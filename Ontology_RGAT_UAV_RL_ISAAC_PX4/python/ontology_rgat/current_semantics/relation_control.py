"""Relation-shuffle control of the ontology R-GAT (guide 8.4; status doc 19).

Guide 8.4: claiming that the SEMANTIC placement of the relation names beats an
arbitrary placement needs a relation-shuffle control with the same architecture,
parameters and head, registered as its own plan, and no favourable shuffle may be
chosen after the results.

A shuffle keeps the eight nodes, the fourteen semantic (source, target) pairs, the
eight self loops, the six relation names and how many edges each labels (3, 2, 3,
3, 3 and 8 self).  The encoder therefore has the typed arm's exact parameters,
initialisation (same model seed -> bit-identical tensors) and two-hop reach.  Only
WHICH semantic edges share a weight matrix changes: the fourteen semantic labels
are permuted uniformly from a declared stream, and a permutation under which any
typed relation's edge set survives intact (whatever its name) is redrawn, so no
shuffle is the typed graph with renamed weights.

The control is a fifth arm of development/1: the same training seeds, training
episodes, validation and evaluation environments (common random numbers; it
selects nothing on them), compared with the stored development/1 typed models
paired by training seed.  The current code reproduces the stored typed
evaluation exactly and its training to ~1e-8 per update (floating point), so the
stored models are a sample of the same protocol.  Local plant only.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import datetime as dt
import itertools
import json
import math
import multiprocessing
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import torch
import yaml

from . import GRAPH_VERSION, SCHEMA_ID
from .contracts import NODE_ORDER, PROJECT_ROOT, CurrentSemanticsConfig, contract_manifest, \
    load_config
from .graph import SELF_EDGES, SEMANTIC_EDGES, TYPED_GRAPH, TYPED_RELATIONS, GraphSpec

PROTOCOL = "relation-shuffle/1"
BASE_PROTOCOL = "development/1"
TYPED_ARM = "ontology_rgat_ppo"
ARM_PREFIX = "ontology_rgat_shuffled_ppo"
DEFAULT_PLAN_PATH = PROJECT_ROOT / "config" / "current_semantics" / "relation_shuffle_v1.yaml"
REGISTRATION = "relation_shuffle_registration.json"
RESULT = "relation_shuffle.json"
OUT_DIR = "relation_shuffle"
STAGES = ("register-relation-shuffle", "relation-shuffle")
_STREAM = 0x5E1F_0001
AUDIT_EPISODES = 4  # composition_id environments re-flown per model for the relational audit
RULE = (
    "Primary: mean SUCCESS over the ten development/1 cells. Per training seed, delta = the "
    "stored development/1 typed R-GAT minus the mean of that seed's declared shuffles. The "
    "semantic placement is called BETTER only if the two-sided 95 % t-interval of delta over "
    "the five training seeds lies above 0, WORSE only if it lies below 0, otherwise 'not "
    "established'. Every declared shuffle is trained, evaluated and reported; none is "
    "dropped, replaced or re-drawn after training. A shuffle model with no checkpoint at "
    "difficulty 1.0 scores 0 in every cell, as in development/1. Secondary, reported and never "
    "decisive: ID mean, four-OOD mean, composition_ood, OOD unsafe, shuffled minus untyped, "
    "the within-seed spread between shuffles, and the relational-output audit.")


# -------------------------------------------------------------------- graphs

def _groups(edges) -> frozenset[frozenset[tuple[str, str]]]:
    by: dict[str, set[tuple[str, str]]] = {}
    for source, target, relation in edges:
        by.setdefault(relation, set()).add((source, target))
    return frozenset(frozenset(pairs) for pairs in by.values())


TYPED_GROUPS = _groups(SEMANTIC_EDGES)


def shuffled_graph(shuffle_seed: int) -> tuple[GraphSpec, int]:
    """(graph, draws): the first uniform permutation of the fourteen semantic labels from
    stream (``_STREAM``, ``shuffle_seed``) that leaves no typed edge set intact."""
    if isinstance(shuffle_seed, bool) or not isinstance(shuffle_seed, int) or shuffle_seed < 0:
        raise ValueError("shuffle_seed must be a non-negative int")
    rng = np.random.default_rng([_STREAM, shuffle_seed])
    labels = [relation for _, _, relation in SEMANTIC_EDGES]
    for draws in itertools.count(1):
        permuted = [labels[i] for i in rng.permutation(len(labels))]
        edges = tuple((s, t, r) for (s, t, _), r in zip(SEMANTIC_EDGES, permuted))
        if not _groups(edges) & TYPED_GROUPS:
            break
    spec = GraphSpec(version=f"{GRAPH_VERSION}+{PROTOCOL}:{shuffle_seed}", nodes=NODE_ORDER,
                     relations=TYPED_RELATIONS, edges=edges + SELF_EDGES)
    return spec, draws


def rand_index(spec: GraphSpec) -> float:
    """Share of the 91 semantic edge pairs whose same/different-relation status agrees
    with the typed graph (1 = the typed partition)."""
    def same(edges):
        relation = {(s, t): r for s, t, r in edges if r != "self"}
        pairs = sorted(relation)
        return [relation[a] == relation[b] for a, b in itertools.combinations(pairs, 2)]
    typed, other = same(TYPED_GRAPH.edges), same(spec.edges)
    return float(np.mean([a == b for a, b in zip(typed, other)]))


def shuffle_arm(shuffle_seed: int) -> str:
    """The arm name a worker rebuilds the graph from; it carries the graph hash, so a
    checkpoint never binds to a graph the rule no longer draws."""
    spec, _ = shuffled_graph(shuffle_seed)
    return f"{ARM_PREFIX}:{shuffle_seed}:{spec.graph_hash[:16]}"


def parse_arm(arm: str) -> int:
    parts = arm.split(":") if isinstance(arm, str) else []
    if len(parts) != 3 or parts[0] != ARM_PREFIX or not parts[1].isdigit():
        raise ValueError(f"unknown arm {arm!r}: neither registered nor a {ARM_PREFIX} arm")
    seed = int(parts[1])
    if shuffled_graph(seed)[0].graph_hash[:16] != parts[2]:
        raise ValueError(f"{arm}: the shuffle rule no longer draws this graph")
    return seed


def shuffled_model(arm: str, config: CurrentSemanticsConfig, *, seed: int):
    """The typed R-GAT arm with its edges' relation indices replaced by the shuffle's:
    same parameters and initialisation, same topology, different weight sharing."""
    from .models import CurrentActorCritic
    spec, _ = shuffled_graph(parse_arm(arm))
    model = CurrentActorCritic(TYPED_ARM, config, seed=seed)
    encoder = model.encoder
    source, target, relation = spec.edge_index()
    if not (np.array_equal(source, encoder.source.numpy())
            and np.array_equal(target, encoder.target.numpy())):
        raise AssertionError("a shuffle must keep the typed graph's (source, target) order")
    encoder.spec = spec
    with torch.no_grad():
        encoder.relation.copy_(torch.as_tensor(relation, dtype=torch.long))
    model.arm = arm
    return model


# ---------------------------------------------------------------------- plan

@dataclass(frozen=True)
class ShufflePlan:
    protocol: str
    base_protocol: str
    shuffles_per_training_seed: int
    shuffle_seed_base: int
    workers: int

    def shuffle_seeds(self, training_seeds) -> dict[int, tuple[int, ...]]:
        return {s: tuple(self.shuffle_seed_base + 10 * s + k
                         for k in range(self.shuffles_per_training_seed))
                for s in training_seeds}


def _int(value: Any, where: str, low: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < low:
        raise ValueError(f"{where} must be an int >= {low}")
    return value


def load_plan(path: str | Path, config: CurrentSemanticsConfig):
    """(ShufflePlan, the development/1 plan it extends), validated."""
    from .development import DEFAULT_DEVELOPMENT_PATH, DEVELOPMENT_TASK, load_development
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    keys = set(ShufflePlan.__dataclass_fields__)
    if not isinstance(data, dict) or set(data) != keys:
        raise ValueError(f"relation-shuffle plan needs exactly the keys {sorted(keys)}")
    if data["protocol"] != PROTOCOL or data["base_protocol"] != BASE_PROTOCOL:
        raise ValueError(f"protocol must be {PROTOCOL} on {BASE_PROTOCOL}")
    if DEVELOPMENT_TASK[BASE_PROTOCOL] != config.task.version:
        raise ValueError(f"{PROTOCOL} extends {BASE_PROTOCOL}, a "
                         f"{DEVELOPMENT_TASK[BASE_PROTOCOL]} plan; this config is "
                         f"{config.task.version}")
    plan = ShufflePlan(protocol=PROTOCOL, base_protocol=BASE_PROTOCOL,
                       shuffles_per_training_seed=_int(data["shuffles_per_training_seed"],
                                                       "shuffles_per_training_seed", 1),
                       shuffle_seed_base=_int(data["shuffle_seed_base"], "shuffle_seed_base", 0),
                       workers=_int(data["workers"], "workers", 1))
    if plan.shuffles_per_training_seed > 10:
        raise ValueError("shuffles_per_training_seed > 10 overflows the seed layout")
    development = load_development(DEFAULT_DEVELOPMENT_PATH, config)
    graphs = [shuffled_graph(s)[0] for seeds in plan.shuffle_seeds(
        development.training_seeds).values() for s in seeds]
    if len({_groups(g.edges) for g in graphs}) != len(graphs):
        raise ValueError("two declared shuffles draw the same partition")
    return plan, development


def graphs_of(plan: ShufflePlan, development) -> dict[str, dict[str, Any]]:
    out = {}
    for training_seed, seeds in plan.shuffle_seeds(development.training_seeds).items():
        for k, s in enumerate(seeds):
            spec, draws = shuffled_graph(s)
            out[f"{training_seed}/{k}"] = {
                "training_seed": training_seed, "index": k, "shuffle_seed": s,
                "arm": shuffle_arm(s), "graph_hash": spec.graph_hash, "draws": draws,
                "rand_index_vs_typed": rand_index(spec),
                "groups": {r: sorted(f"{a}->{b}" for a, b, rr in spec.edges if rr == r)
                           for r in TYPED_RELATIONS if r != "self"}}
    return out


def plan_hash(plan: ShufflePlan, development, config: CurrentSemanticsConfig) -> str:
    from ..direct_policy.contracts import digest
    from .development import plan_hash as development_hash
    return digest({"plan": asdict(plan), "development_plan_hash":
                   development_hash(development, config),
                   "graphs": {k: v["graph_hash"] for k, v in graphs_of(plan, development).items()},
                   "rule": RULE, "audit_episodes": AUDIT_EPISODES})


def _model_root(root: Path, training_seed: int, k: int) -> Path:
    return root / OUT_DIR / f"seed{training_seed}" / f"shuffle{k}"


def _development_model(root: Path, training_seed: int, arm: str) -> Path:
    return root / "development" / f"seed{training_seed}" / arm


def base_blocker(config: CurrentSemanticsConfig, development, root: Path) -> str | None:
    """None when development/1 is registered under this plan and every typed and untyped
    model it compares against is complete; otherwise why not."""
    from .development import REGISTRATION as DEV_REGISTRATION, plan_hash as development_hash
    path = root / DEV_REGISTRATION
    if not path.exists():
        return f"{BASE_PROTOCOL} is not registered in {root}"
    if json.loads(path.read_text(encoding="utf-8")).get("plan_hash") != \
            development_hash(development, config):
        return f"{root} registers another {BASE_PROTOCOL} plan or contract"
    for training_seed in development.training_seeds:
        for arm in (TYPED_ARM, "untyped_gnn_ppo"):
            summary = _development_model(root, training_seed, arm) / "summary.json"
            if not summary.exists() or not json.loads(
                    summary.read_text(encoding="utf-8")).get("complete"):
                return f"{BASE_PROTOCOL} model {training_seed}/{arm} is not complete"
    return None


def resolved(plan: ShufflePlan, development, config: CurrentSemanticsConfig) -> dict[str, Any]:
    g4 = config.g4
    models = len(development.training_seeds) * plan.shuffles_per_training_seed
    from .development import CELLS
    return {"protocol": PROTOCOL, "base_protocol": BASE_PROTOCOL,
            "plan_hash": plan_hash(plan, development, config), "plan": asdict(plan),
            "training_seeds": list(development.training_seeds), "models": models,
            "updates": g4.updates, "episodes_per_update": g4.episodes_per_update,
            "training_episodes": models * g4.updates * g4.episodes_per_update,
            "evaluation_episodes": models * len(CELLS) * development.episodes_per_cell,
            "graphs": graphs_of(plan, development), "rule": RULE,
            "seeds": f"{BASE_PROTOCOL}'s training, validation and evaluation seeds, reused "
                     "on purpose (a paired fifth arm); nothing is selected on them",
            "backend": "spatial_local_v12", "trains": True, "starts_isaac": False}


# -------------------------------------------------------------- registration

def register(config_path: Path, plan_path: Path, root: Path) -> tuple[int, dict[str, Any]]:
    config = load_config(config_path)
    plan, development = load_plan(plan_path, config)
    blocker = base_blocker(config, development, root)
    if blocker:
        return 3, {"stage": STAGES[0], "status": "REFUSED", "reason": blocker}
    digest_ = plan_hash(plan, development, config)
    path = root / REGISTRATION
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("plan_hash") == digest_:
            return 0, {**old, "status": "ALREADY_REGISTERED"}
        return 3, {"stage": STAGES[0], "status": "REFUSED",
                   "reason": f"{path.name} registers another plan ({old.get('plan_hash')}); "
                             "archive it before registering a new one"}
    from .models import CurrentActorCritic
    typed = CurrentActorCritic(TYPED_ARM, config, seed=development.training_seeds[0])
    checks = {}
    for key, graph in graphs_of(plan, development).items():
        shuffled = shuffled_model(graph["arm"], config, seed=development.training_seeds[0])
        same_init = all(torch.equal(a, b) for a, b in zip(typed.state_dict().values(),
                                                           shuffled.state_dict().values()))
        checks[key] = {"parameter_report_equal": shuffled.parameter_report()
                       == typed.parameter_report(), "initialisation_identical": same_init,
                       "relation_indices_differ": not torch.equal(
                           typed.encoder.relation, shuffled.encoder.relation)}
    if not all(all(c.values()) for c in checks.values()):
        raise AssertionError(f"a shuffle is not the typed arm with re-labelled edges: {checks}")
    manifest = contract_manifest(config)
    return 0, {"schema_id": SCHEMA_ID, "stage": STAGES[0], "status": "REGISTERED",
               "registered_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               "plan_file": str(plan_path), **resolved(plan, development, config),
               "construction_checks": checks,
               "contract": {k: manifest[k] for k in ("algorithm_hash", "task_hash",
                                                     "feature_hash", "graph_hash",
                                                     "calibration_hash", "g4_hash")}}


def check_registration(plan: ShufflePlan, development, config: CurrentSemanticsConfig,
                       root: Path) -> str | None:
    path = root / REGISTRATION
    if not path.exists():
        return f"no {PROTOCOL} registration: run --stage {STAGES[0]} first"
    old = json.loads(path.read_text(encoding="utf-8"))
    if old.get("plan_hash") != plan_hash(plan, development, config):
        return f"the plan or contract differs from the registration ({old.get('plan_hash')})"
    return None


# --------------------------------------------------------------------- audit

def _audit_work(task: tuple) -> list:
    from .pilot import _WORKER, run_episode
    arm, state, seed = task
    episode = run_episode(_WORKER["config"], arm, state, seed=seed, difficulty=1.0,
                          deterministic=True, action_seed=seed + 9_000_000)
    return [] if episode.get("refused") else episode["observations"].tolist()


def relational_audit(pool: Any, development, arm: str, state: dict[str, Any], config,
                     seed_index: int) -> dict[str, Any]:
    """The graph readout g8 and how much it moves the actor's mean, on the decisions of
    the first ``AUDIT_EPISODES`` composition_id environments of this training seed:
    rms of g8, its across-decision sd, and the rms change of the latent mean when g8 is
    replaced by its mean over the decisions (the state-dependent graph signal removed)."""
    from .features import build_features
    from .models import build_model, tensors_from_frames
    seeds = [development.eval_seed(seed_index, 0, k) for k in range(AUDIT_EPISODES)]
    vectors = [v for rows in pool.map(_audit_work, [(arm, state, s) for s in seeds]) for v in rows]
    model = build_model(arm, config, seed=0)
    model.load_state_dict(state)
    model.eval()
    raw, flat, nodes = tensors_from_frames([build_features(np.asarray(v), config)
                                            for v in vectors])
    with torch.no_grad():
        g8 = model.graph_readout(nodes)
        mean = model(raw, flat, nodes)
        ablated = model.actor_head(torch.cat((raw, g8.mean(0, keepdim=True).expand_as(g8)), -1))
    return {"decisions": len(vectors), "g8_rms": float(g8.square().mean().sqrt()),
            "g8_sd_across_decisions": float(g8.std(0).square().mean().sqrt()),
            "latent_mean_change_rms": float((mean - ablated).square().sum(-1).mean().sqrt())}


# ---------------------------------------------------------------------- gate

def _t_interval(values: list[float]) -> dict[str, Any]:
    from scipy import stats
    x = np.asarray(values, dtype=float)
    n, m = len(x), float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else math.nan
    half = float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)) if n > 1 else math.nan
    return {"per_seed": [float(v) for v in x], "mean": m, "sd": sd,
            "t": m / (sd / math.sqrt(n)) if n > 1 and sd > 0 else None,
            "ci95": [m - half, m + half]}


def judge(development, typed: dict[int, dict], untyped: dict[int, dict],
          shuffles: dict[int, list[dict]]) -> dict[str, Any]:
    """``typed``/``untyped``: training seed -> development/1 summary; ``shuffles``:
    training seed -> that seed's shuffle summaries in declared order."""
    from .development import CELLS, FOUR_OOD, ID_CELLS, OOD_CELLS
    seeds = list(development.training_seeds)
    metrics = {"all_cells": (CELLS, "success"), "id_mean": (ID_CELLS, "success"),
               "four_ood_mean": (FOUR_OOD, "success"),
               "composition_ood": (("composition_ood",), "success"),
               "ood_unsafe": (OOD_CELLS, "unsafe")}

    def score(summary, cells, key):
        return float(np.mean([summary["cells"][c][key] for c in cells]))

    out = {}
    for name, (cells, key) in metrics.items():
        t = [score(typed[s], cells, key) for s in seeds]
        u = [score(untyped[s], cells, key) for s in seeds]
        sh = [[score(x, cells, key) for x in shuffles[s]] for s in seeds]
        mean_sh = [float(np.mean(v)) for v in sh]
        out[name] = {"typed": t, "untyped": u, "shuffles": sh,
                     "typed_minus_shuffled": _t_interval([a - b for a, b in zip(t, mean_sh)]),
                     "shuffled_minus_untyped": _t_interval([a - b for a, b in zip(mean_sh, u)]),
                     "typed_minus_untyped": _t_interval([a - b for a, b in zip(t, u)]),
                     "within_seed_shuffle_spread": (
                         float(np.mean([max(v) - min(v) for v in sh]))
                         if all(len(v) > 1 for v in sh) else None)}
    primary = out["all_cells"]["typed_minus_shuffled"]
    low, high = primary["ci95"]
    verdict = "better" if low > 0 else "worse" if high < 0 else "not established"
    validation = {f"{s}/{k}": x.get("validation") for s in seeds
                  for k, x in enumerate(shuffles[s])}
    return {"rule": RULE, "primary": primary, "semantic_placement": verdict,
            "metrics": out, "shuffle_validation": validation}


# ----------------------------------------------------------------------- run

def run(config_path: Path, plan_path: Path, root: Path, *, log: Callable[[str], None] = print,
        updates: int | None = None, episodes_per_cell: int | None = None,
        workers: int | None = None) -> tuple[int, dict[str, Any]]:
    """Train, select, evaluate and audit every declared shuffle; resumable per model.
    ``updates``/``episodes_per_cell``/``workers`` are for tests: the result is then marked
    incomplete and carries no verdict."""
    from dataclasses import replace
    from .development import (_status, _validation_of, arm_config, evaluate_model)
    from .experiment import write_json
    from .models import build_model, load_checkpoint
    from .pilot import _init_worker, train_seed
    config = load_config(config_path)
    plan, development = load_plan(plan_path, config)
    blocker = base_blocker(config, development, root) or \
        check_registration(plan, development, config, root)
    if blocker:
        return 3, {"stage": STAGES[1], "status": "REFUSED", "reason": blocker}
    digest_ = plan_hash(plan, development, config)
    if episodes_per_cell is not None:
        development = replace(development, episodes_per_cell=int(episodes_per_cell))
    complete = updates is None and episodes_per_cell is None
    graphs = graphs_of(plan, development)
    shuffles: dict[int, list[dict]] = {s: [] for s in development.training_seeds}
    audits: dict[str, Any] = {}
    started = time.monotonic()
    count = plan.workers if workers is None else int(workers)

    def load(arm, acfg, path):
        model = build_model(arm, acfg, seed=0)
        load_checkpoint(path, model, manifest=contract_manifest(acfg))
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    with ProcessPoolExecutor(max_workers=count, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_init_worker, initargs=(str(config_path),)) as pool:
        for seed_index, training_seed in enumerate(development.training_seeds):
            for k in range(plan.shuffles_per_training_seed):
                graph = graphs[f"{training_seed}/{k}"]
                arm, model_root = graph["arm"], _model_root(root, training_seed, k)
                summary_path = model_root / "summary.json"
                acfg = arm_config(config, development, arm)
                if summary_path.exists():
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    if summary.get("plan_hash") != digest_ or summary.get("complete") != complete:
                        raise RuntimeError(f"{summary_path} belongs to another run; archive it")
                    shuffles[training_seed].append(summary)
                    log(f"{PROTOCOL}: {training_seed}/shuffle{k} already done")
                    continue
                log(f"{PROTOCOL}: training {training_seed}/shuffle{k} ({arm})")
                _status(model_root, "training")
                trained = train_seed(acfg, config_path, training_seed, model_root, pool,
                                     log=log, updates=updates)
                best = model_root / "best.pt"
                summary = {"plan_hash": digest_, "complete": complete, "arm": arm,
                           "graph": graph, "training": trained,
                           "validation": _validation_of(model_root, trained["best_update"]),
                           "checkpoint": str(best) if best.exists() else None}
                if best.exists():
                    state = load(arm, acfg, best)
                    log(f"{PROTOCOL}: evaluating {training_seed}/shuffle{k}")
                    summary["cells"] = evaluate_model(
                        pool, development, arm, state, seed_index,
                        progress=lambda name, index, root_=model_root: _status(
                            root_, "evaluating", cell=name, cell_index=index,
                            cells=len(development.cells)))
                    summary["audit"] = relational_audit(pool, development, arm, state, acfg,
                                                        seed_index)
                else:
                    from .development import CELLS
                    summary["cells"] = {c: {"episodes": 0, "success": 0.0, "unsafe": 0.0,
                                            "timeout": 0.0, "statuses": {},
                                            "note": "no checkpoint reached difficulty 1.0"}
                                        for c in CELLS}
                    summary["audit"] = None
                write_json(summary_path, summary)
                _status(model_root, "done")
                shuffles[training_seed].append(summary)
                log(f"{PROTOCOL}: {training_seed}/shuffle{k} all-cells success "
                    f"{np.mean([v['success'] for v in summary['cells'].values()]):.3f}")
        typed = {s: json.loads((_development_model(root, s, TYPED_ARM) / "summary.json")
                               .read_text(encoding="utf-8")) for s in development.training_seeds}
        untyped = {s: json.loads((_development_model(root, s, "untyped_gnn_ppo") / "summary.json")
                                 .read_text(encoding="utf-8"))
                   for s in development.training_seeds}
        for seed_index, s in enumerate(development.training_seeds):
            acfg = arm_config(config, development, TYPED_ARM)
            path = Path(typed[s]["checkpoint"])
            audits[f"{s}/typed"] = relational_audit(pool, development, TYPED_ARM,
                                                    load(TYPED_ARM, acfg, path), acfg, seed_index)
    verdict = judge(development, typed, untyped, shuffles) if complete else None
    payload = {"schema_id": SCHEMA_ID, "stage": STAGES[1], "protocol": PROTOCOL,
               "plan_hash": digest_, "procedure_complete": complete, "verdict": verdict,
               "graphs": graphs,
               "audit": {**audits, **{f"{s}/shuffle{k}": x.get("audit")
                                      for s, v in shuffles.items() for k, x in enumerate(v)}},
               "models": {f"{s}/shuffle{k}": {kk: vv for kk, vv in x.items() if kk != "cells"}
                          | {"cells": {c: {kk: vv for kk, vv in cv.items() if kk != "per_seed"}
                                       for c, cv in x["cells"].items()}}
                          for s, v in shuffles.items() for k, x in enumerate(v)},
               "wall_s": time.monotonic() - started,
               "note": "local plant only; compared with the stored development/1 typed and "
                       "untyped models paired by training seed"}
    return 0, payload
