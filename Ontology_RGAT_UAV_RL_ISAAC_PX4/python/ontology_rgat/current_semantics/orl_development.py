"""orl-development/1, /2 and /3: ORL as a paired arm of development/1 (status doc 20).

The question is the CNN-vs-MLP one: with the same phi44 input and the same parameter
budget, does putting the ontology into the policy's structure (orl.py) beat the
unstructured semantic MLP?  ORL trains on development/1's training seeds, training
episodes, validation and evaluation environments (common random numbers; nothing is
selected on them) and is compared, paired by training seed, with the stored
development/1 models.  The rule is fixed here and hashed into the registration before
any ORL development model trains; registration also requires a passed ORL G4
learnability pilot under the same ORL hash.  Local plant only.

Each protocol binds one ORL version (``PROTOCOLS``) and writes under its own directory
(``OUTS``): orl-development/1 is orl/1 in ``<root>/orl``, orl-development/2 is orl/2 in
``<root>/orl2``.  orl/1's gate turned out inert (status doc 20.4), so /2 adds the GATE
PROBE (``GATE_PROBE``).  It flies each selected checkpoint through the training blackout
and measures whether the gate closes when the policy's own estimate says the pad has not
been seen.  Registration of /2 requires the pilot checkpoints' gates to be active.
orl/2's gate learned but its closed branch still sank (status doc 20.6), so /3 (orl/3, in
``<root>/orl3``) uses the second probe (``GATE_PROBES["orl/3"]``): the same drop AND a
blind sink no faster than ``max_blind_sink_m_s`` by the policy's own vz.  It also reports,
never decisively, the final checkpoint of each pilot seed.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
import datetime as dt
import json
import multiprocessing
from pathlib import Path
import time
from types import MappingProxyType
from typing import Any, Callable

import numpy as np
import yaml

from . import SCHEMA_ID
from .contracts import PROJECT_ROOT, CurrentSemanticsConfig, contract_manifest, load_config
from .orl import ARMS, CURRENT_VERSION, HASHES, spec_dict

PROTOCOLS = MappingProxyType({"orl-development/1": "orl/1", "orl-development/2": "orl/2",
                              "orl-development/3": "orl/3"})
BASE_PROTOCOL = "development/1"
OUTS = MappingProxyType({"orl/1": "orl", "orl/2": "orl2", "orl/3": "orl3"})
DEFAULT_PLAN_PATHS = MappingProxyType({
    version: PROJECT_ROOT / "config" / "current_semantics"
    / f"orl_development_v{version.split('/')[1]}.yaml" for version in OUTS})
DEFAULT_PLAN_PATH = DEFAULT_PLAN_PATHS[CURRENT_VERSION]
REGISTRATION = "orl_development_registration.json"
RESULT = "orl_development.json"
LEARNABILITY = "orl_learnability.json"
STAGES = ("register-orl-development", "orl-develop")
PRIMARY = "semantic_mlp_ppo"
SECONDARY = ("raw_ppo", "untyped_gnn_ppo", "ontology_rgat_ppo")
RULE = (
    "Primary: mean SUCCESS over the ten development/1 cells; per training seed, delta = ORL "
    "minus the stored development/1 semantic_mlp_ppo (same phi44 input, same budget, no "
    "structure). ORL is called BETTER only if the two-sided 95 % t-interval of delta over the "
    "five training seeds lies above 0, WORSE only if it lies below 0, otherwise 'not "
    "established'. Every ORL model is trained, evaluated and reported; none is dropped. A "
    "model with no checkpoint at difficulty 1.0 scores 0 in every cell, as in development/1. "
    "Secondary, reported and never decisive: ORL minus raw_ppo, untyped_gnn_ppo and "
    "ontology_rgat_ppo on the same metric, and ID mean, four-OOD mean, composition_ood and "
    "OOD unsafe for every comparison; validation learnability of each ORL model.")
# The gate probe (orl/2, status doc 20.5).  Probe seeds 999101-999124 were never used by
# any registered family, audited range or diagnostic (contracts.DIAGNOSTIC_SEED_RANGES).
# The blackout is the TRAINING blackout (g4.training_randomization.vision_blackout), forced
# on, so the probe is in distribution.  Decisions are classed by the policy's OWN decoded
# observation: estimate valid and estimated height above ``high_m``, and a vision age of at
# most ``fresh_age_s`` (fresh) or above ``blind_age_s`` (blind; the age at which the
# reward-/4 lost-view field and the 20.4 blind-brake overlay start).
GATE_PROBE = MappingProxyType({
    "protocol": "orl-gate-probe/1", "seeds": tuple(range(999101, 999125)),
    "split": "in_distribution", "difficulty": 1.0, "deterministic": True,
    "blackout": "the training vision_blackout with probability 1", "high_m": 0.5,
    "fresh_age_s": 0.2, "blind_age_s": 0.5, "min_decisions": 50, "min_gate_drop": 0.2})
GATE_RULE = (
    "A checkpoint's gate is ACTIVE when, over the probe episodes, at least min_decisions "
    "fresh and min_decisions blind decisions occurred above high_m and the mean gate "
    "sigmoid(gate(E)) on fresh decisions exceeds the mean on blind ones by at least "
    "min_gate_drop. orl-development/2 registers only after every selected pilot checkpoint "
    "passed G4 AND has an active gate.")
RULE_V2 = RULE + (
    " orl-development/2 additionally probes the gate of every development model "
    "(GATE_PROBE, same seeds). The verdict is reported with the number of models whose "
    "gate is active; if any is inert the verdict is not read as a test of the gate.")
# The second probe (orl/3, status doc 20.7).  The same seeds, blackout and classes as the
# first.  ``max_blind_sink_m_s`` adds the closed gate's PURPOSE: on blind decisions the
# policy's own estimate of its vertical velocity (decoded drone_vz, up positive) averages a
# sink of at most that speed.  orl/2's pilot read -0.17..-0.20 (seed 0) and -0.25..-0.32
# (seed 1) there (status doc 20.6).  ``report_final``: each pilot seed's final.pt is probed
# and reported as well, never decisively (its best.pt can predate the gate, 20.6).
GATE_PROBE_V2 = MappingProxyType({**GATE_PROBE, "protocol": "orl-gate-probe/2",
                                  "max_blind_sink_m_s": 0.10, "report_final": True})
GATE_RULE_V2 = GATE_RULE + (
    " Probe 2 (orl/3) additionally requires the mean decoded own vertical velocity on the "
    "blind decisions to be at least -max_blind_sink_m_s (a closed gate must stop the "
    "descent, not only the gate value move). The final checkpoint of each pilot seed is "
    "probed and reported; only the selected (best) checkpoints decide registration.")
GATE_PROBES = MappingProxyType({"orl/2": GATE_PROBE, "orl/3": GATE_PROBE_V2})
GATE_RULES = MappingProxyType({"orl/2": GATE_RULE, "orl/3": GATE_RULE_V2})
RULE_V3 = RULE + (
    " orl-development/3 additionally probes the gate of every development model with "
    "probe 2 (GATE_PROBES['orl/3'], same seeds). The verdict is reported with the number "
    "of models whose gate is active; if any is inert the verdict is not read as a test of "
    "the gate.")
RULES = MappingProxyType({"orl-development/1": RULE, "orl-development/2": RULE_V2,
                          "orl-development/3": RULE_V3})


@dataclass(frozen=True)
class ORLPlan:
    protocol: str
    base_protocol: str
    arm: str
    workers: int

    @property
    def version(self) -> str:
        return PROTOCOLS[self.protocol]


def plan_version(path: str | Path) -> str:
    """The ORL version a plan file's protocol binds (before any other check)."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    protocol = data.get("protocol") if isinstance(data, dict) else None
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {sorted(PROTOCOLS)}, got {protocol!r}")
    return PROTOCOLS[protocol]


def load_plan(path: str | Path, config: CurrentSemanticsConfig):
    from .development import DEFAULT_DEVELOPMENT_PATH, DEVELOPMENT_TASK, load_development
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    keys = set(ORLPlan.__dataclass_fields__)
    if not isinstance(data, dict) or set(data) != keys:
        raise ValueError(f"ORL development plan needs exactly the keys {sorted(keys)}")
    if data["protocol"] not in PROTOCOLS or data["base_protocol"] != BASE_PROTOCOL:
        raise ValueError(f"protocol must be one of {sorted(PROTOCOLS)} on {BASE_PROTOCOL}")
    protocol = data["protocol"]
    arm = ARMS[PROTOCOLS[protocol]]
    if data["arm"] != arm:
        raise ValueError(f"plan arm {data['arm']!r} is not this code's ORL arm {arm} "
                         f"({PROTOCOLS[protocol]})")
    if DEVELOPMENT_TASK[BASE_PROTOCOL] != config.task.version:
        raise ValueError(f"{protocol} extends {BASE_PROTOCOL}, a "
                         f"{DEVELOPMENT_TASK[BASE_PROTOCOL]} plan; this config is "
                         f"{config.task.version}")
    workers = data["workers"]
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive int")
    plan = ORLPlan(protocol, BASE_PROTOCOL, arm, workers)
    return plan, load_development(DEFAULT_DEVELOPMENT_PATH, config)


def _probe_spec(version: str = "orl/2") -> dict[str, Any]:
    return {**{k: list(v) if isinstance(v, tuple) else v
               for k, v in GATE_PROBES[version].items()}, "rule": GATE_RULES[version]}


def probe_hash(version: str = "orl/2") -> str:
    from ..direct_policy.contracts import digest
    return digest(_probe_spec(version))


def plan_hash(plan: ORLPlan, development, config: CurrentSemanticsConfig) -> str:
    from ..direct_policy.contracts import digest
    from .development import plan_hash as development_hash
    record = {"plan": asdict(plan), "orl_hash": HASHES[plan.version],
              "rule": RULES[plan.protocol],
              "development_plan_hash": development_hash(development, config)}
    if plan.version != "orl/1":  # orl/1's registered hash predates the probe
        record["gate_probe"] = _probe_spec(plan.version)
    return digest(record)


def out_dir(root: Path, version: str) -> Path:
    return root / OUTS[version]


def _dev_model(root: Path, training_seed: int, arm: str) -> Path:
    return root / "development" / f"seed{training_seed}" / arm


def _model_root(root: Path, training_seed: int, version: str) -> Path:
    return out_dir(root, version) / "development" / f"seed{training_seed}"


def blocker(config: CurrentSemanticsConfig, development, root: Path,
            version: str = CURRENT_VERSION) -> str | None:
    """None when development/1 is complete for every compared arm and the ORL G4 pilot
    passed under this ORL hash and contract (orl/2: with an active gate); otherwise why
    not."""
    from .development import REGISTRATION as DEV_REGISTRATION, plan_hash as development_hash
    path = root / DEV_REGISTRATION
    if not path.exists() or json.loads(path.read_text(encoding="utf-8")).get("plan_hash") \
            != development_hash(development, config):
        return f"{BASE_PROTOCOL} is not registered in {root} under this plan"
    for s in development.training_seeds:
        for arm in (PRIMARY, *SECONDARY):
            summary = _dev_model(root, s, arm) / "summary.json"
            if not summary.exists() or not json.loads(
                    summary.read_text(encoding="utf-8")).get("complete"):
                return f"{BASE_PROTOCOL} model {s}/{arm} is not complete"
    return pilot_blocker(config, root, version)


def pilot_blocker(config: CurrentSemanticsConfig, root: Path,
                  version: str = CURRENT_VERSION) -> str | None:
    """None when ``<root>/<OUTS[version]>/orl_learnability.json`` passed under this ORL
    spec and this contract (orl/2, orl/3: and the version's gate probe found every gate
    active).
    ``pilot.run_pilot(arm=...)`` runs the registered G4 protocol with ``g4.arm`` replaced,
    so its ``g4_hash`` is the protocol's hash WITH the ORL arm, not the Raw one."""
    arm = ARMS[version]
    pilot = out_dir(root, version) / LEARNABILITY
    if not pilot.exists():
        return f"no ORL G4 learnability pilot ({pilot.name})"
    data = json.loads(pilot.read_text(encoding="utf-8"))
    manifest = contract_manifest(replace(config, g4=replace(config.g4, arm=arm)))
    if (data.get("orl") or {}).get("hash") != HASHES[version] or data.get("arm") != arm:
        return "the ORL learnability pilot ran another ORL spec"
    if any(data.get("hashes", {}).get(k) != manifest[k]
           for k in ("g4_hash", "algorithm_hash", "task_hash", "calibration_hash")):
        return "the ORL learnability pilot ran under another contract"
    if not (data.get("protocol_complete") and data.get("learnability_passed")):
        return "the ORL learnability pilot did not pass"
    if version != "orl/1":
        probe = data.get("gate_probe") or {}
        if probe.get("probe_hash") != probe_hash(version):
            return "the ORL pilot's gate was not probed under this gate probe"
        if not probe.get("gate_active"):
            return ("the ORL pilot's gate is inert (gate probe); a development run would not "
                    "test the gate")
    return None


def resolved(plan: ORLPlan, development, config: CurrentSemanticsConfig) -> dict[str, Any]:
    from .development import CELLS
    g4, models = config.g4, len(development.training_seeds)
    extra = ({} if plan.version == "orl/1" else
             {"gate_probe": _probe_spec(plan.version), "probe_hash": probe_hash(plan.version)})
    return {"protocol": plan.protocol, "base_protocol": BASE_PROTOCOL, "arm": plan.arm,
            "orl_version": plan.version, "orl_hash": HASHES[plan.version],
            "orl_spec": spec_dict(plan.version), **extra,
            "plan_hash": plan_hash(plan, development, config),
            "training_seeds": list(development.training_seeds), "models": models,
            "updates": g4.updates, "episodes_per_update": g4.episodes_per_update,
            "training_episodes": models * g4.updates * g4.episodes_per_update,
            "evaluation_episodes": models * len(CELLS) * development.episodes_per_cell,
            "rule": RULES[plan.protocol], "primary_comparison": PRIMARY,
            "secondary": list(SECONDARY),
            "seeds": f"{BASE_PROTOCOL}'s training, validation and evaluation seeds, reused on "
                     "purpose (a paired arm); nothing is selected on them",
            "backend": "spatial_local_v12", "trains": True, "starts_isaac": False}


def register(config_path: Path, plan_path: Path, root: Path) -> tuple[int, dict[str, Any]]:
    config = load_config(config_path)
    plan, development = load_plan(plan_path, config)
    reason = blocker(config, development, root, plan.version)
    if reason:
        return 3, {"stage": STAGES[0], "status": "REFUSED", "reason": reason}
    digest_ = plan_hash(plan, development, config)
    path = out_dir(root, plan.version) / REGISTRATION
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("plan_hash") == digest_:
            return 0, {**old, "status": "ALREADY_REGISTERED"}
        return 3, {"stage": STAGES[0], "status": "REFUSED",
                   "reason": f"{path.name} registers another plan; archive it first"}
    manifest = contract_manifest(config)
    return 0, {"schema_id": SCHEMA_ID, "stage": STAGES[0], "status": "REGISTERED",
               "registered_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               "plan_file": str(plan_path), **resolved(plan, development, config),
               "contract": {k: manifest[k] for k in ("algorithm_hash", "task_hash",
                                                     "feature_hash", "calibration_hash",
                                                     "g4_hash")}}


def check_registration(plan, development, config, root: Path) -> str | None:
    path = out_dir(root, plan.version) / REGISTRATION
    if not path.exists():
        return f"no {plan.protocol} registration: run --stage {STAGES[0]} first"
    if json.loads(path.read_text(encoding="utf-8")).get("plan_hash") != \
            plan_hash(plan, development, config):
        return "the plan, the ORL spec or the contract differs from the registration"
    return None


def judge(development, orl: dict[int, dict], stored: dict[str, dict[int, dict]],
          protocol: str = "orl-development/1") -> dict[str, Any]:
    """``orl``: training seed -> ORL summary; ``stored``: arm -> training seed -> summary."""
    from .development import CELLS, FOUR_OOD, ID_CELLS, OOD_CELLS
    from .relation_control import _t_interval
    seeds = list(development.training_seeds)
    metrics = {"all_cells": (CELLS, "success"), "id_mean": (ID_CELLS, "success"),
               "four_ood_mean": (FOUR_OOD, "success"),
               "composition_ood": (("composition_ood",), "success"),
               "ood_unsafe": (OOD_CELLS, "unsafe")}

    def score(summary, cells, key):
        return float(np.mean([summary["cells"][c][key] for c in cells]))

    out = {}
    for name, (cells, key) in metrics.items():
        mine = [score(orl[s], cells, key) for s in seeds]
        out[name] = {"orl": mine, **{arm: [score(stored[arm][s], cells, key) for s in seeds]
                                     for arm in (PRIMARY, *SECONDARY)}}
        for arm in (PRIMARY, *SECONDARY):
            out[name][f"orl_minus_{arm}"] = _t_interval(
                [a - b for a, b in zip(mine, out[name][arm])])
    primary = out["all_cells"][f"orl_minus_{PRIMARY}"]
    low, high = primary["ci95"]
    verdict = {"rule": RULES[protocol], "primary": primary,
               "orl_vs_semantic_mlp": "better" if low > 0 else "worse" if high < 0
               else "not established",
               "metrics": out,
               "validation": {str(s): orl[s].get("validation") for s in seeds}}
    if PROTOCOLS[protocol] != "orl/1":
        active = [s for s in seeds if (orl[s].get("gate_probe") or {}).get("gate_active")]
        verdict["gate_active_models"] = f"{len(active)}/{len(seeds)}"
        verdict["reads_as_test_of_the_gate"] = len(active) == len(seeds)
    return verdict


def run(config_path: Path, plan_path: Path, root: Path, *, log: Callable[[str], None] = print,
        updates: int | None = None, episodes_per_cell: int | None = None,
        workers: int | None = None) -> tuple[int, dict[str, Any]]:
    """Train, select and evaluate ORL on every development/1 training seed; resumable."""
    from .development import CELLS, _status, _validation_of, arm_config, evaluate_model
    from .experiment import write_json
    from .models import build_model, load_checkpoint
    from .pilot import _init_worker, train_seed
    config = load_config(config_path)
    plan, development = load_plan(plan_path, config)
    reason = blocker(config, development, root, plan.version) or \
        check_registration(plan, development, config, root)
    if reason:
        return 3, {"stage": STAGES[1], "status": "REFUSED", "reason": reason}
    digest_ = plan_hash(plan, development, config)
    if episodes_per_cell is not None:
        development = replace(development, episodes_per_cell=int(episodes_per_cell))
    complete = updates is None and episodes_per_cell is None
    arm, version, protocol = plan.arm, plan.version, plan.protocol
    acfg = arm_config(config, development, arm)
    results: dict[int, dict] = {}
    started = time.monotonic()
    count = plan.workers if workers is None else int(workers)
    with ProcessPoolExecutor(max_workers=count, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_init_worker, initargs=(str(config_path),)) as pool:
        for seed_index, s in enumerate(development.training_seeds):
            model_root = _model_root(root, s, version)
            summary_path = model_root / "summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if summary.get("plan_hash") != digest_ or summary.get("complete") != complete:
                    raise RuntimeError(f"{summary_path} belongs to another run; archive it")
                results[s] = summary
                log(f"{protocol}: seed {s} already done")
                continue
            log(f"{protocol}: training seed {s}")
            _status(model_root, "training")
            trained = train_seed(acfg, config_path, s, model_root, pool, log=log, updates=updates)
            best = model_root / "best.pt"
            summary = {"plan_hash": digest_, "complete": complete, "arm": arm,
                       "training": trained,
                       "validation": _validation_of(model_root, trained["best_update"]),
                       "checkpoint": str(best) if best.exists() else None}
            if best.exists():
                model = build_model(arm, acfg, seed=0)
                load_checkpoint(best, model, manifest=contract_manifest(acfg))
                state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                log(f"{protocol}: evaluating seed {s}")
                summary["cells"] = evaluate_model(
                    pool, development, arm, state, seed_index,
                    progress=lambda name, index, root_=model_root: _status(
                        root_, "evaluating", cell=name, cell_index=index,
                        cells=len(development.cells)))
            else:
                summary["cells"] = {c: {"episodes": 0, "success": 0.0, "unsafe": 0.0,
                                        "timeout": 0.0, "statuses": {},
                                        "note": "no checkpoint reached difficulty 1.0"}
                                    for c in CELLS}
            if version != "orl/1":
                summary["gate_probe"] = (probe_model(pool, arm, state) if best.exists()
                                         else {"gate_active": False,
                                               "note": "no checkpoint reached difficulty 1.0"})
                log(f"{protocol}: seed {s} gate active "
                    f"{summary['gate_probe']['gate_active']}")
            write_json(summary_path, summary)
            _status(model_root, "done")
            results[s] = summary
            log(f"{protocol}: seed {s} all-cells success "
                f"{np.mean([v['success'] for v in summary['cells'].values()]):.3f}")
    stored = {arm: {s: json.loads((_dev_model(root, s, arm) / "summary.json")
                                  .read_text(encoding="utf-8"))
                    for s in development.training_seeds} for arm in (PRIMARY, *SECONDARY)}
    verdict = judge(development, results, stored, protocol) if complete else None
    return 0, {"schema_id": SCHEMA_ID, "stage": STAGES[1], "protocol": protocol,
               "plan_hash": digest_, "orl_hash": HASHES[version], "procedure_complete": complete,
               "verdict": verdict,
               "models": {str(s): {k: v for k, v in x.items() if k not in ("cells",
                                                                            "gate_probe")}
                          | {"gate_active": (x.get("gate_probe") or {}).get("gate_active")}
                          | {"cells": {c: {k: v for k, v in cv.items() if k != "per_seed"}
                                       for c, cv in x["cells"].items()}}
                          for s, x in results.items()},
               "wall_s": time.monotonic() - started,
               "note": "local plant only; paired with the stored development/1 models"}


# ------------------------------------------------------------------ gate probe

def _probe_work(task: tuple) -> dict[str, Any]:
    """One probe episode (``GATE_PROBES`` of the arm's version) in a ``pilot._init_worker``
    process: the frozen policy through the forced training blackout, and per decision the
    mirror-averaged gate and gated term with the policy's own decoded estimate."""
    import torch
    from .backend import make_local_backend
    from .features import build_features
    from .models import tensors_from_frames
    from .observation import decode
    from .orl import version_of
    from .pilot import _WORKER, _model, fly_policy
    arm, state, seed = task
    spec = GATE_PROBES[version_of(arm)]
    config = _WORKER["config"]
    blackout = replace(config.g4.training_randomization.vision_blackout, probability=1.0)
    model = _model(config, arm, state)
    backend = make_local_backend(config, difficulty=spec["difficulty"],
                                 randomization=None, scenario=spec["split"],
                                 dropout=blackout)
    try:
        try:
            episode = fly_policy(backend, model, config, seed=int(seed),
                                 deterministic=spec["deterministic"],
                                 generator=torch.Generator().manual_seed(int(seed) + 9_000_000))
        except RuntimeError as exc:  # pre-policy handover refusal: setup, not RL
            if backend.active:
                raise
            return {"seed": int(seed), "refused": str(exc)}
    finally:
        backend.close()
    observations = episode["observations"][:-1]
    _, flat, _ = tensors_from_frames([build_features(o, config) for o in observations])
    with torch.no_grad():
        terms = model.vertical_terms(flat)
    seen = [decode(o) for o in observations]
    return {"seed": int(seed), "refused": None, "status": episode["status"],
            "blackout_s": episode.get("evaluation_dropout_s"),
            "gate": terms["gate"].tolist(), "gated": terms["gated"].tolist(),
            "valid": [bool(x.estimate_valid) for x in seen],
            "vision_age_s": [float(x.vision_age_s) for x in seen],
            "h": [float(x.h) for x in seen], "vz": [float(x.vz) for x in seen]}


def summarize_probe(episodes: list[dict[str, Any]], version: str = "orl/2") -> dict[str, Any]:
    """``GATE_RULES[version]`` on one checkpoint's probe episodes; the own vertical velocity
    is reported whenever the episodes carry it and decides only under probe 2."""
    spec = GATE_PROBES[version]
    flown = [e for e in episodes if not e.get("refused")]
    cat = lambda key: np.concatenate([np.asarray(e[key], dtype=float) for e in flown]) \
        if flown else np.zeros(0)
    gate, gated, age, h = cat("gate"), cat("gated"), cat("vision_age_s"), cat("h")
    valid = cat("valid").astype(bool)
    high = valid & (h > spec["high_m"])
    fresh = high & (age <= spec["fresh_age_s"])
    blind = high & (age > spec["blind_age_s"])
    mean = lambda x, m: float(x[m].mean()) if m.any() else None
    out = {"episodes": len(flown), "refused": len(episodes) - len(flown),
           "statuses": {k: sum(e["status"] == k for e in flown)
                        for k in sorted({e["status"] for e in flown})},
           "decisions": int(gate.size), "fresh_decisions": int(fresh.sum()),
           "blind_decisions": int(blind.sum()),
           "gate_fresh": mean(gate, fresh), "gate_blind": mean(gate, blind),
           "gated_fresh": mean(gated, fresh), "gated_blind": mean(gated, blind),
           "gate_sd": float(gate.std()) if gate.size else None}
    if flown and all("vz" in e for e in flown):
        vz = cat("vz")
        out["vz_fresh"], out["vz_blind"] = mean(vz, fresh), mean(vz, blind)
    enough = min(out["fresh_decisions"], out["blind_decisions"]) >= spec["min_decisions"]
    out["gate_drop"] = (out["gate_fresh"] - out["gate_blind"]) if enough else None
    active = bool(enough and out["gate_drop"] >= spec["min_gate_drop"])
    if "max_blind_sink_m_s" in spec:
        out["blind_sink_ok"] = bool(enough and out.get("vz_blind") is not None
                                    and out["vz_blind"] >= -spec["max_blind_sink_m_s"])
        active = active and out["blind_sink_ok"]
    out["gate_active"] = active
    return out


def probe_model(pool: Any, arm: str, state: dict[str, Any]) -> dict[str, Any]:
    from .orl import version_of
    version = version_of(arm)
    episodes = list(pool.map(_probe_work, [(arm, state, s)
                                           for s in GATE_PROBES[version]["seeds"]]))
    return {"probe_hash": probe_hash(version), **summarize_probe(episodes, version)}


def probe_checkpoints(config_path: Path, arm: str, checkpoints: dict[str, Path], *,
                      workers: int, log: Callable[[str], None] = print,
                      descriptive: dict[str, Path] | None = None) -> dict[str, Any]:
    """The arm version's gate probe on each selected checkpoint (model seed -> best.pt) of an
    ORL G4 pilot; ``gate_active`` only if every checkpoint exists and its gate is active.
    ``descriptive`` checkpoints (probe 2: each seed's final.pt) are probed and reported under
    ``descriptive_models`` and never enter ``gate_active``."""
    from .models import build_model, load_checkpoint
    from .orl import version_of
    from .pilot import _init_worker
    version = version_of(arm)
    config = load_config(config_path)
    config = replace(config, g4=replace(config.g4, arm=arm))
    manifest = contract_manifest(config)
    groups: dict[str, dict[str, Any]] = {"models": {}, "descriptive_models": {}}
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_init_worker, initargs=(str(config_path),)) as pool:
        for group, paths in (("models", checkpoints), ("descriptive_models", descriptive or {})):
            for key, path in paths.items():
                if not Path(path).exists():
                    groups[group][key] = {"checkpoint": None, "gate_active": False}
                    continue
                model = build_model(arm, config, seed=0)
                meta = load_checkpoint(path, model, manifest=manifest)
                state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                found = {"checkpoint": str(path), "update": meta["extra"].get("update"),
                         **probe_model(pool, arm, state)}
                groups[group][key] = found
                log(f"gate probe {key}{' (descriptive)' if group != 'models' else ''}: gate "
                    f"fresh {found['gate_fresh']} blind {found['gate_blind']} drop "
                    f"{found['gate_drop']} vz blind {found.get('vz_blind')} active "
                    f"{found['gate_active']}")
    models = groups["models"]
    report = {"protocol": GATE_PROBES[version]["protocol"], "probe_hash": probe_hash(version),
              "spec": _probe_spec(version), "arm": arm, "models": models,
              "gate_active": bool(models) and all(m["gate_active"] for m in models.values())}
    if descriptive:
        report["descriptive_models"] = groups["descriptive_models"]
    return report
