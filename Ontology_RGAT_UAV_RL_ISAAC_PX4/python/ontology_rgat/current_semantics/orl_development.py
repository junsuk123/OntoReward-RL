"""orl-development/1: ORL as a paired arm of development/1 (status doc 20).

The question is the CNN-vs-MLP one: with the same phi44 input and the same parameter
budget, does putting the ontology into the policy's structure (orl.py) beat the
unstructured semantic MLP?  ORL trains on development/1's training seeds, training
episodes, validation and evaluation environments (common random numbers; nothing is
selected on them) and is compared, paired by training seed, with the stored
development/1 models.  The rule is fixed here and hashed into the registration before
any ORL development model trains; registration also requires a passed ORL G4
learnability pilot under the same ORL hash.  Local plant only.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
import datetime as dt
import json
import multiprocessing
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import yaml

from . import SCHEMA_ID
from .contracts import PROJECT_ROOT, CurrentSemanticsConfig, contract_manifest, load_config
from .orl import ORL_ARM, ORL_HASH, spec_dict

PROTOCOL = "orl-development/1"
BASE_PROTOCOL = "development/1"
DEFAULT_PLAN_PATH = PROJECT_ROOT / "config" / "current_semantics" / "orl_development_v1.yaml"
OUT = "orl"
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


@dataclass(frozen=True)
class ORLPlan:
    protocol: str
    base_protocol: str
    arm: str
    workers: int


def load_plan(path: str | Path, config: CurrentSemanticsConfig):
    from .development import DEFAULT_DEVELOPMENT_PATH, DEVELOPMENT_TASK, load_development
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    keys = set(ORLPlan.__dataclass_fields__)
    if not isinstance(data, dict) or set(data) != keys:
        raise ValueError(f"ORL development plan needs exactly the keys {sorted(keys)}")
    if data["protocol"] != PROTOCOL or data["base_protocol"] != BASE_PROTOCOL:
        raise ValueError(f"protocol must be {PROTOCOL} on {BASE_PROTOCOL}")
    if data["arm"] != ORL_ARM:
        raise ValueError(f"plan arm {data['arm']!r} is not this code's ORL arm {ORL_ARM}")
    if DEVELOPMENT_TASK[BASE_PROTOCOL] != config.task.version:
        raise ValueError(f"{PROTOCOL} extends {BASE_PROTOCOL}, a "
                         f"{DEVELOPMENT_TASK[BASE_PROTOCOL]} plan; this config is "
                         f"{config.task.version}")
    workers = data["workers"]
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive int")
    plan = ORLPlan(PROTOCOL, BASE_PROTOCOL, ORL_ARM, workers)
    return plan, load_development(DEFAULT_DEVELOPMENT_PATH, config)


def plan_hash(plan: ORLPlan, development, config: CurrentSemanticsConfig) -> str:
    from ..direct_policy.contracts import digest
    from .development import plan_hash as development_hash
    return digest({"plan": asdict(plan), "orl_hash": ORL_HASH, "rule": RULE,
                   "development_plan_hash": development_hash(development, config)})


def _dev_model(root: Path, training_seed: int, arm: str) -> Path:
    return root / "development" / f"seed{training_seed}" / arm


def _model_root(root: Path, training_seed: int) -> Path:
    return root / OUT / "development" / f"seed{training_seed}"


def blocker(config: CurrentSemanticsConfig, development, root: Path) -> str | None:
    """None when development/1 is complete for every compared arm and the ORL G4 pilot
    passed under this ORL hash and contract; otherwise why not."""
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
    pilot = root / OUT / LEARNABILITY
    if not pilot.exists():
        return f"no ORL G4 learnability pilot ({pilot.name})"
    data = json.loads(pilot.read_text(encoding="utf-8"))
    manifest = contract_manifest(config)
    if (data.get("orl") or {}).get("hash") != ORL_HASH or data.get("arm") != ORL_ARM:
        return "the ORL learnability pilot ran another ORL spec"
    if any(data.get("hashes", {}).get(k) != manifest[k]
           for k in ("g4_hash", "algorithm_hash", "task_hash", "calibration_hash")):
        return "the ORL learnability pilot ran under another contract"
    if not (data.get("protocol_complete") and data.get("learnability_passed")):
        return "the ORL learnability pilot did not pass"
    return None


def resolved(plan: ORLPlan, development, config: CurrentSemanticsConfig) -> dict[str, Any]:
    from .development import CELLS
    g4, models = config.g4, len(development.training_seeds)
    return {"protocol": PROTOCOL, "base_protocol": BASE_PROTOCOL, "arm": ORL_ARM,
            "orl_hash": ORL_HASH, "orl_spec": spec_dict(),
            "plan_hash": plan_hash(plan, development, config),
            "training_seeds": list(development.training_seeds), "models": models,
            "updates": g4.updates, "episodes_per_update": g4.episodes_per_update,
            "training_episodes": models * g4.updates * g4.episodes_per_update,
            "evaluation_episodes": models * len(CELLS) * development.episodes_per_cell,
            "rule": RULE, "primary_comparison": PRIMARY, "secondary": list(SECONDARY),
            "seeds": f"{BASE_PROTOCOL}'s training, validation and evaluation seeds, reused on "
                     "purpose (a paired arm); nothing is selected on them",
            "backend": "spatial_local_v12", "trains": True, "starts_isaac": False}


def register(config_path: Path, plan_path: Path, root: Path) -> tuple[int, dict[str, Any]]:
    config = load_config(config_path)
    plan, development = load_plan(plan_path, config)
    reason = blocker(config, development, root)
    if reason:
        return 3, {"stage": STAGES[0], "status": "REFUSED", "reason": reason}
    digest_ = plan_hash(plan, development, config)
    path = root / OUT / REGISTRATION
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
    path = root / OUT / REGISTRATION
    if not path.exists():
        return f"no {PROTOCOL} registration: run --stage {STAGES[0]} first"
    if json.loads(path.read_text(encoding="utf-8")).get("plan_hash") != \
            plan_hash(plan, development, config):
        return "the plan, the ORL spec or the contract differs from the registration"
    return None


def judge(development, orl: dict[int, dict], stored: dict[str, dict[int, dict]]) -> dict[str, Any]:
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
    return {"rule": RULE, "primary": primary,
            "orl_vs_semantic_mlp": "better" if low > 0 else "worse" if high < 0
            else "not established",
            "metrics": out,
            "validation": {str(s): orl[s].get("validation") for s in seeds}}


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
    reason = blocker(config, development, root) or \
        check_registration(plan, development, config, root)
    if reason:
        return 3, {"stage": STAGES[1], "status": "REFUSED", "reason": reason}
    digest_ = plan_hash(plan, development, config)
    if episodes_per_cell is not None:
        development = replace(development, episodes_per_cell=int(episodes_per_cell))
    complete = updates is None and episodes_per_cell is None
    acfg = arm_config(config, development, ORL_ARM)
    results: dict[int, dict] = {}
    started = time.monotonic()
    count = plan.workers if workers is None else int(workers)
    with ProcessPoolExecutor(max_workers=count, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_init_worker, initargs=(str(config_path),)) as pool:
        for seed_index, s in enumerate(development.training_seeds):
            model_root = _model_root(root, s)
            summary_path = model_root / "summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if summary.get("plan_hash") != digest_ or summary.get("complete") != complete:
                    raise RuntimeError(f"{summary_path} belongs to another run; archive it")
                results[s] = summary
                log(f"{PROTOCOL}: seed {s} already done")
                continue
            log(f"{PROTOCOL}: training seed {s}")
            _status(model_root, "training")
            trained = train_seed(acfg, config_path, s, model_root, pool, log=log, updates=updates)
            best = model_root / "best.pt"
            summary = {"plan_hash": digest_, "complete": complete, "arm": ORL_ARM,
                       "training": trained,
                       "validation": _validation_of(model_root, trained["best_update"]),
                       "checkpoint": str(best) if best.exists() else None}
            if best.exists():
                model = build_model(ORL_ARM, acfg, seed=0)
                load_checkpoint(best, model, manifest=contract_manifest(acfg))
                state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                log(f"{PROTOCOL}: evaluating seed {s}")
                summary["cells"] = evaluate_model(
                    pool, development, ORL_ARM, state, seed_index,
                    progress=lambda name, index, root_=model_root: _status(
                        root_, "evaluating", cell=name, cell_index=index,
                        cells=len(development.cells)))
            else:
                summary["cells"] = {c: {"episodes": 0, "success": 0.0, "unsafe": 0.0,
                                        "timeout": 0.0, "statuses": {},
                                        "note": "no checkpoint reached difficulty 1.0"}
                                    for c in CELLS}
            write_json(summary_path, summary)
            _status(model_root, "done")
            results[s] = summary
            log(f"{PROTOCOL}: seed {s} all-cells success "
                f"{np.mean([v['success'] for v in summary['cells'].values()]):.3f}")
    stored = {arm: {s: json.loads((_dev_model(root, s, arm) / "summary.json")
                                  .read_text(encoding="utf-8"))
                    for s in development.training_seeds} for arm in (PRIMARY, *SECONDARY)}
    verdict = judge(development, results, stored) if complete else None
    return 0, {"schema_id": SCHEMA_ID, "stage": STAGES[1], "protocol": PROTOCOL,
               "plan_hash": digest_, "orl_hash": ORL_HASH, "procedure_complete": complete,
               "verdict": verdict,
               "models": {str(s): {k: v for k, v in x.items() if k != "cells"}
                          | {"cells": {c: {k: v for k, v in cv.items() if k != "per_seed"}
                                       for c, cv in x["cells"].items()}}
                          for s, x in results.items()},
               "wall_s": time.monotonic() - started,
               "note": "local plant only; paired with the stored development/1 models"}
