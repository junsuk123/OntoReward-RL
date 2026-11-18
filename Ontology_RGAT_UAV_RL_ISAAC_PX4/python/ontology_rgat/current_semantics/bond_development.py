"""bond-development/1: the bond/1 arm (bond.py) as a paired arm of a completed development/1.

Stages (``./run.sh current-semantics-3d --stage …``), all local plant, in ``<root>/bond``:

* ``bond-learnability``   the config's G4 protocol (g4-pilot) with the bond arm;
* ``register-bond-development``  needs that pilot PASSED and the root's development/1
  registered AND complete (the stored arms are the comparison); binds the plan, the
  bond spec hash, the stored development.json hash and the decision rule below;
* ``bond-develop``         trains bond/1 on the development/1 training seeds, train
  seed base, curriculum and PPO, selects by its validation seeds, and evaluates the ten
  development cells on the identical environment seeds -- every episode is paired with
  the stored arms' episodes.  Resumable per training seed.

Decision rule (fixed at registration, before any bond model trains): primary = mean
success over the ten cells, bond/1 minus ontology_rgat_ppo, per training seed;
SUPPORTED iff the 95 % t-interval over the five seeds lies above 0, NOT SUPPORTED iff it
lies below 0, otherwise INCONCLUSIVE.  Secondary (descriptive, no verdict): the same on
the evidence-loss cells (dropout_id, dropout_ood, combined_id, combined_ood), the unsafe
rate, and bond/1 minus each other stored arm.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import datetime as dt
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import time
from typing import Any, Callable

from . import SCHEMA_ID
from . import bond
from . import development as dev
from .contracts import contract_manifest, load_config

PROTOCOL = "bond-development/1"
OUT = "bond"
LEARNABILITY = "bond_learnability.json"
REGISTRATION = "bond_development_registration.json"
RESULT = "bond_development.json"
EVIDENCE_LOSS_CELLS = ("dropout_id", "dropout_ood", "combined_id", "combined_ood")
PRIMARY_REFERENCE = "ontology_rgat_ppo"
RULE = ("primary: mean success over the ten development/1 cells, bond/1 minus "
        "ontology_rgat_ppo, paired per training seed; SUPPORTED iff the 95 % t-interval over "
        "the training seeds lies above 0, NOT SUPPORTED iff below 0, else INCONCLUSIVE. "
        "Secondary, descriptive: evidence-loss cells (dropout_id, dropout_ood, combined_id, "
        "combined_ood), unsafe rate, bond/1 minus every other stored arm.")
# two-sided 95 % Student t quantiles by degrees of freedom
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}


def out_dir(root: Path) -> Path:
    return Path(root) / OUT


def spec_record() -> dict[str, Any]:
    return {"arm": bond.ARM, "spec_hash": bond.SPEC_HASH, "spec": bond.spec_dict()}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------- learnability

def learnability(config_path: Path, root: Path, *, log: Callable[[str], None] = print
                 ) -> dict[str, Any]:
    from .pilot import run_pilot
    out = out_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    path = out / LEARNABILITY
    if path.exists():
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("bond", {}).get("spec_hash") != bond.SPEC_HASH:
            raise ValueError(f"{path} holds another bond spec's pilot; archive it first")
        if report.get("protocol_complete"):
            return report
    report = run_pilot(config_path, out / "g4_pilot", arm=bond.ARM, log=log)
    report["bond"] = spec_record()
    from .experiment import write_json
    write_json(path, report)
    return report


# --------------------------------------------------------------- registration

def _stored(root: Path, config, plan) -> tuple[str | None, dict[str, Any] | None]:
    problem = dev.check_registration(config, plan, root)
    if problem:
        return problem, None
    path = Path(root) / dev.RESULT
    if not path.exists():
        return "development/1 has not finished in this root (no development.json)", None
    stored = json.loads(path.read_text(encoding="utf-8"))
    if stored.get("plan_hash") != dev.plan_hash(plan, config) or not stored.get(
            "procedure_complete"):
        return "development.json is not this root's complete development/1", None
    return None, stored


def blocker(config, plan, root: Path) -> str | None:
    problem, _ = _stored(root, config, plan)
    if problem:
        return problem
    path = out_dir(root) / LEARNABILITY
    if not path.exists():
        return "no bond learnability pilot (run --stage bond-learnability)"
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("bond", {}).get("spec_hash") != bond.SPEC_HASH:
        return "the bond pilot ran another bond spec"
    if not (report.get("protocol_complete") and report.get("learnability_passed")):
        return "the bond learnability pilot did not pass"
    return None


def plan_hash(plan, config) -> str:
    from ..direct_policy.contracts import digest
    return digest({"protocol": PROTOCOL, "development": dev.plan_hash(plan, config),
                   "bond": bond.SPEC_HASH, "rule": RULE})


def register(config_path: Path, plan_path: Path, root: Path) -> tuple[int, dict[str, Any]]:
    config = load_config(config_path)
    plan = dev.load_development(plan_path, config)
    refused = {"stage": "register-bond-development", "status": "REFUSED"}
    problem = blocker(config, plan, root)
    if problem:
        return 3, {**refused, "reason": problem}
    digest = plan_hash(plan, config)
    path = out_dir(root) / REGISTRATION
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("plan_hash") == digest:
            return 0, {**old, "status": "ALREADY_REGISTERED"}
        return 3, {**refused, "reason": f"{path} registers another plan; archive it first"}
    acfg = dev.arm_config(config, plan, bond.ARM)
    manifest = contract_manifest(acfg)
    return 0, {"schema_id": SCHEMA_ID, "stage": "register-bond-development",
               "status": "REGISTERED", "protocol": PROTOCOL, "plan_hash": digest,
               "registered_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               "development_plan_hash": dev.plan_hash(plan, config),
               "stored_development_sha256": _sha256(Path(root) / dev.RESULT),
               "bond": spec_record(), "rule": RULE,
               "training_seeds": list(plan.training_seeds),
               "contract": {k: manifest[k] for k in dev.BOUND_HASHES},
               "note": "registered before any bond/1 development model trained"}


def check_registration(config, plan, root: Path) -> str | None:
    problem = blocker(config, plan, root)
    if problem:
        return problem
    path = out_dir(root) / REGISTRATION
    if not path.exists():
        return "no bond development registration (run --stage register-bond-development)"
    old = json.loads(path.read_text(encoding="utf-8"))
    if old.get("plan_hash") != plan_hash(plan, config):
        return "the plan, bond spec or rule differs from the registration"
    if old.get("stored_development_sha256") != _sha256(Path(root) / dev.RESULT):
        return "the stored development.json changed after registration"
    return None


# ------------------------------------------------------------------- develop

def _mean_success(cells: dict[str, Any], names) -> float:
    return sum(cells[c]["success"] for c in names) / len(names)


def _mean_unsafe(cells: dict[str, Any], names) -> float:
    return sum(cells[c]["unsafe"] for c in names) / len(names)


def _interval(diffs: list[float]) -> dict[str, Any]:
    n = len(diffs)
    mean = sum(diffs) / n
    if n < 2:
        return {"mean": mean, "low": None, "high": None, "per_seed": diffs}
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1))
    half = T95[n - 1] * sd / math.sqrt(n)
    return {"mean": mean, "low": mean - half, "high": mean + half, "per_seed": diffs}


def judge(plan, bond_models: dict[int, dict[str, Any]], stored: dict[str, Any]) -> dict[str, Any]:
    seeds = list(plan.training_seeds)
    all_cells, loss_cells = list(dev.CELLS), list(EVIDENCE_LOSS_CELLS)
    arms = list(plan.arms)

    def arm_cells(seed, arm):
        return stored["models"][f"{seed}/{arm}"]["cells"]

    comparisons = {}
    for arm in arms:
        comparisons[arm] = {
            "all_cells": _interval([_mean_success(bond_models[s]["cells"], all_cells)
                                    - _mean_success(arm_cells(s, arm), all_cells)
                                    for s in seeds]),
            "evidence_loss_cells": _interval(
                [_mean_success(bond_models[s]["cells"], loss_cells)
                 - _mean_success(arm_cells(s, arm), loss_cells) for s in seeds]),
            "unsafe_all_cells": _interval([_mean_unsafe(bond_models[s]["cells"], all_cells)
                                           - _mean_unsafe(arm_cells(s, arm), all_cells)
                                           for s in seeds])}
    primary = comparisons[PRIMARY_REFERENCE]["all_cells"]
    verdict = ("INCONCLUSIVE" if primary["low"] is None else
               "SUPPORTED" if primary["low"] > 0 else
               "NOT SUPPORTED" if primary["high"] < 0 else "INCONCLUSIVE")
    by_cell = {c: {"bond": sum(bond_models[s]["cells"][c]["success"] for s in seeds) / len(seeds),
                   **{a: sum(arm_cells(s, a)[c]["success"] for s in seeds) / len(seeds)
                      for a in arms}} for c in all_cells}
    return {"verdict": verdict, "rule": RULE, "primary": primary,
            "comparisons": comparisons, "success_by_cell": by_cell}


def run(config_path: Path, plan_path: Path, root: Path, *,
        log: Callable[[str], None] = print) -> tuple[int, dict[str, Any]]:
    from .models import build_model, load_checkpoint
    from .pilot import _init_worker, train_seed
    config = load_config(config_path)
    plan = dev.load_development(plan_path, config)
    problem = check_registration(config, plan, root)
    if problem:
        return 3, {"stage": "bond-develop", "status": "REFUSED", "reason": problem}
    _, stored = _stored(root, config, plan)
    acfg = dev.arm_config(config, plan, bond.ARM)
    digest = plan_hash(plan, config)
    base = out_dir(root) / "development"
    base.mkdir(parents=True, exist_ok=True)
    models: dict[int, dict[str, Any]] = {}
    started = time.monotonic()
    from .experiment import write_json
    with ProcessPoolExecutor(max_workers=plan.workers,
                             mp_context=multiprocessing.get_context("spawn"),
                             initializer=_init_worker, initargs=(str(config_path),)) as pool:
        for seed_index, seed in enumerate(plan.training_seeds):
            model_root = base / f"seed{seed}"
            summary_path = model_root / "summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if summary.get("plan_hash") != digest:
                    raise RuntimeError(f"{summary_path} belongs to another run; archive it")
                models[seed] = summary
                log(f"bond-develop: seed {seed} already done")
                continue
            log(f"bond-develop: training seed {seed} ({bond.ARM})")
            trained = train_seed(acfg, config_path, seed, model_root, pool, log=log)
            best = model_root / "best.pt"
            summary = {"plan_hash": digest, "training": trained,
                       "validation": dev._validation_of(model_root, trained["best_update"]),
                       "checkpoint": str(best) if best.exists() else None}
            if best.exists():
                model = build_model(bond.ARM, acfg, seed=0)
                load_checkpoint(best, model, manifest=contract_manifest(acfg))
                state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                log(f"bond-develop: evaluating seed {seed} on {len(plan.cells)} cells")
                cells = dev.evaluate_model(pool, plan, bond.ARM, state, seed_index)
                summary["cells"] = {c: {k: v for k, v in x.items() if k != "per_seed"}
                                    for c, x in cells.items()}
                summary["episodes"] = {c: x["per_seed"] for c, x in cells.items()}
            else:
                summary["cells"] = {c: {"episodes": 0, "success": 0.0, "unsafe": 0.0,
                                        "timeout": 0.0, "statuses": {},
                                        "note": "no checkpoint reached difficulty 1.0"}
                                    for c in dev.CELLS}
            write_json(summary_path, summary)
            models[seed] = summary
            log(f"bond-develop: seed {seed} mean success "
                f"{_mean_success(summary['cells'], dev.CELLS):.3f}")
    payload = {"schema_id": SCHEMA_ID, "stage": "bond-develop", "protocol": PROTOCOL,
               "plan_hash": digest, "procedure_complete": True,
               "models": {str(s): {k: v for k, v in m.items() if k != "episodes"}
                          for s, m in models.items()},
               "judgement": judge(plan, models, stored),
               "wall_s": time.monotonic() - started,
               "note": "local plant only; paired with the stored development/1 arms"}
    return 0, payload
