"""Guide 11: the development stage of spatial_current_semantics_v1 (local plant).

Four arms x the registered training seeds, each trained with the G4-passed common
PPO protocol (budget, curriculum, PPO, training randomisation), its checkpoint
selected on the validation seeds, then flown deterministically on ten evaluation
cells (guide 11.2) of 100 environments per training seed that the seed's four
arms share.  The development gate of guide 11.5 is computed from those cells and
nothing else; failing seeds are never dropped and no threshold moves after the
fact.  ``register`` binds the plan file's hash to the main contract before
anything trains; ``run_development`` refuses a plan or contract that differs from
the registration.  The Isaac part (guide 11.6) is separate and needs the user.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
import datetime as dt
import json
import math
import multiprocessing
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import torch
import yaml

from . import SCHEMA_ID
from .contracts import (AUDITED_SEED_RANGES, DIAGNOSTIC_SEED_RANGES, PROJECT_ROOT,
                        CurrentSemanticsConfig, contract_manifest, load_config)

DEVELOPMENT_PROTOCOL = "development/1"
DEFAULT_DEVELOPMENT_PATH = PROJECT_ROOT / "config" / "current_semantics" / "development_v1.yaml"
REGISTRATION = "development_registration.json"
RESULT = "development.json"
CELLS = ("composition_id", "composition_ood", "motion_id", "motion_ood", "dropout_id",
         "dropout_ood", "noise_id", "noise_ood", "combined_id", "combined_ood")
ID_CELLS = tuple(c for c in CELLS if c.endswith("_id"))
OOD_CELLS = tuple(c for c in CELLS if c.endswith("_ood"))
FOUR_OOD = ("motion_ood", "dropout_ood", "noise_ood", "combined_ood")
COMPARISONS = (("mlp", "semantic_mlp_ppo"), ("untyped", "untyped_gnn_ppo"), ("raw", "raw_ppo"))
BOUND_HASHES = ("algorithm_hash", "task_hash", "observation_hash", "feature_hash", "graph_hash",
                "calibration_hash", "g4_hash")
# Seed ranges this stage may not touch beyond the registries: the G4 pilots' training
# episodes (base 3e6 + model seed * 1e6) and every offset the pilots add for actions.
_PILOT_SEEDS = (3_000_000, 4_999_999)


@dataclass(frozen=True)
class EvaluationCell:
    name: str
    split: str
    pad_acceleration_m_s2: tuple[float, float] | None = None
    optical_noise_scale: float = 1.0
    dropout_s: tuple[float, float] | None = None


@dataclass(frozen=True)
class DevelopmentGates:
    validation_deterministic_success_min: float
    validation_sampled_success_min: float
    composition_margin_min: float
    id_drop_max: float
    ood_mean_drop_max: float
    ood_unsafe_increase_max: float


@dataclass(frozen=True)
class DevelopmentPlan:
    protocol: str
    proposed_arm: str
    arms: tuple[str, ...]
    training_seeds: tuple[int, ...]
    train_seed_base: int
    validation_seeds: tuple[int, ...]
    workers: int
    episodes_per_cell: int
    env_seed_base: int
    dropout_start_s: tuple[float, float]
    cells: tuple[EvaluationCell, ...]
    gates: DevelopmentGates

    def cell(self, name: str) -> EvaluationCell:
        return next(c for c in self.cells if c.name == name)

    def train_seed(self, training_seed: int, update: int, episode: int, per_update: int) -> int:
        return self.train_seed_base + training_seed * 1_000_000 + update * per_update + episode

    def eval_seed(self, seed_index: int, cell_index: int, k: int) -> int:
        return self.env_seed_base + seed_index * 100_000 + cell_index * 1_000 + k

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self)))


# ------------------------------------------------------------------- loading

def _section(data: Any, keys: tuple[str, ...], where: str) -> dict:
    if not isinstance(data, dict):
        raise ValueError(f"{where} must be a mapping")
    unknown, missing = set(data) - set(keys), set(keys) - set(data)
    if unknown or missing:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
    return data


def _number(value: Any, where: str, *, low: float = -math.inf) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
            or value < low:
        raise ValueError(f"{where} must be a finite number >= {low}")
    return float(value)


def _range(value: Any, where: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{where} must be [low, high]")
    low, high = (_number(v, where, low=0.0) for v in value)
    if not low < high:
        raise ValueError(f"{where} must have low < high")
    return low, high


def _seeds(value: Any, where: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value or any(
            isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in value):
        raise ValueError(f"{where} must be a non-empty list of non-negative integers")
    if len(set(value)) != len(value):
        raise ValueError(f"{where} must be distinct")
    return tuple(value)


def load_development(path: str | Path, config: CurrentSemanticsConfig) -> DevelopmentPlan:
    """Load and validate the development plan against the main contract."""
    from .models import ARMS
    data = _section(yaml.safe_load(Path(path).read_text(encoding="utf-8")),
                    ("protocol", "proposed_arm", "arms", "training_seeds", "train_seed_base",
                     "validation_seeds", "workers", "evaluation", "gates"), "development")
    if data["protocol"] != DEVELOPMENT_PROTOCOL:
        raise ValueError(f"development.protocol must be {DEVELOPMENT_PROTOCOL!r}")
    if tuple(data["arms"]) != ARMS:
        raise ValueError(f"development.arms must be the four registered arms {ARMS}")
    if data["proposed_arm"] != "ontology_rgat_ppo":
        raise ValueError("development.proposed_arm must be ontology_rgat_ppo (guide 11.3)")
    evaluation = _section(data["evaluation"], ("episodes_per_cell", "env_seed_base",
                                               "dropout_start_s", "cells"),
                          "development.evaluation")
    cells_data = _section(evaluation["cells"], CELLS, "development.evaluation.cells")
    blackout = config.g4.training_randomization.vision_blackout
    cells = []
    for name in CELLS:
        spec = cells_data[name]
        if not isinstance(spec, dict) or "split" not in spec or set(spec) - {
                "split", "pad_acceleration_m_s2", "optical_noise_scale", "dropout_s"}:
            raise ValueError(f"development.evaluation.cells.{name} is malformed")
        cell = EvaluationCell(
            name=name, split=spec["split"],
            pad_acceleration_m_s2=(_range(spec["pad_acceleration_m_s2"], name)
                                   if "pad_acceleration_m_s2" in spec else None),
            optical_noise_scale=_number(spec.get("optical_noise_scale", 1.0), name, low=1.0),
            dropout_s=_range(spec["dropout_s"], name) if "dropout_s" in spec else None)
        cells.append(cell)
    cells = tuple(cells)
    by = {c.name: c for c in cells}
    # Guide 11.2: ID references are the TRAINED ranges, OOD departs from them.
    from .scenario import NOMINAL_PAD_ACCELERATION_M_S2
    trained_dropout = (blackout.duration_min_s, blackout.duration_max_s)
    expect_split = {name: ("composition" if name == "composition_ood" else "in_distribution")
                    for name in CELLS}
    for c in cells:
        if c.split != expect_split[c.name]:
            raise ValueError(f"cell {c.name} must use the {expect_split[c.name]} split")
    for name in ("composition_id", "composition_ood", "motion_id", "noise_id"):
        if (by[name].pad_acceleration_m_s2, by[name].optical_noise_scale, by[name].dropout_s) \
                != (None, 1.0, None):
            raise ValueError(f"cell {name} must be the nominal condition")
    if by["dropout_id"].dropout_s != trained_dropout or by["combined_id"] != replace(
            by["dropout_id"], name="combined_id"):
        raise ValueError("dropout_id and combined_id must carry the trained blackout range "
                         f"{trained_dropout} and nothing else")
    motion, drop, noise = (by["motion_ood"].pad_acceleration_m_s2, by["dropout_ood"].dropout_s,
                           by["noise_ood"].optical_noise_scale)
    if (motion is None or motion[0] < NOMINAL_PAD_ACCELERATION_M_S2[1]
            or (by["motion_ood"].optical_noise_scale, by["motion_ood"].dropout_s) != (1.0, None)):
        raise ValueError("motion_ood must move only the pad acceleration, beyond the trained range")
    if (drop is None or drop[0] < trained_dropout[1]
            or (by["dropout_ood"].pad_acceleration_m_s2, by["dropout_ood"].optical_noise_scale)
            != (None, 1.0)):
        raise ValueError("dropout_ood must move only the dropout, beyond the trained range")
    if not noise > 1.0 or (by["noise_ood"].pad_acceleration_m_s2, by["noise_ood"].dropout_s) \
            != (None, None):
        raise ValueError("noise_ood must move only the optical noise")
    if by["combined_ood"] != EvaluationCell("combined_ood", "in_distribution", motion, noise,
                                            drop):
        raise ValueError("combined_ood must combine exactly the three OOD settings")
    gates = _section(data["gates"], tuple(DevelopmentGates.__dataclass_fields__),
                     "development.gates")
    plan = DevelopmentPlan(
        protocol=DEVELOPMENT_PROTOCOL, proposed_arm="ontology_rgat_ppo", arms=ARMS,
        training_seeds=_seeds(data["training_seeds"], "development.training_seeds"),
        train_seed_base=int(_number(data["train_seed_base"], "train_seed_base", low=0)),
        validation_seeds=_seeds(data["validation_seeds"], "development.validation_seeds"),
        workers=int(_number(data["workers"], "workers", low=1)),
        episodes_per_cell=int(_number(evaluation["episodes_per_cell"], "episodes_per_cell", low=1)),
        env_seed_base=int(_number(evaluation["env_seed_base"], "env_seed_base", low=0)),
        dropout_start_s=_range(evaluation["dropout_start_s"], "dropout_start_s"),
        cells=cells,
        gates=DevelopmentGates(**{k: _number(gates[k], k, low=0.0) for k in gates}))
    _audit_seeds(plan, config)
    return plan


def _audit_seeds(plan: DevelopmentPlan, config: CurrentSemanticsConfig) -> None:
    """Guide 11.4: no seed this stage uses may lie in any recorded or registered use."""
    g4 = config.g4
    per = g4.episodes_per_update
    used = [(name, first, last) for name, first, last, _ in AUDITED_SEED_RANGES]
    used += list(DIAGNOSTIC_SEED_RANGES)
    used += [("g4 pilot training", *_PILOT_SEEDS)]
    taken = set(g4.validation_seeds) | set(g4.test_seeds) | set(config.g3.pilot_seeds) \
        | set(config.isaac_eval.seeds)
    if set(plan.training_seeds) & set(g4.model_seeds):
        raise ValueError("development training seeds reuse a G4 pilot model seed")
    train = [(plan.train_seed(s, 0, 0, per), plan.train_seed(s, g4.updates - 1, per - 1, per))
             for s in plan.training_seeds]
    evaluation = (plan.eval_seed(0, 0, 0),
                  plan.eval_seed(len(plan.training_seeds) - 1, len(CELLS) - 1,
                                 plan.episodes_per_cell - 1))
    if plan.episodes_per_cell > 1000 or len(CELLS) > 100:
        raise ValueError("evaluation seed layout overflows its blocks")
    blocks = [("validation", s, s) for s in plan.validation_seeds] \
        + [("training", a, b) for a, b in train] + [("evaluation", *evaluation)]
    for label, low, high in blocks:
        for name, first, last in used:
            if low <= last and first <= high:
                raise ValueError(f"development {label} seeds {low}-{high} overlap {name}")
        if any(low <= s <= high for s in taken):
            raise ValueError(f"development {label} seeds {low}-{high} overlap a registered seed")
    spans = sorted((a, b) for _, a, b in blocks)
    if any(spans[i][1] >= spans[i + 1][0] for i in range(len(spans) - 1)):
        raise ValueError("development seed blocks overlap each other")


def arm_config(config: CurrentSemanticsConfig, plan: DevelopmentPlan,
               arm: str) -> CurrentSemanticsConfig:
    """The main contract with the G4 protocol re-pointed at one arm and the development
    seeds; budget, curriculum, PPO and randomisation are the G4 pilot's."""
    g4 = replace(config.g4, arm=arm, model_seeds=plan.training_seeds,
                 train_seed_base=plan.train_seed_base, validation_seeds=plan.validation_seeds)
    return replace(config, g4=g4)


def plan_hash(plan: DevelopmentPlan, config: CurrentSemanticsConfig) -> str:
    from ..direct_policy.contracts import digest
    manifest = contract_manifest(config)
    return digest({"plan": plan.to_dict(), "contract": {k: manifest[k] for k in BOUND_HASHES}})


def resolved(plan: DevelopmentPlan, config: CurrentSemanticsConfig) -> dict[str, Any]:
    g4 = config.g4
    models = len(plan.arms) * len(plan.training_seeds)
    return {"protocol": plan.protocol, "plan_hash": plan_hash(plan, config),
            "arms": list(plan.arms), "proposed_arm": plan.proposed_arm,
            "training_seeds": list(plan.training_seeds), "models": models,
            "updates": g4.updates, "episodes_per_update": g4.episodes_per_update,
            "training_episodes": models * g4.updates * g4.episodes_per_update,
            "evaluation_episodes": models * len(CELLS) * plan.episodes_per_cell,
            "cells": {c.name: {k: v for k, v in asdict(c).items() if k != "name"}
                      for c in plan.cells},
            "gates": asdict(plan.gates), "backend": "spatial_local_v12",
            "trains": True, "starts_isaac": False}


# ------------------------------------------------------------- registration

def _g4_passed(config: CurrentSemanticsConfig, root: Path) -> str | None:
    """None when the current contract's G4 pilot passed; otherwise why not."""
    path = root / "g4_pilot.json"
    if not path.exists():
        return "no G4 pilot result for this contract (g4_pilot.json)"
    data = json.loads(path.read_text(encoding="utf-8"))
    manifest = contract_manifest(config)
    stale = [k for k in ("g4_hash", "algorithm_hash", "task_hash", "calibration_hash")
             if data.get("hashes", {}).get(k) != manifest[k]]
    if stale:
        return f"the G4 pilot ran under another contract ({stale})"
    if not (data.get("protocol_complete") and data.get("learnability_passed")):
        return "the G4 pilot under this contract did not pass"
    return None


def register(config_path: Path, plan_path: Path, root: Path) -> tuple[int, dict[str, Any]]:
    config = load_config(config_path)
    plan = load_development(plan_path, config)
    blocker = _g4_passed(config, root)
    if blocker:
        return 3, {"stage": "register-development", "status": "REFUSED", "reason": blocker}
    digest_ = plan_hash(plan, config)
    path = root / REGISTRATION
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("plan_hash") == digest_:
            return 0, {**old, "status": "ALREADY_REGISTERED"}
        return 3, {"stage": "register-development", "status": "REFUSED",
                   "reason": f"{path.name} registers another plan ({old.get('plan_hash')}); "
                             "archive it before registering a new one"}
    manifest = contract_manifest(config)
    payload = {"schema_id": SCHEMA_ID, "stage": "register-development", "status": "REGISTERED",
               "registered_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               "plan_file": str(plan_path), "plan": plan.to_dict(), **resolved(plan, config),
               "contract": {k: manifest[k] for k in BOUND_HASHES},
               "seed_audit": "no seed of this plan lies in AUDITED_SEED_RANGES, "
                             "DIAGNOSTIC_SEED_RANGES, the G4 pilots' training range, or any "
                             "registered validation/test/G3/Isaac seed (load_development)"}
    return 0, payload


def check_registration(config: CurrentSemanticsConfig, plan: DevelopmentPlan,
                       root: Path) -> str | None:
    path = root / REGISTRATION
    if not path.exists():
        return "no development registration: run --stage register-development first"
    old = json.loads(path.read_text(encoding="utf-8"))
    if old.get("plan_hash") != plan_hash(plan, config):
        return ("the plan or the contract differs from the registration "
                f"({old.get('plan_hash')}); develop refuses any override")
    return None


# --------------------------------------------------------------- evaluation

def cell_conditions(config: CurrentSemanticsConfig, cell: dict[str, Any],
                    start: tuple[float, float]) -> tuple[Any, Any]:
    """(PlantFactors, evaluation dropout or None) of one cell (``asdict(EvaluationCell)``):
    the one definition the local cells and the Isaac evaluation (guide 11.6) share."""
    from .scenario import PlantFactors
    blackout = config.g4.training_randomization.vision_blackout
    dropout = (None if cell["dropout_s"] is None else replace(
        blackout, probability=1.0, start_min_s=start[0], start_max_s=start[1],
        duration_min_s=cell["dropout_s"][0], duration_max_s=cell["dropout_s"][1]))
    factors = PlantFactors(
        pad_acceleration_m_s2=(None if cell["pad_acceleration_m_s2"] is None
                               else tuple(cell["pad_acceleration_m_s2"])),
        optical_noise_scale=cell["optical_noise_scale"])
    return factors, dropout


def _cell_work(task: tuple) -> dict[str, Any]:
    from .pilot import _WORKER, run_episode
    arm, state, seed, cell, start, action_seed = task
    config = _WORKER["config"]
    factors, dropout = cell_conditions(config, cell, start)
    episode = run_episode(config, arm, state, seed=seed, difficulty=1.0, deterministic=True,
                          action_seed=action_seed, split=cell["split"], factors=factors,
                          dropout=dropout)
    return {key: episode.get(key) for key in ("seed", "refused", "status", "success",
                                              "decisions", "return", "scenario")}


def unsafe(status: str | None) -> bool:
    from ..spatial.environment import Evaluator
    return status in Evaluator.UNSAFE_STATUSES


def summarize_cell(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    flown = [e for e in episodes if not e.get("refused")]
    statuses: dict[str, int] = {}
    for e in flown:
        statuses[e["status"]] = statuses.get(e["status"], 0) + 1
    n = len(flown)
    return {"episodes": n, "refused": len(episodes) - n,
            "success": sum(e["status"] == "SUCCESS" for e in flown) / n if n else None,
            "unsafe": sum(unsafe(e["status"]) for e in flown) / n if n else None,
            "timeout": (statuses.get("TASK_TIMEOUT") or 0) / n if n else None,
            "statuses": statuses}


def evaluate_model(pool: Any, plan: DevelopmentPlan, arm: str, state: dict[str, Any],
                   seed_index: int, *,
                   progress: Callable[[str, int], None] | None = None) -> dict[str, Any]:
    out = {}
    for cell_index, cell in enumerate(plan.cells):
        if progress is not None:
            progress(cell.name, cell_index)
        seeds = [plan.eval_seed(seed_index, cell_index, k) for k in range(plan.episodes_per_cell)]
        spec = asdict(cell)
        tasks = [(arm, state, s, spec, plan.dropout_start_s, s + 9_000_000) for s in seeds]
        episodes = list(pool.map(_cell_work, tasks, chunksize=2))
        out[cell.name] = {**summarize_cell(episodes), "per_seed": episodes}
    return out


# --------------------------------------------------------------------- gate

def judge(plan: DevelopmentPlan, models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Guide 11.5 on ``models[f"{training_seed}/{arm}"]`` = {validation, cells}."""
    g = plan.gates
    seeds = [str(s) for s in plan.training_seeds]
    proposed = plan.proposed_arm

    def rate(seed, arm, cell, key="success"):
        return models[f"{seed}/{arm}"]["cells"][cell][key]

    def mean(values):
        return float(np.mean(values))

    validation = {}
    for key, model in models.items():
        v = model.get("validation") or {}
        det, smp = v.get("deterministic_success"), v.get("sampled_success")
        validation[key] = {"deterministic_success": det, "sampled_success": smp,
                           "passed": (det is not None and smp is not None
                                      and det >= g.validation_deterministic_success_min
                                      and smp >= g.validation_sampled_success_min)}
    margins = {}
    for label, arm in COMPARISONS:
        diffs = [rate(s, proposed, "composition_ood") - rate(s, arm, "composition_ood")
                 for s in seeds]
        margins[label] = {"per_seed": diffs, "mean": mean(diffs),
                          "passed": mean(diffs) > g.composition_margin_min}
    id_diff = [mean([rate(s, proposed, c) for c in ID_CELLS])
               - mean([rate(s, "raw_ppo", c) for c in ID_CELLS]) for s in seeds]
    ood_diff = [mean([rate(s, proposed, c) for c in FOUR_OOD])
                - mean([rate(s, "raw_ppo", c) for c in FOUR_OOD]) for s in seeds]
    unsafe_diff = [mean([rate(s, proposed, c, "unsafe") for c in OOD_CELLS])
                   - mean([rate(s, "raw_ppo", c, "unsafe") for c in OOD_CELLS]) for s in seeds]
    guards = {
        "id_success_vs_raw": {"per_seed": id_diff, "mean": mean(id_diff),
                              "passed": mean(id_diff) >= -g.id_drop_max},
        "four_ood_success_vs_raw": {"per_seed": ood_diff, "mean": mean(ood_diff),
                                    "passed": mean(ood_diff) >= -g.ood_mean_drop_max},
        "ood_unsafe_vs_raw": {"per_seed": unsafe_diff, "mean": mean(unsafe_diff),
                              "passed": mean(unsafe_diff) <= g.ood_unsafe_increase_max}}
    learnable = all(v["passed"] for v in validation.values())
    passed = (learnable and all(m["passed"] for m in margins.values())
              and all(x["passed"] for x in guards.values()))
    table = {arm: {cell: mean([rate(s, arm, cell) for s in seeds]) for cell in CELLS}
             for arm in plan.arms}
    unsafe_table = {arm: {cell: mean([rate(s, arm, cell, "unsafe") for s in seeds])
                          for cell in CELLS} for arm in plan.arms}
    return {"learnability_passed": learnable, "validation": validation,
            "composition_margins": margins, "guards": guards,
            "development_gate_passed": passed, "confirmation_opened": passed,
            "success_by_arm_and_cell": table, "unsafe_by_arm_and_cell": unsafe_table}


# ---------------------------------------------------------------------- run

STATUS = "status.json"  # per-model phase, read by the live dashboard (dashboard.py)


def _status(model_root: Path, phase: str, **detail: Any) -> None:
    from .experiment import write_json
    model_root.mkdir(parents=True, exist_ok=True)
    write_json(model_root / STATUS, {"phase": phase, "utc": dt.datetime.now(
        dt.timezone.utc).isoformat(timespec="seconds"), **detail})


def _validation_of(model_root: Path, best_update: int | None) -> dict[str, Any]:
    rows = [json.loads(line) for line in (model_root / "validation.jsonl").open()] \
        if (model_root / "validation.jsonl").exists() else []
    row = next((r for r in rows if r["update"] == best_update), None)
    if row is None:
        return {"best_update": best_update, "deterministic_success": None, "sampled_success": None}
    return {"best_update": best_update,
            "deterministic_success": row["deterministic"]["success"],
            "sampled_success": row["sampled"]["success"]}


def run_development(config_path: Path, plan_path: Path, root: Path, *,
                    log: Callable[[str], None] = print,
                    updates: int | None = None, episodes_per_cell: int | None = None,
                    workers: int | None = None) -> tuple[int, dict[str, Any]]:
    """Train, select and evaluate every (training seed, arm); resumable per model.
    ``updates``/``episodes_per_cell``/``workers`` exist for tests only: the result is
    then marked incomplete and carries no gate."""
    from .models import CurrentActorCritic, load_checkpoint
    from .pilot import _init_worker, train_seed
    config = load_config(config_path)
    plan = load_development(plan_path, config)
    blocker = check_registration(config, plan, root)
    if blocker:
        return 3, {"stage": "develop", "status": "REFUSED", "reason": blocker}
    if episodes_per_cell is not None:
        plan = replace(plan, episodes_per_cell=int(episodes_per_cell))
    complete = updates is None and episodes_per_cell is None
    out_root = root / "development"
    out_root.mkdir(parents=True, exist_ok=True)
    models: dict[str, dict[str, Any]] = {}
    started = time.monotonic()
    count = plan.workers if workers is None else int(workers)
    with ProcessPoolExecutor(max_workers=count, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_init_worker, initargs=(str(config_path),)) as pool:
        for seed_index, training_seed in enumerate(plan.training_seeds):
            for arm in plan.arms:
                key = f"{training_seed}/{arm}"
                model_root = out_root / f"seed{training_seed}" / arm
                summary_path = model_root / "summary.json"
                acfg = arm_config(config, plan, arm)
                if summary_path.exists():
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    if summary.get("plan_hash") != plan_hash(plan, config) or \
                            summary.get("complete") != complete:
                        raise RuntimeError(f"{summary_path} belongs to another run; archive it")
                    models[key] = summary
                    log(f"development: {key} already done")
                    continue
                log(f"development: training {key}")
                _status(model_root, "training")
                trained = train_seed(acfg, config_path, training_seed, model_root, pool,
                                     log=log, updates=updates)
                best = model_root / "best.pt"
                summary = {"plan_hash": plan_hash(plan, config), "complete": complete,
                           "training": trained,
                           "validation": _validation_of(model_root, trained["best_update"]),
                           "checkpoint": str(best) if best.exists() else None}
                if best.exists():
                    model = CurrentActorCritic(arm, acfg, seed=0)
                    load_checkpoint(best, model, manifest=contract_manifest(acfg))
                    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    log(f"development: evaluating {key} on {len(plan.cells)} cells")
                    summary["cells"] = evaluate_model(
                        pool, plan, arm, state, seed_index,
                        progress=lambda name, index, root_=model_root: _status(
                            root_, "evaluating", cell=name, cell_index=index,
                            cells=len(plan.cells)))
                else:
                    summary["cells"] = {c: {"episodes": 0, "success": 0.0, "unsafe": 0.0,
                                            "timeout": 0.0, "statuses": {},
                                            "note": "no checkpoint reached difficulty 1.0"}
                                        for c in CELLS}
                from .experiment import write_json
                write_json(summary_path, summary)
                _status(model_root, "done")
                models[key] = summary
                log(f"development: {key} composition_ood "
                    f"{summary['cells']['composition_ood']['success']} "
                    f"composition_id {summary['cells']['composition_id']['success']}")
    verdict = judge(plan, models) if complete else None
    payload = {"schema_id": SCHEMA_ID, "stage": "develop", "protocol": plan.protocol,
               "plan_hash": plan_hash(plan, config), "procedure_complete": complete,
               "models": {k: {kk: vv for kk, vv in v.items() if kk != "cells"}
                          | {"cells": {c: {kk: vv for kk, vv in cv.items() if kk != "per_seed"}
                                       for c, cv in v["cells"].items()}}
                          for k, v in models.items()},
               "verdict": verdict, "wall_s": time.monotonic() - started,
               "note": ("local plant only; the Isaac part of guide 11.6 is separate and needs "
                        "the user's approval")}
    return 0, payload
