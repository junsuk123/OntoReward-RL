"""equal-budget/1 and isaac-equal-budget/1: does ORL's faster curriculum (learning-speed/1)
become better landing at the SAME training budget, on the local plant and on Isaac/PX4?

learning-speed/1 (status doc 21.3) found ORL (``orl/1``) reaching difficulty 1.0 in about
0.55x the updates of PPO and the untyped GNN on fresh seeds.  It measured the curriculum,
not landing.  This protocol measures landing after a fixed number of PPO updates.

Procedure (equal-budget/1).  Each training seed trains the proposed arm and every comparison
arm with the base contract's G4 recipe (budget per update, curriculum, PPO settings, training
randomisation) for ``max(budgets)`` updates, the same episode seeds for all arms of a seed.
The weights after exactly B updates, for every registered budget B, are kept
(``pilot.train_seed(snapshots=...)``); no checkpoint is selected and no validation runs.
Each budget's weights fly the ten development cells (``development.cell_conditions``,
deterministic policy) on environments shared by every arm and budget of a training seed.

Decision rule (``RULE``, hashed into the registration).  A model's score at budget B is
its success rate averaged over the ten cells.  Per comparison arm X and budget B the paired
difference per training seed is score(ORL) - score(X).  The budgets are tested as a fixed
sequence, in the registered order (100, 200, 600): at each budget the two-sided paired-t
interval at 1 - alpha/2 (Bonferroni over the two comparisons) gives BETTER (low > 0),
WORSE (high < 0) or NOT ESTABLISHED; the sequence goes on to the next budget only after a
BETTER or WORSE verdict, so later budgets of a stopped sequence carry no verdict.  The fixed
sequence keeps the family-wise error at alpha without splitting it over budgets.

Procedure (isaac-equal-budget/1).  Every model's weights at one registered budget fly the
same Isaac cases (development cells without the local-only optical noise factor, fresh
environment seeds), deterministically, on an owned Isaac Sim + PX4 stack; the identical
cases also fly on the local plant for a paired transfer record.  Registration binds the
sha256 of every checkpoint before anything flies.  ``ISAAC_RULE``: per comparison arm, the
paired-t interval at 1 - alpha/2 of the per-seed difference in Isaac success rate gives
BETTER / WORSE / NOT ESTABLISHED.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
import datetime as dt
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import yaml

from . import development as dev
from . import orl
from . import speed
from .contracts import (AUDITED_SEED_RANGES, DIAGNOSTIC_SEED_RANGES, PROJECT_ROOT,
                        CurrentSemanticsConfig, contract_manifest, load_config)

PROTOCOL = "equal-budget/1"
ISAAC_PROTOCOL = "isaac-equal-budget/1"
DEFAULT_PATH = PROJECT_ROOT / "config" / "current_semantics" / "equal_budget_v1.yaml"
ISAAC_DEFAULT_PATH = PROJECT_ROOT / "config" / "current_semantics" / "isaac_equal_budget_v1.yaml"
FOLDER = "equal_budget"                      # <root>/equal_budget/
REGISTRATION = "equal_budget_registration.json"
RESULT = "equal_budget.json"
ISAAC_FOLDER = "isaac_equal_budget"          # <root>/isaac_equal_budget/
ISAAC_REGISTRATION = "isaac_equal_budget_registration.json"
ISAAC_RESULT = "isaac_equal_budget.json"
LOCAL_REFERENCE = "local_reference.jsonl"
RULE = (
    "score(model, B) = success rate averaged over the ten development cells, deterministic "
    "policy, weights after exactly B updates (no selection).  Per comparison arm X and budget "
    "B: d = score(proposed, B) - score(X, B) per training seed; two-sided paired-t interval "
    "of mean d at 1 - family_alpha/2 (Bonferroni over the comparisons); BETTER iff low > 0, "
    "WORSE iff high < 0, otherwise not established.  Budgets are a fixed sequence in the "
    "registered order: the next budget is tested only after a BETTER or WORSE verdict; a "
    "stopped sequence's later budgets are reported without a verdict.")
ISAAC_RULE = (
    "per model: Isaac success rate over the registered cases (deterministic).  Per "
    "comparison arm X: d = rate(proposed) - rate(X) per training seed; two-sided paired-t "
    "interval of mean d at 1 - family_alpha/2 (Bonferroni over the comparisons); BETTER iff "
    "low > 0, WORSE iff high < 0, otherwise not established.  The paired local-plant "
    "verdicts on the identical cases are reported, never decisive.")
SPENT_MODEL_SEEDS = speed.SPENT_MODEL_SEEDS | frozenset(range(40, 50))   # + learning-speed/1
SPENT_TRAINING_SEEDS = (*speed.SPENT_TRAINING_SEEDS,
                        ("learning-speed/1 training", 340_000_000, 349_007_199))
SPENT_LOCAL_EVALUATION = (("confirmation/1 evaluation (archived)", 200_000_000, 209_999_999),)
SPENT_ISAAC_SEEDS = (*speed.SPENT_ISAAC_SEEDS,
                     ("confirmation/1 Isaac environments (archived)", 994_201, 994_230),
                     ("monitor probe and demo flights", 999_901, 999_904))
ACTION_OFFSET = 9_000_000      # the development cells' action seed = environment seed + this


# ----------------------------------------------------------------------------- plans

@dataclass(frozen=True)
class BudgetPlan:
    protocol: str
    proposed: str                       # an ORL version, e.g. "orl/1"
    comparisons: tuple[str, ...]        # registered arm names
    training_seeds: tuple[int, ...]
    train_seed_base: int
    budgets: tuple[int, ...]            # strictly increasing: the fixed testing sequence
    workers: int
    family_alpha: float
    episodes_per_cell: int
    env_seed_base: int
    cells_from: str                     # the development protocol whose cells are flown
    dropout_start_s: tuple[float, float]
    cells: tuple[dev.EvaluationCell, ...]

    @property
    def proposed_arm(self) -> str:
        return orl.ARMS[self.proposed]

    @property
    def arms(self) -> tuple[str, ...]:
        return (self.proposed_arm, *self.comparisons)

    @property
    def updates(self) -> int:
        return max(self.budgets)

    def cell(self, name: str) -> dev.EvaluationCell:
        return next(c for c in self.cells if c.name == name)

    def train_seed(self, model_seed: int, update: int, episode: int, per: int) -> int:
        return self.train_seed_base + model_seed * 1_000_000 + update * per + episode

    def eval_seed(self, seed_index: int, cell_index: int, k: int) -> int:
        return self.env_seed_base + seed_index * 100_000 + cell_index * 1_000 + k

    def to_dict(self) -> dict[str, Any]:
        return {**json.loads(json.dumps(asdict(self))), "proposed_arm": self.proposed_arm}


@dataclass(frozen=True)
class IsaacBudgetPlan:
    protocol: str
    source_protocol: str
    budget: int
    mode: str
    family_alpha: float
    conditions: tuple[tuple[str, tuple[int, ...]], ...]   # (development cell, environments)

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(s for _, seeds in self.conditions for s in seeds)

    def to_dict(self) -> dict[str, Any]:
        return {"protocol": self.protocol, "source_protocol": self.source_protocol,
                "budget": self.budget, "mode": self.mode, "family_alpha": self.family_alpha,
                "conditions": {name: list(seeds) for name, seeds in self.conditions}}


def _check_contract(config: CurrentSemanticsConfig) -> None:
    if config.navigation is not None:
        raise ValueError(f"{PROTOCOL} runs on the base navigation contract only "
                         "(spatial_view_contact_v1.yaml); this config carries a navigation "
                         "estimator")


def load_plan(path: str | Path, config: CurrentSemanticsConfig) -> BudgetPlan:
    data = dev._section(yaml.safe_load(Path(path).read_text(encoding="utf-8")),
                        ("protocol", "proposed", "comparisons", "training_seeds",
                         "train_seed_base", "budgets", "workers", "family_alpha", "evaluation"),
                        "equal_budget")
    if data["protocol"] != PROTOCOL:
        raise ValueError(f"equal_budget.protocol must be {PROTOCOL!r}")
    if data["proposed"] not in orl.ARMS:
        raise ValueError(f"equal_budget.proposed must be one of {sorted(orl.ARMS)}")
    comparisons = tuple(data["comparisons"])
    if not comparisons or len(set(comparisons)) != len(comparisons) \
            or not set(comparisons) <= set(speed.COMPARABLE):
        raise ValueError(f"equal_budget.comparisons must be distinct arms of {speed.COMPARABLE}")
    seeds = dev._seeds(data["training_seeds"], "equal_budget.training_seeds")
    if len(seeds) < 3:
        raise ValueError("a paired-t comparison needs at least three training seeds")
    budgets = tuple(int(dev._number(b, "equal_budget.budgets", low=1))
                    for b in (data["budgets"] or []))
    if not budgets or any(a >= b for a, b in zip(budgets, budgets[1:])):
        raise ValueError("equal_budget.budgets must be a strictly increasing list")
    evaluation = dev._section(data["evaluation"], ("episodes_per_cell", "env_seed_base",
                                                   "cells_from"), "equal_budget.evaluation")
    development = dev.load_development(dev.DEFAULT_DEVELOPMENT_PATH, config)
    if evaluation["cells_from"] != development.protocol:
        raise ValueError("equal_budget.evaluation.cells_from must name the registered "
                         f"development protocol {development.protocol!r}")
    episodes = int(dev._number(evaluation["episodes_per_cell"], "episodes_per_cell", low=1))
    if episodes > 1000:
        raise ValueError("episodes_per_cell must be at most 1000 (one seed block per cell)")
    plan = BudgetPlan(
        protocol=PROTOCOL, proposed=data["proposed"], comparisons=comparisons,
        training_seeds=seeds,
        train_seed_base=int(dev._number(data["train_seed_base"], "train_seed_base", low=0)),
        budgets=budgets, workers=int(dev._number(data["workers"], "workers", low=1)),
        family_alpha=speed._alpha(data["family_alpha"], "family_alpha"),
        episodes_per_cell=episodes,
        env_seed_base=int(dev._number(evaluation["env_seed_base"], "env_seed_base", low=0)),
        cells_from=development.protocol, dropout_start_s=development.dropout_start_s,
        cells=development.cells)
    if plan.updates > config.g4.updates:
        raise ValueError(f"the largest budget exceeds the G4 recipe's {config.g4.updates} updates")
    _check_contract(config)
    _audit(plan, config, development)
    return plan


def _used_ranges(config: CurrentSemanticsConfig,
                 development: dev.DevelopmentPlan) -> list[tuple[str, int, int]]:
    used = [(name, first, last) for name, first, last, _ in AUDITED_SEED_RANGES]
    used += list(DIAGNOSTIC_SEED_RANGES) + list(dev.SPENT_DEVELOPMENT_SEEDS)
    used += list(SPENT_ISAAC_SEEDS) + list(SPENT_TRAINING_SEEDS) + list(SPENT_LOCAL_EVALUATION)
    used += [("g4 pilot training", *dev._PILOT_SEEDS)]
    return used


def _taken(config: CurrentSemanticsConfig, development: dev.DevelopmentPlan) -> set[int]:
    return (set(config.g4.validation_seeds) | set(config.g4.test_seeds)
            | set(config.g3.pilot_seeds) | set(config.isaac_eval.seeds)
            | set(development.validation_seeds))


def _audit(plan: BudgetPlan, config: CurrentSemanticsConfig,
           development: dev.DevelopmentPlan) -> None:
    """No seed of this plan may lie in any recorded or registered use, or in another block
    of this plan."""
    g4 = config.g4
    per = g4.episodes_per_update
    if set(plan.training_seeds) & (SPENT_MODEL_SEEDS | set(g4.model_seeds)):
        raise ValueError("equal-budget training seeds reuse a G4, development, confirmation "
                         "or learning-speed model seed")
    last = len(plan.training_seeds) - 1
    blocks = [(f"training seed {s}", plan.train_seed(s, 0, 0, per),
               plan.train_seed(s, plan.updates - 1, per - 1, per)) for s in plan.training_seeds]
    low = plan.eval_seed(0, 0, 0)
    high = plan.eval_seed(last, len(plan.cells) - 1, plan.episodes_per_cell - 1)
    blocks += [("evaluation environments", low, high),
               ("evaluation action seeds", low + ACTION_OFFSET, high + ACTION_OFFSET)]
    used = _used_ranges(config, development)
    taken = _taken(config, development)
    for name, first, final in blocks:
        for other, a, b in used:
            if first <= b and a <= final:
                raise ValueError(f"equal-budget {name} ({first}-{final}) overlaps {other}")
        if any(first <= s <= final for s in taken):
            raise ValueError(f"equal-budget {name} ({first}-{final}) overlaps a registered seed")
    spans = sorted((a, b) for _, a, b in blocks)
    if any(spans[i][1] >= spans[i + 1][0] for i in range(len(spans) - 1)):
        raise ValueError("equal-budget seed blocks overlap each other")


def load_isaac_plan(path: str | Path, config: CurrentSemanticsConfig,
                    plan: BudgetPlan) -> IsaacBudgetPlan:
    data = dev._section(yaml.safe_load(Path(path).read_text(encoding="utf-8")),
                        ("protocol", "source_protocol", "budget", "mode", "family_alpha",
                         "conditions"), "isaac_equal_budget")
    if data["protocol"] != ISAAC_PROTOCOL:
        raise ValueError(f"isaac_equal_budget.protocol must be {ISAAC_PROTOCOL!r}")
    if data["source_protocol"] != plan.protocol:
        raise ValueError(f"isaac_equal_budget.source_protocol must be {plan.protocol!r}")
    budget = int(dev._number(data["budget"], "isaac_equal_budget.budget", low=1))
    if budget not in plan.budgets:
        raise ValueError(f"isaac_equal_budget.budget must be one of {plan.budgets}")
    if data["mode"] != "deterministic":
        raise ValueError("isaac_equal_budget.mode must be deterministic")
    if not isinstance(data["conditions"], dict) or not data["conditions"]:
        raise ValueError("isaac_equal_budget.conditions must map development cells to seeds")
    conditions = []
    for name, seeds in data["conditions"].items():
        if name not in dev.CELLS:
            raise ValueError(f"isaac_equal_budget condition {name!r} is not a development cell")
        if plan.cell(name).optical_noise_scale != 1.0:
            raise ValueError(f"{name} scales the optical noise, which only the local plant "
                             "applies; Isaac cannot fly it")
        conditions.append((name, dev._seeds(seeds, f"isaac_equal_budget.conditions.{name}")))
    iplan = IsaacBudgetPlan(protocol=ISAAC_PROTOCOL, source_protocol=plan.protocol,
                            budget=budget, mode="deterministic",
                            family_alpha=speed._alpha(data["family_alpha"], "family_alpha"),
                            conditions=tuple(conditions))
    if len(set(iplan.seeds)) != len(iplan.seeds):
        raise ValueError("isaac_equal_budget seeds must be distinct across conditions")
    development = dev.load_development(dev.DEFAULT_DEVELOPMENT_PATH, config)
    used = _used_ranges(config, development)
    taken = _taken(config, development)
    last = len(plan.training_seeds) - 1
    used += [("equal-budget evaluation environments", plan.eval_seed(0, 0, 0),
              plan.eval_seed(last, len(plan.cells) - 1, plan.episodes_per_cell - 1))]
    for seed in iplan.seeds:
        hits = [name for name, first, final in used if first <= seed <= final]
        if hits or seed in taken:
            raise ValueError(f"isaac_equal_budget seed {seed} is already used "
                             f"({hits or 'registered'})")
    return iplan


def arm_config(config: CurrentSemanticsConfig, plan: BudgetPlan, arm: str
               ) -> CurrentSemanticsConfig:
    """The base contract's G4 recipe re-pointed at one arm, this plan's seeds and budget."""
    return replace(config, g4=replace(config.g4, arm=arm, model_seeds=plan.training_seeds,
                                      train_seed_base=plan.train_seed_base,
                                      updates=plan.updates))


def plan_hash(plan: BudgetPlan, config: CurrentSemanticsConfig) -> str:
    from ..direct_policy.contracts import digest
    manifest = contract_manifest(config)
    return digest({"plan": plan.to_dict(), "rule": RULE, "orl_hash": orl.HASHES[plan.proposed],
                   "contract": {k: manifest[k] for k in dev.BOUND_HASHES}})


def isaac_plan_hash(iplan: IsaacBudgetPlan, source_hash: str) -> str:
    from ..direct_policy.contracts import digest
    return digest({"plan": iplan.to_dict(), "rule": ISAAC_RULE, "source_plan_hash": source_hash})


def out_dir(root: Path) -> Path:
    return Path(root) / FOLDER


def isaac_dir(root: Path) -> Path:
    return Path(root) / ISAAC_FOLDER


def model_keys(plan: BudgetPlan) -> tuple[str, ...]:
    """Seed-major, so the arms of one seed train close together in time."""
    return tuple(f"{s}/{arm}" for s in plan.training_seeds for arm in plan.arms)


def model_dir(root: Path, key: str) -> Path:
    seed, arm = key.split("/", 1)
    return out_dir(root) / f"seed{seed}" / speed.label(arm)


def checkpoint(root: Path, key: str, budget: int) -> Path:
    return model_dir(root, key) / f"budget_{budget}.pt"


def resolved(plan: BudgetPlan, config: CurrentSemanticsConfig) -> dict[str, Any]:
    n = len(model_keys(plan))
    per_model = len(plan.budgets) * len(plan.cells) * plan.episodes_per_cell
    return {"protocol": PROTOCOL, "plan_hash": plan_hash(plan, config),
            "proposed_arm": plan.proposed_arm, "comparisons": list(plan.comparisons),
            "training_seeds": list(plan.training_seeds), "models": n,
            "budgets": list(plan.budgets), "updates_per_model": plan.updates,
            "episodes_per_update": config.g4.episodes_per_update,
            "training_episodes": n * plan.updates * config.g4.episodes_per_update,
            "evaluation_episodes": n * per_model, "cells": [c.name for c in plan.cells],
            "episodes_per_cell": plan.episodes_per_cell, "rule": RULE,
            "family_alpha": plan.family_alpha, "selects_checkpoints": False,
            "starts_isaac": False}


# ----------------------------------------------------------------- registration

def register(config_path: Path, plan_path: Path, root: Path) -> tuple[int, dict[str, Any]]:
    """Write the registration before any model trains; an existing one must match."""
    from .experiment import write_json
    config = load_config(config_path)
    plan = load_plan(plan_path, config)
    payload = {**resolved(plan, config), "plan_file": str(Path(plan_path).resolve())}
    path = out_dir(root) / REGISTRATION
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("plan_hash") != payload["plan_hash"]:
            return 3, {"stage": "register-budget", "status": "REFUSED",
                       "reason": f"{path} registers another plan"}
        return 0, {"stage": "register-budget", "status": "ALREADY_REGISTERED", **payload}
    if any(out_dir(root).glob("seed*/*/history.jsonl")):
        return 3, {"stage": "register-budget", "status": "REFUSED",
                   "reason": "models were trained before the registration; archive them"}
    payload = {**payload, "registered_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)
    return 0, {"stage": "register-budget", "status": "REGISTERED", **payload}


def check_registration(plan: BudgetPlan, config: CurrentSemanticsConfig, root: Path
                       ) -> str | None:
    path = out_dir(root) / REGISTRATION
    if not path.exists():
        return f"{PROTOCOL} is not registered ({path}); run --stage register-budget first"
    stored = json.loads(path.read_text(encoding="utf-8"))
    if stored.get("plan_hash") != plan_hash(plan, config):
        return f"{path} registers another plan or contract; the plan file or code changed"
    return None


# ------------------------------------------------------------- train and evaluate

def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _trained(directory: Path, plan: BudgetPlan) -> bool:
    """A finished training run of this model: every budget's weights and the full history."""
    history = directory / "history.jsonl"
    return ((directory / "final.pt").exists() and history.exists()
            and all((directory / f"budget_{b}.pt").exists() for b in plan.budgets)
            and len(_rows(history)) == plan.updates)


def load_weights(acfg: CurrentSemanticsConfig, arm: str, path: Path):
    from .models import build_model, load_checkpoint
    model = build_model(arm, acfg, seed=0)
    meta = load_checkpoint(path, model, manifest=contract_manifest(acfg))
    model.eval()
    return model, meta


def evaluate_budget(pool: Any, plan: BudgetPlan, arm: str, state: dict[str, Any],
                    seed_index: int, stream: Path, *,
                    progress: Callable[[str, int], None] | None = None) -> dict[str, Any]:
    """The ten cells for one budget's weights; every episode is appended to ``stream`` and a
    complete stream is reused (an interrupted evaluation resumes cell by cell)."""
    done: dict[str, list[dict[str, Any]]] = {}
    if stream.exists():
        for row in _rows(stream):
            done.setdefault(row["cell"], []).append(row)
    out = {}
    with stream.open("a", encoding="utf-8") as handle:
        for cell_index, cell in enumerate(plan.cells):
            have = done.get(cell.name, [])
            if len(have) != plan.episodes_per_cell:
                if have:
                    raise RuntimeError(f"{stream} holds a partial {cell.name}; archive it")
                if progress is not None:
                    progress(cell.name, cell_index)
                seeds = [plan.eval_seed(seed_index, cell_index, k)
                         for k in range(plan.episodes_per_cell)]
                spec = asdict(cell)
                tasks = [(arm, state, s, spec, plan.dropout_start_s, s + ACTION_OFFSET)
                         for s in seeds]
                have = [{"cell": cell.name, **e}
                        for e in pool.map(dev._cell_work, tasks, chunksize=2)]
                handle.write("".join(json.dumps(r, allow_nan=False) + "\n" for r in have))
                handle.flush()
            out[cell.name] = dev.summarize_cell(have)
    success = [out[c.name]["success"] for c in plan.cells]
    unsafe = [out[c.name]["unsafe"] for c in plan.cells]
    return {"cells": out,
            "mean_success": float(np.mean(success)) if None not in success else None,
            "mean_unsafe": float(np.mean(unsafe)) if None not in unsafe else None}


def run(config_path: Path, plan_path: Path, root: Path, *,
        log: Callable[[str], None] = print, workers: int | None = None) -> tuple[int, dict]:
    """Train and evaluate every registered model nobody holds; any number of processes
    share them.  A model whose training finished but whose evaluation did not is evaluated
    without retraining."""
    import torch
    from .experiment import write_json
    from .pilot import _init_worker, train_seed
    if torch.get_num_threads() != 1:
        return 3, {"stage": "budget", "status": "REFUSED",
                   "reason": f"torch uses {torch.get_num_threads()} threads; training is "
                             "deterministic only single-threaded (run through run.sh)"}
    config = load_config(config_path)
    plan = load_plan(plan_path, config)
    problem = check_registration(plan, config, root)
    if problem:
        return 3, {"stage": "budget", "status": "REFUSED", "reason": problem}
    digest = plan_hash(plan, config)
    keys = model_keys(plan)
    pending = [k for k in keys if speed._summary(model_dir(root, k) / "summary.json",
                                                 digest) is None]
    log(f"budget: {len(keys) - len(pending)} of {len(keys)} models done")
    done_here: list[str] = []
    started = time.monotonic()
    if pending:
        with ProcessPoolExecutor(max_workers=int(workers or plan.workers),
                                 mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_init_worker,
                                 initargs=(str(config_path),)) as pool:
            for key in pending:
                directory = model_dir(root, key)
                if speed._summary(directory / "summary.json", digest) is not None:
                    continue
                resumable = _trained(directory, plan)
                if not claim(directory, keep_training=resumable):
                    continue
                seed_text, arm = key.split("/", 1)
                seed_index = plan.training_seeds.index(int(seed_text))
                acfg = arm_config(config, plan, arm)
                if resumable and _trained(directory, plan):
                    log(f"budget: {key} already trained; evaluating")
                    trained = json.loads((directory / "training.json").read_text(
                        encoding="utf-8")) if (directory / "training.json").exists() else None
                else:
                    log(f"budget: {key} trains {plan.updates} updates "
                        f"(weights kept at {list(plan.budgets)})")
                    for stale in directory.glob("eval_budget_*.jsonl"):
                        stale.unlink()
                    dev._status(directory, "training")
                    trained = train_seed(acfg, config_path, int(seed_text), directory, pool,
                                         log=log, updates=plan.updates,
                                         snapshots=plan.budgets, validate=False)
                    write_json(directory / "training.json", trained)
                rows = _rows(directory / "history.jsonl")
                budgets = {}
                for budget in plan.budgets:
                    path = directory / f"budget_{budget}.pt"
                    model, meta = load_weights(acfg, arm, path)
                    if (meta.get("extra") or {}).get("budget") != budget:
                        raise RuntimeError(f"{path} is not the weights after {budget} updates")
                    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    log(f"budget: evaluating {key} at budget {budget}")
                    result = evaluate_budget(
                        pool, plan, arm, state, seed_index,
                        directory / f"eval_budget_{budget}.jsonl",
                        progress=lambda name, index, b=budget: dev._status(
                            directory, "evaluating", budget=b, cell=name, cell_index=index,
                            cells=len(plan.cells)))
                    budgets[str(budget)] = {
                        "checkpoint": str(path), "sha256": _sha256(path),
                        "difficulty": float(rows[budget - 1]["difficulty"]),
                        "updates_at_full_difficulty": sum(
                            float(r["difficulty"]) >= speed.FULL for r in rows[:budget]),
                        **result}
                    log(f"budget: {key} @ {budget}: ten-cell success "
                        f"{result['mean_success']:.3f} unsafe {result['mean_unsafe']:.3f}")
                write_json(directory / "summary.json", {
                    "protocol": PROTOCOL, "plan_hash": digest, "complete": True, "model": key,
                    "arm": arm, "training_seed": int(seed_text), "seed_index": seed_index,
                    "first_full_update": speed.first_full(rows), "training": trained,
                    "budgets": budgets})
                speed.release(directory)
                dev._status(directory, "done")
                done_here.append(key)
    return 0, {"stage": "budget", "protocol": PROTOCOL, "trained_here": done_here,
               "wall_s": time.monotonic() - started}


def claim(directory: Path, *, keep_training: bool) -> bool:
    """``speed.claim``; it clears a dead run's best.pt/final.pt, so a finished training run
    whose evaluation was interrupted (``keep_training``) is set aside and restored."""
    keep = {}
    if keep_training:
        for name in ("final.pt",):
            path = directory / name
            if path.exists():
                keep[name] = path.read_bytes()
    held = speed.claim(directory)
    if held:
        for name, data in keep.items():
            if not (directory / name).exists():
                (directory / name).write_bytes(data)
    return held


# ------------------------------------------------------------------- analysis

def _verdict(interval: dict[str, Any]) -> str:
    if interval["low"] is not None and interval["low"] > 0.0:
        return "better"
    if interval["high"] is not None and interval["high"] < 0.0:
        return "worse"
    return "not established"


def compare(plan: BudgetPlan, proposed: dict[int, dict[str, Any]],
            other: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """The registered fixed-sequence comparison of one arm pair over the budgets (``RULE``)."""
    seeds = sorted(set(proposed) & set(other))
    confidence = 1.0 - plan.family_alpha / len(plan.comparisons)
    out, stopped_at = {}, None
    for budget in plan.budgets:
        b = str(budget)
        diffs = [proposed[s]["budgets"][b]["mean_success"] - other[s]["budgets"][b]["mean_success"]
                 for s in seeds]
        interval = speed.paired_interval(diffs, confidence)
        found = _verdict(interval)
        tested = stopped_at is None
        out[b] = {"tested": tested, "verdict": found if tested else None,
                  "descriptive_verdict": found, "difference": interval,
                  "proposed_mean": float(np.mean([proposed[s]["budgets"][b]["mean_success"]
                                                  for s in seeds])),
                  "other_mean": float(np.mean([other[s]["budgets"][b]["mean_success"]
                                               for s in seeds])),
                  "proposed_higher": sum(d > 0 for d in diffs),
                  "other_higher": sum(d < 0 for d in diffs)}
        if tested and found == "not established":
            stopped_at = budget
    return {"by_budget": out, "sequence_stopped_at": stopped_at, "seeds": seeds,
            "confidence": confidence}


def report(config_path: Path, plan_path: Path, root: Path) -> dict[str, Any]:
    config = load_config(config_path)
    plan = load_plan(plan_path, config)
    problem = check_registration(plan, config, root)
    if problem:
        raise ValueError(problem)
    digest = plan_hash(plan, config)
    records: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in plan.arms}
    pending = []
    for key in model_keys(plan):
        seed, arm = key.split("/", 1)
        summary = speed._summary(model_dir(root, key) / "summary.json", digest)
        if summary is None:
            pending.append(key)
        else:
            records[arm][int(seed)] = summary
    arms = {}
    for arm, by_seed in records.items():
        per_budget = {}
        for budget in plan.budgets:
            b = str(budget)
            rows = [r["budgets"][b] for r in by_seed.values()]
            if not rows:
                continue
            per_budget[b] = {
                "mean_success": float(np.mean([r["mean_success"] for r in rows])),
                "mean_unsafe": float(np.mean([r["mean_unsafe"] for r in rows])),
                "per_seed_success": {s: r["budgets"][b]["mean_success"]
                                     for s, r in sorted(by_seed.items())},
                "at_full_difficulty": sum(r["difficulty"] >= speed.FULL for r in rows),
                "cells": {c.name: float(np.mean([r["cells"][c.name]["success"] for r in rows]))
                          for c in plan.cells},
                "unsafe_cells": {c.name: float(np.mean([r["cells"][c.name]["unsafe"]
                                                        for r in rows])) for c in plan.cells}}
        arms[arm] = {"models": len(by_seed), "budgets": per_budget,
                     "first_full_update": {s: r["first_full_update"]
                                           for s, r in sorted(by_seed.items())}}
    complete = not pending
    comparisons = ({other: compare(plan, records[plan.proposed_arm], records[other])
                    for other in plan.comparisons} if complete else None)
    verdict = None
    if comparisons is not None:
        verdict = {other: {b: v["verdict"] for b, v in c["by_budget"].items()}
                   for other, c in comparisons.items()}
    return {"protocol": PROTOCOL, "plan_hash": digest, "complete": complete,
            "pending": pending, "proposed_arm": plan.proposed_arm,
            "budgets": list(plan.budgets), "rule": RULE, "arms": arms,
            "comparisons": comparisons, "verdict": verdict}


# --------------------------------------------------------------------- Isaac

def isaac_keys(plan: BudgetPlan) -> tuple[str, ...]:
    return model_keys(plan)


def register_isaac(config_path: Path, plan_path: Path, iplan_path: Path, root: Path
                   ) -> tuple[int, dict[str, Any]]:
    """Bind the Isaac plan to the complete local procedure and to every checkpoint's sha256
    before anything flies."""
    from .experiment import write_json
    config = load_config(config_path)
    plan = load_plan(plan_path, config)
    iplan = load_isaac_plan(iplan_path, config, plan)
    problem = check_registration(plan, config, root)
    if problem:
        return 3, {"stage": "register-isaac-budget", "status": "REFUSED", "reason": problem}
    digest = plan_hash(plan, config)
    missing = [k for k in model_keys(plan)
               if speed._summary(model_dir(root, k) / "summary.json", digest) is None]
    if missing:
        return 3, {"stage": "register-isaac-budget", "status": "REFUSED",
                   "reason": f"{PROTOCOL} is not complete; pending {missing[:5]}..."}
    checkpoints = {k: _sha256(checkpoint(root, k, iplan.budget)) for k in model_keys(plan)}
    payload = {"stage": "register-isaac-budget", "protocol": ISAAC_PROTOCOL,
               "plan_hash": isaac_plan_hash(iplan, digest), "source_plan_hash": digest,
               "plan": iplan.to_dict(), "plan_file": str(Path(iplan_path).resolve()),
               "rule": ISAAC_RULE, "checkpoints": checkpoints,
               "episodes": len(iplan.seeds) * len(checkpoints)}
    path = isaac_dir(root) / ISAAC_REGISTRATION
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("plan_hash") != payload["plan_hash"] \
                or stored.get("checkpoints") != checkpoints:
            return 3, {"stage": "register-isaac-budget", "status": "REFUSED",
                       "reason": f"{path} registers another plan or other checkpoints"}
        return 0, {**payload, "status": "ALREADY_REGISTERED"}
    if (isaac_dir(root) / "episodes.jsonl").exists():
        return 3, {"stage": "register-isaac-budget", "status": "REFUSED",
                   "reason": "episodes were flown before the registration; archive them"}
    payload = {**payload, "status": "REGISTERED",
               "registered_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)
    return 0, payload


def check_isaac_registration(config: CurrentSemanticsConfig, plan: BudgetPlan,
                             iplan: IsaacBudgetPlan, root: Path) -> str | None:
    path = isaac_dir(root) / ISAAC_REGISTRATION
    if not path.exists():
        return f"{ISAAC_PROTOCOL} is not registered ({path}); run register-isaac-budget first"
    stored = json.loads(path.read_text(encoding="utf-8"))
    if stored.get("plan_hash") != isaac_plan_hash(iplan, plan_hash(plan, config)):
        return f"{path} registers another plan or contract"
    for key, digest in (stored.get("checkpoints") or {}).items():
        if _sha256(checkpoint(root, key, iplan.budget)) != digest:
            return f"the checkpoint of {key} changed after the registration"
    return None


def isaac_order(iplan: IsaacBudgetPlan, keys: tuple[str, ...]) -> list[tuple[str, int, str]]:
    """(condition, environment seed, model): condition by condition, environment-major, so an
    interruption leaves every model with the same environments flown."""
    return [(cell, seed, key) for cell, seeds in iplan.conditions for seed in seeds
            for key in keys]


ROW_KEYS = ("status", "decisions", "elapsed_s", "return", "termination", "terminal_truth",
            "max_tilt_deg", "handover_wait_decisions", "scenario", "evaluation_dropout_s")


def isaac_run(config: CurrentSemanticsConfig, plan: BudgetPlan, iplan: IsaacBudgetPlan,
              root: Path, make_backend: Callable[[dev.EvaluationCell], Any], out: Path, *,
              log: Callable[[str], None] = print, resume: bool = False) -> dict[str, Any]:
    """Fly the registered cases on the backends ``make_backend(cell)`` returns (once per
    condition; any backend works, so the logic is testable on the local plant)."""
    import torch
    from .isaac_eval import _plain
    from .pilot import fly_policy
    keys = isaac_keys(plan)
    models = {k: load_weights(arm_config(config, plan, k.split("/", 1)[1]), k.split("/", 1)[1],
                              checkpoint(root, k, iplan.budget)) for k in keys}
    order = isaac_order(iplan, keys)
    out.mkdir(parents=True, exist_ok=True)
    stream = out / "episodes.jsonl"
    rows, done = [], set()
    if resume and stream.exists():
        planned = set(order)
        for row in _rows(stream):
            item = (row["condition"], int(row["seed"]), row["model"])
            if item not in planned or item in done:
                raise ValueError(f"{stream} holds {item}, which this protocol does not plan "
                                 "or flew twice; refusing to merge")
            done.add(item)
            rows.append(row)
    elif stream.exists() and stream.stat().st_size:
        raise ValueError(f"{stream} already holds episodes; resume the same registration")
    log(f"isaac-budget: {len(done)} of {len(order)} episodes already have a verdict")
    backends: dict[str, Any] = {}
    stopped = None
    started = time.monotonic()
    with stream.open("a", encoding="utf-8") as handle:
        for cell_name, seed, key in order:
            if (cell_name, seed, key) in done:
                continue
            model, _ = models[key]
            arm = key.split("/", 1)[1]
            t0 = time.monotonic()
            try:
                if cell_name not in backends:
                    backends[cell_name] = make_backend(plan.cell(cell_name))
                flown = fly_policy(backends[cell_name], model, arm_config(config, plan, arm),
                                   seed=seed, deterministic=True,
                                   generator=torch.Generator().manual_seed(int(seed)),
                                   record=True, trace=False)
            except Exception as exc:  # the stack is no longer trusted
                stopped = {"condition": cell_name, "seed": int(seed), "model": key,
                           "error": f"{type(exc).__name__}: {exc}"}
                with (out / "interruptions.jsonl").open("a", encoding="utf-8") as note:
                    note.write(json.dumps(stopped) + "\n")
                log(f"isaac-budget: stopped at {cell_name} seed {seed} {key}: "
                    f"{stopped['error']}")
                break
            row = _plain({"condition": cell_name, "seed": int(seed), "model": key, "arm": arm,
                          "training_seed": int(key.split("/", 1)[0]),
                          **{k: flown.get(k) for k in ROW_KEYS},
                          "wall_s": time.monotonic() - t0})
            rows.append(row)
            done.add((cell_name, seed, key))
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
            log(f"isaac-budget: [{len(done)}/{len(order)}] {cell_name} seed {seed} {key} -> "
                f"{row['status']} in {row['elapsed_s']:.1f} s (wall {row['wall_s']:.0f} s)")
    return {"rows": rows, "complete": stopped is None and len(done) == len(order),
            "stopped": stopped, "wall_s": time.monotonic() - started,
            "backends": sorted(backends)}


def isaac_local_reference(config_path: Path, plan: BudgetPlan, iplan: IsaacBudgetPlan,
                          root: Path, *, workers: int = 16) -> list[dict[str, Any]]:
    """The identical cases on the calibrated local plant (the development cells' own path)."""
    from .pilot import _init_worker
    config = load_config(config_path)
    keys = isaac_keys(plan)
    states = {}
    for key in keys:
        arm = key.split("/", 1)[1]
        model, _ = load_weights(arm_config(config, plan, arm), arm,
                                checkpoint(root, key, iplan.budget))
        states[key] = {k: v.detach().clone() for k, v in model.state_dict().items()}
    order = isaac_order(iplan, keys)
    tasks = [(key.split("/", 1)[1], states[key], seed, asdict(plan.cell(cell)),
              plan.dropout_start_s, seed + ACTION_OFFSET) for cell, seed, key in order]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_init_worker, initargs=(str(config_path),)) as pool:
        episodes = list(pool.map(dev._cell_work, tasks, chunksize=4))
    return [{"condition": c, "seed": int(s), "model": k, "arm": k.split("/", 1)[1],
             "training_seed": int(k.split("/", 1)[0]), "status": e.get("status"),
             "refused": e.get("refused"), "decisions": e.get("decisions")}
            for (c, s, k), e in zip(order, episodes)]


def _rate(rows: list[dict[str, Any]], test: Callable[[str | None], bool]) -> float | None:
    flown = [r for r in rows if r.get("status") is not None]
    return sum(test(r["status"]) for r in flown) / len(flown) if flown else None


def isaac_report(config: CurrentSemanticsConfig, plan: BudgetPlan, iplan: IsaacBudgetPlan,
                 flown: dict[str, Any], local: list[dict[str, Any]] | None, *,
                 backend: str | None, profile_sha256: str | None) -> dict[str, Any]:
    from .isaac_eval import _plain
    success = lambda s: s == "SUCCESS"
    isaac, local = flown["rows"], local or []
    per_model = {}
    for key in isaac_keys(plan):
        mine = [r for r in isaac if r["model"] == key]
        ref = [r for r in local if r["model"] == key]
        per_model[key] = {"episodes": len(mine), "success": _rate(mine, success),
                          "unsafe": _rate(mine, dev.unsafe),
                          "local_success": _rate(ref, success),
                          "by_condition": {c: _rate([r for r in mine if r["condition"] == c],
                                                    success) for c, _ in iplan.conditions}}
    def mean(values: list[float | None]) -> float | None:
        values = [v for v in values if v is not None]
        return float(np.mean(values)) if values else None

    arms = {}
    for arm in plan.arms:
        keys = [f"{s}/{arm}" for s in plan.training_seeds]
        arms[arm] = {"success": mean([per_model[k]["success"] for k in keys]),
                     "local_success": mean([per_model[k]["local_success"] for k in keys]),
                     "unsafe": mean([per_model[k]["unsafe"] for k in keys]),
                     "by_condition": {c: mean([per_model[k]["by_condition"][c] for k in keys])
                                      for c, _ in iplan.conditions},
                     "episodes": sum(per_model[k]["episodes"] for k in keys)}
    comparisons = None
    if flown["complete"]:
        confidence = 1.0 - iplan.family_alpha / len(plan.comparisons)
        comparisons = {}
        for other in plan.comparisons:
            diffs = [per_model[f"{s}/{plan.proposed_arm}"]["success"]
                     - per_model[f"{s}/{other}"]["success"] for s in plan.training_seeds]
            local_diffs = [per_model[f"{s}/{plan.proposed_arm}"]["local_success"]
                           - per_model[f"{s}/{other}"]["local_success"]
                           for s in plan.training_seeds] if local else None
            interval = speed.paired_interval(diffs, confidence)
            comparisons[other] = {
                "verdict": _verdict(interval), "difference": interval,
                "proposed_higher": sum(d > 0 for d in diffs),
                "other_higher": sum(d < 0 for d in diffs),
                "local_difference": (speed.paired_interval(local_diffs, confidence)
                                     if local_diffs else None)}
    statuses: dict[str, int] = {}
    for r in isaac:
        if r.get("status") is not None:
            statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    by = {(r["condition"], r["seed"], r["model"]): r["status"] for r in local}
    cases = [(r["status"], by.get((r["condition"], r["seed"], r["model"]))) for r in isaac
             if r.get("status") is not None]
    paired = {"cases": len(cases),
              "both_success": sum(a == "SUCCESS" and b == "SUCCESS" for a, b in cases),
              "isaac_only_success": sum(a == "SUCCESS" and b != "SUCCESS" for a, b in cases),
              "local_only_success": sum(a != "SUCCESS" and b == "SUCCESS" for a, b in cases),
              "both_fail": sum(a != "SUCCESS" and b != "SUCCESS" for a, b in cases)} \
        if local else None
    return _plain({
        "protocol": ISAAC_PROTOCOL, "source_protocol": plan.protocol,
        "plan_hash": isaac_plan_hash(iplan, plan_hash(plan, config)),
        "backend": backend, "profile_sha256": profile_sha256, "plan": iplan.to_dict(),
        "rule": ISAAC_RULE, "protocol_complete": flown["complete"], "stopped": flown["stopped"],
        "episodes": len(isaac), "wall_s": flown["wall_s"], "arms": arms,
        "per_model": per_model, "comparisons": comparisons, "isaac_statuses": statuses,
        "paired_with_local": paired,
        "verdict": ({k: v["verdict"] for k, v in comparisons.items()} if comparisons
                    else None)})
