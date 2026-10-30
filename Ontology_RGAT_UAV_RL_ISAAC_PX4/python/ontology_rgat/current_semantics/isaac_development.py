"""Guide 11.6: the development models flown on Isaac/PX4 (protocol isaac-development/1).

After a passed development gate, EVERY development model -- each training seed x
each arm, its validation-selected ``best.pt``, never the best seed alone -- flies
the SAME registered Isaac cases, deterministically.  A condition names one of the
registered development cells, so its split, dropout and pad acceleration are the
local cell's own definition (``development.cell_conditions``); the task-v5 draw of
every episode is made by the learner and checked against Isaac's acknowledgement
(``scenario.check_isaac_reset``).  The optical-noise factor is a local-plant
mechanism, so cells carrying it cannot be flown.

``register`` binds the plan to the development registration and to the sha256 of
every checkpoint before anything flies; ``run`` refuses anything else, flies
condition by condition, environment-major (every model meets an environment before
the next one starts, so an interruption leaves comparable coverage), appends every
verdict as it lands and resumes an interrupted flight of the same registration.
``local_reference`` flies the identical cases on the calibrated local plant, so each
Isaac verdict has a paired local one.  This is an integration measurement of the
current deployment, not an Isaac superiority test (guide 11.6.5): no pass criterion.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import datetime as dt
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
from typing import Any, Callable

import torch
import yaml

from . import SCHEMA_ID
from . import development as dev
from .contracts import (AUDITED_SEED_RANGES, DIAGNOSTIC_SEED_RANGES, PROJECT_ROOT,
                        CurrentSemanticsConfig, contract_manifest, load_config)

ISAAC_DEVELOPMENT_PROTOCOL = "isaac-development/1"
DEFAULT_PATH = PROJECT_ROOT / "config" / "current_semantics" / "isaac_development_v1.yaml"
REGISTRATION = "isaac_development_registration.json"
RESULT = "isaac_development.json"
OUT_DIR = "isaac_development"
LOCAL_REFERENCE = "local_reference.jsonl"


@dataclass(frozen=True)
class IsaacDevelopmentPlan:
    conditions: tuple[tuple[str, tuple[int, ...]], ...]  # (development cell, environment seeds)
    mode: str = "deterministic"
    protocol: str = ISAAC_DEVELOPMENT_PROTOCOL
    development_protocol: str = dev.DEVELOPMENT_PROTOCOL

    def to_dict(self) -> dict[str, Any]:
        return {"protocol": self.protocol, "development_protocol": self.development_protocol,
                "mode": self.mode,
                "conditions": {name: list(seeds) for name, seeds in self.conditions}}

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(s for _, seeds in self.conditions for s in seeds)


def load_plan(path: str | Path, config: CurrentSemanticsConfig,
              dev_plan: dev.DevelopmentPlan) -> IsaacDevelopmentPlan:
    data = dev._section(yaml.safe_load(Path(path).read_text(encoding="utf-8")),
                        ("protocol", "development_protocol", "mode", "conditions"),
                        "isaac_development")
    if data["protocol"] != ISAAC_DEVELOPMENT_PROTOCOL:
        raise ValueError(f"isaac_development.protocol must be {ISAAC_DEVELOPMENT_PROTOCOL!r}")
    if data["development_protocol"] != dev_plan.protocol:
        raise ValueError("isaac_development.development_protocol must name the registered "
                         f"development protocol {dev_plan.protocol!r}")
    if data["mode"] != "deterministic":
        raise ValueError("isaac_development.mode must be deterministic (guide 11.6.3)")
    if not isinstance(data["conditions"], dict) or not data["conditions"]:
        raise ValueError("isaac_development.conditions must map development cells to seeds")
    conditions = []
    for name, seeds in data["conditions"].items():
        if name not in dev.CELLS:
            raise ValueError(f"isaac_development condition {name!r} is not a development cell")
        if dev_plan.cell(name).optical_noise_scale != 1.0:
            raise ValueError(f"{name} scales the optical noise, which only the local plant "
                             "applies; Isaac cannot fly it")
        conditions.append((name, dev._seeds(seeds, f"isaac_development.conditions.{name}")))
    plan = IsaacDevelopmentPlan(conditions=tuple(conditions))
    if len(set(plan.seeds)) != len(plan.seeds):
        raise ValueError("isaac_development seeds must be distinct across conditions")
    _audit(plan, config, dev_plan)
    return plan


def _audit(plan: IsaacDevelopmentPlan, config: CurrentSemanticsConfig,
           dev_plan: dev.DevelopmentPlan) -> None:
    """Guide 11.4: no environment seed may lie in any recorded or registered use."""
    g4 = config.g4
    per = g4.episodes_per_update
    used = [(name, first, last) for name, first, last, _ in AUDITED_SEED_RANGES]
    used += list(DIAGNOSTIC_SEED_RANGES)
    used += [("g4 pilot training", *dev._PILOT_SEEDS)]
    used += [("development training", dev_plan.train_seed(s, 0, 0, per),
              dev_plan.train_seed(s, g4.updates - 1, per - 1, per))
             for s in dev_plan.training_seeds]
    used += [("development evaluation", dev_plan.eval_seed(0, 0, 0),
              dev_plan.eval_seed(len(dev_plan.training_seeds) - 1, len(dev.CELLS) - 1,
                                 dev_plan.episodes_per_cell - 1))]
    taken = (set(g4.validation_seeds) | set(g4.test_seeds) | set(config.g3.pilot_seeds)
             | set(config.isaac_eval.seeds) | set(dev_plan.validation_seeds))
    for seed in plan.seeds:
        hits = [name for name, first, last in used if first <= seed <= last]
        if hits or seed in taken:
            raise ValueError(f"isaac_development seed {seed} is already used "
                             f"({hits or 'registered'})")


def model_keys(dev_plan: dev.DevelopmentPlan) -> tuple[str, ...]:
    return tuple(f"{seed}/{arm}" for seed in dev_plan.training_seeds for arm in dev_plan.arms)


def checkpoint_path(root: Path, key: str) -> Path:
    seed, arm = key.split("/")
    return root / "development" / f"seed{seed}" / arm / "best.pt"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan_hash(plan: IsaacDevelopmentPlan, development_plan_hash: str) -> str:
    from ..direct_policy.contracts import digest
    return digest({"plan": plan.to_dict(), "development_plan_hash": development_plan_hash})


def _development_passed(config: CurrentSemanticsConfig, dev_plan: dev.DevelopmentPlan,
                        root: Path) -> tuple[str | None, str | None]:
    """(reason it is not, the registered development plan hash)."""
    problem = dev.check_registration(config, dev_plan, root)
    if problem:
        return problem, None
    registration = json.loads((root / dev.REGISTRATION).read_text(encoding="utf-8"))
    result_path = root / dev.RESULT
    if not result_path.exists():
        return "the development stage has not finished (no development.json)", None
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("plan_hash") != registration["plan_hash"]:
        return "development.json belongs to another development plan", None
    if not (result.get("procedure_complete")
            and (result.get("verdict") or {}).get("development_gate_passed")):
        return "guide 11.6 flies the development models only after a PASSED development gate", None
    return None, registration["plan_hash"]


def register(config_path: Path, plan_path: Path, dev_plan_path: Path,
             root: Path) -> tuple[int, dict[str, Any]]:
    config = load_config(config_path)
    dev_plan = dev.load_development(dev_plan_path, config)
    plan = load_plan(plan_path, config, dev_plan)
    refused = {"stage": "register-isaac-development", "status": "REFUSED"}
    problem, dev_hash = _development_passed(config, dev_plan, root)
    if problem:
        return 3, {**refused, "reason": problem}
    digest = plan_hash(plan, dev_hash)
    existing = root / REGISTRATION
    if existing.exists():
        recorded = json.loads(existing.read_text(encoding="utf-8"))
        if recorded.get("plan_hash") == digest:
            return 0, {**recorded, "status": "ALREADY_REGISTERED"}
        return 3, {**refused, "reason": f"{existing} registers another plan "
                                        f"({recorded.get('plan_hash')}); archive it first"}
    checkpoints = {}
    for key in model_keys(dev_plan):
        path = checkpoint_path(root, key)
        if not path.exists():
            return 3, {**refused, "reason": f"development model {key} has no best.pt ({path})"}
        checkpoints[key] = {"path": str(path), "sha256": _sha256(path)}
    return 0, {"schema_id": SCHEMA_ID, "stage": "register-isaac-development",
               "status": "REGISTERED", "protocol": ISAAC_DEVELOPMENT_PROTOCOL,
               "plan_hash": digest, "development_plan_hash": dev_hash,
               "plan_file": str(plan_path), "plan": plan.to_dict(),
               "checkpoints": checkpoints,
               "episodes": len(plan.seeds) * len(checkpoints),
               "registered_at": dt.datetime.now(dt.timezone.utc).isoformat(),
               "note": ("registered before any development model flew on Isaac; conditions "
                        "and seeds fixed here, no pass criterion (guide 11.6.5)")}


def check_registration(config: CurrentSemanticsConfig, plan: IsaacDevelopmentPlan,
                       dev_plan: dev.DevelopmentPlan, root: Path) -> str | None:
    problem, dev_hash = _development_passed(config, dev_plan, root)
    if problem:
        return problem
    path = root / REGISTRATION
    if not path.exists():
        return "no Isaac development registration; run --stage register-isaac-development"
    recorded = json.loads(path.read_text(encoding="utf-8"))
    if recorded.get("plan_hash") != plan_hash(plan, dev_hash):
        return "the Isaac development plan differs from its registration"
    for key in model_keys(dev_plan):
        want = (recorded.get("checkpoints") or {}).get(key, {}).get("sha256")
        path = checkpoint_path(root, key)
        if not path.exists() or want != _sha256(path):
            return f"development checkpoint {key} differs from the registered one"
    return None


def episode_order(plan: IsaacDevelopmentPlan,
                  keys: tuple[str, ...]) -> list[tuple[str, int, str]]:
    """(condition, environment seed, model): condition by condition, environment-major."""
    return [(cell, seed, key) for cell, seeds in plan.conditions for seed in seeds
            for key in keys]


def load_models(config: CurrentSemanticsConfig, dev_plan: dev.DevelopmentPlan,
                root: Path) -> dict[str, tuple[Any, CurrentSemanticsConfig, dict[str, Any]]]:
    """Every development model under its own arm's contract (load_checkpoint refuses any
    other).  key -> (model, arm config, {checkpoint, update})."""
    from .models import CurrentActorCritic, load_checkpoint
    out = {}
    for key in model_keys(dev_plan):
        arm = key.split("/")[1]
        acfg = dev.arm_config(config, dev_plan, arm)
        model = CurrentActorCritic(arm, acfg, seed=0)
        path = checkpoint_path(root, key)
        meta = load_checkpoint(path, model, manifest=contract_manifest(acfg))
        model.eval()
        out[key] = (model, acfg, {"checkpoint": str(path),
                                  "update": (meta["extra"] or {}).get("update")})
    return out


def _plain(value: Any) -> Any:
    from .isaac_eval import _plain as plain
    return plain(value)


ROW_KEYS = ("status", "decisions", "elapsed_s", "return", "termination", "terminal_truth",
            "max_tilt_deg", "handover_wait_decisions", "scenario", "evaluation_dropout_s")


def run(config: CurrentSemanticsConfig, plan: IsaacDevelopmentPlan,
        dev_plan: dev.DevelopmentPlan, root: Path,
        make_backend: Callable[[dev.EvaluationCell], Any], out_dir: Path, *,
        log: Callable[[str], None] = print, trace: bool = True,
        resume: bool = False) -> dict[str, Any]:
    """Fly the registered cases on the backends ``make_backend(cell)`` returns (called once
    per condition; any backend works, so the logic is testable on the local plant)."""
    from .pilot import fly_policy
    models = load_models(config, dev_plan, root)
    order = episode_order(plan, tuple(models))
    out_dir.mkdir(parents=True, exist_ok=True)
    stream = out_dir / "episodes.jsonl"
    rows: list[dict[str, Any]] = []
    done: set[tuple[str, int, str]] = set()
    if resume:
        planned = set(order)
        for line in (stream.read_text(encoding="utf-8").splitlines()
                     if stream.exists() else []):
            row = json.loads(line)
            item = (row["condition"], int(row["seed"]), row["model"])
            if item not in planned or item in done:
                raise ValueError(f"{stream} holds {item}, which this protocol does not plan "
                                 "or flew twice; refusing to merge")
            done.add(item)
            rows.append(row)
    elif stream.exists() and stream.stat().st_size:
        raise ValueError(f"{stream} already holds episodes; resume the same registration "
                         "or use a fresh directory")
    resumed = len(done)
    log(f"isaac-development: {resumed} of {len(order)} episodes already have a verdict")
    backends: dict[str, Any] = {}
    interruptions_path = out_dir / "interruptions.jsonl"
    stopped = None
    started = time.monotonic()
    with stream.open("a", encoding="utf-8") as handle:
        for cell_name, seed, key in order:
            if (cell_name, seed, key) in done:
                continue
            model, acfg, _ = models[key]
            t0 = time.monotonic()
            try:
                if cell_name not in backends:
                    backends[cell_name] = make_backend(dev_plan.cell(cell_name))
                flown = fly_policy(backends[cell_name], model, acfg, seed=seed,
                                   deterministic=True,
                                   generator=torch.Generator().manual_seed(int(seed)),
                                   record=True, trace=trace)
            except Exception as exc:  # the stack is no longer trusted
                stopped = {"condition": cell_name, "seed": int(seed), "model": key,
                           "error": f"{type(exc).__name__}: {exc}"}
                with interruptions_path.open("a", encoding="utf-8") as out:
                    out.write(json.dumps(stopped) + "\n")
                log(f"isaac-development: stopped at {cell_name} seed {seed} model {key}: "
                    f"{stopped['error']}")
                break
            training_seed, arm = key.split("/")
            row = _plain({"condition": cell_name, "seed": int(seed), "model": key, "arm": arm,
                          "training_seed": int(training_seed),
                          **{k: flown.get(k) for k in ROW_KEYS},
                          "wall_s": time.monotonic() - t0})
            if trace:
                traces = out_dir / "traces"
                traces.mkdir(exist_ok=True)
                name = f"{cell_name}_seed{seed}_{key.replace('/', '_')}.jsonl"
                with (traces / name).open("w", encoding="utf-8") as out:
                    for item in flown["trace"]:
                        out.write(json.dumps(_plain(item), allow_nan=False) + "\n")
            rows.append(row)
            done.add((cell_name, seed, key))
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
            log(f"isaac-development: [{len(done)}/{len(order)}] {cell_name} seed {seed} "
                f"{key} -> {row['status']} in {row['elapsed_s']:.1f} s "
                f"(wall {row['wall_s']:.0f} s)")
    complete = stopped is None and len(done) == len(order)
    interruptions = ([json.loads(line) for line in
                      interruptions_path.read_text(encoding="utf-8").splitlines()]
                     if interruptions_path.exists() else [])
    return {"rows": rows, "complete": complete, "stopped": stopped,
            "interruptions": interruptions, "resumed_with": resumed,
            "models": {k: info for k, (_, _, info) in models.items()},
            "wall_s": time.monotonic() - started, "backends": sorted(backends)}


# ------------------------------------------------------------ local reference

def local_reference(config_path: Path, plan: IsaacDevelopmentPlan,
                    dev_plan: dev.DevelopmentPlan, root: Path, *,
                    workers: int | None = None) -> list[dict[str, Any]]:
    """The identical registered cases on the calibrated local plant (development cells'
    own code path, ``development._cell_work``), for a paired local verdict per case."""
    from .pilot import _init_worker
    config = load_config(config_path)
    models = load_models(config, dev_plan, root)
    tasks, index = [], []
    for cell_name, seed, key in episode_order(plan, tuple(models)):
        model = models[key][0]
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        tasks.append((key.split("/")[1], state, seed, asdict(dev_plan.cell(cell_name)),
                      dev_plan.dropout_start_s, seed + 9_000_000))
        index.append((cell_name, seed, key))
    with ProcessPoolExecutor(max_workers=workers or dev_plan.workers,
                             mp_context=multiprocessing.get_context("spawn"),
                             initializer=_init_worker, initargs=(str(config_path),)) as pool:
        episodes = list(pool.map(dev._cell_work, tasks, chunksize=4))
    return [_plain({"condition": c, "seed": s, "model": k, "arm": k.split("/")[1],
                    "training_seed": int(k.split("/")[0]), "status": e.get("status"),
                    "refused": e.get("refused"), "decisions": e.get("decisions"),
                    "scenario": e.get("scenario")})
            for (c, s, k), e in zip(index, episodes)]


# ---------------------------------------------------------------------- report

def _rate(rows: list[dict[str, Any]], test: Callable[[str | None], bool]) -> float | None:
    flown = [r for r in rows if r.get("status") is not None]
    return sum(test(r["status"]) for r in flown) / len(flown) if flown else None


def _mean(values: list[float | None]) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarize(plan: IsaacDevelopmentPlan, dev_plan: dev.DevelopmentPlan,
              isaac: list[dict[str, Any]],
              local: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Per condition x arm: success and unsafe as the mean over training seeds of each
    model's rate (the training seed is the unit, guide 11.5), and the paired local
    verdicts on the same cases."""
    success = lambda s: s == "SUCCESS"
    tables: dict[str, Any] = {}
    for cell, _ in plan.conditions:
        tables[cell] = {}
        for arm in dev_plan.arms:
            per_seed = {}
            for seed in dev_plan.training_seeds:
                key = f"{seed}/{arm}"
                own = [r for r in isaac if r["condition"] == cell and r["model"] == key]
                mine = [r for r in (local or []) if r["condition"] == cell and r["model"] == key]
                per_seed[seed] = {"episodes": len(own), "success": _rate(own, success),
                                  "unsafe": _rate(own, dev.unsafe),
                                  "local_success": _rate(mine, success) if local else None}
            tables[cell][arm] = {
                "success": _mean([v["success"] for v in per_seed.values()]),
                "unsafe": _mean([v["unsafe"] for v in per_seed.values()]),
                "local_success": _mean([v["local_success"] for v in per_seed.values()]),
                "per_seed": per_seed}
    paired = None
    if local:
        by = {(r["condition"], r["seed"], r["model"]): r["status"] for r in local}
        cases = [(r["status"], by.get((r["condition"], r["seed"], r["model"]))) for r in isaac
                 if r.get("status") is not None]
        paired = {"cases": len(cases),
                  "both_success": sum(a == "SUCCESS" and b == "SUCCESS" for a, b in cases),
                  "isaac_only_success": sum(a == "SUCCESS" and b != "SUCCESS" for a, b in cases),
                  "local_only_success": sum(a != "SUCCESS" and b == "SUCCESS" for a, b in cases),
                  "both_fail": sum(a != "SUCCESS" and b != "SUCCESS" for a, b in cases)}
    statuses: dict[str, int] = {}
    for r in isaac:
        if r.get("status") is not None:
            statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    return {"by_condition_and_arm": tables, "paired_with_local": paired,
            "isaac_statuses": statuses}


def report(config: CurrentSemanticsConfig, plan: IsaacDevelopmentPlan,
           dev_plan: dev.DevelopmentPlan, flown: dict[str, Any], *,
           backend: str | None, profile_sha256: str | None,
           local: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    manifest = contract_manifest(config)
    return _plain({
        "schema_id": SCHEMA_ID, "stage": "Isaac evaluation of the development models (11.6)",
        "protocol": ISAAC_DEVELOPMENT_PROTOCOL, "backend": backend,
        "profile_sha256": profile_sha256, "plan": plan.to_dict(), "mode": "deterministic",
        "hashes": {k: manifest[k] for k in dev.BOUND_HASHES},
        "protocol_complete": flown["complete"], "stopped": flown["stopped"],
        "interruptions": flown["interruptions"], "models": flown["models"],
        "episodes": len(flown["rows"]), "wall_s": flown["wall_s"],
        **summarize(plan, dev_plan, flown["rows"], local),
        "note": ("an integration measurement of the current deployment on fixed paired cases "
                 "(guide 11.6.5): not an Isaac superiority test, no pass criterion")})
