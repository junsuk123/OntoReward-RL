"""learning-speed-extension/1: if training simply continues, when do the confirmation models
that never reached difficulty 1.0 within the PPO budget reach it?

User request 2026-11-20: most of the confirmation's R-GAT margin came from comparison models
that never left the success-gated curriculum inside 300 updates, so record when they would
have, and read the learning speed of every arm.  EXPLORATORY: the metric was chosen after the
confirmation results were seen; a confirmatory test needs fresh seeds that register it first.

Method, an exact continuation.  The confirmation's training is deterministic -- seeded
episodes, an order-preserving pool, private generators and single-threaded math (``run.sh``
sets ``OMP_NUM_THREADS``/``MKL_NUM_THREADS`` to 1) -- so training the same model seed again
with a larger budget reproduces the registered updates bit for bit and then goes on exactly as
the registered run would have under that budget.  The checkpoint alone could not do this: it
carries no Adam moments, generator or curriculum window.  Every replayed row is compared with
the registered history (wall time excluded); one mismatch marks the record as NOT a
continuation, and ``times`` then refuses it.

Event: the first update trained at difficulty 1.0, the definition every registered history is
read with.  Training stops at that update or at ``CAP`` updates (the budget of confirmation/1
before it was miniaturised, and of development/1), whichever comes first; a model still below
1.0 at the cap is censored there.  EVERY censored model is extended -- the proposed arm's too
-- so no comparison is built by extending one side only.
"""
from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from . import confirmation as cf
from . import development as dev
from .contracts import CurrentSemanticsConfig, load_config

PROTOCOL = "learning-speed-extension/1"
CAP = 600
FOLDER = "extension"                      # <root>/<confirmation folder>/extension/
REGISTRATION = "extension_registration.json"
RESULT = "extension.json"
FULL = 1.0
_IGNORED = ("wall_s",)                    # the only field a replay may change


def first_full(rows: list[dict[str, Any]]) -> int | None:
    """The first update trained at difficulty 1.0, or None."""
    return next((int(r["update"]) for r in rows if float(r["difficulty"]) >= FULL), None)


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def same_row(a: dict[str, Any], b: dict[str, Any]) -> bool:
    keep = lambda row: {k: v for k, v in row.items() if k not in _IGNORED}
    return keep(a) == keep(b)


def out_dir(root: Path, plan: cf.ConfirmationPlan) -> Path:
    return cf.out_dir(root, plan) / FOLDER


def model_dir(root: Path, plan: cf.ConfirmationPlan, key: str) -> Path:
    seed, arm = key.split("/")
    return out_dir(root, plan) / f"seed{seed}" / arm


def censored_keys(plan: cf.ConfirmationPlan, config: CurrentSemanticsConfig,
                  root: Path) -> tuple[str, ...]:
    """Every model of the (amended) confirmation whose registered history never reached
    difficulty 1.0; refused unless the confirmation's training is complete."""
    done = set(cf.completed_models(plan, config, root))
    missing = [k for k in cf.model_keys(plan) if k not in done]
    if missing:
        raise ValueError(f"confirmation training is incomplete: {missing}")
    return tuple(k for k in cf.model_keys(plan)
                 if first_full(read_rows(cf.model_root(root, plan, k) / "history.jsonl"))
                 is None)


def _digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def registration(config_path: Path, plan_path: Path, root: Path) -> dict[str, Any]:
    config = load_config(config_path)
    plan = cf.effective_plan(cf.load_plan(plan_path, config), config, root)
    keys = censored_keys(plan, config, root)
    return {"protocol": PROTOCOL, "confirmation_protocol": plan.protocol,
            "confirmation_plan_hash": cf.plan_hash(plan, config),
            "registered_budget": int(plan.updates), "cap": CAP,
            "event": "first update trained at difficulty 1.0",
            "models": list(keys),
            "registered_history_sha256": {
                k: _digest(cf.model_root(root, plan, k) / "history.jsonl") for k in keys},
            "analysis": (
                "exploratory: per training seed, R-GAT against each comparison arm; a model "
                "censored at the cap is slower than any observed event, two censored models "
                "tie; exact two-sided sign test over the untied seeds; restricted mean "
                "(censored = cap) reported, never decisive"),
            "note": "every censored model is extended, the proposed arm's included"}


def register(config_path: Path, plan_path: Path, root: Path) -> tuple[int, dict[str, Any]]:
    """Write the extension's registration before any extension training; an existing one
    must match exactly."""
    from .experiment import write_json
    payload = registration(config_path, plan_path, root)
    config = load_config(config_path)
    plan = cf.load_plan(plan_path, config)
    path = out_dir(root, plan) / REGISTRATION
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        stored.pop("registered_utc", None)
        if stored != payload:
            return 3, {"stage": "register-extension", "status": "REFUSED",
                       "reason": f"{path} exists and differs from the current registration"}
        return 0, {"stage": "register-extension", "status": "ALREADY_REGISTERED", **payload}
    import datetime as dt
    payload = {**payload, "registered_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)
    return 0, {"stage": "register-extension", "status": "REGISTERED", **payload}


def load_registration(root: Path, plan: cf.ConfirmationPlan) -> dict[str, Any]:
    path = out_dir(root, plan) / REGISTRATION
    if not path.exists():
        raise ValueError(f"{PROTOCOL} is not registered ({path}); run --stage "
                         "register-extension first")
    return json.loads(path.read_text(encoding="utf-8"))


def _record(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if data.get("protocol") == PROTOCOL and data.get("complete") else None


def run(config_path: Path, plan_path: Path, root: Path, *,
        log: Callable[[str], None] = print, workers: int | None = None) -> tuple[int, dict]:
    """Extend every registered model nobody holds (any number of processes may share them)."""
    import torch
    from .experiment import write_json
    from .pilot import _init_worker, train_seed
    if torch.get_num_threads() != 1:
        return 3, {"stage": "extend", "status": "REFUSED",
                   "reason": f"torch uses {torch.get_num_threads()} threads; the replay is "
                             "exact only single-threaded (run through run.sh)"}
    config = load_config(config_path)
    plan = cf.effective_plan(cf.load_plan(plan_path, config), config, root)
    record = load_registration(root, plan)
    budget = int(record["registered_budget"])
    from dataclasses import replace
    extended = replace(plan, updates=CAP)
    pending = [k for k in record["models"] if _record(model_dir(root, plan, k) / "summary.json")
               is None]
    log(f"extend: {len(record['models']) - len(pending)} of {len(record['models'])} done")
    done_here: list[str] = []
    started = time.monotonic()
    if pending:
        with ProcessPoolExecutor(max_workers=int(workers or plan.models.workers),
                                 mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_init_worker,
                                 initargs=(str(config_path),)) as pool:
            for key in pending:
                directory = model_dir(root, plan, key)
                if _record(directory / "summary.json") is not None or not cf.claim(directory):
                    continue
                registered = cf.model_root(root, plan, key) / "history.jsonl"
                if _digest(registered) != record["registered_history_sha256"][key]:
                    raise RuntimeError(f"{key}: the registered history changed")
                seed_text, arm = key.split("/")
                acfg = dev.arm_config(cf.budget_config(config, extended), plan.models, arm)
                log(f"extend: {key} replays {budget} registered updates, then continues to "
                    f"difficulty 1.0 or update {CAP}")
                dev._status(directory, "training")
                trained = train_seed(acfg, config_path, int(seed_text), directory, pool,
                                     log=log, stop_at_full=True)
                rows, reference = read_rows(directory / "history.jsonl"), read_rows(registered)
                compared = min(len(rows), len(reference))
                mismatch = next((i for i in range(compared)
                                 if not same_row(rows[i], reference[i])), None)
                event = first_full(rows)
                summary = {
                    "protocol": PROTOCOL, "complete": True, "model": key,
                    "registered_budget": budget, "cap": CAP,
                    "first_full_update": event, "censored": event is None,
                    "censored_at": None if event is not None else CAP,
                    "final_difficulty": float(rows[-1]["difficulty"]),
                    "updates_run": len(rows),
                    "replay": {"rows_compared": compared, "identical": mismatch is None,
                               "first_mismatch": mismatch},
                    "training": trained}
                write_json(directory / "summary.json", summary)
                cf.release(directory)
                dev._status(directory, "done")
                done_here.append(key)
                log(f"extend: {key} first difficulty-1.0 update "
                    f"{event if event is not None else f'none by {CAP}'}; replay "
                    f"{'identical' if mismatch is None else f'DIFFERS at row {mismatch}'}")
    return 0, {"stage": "extend", "protocol": PROTOCOL, "extended_here": done_here,
               "wall_s": time.monotonic() - started}


# --------------------------------------------------------------- analysis

def times(config: CurrentSemanticsConfig, plan: cf.ConfirmationPlan, root: Path
          ) -> dict[str, dict[str, Any]]:
    """Time to difficulty 1.0 of every model: the registered history where it got there
    within the budget, else the extension record (censored at ``CAP``)."""
    record = load_registration(root, plan)
    out: dict[str, dict[str, Any]] = {}
    for key in cf.model_keys(plan):
        rows = read_rows(cf.model_root(root, plan, key) / "history.jsonl")
        event = first_full(rows)
        if event is not None:
            out[key] = {"update": event, "censored": False, "source": "registered"}
            continue
        if key not in record["models"]:
            raise ValueError(f"{key} never reached difficulty 1.0 but is not registered")
        ext = _record(model_dir(root, plan, key) / "summary.json")
        if ext is None:
            out[key] = {"update": None, "censored": None, "source": "pending"}
            continue
        if not ext["replay"]["identical"]:
            raise ValueError(f"{key}: the replay is not the registered run "
                             f"(row {ext['replay']['first_mismatch']})")
        out[key] = {"update": ext["first_full_update"] if not ext["censored"] else CAP,
                    "censored": bool(ext["censored"]), "source": "extension",
                    "final_difficulty": ext["final_difficulty"]}
    return out


def compare(proposed: dict[int, dict[str, Any]], other: dict[int, dict[str, Any]]
            ) -> dict[str, Any]:
    """Paired by training seed: negative difference = the proposed arm got there first.  A
    censored time is a lower bound, so it loses to any observed event; two censored tie."""
    from scipy import stats
    seeds = sorted(set(proposed) & set(other))
    rows, wins, losses, ties = [], 0, 0, 0
    for s in seeds:
        a, b = proposed[s], other[s]
        if a["censored"] and b["censored"]:
            sign = 0
        elif a["censored"]:
            sign = 1
        elif b["censored"]:
            sign = -1
        else:
            sign = int(np.sign(a["update"] - b["update"]))
        wins += sign < 0
        losses += sign > 0
        ties += sign == 0
        rows.append({"seed": s, "proposed": a["update"], "proposed_censored": a["censored"],
                     "other": b["update"], "other_censored": b["censored"],
                     "difference": a["update"] - b["update"],
                     "difference_is_bound": bool(a["censored"] or b["censored"])})
    untied = wins + losses
    p = float(stats.binomtest(wins, untied, 0.5).pvalue) if untied else math.nan
    diffs = [r["difference"] for r in rows]
    return {"seeds": len(rows), "proposed_first": wins, "other_first": losses, "ties": ties,
            "sign_test_p": p,
            "restricted_mean_difference": float(np.mean(diffs)) if diffs else math.nan,
            "per_seed": rows}


def report(config_path: Path, plan_path: Path, root: Path) -> dict[str, Any]:
    config = load_config(config_path)
    plan = cf.effective_plan(cf.load_plan(plan_path, config), config, root)
    record = load_registration(root, plan)
    all_times = times(config, plan, root)
    pending = [k for k, v in all_times.items() if v["source"] == "pending"]
    by_arm: dict[str, dict[int, dict[str, Any]]] = {}
    for key, value in all_times.items():
        seed, arm = key.split("/")
        by_arm.setdefault(arm, {})[int(seed)] = value
    summary = {arm: {"observed": sum(not v["censored"] for v in m.values()
                                     if v["censored"] is not None),
                     "censored_at_cap": sum(bool(v["censored"]) for v in m.values()),
                     "restricted_mean_update": (float(np.mean([v["update"] for v in m.values()]))
                                                if not pending else None),
                     "median_update": (float(np.median([v["update"] for v in m.values()]))
                                       if not pending else None)}
               for arm, m in by_arm.items()}
    comparisons = ({other: compare(by_arm[cf.PROPOSED], by_arm[other])
                    for other in plan.analysis.primary_comparisons} if not pending else None)
    return {"protocol": PROTOCOL, "confirmation_protocol": plan.protocol,
            "complete": not pending, "pending": pending, "cap": CAP,
            "registered_budget": record["registered_budget"], "times": all_times,
            "arms": summary, "comparisons": comparisons,
            "reading": "exploratory (metric chosen after the confirmation results); a "
                       "confirmatory claim needs fresh seeds with this metric registered"}
