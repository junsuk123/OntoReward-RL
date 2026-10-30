"""The selected G4 checkpoints flown on Isaac/PX4 (``isaac_eval``, protocol isaac-eval/1).

User decision 2026-10-28: training runs on the Isaac-calibrated local plant and
every selected-checkpoint evaluation runs on Isaac.  This module takes an OPEN
``CurrentSpatialBackend`` (the caller owns the stack), loads each model seed's
``best.pt`` under the current hashes, and flies the deterministic policy on the
registered seeds, seed-major so an interrupted flight leaves every checkpoint
with comparable coverage.  Every episode is appended to ``episodes.jsonl`` as it
ends.  An exception during a flight stops the evaluation: the stack is then not
trusted, and the report says the protocol is incomplete instead of guessing.
``resume`` continues an interrupted flight of the SAME protocol in a fresh stack:
every pair that already has a verdict is kept and skipped, the pair that was
interrupted (no verdict: infrastructure, not RL) is flown again from its reset,
and every interruption stays in ``interruptions.jsonl`` and in the report.
Any backend works, so the logic is testable on the local plant.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import torch

from . import SCHEMA_ID
from .contracts import DIAGNOSTIC_SEED_RANGES, CurrentSemanticsConfig, contract_manifest

HASH_KEYS = ("isaac_eval_hash", "g4_hash", "algorithm_hash", "task_hash", "calibration_hash",
             "g3_hash")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        return value.item()
    return value


def _summary(episodes: list[dict[str, Any]], minimum: float) -> dict[str, Any]:
    flown = [e for e in episodes if e.get("status") is not None]
    statuses: dict[str, int] = {}
    for e in flown:
        statuses[e["status"]] = statuses.get(e["status"], 0) + 1
    success = (sum(e["status"] == "SUCCESS" for e in flown) / len(flown)) if flown else None
    return {"episodes": len(flown), "statuses": statuses, "success": success,
            "meets_criterion": None if success is None else success >= minimum,
            "cleanup_confirmed": all(bool((e.get("termination") or {}).get("cleanup_confirmed"))
                                     for e in flown)}


def diagnostic_seed(config: CurrentSemanticsConfig, seed: int) -> bool:
    """A seed a diagnostic re-flight may use: a registered isaac_eval seed, or one already
    spent on a recorded diagnosis (contracts.DIAGNOSTIC_SEED_RANGES).  A fresh seed is
    never burnt on a diagnosis, so a later registration can still use it."""
    return (seed in config.isaac_eval.seeds
            or any(first <= seed <= last for _, first, last in DIAGNOSTIC_SEED_RANGES))


def check_pairs(config: CurrentSemanticsConfig, pairs: Sequence[tuple[Any, Any]],
                checkpoints: Mapping[str, Any]) -> list[tuple[str, int]]:
    """The validated (model seed, seed) pairs of a diagnostic re-flight; raises before
    anything flies (the CLI calls it before a stack starts)."""
    pairs = [(str(key), int(seed)) for key, seed in pairs]
    bad = [p for p in pairs if p[0] not in checkpoints or not diagnostic_seed(config, p[1])]
    if not pairs or bad:
        raise ValueError("diagnostic pairs must name a checkpoint and a registered isaac_eval "
                         f"seed or an already-diagnostic seed; refused {bad or pairs}")
    return pairs


def run_isaac_evaluation(config: CurrentSemanticsConfig, backend: Any,
                         checkpoints: Mapping[str, Path], out_dir: Path, *,
                         seeds: tuple[int, ...] | None = None,
                         log: Callable[[str], None] = print,
                         trace: bool = True,
                         pairs: Sequence[tuple[str, int]] | None = None,
                         resume: bool = False,
                         foreign_task: bool = False) -> dict[str, Any]:
    """Seed-major over every checkpoint, or only ``pairs`` of (model seed, seed): a
    diagnostic re-flight of registered seeds that never completes the protocol."""
    from .models import CurrentActorCritic, load_checkpoint
    from .pilot import fly_policy

    registered = config.isaac_eval
    chosen = tuple(registered.seeds if seeds is None else seeds)
    foreign = sorted(set(chosen) - set(registered.seeds))
    if not chosen or foreign:
        raise ValueError(f"isaac_eval flies only isaac_eval.seeds; refused {foreign or chosen}")
    if not checkpoints:
        raise ValueError("no checkpoint to evaluate")
    if foreign_task and pairs is None:
        raise ValueError("a checkpoint trained on another task flies only in a diagnostic "
                         "re-flight (pairs), never in the registered evaluation")
    if pairs is not None:
        pairs = check_pairs(config, pairs, checkpoints)
    order = pairs or [(key, seed) for seed in chosen for key in checkpoints]
    manifest = contract_manifest(config)
    models = {}
    for key, path in checkpoints.items():
        model = CurrentActorCritic(config.g4.arm, config, seed=0)
        meta = load_checkpoint(path, model, manifest=manifest,  # refuses a stale contract
                               allow_foreign_task=foreign_task)
        model.eval()
        models[key] = (model, {"checkpoint": str(path), "update": meta["extra"].get("update"),
                               "checkpoint_task_hash": meta["metadata"]["task_hash"]})
    out_dir.mkdir(parents=True, exist_ok=True)
    stream = out_dir / "episodes.jsonl"
    episodes: dict[str, list[dict[str, Any]]] = {key: [] for key in models}
    done: set[tuple[str, int]] = set()
    if resume:
        if pairs is not None:
            raise ValueError("resume continues the registered protocol, not a diagnostic")
        planned = set(order)
        for line in (stream.read_text(encoding="utf-8").splitlines()
                     if stream.exists() else []):
            row = json.loads(line)
            pair = (str(row["model_seed"]), int(row["seed"]))
            if pair not in planned or pair in done:
                raise ValueError(f"{stream} holds {pair}, which this protocol does not plan "
                                 "or flew twice; refusing to merge")
            done.add(pair)
            episodes[pair[0]].append(row)
    elif stream.exists() and stream.stat().st_size:
        raise ValueError(f"{stream} already holds episodes; pass resume=True to continue "
                         "the same protocol or use a fresh directory")
    log(f"isaac-eval: {len(done)} of {len(order)} episodes already have a verdict")
    interruptions_path = out_dir / "interruptions.jsonl"
    stopped = None
    started = time.monotonic()
    with stream.open("a", encoding="utf-8") as handle:
        for key, seed in order:
            if (key, seed) in done:
                continue
            model = models[key][0]
            t0 = time.monotonic()
            try:
                flown = fly_policy(backend, model, config, seed=seed, deterministic=True,
                                   generator=torch.Generator().manual_seed(int(seed)),
                                   record=True, trace=trace)
            except Exception as exc:  # the stack is no longer trusted
                stopped = {"model_seed": key, "seed": int(seed),
                           "error": f"{type(exc).__name__}: {exc}"}
                with interruptions_path.open("a", encoding="utf-8") as out:
                    out.write(json.dumps(stopped) + "\n")
                log(f"isaac-eval: stopped at model seed {key}, seed {seed}: {stopped['error']}")
                break
            row = {"model_seed": key, "seed": int(seed),
                   **{k: flown[k] for k in ("status", "decisions", "elapsed_s", "return",
                                            "reference_return", "termination",
                                            "terminal_truth", "max_tilt_deg",
                                            "handover_wait_decisions")},
                   "wall_s": time.monotonic() - t0}
            row = _plain(row)
            if trace:  # one decision per line, evaluator-only truth included
                traces = out_dir / "traces"
                traces.mkdir(exist_ok=True)
                with (traces / f"model{key}_seed{seed}.jsonl").open("w") as out:
                    for item in flown["trace"]:
                        out.write(json.dumps(_plain(item), allow_nan=False) + "\n")
            episodes[key].append(row)
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
            log(f"isaac-eval: model seed {key} seed {seed} -> {row['status']} "
                f"in {row['elapsed_s']:.1f} s ({row['decisions']} decisions)")
    minimum = registered.deterministic_success_min
    results = {key: {**info, "deterministic": _summary(episodes[key], minimum),
                     "episodes": episodes[key]}
               for key, (_, info) in models.items()}
    # Only the full registered seed list completes the protocol; a subset is a probe.
    complete = (pairs is None and stopped is None and chosen == tuple(registered.seeds)
                and all(len(episodes[k]) == len(chosen) for k in models))
    interruptions = ([json.loads(line) for line in
                      interruptions_path.read_text(encoding="utf-8").splitlines()]
                     if interruptions_path.exists() else [])
    passed = (complete and all(r["deterministic"]["meets_criterion"]
                               and r["deterministic"]["cleanup_confirmed"]
                               for r in results.values()))
    return _plain({
        "schema_id": SCHEMA_ID, "stage": "Isaac evaluation of the selected G4 checkpoints",
        "protocol": registered.protocol, "backend": getattr(backend, "name", None),
        "profile_sha256": getattr(getattr(backend, "flight", None), "profile_sha256", None),
        "arm": config.g4.arm, "seeds": list(chosen), "mode": "deterministic",
        "criterion": {"deterministic_success_min": minimum},
        "hashes": {key: manifest[key] for key in HASH_KEYS},
        "protocol_complete": complete, "stopped": stopped,
        "foreign_task": bool(foreign_task),
        "interruptions": interruptions, "resumed_with": len(done),
        "isaac_eval_passed": bool(passed), "results": results,
        "wall_s": time.monotonic() - started,
        "note": ("deterministic policy of each model seed's best.pt; a small integration "
                 "sample, not a robustness or representation claim"),
    })
