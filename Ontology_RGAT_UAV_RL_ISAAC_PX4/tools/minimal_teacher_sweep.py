#!/usr/bin/env python3
"""Sweep the minimal-contract teacher's gains on the local plant.

A ceiling is a ceiling only if its gains were swept, and only if the best cell
is not on the grid boundary (AGENTS.md: twice the "ceiling" was the edge of a
small grid). This reports, per cell, landing / unsafe / abort / timeout over
the given seeds, the best cell, and which of its gains sit on a boundary.

Every episode also records the pad-loss classifier against simulator truth,
so the same run yields the confusion matrix (``--confusion``):

  truth TERMINAL  body <= terminal_entry_height and |truth offset| <= terminal_offset
  truth OVER_PAD  |truth offset| <= pad half width (contact there is on the pad)
  predicted       the shared classifier's class at that decision

The costly error is predicted TERMINAL while truth is not even OVER_PAD: the
supervisor then commits a descent onto ground it cannot see.

    python tools/minimal_teacher_sweep.py --seeds 24 --out results/minimal_teacher_sweep.json
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

# On minimal-landing-ontology/4 the integral gains and the descent profile are
# ontology constants (TrackingBias, descent_reference), so only the feedback
# gains are swept. Sweeps r1-r3 (ontology /3) swept them too; r3's plateau
# cell (ki 0.1, kzi 0.5, sink 0.2, align 0.4) became those constants.
GRID = {
    "kp": (0.25, 0.4, 0.6, 0.8),
    "kd": (0.5, 0.8, 1.1, 1.4),
    "kz": (1.0, 1.5, 2.0),
}
UNSAFE = {"UNSAFE_CONTACT", "MISSED_PAD_CONTACT", "SAFETY_ENVELOPE_VIOLATION",
          "UNAUTHORIZED_CONTACT"}


def run_episode(gains: dict, seed: int, difficulty: float = 1.0):
    from ontology_rgat.minimal.constants import DEFAULT_CONSTANTS as C
    from ontology_rgat.minimal.local_env import MinimalLandingEnv
    from ontology_rgat.minimal.pad_loss import (LOST, TERMINAL_OCCLUSION,
                                                 TRANSIENT_DROPOUT, VISIBLE)
    from ontology_rgat.minimal.teacher import MinimalTeacher, TeacherGains
    env = MinimalLandingEnv(difficulty=difficulty)
    obs, _ = env.reset(seed=seed)
    teacher = MinimalTeacher(TeacherGains(**gains))
    confusion = Counter()
    done, info = False, {}
    while not done:
        obs, _, done, info = env.step(teacher.act(obs))
        cls = info["pad_class"]
        if cls in (VISIBLE, TRANSIENT_DROPOUT):
            continue
        truth = np.asarray(info["truth_pad_minus_body"])
        offset, height = float(np.hypot(truth[0], truth[1])), -float(truth[2])
        if height <= C.terminal_entry_height_m and offset <= C.terminal_offset_m:
            label = "TERMINAL"
        elif offset <= C.pad_half_width_m:
            label = "OVER_PAD"
        else:
            label = "OFF_PAD"
        predicted = "TERMINAL" if cls == TERMINAL_OCCLUSION else "LOST"
        confusion[f"{predicted}|{label}"] += 1
    return info["status"], float(info["elapsed_s"]), dict(confusion)


def run_cell(args):
    gains, seeds, difficulty = args
    results = [run_episode(gains, s, difficulty) for s in seeds]
    statuses = Counter(r[0] for r in results)
    confusion = Counter()
    for r in results:
        confusion.update(r[2])
    n = len(seeds)
    return {
        "gains": gains,
        "landing": statuses["SUCCESS"] / n,
        "unsafe": sum(statuses[s] for s in UNSAFE) / n,
        "abort": statuses["SAFE_ABORT"] / n,
        "timeout": statuses["TASK_TIMEOUT"] / n,
        "statuses": dict(statuses),
        "mean_success_time_s": float(np.mean([r[1] for r in results if r[0] == "SUCCESS"]))
        if statuses["SUCCESS"] else None,
        "confusion": dict(confusion),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seeds", type=int, default=24)
    parser.add_argument("--seed-base", type=int, default=4100)
    parser.add_argument("--difficulty", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--grid", type=json.loads, default=None,
                        help="JSON object overriding GRID entries")
    args = parser.parse_args()
    grid = {**GRID, **(args.grid or {})}
    seeds = list(range(args.seed_base, args.seed_base + args.seeds))
    names = list(grid)
    cells = [dict(zip(names, values)) for values in itertools.product(*grid.values())]
    started = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run_cell, [(c, seeds, args.difficulty) for c in cells]))
    rows.sort(key=lambda r: (-r["landing"], r["unsafe"]))
    best = rows[0]
    boundary = [k for k in names if len(grid[k]) > 1
                and best["gains"][k] in (min(grid[k]), max(grid[k]))]
    plateau = [r for r in rows if r["landing"] >= best["landing"] - 1.0 / len(seeds)]
    confusion = Counter()
    for r in rows:
        confusion.update(r["confusion"])
    summary = {
        "contract": "minimal-landing-obs/2 on SpatialConfig() local plant",
        "seeds": seeds, "difficulty": args.difficulty, "grid": grid,
        "cells": len(rows), "elapsed_s": time.time() - started,
        "best": best, "best_on_boundary": boundary,
        "plateau_cells": len(plateau),
        "median_landing": float(np.median([r["landing"] for r in rows])),
        "confusion_all_cells": dict(confusion),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in (
        "cells", "elapsed_s", "best", "best_on_boundary", "plateau_cells",
        "median_landing", "confusion_all_cells")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
