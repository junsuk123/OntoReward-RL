#!/usr/bin/env python3
"""Evaluate every selected checkpoint in one or more runs on ONE wide seed set.

Why this exists
---------------
``train_arm`` selects on ``VALIDATION_SEEDS``, which is two episodes. A rate
built from two episodes can only be 0.000, 0.500 or 1.000, so the per-arm
numbers in a training summary have no resolution and cannot order the arms.
The pipeline's own ``--stage evaluate`` uses ``TEST_SEEDS``, also two.

This loads each run's selected checkpoint and replays it on a single explicit
held-out seed range shared by every arm and seed, so the comparison is paired:
same episodes, same contract, same supervisor.

Reporting rules this enforces
-----------------------------
* The seed range is held out and is never used for selection. Pass a range
  disjoint from both ``VALIDATION_SEEDS`` and any training seed.
* ``relational_health`` is carried through per run. If the proposed arm's
  relational path was inert, that run's ``ppo_ontology_rgat`` row is a copy of
  ``ppo_semantic_flat`` and the comparison means nothing -- the summary below
  says so rather than letting the number stand.
* Spread across training seeds is reported, not just the mean. On two seeds the
  spread swamped the between-arm differences, which is what motivated widening.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from ontology_rgat.spatial.core import SpatialConfig, schema_for
from ontology_rgat.spatial.runtime_contract import deployment_profile
from ontology_rgat.spatial.training import VALIDATION_SEEDS, evaluate, load_agent
from ontology_rgat.two_axis.artifacts import json_text
from ontology_rgat.two_axis.config import CurriculumConfig, GraphConfig
from ontology_rgat.two_axis.models import POLICY_MODES


def config_from_plan(directory):
    """Rebuild the exact config a run trained under.

    A checkpoint only loads against a byte-identical `config_sha256`, and the
    run may have overridden things the defaults do not carry -- the staged
    adaptation fraction, the horizon, the Isaac profile hash. Reconstructing
    from the run's own plan is the only way to stay on its contract.
    """
    plan = json.loads((directory / "plan.json").read_text())
    if "config" not in plan:
        # A behaviour-clone directory (tools/clone_spatial_teacher.py) records
        # only its signature: the clone trains on the contract defaults for
        # its schema. Rebuild those and insist on the same config hash, so a
        # clone can sit in the same table as the cells fine-tuned from it.
        schema = schema_for(plan["signature"]["schema"])
        cfg = replace(SpatialConfig(), schema=schema,
                      isaac_profile_sha256=deployment_profile(schema)["sha256"])
        if cfg.signature["config_sha256"] != plan["signature"]["config_sha256"]:
            raise ValueError(f"{directory}: clone signature does not match the "
                             "contract defaults; it was not cloned on them")
        return cfg
    payload = plan["config"]
    nested = {"curriculum": CurriculumConfig(**payload["curriculum"]),
              "ontology": GraphConfig(**payload["ontology"])}
    return SpatialConfig(**{**payload, **nested})

METRICS = ("landing_rate", "unsafe_rate", "safe_abort_rate",
           "task_timeout_rate", "mean_return")


def runs_in(directory):
    for path in sorted((directory / "runs").glob("*/summary.json")):
        yield path.parent, json.loads(path.read_text())


def main():
    # The pipeline runs this beside training workers that each pin one
    # thread; an unpinned evaluator spin-waits in OpenMP under that load and
    # burns a CPU-hour per minute of progress.
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("outputs", type=Path, nargs="+",
                        help="one or more pipeline output directories to pool")
    parser.add_argument("--seeds", type=int, nargs=2, default=[4100, 4148],
                        metavar=("START", "STOP"),
                        help="held-out evaluation range, never used for selection")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    seeds = tuple(range(*args.seeds))
    if not seeds:
        parser.error("evaluation seed range is empty")
    overlap = set(seeds) & set(VALIDATION_SEEDS)
    if overlap:
        parser.error(f"evaluation seeds overlap the selection seeds: {sorted(overlap)}")

    rows, inert = [], []
    cfg = None
    for directory in args.outputs:
        cfg = config_from_plan(directory)
        for run, summary in runs_in(directory):
            name = summary.get("selected_checkpoint")
            if not name:
                print(f"[compare] {run.name}: no selected checkpoint, skipped", flush=True)
                continue
            training_seed = summary["seed"]
            if training_seed in set(seeds):
                parser.error(f"evaluation seeds overlap training seed {training_seed}")
            agent, _meta = load_agent(run / name, cfg)
            health = summary.get("relational_health") or {}
            if health.get("applicable") and not health.get("active"):
                inert.append(f"{directory.name}/{run.name}")
            result = evaluate(agent, cfg, seeds=seeds)
            row = {"output": directory.name, "run": run.name,
                   "mode": summary["mode"], "seed": training_seed,
                   "checkpoint": name,
                   "relational_active": health.get("active"),
                   **{key: result[key] for key in METRICS}}
            rows.append(row)
            print(f"[compare] {row['mode']:22s} seed{row['seed']} "
                  + " ".join(f"{k}={row[k]:.3f}" for k in METRICS), flush=True)

    print(f"\n=== pooled over {len(seeds)} held-out episodes per run ===")
    print(f"{'arm':24s} {'runs':>4s} " + " ".join(f"{m:>14s}" for m in METRICS))
    summary_rows = {}
    for mode in POLICY_MODES:
        group = [r for r in rows if r["mode"] == mode]
        if not group:
            continue
        cells = []
        stats = {}
        for metric in METRICS:
            values = [r[metric] for r in group]
            mean = statistics.fmean(values)
            spread = statistics.pstdev(values) if len(values) > 1 else 0.0
            stats[metric] = {"mean": mean, "sd": spread,
                             "min": min(values), "max": max(values)}
            cells.append(f"{mean:7.3f}+-{spread:5.3f}")
        summary_rows[mode] = stats
        print(f"{mode:24s} {len(group):4d} " + " ".join(f"{c:>14s}" for c in cells))

    print("\nmean +- population sd ACROSS TRAINING SEEDS. Read the sd before the mean: "
          "on two seeds it swamped every between-arm difference.")
    if inert:
        print(f"\nWARNING: the relational path was INERT in {len(inert)} run(s) -- those "
              f"ppo_ontology_rgat rows are numerically ppo_semantic_flat and carry no "
              f"claim about the representation:\n   " + "\n   ".join(inert))
    if args.report:
        args.report.write_text(json_text(
            {"evaluation_seeds": list(seeds), "contract": cfg.signature,
             "selection_seeds_excluded": list(VALIDATION_SEEDS),
             "rows": rows, "per_arm": summary_rows,
             "inert_relational_runs": inert}, indent=2) + "\n")
        print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
