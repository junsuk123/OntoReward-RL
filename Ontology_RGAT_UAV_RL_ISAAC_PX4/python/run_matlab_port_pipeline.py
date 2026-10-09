#!/usr/bin/env python3
"""One explicit launcher for the complete MATLAB-port verification pipeline."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "python" / "run_matlab_port.py"


def command(stage, dimension, output, args, *extra):
    return [sys.executable, str(RUNNER), "--stage", stage,
            "--dimension", str(dimension), "--output", str(output),
            "--methods", *args.methods, *map(str, extra)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT/"results"/"matlab_port"/
                        f"pipeline_{datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument("--fixture", type=Path,
                        default=ROOT/"results"/"matlab_port"/"matlab_golden.json")
    parser.add_argument("--methods", nargs="+", choices=("ppo", "onto_rgat_ppo",
                        "shuffled_rgat_ppo"), default=["ppo", "onto_rgat_ppo"])
    parser.add_argument("--updates", type=int, default=750)
    parser.add_argument("--episodes-per-update", type=int, default=6)
    parser.add_argument("--validation-episodes", type=int, default=100)
    parser.add_argument("--evaluation-episodes", type=int, default=200)
    parser.add_argument("--test-seed-start", type=int, default=3001)
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--isaac-episodes", type=int, default=1)
    parser.add_argument("--isaac-seed-start", type=int, default=12000)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--planar-checkpoint-root", type=Path)
    parser.add_argument("--spatial-checkpoint-root", type=Path)
    parser.add_argument("--no-isaac", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--adopt-stack", action="store_true")
    parser.add_argument("--reset-recoveries", type=int, default=2)
    parser.add_argument("--fresh-stack-per-episode", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.skip_training and (args.planar_checkpoint_root is None
                               or args.spatial_checkpoint_root is None):
        parser.error("--skip-training requires both checkpoint roots")
    args.run_root.mkdir(parents=True, exist_ok=True)
    records = []

    def run(name, cmd):
        records.append({"name": name, "command": cmd, "status": "PLANNED"})
        print(json.dumps({"pipeline_stage": name, "command": cmd}), flush=True)
        if args.dry_run:
            return
        subprocess.run(cmd, cwd=ROOT, check=True)
        records[-1]["status"] = "PASSED"
        (args.run_root/"pipeline_progress.json").write_text(
            json.dumps(records, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")

    for dimension in (2, 3):
        base = args.run_root/f"{dimension}d"
        run(f"audit-{dimension}d", command("audit", dimension, base/"audit", args))
        if dimension == 2:
            run("matlab-parity", command("parity", 2, base/"parity", args,
                                         "--fixture", args.fixture))
        run(f"smoke-{dimension}d", command("smoke", dimension, base/"smoke", args,
                                            "--steps", min(args.steps, 32)))
        supplied = (args.planar_checkpoint_root if dimension == 2
                    else args.spatial_checkpoint_root)
        checkpoint_root = supplied or base/"train"
        if not args.skip_training:
            run(f"train-{dimension}d", command(
                "train", dimension, checkpoint_root, args,
                "--updates", args.updates,
                "--episodes-per-update", args.episodes_per_update,
                "--validation-episodes", args.validation_episodes,
                "--steps", args.steps))
        common = ("--checkpoint-root", checkpoint_root,
                  "--evaluation-episodes", args.evaluation_episodes,
                  "--seed", args.test_seed_start,
                  "--steps", args.steps)
        run(f"evaluate-deterministic-{dimension}d", command(
            "evaluate", dimension, base/"evaluate_deterministic", args, *common))
        run(f"evaluate-sampled-{dimension}d", command(
            "evaluate", dimension, base/"evaluate_sampled", args,
            *common, "--sample-actions"))
        if not args.no_isaac:
            isaac_extra = ["--backend", "isaac", "--allow-isaac",
                           "--checkpoint-root", checkpoint_root,
                           "--evaluation-episodes", args.isaac_episodes,
                           "--seed", args.isaac_seed_start,
                           "--steps", args.steps,
                           "--reset-recoveries", args.reset_recoveries]
            if args.headless:
                isaac_extra.append("--headless")
            if args.adopt_stack:
                isaac_extra.append("--adopt-stack")
            if args.fresh_stack_per_episode:
                isaac_extra.append("--fresh-stack-per-episode")
            run(f"isaac-{dimension}d", command(
                "evaluate", dimension, base/"isaac", args, *isaac_extra))
        run(f"report-{dimension}d", command("report", dimension, base/"report", args))

    status = "NOT_RUN" if args.dry_run else "PASSED"
    analysis = {"status": status, "dimensions": {},
                "interpretation": (
                    "single training seed; report execution and paired outcomes, "
                    "not representation superiority")}
    if not args.dry_run:
        for dimension in (2, 3):
            base = args.run_root/f"{dimension}d"
            supplied = (args.planar_checkpoint_root if dimension == 2
                        else args.spatial_checkpoint_root)
            checkpoint_root = supplied or base/"train"
            summary_path = checkpoint_root/"summary.json"
            training = (json.loads(summary_path.read_text())
                        if summary_path.is_file() else {"methods": {}})
            deterministic = json.loads(
                (base/"evaluate_deterministic"/"evaluate.json").read_text())
            sampled = json.loads((base/"evaluate_sampled"/"evaluate.json").read_text())

            def metrics(payload):
                return {method: {key: row[key] for key in
                        ("success_rate", "unsafe_rate", "mean_return")}
                        for method, row in payload["methods"].items()}

            row = {
                "checkpoint_root": str(checkpoint_root),
                "selected_validation": {
                    method: values.get("best")
                    for method, values in training["methods"].items()},
                "training_summary_available": summary_path.is_file(),
                "deterministic_test": metrics(deterministic),
                "sampled_test": metrics(sampled),
            }
            if not args.no_isaac:
                live = json.loads((base/"isaac"/"evaluate.json").read_text())
                row["isaac"] = metrics(live)
                row["isaac_episodes"] = {
                    method: values["episodes"] for method, values in live["methods"].items()
                }
            analysis["dimensions"][f"{dimension}d"] = row
        (args.run_root/"analysis.json").write_text(
            json.dumps(analysis, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    report = {"status": status, "run_root": str(args.run_root),
              "isaac_requested": not args.no_isaac,
              "training_executed": not args.skip_training,
              "training_artifacts_reused": bool(args.skip_training),
              "analysis_artifact": (None if args.dry_run
                                    else str(args.run_root/"analysis.json")),
              "stages": records}
    (args.run_root/"pipeline_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
