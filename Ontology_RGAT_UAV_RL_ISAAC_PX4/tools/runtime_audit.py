#!/usr/bin/env python3
"""Read-only process/artifact audit. Does not connect to or reset a vehicle."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FLIGHT_PROGRAMS = {"landing_world.py", "MicroXRCEAgent", "px4", "ros2_gateway",
                   "run_two_pipeline.py", "run_three_pipeline.py"}
LEARNER_PROGRAMS = {"run_two_axis_pipeline.py", "run_spatial_pipeline.py"}


def relevant_programs(tokens):
    """Inspect executable/script tokens only, never a shell's command text."""
    if not tokens or Path(tokens[0]).name in {"bash", "sh", "rg", "grep", "timeout"}:
        return []
    names = {Path(token).name for token in tokens[:3] if token}
    return sorted(names & (FLIGHT_PROGRAMS | LEARNER_PROGRAMS))


def process_snapshot():
    processes = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            tokens = (proc / "cmdline").read_bytes().decode(errors="replace").split("\0")
            programs = relevant_programs(tokens)
            if programs:
                processes.append({"pid": int(proc.name), "programs": programs})
        except (OSError, IndexError):
            pass
    return processes


def audit(root=ROOT):
    processes = process_snapshot()
    flight = any(set(p["programs"]) & FLIGHT_PROGRAMS for p in processes)
    result = {
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True, "processes": processes,
        "flight_state": "running_unverified" if flight else "stopped",
        "pipeline_state": "running_unverified" if any(
            set(p["programs"]) & LEARNER_PROGRAMS for p in processes) else "stopped",
        "operational_stability_verified": False,
        "note": "Process presence is not health. No heartbeat or commanded-flight test was performed.",
        "artifacts": {},
    }
    for directory in sorted((root / "results/two_axis_gate_release/runs").glob("*/summary.json")):
        payload = json.loads(directory.read_text(), parse_constant=lambda value: None)
        result["artifacts"][payload["arm"]] = {
            "path": str(directory.relative_to(root)),
            "environment_steps": payload["environment_steps"],
            "last_training_window": payload["training_rates_last_400"],
            "selected_validation": payload["selected_checkpoint"],
            "test_evidence": "not evaluated by this audit",
        }
    evaluation = root / "results/isaac_pilot_no_bc/evaluation/per_episode.csv"
    if evaluation.exists():
        with evaluation.open() as handle:
            rows = list(csv.DictReader(handle))
        key = "method" if rows and "method" in rows[0] else "pipeline"
        result["isaac_pilot_evaluation_rows"] = dict(Counter(row.get(key, "unknown") for row in rows))
    runs = root / "logs/runs"
    if runs.exists():
        dirs = sorted(runs.glob("run-*"))
        result["stack_run_directories"] = len(dirs)
        result["latest_stack_directory"] = dirs[-1].name if dirs else None
    ports = subprocess.run(["ss", "-lnuH"], text=True, capture_output=True, timeout=3)
    result["bound_flight_udp_ports"] = [port for port in (8888, *range(14650,14658))
        if any(line.split()[3].rsplit(":",1)[-1] == str(port)
               for line in ports.stdout.splitlines() if len(line.split()) >= 4)]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="optionally save the diagnostic JSON")
    args = parser.parse_args()
    payload = json.dumps(audit(), indent=2, ensure_ascii=False, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
