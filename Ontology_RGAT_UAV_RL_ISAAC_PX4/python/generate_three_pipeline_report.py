#!/usr/bin/env python3
"""Regenerate physical-metric reports from completed three-pipeline CSVs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ontology_rgat.evaluation import write_three_pipeline_outputs
from ontology_rgat.pipelines import primary_pipeline_ids


def _read(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path,
                        default=ROOT / "results/three_pipeline/full")
    args = parser.parse_args()
    evaluations = _read(args.results_dir / "evaluation/per_episode.csv")
    training = []
    for pipeline in primary_pipeline_ids():
        path = args.results_dir / f"training/{pipeline}.csv"
        if path.is_file():
            training.extend(_read(path))
    manifest_path = args.results_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    write_three_pipeline_outputs(
        evaluations, training, args.results_dir,
        reward_design_episodes=int(manifest.get("N_reward_design", 0)),
        reward_design_steps=int(manifest.get(
            "reward_design_environment_steps", 0)),
        estimator_warmup_episodes=int(manifest.get(
            "N_estimator_warmup", 0)),
        estimator_warmup_steps=int(manifest.get(
            "estimator_warmup_environment_steps", 0)))
    print(f"Reports regenerated under {args.results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
