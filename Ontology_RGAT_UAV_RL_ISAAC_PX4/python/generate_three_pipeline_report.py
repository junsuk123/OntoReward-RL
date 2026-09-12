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
    parser.add_argument(
        "--replicate-dirs", nargs="+", type=Path,
        help="explicit completed replicate directories to combine")
    parser.add_argument(
        "--output-dir", type=Path,
        help="combined report destination; defaults to RESULTS_DIR/combined")
    args = parser.parse_args()

    sources = list(args.replicate_dirs or [args.results_dir])
    if args.replicate_dirs is None:
        sources.extend(sorted(
            path for path in args.results_dir.glob("replicate_*")
            if (path / "manifest.json").is_file()))
    output_dir = (args.output_dir or
                  (args.results_dir / "combined" if len(sources) > 1
                   else args.results_dir))
    evaluations = []
    training = []
    manifests = []
    seen_replicates = set()
    for source in sources:
        manifest_path = source / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing replicate manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        replicate = str(manifest.get("training_replicate", 0))
        if replicate in seen_replicates:
            raise ValueError(f"duplicate training replicate {replicate}: {source}")
        seen_replicates.add(replicate)
        manifests.append(manifest)
        evaluation_path = source / "evaluation/per_episode.csv"
        if not evaluation_path.is_file():
            raise FileNotFoundError(f"missing replicate evaluation: {evaluation_path}")
        replicate_evaluations = _read(evaluation_path)
        for row in replicate_evaluations:
            row["training_replicate"] = replicate
        evaluations.extend(replicate_evaluations)
        for pipeline in primary_pipeline_ids():
            path = source / f"training/{pipeline}.csv"
            if not path.is_file():
                continue
            rows = _read(path)
            for row in rows:
                row["training_replicate"] = replicate
            training.extend(rows)

    def mean_cost(key):
        values = [int(manifest.get(key, 0)) for manifest in manifests]
        return int(round(sum(values) / max(len(values), 1)))

    write_three_pipeline_outputs(
        evaluations, training, output_dir,
        reward_design_episodes=mean_cost("N_reward_design"),
        reward_design_steps=mean_cost("reward_design_environment_steps"),
        estimator_warmup_episodes=mean_cost("N_estimator_warmup"),
        estimator_warmup_steps=mean_cost("estimator_warmup_environment_steps"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "combined_manifest.json").write_text(json.dumps({
        "format": "ontology_rgat.three_pipeline_combined_report/1",
        "source_directories": [str(path.resolve()) for path in sources],
        "training_replicates": sorted(seen_replicates),
        "confidence_interval": (
            "hierarchical training-replicate/episode bootstrap"),
        "reward_design_cost_per_replicate": mean_cost("N_reward_design"),
        "replicate_costs": {
            str(manifest.get("training_replicate", 0)): {
                "N_PPO": int(manifest.get("N_PPO", 0)),
                "N_reward_design": int(manifest.get("N_reward_design", 0)),
                "N_estimator_warmup": int(manifest.get(
                    "N_estimator_warmup", 0)),
            } for manifest in manifests
        },
    }, indent=2), encoding="utf-8")
    print(f"Reports regenerated under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
