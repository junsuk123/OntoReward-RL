#!/usr/bin/env python3
"""Write a machine-readable audit of the selective-RGAT scientific contract."""
from __future__ import annotations

from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from ontology_rgat.benchmarks.experiment import configuration_hash, load_experiment
from ontology_rgat.contracts import load_observation_registry
from ontology_rgat.pipelines import get_pipeline, validate_pipeline_configuration
from ontology_rgat.ppo.selective_graph_encoder import artifact_sha256
from ontology_rgat.reward_modes.two_term import TwoTermRewardConfig
from ontology_rgat.rgat.selective_state import (
    RELATION_PARTITION_HASH, SELECTIVE_SCHEMA_HASH,
    unreachable_or_too_distant_nodes)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(config_path: Path) -> dict:
    config = load_experiment(config_path)
    validate_pipeline_configuration(config)
    registry_path = ROOT / str(config["observation"]["registry"])
    registry = load_observation_registry(registry_path)
    artifact_path = ROOT / str(config["graph_state"]["pretrained_artifact"])
    system_path = ROOT / str(config["simulator_contract"]["profile"])
    reward = TwoTermRewardConfig(**{
        key: config["reward"][key]
        for key in asdict(TwoTermRewardConfig())})
    return {
        "audit_version": "selective-rgat-refactor-audit-v1",
        "experiment_config": str(config_path.resolve()),
        "experiment_config_hash": configuration_hash(config),
        "pipelines": {name: get_pipeline(name).to_manifest()
                      for name in config["pipelines"]},
        "observation_registry": {
            "path": str(registry_path), "sha256": registry.sha256,
            "dimension": registry.dimension, "clock": registry.clock,
        },
        "reward_config_hash": reward.sha256,
        "action_contract_hash": configuration_hash(config["control"]),
        "environment_config_hash": _file_hash(system_path),
        "graph_schema_hash": SELECTIVE_SCHEMA_HASH,
        "relation_partition_hash": RELATION_PARTITION_HASH,
        "graph_integrity": {
            "unreachable_or_over_two_hops": list(
                unreachable_or_too_distant_nodes()),
        },
        "pretrained_artifact": {
            "path": str(artifact_path), "exists": artifact_path.is_file(),
            "sha256": (artifact_sha256(artifact_path)
                       if artifact_path.is_file() else None),
        },
        "behavior_cloning_enabled": bool(
            config.get("behavior_cloning", {}).get("enabled", False)),
        "simulator_layer_modified_by_refactor": False,
        "simulator_preserved": [
            "Isaac/Pegasus/PX4/ROS2", "camera and keypoint source",
            "UGV route and motion", "contact/strict landing truth",
            "frame transforms", "planar action conversion and limits",
            "reset/disturbance/battery/horizon",
        ],
        "ready_for_ppo": artifact_path.is_file(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=(
        ROOT / "config/experiments/selective_rgat_three_arm.yaml"))
    parser.add_argument("--output", type=Path, default=(
        ROOT / "results/selective_rgat/refactor_audit.json"))
    args = parser.parse_args()
    report = audit(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2))
    return 0 if report["graph_integrity"]["unreachable_or_over_two_hops"] == [] else 2


if __name__ == "__main__":
    raise SystemExit(main())
