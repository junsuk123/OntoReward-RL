"""Catch up on whatever was collected but not yet learned.

Run at every system start. It asks two questions in order:

1. Is there a collection session on disk the cumulative dataset has not
   absorbed? Absorb it -- SAM-refine the simulator's occlusion-aware prompts and
   append the frames.
2. Is there an absorbed session the deployed weights have never trained on? Then
   fine-tune from the latest weights over the whole cumulative dataset and let
   the quality gate decide whether the result is allowed to be deployed.

If neither is true it does nothing and says so, which is what makes it safe to
put in front of every launch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from simlab.config import load_config
from simlab.config.schema import SceneConfig
from simlab.ml.auto_label_train import (
    dataset_images,
    ingest_pending,
    train_and_gate,
)
from simlab.ml.lineage import (
    active_model_for_signature,
    dataset_dir,
    drone_model_signature,
    lineage_dir,
    untrained_sessions,
)
from simlab.utils.paths import resolve_path


def load_perception_config(path: str | Path = "configs/perception.yaml") -> Dict[str, Any]:
    return yaml.safe_load(resolve_path(path).read_text(encoding="utf-8"))


def catch_up(
    perception: Dict[str, Any],
    scene: SceneConfig,
    force_train: bool = False,
    force_base_model: bool = False,
) -> Dict[str, Any]:
    """Ingest pending sessions, then train if anything is still unlearned."""
    artifacts = resolve_path(perception["artifacts_root"])
    signature = drone_model_signature(scene)
    dataset = dataset_dir(artifacts, signature)
    lineage = lineage_dir(artifacts, signature)
    raw_root = resolve_path(scene.scenarios.dataset_root)

    ingested = ingest_pending(raw_root, dataset, perception)
    if ingested:
        print(f"[catch-up] ingested {len(ingested)} new session(s): {', '.join(ingested)}")
    else:
        print("[catch-up] no new collection sessions to ingest")

    pending = untrained_sessions(dataset, lineage)
    images = dataset_images(dataset)
    minimum = int(perception["training"]["minimum_images"])
    report: Dict[str, Any] = {
        "signature": signature,
        "dataset": str(dataset),
        "dataset_images": images,
        "ingested_sessions": ingested,
        "untrained_sessions": pending,
        "active_model": active_model_for_signature(artifacts, signature),
        "trained": False,
        "accepted": False,
    }

    if not pending and not force_train:
        print("[catch-up] every collected session is already learned; skipping training")
        report["reason"] = "up_to_date"
        return report
    if images < minimum:
        print(
            f"[catch-up] {images} labelled images on hand, {minimum} needed; "
            "collect more before training"
        )
        report["reason"] = "insufficient_images"
        return report

    print(
        f"[catch-up] {len(pending)} unlearned session(s); fine-tuning over "
        f"{images} cumulative images"
    )
    accepted = train_and_gate(
        dataset,
        artifacts,
        perception,
        signature,
        force_base_model=force_base_model,
        sessions=pending,
    )
    report.update({"trained": True, "accepted": accepted, "reason": "trained"})
    report["active_model"] = active_model_for_signature(artifacts, signature)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/perception.yaml")
    parser.add_argument("--scene", default="configs/default.yaml")
    parser.add_argument(
        "--force", action="store_true", help="train even when nothing is unlearned"
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="start this run from the base weights, keeping the cumulative dataset",
    )
    args = parser.parse_args()
    report = catch_up(
        load_perception_config(args.config),
        load_config(args.scene),
        force_train=args.force,
        force_base_model=args.force_retrain,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
