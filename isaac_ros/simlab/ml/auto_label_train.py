"""Refine simulator prompts with SAM, train YOLO, and enforce a metric gate.

Collection and training are deliberately decoupled. Every episode drops a raw
session on disk; ingestion folds any session the dataset has not seen into one
cumulative dataset; training runs from the latest weights over that whole
dataset whenever a session exists that the deployed model has not learned yet.
Fine-tuning on the cumulative set rather than only on the new frames is what
keeps a run in a new environment from erasing the previous ones.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import yaml
from ultralytics import SAM, YOLO

from simlab.ml.lineage import (
    active_model_for_signature,
    ingested_sessions,
    latest_model_for_signature,
    lineage_dir,
    pending_raw_sessions,
    record_trained_sessions,
    write_model_pointer,
)


def box_iou(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def mask_box(mask: np.ndarray) -> List[float] | None:
    ys, xs = np.where(mask > 0.5)
    if not len(xs):
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def refine_boxes(model: SAM, image: np.ndarray, prompts: List[List[float]], minimum: float):
    if not prompts:
        return [], []
    try:
        prediction = model.predict(
            source=image,
            bboxes=np.asarray(prompts, dtype=np.float32),
            device=0 if torch.cuda.is_available() else "cpu",
            verbose=False,
        )[0]
        masks = prediction.masks.data.cpu().numpy() if prediction.masks is not None else []
    except Exception as exc:
        print(f"[auto-label] SAM failed, using simulator prompts: {exc}")
        masks = []
    boxes, scores = [], []
    for index, prompt in enumerate(prompts):
        refined = mask_box(masks[index]) if index < len(masks) else None
        consistency = box_iou(prompt, refined) if refined else 0.0
        boxes.append(refined if refined and consistency >= minimum else prompt)
        scores.append(consistency)
    return boxes, scores


def yolo_line(box: List[float], width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    return (
        f"0 {((x1 + x2) * 0.5 / width):.8f} {((y1 + y2) * 0.5 / height):.8f} "
        f"{((x2 - x1) / width):.8f} {((y2 - y1) / height):.8f}"
    )


# -- ingestion --------------------------------------------------------------
def _visible_objects(record: dict, minimum_visibility: float) -> List[dict]:
    """Objects the camera could see. Records without a visibility field predate
    occlusion-aware collection and are taken at face value."""
    return [
        obj
        for obj in record.get("objects", [])
        if float(obj.get("visibility", 1.0)) >= minimum_visibility
    ]


def _select_records(raw: Path, cfg: dict) -> Tuple[List[dict], int]:
    """Split a session into labelled frames plus a bounded share of negatives.

    A frame where every aircraft is hidden is not a broken sample -- it is the
    background a detector must not fire on, and occlusion episodes produce a lot
    of them. Keeping a capped fraction teaches precision; keeping all of them
    would drown the positives.
    """
    minimum_visibility = float(cfg["auto_label"].get("label_min_visibility", 0.0))
    negative_fraction = float(cfg["auto_label"].get("negative_fraction", 0.0))
    lines = (raw / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    positives, negatives = [], []
    for record in records:
        objects = _visible_objects(record, minimum_visibility)
        record["objects"] = objects
        (positives if objects else negatives).append(record)
    generator = random.Random(int(cfg["training"]["seed"]))
    generator.shuffle(negatives)
    keep = min(len(negatives), int(len(positives) * negative_fraction))
    chosen = positives + negatives[:keep]
    generator.shuffle(chosen)
    return chosen, keep


def ingest_session(raw: Path, dataset: Path, cfg: dict, sam: SAM | None = None) -> int:
    """Fold one collection session into the cumulative dataset. Returns images."""
    session_id = raw.name
    session_record = dataset / "sessions" / f"{session_id}.json"
    if session_record.exists():
        return 0
    records, negatives = _select_records(raw, cfg)
    if not records:
        print(f"[auto-label] {session_id}: no usable frames, skipped")
        return 0

    split_at = min(len(records) - 1, max(1, round(len(records) * 0.8)))
    for split in ("train", "val"):
        (dataset / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset / "labels" / split).mkdir(parents=True, exist_ok=True)
    (dataset / "by_drone").mkdir(parents=True, exist_ok=True)
    session_record.parent.mkdir(parents=True, exist_ok=True)

    if sam is None:
        sam = SAM(cfg["auto_label"]["sam_model"])
    minimum = float(cfg["auto_label"]["sam_min_prompt_iou"])
    accepted_sam, total = 0, 0
    for index, record in enumerate(records):
        source = raw / record["image"]
        image = cv2.imread(str(source))
        if image is None:
            continue
        prompts = [obj["bbox_xyxy"] for obj in record["objects"]]
        boxes, scores = refine_boxes(sam, image, prompts, minimum)
        split = "train" if index < split_at else "val"
        # Simulation timestamps restart on every launch, so prefix the session to
        # prevent a later run from overwriting an earlier image and label.
        target_image = dataset / "images" / split / f"{session_id}_{source.name}"
        shutil.copy2(source, target_image)
        label = dataset / "labels" / split / f"{target_image.stem}.txt"
        lines = [yolo_line(box, record["width"], record["height"]) for box in boxes]
        # An empty label file is how YOLO is told "this frame is background".
        label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        for obj, box, score in zip(record["objects"], boxes, scores):
            accepted_sam += int(score >= minimum)
            total += 1
            item = {
                "image": str(target_image),
                "camera": record["camera"],
                "stamp_ns": record["stamp_ns"],
                "scenario": record.get("scenario"),
                "environment": record.get("environment"),
                "bbox_xyxy": box,
                "visibility": obj.get("visibility"),
                "state": obj.get("state"),
                "sam_prompt_iou": round(score, 5),
                "label_source": "sam" if score >= minimum else "simulator_projection",
            }
            with (dataset / "by_drone" / f"{obj['drone_id']}.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(item, sort_keys=True) + "\n")

    session_meta = raw / "session.json"
    meta = json.loads(session_meta.read_text(encoding="utf-8")) if session_meta.is_file() else {}
    session_record.write_text(
        json.dumps(
            {
                "session": session_id,
                "images": len(records),
                "objects": total,
                "negatives": negatives,
                "scenario": meta.get("scenario"),
                "environment": meta.get("environment"),
                "episode_id": meta.get("episode_id"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[auto-label] {session_id}: images={len(records)} (negatives={negatives}), "
        f"objects={total}, SAM-accepted={accepted_sam}"
    )
    return len(records)


def write_data_yaml(dataset: Path) -> Path:
    data_yaml = {
        "path": str(dataset.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "drone"},
    }
    path = dataset / "data.yaml"
    path.write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    return path


def dataset_images(dataset: Path) -> int:
    return sum(
        len(list((dataset / "images" / split).glob("*.jpg")))
        for split in ("train", "val")
        if (dataset / "images" / split).is_dir()
    )


def ingest_pending(raw_root: Path, dataset: Path, cfg: dict) -> List[str]:
    """Absorb every collected session the dataset is missing, oldest first."""
    pending = pending_raw_sessions(raw_root, dataset)
    if not pending:
        return []
    sam = SAM(cfg["auto_label"]["sam_model"])
    ingested = []
    for raw in pending:
        if ingest_session(raw, dataset, cfg, sam):
            ingested.append(raw.name)
    if ingested:
        write_data_yaml(dataset)
    return ingested


def build_dataset(raw: Path, dataset: Path, cfg: dict) -> int:
    """Ingest a single session and refresh the dataset manifest."""
    count = ingest_session(raw, dataset, cfg)
    if not count:
        raise RuntimeError(f"collection session {raw.name} produced no usable frames")
    write_data_yaml(dataset)
    total = dataset_images(dataset)
    minimum = int(cfg["training"]["minimum_images"])
    if total < minimum:
        raise RuntimeError(f"training needs {minimum} labelled images; the dataset holds {total}")
    return count


# -- training ---------------------------------------------------------------
def train_and_gate(
    dataset: Path,
    artifacts: Path,
    cfg: dict,
    signature: str,
    force_base_model: bool = False,
    sessions: Sequence[str] = (),
) -> bool:
    training = cfg["training"]
    detector = cfg["detector"]
    previous = None if force_base_model else latest_model_for_signature(artifacts, signature)
    initial_model = previous or detector["base_model"]
    print(f"[continual-train] initial weights: {initial_model}")
    model = YOLO(initial_model)
    run_name = f"drone_yolo26s_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model.train(
        data=str(dataset / "data.yaml"),
        epochs=int(training["epochs"]),
        patience=int(training["patience"]),
        batch=int(training["batch"]),
        imgsz=int(training["image_size"]),
        seed=int(training["seed"]),
        device=0 if torch.cuda.is_available() else "cpu",
        project=str((artifacts / "training").resolve()),
        name=run_name,
        exist_ok=False,
        verbose=True,
    )
    best = Path(model.trainer.best)
    metrics = YOLO(str(best)).val(
        data=str(dataset / "data.yaml"),
        imgsz=int(training["image_size"]),
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=False,
    )
    score = float(metrics.box.map)
    threshold = float(training["acceptance_map50_95"])
    lineage = lineage_dir(artifacts, signature)
    lineage.mkdir(parents=True, exist_ok=True)
    active_report_path = lineage / "active_metrics.json"
    active_score = None
    if active_report_path.exists():
        active_score = float(json.loads(active_report_path.read_text(encoding="utf-8"))["score"])
    required_score = max(threshold, active_score) if active_score is not None else threshold
    accepted = score >= required_score
    report = {
        "model": str(best.resolve()),
        "initial_model": str(initial_model),
        "signature": signature,
        "dataset": str(dataset.resolve()),
        "metric": "mAP50-95",
        "score": score,
        "threshold": threshold,
        "incumbent_score": active_score,
        "required_score": required_score,
        "accepted": accepted,
        "sessions_learned": list(sessions),
        "dataset_images": dataset_images(dataset),
    }
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (lineage / "metrics_history.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(report, sort_keys=True) + "\n")
    # Even a rejected candidate remains the next fine-tuning starting point.
    # Deployment is stricter: it only advances when the candidate clears both
    # the absolute gate and the incumbent's score.
    write_model_pointer(lineage / "latest_model.txt", best)
    # The sessions are learned either way -- the weights have seen them, and
    # re-running the same fine-tune on the same data would only repeat itself.
    record_trained_sessions(lineage, sessions or ingested_sessions(dataset), best)
    if accepted:
        write_model_pointer(lineage / "active_model.txt", best)
        write_model_pointer(artifacts / "active_model.txt", best)
        (artifacts / "active_signature.txt").write_text(signature + "\n", encoding="utf-8")
        active_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"[quality-gate] mAP50-95={score:.4f}, required={required_score:.4f}, "
        f"accepted={accepted}"
    )
    return accepted


def accepted_model(cfg: dict, signature: str) -> str | None:
    artifacts = Path(cfg["artifacts_root"])
    return active_model_for_signature(artifacts, signature)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/perception.yaml")
    parser.add_argument("--raw", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--signature", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset = Path(args.dataset)
    build_dataset(Path(args.raw), dataset, cfg)
    if not train_and_gate(
        dataset, Path(cfg["artifacts_root"]), cfg, args.signature, sessions=[Path(args.raw).name]
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
