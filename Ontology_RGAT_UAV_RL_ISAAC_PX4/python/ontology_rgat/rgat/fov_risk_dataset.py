"""Episode-separated labels for future *geometric* field-of-view loss.

The supervised target is whether the landing-pad centre leaves the camera
frustum within the prediction horizon, as defined once by
``isaac_sim/keypoint_geometry.geometric_pad_center_in_fov``.  It is deliberately
not marker-detection success and not learned keypoint confidence: a model
trained on detector failures would predict future *detector* failures, which is
a different quantity from the one the proposed reward is meant to discourage.

The labels are simulator truth and are offline-only.  They are never handed to
R-GAT inference, which sees the eight visual features of ``fov_graph.py``.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .fov_graph import (FOV_GRAPH_INPUT_DIM, FOV_GRAPH_VERSION,
                        FOV_NODE_NAMES, FOV_RELATION_NAMES)


FOV_RISK_DATASET_FORMAT = "ontology_rgat.future_fov_loss/1"


def horizon_steps(horizon_seconds: float, control_hz: float) -> int:
    seconds = float(horizon_seconds)
    frequency = float(control_hz)
    if not np.isfinite(seconds) or seconds <= 0.0:
        raise ValueError("prediction horizon must be positive")
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("control frequency must be positive")
    return max(1, int(round(seconds * frequency)))


def future_fov_loss_labels(geometric_in_fov, prediction_steps: int) -> np.ndarray:
    """1 when geometric pad-centre FOV is lost within the next N control steps.

    At 10 Hz control and a 1.0 s horizon, ``prediction_steps`` is 10 and the
    label covers samples ``t+1 .. t+10``.
    """
    visible = np.asarray(geometric_in_fov, dtype=bool).reshape(-1)
    steps = int(prediction_steps)
    if visible.size == 0 or steps < 1:
        raise ValueError("FOV labels require a trajectory and positive horizon")
    labels = np.zeros(visible.size, dtype=np.float32)
    for index in range(visible.size):
        stop = min(visible.size, index + steps + 1)
        labels[index] = float(np.any(~visible[index + 1:stop]))
    return labels


def build_fov_risk_dataset(episodes: Sequence[Mapping], *, prediction_steps: int) -> dict:
    matrices, labels, metadata = [], [], []
    for episode in episodes:
        episode_id = int(episode["episode_id"])
        seed = int(episode["seed"])
        samples = list(episode["samples"])
        if not samples:
            raise ValueError("FOV-risk episode cannot be empty")
        for sample in samples:
            if set(sample) != {"graph_X", "geometric_in_fov"}:
                raise ValueError(
                    "FOV-risk samples accept only graph_X and geometric_in_fov; "
                    "the label must come from simulator pad-centre geometry")
        episode_labels = future_fov_loss_labels(
            [sample["geometric_in_fov"] for sample in samples], prediction_steps)
        for time_index, (sample, label) in enumerate(zip(samples, episode_labels)):
            matrix = np.asarray(sample["graph_X"], dtype=np.float32)
            if matrix.shape != (FOV_GRAPH_INPUT_DIM, len(FOV_NODE_NAMES)):
                raise ValueError("FOV-risk graph feature shape mismatch")
            if not np.isfinite(matrix).all():
                raise ValueError("FOV-risk graph contains non-finite values")
            matrices.append(matrix.T)
            labels.append(label)
            metadata.append((episode_id, time_index, seed))
    if not matrices:
        raise ValueError("FOV-risk dataset requires at least one episode")
    return {
        "X": np.stack(matrices).astype(np.float32),
        "y": np.asarray(labels, dtype=np.float32),
        "meta": np.asarray(metadata, dtype=np.int64),
        "prediction_steps": int(prediction_steps),
    }


def validate_fov_risk_dataset(dataset: Mapping) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(dataset["X"], dtype=np.float32)
    y = np.asarray(dataset["y"], dtype=np.float32).reshape(-1)
    meta = np.asarray(dataset["meta"], dtype=np.int64)
    expected = (len(FOV_NODE_NAMES), FOV_GRAPH_INPUT_DIM)
    if X.ndim != 3 or X.shape[1:] != expected:
        raise ValueError("FOV-risk dataset X shape mismatch")
    if y.shape != (X.shape[0],) or meta.shape != (X.shape[0], 3):
        raise ValueError("FOV-risk labels/metadata shape mismatch")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("FOV-risk dataset contains non-finite values")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("FOV-risk labels must be binary")
    if int(dataset.get("prediction_steps", 0)) < 1:
        raise ValueError("FOV-risk dataset prediction_steps must be positive")
    return X, y, meta


def split_by_episode(dataset: Mapping, *, validation_fraction: float = 0.2,
                     seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    _, _, meta = validate_fov_risk_dataset(dataset)
    episodes = np.unique(meta[:, 0])
    if episodes.size < 2:
        raise ValueError("episode-level split requires at least two episodes")
    fraction = float(validation_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("validation_fraction must be in (0,1)")
    shuffled = episodes.copy()
    np.random.default_rng(int(seed)).shuffle(shuffled)
    validation_count = min(len(shuffled) - 1, max(1, int(round(len(shuffled) * fraction))))
    validation_episodes = set(int(value) for value in shuffled[:validation_count])
    validation = np.asarray(
        [int(episode) in validation_episodes for episode in meta[:, 0]], dtype=bool)
    training = ~validation
    if set(meta[training, 0]) & set(meta[validation, 0]):
        raise AssertionError("episode leakage across FOV-risk train/validation split")
    return training, validation


def dataset_digest(dataset: Mapping) -> str:
    X, y, meta = validate_fov_risk_dataset(dataset)
    digest = hashlib.sha256()
    for name, value in (("X", X), ("y", y), ("meta", meta)):
        array = np.ascontiguousarray(value)
        digest.update(name.encode("ascii"))
        digest.update(array.tobytes())
    digest.update(str(int(dataset["prediction_steps"])).encode("ascii"))
    return digest.hexdigest()


def save_fov_risk_dataset(dataset: Mapping, path: str | Path, *,
                          config_hash: str, seed: int,
                          horizon_seconds: float, control_hz: float) -> dict:
    X, y, meta = validate_fov_risk_dataset(dataset)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, X=X, y=y, meta=meta,
                            prediction_steps=int(dataset["prediction_steps"]))
    os.replace(temporary, path)
    manifest = {
        "format": FOV_RISK_DATASET_FORMAT,
        "dataset_version": FOV_RISK_DATASET_FORMAT,
        "graph_version": FOV_GRAPH_VERSION,
        "node_names": list(FOV_NODE_NAMES),
        "relation_names": list(FOV_RELATION_NAMES),
        "dataset_config_hash": str(config_hash),
        "seed": int(seed),
        "prediction_horizon_seconds": float(horizon_seconds),
        "control_hz": float(control_hz),
        "prediction_horizon_steps": int(dataset["prediction_steps"]),
        "episodes": int(len(np.unique(meta[:, 0]))),
        "samples": int(len(y)),
        "positive_samples": int(np.sum(y)),
        "dataset_sha256": dataset_digest(dataset),
        "feature_source": "eight visual/keypoint history features only",
        "label_source": (
            "geometric pad-centre camera-frustum visibility (simulator truth, "
            "offline only)"),
        "forbidden_inputs": ["simulator truth", "relative-state estimate",
                             "critic state", "marker detection quality"],
    }
    manifest_path = path.with_suffix(".manifest.json")
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    return manifest


def load_fov_risk_dataset(path: str | Path, *, config_hash: str | None = None):
    path = Path(path)
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != FOV_RISK_DATASET_FORMAT:
        raise ValueError("unsupported FOV-risk dataset format")
    if manifest.get("graph_version") != FOV_GRAPH_VERSION:
        raise ValueError("FOV-risk dataset graph version mismatch")
    if config_hash is not None and manifest.get("dataset_config_hash") != config_hash:
        raise ValueError("FOV-risk dataset configuration mismatch")
    with np.load(path, allow_pickle=False) as payload:
        dataset = {name: payload[name] for name in ("X", "y", "meta")}
        dataset["prediction_steps"] = int(payload["prediction_steps"])
    validate_fov_risk_dataset(dataset)
    if dataset_digest(dataset) != manifest.get("dataset_sha256"):
        raise ValueError("FOV-risk dataset checksum mismatch")
    return dataset, manifest

