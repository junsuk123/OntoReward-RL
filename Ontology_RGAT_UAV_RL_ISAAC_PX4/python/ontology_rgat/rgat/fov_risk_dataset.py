"""Episode-separated targets for future *geometric* field-of-view loss.

The supervised target is the fraction of the next ``H`` control steps in which
the landing-pad centre is outside the camera frustum, as defined once by
``isaac_sim/keypoint_geometry.geometric_pad_center_in_fov``::

    y_t = sum(1 - V_centre[t+k] for k in 1..H) / H

It is deliberately not marker-detection success and not learned keypoint
confidence: a model trained on detector failures would predict future
*detector* failures, which is a different quantity from the one the proposed
reward is meant to discourage.

Samples whose full ``H``-step future was not observed -- the tail of every
episode -- carry ``valid=False`` and a NaN target.  They are never filled with
zero and episodes are never concatenated, so an unobserved future cannot be
read as "the pad stayed visible".

The targets are simulator truth and are offline-only.  They are never handed to
R-GAT inference, which sees the ten visual features of ``fov_graph.py``.
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


FOV_RISK_DATASET_FORMAT = "ontology_rgat.future_fov_unavailability/2"


def horizon_steps(horizon_seconds: float, control_hz: float) -> int:
    seconds = float(horizon_seconds)
    frequency = float(control_hz)
    if not np.isfinite(seconds) or seconds <= 0.0:
        raise ValueError("prediction horizon must be positive")
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("control frequency must be positive")
    return max(1, int(round(seconds * frequency)))


def future_fov_unavailability_targets(geometric_in_fov, prediction_steps: int):
    """Return ``(y, valid)`` for one episode's visibility timeline.

    ``y[t]`` is the fraction of steps ``t+1 .. t+H`` with the geometric pad
    centre outside the frustum.  ``valid[t]`` is False, and ``y[t]`` is NaN,
    when fewer than ``H`` future steps were observed.  At 10 Hz control and a
    1.0 s horizon, ``prediction_steps`` is 10 and the last ten samples of every
    episode are masked.
    """
    visible = np.asarray(geometric_in_fov, dtype=bool).reshape(-1)
    steps = int(prediction_steps)
    if visible.size == 0 or steps < 1:
        raise ValueError("FOV targets require a trajectory and positive horizon")
    y = np.full(visible.size, np.nan, dtype=np.float32)
    valid = np.zeros(visible.size, dtype=bool)
    for index in range(visible.size):
        stop = index + steps + 1
        if stop > visible.size:
            break
        window = visible[index + 1:stop]
        y[index] = float(np.mean(~window))
        valid[index] = True
    return y, valid


def build_fov_risk_dataset(episodes: Sequence[Mapping], *, prediction_steps: int) -> dict:
    matrices, targets, masks, metadata = [], [], [], []
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
                    "the target must come from simulator pad-centre geometry")
        episode_targets, episode_valid = future_fov_unavailability_targets(
            [sample["geometric_in_fov"] for sample in samples], prediction_steps)
        for time_index, sample in enumerate(samples):
            matrix = np.asarray(sample["graph_X"], dtype=np.float32)
            if matrix.shape != (FOV_GRAPH_INPUT_DIM, len(FOV_NODE_NAMES)):
                raise ValueError("FOV-risk graph feature shape mismatch")
            if not np.isfinite(matrix).all():
                raise ValueError("FOV-risk graph contains non-finite values")
            matrices.append(matrix.T)
            targets.append(episode_targets[time_index])
            masks.append(episode_valid[time_index])
            metadata.append((episode_id, time_index, seed))
    if not matrices:
        raise ValueError("FOV-risk dataset requires at least one episode")
    return {
        "X": np.stack(matrices).astype(np.float32),
        "y": np.asarray(targets, dtype=np.float32),
        "valid": np.asarray(masks, dtype=bool),
        "meta": np.asarray(metadata, dtype=np.int64),
        "prediction_steps": int(prediction_steps),
    }


def validate_fov_risk_dataset(dataset: Mapping):
    X = np.asarray(dataset["X"], dtype=np.float32)
    y = np.asarray(dataset["y"], dtype=np.float32).reshape(-1)
    valid = np.asarray(dataset["valid"], dtype=bool).reshape(-1)
    meta = np.asarray(dataset["meta"], dtype=np.int64)
    expected = (len(FOV_NODE_NAMES), FOV_GRAPH_INPUT_DIM)
    if X.ndim != 3 or X.shape[1:] != expected:
        raise ValueError("FOV-risk dataset X shape mismatch")
    if (y.shape != (X.shape[0],) or valid.shape != (X.shape[0],)
            or meta.shape != (X.shape[0], 3)):
        raise ValueError("FOV-risk targets/mask/metadata shape mismatch")
    if not np.isfinite(X).all():
        raise ValueError("FOV-risk dataset contains non-finite graphs")
    if not valid.any():
        raise ValueError("FOV-risk dataset has no sample with an observed future")
    if not np.isfinite(y[valid]).all():
        raise ValueError("FOV-risk dataset has a non-finite supervised target")
    if np.any((y[valid] < 0.0) | (y[valid] > 1.0)):
        raise ValueError("FOV-risk targets must be time fractions in [0,1]")
    if np.isfinite(y[~valid]).any():
        raise ValueError(
            "masked FOV-risk samples must stay NaN; an unobserved future is "
            "not evidence that the pad remained visible")
    if int(dataset.get("prediction_steps", 0)) < 1:
        raise ValueError("FOV-risk dataset prediction_steps must be positive")
    return X, y, valid, meta


def split_by_episode(dataset: Mapping, *, validation_fraction: float = 0.2,
                     seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Episode-disjoint masks over supervised (``valid``) samples only."""
    _, _, valid, meta = validate_fov_risk_dataset(dataset)
    episodes = np.unique(meta[valid, 0])
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
    validation &= valid
    training = (~validation) & valid
    if set(meta[training, 0]) & set(meta[validation, 0]):
        raise AssertionError("episode leakage across FOV-risk train/validation split")
    return training, validation


def dataset_digest(dataset: Mapping) -> str:
    X, y, valid, meta = validate_fov_risk_dataset(dataset)
    digest = hashlib.sha256()
    for name, value in (("X", X), ("y", np.nan_to_num(y, nan=-1.0)),
                        ("valid", valid), ("meta", meta)):
        array = np.ascontiguousarray(value)
        digest.update(name.encode("ascii"))
        digest.update(array.tobytes())
    digest.update(str(int(dataset["prediction_steps"])).encode("ascii"))
    return digest.hexdigest()


def save_fov_risk_dataset(dataset: Mapping, path: str | Path, *,
                          config_hash: str, seed: int,
                          horizon_seconds: float, control_hz: float) -> dict:
    X, y, valid, meta = validate_fov_risk_dataset(dataset)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, X=X, y=y, valid=valid, meta=meta,
                            prediction_steps=int(dataset["prediction_steps"]))
    os.replace(temporary, path)
    supervised = y[valid]
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
        "supervised_samples": int(np.count_nonzero(valid)),
        "masked_tail_samples": int(np.count_nonzero(~valid)),
        "target": "fraction of the next H steps with the pad centre outside the FOV",
        "target_mean": float(np.mean(supervised)),
        "target_nonzero_fraction": float(np.mean(supervised > 0.0)),
        "target_saturated_fraction": float(np.mean(supervised >= 1.0)),
        "dataset_sha256": dataset_digest(dataset),
        "feature_source": "ten visual/keypoint history features only",
        "target_source": (
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
        dataset = {name: payload[name] for name in ("X", "y", "valid", "meta")}
        dataset["prediction_steps"] = int(payload["prediction_steps"])
    validate_fov_risk_dataset(dataset)
    if dataset_digest(dataset) != manifest.get("dataset_sha256"):
        raise ValueError("FOV-risk dataset checksum mismatch")
    return dataset, manifest
