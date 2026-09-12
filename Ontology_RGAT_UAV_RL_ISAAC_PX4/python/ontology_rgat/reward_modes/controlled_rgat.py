"""Empirical R-GAT reward design for the controlled Shin benchmark."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..config import default_config
from ..rgat.dataset import load_dataset, merge_datasets, save_dataset
from ..rgat.train import train_potential
from ..semantic import OntologyGraph
from .controlled_potential import (EMPIRICAL_PROVENANCE, FEATURES,
                                   NORMALIZATION, controlled_costs)


NODE_NAMES = tuple(FEATURES) + ("SafeLanding",)
RELATION_NAMES = ("contributes", "self")
DATASET_FORMAT = "ontology_rgat.controlled_rollouts/2"
ARTIFACT_FORMAT = "ontology_rgat.controlled_reward/3"


def controlled_graph(values=None):
    values = np.zeros(6) if values is None else np.asarray(values, dtype=float)
    if values.shape != (6,):
        raise ValueError("controlled ontology needs six node values")
    src = np.asarray([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5], dtype=np.int64)
    dst = np.asarray([5, 5, 5, 5, 5, 0, 1, 2, 3, 4, 5], dtype=np.int64)
    rel = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.int64)
    features = np.zeros((10, 6), dtype=float)
    features[0] = values
    features[1] = 1.0 - values
    features[2, :5] = 1.0
    features[3] = 1.0
    features[4:] = np.eye(6)
    return OntologyGraph(
        X=features, src=src, dst=dst, rel=rel, goal_node=5,
        node_names=NODE_NAMES, relation_names=RELATION_NAMES)


def episode_rollout_dataset(rows: Sequence[dict[str, Any]], metric: dict[str, Any],
                            *, episode: int, seed: int, gamma: float = 0.99,
                            sample_stride: int = 3) -> dict[str, Any]:
    """Convert one real Isaac/PX4 rollout into discounted-outcome graphs.

    Graph features come from the recurrent visual estimator and modeled
    onboard battery reserve. Simulator truth determines the already-reported
    safe-landing outcome but is never copied into ``X``.
    """
    if not rows:
        raise ValueError("cannot build an R-GAT dataset from an empty rollout")
    stride = max(1, int(sample_stride))
    gamma = float(gamma)
    if not 0.0 < gamma <= 1.0:
        raise ValueError("R-GAT outcome discount must be in (0, 1]")
    success = float(metric["paper_success"])
    if success not in (0.0, 1.0):
        raise ValueError("paper_success must be a binary safe-landing outcome")
    outcome = 2.0 * success - 1.0
    total = len(rows)
    indices = list(range(0, total, stride))
    X = np.stack([
        controlled_graph(np.r_[controlled_costs(
            rows[index]["estimate"], rows[index].get("battery_reserve", 1.0)),
            0.0]).X.T
        for index in indices
    ]).astype(np.float32)
    y = np.asarray([
        outcome * gamma ** (total - index - 1) for index in indices
    ], dtype=np.float32)
    # episode, control step, safe-landing success, environment seed
    meta = np.asarray([
        (int(episode), int(index), success, int(seed)) for index in indices
    ], dtype=np.float64)
    return {"X": X, "y": y, "meta": meta, "graph": controlled_graph()}


def merge_rollout_datasets(previous: dict[str, Any] | None,
                           current: dict[str, Any]) -> dict[str, Any]:
    """Append an empirical episode using the existing shape checks."""
    return merge_datasets(previous, current)


def _dataset_digest(dataset: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in ("X", "y", "meta"):
        value = np.ascontiguousarray(dataset[key])
        digest.update(key.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.dtype.str.encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _dataset_manifest_path(path: str | Path) -> Path:
    return Path(path).with_suffix(".manifest.json")


def _validated_rollout_arrays(dataset: dict[str, Any]):
    X = np.asarray(dataset["X"], dtype=np.float32)
    y = np.asarray(dataset["y"], dtype=np.float32).reshape(-1)
    meta = np.asarray(dataset["meta"], dtype=np.float64)
    if X.ndim != 3 or X.shape[1:] != (6, 10) or y.shape != (X.shape[0],):
        raise ValueError("controlled rollout dataset has an invalid feature/label shape")
    if meta.shape != (X.shape[0], 4):
        raise ValueError("controlled rollout metadata must be [samples,4]")
    if not np.isfinite(X).all() or not np.isfinite(y).all() or not np.isfinite(meta).all():
        raise ValueError("controlled rollout dataset contains non-finite values")
    if not np.isin(meta[:, 2], (0.0, 1.0)).all():
        raise ValueError("controlled rollout outcomes must be binary safe landings")
    if not np.equal(meta[:, (0, 1, 3)], np.floor(meta[:, (0, 1, 3)])).all():
        raise ValueError("controlled rollout episode, step and seed metadata must be integers")
    return X, y, meta


def _episode_summary(meta: np.ndarray) -> dict[int, tuple[int, int]]:
    summary: dict[int, tuple[int, int]] = {}
    for row in meta:
        episode = int(row[0])
        outcome_seed = (int(row[2]), int(row[3]))
        if episode in summary and summary[episode] != outcome_seed:
            raise ValueError("one R-GAT rollout has conflicting outcomes or seeds")
        summary[episode] = outcome_seed
    return summary


def save_rollout_dataset(dataset: dict[str, Any], path: str | Path, *,
                         config_hash: str, source_checkpoint_sha256: str,
                         completed_seeds: Sequence[int]) -> dict[str, Any]:
    """Atomically persist real rollout samples and auditable provenance."""
    X, y, meta = _validated_rollout_arrays(dataset)
    episode_summary = _episode_summary(meta)
    episode_outcomes = {key: value[0] for key, value in episode_summary.items()}
    episode_seeds = {key: value[1] for key, value in episode_summary.items()}
    completed = [int(seed) for seed in completed_seeds]
    if len(set(completed)) != len(completed):
        raise ValueError("completed R-GAT rollout seeds must be unique")
    if (len(completed) != len(episode_summary) or
            set(completed) != set(episode_seeds.values())):
        raise ValueError("completed seeds do not match R-GAT rollout metadata")
    path = Path(path)
    save_dataset({"X": X, "y": y, "meta": meta,
                  "graph": dataset["graph"]}, path)
    manifest = {
        "format": DATASET_FORMAT,
        "dataset_provenance": EMPIRICAL_PROVENANCE,
        "dataset_config_hash": str(config_hash),
        "source_policy": "trained_shin2026_recurrent_actor",
        "source_checkpoint_sha256": str(source_checkpoint_sha256),
        "feature_source": "recurrent visual estimate plus modeled onboard battery reserve",
        "label_source": "Isaac/PX4 physical pad-contact success versus terminal failure",
        "samples": int(X.shape[0]),
        "episodes": len(episode_outcomes),
        "successful_episodes": int(sum(episode_outcomes.values())),
        "completed_seeds": completed,
        "dataset_sha256": _dataset_digest({"X": X, "y": y, "meta": meta}),
    }
    manifest_path = _dataset_manifest_path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False),
                         encoding="utf-8")
    os.replace(temporary, manifest_path)
    return manifest


def load_rollout_dataset(path: str | Path, *, config_hash: str,
                         source_checkpoint_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resume only a dataset produced by this exact experiment and policy."""
    path = Path(path)
    manifest = json.loads(_dataset_manifest_path(path).read_text(encoding="utf-8"))
    if manifest.get("format") != DATASET_FORMAT:
        raise ValueError("unsupported controlled rollout dataset format")
    if manifest.get("dataset_provenance") != EMPIRICAL_PROVENANCE:
        raise ValueError("R-GAT dataset is not made from actual Isaac/PX4 rollouts")
    if manifest.get("dataset_config_hash") != str(config_hash):
        raise ValueError("R-GAT rollout dataset configuration mismatch")
    if manifest.get("source_checkpoint_sha256") != str(source_checkpoint_sha256):
        raise ValueError("R-GAT rollout dataset source-policy checkpoint mismatch")
    dataset = load_dataset(path)
    _, _, meta = _validated_rollout_arrays(dataset)
    if _dataset_digest(dataset) != manifest.get("dataset_sha256"):
        raise ValueError("R-GAT rollout dataset digest mismatch")
    episode_summary = _episode_summary(meta)
    episode_outcomes = {key: value[0] for key, value in episode_summary.items()}
    episode_seeds = {key: value[1] for key, value in episode_summary.items()}
    if (manifest.get("samples") != int(meta.shape[0]) or
            manifest.get("episodes") != len(episode_outcomes) or
            manifest.get("successful_episodes") != int(sum(episode_outcomes.values())) or
            set(manifest.get("completed_seeds", [])) != set(episode_seeds.values())):
        raise ValueError("R-GAT rollout manifest counts or seeds do not match its data")
    return dataset, manifest


def prepare_controlled_rgat_artifact(
        path: str | Path, dataset: dict[str, Any], *, config_hash: str,
        source_checkpoint_sha256: str, dataset_path: str | Path,
        mode="quick", seed=42, epochs=None):
    """Train, distill and atomically freeze an empirical reward artifact."""
    X, y, meta = _validated_rollout_arrays(dataset)
    recorded, dataset_manifest = load_rollout_dataset(
        dataset_path, config_hash=config_hash,
        source_checkpoint_sha256=source_checkpoint_sha256)
    if _dataset_digest(recorded) != _dataset_digest({"X": X, "y": y, "meta": meta}):
        raise ValueError("in-memory R-GAT data does not match the recorded rollout dataset")
    episode_summary = _episode_summary(meta)
    episode_outcomes = {key: value[0] for key, value in episode_summary.items()}
    if set(episode_outcomes.values()) != {0, 1}:
        raise ValueError(
            "empirical R-GAT data needs both successful and failed physical-contact "
            "episodes; collect more Shin behavior rollouts")

    cfg = default_config(mode, "sitl")
    cfg.seed = int(seed)
    cfg.ontology.node_names = list(NODE_NAMES)
    cfg.ontology.relation_names = list(RELATION_NAMES)
    cfg.ontology.n_nodes = 6
    cfg.ontology.n_relations = 2
    cfg.ontology.in_dim = 10
    cfg.rgat.epochs = int(epochs if epochs is not None else (10 if mode == "quick" else 80))
    cfg.rgat.batch_size = 64
    # The non-stabilized softmax overflowed in the supplied full-run log at
    # epoch 38. Subtracting the segment maximum is algebraically equivalent.
    cfg.rgat.stable_softmax = True
    cfg.rgat.lr = 5e-4
    model, history = train_potential(
        {"X": X, "y": y, "meta": meta, "graph": controlled_graph(),
         "split_by_episode": True},
        cfg, verbose=True)
    losses = np.r_[history["train_loss"], history["val_loss"]]
    if not np.isfinite(losses).all():
        raise FloatingPointError("R-GAT produced a non-finite training metric")

    baseline = model.predict_batch(X)
    importance = []
    signed = []
    for node in range(len(FEATURES)):
        counterfactual = X.copy()
        counterfactual[:, node, 0] = 0.0
        counterfactual[:, node, 1] = 1.0
        delta = model.predict_batch(counterfactual) - baseline
        importance.append(float(np.mean(np.abs(delta))))
        signed.append(float(np.mean(delta)))
    raw = np.asarray(importance, dtype=float)
    if not np.isfinite(raw).all() or float(raw.sum()) <= 1e-12:
        raise FloatingPointError("R-GAT counterfactual importance is non-finite or zero")
    weights = raw / raw.sum()
    dataset_sha256 = _dataset_digest({"X": X, "y": y, "meta": meta})
    payload = {
        "format": ARTIFACT_FORMAT,
        "profile": "controlled_landing", "frozen": True,
        "provenance": "rgat_distillation",
        "dataset_provenance": EMPIRICAL_PROVENANCE,
        "dataset_note": "actual recurrent Shin-policy flights in Isaac Sim/Pegasus/PX4",
        "dataset_config_hash": str(config_hash),
        "dataset_path": str(Path(dataset_path).resolve()),
        "dataset_sha256": dataset_sha256,
        "source_policy": "trained_shin2026_recurrent_actor",
        "source_checkpoint_sha256": str(source_checkpoint_sha256),
        "feature_source": "recurrent visual estimate plus modeled onboard battery reserve",
        "target": "discounted physical pad-contact outcome (+1 success, -1 failure)",
        "features": list(FEATURES),
        "normalization": NORMALIZATION,
        "weights": {name: float(value) for name, value in zip(FEATURES, weights)},
        "importance": {name: value for name, value in zip(FEATURES, importance)},
        "signed_effect": {name: value for name, value in zip(FEATURES, signed)},
        "dataset_samples": int(X.shape[0]),
        "dataset_episodes": len(episode_outcomes),
        "dataset_successful_episodes": int(sum(episode_outcomes.values())),
        "validation_split": "held-out rollout episodes",
        "recorded_dataset_manifest": dataset_manifest,
        "rgat_epochs": cfg.rgat.epochs,
        "rgat_learning_rate": float(cfg.rgat.lr),
        "rgat_stable_softmax": True,
        "rgat_final_validation_mse": float(history["val_loss"][-1]),
        "seed": int(seed),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           allow_nan=False)
    payload["design_id"] = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False),
                         encoding="utf-8")
    os.replace(temporary, path)
    return path, payload
