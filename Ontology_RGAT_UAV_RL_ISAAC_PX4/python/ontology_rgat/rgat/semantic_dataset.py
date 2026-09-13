"""Estimator-free semantic rollout data and frozen direct R-GAT potential."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ..config import default_config
from ..perception.semantic_observation import (
    FORBIDDEN_SEMANTIC_FIELDS, SEMANTIC_FEATURE_NAMES,
    SEMANTIC_GRAPH_INPUT_DIM, SEMANTIC_GRAPH_VERSION, SEMANTIC_NODE_NAMES,
    SEMANTIC_RELATION_NAMES, SemanticObservation, semantic_graph)
from ..semantic import OntologyGraph
from .dataset import load_dataset, save_dataset
from .model import build_potential
from .train import train_potential


SEMANTIC_DATASET_FORMAT = "ontology_rgat.semantic_rollouts/2-recovery-aware"
SEMANTIC_MODEL_FORMAT = "ontology_rgat.semantic_potential/3-stratified-best"
SEMANTIC_DATASET_PROVENANCE = "isaac_px4_estimator_free_recovery_behavior_mixture"
SEMANTIC_SAMPLE_FIELDS = frozenset({"graph_X", "step_id"})
MONOTONIC_COUNTERFACTUAL_NAMES = (
    "perception_degraded", "recovery_degraded", "battery_degraded")


def _forbidden_name(name: object) -> bool:
    normalized = str(name).strip().lower().replace("-", "_").replace("/", "_")
    return any(token in normalized for token in FORBIDDEN_SEMANTIC_FIELDS)


def assert_no_privileged_semantic_fields(value: Any, path: str = "sample") -> None:
    """Recursively reject semantic dataset metadata with privileged provenance."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            if _forbidden_name(key):
                raise ValueError(f"forbidden semantic rollout field at {path}.{key}")
            assert_no_privileged_semantic_fields(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_no_privileged_semantic_fields(child, f"{path}.{index}")


def _empty_graph() -> OntologyGraph:
    return semantic_graph(SemanticObservation(
        keypoint_confidence=0.0, visible_keypoint_fraction=0.0,
        image_alignment=0.0, apparent_target_scale=0.0,
        image_plane_motion_safety=0.0, scale_rate_safety=0.0,
        visibility_memory=0.0, reacquisition_trend=0.0,
        vertical_motion_safety=0.0, attitude_stability=0.0,
        battery_risk=0.0, visual_loss_risk=1.0,
        centroid_xy=(0.0, 0.0), raw_scale=0.0,
        visual_loss_duration_s=0.0))


def semantic_monotonic_counterfactuals(X) -> np.ndarray:
    """Build adverse semantic interventions for monotonic potential training.

    Inputs and outputs use the dataset orientation ``[sample,node,feature]``.
    Each intervention removes only estimator-free evidence: current perception,
    recent visibility/reacquisition, or onboard energy margin.  It never adds
    simulator truth or a metric target state.
    """
    source = np.asarray(X, dtype=np.float32)
    if source.ndim != 3 or source.shape[1:] != (
            len(SEMANTIC_NODE_NAMES), SEMANTIC_GRAPH_INPUT_DIM):
        raise ValueError("semantic monotonicity audit requires [sample,node,feature]")
    names = {name: index for index, name in enumerate(SEMANTIC_NODE_NAMES)}

    def set_value(value, node, level):
        index = names[node]
        value[:, index, 0] = float(level)
        value[:, index, 1] = 1.0 - float(level)

    perception = source.copy()
    for node in ("KeypointConfidence", "VisibleKeypointFraction",
                 "ImageAlignment", "PerceptionQuality"):
        set_value(perception, node, 0.0)
    recovery = source.copy()
    for node in ("VisibilityMemory", "ReacquisitionTrend", "RecoveryState"):
        set_value(recovery, node, 0.0)
    set_value(recovery, "VisualLossRisk", 1.0)
    battery = source.copy()
    set_value(battery, "BatteryRisk", 1.0)
    return np.stack((perception, recovery, battery), axis=1)


def semantic_episode_dataset(samples: Sequence[Mapping[str, Any]], *,
                             success: bool, episode_id: int, seed: int,
                             gamma_design: float = 0.99,
                             sample_stride: int = 3) -> dict[str, Any]:
    """Label one estimator-free semantic trajectory by discounted outcome."""
    if not samples:
        raise ValueError("cannot build semantic R-GAT data from an empty trajectory")
    gamma_design = float(gamma_design)
    if not 0.0 < gamma_design <= 1.0:
        raise ValueError("R-GAT outcome discount must be in (0, 1]")
    stride = max(1, int(sample_stride))
    for sample in samples:
        assert_no_privileged_semantic_fields(sample)
        if set(sample) != SEMANTIC_SAMPLE_FIELDS:
            raise ValueError(
                f"semantic rollout samples must contain exactly {sorted(SEMANTIC_SAMPLE_FIELDS)}")
    indices = list(range(0, len(samples), stride))
    graph = _empty_graph()
    matrices = []
    steps = []
    for index in indices:
        matrix = np.asarray(samples[index]["graph_X"], dtype=np.float32)
        if matrix.shape != (SEMANTIC_GRAPH_INPUT_DIM, len(SEMANTIC_NODE_NAMES)):
            raise ValueError("semantic rollout graph has an incompatible feature shape")
        if not np.isfinite(matrix).all():
            raise ValueError("semantic rollout graph contains non-finite values")
        matrices.append(matrix.T)
        steps.append(int(samples[index]["step_id"]))
    outcome = 1.0 if bool(success) else -1.0
    total = len(samples)
    y = np.asarray([
        outcome * gamma_design ** (total - index - 1) for index in indices
    ], dtype=np.float32)
    meta = np.asarray([
        (int(episode_id), step, float(bool(success)), int(seed))
        for step in steps
    ], dtype=np.float64)
    return {"X": np.stack(matrices), "y": y, "meta": meta,
            "graph": graph, "split_by_episode": True}


def merge_semantic_datasets(previous: dict[str, Any] | None,
                            current: dict[str, Any]) -> dict[str, Any]:
    if previous is None:
        return current
    old_x = np.asarray(previous["X"], dtype=np.float32)
    new_x = np.asarray(current["X"], dtype=np.float32)
    if old_x.ndim != 3 or old_x.shape[1:] != new_x.shape[1:]:
        raise ValueError("semantic rollout graph schemas do not match")
    return {
        "X": np.concatenate((old_x, new_x)),
        "y": np.concatenate((np.asarray(previous["y"], dtype=np.float32),
                              np.asarray(current["y"], dtype=np.float32))),
        "meta": np.concatenate((np.asarray(previous["meta"], dtype=np.float64),
                                 np.asarray(current["meta"], dtype=np.float64))),
        "graph": current["graph"], "split_by_episode": True,
    }


def _validate_dataset(dataset: dict[str, Any], *, require_both_classes=False):
    X = np.asarray(dataset["X"], dtype=np.float32)
    y = np.asarray(dataset["y"], dtype=np.float32).reshape(-1)
    meta = np.asarray(dataset["meta"], dtype=np.float64)
    expected = (len(SEMANTIC_NODE_NAMES), SEMANTIC_GRAPH_INPUT_DIM)
    if X.ndim != 3 or X.shape[1:] != expected or y.shape != (X.shape[0],):
        raise ValueError("semantic dataset has an invalid feature or label shape")
    if meta.shape != (X.shape[0], 4):
        raise ValueError("semantic dataset metadata must be [samples,4]")
    if not np.isfinite(X).all() or not np.isfinite(y).all() or not np.isfinite(meta).all():
        raise ValueError("semantic dataset contains non-finite values")
    outcomes = set(int(value) for value in np.unique(meta[:, 2]))
    if not outcomes.issubset({0, 1}):
        raise ValueError("semantic rollout outcomes must be binary")
    if require_both_classes and outcomes != {0, 1}:
        raise ValueError("semantic R-GAT data requires successful and failed episodes")
    if not np.equal(meta[:, (0, 1, 3)], np.floor(meta[:, (0, 1, 3)])).all():
        raise ValueError("semantic episode, step and seed metadata must be integers")
    return X, y, meta


def validate_semantic_dataset(dataset: dict[str, Any], *,
                              require_both_classes=False):
    """Public validation boundary used before any R-GAT optimizer is created."""
    return _validate_dataset(dataset, require_both_classes=require_both_classes)


def semantic_dataset_digest(dataset: dict[str, Any]) -> str:
    X, y, meta = _validate_dataset(dataset)
    digest = hashlib.sha256()
    for name, value in (("X", X), ("y", y), ("meta", meta)):
        array = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _manifest_path(path: str | Path) -> Path:
    return Path(path).with_suffix(".manifest.json")


def save_semantic_dataset(dataset: dict[str, Any], path: str | Path, *,
                          config_hash: str,
                          source_behavior_policy: Mapping[str, Any],
                          completed_seeds: Sequence[int],
                          environment_steps: int | None = None,
                          recovery_statistics: Mapping[str, Any] | None = None,
                          ) -> dict[str, Any]:
    X, y, meta = _validate_dataset(dataset)
    assert_no_privileged_semantic_fields(source_behavior_policy, "source_behavior_policy")
    episodes = np.unique(meta[:, 0]).astype(int)
    outcomes = {episode: int(meta[meta[:, 0] == episode, 2][0]) for episode in episodes}
    seeds = [int(value) for value in completed_seeds]
    if len(seeds) != len(set(seeds)) or set(seeds) != set(meta[:, 3].astype(int)):
        raise ValueError("completed semantic rollout seeds do not match dataset metadata")
    path = Path(path)
    save_dataset({"X": X, "y": y, "meta": meta,
                  "graph": dataset.get("graph", _empty_graph())}, path)
    manifest = {
        "format": SEMANTIC_DATASET_FORMAT,
        "graph_schema_version": SEMANTIC_GRAPH_VERSION,
        "dataset_provenance": SEMANTIC_DATASET_PROVENANCE,
        "dataset_config_hash": str(config_hash),
        "feature_names": list(SEMANTIC_FEATURE_NAMES),
        "node_names": list(SEMANTIC_NODE_NAMES),
        "relation_names": list(SEMANTIC_RELATION_NAMES),
        "normalization": {
            "keypoints": "normalized image coordinates [-1,1]",
            "apparent_scale": "RMS/0.75 clipped [0,1]",
            "image_motion": "1-centroid_speed/4 clipped [0,1]",
            "scale_rate": "1-abs(scale_rate)/2 clipped [0,1]",
            "vertical_motion": "exp(-abs(vz)/0.6)",
            "attitude": "exp(-tilt/radians(22))",
            "battery_risk": "1-clipped onboard reserve",
            "visible_keypoints": "fraction of heatmaps above entropy-confidence gate",
            "visibility_memory": "1.5 s exponentially decayed recent visibility",
            "reacquisition_trend": "positive confidence recovery, clipped [0,1]",
            "visual_loss_risk": "consecutive loss duration / 2 s, clipped [0,1]",
        },
        "feature_source": "keypoints, heatmaps, UAV proprioception, onboard battery reserve",
        "label_source": "Isaac/PX4 terminal physical landing outcome only",
        "target": "gamma_design^(T-t-1) * (+1 success or -1 failure)",
        "source_behavior_policy": dict(source_behavior_policy),
        "samples": int(X.shape[0]), "episodes": int(len(episodes)),
        "successful_episodes": int(sum(outcomes.values())),
        "environment_steps": (None if environment_steps is None
                              else int(environment_steps)),
        "completed_seeds": seeds,
        "dataset_sha256": semantic_dataset_digest(dataset),
        "forbidden_graph_inputs": sorted(FORBIDDEN_SEMANTIC_FIELDS),
        "recovery_statistics": dict(recovery_statistics or {}),
    }
    target = _manifest_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False),
                         encoding="utf-8")
    os.replace(temporary, target)
    return manifest


def load_semantic_dataset(path: str | Path, *, config_hash: str | None = None):
    path = Path(path)
    manifest = json.loads(_manifest_path(path).read_text(encoding="utf-8"))
    if manifest.get("format") != SEMANTIC_DATASET_FORMAT:
        raise ValueError("unsupported semantic rollout dataset format")
    if manifest.get("graph_schema_version") != SEMANTIC_GRAPH_VERSION:
        raise ValueError("semantic rollout graph schema mismatch")
    if config_hash is not None and manifest.get("dataset_config_hash") != str(config_hash):
        raise ValueError("semantic rollout configuration mismatch")
    assert_no_privileged_semantic_fields(
        manifest.get("source_behavior_policy", {}), "source_behavior_policy")
    dataset = load_dataset(path)
    _validate_dataset(dataset)
    if semantic_dataset_digest(dataset) != manifest.get("dataset_sha256"):
        raise ValueError("semantic rollout dataset digest mismatch")
    return dataset, manifest


def semantic_rgat_config(mode: str, seed: int, settings=None):
    settings = dict(settings or {})
    cfg = default_config(mode, "sitl")
    cfg.seed = int(seed)
    cfg.ontology.node_names = list(SEMANTIC_NODE_NAMES)
    cfg.ontology.relation_names = list(SEMANTIC_RELATION_NAMES)
    cfg.ontology.n_nodes = len(SEMANTIC_NODE_NAMES)
    cfg.ontology.n_relations = len(SEMANTIC_RELATION_NAMES)
    cfg.ontology.in_dim = SEMANTIC_GRAPH_INPUT_DIM
    cfg.rgat.hidden_dim = int(settings.get("hidden_dim", 24))
    cfg.rgat.rel_dim = int(settings.get("relation_dim", 6))
    cfg.rgat.epochs = int(settings.get("epochs", 10 if mode == "quick" else 80))
    cfg.rgat.batch_size = int(settings.get("batch_size", 64))
    cfg.rgat.lr = float(settings.get("learning_rate", 5e-4))
    cfg.rgat.val_fraction = float(settings.get("validation_fraction", 0.2))
    cfg.rgat.stable_softmax = True
    cfg.rgat.monotonic_weight = float(settings.get("monotonic_weight", 0.25))
    cfg.rgat.monotonic_margin = float(settings.get("monotonic_margin", 0.01))
    cfg.rgat.minimum_monotonic_compliance = float(
        settings.get("minimum_monotonic_compliance", 0.0))
    cfg.device.rgat = str(settings.get("device", "auto"))
    cfg.device.compile = False
    return cfg


def _model_digest(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def prepare_semantic_rgat_artifact(path: str | Path, dataset: dict[str, Any], *,
                                   dataset_path: str | Path, config_hash: str,
                                   mode="quick", seed=42, settings=None):
    """Train and freeze the direct R-GAT output used as primary potential."""
    X, y, meta = _validate_dataset(dataset, require_both_classes=True)
    recorded, dataset_manifest = load_semantic_dataset(
        dataset_path, config_hash=config_hash)
    if semantic_dataset_digest(recorded) != semantic_dataset_digest(dataset):
        raise ValueError("in-memory semantic data differs from persisted dataset")
    cfg = semantic_rgat_config(mode, seed, settings)
    monotonic_X = semantic_monotonic_counterfactuals(X)
    model, history = train_potential(
        {"X": X, "y": y, "meta": meta, "graph": _empty_graph(),
         "split_by_episode": True, "monotonic_X": monotonic_X},
        cfg, verbose=True)
    losses = np.r_[history["train_loss"], history["val_loss"]]
    if not np.isfinite(losses).all():
        raise FloatingPointError("semantic R-GAT produced non-finite losses")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model_sha = _model_digest(model)
    prediction = np.asarray(model.predict_batch(X), dtype=np.float64)
    counterfactual_shape = monotonic_X.shape
    counterfactual_prediction = np.asarray(model.predict_batch(
        monotonic_X.reshape(-1, *monotonic_X.shape[2:])), dtype=np.float64
    ).reshape(counterfactual_shape[:2])
    monotonic_delta = prediction[:, None] - counterfactual_prediction
    changed = np.max(np.abs(monotonic_X - X[:, None]), axis=(-1, -2)) > 1e-7
    monotonic_compliance = float(np.mean(
        (monotonic_delta >= 0.0) | ~changed))
    monotonic_margin_compliance = float(np.mean(
        (monotonic_delta >= float(cfg.rgat.monotonic_margin)) | ~changed))
    minimum_compliance = float(cfg.rgat.minimum_monotonic_compliance)
    if monotonic_compliance < minimum_compliance:
        raise RuntimeError(
            f"semantic potential monotonic compliance {monotonic_compliance:.1%} "
            f"is below required {minimum_compliance:.1%}")
    metadata = {
        "format": SEMANTIC_MODEL_FORMAT, "frozen": True,
        "profile": "three_pipeline_onto_no_se",
        "graph_schema_version": SEMANTIC_GRAPH_VERSION,
        "dataset_config_hash": str(config_hash),
        "dataset_sha256": semantic_dataset_digest(dataset),
        "dataset_path": str(Path(dataset_path).resolve()),
        "dataset_provenance": SEMANTIC_DATASET_PROVENANCE,
        "dataset_manifest": dataset_manifest,
        "feature_names": list(SEMANTIC_FEATURE_NAMES),
        "node_names": list(SEMANTIC_NODE_NAMES),
        "relation_names": list(SEMANTIC_RELATION_NAMES),
        "normalization": dataset_manifest["normalization"],
        "source_behavior_policy": dataset_manifest["source_behavior_policy"],
        "recovery_statistics": dataset_manifest.get("recovery_statistics", {}),
        "target": dataset_manifest["target"],
        "loss": "mean((Phi_theta(G)-target)^2) + output_l2*mean(Phi^2)",
        "validation_split": "whole held-out episodes",
        "train_episode_ids": history.get("train_episode_ids", []),
        "validation_episode_ids": history.get("validation_episode_ids", []),
        "hyperparameters": {
            "hidden_dim": int(cfg.rgat.hidden_dim),
            "relation_dim": int(cfg.rgat.rel_dim),
            "epochs": int(cfg.rgat.epochs),
            "batch_size": int(cfg.rgat.batch_size),
            "learning_rate": float(cfg.rgat.lr),
            "validation_fraction": float(cfg.rgat.val_fraction),
            "stable_softmax": True,
            "monotonic_weight": float(cfg.rgat.monotonic_weight),
            "monotonic_margin": float(cfg.rgat.monotonic_margin),
            "minimum_monotonic_compliance": minimum_compliance,
        },
        "final_train_objective": float(history["train_loss"][-1]),
        "final_validation_mse": float(history["val_loss"][-1]),
        "seed": int(seed), "model_sha256": model_sha,
        "potential": "Phi(G)=frozen_R_GAT(G)",
        "monotonic_counterfactuals": list(MONOTONIC_COUNTERFACTUAL_NAMES),
        "monotonic_compliance": monotonic_compliance,
        "monotonic_margin_compliance": monotonic_margin_compliance,
        "minimum_monotonic_delta": float(np.min(monotonic_delta[changed])),
        "mean_monotonic_delta": float(np.mean(monotonic_delta[changed])),
    }
    canonical = json.dumps(metadata, sort_keys=True, separators=(",", ":"),
                           allow_nan=False)
    metadata["design_id"] = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": SEMANTIC_MODEL_FORMAT,
        "state_dict": {name: value.detach().cpu()
                       for name, value in model.state_dict().items()},
        "metadata": metadata,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    manifest_path = _manifest_path(path)
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    manifest_tmp.write_text(json.dumps(metadata, indent=2, allow_nan=False),
                            encoding="utf-8")
    os.replace(manifest_tmp, manifest_path)
    history_path = path.parent / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "epoch", "train_loss", "val_loss", "monotonic_compliance"))
        writer.writeheader()
        compliance = history.get("monotonic_compliance", [])
        for index, (train, val) in enumerate(
                zip(history["train_loss"], history["val_loss"]), start=1):
            writer.writerow({"epoch": index, "train_loss": train, "val_loss": val,
                             "monotonic_compliance": (
                                 compliance[index - 1] if index <= len(compliance) else "")})
    return path, metadata


class FrozenSemanticRGATPotential:
    """Immutable direct R-GAT inference wrapper for PPO reward shaping."""

    def __init__(self, path: str | Path, *, expected_config_hash: str | None = None):
        self.path = Path(path).resolve()
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata") or {}
        if payload.get("format") != SEMANTIC_MODEL_FORMAT:
            raise ValueError("semantic potential artifact format mismatch")
        if metadata.get("graph_schema_version") != SEMANTIC_GRAPH_VERSION:
            raise ValueError("semantic potential graph schema mismatch")
        if not bool(metadata.get("frozen", False)):
            raise ValueError("semantic R-GAT must be frozen before PPO")
        if expected_config_hash is not None and (
                metadata.get("dataset_config_hash") != str(expected_config_hash)):
            raise ValueError("semantic R-GAT configuration mismatch")
        cfg = semantic_rgat_config(
            "quick", int(metadata.get("seed", 42)), metadata.get("hyperparameters"))
        self.model = build_potential(cfg, _empty_graph(), device="cpu")
        self.model.load_state_dict(payload["state_dict"])
        if _model_digest(self.model) != metadata.get("model_sha256"):
            raise ValueError("semantic R-GAT model digest mismatch")
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.design_id = str(metadata["design_id"])
        self.dataset_sha256 = str(metadata["dataset_sha256"])
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.metadata = metadata

    def __call__(self, graph: OntologyGraph) -> float:
        if not isinstance(graph, OntologyGraph):
            raise TypeError("semantic potential accepts only an OntologyGraph")
        if tuple(graph.node_names) != SEMANTIC_NODE_NAMES:
            raise ValueError("semantic potential node schema mismatch")
        if tuple(graph.relation_names) != SEMANTIC_RELATION_NAMES:
            raise ValueError("semantic potential relation schema mismatch")
        if graph.X.shape != (SEMANTIC_GRAPH_INPUT_DIM, len(SEMANTIC_NODE_NAMES)):
            raise ValueError("semantic potential feature schema mismatch")
        return self.model.predict(graph)

    def explain(self, graph: OntologyGraph):
        return self.model.explain(graph)
