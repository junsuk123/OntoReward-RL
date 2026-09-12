"""Estimator-free visual semantics for the primary ontology pipeline.

Only keypoint-network outputs, UAV proprioception and onboard battery reserve
enter this module.  It neither accepts nor reconstructs metric platform pose or
velocity.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from ..mathx import quat_to_euler_zyx
from ..semantic import OntologyGraph


SEMANTIC_FEATURE_NAMES = (
    "keypoint_confidence",
    "image_alignment",
    "apparent_target_scale",
    "image_plane_motion_safety",
    "scale_rate_safety",
    "vertical_motion_safety",
    "attitude_stability",
    "battery_risk",
)

SEMANTIC_NODE_NAMES = (
    "KeypointConfidence", "ImageAlignment", "ApparentScale",
    "ImagePlaneMotion", "ScaleRate", "VerticalMotionSafety",
    "AttitudeStability", "BatteryRisk", "PerceptionQuality",
    "ApproachState", "ApproachStability", "DescentSafety", "SafeLanding",
)
SEMANTIC_RELATION_NAMES = ("indicates", "supports", "constrains", "self")
SEMANTIC_GRAPH_VERSION = "ontology_rgat.semantic_graph/1"
SEMANTIC_GRAPH_INPUT_DIM = 6 + len(SEMANTIC_NODE_NAMES)

# Substring matching intentionally rejects aliases such as
# ``simulator_relative_position`` rather than merely a short exact list.
FORBIDDEN_SEMANTIC_FIELDS = frozenset({
    "estimate", "estimated_relative_state", "true_relative_state",
    "relative_position", "relative_velocity", "platform_position",
    "platform_velocity", "simulator_truth", "ground_truth", "critic",
    "pad_position", "pad_velocity", "deck_position", "deck_velocity",
    "gnss_platform", "privileged",
})


def _normalise_name(value: object) -> str:
    return str(value).strip().lower().replace("-", "_").replace("/", "_")


def assert_semantic_payload_safe(payload: Mapping[str, Any]) -> None:
    """Reject privileged provenance recursively before graph construction."""
    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                name = _normalise_name(key)
                if any(token in name for token in FORBIDDEN_SEMANTIC_FIELDS):
                    raise ValueError(f"forbidden semantic field at {path}{key}: {name}")
                visit(child, f"{path}{key}.")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{path}{index}.")
    visit(payload, "semantic.")


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    output = np.asarray(value, dtype=np.float64)
    if output.shape != shape or not np.isfinite(output).all():
        raise ValueError(f"{name} must have finite shape {shape}")
    return output


@dataclass(frozen=True)
class SemanticObservation:
    """Eight bounded, physically interpretable observation-level quantities."""

    keypoint_confidence: float
    image_alignment: float
    apparent_target_scale: float
    image_plane_motion_safety: float
    scale_rate_safety: float
    vertical_motion_safety: float
    attitude_stability: float
    battery_risk: float
    centroid_xy: tuple[float, float]
    raw_scale: float

    def __post_init__(self) -> None:
        for name in SEMANTIC_FEATURE_NAMES:
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"semantic feature {name} must be finite and in [0, 1]")
            object.__setattr__(self, name, value)
        centroid = _finite_array(self.centroid_xy, (2,), "centroid_xy")
        if np.any(np.abs(centroid) > 1.0 + 1e-6):
            raise ValueError("centroid_xy must use normalized image coordinates")
        object.__setattr__(self, "centroid_xy", tuple(float(v) for v in centroid))
        if not math.isfinite(float(self.raw_scale)) or float(self.raw_scale) < 0.0:
            raise ValueError("raw_scale must be finite and non-negative")
        object.__setattr__(self, "raw_scale", float(self.raw_scale))

    @property
    def feature_vector(self) -> np.ndarray:
        return np.asarray([getattr(self, name) for name in SEMANTIC_FEATURE_NAMES],
                          dtype=np.float32)


def semantic_observation(keypoints, heatmaps, proprioception, *,
                         battery_reserve: float = 1.0,
                         previous: SemanticObservation | None = None,
                         dt: float = 0.1) -> SemanticObservation:
    """Calculate bounded semantics without estimating metric relative state.

    Normalization is explicit: normalized keypoints live in ``[-1, 1]``;
    apparent RMS scale is saturated at 0.75 image units; centroid motion is
    saturated at 4 normalized image units/s; scale rate at 2 units/s; vertical
    speed and tilt decay exponentially with 0.6 m/s and 22 degree scales.
    """
    points = _finite_array(keypoints, (6, 2), "keypoints")
    logits = np.asarray(heatmaps, dtype=np.float64)
    if logits.ndim != 3 or logits.shape[0] != 6 or not np.isfinite(logits).all():
        raise ValueError("heatmaps must have finite shape 6xHxW")
    if logits.shape[1] < 2 or logits.shape[2] < 2:
        raise ValueError("heatmaps require at least two pixels per axis")
    proprio = _finite_array(proprioception, (7,), "proprioception")
    dt = float(dt)
    reserve = float(battery_reserve)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("semantic observation dt must be positive")
    if not math.isfinite(reserve):
        raise ValueError("battery reserve must be finite")

    flat = logits.reshape(6, -1)
    flat = flat - flat.max(axis=1, keepdims=True)
    probability = np.exp(flat)
    probability /= probability.sum(axis=1, keepdims=True)
    entropy = -(probability * np.log(probability + 1e-12)).sum(axis=1)
    confidence = float(np.clip(
        1.0 - np.mean(entropy) / math.log(probability.shape[1]), 0.0, 1.0))

    centroid = points.mean(axis=0)
    alignment = 1.0 - float(np.clip(np.linalg.norm(centroid) / math.sqrt(2.0),
                                    0.0, 1.0))
    scale = float(np.sqrt(np.mean(np.sum((points - centroid) ** 2, axis=1))))
    apparent_scale = float(np.clip(scale / 0.75, 0.0, 1.0))
    if previous is None:
        motion_safety = 1.0
        scale_rate_safety = 1.0
    else:
        motion_rate = np.linalg.norm(
            centroid - np.asarray(previous.centroid_xy, dtype=float)) / dt
        scale_rate = abs(scale - previous.raw_scale) / dt
        motion_safety = 1.0 - float(np.clip(motion_rate / 4.0, 0.0, 1.0))
        scale_rate_safety = 1.0 - float(np.clip(scale_rate / 2.0, 0.0, 1.0))
    vertical_safety = float(np.exp(-abs(float(proprio[2])) / 0.6))
    quaternion = proprio[3:]
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-8:
        raise ValueError("proprioception quaternion must have non-zero norm")
    roll, pitch, _ = quat_to_euler_zyx(quaternion / norm)
    attitude = float(np.exp(-np.linalg.norm([roll, pitch]) / math.radians(22.0)))
    return SemanticObservation(
        keypoint_confidence=confidence,
        image_alignment=alignment,
        apparent_target_scale=apparent_scale,
        image_plane_motion_safety=motion_safety,
        scale_rate_safety=scale_rate_safety,
        vertical_motion_safety=vertical_safety,
        attitude_stability=attitude,
        battery_risk=1.0 - float(np.clip(reserve, 0.0, 1.0)),
        centroid_xy=tuple(centroid), raw_scale=scale)


def semantic_observation_from_payload(payload: Mapping[str, Any], *,
                                      previous: SemanticObservation | None = None,
                                      dt: float = 0.1) -> SemanticObservation:
    """Checked mapping boundary used by dataset/online graph callers."""
    assert_semantic_payload_safe(payload)
    allowed = {"keypoints", "heatmaps", "proprioception", "battery_reserve"}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unknown semantic fields: {sorted(unknown)}")
    required = {"keypoints", "heatmaps", "proprioception"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"missing semantic fields: {sorted(missing)}")
    return semantic_observation(
        payload["keypoints"], payload["heatmaps"], payload["proprioception"],
        battery_reserve=payload.get("battery_reserve", 1.0),
        previous=previous, dt=dt)


def semantic_graph(observation: SemanticObservation) -> OntologyGraph:
    """Build the fixed interpretable ontology used by the direct R-GAT."""
    if not isinstance(observation, SemanticObservation):
        raise TypeError("semantic graph requires a SemanticObservation")
    observed = observation.feature_vector.astype(np.float64)
    values = np.r_[
        observed,
        np.mean(observed[[0, 1]]),       # PerceptionQuality
        observed[2],                    # ApproachState
        np.mean(observed[[3, 4]]),       # ApproachStability
        np.mean(observed[[5, 6]]),       # DescentSafety
        0.0,                            # SafeLanding is learned, not hand-filled
    ]
    n_nodes = len(SEMANTIC_NODE_NAMES)
    features = np.zeros((SEMANTIC_GRAPH_INPUT_DIM, n_nodes), dtype=np.float32)
    features[0] = values
    features[1] = 1.0 - values
    features[2, :8] = 1.0               # direct observation nodes
    features[3, 8:12] = 1.0             # semantic intermediate nodes
    features[4, 7] = 1.0                # BatteryRisk is adverse
    features[5, 12] = 1.0               # goal node
    features[6:] = np.eye(n_nodes, dtype=np.float32)

    names = {name: index for index, name in enumerate(SEMANTIC_NODE_NAMES)}
    edge_specs = [
        ("KeypointConfidence", "PerceptionQuality", "indicates"),
        ("ImageAlignment", "PerceptionQuality", "indicates"),
        ("ApparentScale", "ApproachState", "indicates"),
        ("ImagePlaneMotion", "ApproachStability", "indicates"),
        ("ScaleRate", "ApproachStability", "indicates"),
        ("PerceptionQuality", "SafeLanding", "supports"),
        ("ApproachState", "ApproachStability", "supports"),
        ("ApproachStability", "SafeLanding", "supports"),
        ("VerticalMotionSafety", "DescentSafety", "supports"),
        ("AttitudeStability", "DescentSafety", "supports"),
        ("DescentSafety", "SafeLanding", "supports"),
        ("BatteryRisk", "SafeLanding", "constrains"),
    ]
    relation = {name: index for index, name in enumerate(SEMANTIC_RELATION_NAMES)}
    src = [names[source] for source, _, _ in edge_specs] + list(range(n_nodes))
    dst = [names[target] for _, target, _ in edge_specs] + list(range(n_nodes))
    rel = [relation[kind] for _, _, kind in edge_specs] + [relation["self"]] * n_nodes
    return OntologyGraph(
        X=features, src=np.asarray(src, dtype=np.int64),
        dst=np.asarray(dst, dtype=np.int64), rel=np.asarray(rel, dtype=np.int64),
        goal_node=names["SafeLanding"], node_names=SEMANTIC_NODE_NAMES,
        relation_names=SEMANTIC_RELATION_NAMES)
