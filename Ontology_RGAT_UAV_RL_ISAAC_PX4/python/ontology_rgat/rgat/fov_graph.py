"""Strict visual-only ontology for future field-of-view loss prediction.

The graph accepts exactly the eight observable quantities named in the paper
contract.  It deliberately has no argument through which simulator truth, the
six-dimensional relative-state estimate, or critic state can enter.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..semantic import OntologyGraph


FOV_FEATURE_NAMES = (
    "keypoint_confidence",
    "visible_keypoint_fraction",
    "fov_margin",
    "apparent_target_scale",
    "image_plane_motion",
    "scale_rate",
    "visibility_memory",
    "reacquisition_trend",
)
FOV_NODE_NAMES = (
    "KeypointConfidence",
    "VisibleKeypointFraction",
    "FOVMargin",
    "ApparentTargetScale",
    "ImagePlaneMotion",
    "ScaleRate",
    "VisibilityMemory",
    "ReacquisitionTrend",
    "PerceptionQuality",
    "BoundarySafety",
    "TargetMotion",
    "VisualObservability",
    "FOVRetention",
)
FOV_RELATION_NAMES = ("indicates", "supports", "constrains", "self")
FOV_GRAPH_VERSION = "ontology_rgat.fov_retention/1"
FOV_GRAPH_INPUT_DIM = 4 + len(FOV_NODE_NAMES)
FOV_GOAL_NODE = "FOVRetention"


@dataclass(frozen=True)
class FOVSemanticObservation:
    """The complete and exclusive input boundary of the proposed branch."""

    keypoint_confidence: float
    visible_keypoint_fraction: float
    fov_margin: float
    apparent_target_scale: float
    image_plane_motion: float
    scale_rate: float
    visibility_memory: float
    reacquisition_trend: float

    def __post_init__(self) -> None:
        for name in FOV_FEATURE_NAMES:
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"FOV semantic feature {name} must be in [0,1]")
            object.__setattr__(self, name, value)

    @property
    def feature_vector(self) -> np.ndarray:
        return np.asarray(
            [getattr(self, name) for name in FOV_FEATURE_NAMES], dtype=np.float32)


def fov_margin(centroid_xy) -> float:
    """Distance-to-boundary margin for normalized image coordinates."""
    centroid = np.asarray(centroid_xy, dtype=np.float64)
    if centroid.shape != (2,) or not np.isfinite(centroid).all():
        raise ValueError("centroid_xy must have finite shape (2,)")
    if np.any(np.abs(centroid) > 1.0 + 1e-6):
        raise ValueError("centroid_xy must be normalized to [-1,1]")
    return float(np.clip(1.0 - np.max(np.abs(centroid)), 0.0, 1.0))


def fov_observation_from_visual_semantics(observation) -> FOVSemanticObservation:
    """Project the keypoint history to the eight allowed FOV quantities.

    The legacy semantic extractor also computes proprioceptive diagnostics.
    This projection accesses none of them; only keypoint/image history fields
    cross into the proposed ontology graph.
    """
    return FOVSemanticObservation(
        keypoint_confidence=observation.keypoint_confidence,
        visible_keypoint_fraction=observation.visible_keypoint_fraction,
        fov_margin=fov_margin(observation.centroid_xy),
        apparent_target_scale=observation.apparent_target_scale,
        image_plane_motion=1.0 - observation.image_plane_motion_safety,
        scale_rate=1.0 - observation.scale_rate_safety,
        visibility_memory=observation.visibility_memory,
        reacquisition_trend=observation.reacquisition_trend,
    )


def build_fov_graph(observation: FOVSemanticObservation) -> OntologyGraph:
    """Build the deterministic 13-node ontology required by the method."""
    if not isinstance(observation, FOVSemanticObservation):
        raise TypeError("FOV graph requires FOVSemanticObservation")
    direct = observation.feature_vector.astype(np.float64)
    values = np.r_[
        direct,
        np.mean(direct[[0, 1]]),       # PerceptionQuality
        direct[2],                     # BoundarySafety
        np.mean(direct[[4, 5]]),       # TargetMotion (magnitude/risk)
        np.mean(direct[[6, 7]]),       # VisualObservability
        0.0,                           # learned goal readout
    ]
    node_count = len(FOV_NODE_NAMES)
    X = np.zeros((FOV_GRAPH_INPUT_DIM, node_count), dtype=np.float32)
    X[0] = values
    X[1] = 1.0 - values
    X[2, :len(FOV_FEATURE_NAMES)] = 1.0
    X[3, len(FOV_FEATURE_NAMES):-1] = 1.0
    X[3, -1] = 1.0
    X[4:] = np.eye(node_count, dtype=np.float32)

    edges = (
        ("KeypointConfidence", "PerceptionQuality", "indicates"),
        ("VisibleKeypointFraction", "PerceptionQuality", "indicates"),
        ("FOVMargin", "BoundarySafety", "indicates"),
        ("ImagePlaneMotion", "TargetMotion", "indicates"),
        ("ScaleRate", "TargetMotion", "indicates"),
        ("VisibilityMemory", "VisualObservability", "supports"),
        ("ReacquisitionTrend", "VisualObservability", "supports"),
        ("PerceptionQuality", "FOVRetention", "supports"),
        ("BoundarySafety", "FOVRetention", "supports"),
        ("VisualObservability", "FOVRetention", "supports"),
        ("TargetMotion", "FOVRetention", "constrains"),
    )
    nodes = {name: index for index, name in enumerate(FOV_NODE_NAMES)}
    relations = {name: index for index, name in enumerate(FOV_RELATION_NAMES)}
    src = [nodes[source] for source, _, _ in edges] + list(range(node_count))
    dst = [nodes[target] for _, target, _ in edges] + list(range(node_count))
    rel = [relations[kind] for _, _, kind in edges]
    rel += [relations["self"]] * node_count
    return OntologyGraph(
        X=X,
        src=np.asarray(src, dtype=np.int64),
        dst=np.asarray(dst, dtype=np.int64),
        rel=np.asarray(rel, dtype=np.int64),
        goal_node=nodes[FOV_GOAL_NODE],
        node_names=FOV_NODE_NAMES,
        relation_names=FOV_RELATION_NAMES,
    )


def empty_fov_graph() -> OntologyGraph:
    return build_fov_graph(FOVSemanticObservation(*([0.0] * 8)))

