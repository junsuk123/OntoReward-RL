"""Strict visual-only ontology for future field-of-view unavailability.

The graph accepts exactly the ten observable quantities named in the ontology
contract (``config/ontology/fov_recovery.yaml``).  It deliberately has no
argument through which simulator truth, the six-dimensional relative-state
estimate, or critic state can enter.

Two properties are enforced structurally rather than by convention:

* every declared input node has a non-self path to the output node, so no
  declared feature can be silently ignored by the schema itself, and
* a stale keypoint measurement is never presented as a currently comfortable
  boundary margin.  ``fov_margin`` is the margin of the *last trustworthy*
  centroid, so it is published together with ``measurement_validity`` and
  ``measurement_age`` and the derived ``BoundarySafety`` seed is gated by
  validity (contract rule ``R-03``).

The relation names are structural edge types.  They are not enforced
monotonicity constraints, and nothing in this module guarantees that the
trained model respects the sign a name suggests.
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
    "measurement_validity",
    "measurement_age",
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
    "MeasurementValidity",
    "MeasurementAge",
    "PerceptionQuality",
    "BoundarySafety",
    "TargetMotion",
    "VisualObservability",
    "FutureFOVUnavailability",
)
FOV_RELATION_NAMES = ("indicates", "supports", "constrains", "self")
FOV_GRAPH_VERSION = "ontology_rgat.future_fov_unavailability/2"
FOV_GRAPH_INPUT_DIM = 4 + len(FOV_NODE_NAMES)
FOV_GOAL_NODE = "FutureFOVUnavailability"

# Declared, testable edges of the contract. ``self`` loops are appended by
# ``build_fov_graph`` for every node.
FOV_GRAPH_EDGES = (
    ("KeypointConfidence", "PerceptionQuality", "indicates"),
    ("VisibleKeypointFraction", "PerceptionQuality", "indicates"),
    ("MeasurementValidity", "PerceptionQuality", "indicates"),
    ("FOVMargin", "BoundarySafety", "indicates"),
    ("ApparentTargetScale", "BoundarySafety", "constrains"),
    ("MeasurementValidity", "BoundarySafety", "constrains"),
    ("MeasurementAge", "BoundarySafety", "constrains"),
    ("ImagePlaneMotion", "TargetMotion", "indicates"),
    ("ScaleRate", "TargetMotion", "indicates"),
    ("VisibilityMemory", "VisualObservability", "supports"),
    ("ReacquisitionTrend", "VisualObservability", "supports"),
    ("MeasurementAge", "VisualObservability", "constrains"),
    ("PerceptionQuality", "FutureFOVUnavailability", "supports"),
    ("BoundarySafety", "FutureFOVUnavailability", "supports"),
    ("VisualObservability", "FutureFOVUnavailability", "supports"),
    ("TargetMotion", "FutureFOVUnavailability", "constrains"),
)


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
    measurement_validity: float
    measurement_age: float

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
    """Distance-to-boundary margin for normalized image coordinates.

    The caller owns the question of whether the centroid is a *current*
    measurement; this function only converts coordinates to a margin.
    """
    centroid = np.asarray(centroid_xy, dtype=np.float64)
    if centroid.shape != (2,) or not np.isfinite(centroid).all():
        raise ValueError("centroid_xy must have finite shape (2,)")
    if np.any(np.abs(centroid) > 1.0 + 1e-6):
        raise ValueError("centroid_xy must be normalized to [-1,1]")
    return float(np.clip(1.0 - np.max(np.abs(centroid)), 0.0, 1.0))


def fov_observation_from_visual_semantics(observation) -> FOVSemanticObservation:
    """Project the keypoint history to the ten allowed FOV quantities.

    The legacy semantic extractor also computes proprioceptive diagnostics.
    This projection accesses none of them; only keypoint/image history fields
    cross into the proposed ontology graph.  ``visual_loss_duration_s`` is
    zero exactly when the current frame produced a usable pad geometry, and
    ``visual_loss_risk`` is its bounded age, so both missingness signals come
    from the same observed keypoint history and add no privileged input.
    """
    validity = 1.0 if float(observation.visual_loss_duration_s) <= 0.0 else 0.0
    return FOVSemanticObservation(
        keypoint_confidence=observation.keypoint_confidence,
        visible_keypoint_fraction=observation.visible_keypoint_fraction,
        fov_margin=fov_margin(observation.centroid_xy),
        apparent_target_scale=observation.apparent_target_scale,
        image_plane_motion=1.0 - observation.image_plane_motion_safety,
        scale_rate=1.0 - observation.scale_rate_safety,
        visibility_memory=observation.visibility_memory,
        reacquisition_trend=observation.reacquisition_trend,
        measurement_validity=validity,
        measurement_age=observation.visual_loss_risk,
    )


def derived_node_values(direct: np.ndarray) -> np.ndarray:
    """Seed values of the four derived nodes, in ``FOV_NODE_NAMES`` order.

    ``BoundarySafety`` multiplies the margin by the measurement validity so a
    stale centroid cannot enter the graph as a currently comfortable margin
    (contract rule ``R-03``).  The remaining seeds are plain means; the model
    is free to reweight them.
    """
    index = {name: position for position, name in enumerate(FOV_FEATURE_NAMES)}
    perception = float(np.mean(direct[[index["keypoint_confidence"],
                                       index["visible_keypoint_fraction"],
                                       index["measurement_validity"]]]))
    boundary = float(direct[index["fov_margin"]]
                     * direct[index["measurement_validity"]])
    motion = float(np.mean(direct[[index["image_plane_motion"],
                                   index["scale_rate"]]]))
    observability = float(np.mean(direct[[index["visibility_memory"],
                                          index["reacquisition_trend"]]])
                          * (1.0 - direct[index["measurement_age"]]))
    return np.asarray([perception, boundary, motion, observability],
                      dtype=np.float64)


def build_fov_graph(observation: FOVSemanticObservation) -> OntologyGraph:
    """Build the deterministic 15-node ontology required by the method."""
    if not isinstance(observation, FOVSemanticObservation):
        raise TypeError("FOV graph requires FOVSemanticObservation")
    direct = observation.feature_vector.astype(np.float64)
    values = np.r_[
        direct,
        derived_node_values(direct),
        0.0,                           # learned goal readout
    ]
    node_count = len(FOV_NODE_NAMES)
    X = np.zeros((FOV_GRAPH_INPUT_DIM, node_count), dtype=np.float32)
    X[0] = values
    X[1] = 1.0 - values
    X[2, :len(FOV_FEATURE_NAMES)] = 1.0
    X[3, len(FOV_FEATURE_NAMES):] = 1.0
    X[4:] = np.eye(node_count, dtype=np.float32)

    nodes = {name: index for index, name in enumerate(FOV_NODE_NAMES)}
    relations = {name: index for index, name in enumerate(FOV_RELATION_NAMES)}
    src = [nodes[source] for source, _, _ in FOV_GRAPH_EDGES] + list(range(node_count))
    dst = [nodes[target] for _, target, _ in FOV_GRAPH_EDGES] + list(range(node_count))
    rel = [relations[kind] for _, _, kind in FOV_GRAPH_EDGES]
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


def unreachable_input_nodes() -> tuple[str, ...]:
    """Declared input nodes with no non-self path to ``FOV_GOAL_NODE``.

    An empty tuple is the contract requirement checked by the acceptance
    tests; a non-empty one names features that the schema silently ignores.
    """
    successors: dict[str, list[str]] = {name: [] for name in FOV_NODE_NAMES}
    for source, target, _ in FOV_GRAPH_EDGES:
        successors[source].append(target)
    reaching: set[str] = set()
    for name in FOV_NODE_NAMES:
        seen: set[str] = set()
        stack = list(successors[name])
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(successors[current])
        if FOV_GOAL_NODE in seen:
            reaching.add(name)
    return tuple(name for name in FOV_NODE_NAMES[:len(FOV_FEATURE_NAMES)]
                 if name not in reaching)


def empty_fov_graph() -> OntologyGraph:
    return build_fov_graph(FOVSemanticObservation(*([0.0] * len(FOV_FEATURE_NAMES))))
