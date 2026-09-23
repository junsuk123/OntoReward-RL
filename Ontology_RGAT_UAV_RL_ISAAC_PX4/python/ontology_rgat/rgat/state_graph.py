"""The minimal ontology situation graph used as the policy's state, G_t.

This is the new methodology. The ontology no longer infers reward-term weights
and no longer contributes an additive reward term: it *is* the state
representation the proposed arm's actor and critic read. Both learned arms
share one reward, one action space, one environment and one PPO configuration,
so the single experimental factor is whether the policy sees the ontology graph
or not.

Scope: minimal core contract
----------------------------
The retired 15-node future-FOV schema and the 18-node semantic schema both
existed to feed a *readout* -- a scalar the reward consumed. A state
representation has a different failure mode: anything the graph cannot express
is information the policy loses. The schema here is therefore the smallest one
that still has something for a relational network to do, which the reduced 2-D
study measured as three properties (``docs/ONTOLOGY_RGAT_STATE.md``):

* a two-stage structure, risk node -> intermediate node -> goal node. Collapse
  it and there is no message to pass;
* four distinct relation types. Merge them and "relational" attention is just
  attention;
* every declared input node on a non-self path to the goal. Otherwise the
  schema silently drops a feature.

Nine nodes, four relations, twelve declared edges and nine self-loops is what
satisfies all three. It is not a reduction of the FOV schema -- that one is
kept, unchanged, under ``fov_graph.py`` for the legacy reward arm.

Information boundary
--------------------
The only constructor argument is a :class:`SemanticObservation`, whose fields
come from the frozen keypoint encoder and the vehicle's own proprioception.
There is no argument through which simulator truth, the six-dimensional
relative-state estimate, the geometric pad-centre FOV label or critic state
could enter -- the same structural guarantee ``fov_graph`` has, for the same
reason.

Two adaptations the reduced study found necessary
-------------------------------------------------
1. **Soft saturation.** The quantities span two orders of magnitude -- range
   from 10 m to 0.05 m -- and a clipped ``min(1, x/scale)`` cannot cover both
   ends with one constant. Every magnitude here uses ``x / (x + scale)``, so
   ``scale`` is the value at which the node reads 0.5 rather than the point at
   which it stops reading anything.

2. **Direction is a separate channel.** Ontology node values are magnitudes of
   risk, which is all a reward-weight design needs. A policy that cannot tell
   ahead from behind cannot control. Row 5 of the node-feature matrix carries
   a ``[-1, 1]`` sign for the quantity the node already represents; it adds no
   new physical quantity and changes no node, edge or relation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..semantic import OntologyGraph


__all__ = [
    "STATE_GRAPH_VERSION", "STATE_NODE_NAMES", "STATE_RELATION_NAMES",
    "STATE_GRAPH_EDGES", "STATE_GRAPH_INPUT_DIM", "STATE_GOAL_NODE",
    "STATE_RISK_NODES", "StateGraphGeometry", "StateGraphScales",
    "state_node_values", "state_node_signs", "build_state_graph",
    "empty_state_graph", "unreachable_state_input_nodes",
]


STATE_GRAPH_VERSION = "ontology_rgat.planar_situation_state/1"

STATE_NODE_NAMES = (
    # Risk nodes: magnitude of an adverse quantity, in [0, 1).
    "AlignmentError",
    "DescentRate",
    "TargetMotion",
    "FOVMargin",
    "MeasurementAge",
    "RelativeRange",
    # Support node.
    "PadVisibility",
    # Intermediate node -- the whole reason a second stage exists.
    "TouchdownSafety",
    # Goal node. Always zero; see ``state_node_values``.
    "SafeLanding",
)
STATE_RELATION_NAMES = ("degrades", "supports", "contributes", "self")
STATE_GOAL_NODE = "SafeLanding"
STATE_RISK_NODES = STATE_NODE_NAMES[:6]

# Declared edges. ``self`` loops are appended for every node by
# ``build_state_graph``. Two stages: risk -> {PadVisibility, TouchdownSafety}
# -> SafeLanding, with the two shortcuts the reduced study kept because the
# quantities they carry bear on the outcome directly.
STATE_GRAPH_EDGES = (
    ("AlignmentError", "TouchdownSafety", "degrades"),
    ("DescentRate", "TouchdownSafety", "degrades"),
    ("FOVMargin", "PadVisibility", "degrades"),
    ("FOVMargin", "TouchdownSafety", "degrades"),
    ("TargetMotion", "PadVisibility", "degrades"),
    ("MeasurementAge", "PadVisibility", "degrades"),
    ("MeasurementAge", "SafeLanding", "degrades"),
    ("RelativeRange", "TouchdownSafety", "degrades"),
    ("RelativeRange", "SafeLanding", "degrades"),
    ("PadVisibility", "TouchdownSafety", "supports"),
    ("PadVisibility", "SafeLanding", "contributes"),
    ("TouchdownSafety", "SafeLanding", "contributes"),
)

# Rows of the node-feature matrix, in order:
#   0  node value
#   1  1 - node value
#   2  risk-node indicator
#   3  bias
#   4  direction sign in [-1, 1]
#   5+ node identity one-hot
STATE_FEATURE_ROWS = 5
STATE_GRAPH_INPUT_DIM = STATE_FEATURE_ROWS + len(STATE_NODE_NAMES)


@dataclass(frozen=True)
class StateGraphGeometry:
    """Fixed camera constants. A profile property, not an observation."""

    nadir_column: float = -0.577
    tan_half_horizontal: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.nadir_column) or abs(self.nadir_column) > 1.0:
            raise ValueError("the nadir column must be a normalized image coordinate")
        if not math.isfinite(self.tan_half_horizontal) or self.tan_half_horizontal <= 0.0:
            raise ValueError("the horizontal half-angle tangent must be positive")


@dataclass(frozen=True)
class StateGraphScales:
    """Half-value points of the soft saturations, in the quantity's own units."""

    bearing: float = 0.35            # tangent units from nadir
    descent_speed_m_s: float = 0.45
    image_motion: float = 0.35       # normalized image units per second
    fov_margin: float = 0.35         # normalized image units of remaining margin
    range_m: float = 3.0
    # Apparent RMS keypoint scale at one metre of altitude; converts the only
    # available inverse-range signal into metres.
    reference_scale: float = 0.06
    range_bounds_m: tuple[float, float] = (0.30, 12.0)

    def __post_init__(self) -> None:
        for name in ("bearing", "descent_speed_m_s", "image_motion",
                     "fov_margin", "range_m", "reference_scale"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"state-graph scale {name} must be positive")
        low, high = (float(bound) for bound in self.range_bounds_m)
        if not 0.0 < low <= high:
            raise ValueError("state-graph range bounds must be positive and ordered")


def _soft(magnitude: float, scale: float) -> float:
    """``x / (x + scale)`` -- bounded in [0, 1), never a zero gradient."""
    value = abs(float(magnitude))
    return float(value / (value + float(scale)))


def _bearing_tangent(observation, geometry: StateGraphGeometry) -> float:
    centroid = np.asarray(observation.centroid_xy, dtype=float).reshape(-1)
    return float((centroid[0] - geometry.nadir_column)
                 * geometry.tan_half_horizontal)


def _altitude_estimate(observation, scales: StateGraphScales) -> float:
    scale = max(float(observation.raw_scale), 1e-3)
    low, high = scales.range_bounds_m
    return float(np.clip(scales.reference_scale / scale, low, high))


def state_node_values(observation, *, geometry: StateGraphGeometry | None = None,
                      scales: StateGraphScales | None = None,
                      previous=None, dt: float = 0.1) -> np.ndarray:
    """Node values in ``STATE_NODE_NAMES`` order, each in ``[0, 1]``.

    ``SafeLanding`` is always exactly zero. It is the node the two-stage
    structure terminates on, and writing anything derived from the outcome into
    it would leak the answer into the state.
    """
    geometry = geometry or StateGraphGeometry()
    scales = scales or StateGraphScales()
    step = float(dt)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("the state graph needs a positive control period")

    bearing = _bearing_tangent(observation, geometry)
    altitude = _altitude_estimate(observation, scales)
    distance = altitude * math.hypot(1.0, bearing)

    # ``image_plane_motion_safety`` is already the bounded safety form of the
    # centroid rate; invert it into the risk the node names.
    motion_risk = 1.0 - float(observation.image_plane_motion_safety)
    # ``vertical_motion_safety`` is exp(-|vz| / 0.6); recover |vz| so the soft
    # saturation is applied to the physical quantity rather than to an already
    # squashed one.
    vertical_safety = min(max(float(observation.vertical_motion_safety), 1e-6), 1.0)
    descent_speed = -0.6 * math.log(vertical_safety)

    centroid = np.asarray(observation.centroid_xy, dtype=float).reshape(-1)
    margin = float(np.clip(1.0 - float(np.max(np.abs(centroid))), 0.0, 1.0))

    visibility = float(np.clip(
        float(observation.keypoint_confidence)
        * float(observation.visible_keypoint_fraction), 0.0, 1.0))
    # A support node: high when the pad is being seen, and the memory term
    # keeps it from collapsing on a single dropped frame.
    pad_visibility = float(np.clip(
        max(visibility, float(observation.visibility_memory)
            * (1.0 - float(observation.visual_loss_risk))), 0.0, 1.0))

    alignment_risk = _soft(bearing, scales.bearing)
    descent_risk = _soft(descent_speed, scales.descent_speed_m_s)
    target_motion = float(np.clip(motion_risk, 0.0, 1.0))
    fov_risk = 1.0 - _soft(margin, scales.fov_margin)
    measurement_age = float(np.clip(observation.visual_loss_risk, 0.0, 1.0))
    range_risk = _soft(distance, scales.range_m)

    # The intermediate node is a product of the conditions a touchdown needs,
    # which is the form the original ontology used and the only place in this
    # schema where several risks are combined before the network sees them.
    touchdown_safety = float(np.clip(
        (1.0 - alignment_risk) * (1.0 - descent_risk)
        * (1.0 - range_risk) * pad_visibility, 0.0, 1.0))

    return np.asarray([
        alignment_risk, descent_risk, target_motion, fov_risk,
        measurement_age, range_risk, pad_visibility, touchdown_safety,
        0.0,
    ], dtype=np.float64)


def state_node_signs(observation, *, geometry: StateGraphGeometry | None = None,
                     previous=None, committed: bool = False) -> np.ndarray:
    """Direction channel, in ``STATE_NODE_NAMES`` order, each in ``[-1, 1]``.

    Every entry is the sign of the quantity the node already carries. The one
    that is not a sign in the arithmetic sense is ``PadVisibility``: it reports
    which branch of the approach the vehicle is in (+1 tracking, -1 searching,
    0 committed to the flare). The reduced study measured this as the single
    largest correction to the representation -- the guidance law it is compared
    against uses *opposite* vertical commands in the tracking and search
    branches, so a state without the branch cannot reproduce either. It is the
    same information the baseline actor already receives through the image and
    its own recurrence; it is not privileged.
    """
    geometry = geometry or StateGraphGeometry()
    bearing = _bearing_tangent(observation, geometry)
    direction = float(np.clip(bearing / max(abs(bearing), 1e-9), -1.0, 1.0)
                      ) if abs(bearing) > 1e-9 else 0.0

    vertical_safety = min(max(float(observation.vertical_motion_safety), 1e-6), 1.0)
    descent_speed = -0.6 * math.log(vertical_safety)
    # ``vertical_motion_safety`` is an even function of vz, so the sign has to
    # come from somewhere that still has one. The descent branch is the normal
    # case; a climb is only commanded while the pad is not visible, which the
    # measurement-age node already exposes.
    descent_sign = -1.0 if descent_speed > 1e-6 else 0.0
    if float(observation.visual_loss_risk) > 0.0:
        descent_sign = +1.0

    motion_sign = 0.0
    if previous is not None:
        previous_bearing = _bearing_tangent(previous, geometry)
        delta = bearing - previous_bearing
        if abs(delta) > 1e-9:
            motion_sign = math.copysign(1.0, delta)

    trustworthy = (float(observation.visible_keypoint_fraction) >= 0.5
                   and float(observation.visual_loss_risk) <= 0.0)
    if committed:
        visibility_sign = 0.0
    else:
        visibility_sign = 1.0 if trustworthy else -1.0

    return np.asarray([
        direction,            # AlignmentError: pad ahead (+) or behind (-)
        descent_sign,         # DescentRate: descending (-) or climbing (+)
        motion_sign,          # TargetMotion: drifting forward (+) or aft (-)
        direction,            # FOVMargin: which edge the centroid is nearing
        0.0,                  # MeasurementAge: elapsed time has no direction
        0.0,                  # RelativeRange: a distance has no direction
        visibility_sign,      # PadVisibility: tracking / searching / committed
        0.0,                  # TouchdownSafety: a product of magnitudes
        0.0,                  # SafeLanding: the goal node
    ], dtype=np.float64)


def build_state_graph(observation, *, geometry: StateGraphGeometry | None = None,
                      scales: StateGraphScales | None = None,
                      previous=None, committed: bool = False,
                      dt: float = 0.1) -> OntologyGraph:
    """Build ``G_t`` for one control step."""
    values = state_node_values(observation, geometry=geometry, scales=scales,
                               previous=previous, dt=dt)
    signs = state_node_signs(observation, geometry=geometry, previous=previous,
                             committed=committed)
    node_count = len(STATE_NODE_NAMES)
    X = np.zeros((STATE_GRAPH_INPUT_DIM, node_count), dtype=np.float32)
    X[0] = values
    X[1] = 1.0 - values
    X[2, :len(STATE_RISK_NODES)] = 1.0
    X[3] = 1.0
    X[4] = signs
    X[STATE_FEATURE_ROWS:] = np.eye(node_count, dtype=np.float32)

    nodes = {name: index for index, name in enumerate(STATE_NODE_NAMES)}
    relations = {name: index for index, name in enumerate(STATE_RELATION_NAMES)}
    src = [nodes[source] for source, _, _ in STATE_GRAPH_EDGES] + list(range(node_count))
    dst = [nodes[target] for _, target, _ in STATE_GRAPH_EDGES] + list(range(node_count))
    rel = [relations[kind] for _, _, kind in STATE_GRAPH_EDGES]
    rel += [relations["self"]] * node_count
    return OntologyGraph(
        X=X,
        src=np.asarray(src, dtype=np.int64),
        dst=np.asarray(dst, dtype=np.int64),
        rel=np.asarray(rel, dtype=np.int64),
        goal_node=nodes[STATE_GOAL_NODE],
        node_names=STATE_NODE_NAMES,
        relation_names=STATE_RELATION_NAMES,
    )


def empty_state_graph() -> OntologyGraph:
    """A zero-valued graph, used to build the fixed topology."""
    node_count = len(STATE_NODE_NAMES)
    X = np.zeros((STATE_GRAPH_INPUT_DIM, node_count), dtype=np.float32)
    X[1] = 1.0
    X[2, :len(STATE_RISK_NODES)] = 1.0
    X[3] = 1.0
    X[STATE_FEATURE_ROWS:] = np.eye(node_count, dtype=np.float32)
    nodes = {name: index for index, name in enumerate(STATE_NODE_NAMES)}
    relations = {name: index for index, name in enumerate(STATE_RELATION_NAMES)}
    src = [nodes[source] for source, _, _ in STATE_GRAPH_EDGES] + list(range(node_count))
    dst = [nodes[target] for _, target, _ in STATE_GRAPH_EDGES] + list(range(node_count))
    rel = [relations[kind] for _, _, kind in STATE_GRAPH_EDGES]
    rel += [relations["self"]] * node_count
    return OntologyGraph(
        X=X, src=np.asarray(src, dtype=np.int64),
        dst=np.asarray(dst, dtype=np.int64), rel=np.asarray(rel, dtype=np.int64),
        goal_node=nodes[STATE_GOAL_NODE], node_names=STATE_NODE_NAMES,
        relation_names=STATE_RELATION_NAMES)


def unreachable_state_input_nodes() -> tuple[str, ...]:
    """Declared input nodes with no non-self path to the goal node.

    An empty tuple is the contract requirement; a non-empty one names features
    the schema would silently ignore.
    """
    successors: dict[str, list[str]] = {name: [] for name in STATE_NODE_NAMES}
    for source, target, _ in STATE_GRAPH_EDGES:
        successors[source].append(target)
    unreachable = []
    for name in STATE_NODE_NAMES[:-1]:
        seen: set[str] = set()
        stack = list(successors[name])
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(successors[current])
        if STATE_GOAL_NODE not in seen:
            unreachable.append(name)
    return tuple(unreachable)
