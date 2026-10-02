"""A 3D snapshot of the ontology graph with the R-GAT's own attention on it.

The dashboard already answers "is the run going somewhere"; this answers the
question a reviewer asks next, which is *what the potential learned to look
at*. The schema is fixed, so the interesting quantity is not the topology but
the second-layer attention over it: which relation carries the signal into
SafeLanding at this point in training, and how that changes once the markers
drop out or the deck starts moving.

Two things are worth being explicit about.

*The layout is computed, not drawn.* Nodes are placed by longest-path depth
towards the goal and then spread around a ring inside their layer, so the
picture follows the edge list rather than a hand-tuned table that would drift
the moment the schema gains a node. RViz's overlay keeps its flat table --
there the graph hangs beside a real vehicle and has to stay readable from one
viewpoint -- so the two views are deliberately laid out by different rules.

*Attention is learned importance, not causal proof.* A thick edge means the
layer weighted it, nothing stronger; the same caveat the RViz overlay and the
paper carry.
"""
from __future__ import annotations

import math
import time
from typing import Any, Sequence

import numpy as np

from ..rgat.state_graph import STATE_RISK_NODES
from ..semantic import GOAL_NODE, N_NODES, RISK_NODES
from .live import STORE, LiveStore

__all__ = [
    "layer_of", "layout_3d", "graph_payload", "sensor_ontology_provenance",
    "GraphPublisher",
]


def layer_of(src: Sequence[int], dst: Sequence[int], n_nodes: int) -> np.ndarray:
    """Longest-path depth of every node, ignoring self-loops.

    The ontology is a DAG once the self-relation is dropped, so a node's depth
    is one past its deepest predecessor. Sources -- the raw semantic channels --
    land at zero and the goal ends up furthest right, which is the reading
    order the schema was written in.
    """
    src = np.asarray(src, dtype=np.int64).reshape(-1)
    dst = np.asarray(dst, dtype=np.int64).reshape(-1)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    layer = np.zeros(int(n_nodes), dtype=np.int64)
    # n_nodes relaxations settle any DAG; a cycle (which the schema forbids)
    # would stop making progress rather than spin.
    for _ in range(int(n_nodes)):
        changed = False
        for s, d in zip(src, dst):
            if layer[d] < layer[s] + 1:
                layer[d] = layer[s] + 1
                changed = True
        if not changed:
            break
    return layer


def layout_3d(src: Sequence[int], dst: Sequence[int], n_nodes: int,
              goal_node: int = GOAL_NODE) -> np.ndarray:
    """``[n_nodes, 3]`` positions: depth along x, a ring in the y-z plane.

    The ring is what makes this worth rendering in three dimensions at all. A
    layer with eight raw channels drawn as a flat column crosses most of its
    outgoing edges over each other; spread around a circle, every edge into the
    next layer has its own line of sight, and rotating the view separates the
    few that still overlap.
    """
    layer = layer_of(src, dst, n_nodes)
    depth = int(layer.max()) + 1
    pos = np.zeros((int(n_nodes), 3), dtype=float)
    x_span = 4.0
    for level in range(depth):
        members = np.flatnonzero(layer == level)
        x = -x_span / 2.0 + x_span * (level / max(depth - 1, 1))
        if members.size == 1:
            pos[members[0]] = (x, 0.0, 0.0)
            continue
        # Roughly constant arc length between neighbours, so a crowded layer
        # opens out instead of packing its nodes together -- capped, or the
        # nine raw channels would push the ring wider than the graph is deep
        # and the depth axis would stop reading as depth.
        radius = min(1.7, max(0.75, 0.30 * members.size))
        # Golden-angle phase per layer: consecutive rings do not line up, so an
        # edge between them is never hidden exactly behind a node.
        phase = 2.399963 * level
        for i, node in enumerate(members):
            angle = phase + 2.0 * math.pi * i / members.size
            pos[node] = (x, radius * math.cos(angle), radius * math.sin(angle))
    # The goal sits on the axis whatever its ring would have said: it is the
    # read-out node, and every edge in the picture is on its way there.
    pos[int(goal_node)] = (x_span / 2.0, 0.0, 0.0)
    return pos


_GEOMETRY: dict[Any, tuple[np.ndarray, np.ndarray]] = {}


def _geometry(src, dst, n_nodes: int, goal_node: int) -> tuple[np.ndarray, np.ndarray]:
    """Positions and depths for a topology, computed once.

    The schema is fixed for a whole run and this is called from inside the
    50 Hz control loop's monitor, so re-deriving the same layout every few
    steps would be pure waste. Keyed on the edge list, so a schema change still
    produces a new layout rather than a stale one.
    """
    key = (int(n_nodes), int(goal_node),
           np.asarray(src, dtype=np.int64).tobytes(),
           np.asarray(dst, dtype=np.int64).tobytes())
    cached = _GEOMETRY.get(key)
    if cached is None:
        cached = _GEOMETRY[key] = (layout_3d(src, dst, n_nodes, goal_node),
                                   layer_of(src, dst, n_nodes))
    return cached


# Which nodes mean "more is worse", per schema that this view can be handed.
# Matched by NAME rather than by index: the legacy 14-node schema's
# ``RISK_NODES`` index tuple describes that schema alone, and applying it to
# the 9-node situation graph coloured four unrelated nodes as risks.
_RISK_NODE_NAMES = frozenset(STATE_RISK_NODES) | {
    "PositionError", "DescentSpeed", "FovMargin", "FOVMargin",
    "SearchDuration", "RelativeDistance", "MeasurementAge", "VisualLossRisk",
    "BatteryRisk", "ImagePlaneMotion", "ScaleRate", "TargetMotion",
}


def _role(index: int, goal_node: int, name: str = "",
          legacy_indices: bool = False) -> str:
    if index == goal_node:
        return "goal"
    if str(name) in _RISK_NODE_NAMES or "risk" in str(name).lower():
        return "risk"
    if legacy_indices and index in RISK_NODES:
        return "risk"
    return "support"


def _plain(value: Any) -> Any:
    """Recursively convert a model trace into strict JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


_SEMANTIC_CHANNELS = (
    ("centroid_x", "Keypoint centroid x", "normalized image", "context"),
    ("centroid_y", "Keypoint centroid y", "normalized image", "context"),
    ("raw_target_scale", "Raw target scale", "normalized image", "context"),
    ("keypoint_confidence", "Keypoint confidence", "0..1", "semantic"),
    ("visible_keypoint_fraction", "Visible keypoint fraction", "0..1", "semantic"),
    ("image_alignment", "Image alignment", "0..1", "semantic"),
    ("apparent_target_scale", "Apparent target scale", "0..1", "semantic"),
    ("image_plane_motion_safety", "Image motion safety", "0..1", "context"),
    ("scale_rate_safety", "Scale-rate safety", "0..1", "context"),
    ("visibility_memory", "Visibility memory", "0..1", "context"),
    ("reacquisition_trend", "Reacquisition trend", "0..1", "context"),
    ("vertical_motion_safety", "Vertical-motion safety", "0..1", "semantic"),
    ("attitude_stability", "Attitude stability", "0..1", "semantic"),
    ("battery_risk", "Battery risk", "0..1", "semantic"),
    ("visual_loss_risk", "Visual-loss risk", "0..1", "context"),
    ("visual_loss_duration_s", "Visual-loss duration", "s", "context"),
)

_SENSOR_CHANNEL_LINKS = (
    ("landing_camera", "centroid_x", "confidence-weighted keypoint centroid"),
    ("landing_camera", "centroid_y", "confidence-weighted keypoint centroid"),
    ("landing_camera", "raw_target_scale", "RMS keypoint spread"),
    ("keypoint_encoder", "keypoint_confidence", "heatmap entropy + visibility"),
    ("keypoint_encoder", "visible_keypoint_fraction", "visible keypoints / 6"),
    ("keypoint_encoder", "image_alignment", "centroid distance from image centre"),
    ("keypoint_encoder", "apparent_target_scale", "reliability-weighted scale"),
    ("landing_camera", "image_plane_motion_safety", "centroid delta / dt"),
    ("temporal_context", "image_plane_motion_safety", "previous frame"),
    ("landing_camera", "scale_rate_safety", "scale delta / dt"),
    ("temporal_context", "scale_rate_safety", "previous frame"),
    ("keypoint_encoder", "visibility_memory", "current reliability"),
    ("temporal_context", "visibility_memory", "1.5 s exponential memory"),
    ("keypoint_encoder", "reacquisition_trend", "confidence recovery"),
    ("temporal_context", "reacquisition_trend", "previous confidence"),
    ("px4_odometry", "vertical_motion_safety", "exp(-|vz| / 0.6)"),
    ("px4_imu", "attitude_stability", "exp(-tilt / 22 deg)"),
    ("battery_monitor", "battery_risk", "1 - reserve"),
    ("keypoint_encoder", "visual_loss_risk", "keypoint dropout"),
    ("temporal_context", "visual_loss_risk", "loss duration / 2 s"),
    ("temporal_context", "visual_loss_duration_s", "consecutive dropout time"),
)

_ONTOLOGY_CHANNEL_LINKS = (
    ("centroid_x", "AlignmentError", "bearing from camera nadir column"),
    ("vertical_motion_safety", "DescentRate", "recover |vz|, then soft saturation"),
    ("image_plane_motion_safety", "TargetMotion", "1 - motion safety"),
    ("centroid_x", "FOVMargin", "distance to horizontal image edge"),
    ("centroid_y", "FOVMargin", "distance to vertical image edge"),
    ("visual_loss_risk", "MeasurementAge", "bounded observation age"),
    ("raw_target_scale", "RelativeRange", "inverse apparent scale"),
    ("keypoint_confidence", "PadVisibility", "confidence × visible fraction"),
    ("visible_keypoint_fraction", "PadVisibility", "confidence × visible fraction"),
    ("visibility_memory", "PadVisibility", "dropout memory support"),
    ("visual_loss_risk", "PadVisibility", "suppresses stale memory"),
    # Selective-RGAT v1: raw evidence nodes expose which onboard channels form
    # each semantic/context relation before message passing.
    ("centroid_x", "LongitudinalTrackingEvidence", "signed image bearing"),
    ("image_plane_motion_safety", "LongitudinalTrackingEvidence",
     "causal centroid-rate context"),
    ("centroid_x", "VisualRetentionEvidence", "horizontal FOV margin"),
    ("centroid_y", "VisualRetentionEvidence", "vertical FOV margin"),
    ("visible_keypoint_fraction", "VisualRetentionEvidence", "current visibility"),
    ("visibility_memory", "VisualRetentionEvidence", "retention history"),
    ("vertical_motion_safety", "ContactKinematicsEvidence", "observable vertical rate"),
    ("raw_target_scale", "ContactKinematicsEvidence", "apparent approach scale"),
    ("keypoint_confidence", "StateReliabilityEvidence", "measurement confidence"),
    ("visible_keypoint_fraction", "StateReliabilityEvidence", "validity mask"),
    ("visual_loss_risk", "StateReliabilityEvidence", "age/stale context"),
)


def sensor_ontology_provenance(observation, graph) -> dict[str, Any]:
    """Describe the live, non-privileged path from sensors into ``G_t``.

    This mirrors the transformations in :mod:`ontology_rgat.rgat.state_graph`;
    it does not infer causality from R-GAT attention.
    """
    centroid = np.asarray(observation.centroid_xy, dtype=float).reshape(-1)
    values = {
        "centroid_x": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "raw_target_scale": float(observation.raw_scale),
        "visual_loss_duration_s": float(observation.visual_loss_duration_s),
    }
    values.update({name: float(getattr(observation, name))
                   for name, *_ in _SEMANTIC_CHANNELS
                   if hasattr(observation, name)})
    vertical_safety = max(1e-6, min(1.0, values["vertical_motion_safety"]))
    sensors = [
        {"id": "landing_camera", "label": "Landing camera", "kind": "sensor",
         "readings": [
             {"label": "centroid x", "value": values["centroid_x"], "unit": "norm"},
             {"label": "centroid y", "value": values["centroid_y"], "unit": "norm"},
             {"label": "raw scale", "value": values["raw_target_scale"], "unit": "norm"},
         ]},
        {"id": "keypoint_encoder", "label": "Frozen 6-keypoint encoder",
         "kind": "inference", "readings": [
             {"label": "confidence", "value": values["keypoint_confidence"], "unit": "0..1"},
             {"label": "visible", "value": values["visible_keypoint_fraction"], "unit": "fraction"},
         ]},
        {"id": "temporal_context", "label": "Visual history", "kind": "context",
         "readings": [{"label": "loss age", "value": values["visual_loss_duration_s"],
                       "unit": "s"}]},
        {"id": "px4_odometry", "label": "PX4 local odometry", "kind": "sensor",
         "readings": [{"label": "|vertical speed|",
                       "value": float(-0.6 * math.log(vertical_safety)),
                       "unit": "m/s"}]},
        {"id": "px4_imu", "label": "PX4 IMU attitude", "kind": "sensor",
         "readings": [{"label": "stability", "value": values["attitude_stability"],
                       "unit": "0..1"}]},
        {"id": "battery_monitor", "label": "PX4 battery monitor", "kind": "sensor",
         "readings": [{"label": "reserve", "value": 1.0 - values["battery_risk"],
                       "unit": "fraction"}]},
    ]
    semantics = [
        {"id": name, "label": label, "value": values.get(name),
         "unit": unit, "kind": kind}
        for name, label, unit, kind in _SEMANTIC_CHANNELS
    ]
    node_names = {str(name) for name in graph.node_names}
    connections = [
        {"source": source, "target": target, "stage": "sensor_to_semantic",
         "transform": transform}
        for source, target, transform in _SENSOR_CHANNEL_LINKS
    ]
    connections.extend(
        {"source": source, "target": target, "stage": "semantic_to_ontology",
         "transform": transform}
        for source, target, transform in _ONTOLOGY_CHANNEL_LINKS
        if target in node_names)
    return {
        "format": "ontology-rgat-sensor-provenance-v1",
        "sensors": sensors,
        "semantics": semantics,
        "connections": connections,
        "boundary": [
            "No simulator truth", "No geometric FOV label",
            "No relative-pose estimate", "Onboard camera/PX4 signals only",
        ],
    }


def graph_payload(graph, values: Sequence[float] | None = None, *,
                  potential=None, source: str = "", phi: float | None = None,
                  extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Nodes, edges and attention for one graph, ready to serialise.

    ``values`` is the node activation vector; when it is not given it is read
    back off the feature matrix, whose first row is exactly that (see
    :func:`ontology_rgat.semantic.node_features`). ``potential`` is optional --
    without it the picture is the schema alone, which is what the dashboard
    shows before the R-GAT has been trained.
    """
    src = np.asarray(graph.src, dtype=int).reshape(-1)
    dst = np.asarray(graph.dst, dtype=int).reshape(-1)
    rel = np.asarray(graph.rel, dtype=int).reshape(-1)
    n_nodes = len(graph.node_names)
    if values is None:
        values = np.asarray(graph.X, dtype=float)[0, :]
    values = np.asarray(values, dtype=float).reshape(-1)

    alpha = None
    alpha_heads = None
    relation_mean = None
    node_embeddings = None
    model_trace = None
    if potential is not None:
        try:
            explained = potential.explain(graph)
            if explained.get("edge_alpha") is not None:
                alpha = np.asarray(
                    explained["edge_alpha"], dtype=float).reshape(-1)
            if explained.get("edge_alpha_heads") is not None:
                alpha_heads = np.asarray(
                    explained["edge_alpha_heads"], dtype=float)
                if alpha_heads.ndim == 1:
                    alpha_heads = alpha_heads[:, None]
            if explained.get("relation_mean") is not None:
                relation_mean = np.asarray(
                    explained["relation_mean"], dtype=float)
            if explained.get("node_embeddings") is not None:
                node_embeddings = np.asarray(
                    explained["node_embeddings"], dtype=float)
            model_trace = _plain(explained.get("model"))
        except Exception as exc:                       # pragma: no cover - defensive
            # A view is never worth taking a run down with it.
            print(f"WARNING: attention read-out failed for the 3D graph: {exc}")
            alpha = None

    pos, layers = _geometry(src, dst, n_nodes, int(graph.goal_node))
    nodes = []
    for i, (name, layer) in enumerate(zip(graph.node_names, layers)):
        node = {
            "name": str(name),
            "value": float(values[i]) if i < values.size else 0.0,
            "pos": [round(float(c), 4) for c in pos[i]],
            "layer": int(layer),
            "role": _role(i, int(graph.goal_node), str(name),
                          # The index tuple only describes the 14-node
                          # legacy schema, so it is consulted only for it.
                          legacy_indices=n_nodes == N_NODES),
        }
        if (node_embeddings is not None and node_embeddings.ndim == 2
                and i < node_embeddings.shape[0]):
            embedding = node_embeddings[i]
            node["embedding_mean"] = float(np.mean(embedding))
            node["embedding_l2"] = float(np.linalg.norm(embedding))
        nodes.append(node)

    edges = []
    for e in range(src.size):
        edge = {"s": int(src[e]), "d": int(dst[e]), "r": int(rel[e])}
        if alpha is not None and e < alpha.size:
            edge["a"] = round(float(alpha[e]), 6)
        if (alpha_heads is not None and alpha_heads.ndim == 2
                and e < alpha_heads.shape[0]):
            edge["heads"] = [round(float(value), 6)
                             for value in alpha_heads[e]]
        edges.append(edge)

    relations = []
    for r, name in enumerate(graph.relation_names):
        entry = {"name": str(name)}
        if relation_mean is not None and r < relation_mean.size:
            entry["mean"] = round(float(relation_mean[r]), 6)
        relations.append(entry)

    payload: dict[str, Any] = {
        "source": source,
        "attention": alpha is not None,
        "nodes": nodes,
        "edges": edges,
        "relations": relations,
        "goal_node": int(graph.goal_node),
        "time": time.time(),
    }
    if model_trace is not None:
        payload["model"] = model_trace
    if phi is not None and np.isfinite(phi):
        payload["phi"] = float(phi)
    if extra:
        payload.update(extra)
    return payload


class GraphPublisher:
    """Throttled writer of graph snapshots into the live store.

    The potential is an attribute rather than a constructor argument because
    the pipeline has an episode monitor running before the R-GAT exists: the
    dataset stage publishes the bare schema, and the same publisher starts
    carrying attention the moment training hands a model over.
    """

    def __init__(self, cfg, store: LiveStore | None = None, potential=None):
        self.cfg = cfg
        self.store = store or STORE
        self.potential = potential
        opt = getattr(cfg.viz, "graph3d", None)
        self.enabled = bool(getattr(opt, "enabled", True))
        self.every = max(1, int(getattr(opt, "every", 5)))
        self._counter = 0

    def publish(self, graph, values=None, *, source: str = "",
                phi: float | None = None, force: bool = False,
                extra: dict[str, Any] | None = None) -> None:
        if not self.enabled or graph is None:
            return
        index = self._counter
        self._counter += 1
        if index % self.every and not force:
            return
        self.store.graph(graph_payload(graph, values, potential=self.potential,
                                       source=source, phi=phi, extra=extra))

    def clear(self) -> None:
        self._counter = 0
