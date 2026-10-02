"""Causal ontology situation graph for selective relation gating."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from ..contracts.observation import CausalObservationPacket, ObservationRegistry
from ..semantic import OntologyGraph


SELECTIVE_GRAPH_VERSION = "ontology_rgat.selective_situation_graph/1"
SELECTIVE_NODE_NAMES = (
    "LongitudinalTrackingEvidence", "VisualRetentionEvidence",
    "ContactKinematicsEvidence", "StateReliabilityEvidence",
    "LongitudinalTracking", "VisualRetention", "ContactKinematics",
    "StateReliability", "SafeObservableLanding",
)
SELECTIVE_RELATION_NAMES = (
    "Tracking_evidence", "VisualRetention_evidence",
    "ContactKinematics_evidence", "StateReliability_evidence",
    "concept_to_landing_goal", "self",
)
ADAPTIVE_RELATIONS = SELECTIVE_RELATION_NAMES[:4]
INVARIANT_RELATIONS = SELECTIVE_RELATION_NAMES[4:]
SELECTIVE_GOAL_NODE = "SafeObservableLanding"
SELECTIVE_GRAPH_EDGES = (
    ("LongitudinalTrackingEvidence", "LongitudinalTracking", "Tracking_evidence"),
    ("VisualRetentionEvidence", "VisualRetention", "VisualRetention_evidence"),
    ("ContactKinematicsEvidence", "ContactKinematics", "ContactKinematics_evidence"),
    ("StateReliabilityEvidence", "StateReliability", "StateReliability_evidence"),
    ("LongitudinalTracking", SELECTIVE_GOAL_NODE, "concept_to_landing_goal"),
    ("VisualRetention", SELECTIVE_GOAL_NODE, "concept_to_landing_goal"),
    ("ContactKinematics", SELECTIVE_GOAL_NODE, "concept_to_landing_goal"),
    ("StateReliability", SELECTIVE_GOAL_NODE, "concept_to_landing_goal"),
)
SELECTIVE_FEATURE_ROWS = 4
SELECTIVE_GRAPH_INPUT_DIM = SELECTIVE_FEATURE_ROWS + len(SELECTIVE_NODE_NAMES)


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


SELECTIVE_SCHEMA_HASH = _digest({
    "version": SELECTIVE_GRAPH_VERSION, "nodes": SELECTIVE_NODE_NAMES,
    "relations": SELECTIVE_RELATION_NAMES, "edges": SELECTIVE_GRAPH_EDGES,
    "feature_rows": SELECTIVE_FEATURE_ROWS,
})
RELATION_PARTITION_HASH = _digest({
    "adaptive": ADAPTIVE_RELATIONS, "invariant": INVARIANT_RELATIONS})


def _field(packet: CausalObservationPacket, registry: ObservationRegistry,
           name: str) -> np.ndarray:
    offset = 0
    for spec in registry.fields:
        size = int(spec["size"])
        if spec["name"] == name:
            return packet.values[offset:offset + size]
        offset += size
    raise KeyError(name)


def selective_node_values(packet: CausalObservationPacket,
                          registry: ObservationRegistry) -> np.ndarray:
    packet.assert_registry(registry)
    centroid = _field(packet, registry, "centroid_xy")
    rate = _field(packet, registry, "centroid_rate_xy")
    visibility = float(_field(packet, registry, "visible_fraction")[0])
    margin = float(_field(packet, registry, "fov_margin")[0])
    scale = float(_field(packet, registry, "apparent_scale")[0])
    vertical = float(_field(packet, registry, "body_velocity")[2])
    valid = float(_field(packet, registry, "measurement_valid")[0])
    stale = float(_field(packet, registry, "stale")[0])
    age = float(_field(packet, registry, "measurement_age_s")[0])
    tracking = float(np.tanh(abs(float(centroid[0]))))
    retention = float(np.clip(0.5 * visibility + 0.5 * margin, 0.0, 1.0))
    contact = float(np.clip(0.5 * np.tanh(abs(vertical))
                            + 0.5 * np.tanh(abs(scale)), 0.0, 1.0))
    reliability = float(np.clip(valid * (1.0 - stale)
                                * np.exp(-max(age, 0.0)), 0.0, 1.0))
    values = [tracking, retention, contact, reliability,
              0.0, 0.0, 0.0, 0.0, 0.0]
    directions = [float(np.tanh(centroid[0])), float(np.tanh(rate[0])),
                  float(np.tanh(vertical)), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return np.asarray([values, directions], dtype=np.float32)


def build_selective_graph(packet: CausalObservationPacket, *,
                          registry: ObservationRegistry) -> OntologyGraph:
    """Build the graph; its signature deliberately has no truth argument."""
    dynamic = selective_node_values(packet, registry)
    count = len(SELECTIVE_NODE_NAMES)
    X = np.zeros((SELECTIVE_GRAPH_INPUT_DIM, count), dtype=np.float32)
    X[0:2] = dynamic
    X[2] = 1.0
    X[3, :4] = 1.0
    X[SELECTIVE_FEATURE_ROWS:] = np.eye(count, dtype=np.float32)
    nodes = {name: index for index, name in enumerate(SELECTIVE_NODE_NAMES)}
    relations = {name: index for index, name in enumerate(SELECTIVE_RELATION_NAMES)}
    src = [nodes[a] for a, _, _ in SELECTIVE_GRAPH_EDGES] + list(range(count))
    dst = [nodes[b] for _, b, _ in SELECTIVE_GRAPH_EDGES] + list(range(count))
    rel = [relations[r] for _, _, r in SELECTIVE_GRAPH_EDGES]
    rel.extend([relations["self"]] * count)
    return OntologyGraph(X, np.asarray(src), np.asarray(dst), np.asarray(rel),
                         nodes[SELECTIVE_GOAL_NODE], SELECTIVE_NODE_NAMES,
                         SELECTIVE_RELATION_NAMES)


def selective_topology_hash(graph: OntologyGraph) -> str:
    return _digest({"src": graph.src.tolist(), "dst": graph.dst.tolist(),
                    "rel": graph.rel.tolist(), "nodes": graph.node_names,
                    "relations": graph.relation_names})


def unreachable_or_too_distant_nodes() -> tuple[str, ...]:
    successors = {name: [] for name in SELECTIVE_NODE_NAMES}
    for source, target, _ in SELECTIVE_GRAPH_EDGES:
        successors[source].append(target)
    bad = []
    for source in SELECTIVE_NODE_NAMES[:-1]:
        frontier = [(source, 0)]
        seen = set()
        distance = None
        while frontier:
            node, depth = frontier.pop(0)
            if node == SELECTIVE_GOAL_NODE:
                distance = depth
                break
            if node in seen or depth >= 2:
                continue
            seen.add(node)
            frontier.extend((child, depth + 1) for child in successors[node])
        if distance is None or distance > 2:
            bad.append(source)
    return tuple(bad)
