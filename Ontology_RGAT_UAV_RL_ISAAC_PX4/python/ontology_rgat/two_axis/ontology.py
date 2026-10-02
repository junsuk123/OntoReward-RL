"""Compact typed context graph derived only from the causal packet."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from ..contracts.observation import CausalObservationPacket, ObservationRegistry
from .contracts import load_v2_registry, packet_fields


SCHEMA_VERSION = "compact_landing_context_graph_v1"
SEMANTIC_NODES = (
    "PadVisibility", "PadMotion", "DroneTranslation", "DroneAttitude",
    "RelativeTracking", "TrackingCorrection", "ViewRecovery",
    "DescentEligibility", "LandingInhibit")
QUERY_NODES = ("PolicyNode", "ValueNode")
NODE_NAMES = SEMANTIC_NODES + QUERY_NODES
NODE_CLASSES = (
    "ObservationState", "MotionEstimate", "OwnState", "OwnState",
    "RelativeState", "DecisionContext", "DecisionContext",
    "DecisionContext", "DecisionContext", "ReadoutQuery", "ReadoutQuery")
RELATION_NAMES = (
    "informs", "affects_visibility", "supports", "inhibits",
    "contributes", "self")
FEATURE_CHANNELS = (
    "primary", "secondary", "tertiary", "valid", "confidence", "support",
    "inhibit", "direction", "uncertainty", "bias")

DECLARED_EDGES = (
    ("PadMotion", "informs", "RelativeTracking"),
    ("DroneTranslation", "informs", "RelativeTracking"),
    ("DroneAttitude", "affects_visibility", "PadVisibility"),
    ("DroneTranslation", "affects_visibility", "PadVisibility"),
    ("PadMotion", "informs", "TrackingCorrection"),
    ("RelativeTracking", "informs", "TrackingCorrection"),
    ("PadVisibility", "informs", "ViewRecovery"),
    ("DroneAttitude", "informs", "ViewRecovery"),
    ("RelativeTracking", "informs", "ViewRecovery"),
    ("PadVisibility", "supports", "DescentEligibility"),
    ("RelativeTracking", "informs", "DescentEligibility"),
    ("DroneTranslation", "informs", "DescentEligibility"),
    ("DroneAttitude", "informs", "DescentEligibility"),
    ("PadVisibility", "informs", "LandingInhibit"),
    ("RelativeTracking", "informs", "LandingInhibit"),
    ("DroneTranslation", "informs", "LandingInhibit"),
    ("LandingInhibit", "inhibits", "DescentEligibility"),
)


def _edges() -> tuple[tuple[str, str, str], ...]:
    contributes = tuple((source, "contributes", query)
                        for source in SEMANTIC_NODES for query in QUERY_NODES)
    self_edges = tuple((node, "self", node) for node in NODE_NAMES)
    return DECLARED_EDGES + contributes + self_edges


GRAPH_EDGES = _edges()


def _schema_payload() -> dict:
    return {"version": SCHEMA_VERSION, "nodes": NODE_NAMES,
            "classes": NODE_CLASSES, "relations": RELATION_NAMES,
            "edges": GRAPH_EDGES, "features": FEATURE_CHANNELS,
            "identity_features": NODE_NAMES, "inverse_edges": False}


GRAPH_SCHEMA_HASH = hashlib.sha256(json.dumps(
    _schema_payload(), sort_keys=True, separators=(",", ":"))
    .encode("utf-8")).hexdigest()


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _soft_positive(value: float, scale: float = 1.0) -> float:
    return _clip01(0.5 + 0.5 * math.tanh(value / max(scale, 1e-6)))


def behavior_case_scores(values: dict[str, float]) -> dict[str, float]:
    """Bounded engineering-prior support, never an acceleration label."""
    ex, rv = values["exEstimate"], values["relativeVxEstimate"]
    ax = values["padAxEstimate"]
    detected, valid = values["detected"], values["trackInitialized"]
    age = values["timeSinceLastDetection"] * 3.0
    uncertainty = max(values["positionStd"], values["velocityStd"])
    margin = values["predictedFovMargin"]
    pitch = abs(values["sinTheta"])
    rate = abs(values["pitchRate"])
    height = max(0.0, values["h"])
    deadline_pressure = 1.0 - _clip01(values["remainingMissionTime"])
    alignment = 1.0 - _clip01(abs(ex) * 2.0)
    relative_stable = 1.0 - _clip01(abs(rv) * 2.0)
    scores = {
        "C01": _soft_positive(ax, .25) * _soft_positive(rv, .25)
               * _soft_positive(ex, .25) * valid,
        "C02": _soft_positive(-ex * rv, .15) * _clip01(abs(rv) * 2.0),
        "C03": (1.0 - detected) * _clip01(abs(values["predictedBearing"]))
               * _clip01(pitch + rate),
        "C04": (1.0 - detected) * _soft_positive(margin, .2)
               * (1.0 - _clip01(pitch + rate)),
        "C05": (1.0 - detected) * _clip01(1.0 - age / 0.5)
               * (1.0 - uncertainty),
        "C06": detected * valid * (1.0 - uncertainty),
        "C07": detected * valid * uncertainty,
        "C08": valid * detected * alignment * relative_stable
               * (1.0 - _clip01(pitch + rate)),
        "C09": _clip01((1.0 - height) *
                        (abs(values["vz"]) + abs(rv) + pitch + rate)),
        "C10": _clip01(max(age / 3.0, uncertainty)),
        "C11": (1.0 - detected) * _clip01(-margin) * _clip01(1.0 - height),
        "C12": deadline_pressure * (1.0 - alignment * relative_stable),
    }
    return {key: _clip01(value) for key, value in scores.items()}


@dataclass(frozen=True)
class ContextGraph:
    X: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    rel: np.ndarray
    node_names: tuple[str, ...] = NODE_NAMES
    relation_names: tuple[str, ...] = RELATION_NAMES
    schema_hash: str = GRAPH_SCHEMA_HASH
    packet_registry_hash: str = ""

    def __post_init__(self) -> None:
        expected = (len(NODE_NAMES), len(FEATURE_CHANNELS) + len(NODE_NAMES))
        if np.asarray(self.X).shape != expected:
            raise ValueError(f"context graph features must have shape {expected}")
        if not np.isfinite(self.X).all():
            raise ValueError("context graph features must be finite")


def build_context_graph(packet: CausalObservationPacket,
                        registry: ObservationRegistry | None = None) -> ContextGraph:
    registry = registry or load_v2_registry()
    values = packet_fields(packet, registry)
    scores = behavior_case_scores(values)
    base = np.zeros((len(NODE_NAMES), len(FEATURE_CHANNELS)), dtype=np.float32)

    def put(name: str, **features: float) -> None:
        row = NODE_NAMES.index(name)
        for key, value in features.items():
            base[row, FEATURE_CHANNELS.index(key)] = float(value)

    put("PadVisibility", primary=values["detected"],
        secondary=values["measuredBearing"], tertiary=values["predictedFovMargin"],
        valid=values["bearingValid"], confidence=values["detectionConfidence"],
        uncertainty=values["timeSinceLastDetection"])
    put("PadMotion", primary=values["padVxEstimate"],
        secondary=values["padAxEstimate"], tertiary=values["velocityStd"],
        valid=values["trackInitialized"],
        confidence=1.0 - values["positionStd"],
        uncertainty=values["accelerationStd"])
    put("DroneTranslation", primary=values["h"], secondary=values["vx"],
        tertiary=values["vz"], valid=1.0, confidence=1.0)
    put("DroneAttitude", primary=values["sinTheta"],
        secondary=values["cosTheta"], tertiary=values["pitchRate"],
        valid=1.0, confidence=1.0, direction=values["sinTheta"])
    correction = math.tanh(values["exEstimate"] + values["relativeVxEstimate"])
    put("RelativeTracking", primary=values["exEstimate"],
        secondary=values["relativeVxEstimate"],
        tertiary=values["predictedFovMargin"],
        valid=values["trackInitialized"],
        confidence=1.0 - values["positionStd"],
        uncertainty=max(values["positionStd"], values["velocityStd"]),
        direction=math.tanh(values["exEstimate"]))
    put("TrackingCorrection", primary=correction, direction=correction,
        valid=values["trackInitialized"], support=max(scores["C01"], scores["C02"]),
        confidence=1.0 - values["velocityStd"])
    put("ViewRecovery", primary=1.0 - values["detected"],
        secondary=values["predictedBearing"], direction=-math.tanh(
            values["predictedBearing"]), valid=values["trackInitialized"],
        support=max(scores["C03"], scores["C04"], scores["C05"], scores["C11"]),
        uncertainty=values["positionStd"])
    put("DescentEligibility", primary=scores["C08"], support=scores["C08"],
        inhibit=max(scores["C09"], scores["C10"]),
        valid=values["trackInitialized"], confidence=values["detectionConfidence"])
    inhibit = max(scores["C07"], scores["C09"], scores["C10"], scores["C12"],
                  values["landingInhibited"], values["abortRequested"])
    put("LandingInhibit", primary=inhibit, inhibit=inhibit, support=inhibit,
        valid=1.0, confidence=1.0 - values["positionStd"],
        uncertainty=max(values["positionStd"], values["velocityStd"]))
    # Readout queries contain constants/identity only, never outcomes or commands.
    put("PolicyNode", bias=1.0)
    put("ValueNode", bias=1.0)
    identity = np.eye(len(NODE_NAMES), dtype=np.float32)
    X = np.concatenate((base, identity), axis=1)
    index = {name: i for i, name in enumerate(NODE_NAMES)}
    relation = {name: i for i, name in enumerate(RELATION_NAMES)}
    src = np.array([index[s] for s, _, _ in GRAPH_EDGES], dtype=np.int64)
    dst = np.array([index[d] for _, _, d in GRAPH_EDGES], dtype=np.int64)
    rel = np.array([relation[r] for _, r, _ in GRAPH_EDGES], dtype=np.int64)
    return ContextGraph(X, src, dst, rel,
                        packet_registry_hash=packet.registry_sha256)


def semantic_flat_features(graph: ContextGraph) -> np.ndarray:
    """Information-matched control: identical node values, no edges."""
    return np.asarray(graph.X, dtype=np.float32).reshape(-1).copy()


def query_reachable(source: str, query: str) -> bool:
    successors: dict[str, list[str]] = {name: [] for name in NODE_NAMES}
    for src, relation, dst in GRAPH_EDGES:
        if relation != "self":
            successors[src].append(dst)
    pending, seen = [source], set()
    while pending:
        node = pending.pop()
        if node == query:
            return True
        if node not in seen:
            seen.add(node)
            pending.extend(successors[node])
    return False
