"""The minimal landing ontology: 12 nodes, 6 relations, 8 uniform channels.

TBox (this module's constants) is static and hashed; an ABox instance is built
per observation by ``MinimalOntology.build``. The edge SET never changes. A
conditional relation is weakened through its edge weight in [0, 1] instead of
removed, so every graph has the same tensor shape and batches.

Every judgement the observation deliberately leaves out -- pad velocity,
visibility, why the pad is not visible, the bias a constant force leaves in
the track (``TrackingBias``, see bias.py for why it is required), readiness,
inhibit -- is made here.
Absolute own x/y never enter a feature: landing is translation invariant.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from . import OBSERVATION_SCHEMA_ID, ONTOLOGY_SCHEMA_ID
from .constants import DEFAULT_CONSTANTS, LandingConstants
from .observation import LandingObservation
from .bias import TrackingBias
from .pad_loss import PadLossAssessment, PadMemory, bearing_fraction

NODES: tuple[str, ...] = (
    "UAV", "LandingPad", "CameraObservation", "Localization", "PadMemory",
    "RelativePosition", "RelativeVelocity", "Visibility", "TerminalOcclusion",
    "TargetLost", "DescentReadiness", "LandingInhibit", "TrackingBias")
NODE_INDEX = {name: i for i, name in enumerate(NODES)}
NODE_CLASSES: tuple[str, ...] = ("Entity", "Observation", "Situation", "Decision")
NODE_CLASS_OF: tuple[int, ...] = (0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 2)
RELATIONS: tuple[str, ...] = (
    "observes", "relative_to", "informs", "supports", "inhibits", "self")
REL = {name: i for i, name in enumerate(RELATIONS)}
FEATURE_CHANNELS: tuple[str, ...] = (
    "c0", "c1", "c2", "magnitude", "validity", "freshness", "urgency", "typeId")
FEATURE_DIM = len(FEATURE_CHANNELS)

DECLARED_EDGES: tuple[tuple[str, str, str], ...] = (
    ("CameraObservation", "observes", "LandingPad"),
    ("Localization", "observes", "UAV"),
    ("UAV", "relative_to", "RelativePosition"),
    ("LandingPad", "relative_to", "RelativePosition"),
    ("UAV", "relative_to", "RelativeVelocity"),
    ("LandingPad", "relative_to", "RelativeVelocity"),
    ("CameraObservation", "informs", "PadMemory"),
    ("Localization", "informs", "PadMemory"),
    ("PadMemory", "informs", "RelativePosition"),
    ("RelativePosition", "informs", "Visibility"),
    ("RelativeVelocity", "informs", "Visibility"),
    ("CameraObservation", "informs", "Visibility"),
    ("PadMemory", "informs", "TerminalOcclusion"),
    ("PadMemory", "informs", "TargetLost"),
    ("RelativePosition", "informs", "TerminalOcclusion"),
    ("RelativeVelocity", "informs", "TerminalOcclusion"),
    ("Visibility", "informs", "TargetLost"),
    ("RelativePosition", "informs", "DescentReadiness"),
    ("RelativeVelocity", "informs", "DescentReadiness"),
    ("CameraObservation", "informs", "LandingInhibit"),
    ("Localization", "informs", "LandingInhibit"),
    ("TargetLost", "informs", "LandingInhibit"),
    ("Visibility", "supports", "DescentReadiness"),
    ("TerminalOcclusion", "supports", "DescentReadiness"),
    ("LandingInhibit", "inhibits", "DescentReadiness"),
    ("TargetLost", "inhibits", "DescentReadiness"),
    ("TerminalOcclusion", "inhibits", "TargetLost"),
    ("RelativePosition", "informs", "TrackingBias"),
    ("UAV", "informs", "TrackingBias"),
    ("TrackingBias", "informs", "DescentReadiness"),
)
EDGES: tuple[tuple[str, str, str], ...] = DECLARED_EDGES + tuple(
    (n, "self", n) for n in NODES)
EDGE_SOURCE = np.array([NODE_INDEX[s] for s, _, _ in EDGES], dtype=np.int64)
EDGE_TARGET = np.array([NODE_INDEX[t] for _, _, t in EDGES], dtype=np.int64)
EDGE_RELATION = np.array([REL[r] for _, r, _ in EDGES], dtype=np.int64)
EDGE_INDEX = {edge: i for i, edge in enumerate(EDGES)}
#: Readout groups: observation, entity, situation, decision.
READOUT_GROUPS: tuple[tuple[int, ...], ...] = (
    (2, 3, 4, 7), (0, 1), (5, 6, 8, 9, 12), (10, 11))
DECISION_NODE = NODE_INDEX["DescentReadiness"]

#: Edges whose weight is a function of the observation; every other edge is 1.
#: value: which weight term applies.
CONDITIONAL_EDGES = {
    ("CameraObservation", "observes", "LandingPad"): "pad_freshness",
    ("LandingPad", "relative_to", "RelativePosition"): "pad_freshness",
    ("LandingPad", "relative_to", "RelativeVelocity"): "pad_freshness",
    ("PadMemory", "informs", "RelativePosition"): "memory",
    ("TerminalOcclusion", "supports", "DescentReadiness"): "terminal",
    ("TerminalOcclusion", "inhibits", "TargetLost"): "terminal",
    ("TargetLost", "inhibits", "DescentReadiness"): "lost",
    ("TargetLost", "informs", "LandingInhibit"): "lost",
}

# Normalization scales.
SPEED_SCALE = 3.0
OFFSET_SCALE = 3.0
HEIGHT_SCALE = 8.0


def schema_dict(constants: LandingConstants = DEFAULT_CONSTANTS) -> dict:
    return {
        "schema_id": ONTOLOGY_SCHEMA_ID,
        "observation_schema_id": OBSERVATION_SCHEMA_ID,
        "nodes": NODES, "node_classes": NODE_CLASSES, "node_class_of": NODE_CLASS_OF,
        "relations": RELATIONS, "edges": EDGES, "features": FEATURE_CHANNELS,
        "readout_groups": READOUT_GROUPS,
        "conditional_edges": sorted([list(k) + [v] for k, v in CONDITIONAL_EDGES.items()]),
        "scales": {"speed": SPEED_SCALE, "offset": OFFSET_SCALE, "height": HEIGHT_SCALE},
        "constants": constants.as_dict(),
    }


def schema_hash(constants: LandingConstants = DEFAULT_CONSTANTS) -> str:
    return hashlib.sha256(json.dumps(
        schema_dict(constants), sort_keys=True, default=list).encode()).hexdigest()


@dataclass(frozen=True)
class GraphInstance:
    stamp: float
    features: np.ndarray        # (13, 8) float32 in [-1, 1]
    edge_weight: np.ndarray     # (E,) float32 in [0, 1]
    assessment: PadLossAssessment
    schema_hash: str


def _clip(x) -> float:
    return float(np.clip(x, -1.0, 1.0))


class MinimalOntology:
    """Builds one ABox instance per observation; holds the pad memory."""

    def __init__(self, constants: LandingConstants = DEFAULT_CONSTANTS):
        self.constants = constants
        self.memory = PadMemory(constants)
        self.bias = TrackingBias(constants)
        self.hash = schema_hash(constants)

    def reset(self) -> None:
        self.memory = PadMemory(self.constants)
        self.bias = TrackingBias(self.constants)

    def build(self, obs: LandingObservation) -> GraphInstance:
        c = self.constants
        a = self.memory.update(obs)
        self.bias.update(obs, a)
        b = self.bias.normalized()
        f_pad, f_own = a.pad_freshness, a.own_freshness
        own_valid = float(obs.own_valid)
        seen = float(obs.ever_detected)
        v_own = np.asarray(obs.own_velocity, float)
        rel = a.relative_position
        h = -float(rel[2])
        d_xy = float(np.hypot(rel[0], rel[1]))
        v_rel = a.relative_velocity if a.relative_velocity_valid else np.zeros(3)
        v_rel_valid = float(a.relative_velocity_valid)
        v_pad = a.pad_velocity
        half = c.min_half_fov_rad
        h_cam = c.camera_height(h)

        def bearing(axis_offset: float) -> float:
            return math.atan2(axis_offset, max(h_cam, 1e-3)) / half if seen else 0.0

        bx, by = bearing(rel[0]), bearing(rel[1])
        margin_x, margin_y = 1 - abs(bx), 1 - abs(by)
        predicted = rel + v_rel * c.visibility_horizon_s
        predicted_margin = 1 - bearing_fraction(predicted, c) if seen else -1.0
        visibility_urgency = max(0.0, -predicted_margin)

        allowed_sink = c.allowed_sink(h)
        sink_excess = max(0.0, -float(v_own[2]) - allowed_sink)
        align = 1 - min(d_xy / c.descent_gate_width(h), 1.0) if seen else 0.0
        slow_xy = 1 - min(float(np.hypot(v_rel[0], v_rel[1])) / c.touchdown_xy_speed_m_s, 1.0)
        sink_ok = 1 - min(sink_excess / c.touchdown_z_speed_m_s, 1.0)
        evidence = max(f_pad, a.terminal_score)
        readiness = align * slow_xy * sink_ok * evidence * own_valid

        stale = (1 - f_pad) * (1 - a.terminal_score)
        loc_invalid = 1 - own_valid * f_own
        inhibit = max(stale, loc_invalid, a.lost_score)
        age_s = (obs.stamp - obs.pad_capture_stamp) if seen else c.pad_age_cap_s
        terminal_age = min(age_s / c.terminal_window_s, 1.0)

        rows = [
            # UAV
            [*(v_own / SPEED_SCALE), np.linalg.norm(v_own) / SPEED_SCALE, own_valid,
             f_own, max(0.0, -v_own[2]) / c.touchdown_z_speed_m_s],
            # LandingPad
            [*(v_pad / SPEED_SCALE), np.linalg.norm(v_pad) / SPEED_SCALE,
             v_rel_valid, f_pad, 0.0],
            # CameraObservation
            [bx, by, 0.0, math.hypot(bx, by), float(obs.pad_detected), f_pad, 0.0],
            # Localization
            [0.0, 0.0, 0.0, 0.0, own_valid, f_own, 1 - own_valid],
            # PadMemory
            [rel[0] / OFFSET_SCALE, rel[1] / OFFSET_SCALE, a.last_height_m / HEIGHT_SCALE,
             1 - min(a.last_bearing_fraction, 1.0), seen, f_pad, terminal_age],
            # RelativePosition
            [rel[0] / OFFSET_SCALE, rel[1] / OFFSET_SCALE, h / HEIGHT_SCALE,
             d_xy / OFFSET_SCALE, seen, f_pad,
             d_xy / max(c.descent_gate_width(h), 1e-6) if seen else 1.0],
            # RelativeVelocity
            [*(v_rel / SPEED_SCALE), np.hypot(v_rel[0], v_rel[1]) / SPEED_SCALE,
             v_rel_valid, f_pad,
             np.hypot(v_rel[0], v_rel[1]) / c.touchdown_xy_speed_m_s],
            # Visibility
            [margin_x, margin_y, predicted_margin, min(margin_x, margin_y, predicted_margin),
             float(obs.pad_detected), f_pad, visibility_urgency],
            # TerminalOcclusion
            [1 - min(a.last_height_m / c.terminal_entry_height_m, 1.0) if seen else 0.0,
             1 - min(a.last_offset_m / c.terminal_offset_m, 1.0) if seen else 0.0,
             slow_xy, a.terminal_score, own_valid, 1 - terminal_age, 1 - a.terminal_score],
            # TargetLost
            [float(bool(a.reason & 1)), float(bool(a.reason & 2)), float(bool(a.reason & 4)),
             a.lost_score, 1.0, 1.0, a.lost_score],
            # DescentReadiness
            [align, slow_xy, sink_ok, readiness, own_valid * seen, min(f_pad, f_own),
             1 - readiness],
            # LandingInhibit
            [stale, loc_invalid, a.lost_score, inhibit, 1.0, 1.0, inhibit],
            # TrackingBias
            [b[0], b[1], b[2], float(np.max(np.abs(b))), seen, f_pad,
             float(np.max(np.abs(b)))],
        ]
        count = len(rows)
        features = np.array(
            [[_clip(v) for v in row] + [(i + 1) / count] for i, row in enumerate(rows)],
            dtype=np.float32)

        terms = {"pad_freshness": f_pad if seen else 0.0,
                 "memory": (1 - f_pad) if seen else 0.0,
                 "terminal": a.terminal_score, "lost": a.lost_score}
        weights = np.ones(len(EDGES), dtype=np.float32)
        for edge, term in CONDITIONAL_EDGES.items():
            weights[EDGE_INDEX[edge]] = float(np.clip(terms[term], 0.0, 1.0))
        return GraphInstance(obs.stamp, features, weights, a, self.hash)
