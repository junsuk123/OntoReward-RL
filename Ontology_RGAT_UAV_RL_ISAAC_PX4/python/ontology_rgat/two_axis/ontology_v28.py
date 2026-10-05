"""Nine-node/12-channel causal graph ported from upstream v2.8.

Source: contextSchema.m/contextGraph.m at 7efcc10995557bc8435304216ad2ced70ab1da00.
The Python packet is normalized already; decode its units before applying the
reference equations. No simulator truth, labels, actions or outcomes enter it.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import numpy as np

from ..landing.plane_graph import (AxisContext, SharedContext, build_plane,
                                   descent_eligibility)
from .contracts import packet_fields, REFERENCE_SCALES
from .ontology import DECLARED_EDGES, SEMANTIC_NODES

SCHEMA_VERSION = "compact_context_graph_v3_grouped"
NODE_NAMES = SEMANTIC_NODES
RELATION_NAMES = ("informs", "affects_visibility", "supports", "inhibits", "self")
FEATURE_CHANNELS = ("primary", "signedPrimary", "secondary", "signedSecondary",
                    "validity", "confidence", "uncertainty", "trend", "urgency",
                    "remainingTime", "bias", "typeId")
GRAPH_EDGES = DECLARED_EDGES + tuple((node, "self", node) for node in NODE_NAMES)
READOUT_GROUPS = ((0, 6), (1, 4, 5), (2, 3), (7, 8))
GRAPH_SCHEMA_HASH = hashlib.sha256(json.dumps({
    "version": SCHEMA_VERSION, "nodes": NODE_NAMES, "relations": RELATION_NAMES,
    "edges": GRAPH_EDGES, "features": FEATURE_CHANNELS, "groups": READOUT_GROUPS,
    "packet_decoding": "python-causal-v3-smooth-v28",
}, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class GroupedContextGraph:
    X: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    rel: np.ndarray
    packet_registry_hash: str
    schema_hash: str = GRAPH_SCHEMA_HASH
    node_names: tuple[str, ...] = NODE_NAMES
    relation_names: tuple[str, ...] = RELATION_NAMES

    def __post_init__(self):
        if self.X.shape != (9, 12) or not np.isfinite(self.X).all():
            raise ValueError("v2.8 graph requires finite [9, 12] features")


def build_context_graph(packet, registry, config) -> GroupedContextGraph:
    p = packet_fields(packet, registry)
    if registry.schema == "ontology_rgat.planar_causal_packet/3-v28":
        raw = {}
        for name,scale in REFERENCE_SCALES.items():
            scale = (config.camera.fov_rad/2 if scale == "half_fov" else
                     config.estimator.prolonged_loss_s if scale == "prolonged_loss" else scale)
            raw[name] = float(scale)*p[name]/max(1-abs(p[name]),1e-9)
        linear_scales = {"h":8., "vx":4., "vz":3., "pitchRate":math.pi/2,
            "exEstimate":12., "relativeVxEstimate":4., "padVxEstimate":4., "padAxEstimate":2.,
            "measuredBearing":config.camera.fov_rad/2, "predictedBearing":config.camera.fov_rad/2,
            "predictedFovMargin":config.camera.fov_rad/2, "timeSinceLastDetection":3.}
        for name,scale in linear_scales.items():
            p[name] = raw[name]/scale
        for name in ("positionStd","velocityStd","accelerationStd"):
            p[name] = raw[name]/(1+raw[name])
    # The registry uses s/(1+s) for uncertainty, not physical standard deviation.
    def std(name, scale):
        u = min(max(p[name], 0.0), 1.0 - 1e-9)
        return min(u / (1.0 - u) / scale, 1.0)
    pos_u, vel_u, acc_u = std("positionStd", 5), std("velocityStd", 5), std("accelerationStd", 3)
    age = min(p["timeSinceLastDetection"] * 3 / config.estimator.prolonged_loss_s, 1)
    theta = math.atan2(p["sinTheta"], p["cosTheta"])
    ex, rv = p["exEstimate"] * 12, p["relativeVxEstimate"] * 4
    vx, vz, h = p["vx"] * 4, p["vz"] * 3, p["h"] * 8
    pv, pa = p["padVxEstimate"] * 4, p["padAxEstimate"] * 2
    half_fov = config.camera.fov_rad / 2
    # The planar route has exactly one horizontal axis, so its joined risk and
    # its per-axis risk are the same number. Keeping them distinct is what lets
    # the 3D route reuse this definition unchanged.
    pitch_rate = p["pitchRate"] * math.radians(90)
    axis = AxisContext(
        pad_velocity=pv, pad_acceleration=pa, own_velocity=vx,
        relative_position=ex, relative_velocity=rv,
        measured_bearing=p["measuredBearing"] * half_fov,
        predicted_bearing=p["predictedBearing"] * half_fov,
        fov_margin=p["predictedFovMargin"] * half_fov, half_fov=half_fov,
        tilt=theta, tilt_rate=pitch_rate)
    shared = SharedContext(
        height=h, descent_rate=vz, detected=p["detected"],
        bearing_valid=p["bearingValid"],
        detection_confidence=p["detectionConfidence"],
        track_initialized=p["trackInitialized"], age=age,
        position_uncertainty=pos_u, velocity_uncertainty=vel_u,
        acceleration_uncertainty=acc_u,
        landing_inhibited=p["landingInhibited"],
        abort_requested=p["abortRequested"],
        remaining_time=p["remainingMissionTime"],
        joint_position_risk=min(abs(ex) / 3, 1),
        joint_speed_risk=min(abs(rv) / 3, 1),
        tilt_limit=config.dynamics.pitch_limit_rad,
        tilt_rate_limit=config.dynamics.pitch_rate_limit_rad_s,
        tilt_rate_scale=math.radians(90))
    attitude = min(abs(theta) / shared.tilt_limit, 1)
    rate = min(abs(pitch_rate) / shared.tilt_rate_limit, 1)
    X = build_plane(axis, shared,
                    eligibility=descent_eligibility(shared, attitude, rate))
    indices = {name: i for i,name in enumerate(NODE_NAMES)}
    return GroupedContextGraph(X,
        np.array([indices[s] for s,_,_ in GRAPH_EDGES]),
        np.array([indices[d] for _,_,d in GRAPH_EDGES]),
        np.array([RELATION_NAMES.index(r) for _,r,_ in GRAPH_EDGES]), packet.registry_sha256)


def schema_hash(config) -> str:
    if config.ontology.schema == SCHEMA_VERSION:
        return GRAPH_SCHEMA_HASH
    from .ontology import GRAPH_SCHEMA_HASH as legacy_hash
    return legacy_hash
