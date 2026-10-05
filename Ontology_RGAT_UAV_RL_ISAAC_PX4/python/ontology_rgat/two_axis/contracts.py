"""Causal packet construction and strict experiment signatures."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping
from functools import lru_cache

import numpy as np

from ..contracts.observation import (CausalObservationPacket,
                                     ObservationRegistry,
                                     load_observation_registry)
from ..contracts.signature import CheckpointSignature
from ..landing.packet import (MISSION_LIMIT as _MISSION_LIMIT,
                              PLANAR_AXES as _PLANAR_AXES, UNSCALED as _UNSCALED,
                              field_specs as _field_specs)
from .config import DEFAULT_REGISTRY_PATH, ExperimentConfig
from .dynamics import PlanarState
from .estimation import TrackEstimate
from .sensing import PadMeasurement


def load_v2_registry() -> ObservationRegistry:
    return load_observation_registry(DEFAULT_REGISTRY_PATH)


#: Derived from the axis-generic field spec rather than restated, so the
#: forward normalization here, its inverse in ontology_v28.build_context_graph
#: and the 3D packet cannot disagree about a scale. The spec's 1-axis names are
#: upstream's own, which is why this is a straight projection.
#: remainingMissionTime is clipped against the mission limit rather than
#: smooth-signed, so it is excluded here and handled by its caller.
REFERENCE_SCALES = {
    name: spec.scale for name, spec, _ in _field_specs(_PLANAR_AXES)
    if spec.scale not in _UNSCALED and spec.scale != _MISSION_LIMIT
}


def normalize_reference_fields(raw, *, half_fov, prolonged_loss_s=3., mission_limit_s=70.):
    values = dict(raw)
    for name, scale in REFERENCE_SCALES.items():
        scale = half_fov if scale == "half_fov" else prolonged_loss_s if scale == "prolonged_loss" else scale
        values[name] = float(raw[name])/(abs(float(raw[name]))+float(scale))
    if "remainingMissionTime" in values:
        values["remainingMissionTime"] = float(np.clip(raw["remainingMissionTime"]/mission_limit_s,0,1))
    return values


@lru_cache(maxsize=1)
def load_reference_registry() -> ObservationRegistry:
    """Same causal provenance/order, separately hashed v2.8 normalization."""
    base = load_v2_registry()
    fields = []
    for field in base.fields:
        value = dict(field)
        name = value["name"]
        if name in REFERENCE_SCALES:
            value["normalization"] = f"signed_x_over_abs_x_plus_{REFERENCE_SCALES[name]}"
        elif name == "remainingMissionTime":
            value["normalization"] = "remaining_seconds_over_mission_limit"
        fields.append(value)
    payload = {"schema": "ontology_rgat.planar_causal_packet/3-v28",
        "clock": base.clock, "fields": fields, "forbidden_sources": base.forbidden_sources}
    digest = hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",", ":")).encode()).hexdigest()
    return ObservationRegistry(payload["schema"], base.clock, tuple(fields),base.forbidden_sources,digest)


def _bounded(value: float) -> float:
    value = max(0.0, float(value))
    return value / (1.0 + value)


def make_causal_packet(*, state: PlanarState, track: TrackEstimate,
                       measurement: PadMeasurement, previous_action: np.ndarray,
                       landing_inhibited: bool, abort_requested: bool,
                       mission_deadline_s: float, fov_rad: float,
                       prediction_horizon_s: float = 0.0,
                       reference_normalization: bool = False,
                       mission_limit_s: float = 70.0, prolonged_loss_s: float = 3.0,
                       registry: ObservationRegistry | None = None
                       ) -> CausalObservationPacket:
    """Build policy input without accepting current pad truth as an argument."""
    registry = registry or (load_reference_registry() if reference_normalization else load_v2_registry())
    previous = np.asarray(previous_action, dtype=float).reshape(2)
    ex_est = track.pad_x_m - state.x_m if track.initialized else 0.0
    rel_v_est = track.pad_vx_m_s - state.vx_m_s if track.initialized else 0.0
    h = max(0.0, state.z_m)
    future_ex = ex_est + rel_v_est*prediction_horizon_s + .5*track.pad_ax_m_s2*prediction_horizon_s**2
    future_theta = state.theta_rad + state.pitch_rate_rad_s*prediction_horizon_s
    depth = h * math.cos(future_theta) - future_ex * math.sin(future_theta)
    lateral = future_ex * math.cos(future_theta) + h * math.sin(future_theta)
    predicted_bearing = math.atan2(lateral, depth) if depth > 0.0 else 0.0
    half_fov = 0.5 * float(fov_rad)
    predicted_margin = half_fov - abs(predicted_bearing) if depth > 0.0 else -half_fov
    if not track.initialized:
        predicted_bearing = predicted_margin = 0.0
    age = track.time_since_detection_s
    if not np.isfinite(age):
        age = mission_deadline_s
    fields: Mapping[str, object] = {
        "h": [h / 8.0], "vx": [state.vx_m_s / 4.0],
        "vz": [state.vz_m_s / 3.0], "sinTheta": [math.sin(state.theta_rad)],
        "cosTheta": [math.cos(state.theta_rad)],
        "pitchRate": [state.pitch_rate_rad_s / math.radians(90.0)],
        "exEstimate": [ex_est / 12.0],
        "relativeVxEstimate": [rel_v_est / 4.0],
        "padVxEstimate": [track.pad_vx_m_s / 4.0 if track.initialized else 0.0],
        "padAxEstimate": [track.pad_ax_m_s2 / 2.0 if track.initialized else 0.0],
        "positionStd": [_bounded(track.position_std_m)],
        "velocityStd": [_bounded(track.velocity_std_m_s)],
        "accelerationStd": [_bounded(track.acceleration_std_m_s2)],
        "trackInitialized": [float(track.initialized)],
        "detected": [float(measurement.detected)],
        "measuredBearing": [measurement.bearing_rad / half_fov
                            if measurement.bearing_valid else 0.0],
        "bearingValid": [float(measurement.bearing_valid)],
        "detectionConfidence": [measurement.confidence],
        "timeSinceLastDetection": [min(age / 3.0, 2.0)],
        "predictedBearing": [predicted_bearing / half_fov],
        "predictedFovMargin": [predicted_margin / half_fov],
        "remainingMissionTime": [max(0.0, mission_deadline_s - state.time_s)
                                 / mission_deadline_s],
        "previousNormalizedActionX": [float(np.clip(previous[0], -1.0, 1.0))],
        "previousNormalizedActionZ": [float(np.clip(previous[1], -1.0, 1.0))],
        "landingInhibited": [float(landing_inhibited)],
        "abortRequested": [float(abort_requested)],
    }
    if reference_normalization:
        raw = {"h": h, "vx": state.vx_m_s, "vz": state.vz_m_s,
            "pitchRate": state.pitch_rate_rad_s, "exEstimate": ex_est,
            "relativeVxEstimate": rel_v_est, "padVxEstimate": track.pad_vx_m_s,
            "padAxEstimate": track.pad_ax_m_s2, "positionStd": track.position_std_m,
            "velocityStd": track.velocity_std_m_s, "accelerationStd": track.acceleration_std_m_s2,
            "measuredBearing": measurement.bearing_rad if measurement.bearing_valid else 0.,
            "predictedBearing": predicted_bearing, "predictedFovMargin": predicted_margin,
            "timeSinceLastDetection": age}
        for name,value in normalize_reference_fields(raw,half_fov=half_fov,
                prolonged_loss_s=prolonged_loss_s,mission_limit_s=mission_limit_s).items():
            fields[name] = [value]
        fields["remainingMissionTime"] = [max(0.,mission_deadline_s-state.time_s)/mission_limit_s]
    return CausalObservationPacket.from_fields(
        fields, timestamp_s=state.time_s, registry=registry)


def packet_fields(packet: CausalObservationPacket,
                  registry: ObservationRegistry | None = None) -> dict[str, float]:
    registry = registry or load_v2_registry()
    packet.assert_registry(registry)
    result: dict[str, float] = {}
    cursor = 0
    for field in registry.fields:
        size = int(field["size"])
        values = packet.values[cursor:cursor + size]
        if size != 1:
            raise ValueError("v2 packet access expects scalar registry fields")
        result[str(field["name"])] = float(values[0])
        cursor += size
    return result


def _digest(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def experiment_signature(config: ExperimentConfig, *, graph_schema_hash: str,
                         registry: ObservationRegistry | None = None
                         ) -> CheckpointSignature:
    registry = registry or (load_reference_registry()
        if config.ontology.schema == "compact_context_graph_v3_grouped" else load_v2_registry())
    return CheckpointSignature(
        algorithm_version=config.algorithm_version,
        environment_config_hash=config.sha256,
        observation_registry_hash=registry.sha256,
        reward_config_hash=_digest(config.reward),
        action_contract_hash=_digest({"dimension": 2, "channels": (
            "longitudinal_acceleration", "vertical_acceleration"),
            "limits": (config.dynamics.ax_max_m_s2, config.dynamics.az_max_m_s2)}),
        graph_schema_hash=graph_schema_hash,
        pretrained_artifact_hash=(_digest({"regime": "train-only-masked-same-time-v28",
            "episodes": config.ontology.pretrain_episodes,
            "epochs": config.ontology.pretrain_epochs}) if config.ontology.pretrain_episodes
            else "none-end-to-end-ppo"),
        relation_partition_hash=graph_schema_hash,
        normalization_hash=registry.sha256,
        seed_contract_hash=_digest("independent-scenario-sensor-policy-time-paired-v2"),
        training_budget_hash=_digest("caller-declared-environment-steps"))
