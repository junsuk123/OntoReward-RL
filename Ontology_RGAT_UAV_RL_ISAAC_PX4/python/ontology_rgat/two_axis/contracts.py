"""Causal packet construction and strict experiment signatures."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping

import numpy as np

from ..contracts.observation import (CausalObservationPacket,
                                     ObservationRegistry,
                                     load_observation_registry)
from ..contracts.signature import CheckpointSignature
from .config import DEFAULT_REGISTRY_PATH, ExperimentConfig
from .dynamics import PlanarState
from .estimation import TrackEstimate
from .sensing import PadMeasurement


def load_v2_registry() -> ObservationRegistry:
    return load_observation_registry(DEFAULT_REGISTRY_PATH)


def _bounded(value: float) -> float:
    value = max(0.0, float(value))
    return value / (1.0 + value)


def make_causal_packet(*, state: PlanarState, track: TrackEstimate,
                       measurement: PadMeasurement, previous_action: np.ndarray,
                       landing_inhibited: bool, abort_requested: bool,
                       mission_deadline_s: float, fov_rad: float,
                       registry: ObservationRegistry | None = None
                       ) -> CausalObservationPacket:
    """Build policy input without accepting current pad truth as an argument."""
    registry = registry or load_v2_registry()
    previous = np.asarray(previous_action, dtype=float).reshape(2)
    ex_est = track.pad_x_m - state.x_m if track.initialized else 0.0
    rel_v_est = track.pad_vx_m_s - state.vx_m_s if track.initialized else 0.0
    h = max(0.0, state.z_m)
    depth = h * math.cos(state.theta_rad) - ex_est * math.sin(state.theta_rad)
    lateral = ex_est * math.cos(state.theta_rad) + h * math.sin(state.theta_rad)
    predicted_bearing = math.atan2(lateral, depth) if depth > 0.0 else 0.0
    half_fov = 0.5 * float(fov_rad)
    predicted_margin = half_fov - abs(predicted_bearing) if depth > 0.0 else -half_fov
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
    registry = registry or load_v2_registry()
    return CheckpointSignature(
        algorithm_version=config.algorithm_version,
        environment_config_hash=config.sha256,
        observation_registry_hash=registry.sha256,
        reward_config_hash=_digest(config.reward),
        action_contract_hash=_digest({"dimension": 2, "channels": (
            "longitudinal_acceleration", "vertical_acceleration"),
            "limits": (config.dynamics.ax_max_m_s2, config.dynamics.az_max_m_s2)}),
        graph_schema_hash=graph_schema_hash,
        pretrained_artifact_hash="none-end-to-end-ppo",
        relation_partition_hash=graph_schema_hash,
        normalization_hash=registry.sha256,
        seed_contract_hash=_digest("independent-scenario-sensor-policy-v1"),
        training_budget_hash=_digest("caller-declared-environment-steps"))
