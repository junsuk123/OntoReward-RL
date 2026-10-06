"""The one nine-node, twelve-channel context plane, built once per axis.

Upstream v2.8 defines the graph over a single horizontal axis
(``contextSchema.m`` / ``contextGraph.m`` at ``ugv_landing_2d_workspace@7efcc10``).
The 3D extension is that same plane evaluated once per ENU horizontal axis,
sharing the vertical, attitude, validity and authorization context -- not a
second graph definition.

It was a second definition until now: ``two_axis/ontology_v28.py`` and
``spatial/context.py`` each built the nine rows from their own packet, and a
change to the ontology in one dimension did not reach the other. The two were
the same function up to exactly two things, both parameterized here:

* rows 4 and 5 use the risk of **this** axis, while rows 7 and 8 use the risk
  joined over all axes. With one axis the two coincide, which is why the 2D
  implementation could use a single scalar and never notice the distinction.
* the attitude-rate channel normalizes by a fixed scale while the attitude-rate
  *risk* normalizes by the dynamics limit. They are equal in the 3D plant and
  different in the 2D one.

Inputs are decoded physical quantities taken from the causal packet. No
simulator truth, action, outcome or label reaches this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

#: Channel order is the contract; it is what the typed relation kernels index.
FEATURE_CHANNELS = ("primary", "signedPrimary", "secondary", "signedSecondary",
                    "validity", "confidence", "uncertainty", "trend", "urgency",
                    "remainingTime", "bias", "typeId")
NODE_COUNT = 9


@dataclass(frozen=True)
class AxisContext:
    """Quantities belonging to one horizontal axis, in physical units."""

    pad_velocity: float            # pad speed along this axis (m/s)
    pad_acceleration: float        # pad acceleration along this axis (m/s^2)
    own_velocity: float            # own speed along this axis (m/s)
    relative_position: float       # pad minus UAV along this axis (m)
    relative_velocity: float       # pad minus UAV rate along this axis (m/s)
    measured_bearing: float        # last measured bearing (rad)
    predicted_bearing: float       # predicted bearing at the horizon (rad)
    fov_margin: float              # predicted margin to the frame edge (rad)
    half_fov: float                # half field of view on this axis (rad)
    tilt: float                    # tilt toward this axis (rad)
    tilt_rate: float               # tilt rate about this axis (rad/s)
    # Extension inputs; zero where the task has no such phenomenon.
    disturbance: float = 0.0       # unmodelled acceleration on this axis (m/s^2)
    estimated_bearing: float = 0.0 # bearing from the current estimate (rad)


@dataclass(frozen=True)
class SharedContext:
    """Quantities every axis plane shares: vertical, validity, authorization."""

    height: float                  # height above the pad (m)
    descent_rate: float            # vertical rate (m/s), negative is descending
    detected: float
    bearing_valid: float
    detection_confidence: float    # raw, before the track-initialized gate
    track_initialized: float
    age: float                     # normalized time since the last detection
    position_uncertainty: float
    velocity_uncertainty: float
    acceleration_uncertainty: float
    landing_inhibited: float
    abort_requested: float
    remaining_time: float
    joint_position_risk: float     # joined over every horizontal axis
    joint_speed_risk: float
    tilt_limit: float              # attitude limit used for the risk channel
    tilt_rate_limit: float         # attitude-rate limit used for the risk channel
    tilt_rate_scale: float         # normalization of the signed rate channel
    # Extension inputs; zero where the task has no such phenomenon.
    disturbance_vertical: float = 0.0
    transport_age: float = 0.0     # capture to use, in seconds
    acceleration_limit: float = 1.0
    vertical_acceleration_limit: float = 1.0
    policy_dt: float = 0.1


def descent_eligibility(shared: SharedContext, attitude_risk: float,
                        rate_risk: float) -> float:
    """Conservative authorization evidence, joined over all horizontal axes.

    A well-aligned x plane must not authorize descent while y is unsafe, so the
    risks entering this product are the joined ones.
    """
    confidence = shared.detection_confidence * shared.track_initialized
    return (confidence * (1 - shared.joint_position_risk)
            * (1 - shared.joint_speed_risk) * (1 - attitude_risk)
            * (1 - rate_risk) * (1 - shared.landing_inhibited))


def _disturbance_row(axis, shared, acc_u):
    """Unmodelled acceleration this axis carries, and what it costs in authority."""
    d = axis.disturbance / max(shared.acceleration_limit, 1e-9)
    dz = shared.disturbance_vertical / max(shared.vertical_acceleration_limit, 1e-9)
    d, dz = float(np.clip(d, -1, 1)), float(np.clip(dz, -1, 1))
    return [abs(d), d, abs(dz), dz, 1.0, 1 - acc_u, acc_u, d, abs(d)]


def _latency_row(axis, shared, pos_u):
    """How far the view has moved since the solve the estimate rests on."""
    half = max(axis.half_fov, 1e-9)
    age = min(shared.transport_age / max(shared.policy_dt, 1e-9), 1.0)
    discrepancy = float(np.clip(
        (axis.estimated_bearing - axis.measured_bearing) / half, -1, 1))
    return [age, discrepancy, abs(discrepancy), axis.measured_bearing / half,
            shared.bearing_valid, shared.detection_confidence, pos_u,
            discrepancy, max(age, abs(discrepancy))]


EXTENSION_ROWS = {
    "DisturbanceEstimate": _disturbance_row,
    "MeasurementLatency": _latency_row,
}


def build_plane(axis: AxisContext, shared: SharedContext, *,
                eligibility: float, extensions=()) -> np.ndarray:
    """One context plane for one horizontal axis, nine rows plus extensions."""
    ex, rv = axis.relative_position, axis.relative_velocity
    pv, pa = axis.pad_velocity, axis.pad_acceleration
    half = axis.half_fov
    measured, predicted, margin = (axis.measured_bearing, axis.predicted_bearing,
                                   axis.fov_margin)
    urgency = max(0.0, -margin / half)
    age = shared.age
    pos_u, vel_u, acc_u = (shared.position_uncertainty,
                           shared.velocity_uncertainty,
                           shared.acceleration_uncertainty)
    confidence = shared.detection_confidence * shared.track_initialized
    # This axis' own risk, which is what the per-axis tracking and correction
    # nodes describe. The authorization nodes below use the joined risk.
    axis_position_risk = min(abs(ex) / 3.0, 1.0)
    axis_speed_risk = min(abs(rv) / 3.0, 1.0)
    attitude_risk = min(abs(axis.tilt) / shared.tilt_limit, 1.0)
    rate_risk = min(abs(axis.tilt_rate) / shared.tilt_rate_limit, 1.0)
    recovery = min(1.0, max(1.0 - shared.detected, age, urgency))
    inhibit = max(shared.landing_inhibited, shared.abort_requested, age,
                  pos_u, vel_u)
    correction = math.tanh(ex / 3.0 + 0.5 * rv / 3.0)
    signed_rate = axis.tilt_rate / shared.tilt_rate_scale
    vz, h, vx = shared.descent_rate, shared.height, axis.own_velocity
    rows = [
        # PadVisibility
        [shared.detected, measured, predicted, margin, shared.bearing_valid,
         shared.detection_confidence, age, predicted, urgency],
        # PadMotion
        [min(abs(pv) / 10, 1), pv / 10, min(abs(pa) / 2, 1), pa / 2,
         shared.track_initialized, confidence, max(vel_u, acc_u), pa / 2, acc_u],
        # DroneTranslation
        [min(h / 8, 1), vz / 1.5, min(abs(vx) / 10, 1), vx / 10, 1, 1, 0,
         vz / 1.5, 0],
        # DroneAttitude
        [attitude_risk, axis.tilt / shared.tilt_limit, rate_risk, signed_rate,
         1, 1, 0, signed_rate, attitude_risk],
        # RelativeTracking
        [axis_position_risk, ex / 3, axis_speed_risk, rv / 3,
         shared.track_initialized, confidence, max(pos_u, vel_u), rv / 3, urgency],
        # TrackingCorrection
        [abs(correction), correction, axis_speed_risk, -rv / 3,
         shared.track_initialized, confidence, max(pos_u, vel_u), pa / 2,
         axis_position_risk],
        # ViewRecovery
        [recovery, -np.sign(predicted) * recovery, urgency, margin / half,
         shared.track_initialized, confidence, max(age, pos_u), predicted / half,
         recovery],
        # DescentEligibility -- joined risk, never this axis alone
        [eligibility, eligibility, 1 - shared.joint_speed_risk,
         -shared.joint_speed_risk, shared.track_initialized, confidence,
         max(pos_u, vel_u, age), vz / 1.5, 1 - eligibility],
        # LandingInhibit
        [inhibit, shared.abort_requested, age, shared.landing_inhibited, 1, 1,
         max(pos_u, vel_u, age), shared.abort_requested, inhibit],
    ]
    for name in extensions:
        if name not in EXTENSION_ROWS:
            raise ValueError(f"no context row defined for extension {name!r}")
        rows.append(EXTENSION_ROWS[name](
            axis, shared, acc_u if name == "DisturbanceEstimate" else pos_u))
    count = len(rows)
    plane = np.array(
        [row + [shared.remaining_time, 1.0, (index + 1) / count]
         for index, row in enumerate(rows)], dtype=np.float32)
    return np.clip(plane, -1.0, 1.0)
