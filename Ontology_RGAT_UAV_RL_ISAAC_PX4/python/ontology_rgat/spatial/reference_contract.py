"""Build the axis-generic causal packet from the spatial estimator.

This is the whole 3D side of the dimensional extension: name the quantity each
declared field wants, per ENU horizontal axis, and hand it to
``landing.observation.normalize``. The field list, its order, its scales and
its registry digest are not decided here -- they are derived from the 2D
reference in ``landing.packet``.

What this replaces: a 35-field positional array with its own scale vector,
extended four times (39/43/47) by appending whatever the previous rung turned
out to lack, with the normalization open-coded against a parallel ``SCALES``
constant and inverted again, differently, in the graph builder.

Everything here comes from ``Estimator`` and the safety decision. No simulator
truth, contact state, outcome or scenario parameter is reachable.
"""
from __future__ import annotations

import math

import numpy as np

from ..landing import observation as packet_io
from ..landing.packet import SPATIAL_AXES
from .context import _attitude  # quaternion -> per-axis thrust-axis tilt

#: Optical image-down is opposite ENU +y at the held yaw. Applying the sign
#: once, here, keeps every packet field in ENU; the graph then needs no
#: per-axis sign convention of its own.
ENU_OPTICAL_SIGN = (1.0, -1.0)
PREDICTION_HORIZON_S = 0.5


def _bearings(relative, quaternion, cfg):
    from .core import camera_bearings

    bearing, depth = camera_bearings(relative, quaternion, cfg)
    return np.asarray(bearing), float(depth)


def raw_fields(est, safety, elapsed, cfg) -> dict[str, float]:
    """Physical quantities for every declared field, keyed by field name."""
    from scipy.spatial.transform import Rotation

    m = est.own
    half = np.asarray(cfg.fov) / 2
    sign = np.asarray(ENU_OPTICAL_SIGN)

    estimated, _ = _bearings(est.r, m.quaternion, cfg)
    # Causal 0.5 s projection, exactly the reference's predicted view.
    predicted_r = (est.r + PREDICTION_HORIZON_S * est.rv
                   - 0.5 * PREDICTION_HORIZON_S ** 2 * est.pad_a)
    rotation = Rotation.from_quat(m.quaternion[[1, 2, 3, 0]])
    future_q = (rotation * Rotation.from_rotvec(
        PREDICTION_HORIZON_S * m.angular_rate)).as_quat()
    predicted, depth = _bearings(predicted_r, future_q[[3, 0, 1, 2]], cfg)
    if depth > 0:
        margin = half - np.abs(predicted)
    else:
        # Behind the camera: no predicted view, and the margin is the full
        # half-angle in deficit rather than a spurious centred bearing.
        predicted, margin = np.zeros(2), -half
    if not est.initialized:
        predicted, margin = np.zeros(2), np.zeros(2)

    measured = np.zeros(2)
    valid = m.optical_position is not None
    if valid:
        measured, _ = _bearings(
            m.optical_position,
            m.quaternion if m.optical_quaternion is None else m.optical_quaternion,
            cfg)
    transport_age = 0.0
    if m.optical_time_s is not None:
        transport_age = max(0.0, m.time_s - m.optical_time_s)

    tilts, tilt_rates = _attitude({
        "qx": m.quaternion[1], "qy": m.quaternion[2], "qz": m.quaternion[3],
        "qw": m.quaternion[0], "wx": m.angular_rate[0], "wy": m.angular_rate[1],
        "wz": m.angular_rate[2]})

    raw = {
        "h": float(est.r[2]),
        "vz": float(m.own_velocity[2]),
        "positionStd": float(est.std),
        "velocityStd": float(est.velocity_std),
        "accelerationStd": float(getattr(est, "acceleration_std", 0.0)),
        "trackInitialized": float(est.initialized),
        "detected": float(est.detected),
        "bearingValid": float(valid),
        "detectionConfidence": float(m.confidence),
        "timeSinceLastDetection": float(min(est.age, 1000.0)),
        "remainingMissionTime": float(max(0.0, cfg.horizon - elapsed)),
        "previousNormalizedActionZ": float(np.clip(est.previous_action[2], -1, 1)),
        "landingInhibited": float(safety.inhibited),
        "abortRequested": float(safety.abort),
        "opticalTransportAge": transport_age,
    }
    for index, axis in enumerate(SPATIAL_AXES):
        raw.update({
            f"ownVelocity_{axis}": float(m.own_velocity[index]),
            f"sinTilt_{axis}": math.sin(float(tilts[index])),
            f"cosTilt_{axis}": math.cos(float(tilts[index])),
            f"tiltRate_{axis}": float(tilt_rates[index]),
            # Upstream's convention is pad minus UAV; the estimator holds
            # UAV minus pad.
            f"relativePosition_{axis}": -float(est.r[index]),
            f"relativeVelocity_{axis}": -float(est.rv[index]),
            f"padVelocity_{axis}": float(est.pad_v[index]),
            f"padAcceleration_{axis}": float(est.pad_a[index]),
            f"measuredBearing_{axis}": float(sign[index] * measured[index]),
            f"predictedBearing_{axis}": float(sign[index] * predicted[index]),
            f"predictedFovMargin_{axis}": float(margin[index]),
            f"previousAction_{axis}": float(
                np.clip(est.previous_action[index], -1, 1)),
            f"estimatedBearing_{axis}": float(sign[index] * estimated[index]),
        })
    return raw


def packet_values(est, safety, elapsed, cfg) -> np.ndarray:
    """The normalized causal packet, in the declared axis-generic order."""
    half = {axis: float(np.asarray(cfg.fov)[index] / 2)
            for index, axis in enumerate(SPATIAL_AXES)}
    return packet_io.normalize(
        raw_fields(est, safety, elapsed, cfg), SPATIAL_AXES, half_fov=half,
        prolonged_loss_s=cfg.loss_timeout, mission_limit_s=cfg.horizon,
        extras=True)
