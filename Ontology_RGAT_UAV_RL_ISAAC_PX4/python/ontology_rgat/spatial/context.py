"""Decode the spatial causal packet into one context plane per ENU axis.

The nine rows and twelve channels are NOT defined here -- they are
``landing.plane_graph``, the same definition the planar route builds its single
plane from. This module only decodes the packet and says which quantity belongs
to an axis and which is shared, so that 3D is literally 2D evaluated once more.

Both planes share conservative 3D descent evidence: a well-aligned x plane
cannot authorize descent while y is unsafe. Input is exclusively the common
normalized causal packet, never an estimator object, future samples, controller
commands, labels or simulator truth.
"""
import math

import numpy as np
from scipy.spatial.transform import Rotation

from ..landing.plane_graph import (AxisContext, SharedContext, build_plane,
                                   descent_eligibility)

#: Spatial attitude envelope used for the attitude risk channels.
TILT_LIMIT_RAD = math.radians(20)
TILT_RATE_LIMIT_RAD_S = math.radians(90)
HORIZONTAL_AXES = ("x", "y")


def _decode(values, cfg):
    """Undo the packet's smooth-signed normalization back to physical units."""
    from .core import FIELDS, SCALES, DIRECT

    p = dict(zip(cfg.packet_fields, np.asarray(values, dtype=float)))
    direct = set(DIRECT.tolist())
    for i, (name, scale) in enumerate(zip(FIELDS, SCALES)):
        if i not in direct:
            p[name] = scale * p[name] / max(1 - abs(p[name]), 1e-7)
    half = np.asarray(cfg.fov) / 2
    for name, scale in zip(("pred_bx", "pred_by", "pred_mx", "pred_my",
                            "meas_bx", "meas_by"), np.r_[half, half, half]):
        p[name] = scale * p[name] / max(1 - abs(p[name]), 1e-7)
    p["accstd"] /= max(1 - p["accstd"], 1e-7)
    return p, half


def _attitude(p):
    """ENU thrust-axis tilt per horizontal axis and its derivative."""
    q = np.array([p[n] for n in ("qx", "qy", "qz", "qw")])
    rotation = Rotation.from_quat(q)
    thrust_axis = rotation.apply([0.0, 0.0, 1.0])
    axis_derivative = np.cross(
        rotation.apply([p["wx"], p["wy"], p["wz"]]), thrust_axis)
    tilts = np.arctan2(thrust_axis[:2], thrust_axis[2])
    tilt_rates = (thrust_axis[2] * axis_derivative[:2]
                  - thrust_axis[:2] * axis_derivative[2]) / np.maximum(
                      thrust_axis[2] ** 2 + thrust_axis[:2] ** 2, 1e-9)
    return tilts, tilt_rates


def reference_context_graphs(values, cfg):
    """One 9x12 plane per ENU horizontal axis, stacked (len(axes), 9, 12)."""
    p, half = _decode(values, cfg)
    tilts, tilt_rates = _attitude(p)
    age = min(p["age"] / cfg.loss_timeout, 1.0)
    pos_u, vel_u, acc_u = (min(p[name] / scale, 1.0) for name, scale in
                           (("posstd", 5.0), ("velstd", 5.0), ("accstd", 3.0)))
    shared = SharedContext(
        height=p["rz"], descent_rate=p["vz"], detected=p["detected"],
        bearing_valid=p["optical_valid"], detection_confidence=p["confidence"],
        track_initialized=p["initialized"], age=age,
        position_uncertainty=pos_u, velocity_uncertainty=vel_u,
        acceleration_uncertainty=acc_u,
        landing_inhibited=p["inhibited"], abort_requested=p["abort"],
        remaining_time=p["remaining"],
        joint_position_risk=min(math.hypot(p["rx"], p["ry"]) / 3.0, 1.0),
        joint_speed_risk=min(math.hypot(p["rvx"], p["rvy"]) / 3.0, 1.0),
        tilt_limit=TILT_LIMIT_RAD, tilt_rate_limit=TILT_RATE_LIMIT_RAD_S,
        tilt_rate_scale=TILT_RATE_LIMIT_RAD_S)
    # Descent evidence is joined over both axes before either plane sees it.
    eligibility = descent_eligibility(
        shared,
        min(float(np.linalg.norm(tilts)) / TILT_LIMIT_RAD, 1.0),
        min(float(np.linalg.norm(tilt_rates)) / TILT_RATE_LIMIT_RAD_S, 1.0))
    planes = []
    for index, name in enumerate(HORIZONTAL_AXES):
        # Optical image-down is opposite ENU +y at held yaw zero.
        sign = 1.0 if index == 0 else -1.0
        planes.append(build_plane(AxisContext(
            pad_velocity=p["pv" + name], pad_acceleration=p["pa" + name],
            own_velocity=p["v" + name],
            # Upstream's convention is pad minus UAV; the packet stores
            # UAV minus pad.
            relative_position=-p["r" + name], relative_velocity=-p["rv" + name],
            measured_bearing=sign * p["meas_b" + name],
            predicted_bearing=sign * p["pred_b" + name],
            fov_margin=p["pred_m" + name], half_fov=half[index],
            tilt=tilts[index], tilt_rate=tilt_rates[index],
        ), shared, eligibility=eligibility))
    return np.asarray(planes, dtype=np.float32)


def axis_context_graphs(values, cfg):
    """One reference plane per ENU axis, from the axis-generic packet.

    The packet already carries ENU-signed bearings and pad-minus-UAV relative
    quantities, and ``landing.observation.decode`` returns physical units using
    the same scales the forward direction used -- so this is a naming step, not
    a second normalization. Contrast ``reference_context_graphs`` above, which
    had to re-derive the inverse of a parallel SCALES constant by hand.
    """
    import math

    from ..landing import observation as packet_io
    from ..landing.ontology import SPATIAL_EXTENSIONS
    from ..landing.packet import SPATIAL_AXES

    half = {axis: float(np.asarray(cfg.fov)[index] / 2)
            for index, axis in enumerate(SPATIAL_AXES)}
    p = packet_io.decode(values, SPATIAL_AXES, half_fov=half,
                         prolonged_loss_s=cfg.loss_timeout,
                         mission_limit_s=cfg.horizon, extras=True)
    tilts = [math.atan2(p[f"sinTilt_{axis}"], p[f"cosTilt_{axis}"])
             for axis in SPATIAL_AXES]
    rates = [p[f"tiltRate_{axis}"] for axis in SPATIAL_AXES]
    shared = SharedContext(
        height=p["h"], descent_rate=p["vz"], detected=p["detected"],
        bearing_valid=p["bearingValid"],
        detection_confidence=p["detectionConfidence"],
        track_initialized=p["trackInitialized"],
        age=min(p["timeSinceLastDetection"] / cfg.loss_timeout, 1.0),
        position_uncertainty=min(p["positionStd"] / 5.0, 1.0),
        velocity_uncertainty=min(p["velocityStd"] / 5.0, 1.0),
        acceleration_uncertainty=min(p["accelerationStd"] / 3.0, 1.0),
        landing_inhibited=p["landingInhibited"],
        abort_requested=p["abortRequested"],
        remaining_time=p["remainingMissionTime"],
        joint_position_risk=min(math.hypot(
            *(p[f"relativePosition_{a}"] for a in SPATIAL_AXES)) / 3.0, 1.0),
        joint_speed_risk=min(math.hypot(
            *(p[f"relativeVelocity_{a}"] for a in SPATIAL_AXES)) / 3.0, 1.0),
        tilt_limit=TILT_LIMIT_RAD, tilt_rate_limit=TILT_RATE_LIMIT_RAD_S,
        tilt_rate_scale=TILT_RATE_LIMIT_RAD_S,
        # Extension context: phenomena the planar task does not contain.
        disturbance_vertical=p["disturbanceEstimateZ"],
        transport_age=p["opticalTransportAge"],
        acceleration_limit=float(np.asarray(cfg.max_acceleration)[:2].max()),
        vertical_acceleration_limit=float(np.asarray(cfg.max_acceleration)[2]),
        policy_dt=cfg.dt)
    eligibility = descent_eligibility(
        shared,
        min(float(np.linalg.norm(tilts)) / TILT_LIMIT_RAD, 1.0),
        min(float(np.linalg.norm(rates)) / TILT_RATE_LIMIT_RAD_S, 1.0))
    planes = [build_plane(AxisContext(
        pad_velocity=p[f"padVelocity_{axis}"],
        pad_acceleration=p[f"padAcceleration_{axis}"],
        own_velocity=p[f"ownVelocity_{axis}"],
        relative_position=p[f"relativePosition_{axis}"],
        relative_velocity=p[f"relativeVelocity_{axis}"],
        measured_bearing=p[f"measuredBearing_{axis}"],
        predicted_bearing=p[f"predictedBearing_{axis}"],
        fov_margin=p[f"predictedFovMargin_{axis}"], half_fov=half[axis],
        tilt=tilts[index], tilt_rate=rates[index],
        disturbance=p[f"disturbanceEstimate_{axis}"],
        estimated_bearing=p[f"estimatedBearing_{axis}"],
    ), shared, eligibility=eligibility, extensions=SPATIAL_EXTENSIONS)
        for index, axis in enumerate(SPATIAL_AXES)]
    return np.asarray(planes, dtype=np.float32)
