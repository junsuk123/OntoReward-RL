"""Proportional-navigation guidance in the sagittal plane, from the image alone.

This is the non-learned control condition of the comparison. It replaces the
image-based visual servo for one reason: the reduced 2-D study the new
methodology comes from uses PN guidance as its reference law, and a comparison
whose control condition is a different law is not the same comparison.

Information boundary
--------------------
The only input is a :class:`~ontology_rgat.perception.semantic_observation.SemanticObservation`
-- the confidence-weighted keypoint centroid, its apparent RMS scale and the
visibility scalars, all from the same frozen encoder the learned actors see --
plus the vehicle's own body velocity, which is part of the actor's 7-D
proprioception. No relative position, no relative velocity, no deck twist, no
simulator truth. A difference between this arm and a learned arm is therefore a
difference in *control*, not in observation.

The law
-------
In the reduced envelope the task is planar, so PN needs one line of sight. The
column axis of the image gives the bearing from nadir,

    e      = (centroid_col - nadir_col) * tan(hfov/2)      [= d / h]
    lambda = atan(e)

and apparent scale gives the range, the same monotone inverse-range signal the
retired servo used:

    h = clip(reference_scale / raw_scale, ...)   (metres of altitude)
    R = h * sqrt(1 + e^2)

With ``lambda_dot`` and ``R_dot`` filtered from those, the commanded
acceleration is the textbook pair -- PN across the line of sight, closing-speed
regulation along it:

    a_perp = N * V_c * lambda_dot,            V_c = -R_dot
    a_los  = -k_close * (v_ref + R_dot),      v_ref = min(v_app, k_app * R)

and when the pad is not trustworthy the law switches to the reduced study's
search branch: climb at a reference rate until it is re-acquired, and stop
climbing once the vehicle is committed to the flare (where the pad legitimately
fills and leaves a downward camera).

The result is decomposed onto the sagittal axes and emitted as the same
three-channel planar action every other arm uses. Its tilt channel carries the
tilt its own longitudinal acceleration implies, ``atan(a_fwd / g)``, so the
control condition exercises the same three channels rather than leaving one
silent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .planar_controller import PLANAR_ACTION_DIM, STANDARD_GRAVITY


__all__ = ["PNGuidanceConfig", "PNGuidanceState", "PNGuidanceController",
           "PN_GUIDANCE_METHOD"]


# The arm id this controller is registered under.
PN_GUIDANCE_METHOD = "pn_guidance_v1"


@dataclass(frozen=True)
class PNGuidanceConfig:
    """Gains, transcribed from the reduced study and re-based on this camera.

    The reduced study works in metres directly; here ``reference_scale`` is the
    apparent RMS keypoint scale at one metre of altitude, which is what turns
    an image bearing into metres. ``range_gain_bounds`` keeps a collapsed or
    saturated scale from producing an unbounded range.
    """

    navigation_gain: float = 3.0            # N
    approach_speed_m_s: float = 0.45        # v_app
    approach_gain: float = 0.60             # k_app  [1/s]
    closing_gain: float = 1.20              # k_close [1/s]
    vertical_gain: float = 2.00             # search-branch climb tracking [1/s]
    climb_reference_m_s: float = 0.35
    minimum_closing_speed_m_s: float = 0.15  # floor on V_c in the PN term
    minimum_range_m: float = 0.15
    # Image-plane geometry.
    reference_scale: float = 0.06
    range_gain_bounds: tuple[float, float] = (0.30, 12.0)
    rate_filter_s: float = 0.30
    # Phase logic, shared in spirit with the retired servo.
    flare_scale: float = 0.20               # commit threshold on apparent scale
    trustworthy_visible_fraction: float = 0.5
    # Descent authority near the deck, so the flare is a landing and not a hover.
    flare_descent_m_s: float = 0.25
    descent_floor_m_s: float = 0.10
    # Exploration noise for demonstration collection; zero when flown as the arm.
    noise_std: float = 0.0


@dataclass
class PNGuidanceState:
    """Per-episode memory. The controller itself is stateless and reusable."""

    previous_bearing: float | None = None
    previous_range: float | None = None
    bearing_rate: float = 0.0
    range_rate: float = 0.0
    committed: bool = False
    search: bool = False
    diagnostics: dict = field(default_factory=dict)

    def reset(self) -> None:
        self.previous_bearing = None
        self.previous_range = None
        self.bearing_rate = 0.0
        self.range_rate = 0.0
        self.committed = False
        self.search = False
        self.diagnostics = {}


class PNGuidanceController:
    """Sagittal-plane PN guidance emitting the shared three-channel action."""

    def __init__(self, config: PNGuidanceConfig | None = None, *,
                 nadir_column: float = -0.577, tan_half_horizontal: float = 1.0,
                 max_longitudinal_acceleration_m_s2: float = 1.2,
                 max_vertical_acceleration_m_s2: float = 0.8,
                 max_longitudinal_tilt_rad: float = math.radians(12.0),
                 dt: float = 0.1):
        self.config = config or PNGuidanceConfig()
        self.nadir_column = float(nadir_column)
        self.tan_half_horizontal = float(tan_half_horizontal)
        self.max_longitudinal_acceleration = float(max_longitudinal_acceleration_m_s2)
        self.max_vertical_acceleration = float(max_vertical_acceleration_m_s2)
        self.max_longitudinal_tilt = float(max_longitudinal_tilt_rad)
        self.dt = float(dt)
        for name, value in (("dt", self.dt),
                            ("longitudinal acceleration limit",
                             self.max_longitudinal_acceleration),
                            ("vertical acceleration limit",
                             self.max_vertical_acceleration),
                            ("tilt limit", self.max_longitudinal_tilt),
                            ("horizontal half-angle tangent",
                             self.tan_half_horizontal)):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"PN guidance {name} must be positive and finite")

    # ----------------------------------------------------------- estimation
    def _geometry(self, semantic) -> tuple[float, float, float]:
        """``(bearing_tangent, altitude_estimate_m, range_m)`` from the image."""
        centroid = np.asarray(semantic.centroid_xy, dtype=float).reshape(-1)
        bearing_tangent = float(
            (centroid[0] - self.nadir_column) * self.tan_half_horizontal)
        scale = max(float(semantic.raw_scale), 1e-3)
        low, high = self.config.range_gain_bounds
        altitude = float(np.clip(self.config.reference_scale / scale, low, high))
        distance = max(altitude * math.hypot(1.0, bearing_tangent),
                       self.config.minimum_range_m)
        return bearing_tangent, altitude, distance

    def _filtered(self, previous: float | None, current: float,
                  carried: float) -> float:
        if previous is None:
            return 0.0
        measured = (current - previous) / self.dt
        tau = max(self.config.rate_filter_s, 0.0)
        if tau <= 0.0:
            return measured
        return carried + (measured - carried) * (self.dt / (tau + self.dt))

    # -------------------------------------------------------------- command
    def action(self, semantic, body_velocity, state: PNGuidanceState, *,
               rng=None) -> np.ndarray:
        """One normalized planar action ``[a_fwd, a_z, tilt]``.

        ``body_velocity`` is the UAV's own FLU body velocity, the first three
        components of the actor's proprioception.
        """
        cfg = self.config
        velocity = np.asarray(body_velocity, dtype=float).reshape(-1)
        if velocity.shape != (3,) or not np.isfinite(velocity).all():
            raise ValueError("PN guidance needs the UAV's own 3-D body velocity")

        bearing_tangent, altitude, distance = self._geometry(semantic)
        bearing = math.atan(bearing_tangent)
        trustworthy = (
            float(semantic.visible_keypoint_fraction)
            >= cfg.trustworthy_visible_fraction
            and float(semantic.visual_loss_risk) <= 0.0)
        apparent = float(semantic.apparent_target_scale)
        state.committed = bool(state.committed) or apparent >= cfg.flare_scale

        if trustworthy:
            state.bearing_rate = self._filtered(
                state.previous_bearing, bearing, state.bearing_rate)
            state.range_rate = self._filtered(
                state.previous_range, distance, state.range_rate)
            state.previous_bearing = bearing
            state.previous_range = distance
        state.search = bool(not trustworthy and not state.committed)

        if state.search:
            # Reduced study, search branch: climb on a reference rate until the
            # pad is re-acquired. Nothing here uses a bearing that is stale.
            vertical = cfg.vertical_gain * (
                cfg.climb_reference_m_s - float(velocity[2]))
            longitudinal = 0.0
            state.diagnostics = {
                "mode": "search", "bearing_rad": bearing,
                "bearing_rate_rad_s": 0.0, "range_m": distance,
                "range_rate_m_s": 0.0, "altitude_estimate_m": altitude,
                "closing_speed_m_s": 0.0, "committed": state.committed}
        else:
            closing_speed = max(-state.range_rate, cfg.minimum_closing_speed_m_s)
            across = cfg.navigation_gain * closing_speed * state.bearing_rate
            reference = min(cfg.approach_speed_m_s, cfg.approach_gain * distance)
            along = -cfg.closing_gain * (reference + state.range_rate)
            # Sagittal unit vectors in (forward, up). The pad sits below the
            # vehicle at a horizontal offset of ``bearing_tangent * altitude``.
            norm = math.hypot(1.0, bearing_tangent)
            los = (bearing_tangent / norm, -1.0 / norm)
            perpendicular = (1.0 / norm, bearing_tangent / norm)
            longitudinal = along * los[0] + across * perpendicular[0]
            vertical = along * los[1] + across * perpendicular[1]
            if state.committed:
                # Past the commit the pad fills and leaves a downward camera,
                # so the flare runs on a scheduled rate rather than on a
                # bearing the geometry no longer supports.
                vertical = min(vertical, -cfg.flare_descent_m_s
                               * cfg.vertical_gain * 0.5)
            elif trustworthy:
                # Never let the closing law hold a perfect hover: creep down.
                vertical = min(vertical,
                               cfg.vertical_gain * (-cfg.descent_floor_m_s
                                                    - float(velocity[2])))
            state.diagnostics = {
                "mode": "flare" if state.committed else "track",
                "bearing_rad": bearing,
                "bearing_rate_rad_s": float(state.bearing_rate),
                "range_m": distance, "range_rate_m_s": float(state.range_rate),
                "altitude_estimate_m": altitude,
                "closing_speed_m_s": float(closing_speed),
                "committed": state.committed}

        action = np.array([
            longitudinal / self.max_longitudinal_acceleration,
            vertical / self.max_vertical_acceleration,
            # The tilt this longitudinal acceleration actually implies.
            math.atan2(longitudinal, STANDARD_GRAVITY) / self.max_longitudinal_tilt,
        ], dtype=float)
        if cfg.noise_std > 0.0:
            generator = rng if rng is not None else np.random.default_rng()
            action = action + generator.normal(0.0, cfg.noise_std,
                                               PLANAR_ACTION_DIM)
        state.diagnostics["action"] = [float(value) for value in action]
        return np.clip(action, -0.999, 0.999)
