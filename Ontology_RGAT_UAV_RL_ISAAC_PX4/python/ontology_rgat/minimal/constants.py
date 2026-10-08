"""Domain knowledge of the minimal contract: geometry, limits, thresholds.

These are TBox constants, not observations. The pad-loss thresholds are
STARTING POINTS taken from the spatial-reference/8-/10 Isaac records (track
lost at body 0.20-0.25 m, 0.2 m gate floor, 3 s commit window); measure the
classifier's confusion matrix on the local plant before trusting them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class LandingConstants:
    # Camera: 90 x 73.7 deg, nadir, 0.16 m below the body (no landing legs).
    half_fov_x_rad: float = math.radians(45.0)
    half_fov_y_rad: float = math.atan(0.75)
    camera_below_body_m: float = 0.16
    # Vehicle and pad.
    touchdown_height_m: float = 0.12
    pad_half_width_m: float = 0.5
    touchdown_xy_speed_m_s: float = 0.35
    touchdown_z_speed_m_s: float = 0.30
    max_acceleration_m_s2: tuple = (2.5, 2.5, 2.0)
    tilt_command_limit_rad: float = math.radians(20.0)
    gravity_m_s2: float = 9.81
    policy_dt_s: float = 0.1
    # Observation.
    pad_age_cap_s: float = 5.0
    frame_valid_age_s: float = 0.5
    # Freshness time constants (f = exp(-age / tau)).
    pad_freshness_tau_s: float = 0.5
    own_freshness_tau_s: float = 0.2
    # Pad velocity from consecutive detections.
    # The window sets the noise, the span sets when an estimate counts. A
    # 0.5 s window with a 0.15 s span declared a -0.26 m/s pad velocity at
    # loss (truth -0.04..-0.12) and the terminal commit matched it into an
    # unsafe contact (combined, seed 4106). Swept on seeds 5000-5095 (window
    # 0.7): span 0.2 -> poor_vision 84 %, combined 51 %, unsafe <= 1 %;
    # spans 0.25-0.35 lost 10-26 points of combined to NO_PAD_VELOCITY aborts.
    pad_velocity_window_s: float = 0.7
    pad_velocity_min_points: int = 3
    pad_velocity_min_span_s: float = 0.2
    visibility_horizon_s: float = 0.5
    # Pad-loss classifier.
    transient_age_s: float = 0.3
    # Local optics lose the last tag at body 0.31-0.36 m, Isaac at 0.20-0.25 m;
    # 0.35 put the local loss on the boundary and a centred descent became
    # HIGH_ALTITUDE -> LOST -> abort (seed 3033).
    terminal_entry_height_m: float = 0.40
    terminal_offset_m: float = 0.2
    terminal_release_offset_m: float = 0.25
    terminal_max_sink_up_m_s: float = 0.05
    terminal_window_s: float = 3.0
    loss_timeout_s: float = 3.0
    own_invalid_timeout_s: float = 1.0
    # Supervisor.
    descent_gate_horizon_s: float = 0.5
    # Canonical descent profile and TrackingBias caps (bias.py).
    profile_align_fraction: float = 0.4
    profile_sink_per_m: float = 0.2
    bias_xy_cap: float = 10.0
    bias_z_cap: float = 2.0
    handover_window_s: float = 2.0
    handover_lateral_cap_m_s2: float = 2.0
    handover_lateral_slew_m_s2: float = 1.5
    terminal_sink_factor: float = 0.8
    sink_brake_deceleration_m_s2: float = 0.8
    sink_response_lag_s: float = 0.3
    vertical_hold_time_constant_s: float = 0.5
    terminal_commit_time_constant_s: float = 0.2
    view_recovery_fraction: float = 0.8
    view_recovery_climb_m_s: float = 0.4
    terminal_lateral_cap_m_s2: float = 1.5
    terminal_velocity_gain: float = 1.0
    terminal_position_gain: float = 0.8
    abort_climb_rate_m_s: float = 0.3
    abort_position_gain: float = 0.8
    abort_velocity_gain: float = 1.2
    abort_lateral_cap_m_s2: float = 1.5
    # Causal disturbance observer (own velocity + own commands only).
    actuation_delay_steps: int = 1
    disturbance_filter: float = 0.2
    disturbance_cap_m_s2: float = 1.0

    @property
    def min_half_fov_rad(self) -> float:
        """Yaw is not observed, so bearings are judged against the narrow axis."""
        return min(self.half_fov_x_rad, self.half_fov_y_rad)

    def camera_height(self, body_height_m: float) -> float:
        return body_height_m - self.camera_below_body_m

    def footprint_half_width(self, body_height_m: float) -> float:
        """Half-width of the ground the camera sees on its narrow axis."""
        return max(self.camera_height(body_height_m), 0.0) * math.tan(self.min_half_fov_rad)

    def descent_gate_width(self, body_height_m: float) -> float:
        return max(self.terminal_offset_m, self.footprint_half_width(body_height_m))

    def allowed_sink(self, body_height_m: float) -> float:
        """Largest sink speed from which touchdown at <= 0.8 x limit is still reachable.

        Braking at ``a`` only starts after the response lag ``L``, during which
        the vehicle keeps sinking at ``v``: v^2 - v_td^2 <= 2a(h - v L), i.e.
        v <= -aL + sqrt(a^2 L^2 + v_td^2 + 2ah). The /9 form without the lag
        term admitted 0.66 m/s at 0.37 m and the local plant (0.1 s actuation
        delay, attitude/thrust response, seeded vertical force) touched down at
        0.32-0.56 m/s from there (seeds 4110, 4115).
        """
        touchdown = self.terminal_sink_factor * self.touchdown_z_speed_m_s
        if body_height_m <= self.terminal_entry_height_m:
            # Inside the terminal zone the remaining height cannot absorb the
            # response lag: the corridor brake of /9. Measured: admitting the
            # lag formula's 0.39 m/s at 0.30 m touched down at 0.31 (seed 4139).
            return touchdown
        height = max(body_height_m - self.touchdown_height_m, 0.0)
        a, lag = self.sink_brake_deceleration_m_s2, self.sink_response_lag_s
        reachable = -a * lag + math.sqrt(a * a * lag * lag + touchdown ** 2 + 2.0 * a * height)
        return max(touchdown, reachable)

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONSTANTS = LandingConstants()
