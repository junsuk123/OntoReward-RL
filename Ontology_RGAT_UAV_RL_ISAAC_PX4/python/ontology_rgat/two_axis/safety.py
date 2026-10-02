"""Common safety supervisor and physical episode termination."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from .config import DynamicsConfig, EstimatorConfig, SafetyConfig
from .dynamics import PlanarState
from .estimation import TrackEstimate


class TerminalReason(str, Enum):
    SUCCESS = "SUCCESS"
    SAFE_ABORT = "SAFE_ABORT"
    UNSAFE_CONTACT = "UNSAFE_CONTACT"
    UNAUTHORIZED_CONTACT = "UNAUTHORIZED_CONTACT"
    MISSED_PAD_CONTACT = "MISSED_PAD_CONTACT"
    SAFETY_ENVELOPE_VIOLATION = "SAFETY_ENVELOPE_VIOLATION"
    TASK_TIMEOUT = "TASK_TIMEOUT"


@dataclass(frozen=True)
class SafetyDecision:
    requested_m_s2: np.ndarray
    applied_m_s2: np.ndarray
    intervened: bool
    reasons: tuple[str, ...]
    landing_inhibited: bool
    abort_requested: bool
    stopping_margin_m: float
    abort_complete: bool


class SafetySupervisor:
    """Bounded braking/hold logic shared bit-for-bit by all policy arms."""

    def __init__(self, dynamics: DynamicsConfig, estimator: EstimatorConfig,
                 safety: SafetyConfig):
        self.dynamics = dynamics
        self.estimator = estimator
        self.safety = safety
        self.abort_latched = False
        self.abort_started_s: float | None = None

    def reset(self) -> None:
        self.abort_latched = False
        self.abort_started_s = None

    def apply(self, requested_m_s2: np.ndarray, state: PlanarState,
              track: TrackEstimate) -> SafetyDecision:
        requested = np.asarray(requested_m_s2, dtype=float).reshape(2)
        clipped = np.array([
            np.clip(requested[0], -self.dynamics.ax_max_m_s2,
                    self.dynamics.ax_max_m_s2),
            np.clip(requested[1], -self.dynamics.az_max_m_s2,
                    self.dynamics.az_max_m_s2)], dtype=float)
        reasons: list[str] = []
        if not np.allclose(clipped, requested):
            reasons.append("physical_command_limit")
        age = track.time_since_detection_s
        uncertain = (not track.initialized or track.position_std_m > 0.75
                     or track.velocity_std_m_s > 1.0)
        landing_inhibited = bool(
            uncertain or age > self.estimator.recent_track_grace_s)
        if landing_inhibited:
            reasons.append("track_not_trustworthy")
        if age >= self.estimator.prolonged_loss_s:
            self.abort_latched = True
            if self.abort_started_s is None:
                self.abort_started_s = state.time_s
            reasons.append("prolonged_visual_loss")

        downward_speed = max(0.0, -state.vz_m_s)
        response_delay = (self.dynamics.thrust_time_constant_s
                          + 2.0 / self.dynamics.pitch_natural_frequency_rad_s)
        braking_accel = max(
            0.1, self.dynamics.max_thrust_weight_ratio
            * self.dynamics.gravity_m_s2 - self.dynamics.gravity_m_s2)
        stopping_distance = (downward_speed * response_delay
                             + downward_speed ** 2 / (2.0 * braking_accel))
        stopping_margin = state.z_m - stopping_distance
        if stopping_margin < self.safety.minimum_abort_hold_height_m:
            landing_inhibited = True
            reasons.append("vertical_stopping_margin")

        applied = clipped.copy()
        if landing_inhibited and applied[1] < 0.0:
            applied[1] = min(self.dynamics.az_max_m_s2,
                             max(0.0, downward_speed / max(response_delay, 1e-6)))
            reasons.append("descent_brake")
        if self.abort_latched:
            applied[0] = float(np.clip(-1.5 * state.vx_m_s,
                                       -self.dynamics.ax_max_m_s2,
                                       self.dynamics.ax_max_m_s2))
            applied[1] = (self.dynamics.az_max_m_s2 if state.vz_m_s < -0.1
                          else float(np.clip(-1.0 * state.vz_m_s,
                                             -self.dynamics.az_max_m_s2,
                                             self.dynamics.az_max_m_s2)))
            reasons.append("abort_braking_hold")
        abort_complete = bool(
            self.abort_latched
            and state.z_m >= self.safety.minimum_abort_hold_height_m
            and abs(state.vz_m_s) <= 0.1 and abs(state.vx_m_s) <= 0.2
            and abs(state.theta_rad) <= self.safety.touchdown_pitch_rad)
        if (self.abort_started_s is not None
                and state.time_s - self.abort_started_s
                > self.safety.maximum_backup_duration_s):
            reasons.append("backup_duration_exceeded")
        return SafetyDecision(
            requested_m_s2=requested, applied_m_s2=applied,
            intervened=not np.allclose(applied, requested),
            reasons=tuple(dict.fromkeys(reasons)),
            landing_inhibited=landing_inhibited,
            abort_requested=self.abort_latched,
            stopping_margin_m=float(stopping_margin), abort_complete=abort_complete)


@dataclass(frozen=True)
class ContactState:
    time_s: float
    longitudinal_error_m: float
    relative_longitudinal_speed_m_s: float
    vertical_speed_m_s: float
    pitch_rad: float
    pitch_rate_rad_s: float


def classify_contact(contact: ContactState, *, authorized: bool,
                     config: SafetyConfig) -> TerminalReason:
    if abs(contact.longitudinal_error_m) > config.pad_half_width_m:
        return TerminalReason.MISSED_PAD_CONTACT
    mechanically_safe = (
        abs(contact.longitudinal_error_m) <= config.touchdown_horizontal_error_m
        and abs(contact.relative_longitudinal_speed_m_s)
        <= config.touchdown_relative_speed_m_s
        and abs(contact.vertical_speed_m_s) <= config.touchdown_vertical_speed_m_s
        and abs(contact.pitch_rad) <= config.touchdown_pitch_rad
        and abs(contact.pitch_rate_rad_s) <= config.touchdown_pitch_rate_rad_s)
    if not mechanically_safe:
        return TerminalReason.UNSAFE_CONTACT
    if not authorized:
        return TerminalReason.UNAUTHORIZED_CONTACT
    return TerminalReason.SUCCESS


def hard_envelope_violation(state: PlanarState, config: SafetyConfig) -> bool:
    values = np.array([state.x_m, state.z_m, state.vx_m_s, state.vz_m_s,
                       state.theta_rad, state.pitch_rate_rad_s, state.thrust_n])
    if not np.isfinite(values).all():
        raise FloatingPointError("invalid numerical state")
    return bool(abs(state.x_m) > config.max_horizontal_range_m
                or state.z_m > config.max_height_m
                or state.vz_m_s < -config.max_descent_speed_m_s)


def interpolate_contact(before: PlanarState, after: PlanarState,
                        pad_before: tuple[float, float],
                        pad_after: tuple[float, float]) -> ContactState | None:
    """Preserve pre-impact conditions and prevent ground-plane tunnelling."""
    if before.z_m <= 0.0 or after.z_m > 0.0:
        return None
    denominator = before.z_m - after.z_m
    fraction = 1.0 if denominator <= 0.0 else before.z_m / denominator
    fraction = float(np.clip(fraction, 0.0, 1.0))
    def lerp(a: float, b: float) -> float:
        return a + fraction * (b - a)
    pad_x = lerp(pad_before[0], pad_after[0])
    pad_v = lerp(pad_before[1], pad_after[1])
    return ContactState(
        time_s=lerp(before.time_s, after.time_s),
        longitudinal_error_m=pad_x - lerp(before.x_m, after.x_m),
        relative_longitudinal_speed_m_s=pad_v - lerp(
            before.vx_m_s, after.vx_m_s),
        vertical_speed_m_s=lerp(before.vz_m_s, after.vz_m_s),
        pitch_rad=lerp(before.theta_rad, after.theta_rad),
        pitch_rate_rad_s=lerp(before.pitch_rate_rad_s,
                              after.pitch_rate_rad_s))
