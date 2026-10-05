"""Common safety supervisor and physical episode termination."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from .config import (CameraConfig, DynamicsConfig, EstimatorConfig,
                     SafetyConfig)
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
    terminal_descent: bool = False
    abort_expired: bool = False
    abort_recovered: bool = False
    touchdown_gate_width_m: float = 0.0


class SafetySupervisor:
    """Bounded braking/hold logic shared bit-for-bit by all policy arms."""

    def __init__(self, dynamics: DynamicsConfig, estimator: EstimatorConfig,
                 safety: SafetyConfig, camera: CameraConfig | None = None):
        self.dynamics = dynamics
        self.estimator = estimator
        self.safety = safety
        self.camera = camera or CameraConfig()
        self.abort_latched = False
        self.abort_started_s: float | None = None
        self.abort_recovery_count = 0
        self.gate_satisfied_at_s: float | None = None

    def reset(self) -> None:
        self.abort_latched = False
        self.abort_started_s = None
        self.abort_recovery_count = 0
        self.gate_satisfied_at_s = None

    def touchdown_gate_width_m(self, height_m: float) -> float:
        """Horizontal error the gate admits at this height.

        Near the ground the camera footprint is much narrower than the nominal
        touchdown tolerance, so a fixed gate lets the controller commit to a
        descent from which the pad leaves the frame. The gate therefore
        tightens with the footprint it can actually observe.
        """
        footprint = max(0.0, float(height_m)) * math.tan(0.5 * self.camera.fov_rad)
        return float(min(self.safety.touchdown_horizontal_error_m, footprint))

    def _attitude_settled(self, state: PlanarState) -> bool:
        return bool(abs(state.theta_rad) <= self.safety.touchdown_pitch_rad
                    and abs(state.pitch_rate_rad_s)
                    <= self.safety.touchdown_pitch_rate_rad_s)

    def touchdown_gate_open(self, state: PlanarState, track: TrackEstimate,
                        *, trustworthy: bool) -> bool:
        """Stricter replacement for the abort-hold margin below the corridor.

        Everything here is causal: the pad position and velocity come from the
        shared estimator, never from the simulator.
        """
        if not trustworthy or not track.initialized:
            return False
        if state.z_m > self.safety.terminal_descent_height_m:
            return False
        margin = self.safety.terminal_descent_speed_margin
        return bool(
            abs(track.pad_x_m - state.x_m)
            <= self.touchdown_gate_width_m(state.z_m)
            and abs(track.pad_vx_m_s - state.vx_m_s)
            <= margin * self.safety.touchdown_relative_speed_m_s
            and self._attitude_settled(state))

    def _touchdown_corridor(self, state: PlanarState, track: TrackEstimate,
                            *, trustworthy: bool) -> bool:
        """Gate, plus a bounded flare commit once the gate becomes geometric.

        The admissible horizontal error shrinks with the camera footprint, so
        in the last few centimetres the gate cannot be satisfied by any
        controller. Rather than widening it, a descent that *did* satisfy it is
        allowed to finish on the causal estimate for a bounded window, and the
        commit is dropped the moment the attitude or the corridor is lost.
        """
        if state.z_m > self.safety.terminal_descent_height_m:
            self.gate_satisfied_at_s = None
            return False
        if self.touchdown_gate_open(state, track, trustworthy=trustworthy):
            self.gate_satisfied_at_s = state.time_s
            return True
        if self.gate_satisfied_at_s is None or not self._attitude_settled(state):
            return False
        committed = (state.time_s - self.gate_satisfied_at_s
                     <= self.safety.terminal_descent_commit_s)
        if not committed:
            self.gate_satisfied_at_s = None
        return bool(committed)

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
        trustworthy = bool(
            not uncertain and age <= self.estimator.recent_track_grace_s)
        landing_inhibited = not trustworthy
        if landing_inhibited:
            reasons.append("track_not_trustworthy")
        abort_recovered = False
        if age >= self.estimator.prolonged_loss_s:
            self.abort_latched = True
            if self.abort_started_s is None:
                self.abort_started_s = state.time_s
            reasons.append("prolonged_visual_loss")
        elif self.abort_latched and trustworthy:
            # Bounded recovery: the pad came back inside the window, so the
            # abort is cleared and the learned policy regains control. Only an
            # expired window below still ends the episode.
            self.abort_latched = False
            self.abort_started_s = None
            self.abort_recovery_count += 1
            abort_recovered = True
            reasons.append("track_reacquired")

        downward_speed = max(0.0, -state.vz_m_s)
        response_delay = (self.dynamics.thrust_time_constant_s
                          + 2.0 / self.dynamics.pitch_natural_frequency_rad_s)
        braking_accel = max(
            0.1, self.dynamics.max_thrust_weight_ratio
            * self.dynamics.gravity_m_s2 - self.dynamics.gravity_m_s2)
        stopping_distance = (downward_speed * response_delay
                             + downward_speed ** 2 / (2.0 * braking_accel))
        stopping_margin = state.z_m - stopping_distance
        # Below the abort-hold height the stopping margin can never be met, so
        # applying it there would inhibit every landing and make an authorized
        # touchdown unreachable. Inside the stricter touchdown corridor the
        # margin is replaced by a descent-speed limit instead of removed.
        terminal_descent = (not self.abort_latched
                            and self._touchdown_corridor(
                                state, track, trustworthy=trustworthy))
        if terminal_descent:
            landing_inhibited = False
            reasons.append("terminal_descent_corridor")
        elif stopping_margin < self.safety.minimum_abort_hold_height_m:
            landing_inhibited = True
            reasons.append("vertical_stopping_margin")

        applied = clipped.copy()
        if landing_inhibited and applied[1] < 0.0:
            applied[1] = min(self.dynamics.az_max_m_s2,
                             max(0.0, downward_speed / max(response_delay, 1e-6)))
            reasons.append("descent_brake")
        if terminal_descent:
            speed_limit = (self.safety.terminal_descent_speed_margin
                           * self.safety.touchdown_vertical_speed_m_s)
            if downward_speed > speed_limit:
                applied[1] = float(np.clip(
                    max(applied[1],
                        (downward_speed - speed_limit) / max(response_delay, 1e-6)),
                    -self.dynamics.az_max_m_s2, self.dynamics.az_max_m_s2))
                reasons.append("touchdown_speed_limit")
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
        abort_expired = bool(
            self.abort_latched and self.abort_started_s is not None
            and state.time_s - self.abort_started_s
            > self.safety.maximum_backup_duration_s)
        if abort_expired:
            reasons.append("backup_duration_exceeded")
        return SafetyDecision(
            requested_m_s2=requested, applied_m_s2=applied,
            intervened=not np.allclose(applied, requested),
            reasons=tuple(dict.fromkeys(reasons)),
            landing_inhibited=landing_inhibited,
            abort_requested=self.abort_latched,
            stopping_margin_m=float(stopping_margin),
            abort_complete=abort_complete, terminal_descent=terminal_descent,
            abort_expired=abort_expired, abort_recovered=abort_recovered,
            touchdown_gate_width_m=self.touchdown_gate_width_m(state.z_m))


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


def hard_envelope_violation(state: PlanarState, config: SafetyConfig, *,
                            pad_x_m: float = 0.0) -> bool:
    """Unrecoverable-state test, measured against the pad rather than the world.

    The pad translates for the whole mission and can finish several hundred
    metres downrange, so bounding the vehicle's absolute ``x`` terminates
    well-flown long episodes that are still tracking the pad from a few
    centimetres away. The separation from the pad is what is actually
    unrecoverable, and it stays comparable to the camera's maximum range.
    """
    values = np.array([state.x_m, state.z_m, state.vx_m_s, state.vz_m_s,
                       state.theta_rad, state.pitch_rate_rad_s, state.thrust_n])
    if not np.isfinite(values).all():
        raise FloatingPointError("invalid numerical state")
    return bool(abs(state.x_m - float(pad_x_m)) > config.max_horizontal_range_m
                or state.z_m > config.max_height_m
                or state.vz_m_s < -config.max_descent_speed_m_s)


def interpolate_contact(before: PlanarState, after: PlanarState,
                        pad_before: tuple[float, float],
                        pad_after: tuple[float, float], *,
                        contact_height_m: float = 0.0) -> ContactState | None:
    """Preserve pre-impact conditions and prevent ground-plane tunnelling.

    ``contact_height_m`` is the landing-gear plane: the vehicle reference point
    never reaches the pad surface, so requiring a 0 m crossing leaves it
    hovering a few centimetres above a pad it is mechanically resting on. The
    0 m plane stays as the numerical fallback when the gear plane was already
    crossed inside an earlier step.
    """
    plane = float(contact_height_m)
    if before.z_m <= plane or after.z_m > plane:
        if plane == 0.0 or before.z_m <= 0.0 or after.z_m > 0.0:
            return None
        plane = 0.0  # fallback: the gear plane was missed, use ground contact
    denominator = before.z_m - after.z_m
    fraction = (1.0 if denominator <= 0.0
                else (before.z_m - plane) / denominator)
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
