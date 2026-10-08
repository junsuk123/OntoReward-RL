"""Safety supervisor on the minimal observation only (laws S1-S6).

Reads ``LandingObservation`` and its own ``PadMemory`` -- the same classifier
the ontology calls -- and NEVER the graph or the policy's internals, so all
three arms fly under one supervisor. No tracker std, detection confidence or
track-initialized flag exists in this contract and none is read.

Priority: ABORT_HOLD > TERMINAL_COMMIT > DESCENT_HOLD > NOMINAL.

S1 saturation      per-axis limits, derived tilt <= 20 deg
S2 handover        first 2 s: |a_xy| <= 2.0, slew <= 1.5 per step, a_z >= 0
                   (not near the pad: the floor broke the easy rung on /12)
S3 descent gate    descent only with VISIBLE/TRANSIENT pad inside the gate NOW
                   and 0.5 s ahead at the current relative velocity, and below
                   the terminal entry height only with a KNOWN relative
                   velocity within the touchdown limit; else the vertical RATE
                   is held, symmetrically (/2). Measured: a position-only gate
                   authorized descent at 0.29 m from one re-detection 0.08 m
                   off-centre while the pad ran away at 0.92 m/s, and the
                   vehicle touched down 0.55 m off the pad (seed 3017).
S4 sink brake      excess sink over ``allowed_sink(h)`` removed in one step (/9)
S5 terminal commit TERMINAL_OCCLUSION: braked 0.24 m/s descent; lateral flown by
                   the supervisor (velocity match + dead-reckoned offset +
                   disturbance cancellation, <= 1.5 m/s^2)
S6 abort hold      LOST for > loss_timeout, or own invalid > 1 s: own-EKF
                   position hold plus climb on all axes; released when the pad
                   is VISIBLE again (bounded recovery, never a terminal)

Every law the supervisor itself flies (S3 hold, S5 commit, S6 abort)
subtracts a causal disturbance estimate. Without it a P-only rate hold settles
where the constant per-episode force balances it: measured on the local plant,
seed 4107, a 0.42 m/s^2 downward force held the "vertical hold" at a steady
-0.21 m/s sink and the vehicle touched down while LOST, and the 0.24 m/s
terminal commit would have arrived at 0.45 m/s, past the 0.30 limit. The
estimate is d = dv/dt - (command active over the interval), low-passed; the
command active over [t-1, t] is the one issued ``actuation_delay_steps``
decisions earlier. Only own EKF velocity and the supervisor's own outputs
enter it -- no truth, no pad measurement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .constants import DEFAULT_CONSTANTS, LandingConstants
from .observation import LandingObservation
from .pad_loss import (LOST, TERMINAL_OCCLUSION, TRANSIENT_DROPOUT, VISIBLE,
                       PadLossAssessment, PadMemory, bearing_fraction)

MODE_NOMINAL, MODE_DESCENT_HOLD, MODE_TERMINAL_COMMIT, MODE_ABORT_HOLD = 0, 1, 2, 3
MODE_NAMES = ("NOMINAL", "DESCENT_HOLD", "TERMINAL_COMMIT", "ABORT_HOLD")


@dataclass(frozen=True)
class SupervisorDecision:
    stamp: float
    mode: int
    requested: np.ndarray
    applied: np.ndarray
    intervened: tuple[bool, bool, bool]
    assessment: PadLossAssessment

    @property
    def mode_name(self) -> str:
        return MODE_NAMES[self.mode]


@dataclass
class MinimalSafetySupervisor:
    constants: LandingConstants = DEFAULT_CONSTANTS
    memory: PadMemory = None
    handover_stamp: float | None = None
    previous_lateral: np.ndarray | None = None
    abort_reference: np.ndarray | None = None
    disturbance: np.ndarray = None
    previous_velocity: np.ndarray | None = None
    previous_stamp: float | None = None
    command_history: list = None

    def __post_init__(self):
        if self.memory is None:
            self.memory = PadMemory(self.constants)
        if self.disturbance is None:
            self.disturbance = np.zeros(3)
        if self.command_history is None:
            self.command_history = []

    def reset(self) -> None:
        self.memory = PadMemory(self.constants)
        self.handover_stamp = self.previous_lateral = self.abort_reference = None
        self.disturbance, self.command_history = np.zeros(3), []
        self.previous_velocity = self.previous_stamp = None

    def _observe_disturbance(self, obs: LandingObservation) -> np.ndarray:
        """Unmodelled acceleration from own velocity and own past commands."""
        c = self.constants
        v = np.asarray(obs.own_velocity, float)
        lag = c.actuation_delay_steps + 1
        if (obs.own_valid and self.previous_velocity is not None
                and len(self.command_history) >= lag):
            dt = obs.stamp - self.previous_stamp
            if 0.0 < dt <= 2 * c.policy_dt_s:
                measured = (v - self.previous_velocity) / dt - self.command_history[-lag]
                self.disturbance = np.clip(
                    (1 - c.disturbance_filter) * self.disturbance
                    + c.disturbance_filter * measured,
                    -c.disturbance_cap_m_s2, c.disturbance_cap_m_s2)
        self.previous_velocity, self.previous_stamp = v.copy(), obs.stamp
        return self.disturbance

    def step(self, obs: LandingObservation, requested) -> SupervisorDecision:
        c = self.constants
        requested = np.asarray(requested, dtype=float).reshape(3)
        a = self.memory.update(obs)
        d_hat = self._observe_disturbance(obs)
        if self.handover_stamp is None:
            self.handover_stamp = obs.stamp
        v = np.asarray(obs.own_velocity, float)
        rel = a.relative_position
        h = -float(rel[2]) if obs.ever_detected else math.inf
        d_xy = float(np.hypot(rel[0], rel[1]))
        limits = np.asarray(c.max_acceleration_m_s2, float)
        cmd = np.clip(requested, -limits, limits)                       # S1
        near_pad = h <= c.terminal_entry_height_m

        # S6: abort hold, latched until the pad is seen again.
        lost_too_long = (a.pad_class == LOST and obs.ever_detected
                         and obs.stamp - obs.pad_capture_stamp > c.loss_timeout_s)
        never_seen_too_long = (not obs.ever_detected
                               and obs.stamp - self.handover_stamp > c.loss_timeout_s)
        own_bad = self.memory.own_invalid_duration(obs.stamp) > c.own_invalid_timeout_s
        if self.abort_reference is not None and a.pad_class == VISIBLE and not own_bad:
            self.abort_reference = None
        if lost_too_long or never_seen_too_long or own_bad or self.abort_reference is not None:
            if self.abort_reference is None:
                self.abort_reference = np.asarray(obs.own_position, float).copy()
            mode = MODE_ABORT_HOLD
            if obs.own_valid:
                error = self.abort_reference[:2] - np.asarray(obs.own_position, float)[:2]
                lateral = c.abort_position_gain * error - c.abort_velocity_gain * v[:2]
            else:
                lateral = -c.abort_velocity_gain * v[:2]
            lateral = _cap_norm(lateral - d_hat[:2], c.abort_lateral_cap_m_s2)
            az = (c.abort_climb_rate_m_s - v[2]) / c.vertical_hold_time_constant_s - d_hat[2]
            cmd = np.array([lateral[0], lateral[1], az])
        elif a.pad_class == TERMINAL_OCCLUSION:                         # S5
            mode = MODE_TERMINAL_COMMIT
            target = -c.terminal_sink_factor * c.touchdown_z_speed_m_s
            # The blind stage is flown by the supervisor on its own dead
            # reckoning: match the pad's last known velocity, close the
            # predicted offset, cancel the disturbance. Capping the POLICY's
            # lateral at 0.5 m/s^2 could not hold a 0.7-0.9 m/s^2 crosswind
            # (combined stress): the relative speed grew to 0.30-0.44 m/s
            # in the last second and touched down past the 0.35 limit
            # (seeds 4102, 4106, 4126). The policy sees the same dead
            # reckoning here, so nothing it knows is lost.
            v_rel_now = a.relative_velocity[:2] if a.relative_velocity_valid else -v[:2]
            lateral = (c.terminal_velocity_gain * v_rel_now
                       + c.terminal_position_gain * rel[:2] - d_hat[:2])
            cmd[:2] = _cap_norm(lateral, c.terminal_lateral_cap_m_s2)
            # Brake hard, speed up gently: a stiff loop accelerating a hovering
            # vehicle down to the target overshot it through the 0.1 s delay
            # and touched down at 0.33 m/s (seed 3034).
            tau = (c.terminal_commit_time_constant_s if v[2] < target
                   else c.vertical_hold_time_constant_s)
            cmd[2] = (target - v[2]) / tau - d_hat[2]
            excess = -v[2] - c.allowed_sink(h)                          # S4 here too
            if excess > 0:
                cmd[2] = max(cmd[2], excess / c.policy_dt_s)
        else:
            gate = c.descent_gate_width(h)
            v_rel_xy = a.relative_velocity[:2] if a.relative_velocity_valid else np.zeros(2)
            ahead = float(np.linalg.norm(rel[:2] + v_rel_xy * c.descent_gate_horizon_s))
            settled = (h > c.terminal_entry_height_m or (
                a.relative_velocity_valid
                and float(np.linalg.norm(v_rel_xy)) <= c.touchdown_xy_speed_m_s))
            authorized = (obs.own_valid and a.pad_class in (VISIBLE, TRANSIENT_DROPOUT)
                          and d_xy <= gate and ahead <= gate and settled)
            mode = MODE_NOMINAL
            if not authorized:                                          # S3
                mode = MODE_DESCENT_HOLD
                hold = -v[2] / c.vertical_hold_time_constant_s - d_hat[2]
                # The hold is symmetric (/2: one-sided braking let exploration
                # ratchet upward into the ceiling), EXCEPT a bounded climb when
                # the pad is about to leave the frame above the terminal zone:
                # the footprint grows with height, and holding blocked exactly
                # the recovery that keeps the pad in view (seed 4138).
                ahead = rel + np.r_[v_rel_xy, 0.0] * c.visibility_horizon_s
                if (obs.ever_detected and h > c.terminal_entry_height_m
                        and bearing_fraction(ahead, c) > c.view_recovery_fraction):
                    climb = (c.view_recovery_climb_m_s - v[2]) / c.vertical_hold_time_constant_s - d_hat[2]
                    hold = max(hold, min(cmd[2], climb))
                cmd[2] = hold
            # S4: never sink faster than the height allows.
            excess = -v[2] - c.allowed_sink(h)
            if excess > 0:
                cmd[2] = max(cmd[2], excess / c.policy_dt_s)

        # S2: handover limits, not inside the terminal phase.
        if (mode != MODE_ABORT_HOLD and obs.stamp - self.handover_stamp < c.handover_window_s):
            lateral = _cap_norm(cmd[:2], c.handover_lateral_cap_m_s2)
            if self.previous_lateral is not None:
                step = lateral - self.previous_lateral
                lateral = self.previous_lateral + _cap_norm(step, c.handover_lateral_slew_m_s2)
            cmd[:2] = lateral
            if not near_pad and mode != MODE_TERMINAL_COMMIT:
                cmd[2] = max(cmd[2], 0.0)

        cmd = np.clip(cmd, -limits, limits)
        # S1 tilt: a_xy / (g + a_z) <= tan(20 deg).
        lateral_cap = max(c.gravity_m_s2 + cmd[2], 0.0) * math.tan(c.tilt_command_limit_rad)
        cmd[:2] = _cap_norm(cmd[:2], lateral_cap)
        self.previous_lateral = cmd[:2].copy()
        self.command_history = (self.command_history + [cmd.copy()])[-4:]
        intervened = tuple(bool(abs(cmd[i] - requested[i]) > 1e-9) for i in range(3))
        return SupervisorDecision(obs.stamp, mode, requested, cmd, intervened, a)


def _cap_norm(vector, cap: float) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= cap or norm == 0.0:
        return vector.copy()
    return vector * (cap / norm)
