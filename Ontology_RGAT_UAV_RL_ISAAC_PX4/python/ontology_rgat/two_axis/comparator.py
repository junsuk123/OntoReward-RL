"""Non-learned PN comparison controller for the two-axis landing task.

This arm exists to answer one question the learned arms cannot: is the task
physically feasible inside the declared envelope? It is **not** a teacher. Its
actions are never recorded for behaviour cloning, never used as a PPO target,
and never shape the reward. It shares the environment, the camera, the causal
estimator, the action space and limits, the safety supervisor, the termination
rules and the evaluation seeds with every learned arm.

Guidance law
------------
Proportional navigation across the line of sight to the *estimated* pad, plus
explicit relative-velocity regulation, because landing on a moving pad is a
rendezvous rather than an intercept and pure PN drives the miss distance to
zero without matching the pad's velocity.

    lambda   = atan2(e_x, h)                 (bearing from nadir)
    a_perp   = N * V_c * lambda_dot          (V_c = closing speed)
    a_match  = k_v * (v_pad - v_own) + k_p * e_x

The approach gate
-----------------
A fixed alignment gate lets the controller commit to a descent from which the
pad leaves the frame: near the ground the camera footprint is far narrower
than the nominal 0.35 m touchdown tolerance. Above
``terminal_descent_height_m`` the gate stays permissive; below it the admitted
error shrinks with the observable footprint and the relative speed, pitch and
pitch rate must all have settled before the descent continues. The gate reuses
the shared supervisor's own test, so the controller and the supervisor cannot
disagree about when a descent is admissible.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config import ExperimentConfig
from .dynamics import PlanarState
from .estimation import TrackEstimate
from .safety import SafetySupervisor


@dataclass(frozen=True)
class PNGains:
    """Chosen by grid search on the train seeds (1000-1023) only.

    The validation and test seeds were never read while tuning, so the PN
    numbers reported on them are out-of-sample for this controller.
    """

    navigation_constant: float = 2.0
    position_gain: float = 0.9
    velocity_gain: float = 1.9
    descent_gain: float = 0.55
    descent_speed_m_s: float = 0.9
    vertical_gain: float = 1.4


class PNLandingController:
    """Causal PN guidance; it reads the shared estimator, never the simulator."""

    def __init__(self, config: ExperimentConfig, *, gains: PNGains | None = None):
        self.config = config
        self.gains = gains or PNGains()
        self.supervisor = SafetySupervisor(
            config.dynamics, config.estimator, config.safety, config.camera)
        self.reset()

    def reset(self) -> None:
        self._previous_bearing: float | None = None
        self._previous_range: float | None = None
        self._previous_time: float | None = None

    def _rates(self, bearing: float, range_m: float, time_s: float
               ) -> tuple[float, float]:
        if (self._previous_time is None
                or time_s <= self._previous_time + 1e-9):
            bearing_rate = range_rate = 0.0
        else:
            dt = time_s - self._previous_time
            bearing_rate = (bearing - float(self._previous_bearing)) / dt
            range_rate = (range_m - float(self._previous_range)) / dt
        self._previous_bearing = bearing
        self._previous_range = range_m
        self._previous_time = time_s
        return bearing_rate, range_rate

    def approach_gate_open(self, state: PlanarState, track: TrackEstimate) -> bool:
        """Permissive above the corridor, footprint-tight inside it."""
        if not track.initialized:
            return False
        error = abs(track.pad_x_m - state.x_m)
        relative = abs(track.pad_vx_m_s - state.vx_m_s)
        if state.z_m > self.config.safety.terminal_descent_height_m:
            return bool(error <= self.config.safety.pad_half_width_m
                        and relative
                        <= 2.0 * self.config.safety.touchdown_relative_speed_m_s)
        trustworthy = bool(
            track.time_since_detection_s <= self.config.estimator.recent_track_grace_s
            and track.position_std_m <= 0.75 and track.velocity_std_m_s <= 1.0)
        # Exactly the supervisor's own test, so the controller and the
        # supervisor cannot disagree about when a descent is admissible.
        return self.supervisor.touchdown_gate_open(
            state, track, trustworthy=trustworthy)

    def act(self, state: PlanarState, track: TrackEstimate) -> np.ndarray:
        """Return the normalized two-axis command for this decision step."""
        dynamics = self.config.dynamics
        if not track.initialized:
            return np.zeros(2, dtype=float)
        error = track.pad_x_m - state.x_m
        relative_speed = track.pad_vx_m_s - state.vx_m_s
        height = max(state.z_m, 1e-3)
        bearing = math.atan2(error, height)
        range_m = math.hypot(error, height)
        bearing_rate, range_rate = self._rates(bearing, range_m, state.time_s)
        closing = -range_rate
        gains = self.gains
        longitudinal = (gains.navigation_constant * closing * bearing_rate
                        * math.cos(bearing)
                        + gains.position_gain * error
                        + gains.velocity_gain * relative_speed)
        if self.approach_gate_open(state, track):
            target_vz = -min(gains.descent_speed_m_s, gains.descent_gain * height)
        else:
            # Stabilise horizontally before continuing the final descent.
            target_vz = 0.0
        vertical = gains.vertical_gain * (target_vz - state.vz_m_s)
        return np.array([
            float(np.clip(longitudinal / dynamics.ax_max_m_s2, -1.0, 1.0)),
            float(np.clip(vertical / dynamics.az_max_m_s2, -1.0, 1.0))],
            dtype=float)
