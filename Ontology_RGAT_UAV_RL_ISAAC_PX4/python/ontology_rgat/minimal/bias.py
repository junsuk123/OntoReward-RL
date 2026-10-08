"""TrackingBias: what a constant, unobserved force leaves behind in the track.

The local plant (and Isaac) applies a constant per-episode force of up to
0.5 m/s^2 against a 2.5 m/s^2 authority. No single minimal observation shows
it, and the ontology does not receive the vehicle's own commands, so it cannot
be estimated directly. What it does leave is a persistent tracking error, and
the time integral of that error IS observable from the observation history:

    lateral   I_xy = int rel_xy dt              (while the pad is detected)
    vertical  I_z  = int (v_ref(h, rel) - v_z) dt

``v_ref`` is the canonical descent profile (TBox): descend only when aligned
within ``profile_align_fraction`` of the descent gate, at a sink that grows
``profile_sink_per_m`` per metre above touchdown, bounded by the supervisor's
``allowed_sink``.

Measured on the local plant (24 nominal seeds, PD teacher kp 0.4 / kd 0.8):
with both integrals 24/24 SUCCESS; lateral only 17/24 with 4 UNSAFE; vertical
only 0/24 (23 TASK_TIMEOUT, stalled 0.5-0.9 m off the pad); neither 0/24. And
behaviour clones of the integrating teacher, whose inputs had no integral,
landed 0-4 % for all three arms with the same stall. A memoryless policy over
an observation without this node cannot land; that is an information gap, not
a learning failure.

The accumulators are clipped at ``bias_xy_cap`` / ``bias_z_cap`` so that a
teacher reading ``clip(k * I, +-1)`` with k = 1/cap behaves exactly like a
PI controller with that clipped accumulator (anti-windup unchanged).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import DEFAULT_CONSTANTS, LandingConstants
from .observation import LandingObservation
from .pad_loss import PadLossAssessment


def descent_reference(rel: np.ndarray, constants: LandingConstants = DEFAULT_CONSTANTS) -> float:
    """Canonical sink target (m/s, positive down) at this relative position."""
    c = constants
    h = -float(rel[2])
    aligned = float(np.hypot(rel[0], rel[1])) <= c.profile_align_fraction * c.descent_gate_width(h)
    if not aligned:
        return 0.0
    return min(c.allowed_sink(h),
               max(c.terminal_sink_factor * c.touchdown_z_speed_m_s,
                   c.profile_sink_per_m * (h - c.touchdown_height_m)))


@dataclass
class TrackingBias:
    constants: LandingConstants = DEFAULT_CONSTANTS
    xy: np.ndarray = field(default_factory=lambda: np.zeros(2))
    z: float = 0.0

    def update(self, obs: LandingObservation, assessment: PadLossAssessment) -> np.ndarray:
        c = self.constants
        dt = c.policy_dt_s
        if not obs.ever_detected:
            return self.value()
        rel = assessment.relative_position
        if obs.pad_detected:
            self.xy = np.clip(self.xy + rel[:2] * dt, -c.bias_xy_cap, c.bias_xy_cap)
        error = -descent_reference(rel, c) - float(obs.own_velocity[2])
        self.z = float(np.clip(self.z + error * dt, -c.bias_z_cap, c.bias_z_cap))
        return self.value()

    def value(self) -> np.ndarray:
        return np.array([self.xy[0], self.xy[1], self.z])

    def normalized(self) -> np.ndarray:
        c = self.constants
        return np.array([self.xy[0] / c.bias_xy_cap, self.xy[1] / c.bias_xy_cap,
                         self.z / c.bias_z_cap])
