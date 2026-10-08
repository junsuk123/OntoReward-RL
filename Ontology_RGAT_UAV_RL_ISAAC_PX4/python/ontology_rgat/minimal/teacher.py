"""A memoryless-in-gains PD teacher on the minimal observation.

It reads ``LandingObservation`` through its own ``PadMemory`` (the shared
classifier: measured relative position while visible, dead-reckoned while
not) and nothing else -- no truth. Its job is to show the task is solvable
under this contract and to label behaviour cloning. A ceiling is only a
ceiling if its gains were swept (AGENTS.md); ``tools/minimal_teacher_sweep.py``
sweeps them and reports whether the best cell sits on the grid boundary.

Horizontal: a_xy = kp * rel_xy + kd * v_rel_xy + int(rel_xy) / cap (rel = pad
- body, so this drives rel and its rate to zero against a moving pad; the
integral removes the offset a constant per-episode force leaves, ~0.6 m at
kp 0.8 without it). Vertical: track a sink target that grows with height while
aligned inside the descent gate, hold altitude otherwise, with an integral on
the rate error for the same reason. The supervisor still applies S1-S6.
The integrals make the teacher history-dependent: fine for a ceiling and for
plain behaviour cloning, wrong for DAgger relabelling (AGENTS.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import DEFAULT_CONSTANTS, LandingConstants
from .observation import LandingObservation
from .bias import TrackingBias, descent_reference
from .pad_loss import PadMemory, bearing_fraction


@dataclass(frozen=True)
class TeacherGains:
    """Swept on minimal-landing-ontology/3 (results/minimal_contract_20261007/
    teacher_sweep_r3.json): an interior cell of a 123-of-243-cell plateau,
    24/24 on validation seeds, 46/48 held out. Revised 2026-10-08 to kp 0.6 /
    kd 1.1 with the velocity boost only on a fresh detection: the boost acting
    on a stale, noisy relative-velocity estimate was what made the teacher
    over-react under poor vision (a calmer clone out-landed it). On seeds
    5000-5095 (not the held-out range): nominal 100/100, fast_pad 100/100,
    strong_wind 88.5 -> 96.9, poor_vision 62.5 -> 75.0, combined 15.6 -> 36.5 %,
    unsafe <= 1 %. Weakening the lateral loop while the pad is hidden was
    tried first and made every stress scenario worse (``hidden_lateral_scale``
    stays 1.0). The integral gains are 1/cap of
    the shared ``TrackingBias`` accumulators, and the descent profile is the
    ontology's (sink 0.2 /m, align 0.4), so the teacher's state IS the graph's."""
    kp: float = 0.6
    kd: float = 1.1
    kz: float = 1.5              # vertical rate loop gain
    kd_boost: float = 0.6        # extra velocity matching above the touchdown speed
    #: Lateral authority kept while the pad is NOT in the current frame,
    #: blended by detection freshness: 1.0 chases the dead-reckoned estimate
    #: as hard as a measurement, which is what made the teacher worse than a
    #: calmer clone under poor vision (stress diagnostics, doc section 15.2).
    hidden_lateral_scale: float = 1.0
    #: Apply ``kd_boost`` only on a fresh measurement with a valid velocity.
    boost_requires_fresh: bool = True


class MinimalTeacher:
    def __init__(self, gains: TeacherGains = TeacherGains(),
                 constants: LandingConstants = DEFAULT_CONSTANTS):
        self.gains, self.constants = gains, constants
        self.reset()

    def reset(self) -> None:
        self.memory = PadMemory(self.constants)
        self.bias = TrackingBias(self.constants)

    def act(self, obs: LandingObservation) -> np.ndarray:
        g, c = self.gains, self.constants
        a = self.memory.update(obs)
        integral = np.clip(self.bias.update(obs, a)
                           / np.array([c.bias_xy_cap, c.bias_xy_cap, c.bias_z_cap]), -1.0, 1.0)
        v_own = np.asarray(obs.own_velocity, float)
        if not obs.ever_detected:
            return np.array([*(-g.kd * v_own[:2]), -g.kz * v_own[2]])
        rel = a.relative_position
        v_rel = a.relative_velocity if a.relative_velocity_valid else -v_own
        # Match a fast pad first: the extra gain acts only on the part of the
        # relative speed above the touchdown limit.
        speed = float(np.linalg.norm(v_rel[:2]))
        excess = max(speed - c.touchdown_xy_speed_m_s, 0.0)
        boost = g.kd_boost * excess / max(speed, 1e-9) * v_rel[:2]
        if g.boost_requires_fresh and not (obs.pad_detected and a.relative_velocity_valid):
            boost = np.zeros(2)
        lateral = g.kp * rel[:2] + g.kd * v_rel[:2] + boost + integral[:2]
        if not obs.pad_detected:
            trust = a.pad_freshness + (1.0 - a.pad_freshness) * g.hidden_lateral_scale
            lateral = integral[:2] + trust * (lateral - integral[:2])
        target = -descent_reference(rel, c)
        # View recovery: when the pad is about to leave the frame above the
        # terminal zone, climb to widen the footprint instead of chasing it
        # out of view (seed 4138: lost at 1.7 m, relative speed 1.45 m/s).
        h = -float(rel[2])
        if h > c.terminal_entry_height_m:
            ahead = rel + np.r_[v_rel[:2], 0.0] * c.visibility_horizon_s
            if bearing_fraction(ahead, c) > c.view_recovery_fraction:
                target = c.view_recovery_climb_m_s
        error = target - v_own[2]
        vertical = g.kz * error + integral[2]
        return np.array([lateral[0], lateral[1], vertical])
