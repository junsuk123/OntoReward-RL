"""The common training reward of spatial_current_semantics_v1 (``spatial-current-reward/4``).

Every arm trains on this one function; it reads simulator truth (and the policy's
own vision age) and nothing else reads it.  The verdict and the terminal table stay the spatial-reference/12
Evaluator's: this module only replaces the dense part.

Why (measured 2026-10-28 on diagnostic seeds 996001-996024, G0_G4_STATUS_KO.md
section 10): the Evaluator's dense terms are a position potential, which an
acceleration action moves only after double integration, and a readiness term
that is exp(-risk/2) and therefore flat beyond ~1 m of the pad.  The correlation
between an action's projection on the approach direction and the next 1-10
rewards was -0.05..0.08 under it, so a PPO advantage carried no information
about which way to push the action.  A potential on the VELOCITY error against
an approach field moves within the 0.1 s actuation delay of the action:
correlation 0.10-0.46 over the next 1-10 rewards on the same rollouts (0.10-0.17
for the untrained policy at nominal, 0.24-0.46 once the mean follows the field
a little).

The potential is
    Phi(s) = -(w_d * sqrt(|e|^2 + h^2) + w_v * |v - v_des(e, h)|)
with e the own-minus-pad horizontal offset, h the height above the contact
height and v the own-minus-pad velocity.  The step reward is the UNDISCOUNTED
difference Phi(s') - Phi(s) (constant Phi pays nothing, so hovering earns no
discount refund), Phi is kept at the terminal state (a fast contact or a far
miss is graded by it), and an explicit time and control cost make waiting cost
something.  The field ends in a contact the verdict accepts: sink at
``touchdown_sink_m_s`` (< the 0.30 m/s limit), descent only inside the
alignment radius (< the pad radius).

/3 (2026-10-29) added two costs for the Isaac handover failure (G0_G4_STATUS_KO.md
10.7, 11.6): a policy that answers a transient estimate with +-2 m/s^2 and
reverses 0.3-0.5 s later drives PX4 past 21 deg.  ``tilt_cost_per_s`` prices the
true thrust-axis tilt above ``tilt_cost_onset_deg``, quadratically up to the 21 deg
envelope.  The other, a command-change cost on mean((u - u_prev)^2), is REMOVED in
/4: it charged the SAMPLED action, so at the initial sigma exploration noise alone
paid ~0.65 per decision (22x the time cost) and pilot 5 dived into the pad for
~340 updates.  No reward on the sampled action can separate the two -- the jumps
worth pricing (0.5-1.2 normalised) are the size of the noise (0.47) -- while the
tilt the plant actually reaches filters the noise out: 0.000 per decision for the
untrained policy, 0.05 for the rough pilot-4 and 0.007 for the smooth pilot-5
policies (pilot6_design_checks, G0_G4_STATUS_KO.md 13).

/4 also makes the field answer a lost pad (``VelocityField.lost_view``): once the
policy's own vision age passes ``after_s``, outside the blind last stretch of a
landing, v_des_z turns from the sink to a climb toward ``search_height_m``.  Every
late Isaac failure on record descended on a stale estimate after the pad left the
view at 0.8-1.3 m and never climbed back to find it.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..spatial.environment import Evaluator
from .contracts import TILT_ENVELOPE_DEG, CommonReward, LostView, VelocityField


def terminal_reward(status: str) -> float:
    """The Evaluator's own terminal table, read from it (never restated)."""
    if status == "RUNNING":
        return 0.0
    if status in Evaluator.UNSAFE_STATUSES:
        return float(Evaluator.NOMINAL_UNSAFE_PENALTY)
    return float(Evaluator.BONUSES.get(status, 0.0))


def _ramp(x: float) -> float:
    return min(max(x, 0.0), 1.0)


def lost_view_weight(vision_age_s: float | None, height: float, distance: float,
                     lost: LostView) -> float:
    """How far the field has turned from the sink to the search climb, in [0, 1].
    ``vision_age_s`` None (no accepted frame yet) counts as an infinite age."""
    if vision_age_s is None or math.isinf(vision_age_s):
        age = 1.0
    elif math.isnan(vision_age_s) or vision_age_s < 0.0:
        raise ValueError("vision age must be None, +inf or a finite non-negative number")
    else:
        age = _ramp((vision_age_s - lost.after_s) / (lost.full_after_s - lost.after_s))
    where = max(_ramp((height - lost.terminal_height_m) / lost.ramp_m),
                _ramp((distance - lost.terminal_radius_m) / lost.ramp_m))
    return age * where


def desired_velocity(offset_xy: Any, height: float, field: VelocityField,
                     vision_age_s: float | None = 0.0) -> np.ndarray:
    """Own-minus-pad velocity the approach asks for at (offset, height above contact),
    given the policy's own vision age (0 = just seen)."""
    e = np.asarray(offset_xy, dtype=np.float64)
    if e.shape != (2,) or not np.all(np.isfinite(e)) or not math.isfinite(height):
        raise ValueError("offset must be a finite (2,) vector and height finite")
    h = max(float(height), 0.0)
    distance = float(np.linalg.norm(e))
    lateral = (-e * min(field.lateral_gain_per_s, field.lateral_speed_max_m_s / distance)
               if distance > 0.0 else np.zeros(2))
    radius = field.alignment_radius_m + field.alignment_radius_per_m * h
    align = min(max(1.0 - distance / radius, 0.0), 1.0)
    sink = min(field.descent_speed_max_m_s,
               field.touchdown_sink_m_s + field.descent_gain_per_s * h)
    lost = field.lost_view
    climb = min(lost.climb_speed_m_s, lost.climb_gain_per_s * max(lost.search_height_m - h, 0.0))
    w = lost_view_weight(vision_age_s, h, distance, lost)
    return np.array([lateral[0], lateral[1], (1.0 - w) * (-align * sink) + w * climb])


def potential(relative_position: Any, relative_velocity: Any, reward: CommonReward,
              contact_height_m: float, vision_age_s: float | None = 0.0) -> float:
    r = np.asarray(relative_position, dtype=np.float64)
    v = np.asarray(relative_velocity, dtype=np.float64)
    if r.shape != (3,) or v.shape != (3,) or not (np.all(np.isfinite(r))
                                                  and np.all(np.isfinite(v))):
        raise ValueError("relative position and velocity must be finite (3,) vectors")
    height = max(float(r[2]) - float(contact_height_m), 0.0)
    distance = math.hypot(float(np.linalg.norm(r[:2])), height)
    error = float(np.linalg.norm(v - desired_velocity(r[:2], height, reward.field,
                                                      vision_age_s)))
    return -(reward.distance_weight_per_m * distance + reward.velocity_weight_per_m_s * error)


class CommonRewardFunction:
    """reset(truth, vision_age_s=...) at handover, then step(...) once per decision.
    ``vision_age_s`` is the policy observation's own vision age (None before the
    first accepted frame); it is the only non-truth input."""

    def __init__(self, reward: CommonReward, contact_height_m: float):
        self.reward = reward
        self.contact_height_m = float(contact_height_m)
        self.previous: float | None = None

    def _phi(self, truth: Any, vision_age_s: float | None) -> float:
        return potential(truth.relative_position, truth.relative_velocity, self.reward,
                         self.contact_height_m, vision_age_s)

    def lost_view(self, truth: Any, vision_age_s: float | None) -> float:
        r = np.asarray(truth.relative_position, dtype=np.float64)
        return lost_view_weight(vision_age_s, max(float(r[2]) - self.contact_height_m, 0.0),
                                float(np.linalg.norm(r[:2])), self.reward.field.lost_view)

    def reset(self, truth: Any, *, vision_age_s: float | None) -> float:
        self.previous = self._phi(truth, vision_age_s)
        return self.previous

    def tilt_excess(self, truth: Any) -> float:
        roll, pitch = (float(a) for a in np.asarray(truth.roll_pitch)[:2])
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, math.cos(roll) * math.cos(pitch)))))
        onset = self.reward.tilt_cost_onset_deg
        return max(0.0, (tilt - onset) / (TILT_ENVELOPE_DEG - onset))

    def step(self, truth: Any, *, status: str, dt: float, action_normalized: Any,
             vision_age_s: float | None) -> tuple[float, dict[str, float]]:
        if self.previous is None:
            raise RuntimeError("reset the reward at handover before the first step")
        if not (math.isfinite(dt) and dt > 0.0):
            raise ValueError("dt must be finite and positive")
        u = np.asarray(action_normalized, dtype=np.float64)
        phi = self._phi(truth, vision_age_s)
        shaping = phi - self.previous
        cost = -(self.reward.time_cost_per_s
                 + self.reward.control_cost_per_s * float(np.mean(u * u))) * float(dt)
        tilt = -self.reward.tilt_cost_per_s * self.tilt_excess(truth) ** 2 * float(dt)
        terminal = terminal_reward(status)
        self.previous = phi
        total = shaping + cost + tilt + terminal
        return float(total), {"potential": phi, "shaping": shaping, "cost": cost,
                              "tilt": tilt, "terminal": terminal}
