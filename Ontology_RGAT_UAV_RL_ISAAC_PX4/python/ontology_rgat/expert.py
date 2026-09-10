"""The behaviour policy the R-GAT dataset is labelled from."""
from __future__ import annotations

import numpy as np

from .config import Config
from .mathx import quat_to_euler_zyx

__all__ = ["expert_action", "PolicySpec"]


def _context(cur) -> tuple[np.ndarray, float]:
    """Deck velocity and energy margin, defaulting to the static full-pack case.

    This controller still has to run against a link that reports neither.
    """
    pad_vel = np.zeros(3)
    margin = 1.0
    if cur is None:
        return pad_vel, margin
    pv = np.asarray(cur.meas.get("pad_velocity", (0.0, 0.0, 0.0)), dtype=float).reshape(-1)
    if pv.size >= 2:
        pad_vel[:2] = pv[:2]
    margin = float(getattr(cur.sem, "energy_margin", 1.0))
    return pad_vel, margin


def expert_action(x: np.ndarray, cfg: Config, cur=None) -> np.ndarray:
    """PX4-compatible behaviour policy without simulator feedforward.

    The state is pad-relative, so the same PD that used to hold a fixed pad now
    chases a moving one: driving pad-relative position and velocity to zero *is*
    tracking the deck. Adding the deck's absolute velocity again would
    double-count its motion and continuously accelerate past a constant-speed
    lorry. The external backend adds two details instead:

    * the horizontal acceleration is rotated into the vehicle's current yaw
      before it becomes roll/pitch, so a yawed aircraft still corrects along
      the requested world ENU axis;
    * the descent is compressed when the energy margin is thin, so the
      demonstrations the R-GAT dataset is labelled from actually contain
      successful low-reserve landings. Without this every low-battery episode is
      a negative example and the potential has nothing to learn from.
    """
    p = np.asarray(x[0:3], dtype=float)
    v = np.asarray(x[3:6], dtype=float)
    rpy = quat_to_euler_zyx(x[6:10])
    _, margin = _context(cur)

    horizontal_error = float(np.linalg.norm(p[:2]))
    relative_speed = float(np.linalg.norm(v[:2]))

    # Nominal profile, then urgency: a thin margin buys descent rate, bounded so
    # the controller never asks for a touchdown speed the criteria would reject.
    vz_nominal = min(0.35, max(0.10, 0.12 + 0.08 * p[2]))
    urgency = min(1.0, max(0.0, 1.0 - margin))
    vz_des = -min(0.95 * cfg.criteria.vz, vz_nominal * (1.0 + 0.6 * urgency))
    # Do not descend beside the lorry. First match its relative motion; in the
    # flare, hold roughly 0.8 m above the roof until the centre and speed gates
    # are both satisfied. This turns a late chase into another alignment pass
    # instead of a guaranteed road impact.
    if p[2] > 1.0 and (horizontal_error > 1.25 or relative_speed > 1.0):
        vz_des = 0.0
    elif p[2] <= 1.0 and (horizontal_error > 0.25 or relative_speed > 0.30):
        vz_des = float(np.clip(0.8 * (0.8 - p[2]), 0.0, 0.35))
    elif p[2] <= 0.5:
        vz_des = max(vz_des, -0.08)
    elif p[2] <= 2.5:
        vz_des = max(vz_des, -0.15)
    # Keep the delayed PX4/Isaac vertical loop damped; safety comes from
    # beginning the low-speed flare early, not from a large feedback gain.
    az_cmd = 2.2 * (vz_des - v[2])
    collective = az_cmd / (cfg.sim.g * cfg.rl.collective_span)

    # Horizontal PD on pad-relative error. This already tracks constant deck
    # velocity because v is UAV velocity minus deck velocity.
    # Near-critical damping is intentional: the kinematic lorry acquires its
    # seeded cruise velocity at handover, so an under-damped chase crosses the
    # narrow roof repeatedly and never opens the safe descent gate.
    ax = -0.90 * p[0] - 2.10 * v[0]
    ay = -0.90 * p[1] - 2.10 * v[1]
    cy, sy = np.cos(rpy[2]), np.sin(rpy[2])
    accel_forward = cy * ax + sy * ay
    accel_left = -sy * ax + cy * ay
    limit = cfg.rl.max_roll_pitch
    pitch_des = float(np.clip(accel_forward / cfg.sim.g, -limit, limit))
    roll_des = float(np.clip(-accel_left / cfg.sim.g, -limit, limit))

    a = np.array([collective, roll_des / limit, pitch_des / limit,
                  -0.6 * rpy[2] / cfg.rl.max_yaw_rate])
    return np.clip(a, -1.0, 1.0)


class PolicySpec:
    """What ``run_episode`` needs to know to produce an action.

    ``kind`` is ``'expert'``, ``'expert_noisy'`` or ``'ppo'``.
    """

    def __init__(self, kind: str, *, agent=None, deterministic: bool = True,
                 noise_std: float = 0.0, rng: np.random.Generator | None = None):
        self.kind = kind.lower()
        self.agent = agent
        self.deterministic = bool(deterministic)
        self.noise_std = float(noise_std)
        self.rng = rng or np.random.default_rng()
        if self.kind not in {"expert", "expert_noisy", "ppo"}:
            raise ValueError(f"Unknown policy type: {kind}")
        if self.kind == "ppo" and agent is None:
            raise ValueError("a 'ppo' policy needs a trained agent")

    def action(self, cur, x: np.ndarray, cfg: Config) -> np.ndarray:
        if self.kind == "expert":
            return expert_action(x, cfg, cur)
        if self.kind == "expert_noisy":
            a = expert_action(x, cfg, cur)
            a = a + self.noise_std * self.rng.standard_normal(4)
            # Occasionally inject a larger perturbation, to create the failure
            # examples the potential needs as negative labels.
            if self.rng.random() < min(0.20, 2.0 * self.noise_std):
                a = a + np.array([0.2, 0.5, 0.5, 0.3]) * self.rng.standard_normal(4)
            return np.clip(a, -1.0, 1.0)
        return self.agent.act(cur.obs, deterministic=self.deterministic)[0]
