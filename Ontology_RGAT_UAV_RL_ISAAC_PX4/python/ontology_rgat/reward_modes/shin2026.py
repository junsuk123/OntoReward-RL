"""Hand-designed Shin-style reward on the common benchmark state.

Table-III equations, weights and terminal values are transcribed from the
paper.  The frame/sign contract follows the paper: relative position is the
platform in the drone body frame and ``Delta z > 0`` is undershoot.

One declared deviation (see ``docs/PAPER_FIDELITY.md``).  Table III's fifth
shaping term is ``-2|omega_z|``: it prices the yaw-rate channel of the paper's
four-dimensional action.  The reduced planar envelope has no yaw channel --
heading is a constraint of the experiment, not a control -- so that term would
be identically zero and the reward would silently drop to four components.  It
is replaced by the same penalty on the channel that took the yaw channel's
place, the longitudinal tilt command, at the same weight: ``-2|tilt|``.  Its
role is unchanged (price gratuitous use of the attitude channel) and it applies
identically to every arm.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..controllers import PLANAR_ACTION_DIM
from .sparse import sparse_terminal_reward


@dataclass(frozen=True)
class ShinRewardConfig:
    lateral_progress_weight: float = 1.0
    vertical_progress_weight: float = 1.0
    vertical_speed_weight: float = 0.5
    undershoot_weight: float = 1.0
    # Table III's -2|omega_z|, re-based onto the planar envelope's tilt channel.
    attitude_weight: float = 2.0
    active_alpha: float = 0.1
    active_beta: float = 1.0
    active_tau: float = 0.01
    active_enabled: bool = True
    success_value: float = 10.0
    failure_value: float = -10.0


def active_perception_reward(next_estimation_loss: float,
                             cfg: ShinRewardConfig) -> float:
    """Section III-C: ``-alpha [beta (L_est[t+1] - tau)]_0^1``."""
    clipped = np.clip(cfg.active_beta * (float(next_estimation_loss) - cfg.active_tau),
                      0.0, 1.0)
    return -cfg.active_alpha * float(clipped)


# Compatibility name retained for callers created before the PDF was supplied.
def active_perception_approximation(current_error: float, next_error: float,
                                    cfg: ShinRewardConfig) -> float:
    del current_error
    return active_perception_reward(next_error, cfg)


class ShinReward:
    def __init__(self, config: ShinRewardConfig | None = None):
        self.config = config or ShinRewardConfig()

    def __call__(self, current_relative_state, next_relative_state, action,
                 *, drone_vertical_velocity: float | None = None,
                 current_estimation_error: float | None = None,
                 next_estimation_loss: float | None = None,
                 next_estimation_error: float | None = None,
                 physical_contact=False, crash=False, excessive_drift=False,
                 battery_depleted=False, terminal=False):
        cfg = self.config
        state = np.asarray(current_relative_state, dtype=float).reshape(-1)
        nxt = np.asarray(next_relative_state, dtype=float).reshape(-1)
        action = np.asarray(action, dtype=float).reshape(-1)
        if state.shape != (6,) or nxt.shape != (6,):
            raise ValueError("manual reward requires two six-dimensional relative states")
        if action.shape != (PLANAR_ACTION_DIM,):
            raise ValueError(
                "manual reward requires the three-dimensional planar action "
                "[a_fwd, a_z, tilt]")
        lateral = float(np.linalg.norm(state[:2]))
        next_lateral = float(np.linalg.norm(nxt[:2]))
        if drone_vertical_velocity is None:
            raise ValueError("Table-III vertical-speed term requires UAV body-frame vz")
        def clip_progress(value: float) -> float:
            return float(np.clip(value, -1.0, 1.0))
        task = sparse_terminal_reward(
                physical_contact=physical_contact, crash=crash,
                excessive_drift=excessive_drift, terminal=terminal,
                battery_depleted=battery_depleted,
                success_value=cfg.success_value, failure_value=cfg.failure_value)
        parts = {
            "task": task,
            "lateral_progress": cfg.lateral_progress_weight * clip_progress(lateral - next_lateral),
            "vertical_progress": (cfg.vertical_progress_weight
                                  * clip_progress(abs(float(state[2])) - abs(float(nxt[2])))
                                  / max(next_lateral, 1.0)),
            "vertical_speed_penalty": (-cfg.vertical_speed_weight
                                       * max(float(drone_vertical_velocity) + 0.5, 0.0)),
            "undershoot_penalty": (-cfg.undershoot_weight * float(nxt[2])
                                    if float(nxt[2]) > 0.0 else 0.0),
            "attitude_penalty": -cfg.attitude_weight * abs(float(action[2])),
            "active_perception": 0.0,
        }
        # The paper's final reward is piecewise: terminal outcomes replace,
        # rather than augment, shaping and active-perception terms.
        if task != 0.0:
            terminal_parts = {name: 0.0 for name in parts}
            terminal_parts["task"] = task
            return float(task), terminal_parts
        if cfg.active_enabled:
            if next_estimation_loss is None:
                next_estimation_loss = next_estimation_error
            if next_estimation_loss is None:
                raise ValueError("active perception is training-only and requires privileged estimation errors")
            parts["active_perception"] = active_perception_reward(
                next_estimation_loss, cfg)
        return float(sum(parts.values())), parts
