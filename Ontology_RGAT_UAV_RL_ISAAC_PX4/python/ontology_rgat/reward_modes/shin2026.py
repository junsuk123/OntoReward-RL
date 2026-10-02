"""Hand-designed Shin-style reward on the common benchmark state.

Table-III equations, weights and terminal values are transcribed from the
paper.  The frame/sign contract follows the paper: relative position is the
platform in the drone body frame and ``Delta z > 0`` is undershoot.

Declared deviations (see ``docs/PAPER_FIDELITY.md``).

1. Table III's fifth shaping term is ``-2|omega_z|``: it prices the yaw-rate
   channel of the paper's four-dimensional action.  The reduced planar envelope
   has no yaw channel -- heading is a constraint of the experiment, not a
   control -- so that term would be identically zero and the reward would
   silently drop to four components.  It is replaced by the same penalty on the
   channel that took the yaw channel's place, the longitudinal tilt command.

2. 2026-10-02 re-weighting.  Deviation 1 preserved Table III's *weight* but not
   its *role*: yaw rate is gratuitous in the paper, whereas tilt is the planar
   envelope's only longitudinal actuator, so ``-2|tilt|`` taxed the approach
   itself.  Measured over 40 episodes per arm, the per-episode decomposition
   was attitude -69.6, vertical speed -48.2, undershoot -40.5, active
   perception -6.0, task -4.3 and lateral progress -3.4 -- the approach signal
   was the smallest term and negative.  Both arms duly stopped approaching; the
   proposed arm touched the deck once in 80 episodes while holding it in view
   from 5.7 m out.  The re-weighting makes the task outbid its own actuator:
   lateral progress 1.0 -> 5.0, attitude 2.0 -> 0.1, success +10 -> +50, and a
   timeout -- previously the only free terminal -- now scores -5.

Every weight is shared by both learned arms, so the comparison's single factor
remains the situation graph.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..controllers import PLANAR_ACTION_DIM
from .sparse import sparse_terminal_reward


@dataclass(frozen=True)
class ShinRewardConfig:
    # These defaults ARE Table III. The Shin baseline reproduction depends on
    # them and ``tests/test_shin2026_paper_fidelity.py`` checks them against an
    # independent transcription of the table, so do not re-tune them here.
    # An experiment that needs different weights declares them in its own
    # config under ``reward:`` -- see ``planar_envelope_reward_config`` and the
    # ``planar_reward_weights`` deviation.
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
    # Table III scores a horizon timeout at zero: it is not one of the paper's
    # terminal outcomes. Experiments that price it do so in their own config.
    timeout_value: float = 0.0


#: ``reward:`` block key -> ``ShinRewardConfig`` field, per sub-block. The sets
#: are closed so a typo in a config cannot silently leave a weight at its paper
#: value, and every listed key is genuinely read: before 2026-10-02 the block
#: was hashed into the manifest but never reached the reward, so editing
#: ``reward.terminal.success`` changed nothing.
REWARD_SETTING_FIELDS = {
    "shaping": {
        "lateral_progress": "lateral_progress_weight",
        "vertical_progress": "vertical_progress_weight",
        "vertical_speed": "vertical_speed_weight",
        "undershoot": "undershoot_weight",
        "attitude": "attitude_weight",
    },
    "terminal": {
        "success": "success_value",
        "failure": "failure_value",
        "timeout": "timeout_value",
    },
    "active_perception": {
        "alpha": "active_alpha",
        "beta": "active_beta",
        "tau": "active_tau",
    },
}


def reward_config_from(settings, *, active_enabled: bool) -> ShinRewardConfig:
    """Build a reward config from an experiment's ``reward:`` block.

    Unlisted keys keep their Table-III value, so a config states only what it
    deviates on and the deviation is readable in one place.

    ``active_perception.enabled`` is deliberately ignored: whether an arm sees
    the active-perception term is a property of the pipeline spec, which is the
    experimental factor, and the caller passes it as ``active_enabled``.
    """
    block = dict(settings or {})
    unknown_blocks = set(block) - set(REWARD_SETTING_FIELDS)
    if unknown_blocks:
        raise ValueError(
            f"unknown reward blocks {sorted(unknown_blocks)}; "
            f"allowed: {sorted(REWARD_SETTING_FIELDS)}")
    overrides: dict[str, float] = {}
    for name, fields in REWARD_SETTING_FIELDS.items():
        section = dict(block.get(name) or {})
        section.pop("enabled", None)
        unknown = set(section) - set(fields)
        if unknown:
            raise ValueError(
                f"unknown reward weights {sorted(unknown)} in '{name}'; "
                f"allowed: {sorted(fields)}")
        for key, value in section.items():
            overrides[fields[key]] = float(value)
    return ShinRewardConfig(active_enabled=bool(active_enabled), **overrides)


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
                success_value=cfg.success_value, failure_value=cfg.failure_value,
                timeout_value=cfg.timeout_value)
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
