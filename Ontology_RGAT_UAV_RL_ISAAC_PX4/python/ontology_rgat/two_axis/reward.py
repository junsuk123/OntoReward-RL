"""Common bounded running costs and single terminal outcome reward."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config import RewardConfig
from .safety import TerminalReason


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    running: float
    terminal: float
    goal_cost: float
    view_cost: float
    control_cost: float
    dt_s: float


def compute_reward(*, ex_true_m: float, h_true_m: float,
                   measured_bearing_rad: float, bearing_valid: bool,
                   normalized_policy_action: np.ndarray, fov_rad: float,
                   dt_s: float, terminal_reason: TerminalReason | None,
                   config: RewardConfig) -> RewardBreakdown:
    action = np.asarray(normalized_policy_action, dtype=float).reshape(2)
    zeta = ((float(ex_true_m) / config.horizontal_scale_m) ** 2
            + (float(h_true_m) / config.height_scale_m) ** 2)
    goal = zeta / (1.0 + zeta)
    if bearing_valid:
        view = min(1.0, (float(measured_bearing_rad) / (0.5 * fov_rad)) ** 2)
    else:
        view = 1.0
    control = 0.5 * float(np.dot(action, action))
    running = -(float(dt_s) / config.reference_duration_s) * (
        config.goal_weight * goal + config.view_weight * view
        + config.control_weight * control)
    terminal = config.bonus(terminal_reason.value if terminal_reason else None)
    total = running + terminal
    if not np.isfinite(total):
        raise FloatingPointError("reward must remain finite")
    return RewardBreakdown(float(total), float(running), float(terminal),
                           float(goal), float(view), float(control), float(dt_s))


def discount_for_dt(dt_s: float, discount_time_constant_s: float) -> float:
    return math.exp(-float(dt_s) / float(discount_time_constant_s))


def conservative_return_bounds(config: RewardConfig,
                               discount_time_constant_s: float,
                               mission_duration_s: float) -> dict[str, tuple[float, float]]:
    running_max = (mission_duration_s / config.reference_duration_s) * (
        config.goal_weight + config.view_weight + config.control_weight)
    terminal_discount = math.exp(-mission_duration_s / discount_time_constant_s)
    result: dict[str, tuple[float, float]] = {}
    for name, bonus in config.terminal_bonus:
        discounted = bonus * terminal_discount
        if bonus >= 0.0:
            result[name] = (discounted - running_max, bonus)
        else:
            result[name] = (bonus - running_max, discounted)
    return result
