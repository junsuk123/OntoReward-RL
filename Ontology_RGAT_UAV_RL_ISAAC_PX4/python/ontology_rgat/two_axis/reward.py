"""Common bounded running costs and single terminal outcome reward."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config import RewardConfig, SafetyConfig
from .safety import TerminalReason


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    running: float
    terminal: float
    goal_cost: float
    horizontal_cost: float
    height_cost: float
    view_cost: float
    control_cost: float
    readiness: float
    readiness_delta_reward: float
    dt_s: float
    potential_shaping: float = 0.0


def _saturate(normalized: float) -> float:
    """Map a normalized error to [0, 1) with a finite gradient everywhere."""
    squared = float(normalized) ** 2
    return squared / (1.0 + squared)


def landing_readiness(*, ex_true_m: float, relative_speed_m_s: float,
                      h_true_m: float, pitch_rad: float,
                      safety: SafetyConfig,
                      vertical_speed_m_s: float = 0.0,
                      pitch_rate_rad_s: float = 0.0,
                      reward_config: RewardConfig | None = None) -> float:
    """Bounded (0, 1] progress towards a mechanically admissible touchdown.

    Every factor is a touchdown admissibility condition scaled by its own
    tolerance, so readiness reaches 1 only where the vehicle could actually
    land, and rises only when it becomes more able to land rather than merely
    lower. It is paid as a difference (see ``compute_reward``), so holding a
    high-readiness state earns nothing and leaving one gives the credit back.

    The factors are smooth rather than clipped. A clipped gate is exactly zero
    once any one condition is outside tolerance, which made the whole product
    zero above 1 m -- the entire approach carried no gradient, and PPO settled
    into hovering to the deadline because nothing paid for closing the gap.
    """
    if reward_config is not None and reward_config.readiness_model == "velocity_target_v28":
        # Both ports use pad-minus-UAV: a pad ahead needs negative relative
        # speed (the UAV catching up), not velocity matching before alignment.
        ex, relative = ex_true_m, relative_speed_m_s
        desired_relative = -np.sign(ex) * min(safety.touchdown_relative_speed_m_s, .6*abs(ex))
        desired_vz = -min(.8*safety.touchdown_vertical_speed_m_s, .5*max(h_true_m,0))
        risk = ((ex/safety.pad_half_width_m)**2
                + (max(h_true_m,0)/reward_config.readiness_height_m)**2
                + ((relative-desired_relative)/safety.touchdown_relative_speed_m_s)**2
                + ((vertical_speed_m_s-desired_vz)/safety.touchdown_vertical_speed_m_s)**2
                + (pitch_rad/safety.touchdown_pitch_rad)**2
                + (pitch_rate_rad_s/safety.touchdown_pitch_rate_rad_s)**2)
        return float(math.exp(-.5*min(risk,100)))
    def soft(value: float, scale: float) -> float:
        normalized = abs(float(value)) / max(float(scale), 1e-9)
        return 1.0 / (1.0 + normalized * normalized)
    alignment = soft(ex_true_m, safety.touchdown_horizontal_error_m)
    relative = soft(relative_speed_m_s, safety.touchdown_relative_speed_m_s)
    attitude = soft(pitch_rad, safety.touchdown_pitch_rad)
    # Sink rate is a touchdown criterion in its own right and was missing:
    # readiness asked the vehicle to get LOW but never to arrive SLOWLY. With
    # it absent, PPO solved the horizontal problem (0.16 m error, 0.07 m/s
    # relative speed at contact, both well inside tolerance) and then dived:
    # 30 of 30 contacts failed on |vz| alone, median 1.10 m/s against a
    # 0.35 m/s limit.
    sink = soft(min(0.0, float(vertical_speed_m_s)),
                safety.touchdown_vertical_speed_m_s)
    descent = soft(max(0.0, h_true_m) - safety.touchdown_height_m,
                   safety.terminal_descent_height_m)
    return float(alignment * relative * attitude * sink * descent)


def compute_reward(*, ex_true_m: float, h_true_m: float,
                   measured_bearing_rad: float, bearing_valid: bool,
                   normalized_policy_action: np.ndarray, fov_rad: float,
                   dt_s: float, terminal_reason: TerminalReason | None,
                   config: RewardConfig, readiness: float = 0.0,
                   previous_readiness: float = 0.0,
                   previous_goal_cost: float | None = None,
                   discount_time_constant_s: float = 70.0) -> RewardBreakdown:
    action = np.asarray(normalized_policy_action, dtype=float).reshape(2)
    # Each axis saturates on its own scale before they are mixed. Sharing one
    # saturation hides a growing longitudinal error behind the altitude term
    # while the vehicle is still high, which teaches descent before tracking.
    horizontal = _saturate(float(ex_true_m) / config.horizontal_scale_m)
    height = _saturate(float(h_true_m) / config.height_scale_m)
    alpha = config.horizontal_priority
    goal = alpha * horizontal + (1.0 - alpha) * height
    if bearing_valid:
        view = min(1.0, (float(measured_bearing_rad) / (0.5 * fov_rad)) ** 2)
    else:
        view = 1.0
    control = 0.5 * float(np.dot(action, action))
    running = -(float(dt_s) / config.reference_duration_s) * (
        config.goal_weight * goal + config.view_weight * view
        + config.control_weight * control)
    # Pay only the change in readiness: a held state earns nothing, so a
    # near-pad hover cannot compete with an actual touchdown.
    readiness_delta = config.readiness_weight * (
        float(readiness) - float(previous_readiness))
    running += readiness_delta
    terminal = config.bonus(terminal_reason.value if terminal_reason else None)
    potential = 0.0
    if config.potential_weight:
        if previous_goal_cost is None:
            raise ValueError("potential shaping requires the preceding goal cost")
        phi_previous = -config.potential_weight * previous_goal_cost
        phi_next = 0.0 if terminal_reason else -config.potential_weight * goal
        potential = discount_for_dt(dt_s, discount_time_constant_s)*phi_next - phi_previous
    total = running + terminal + potential
    if not np.isfinite(total):
        raise FloatingPointError("reward must remain finite")
    return RewardBreakdown(float(total), float(running), float(terminal),
                           float(goal), float(horizontal), float(height),
                           float(view), float(control), float(readiness),
                           float(readiness_delta), float(dt_s), float(potential))


def goal_cost(ex_m: float, h_m: float, config: RewardConfig) -> float:
    return (config.horizontal_priority * _saturate(ex_m/config.horizontal_scale_m)
            + (1-config.horizontal_priority) * _saturate(h_m/config.height_scale_m))


def discount_for_dt(dt_s: float, discount_time_constant_s: float) -> float:
    return math.exp(-float(dt_s) / float(discount_time_constant_s))


def conservative_return_bounds(config: RewardConfig,
                               discount_time_constant_s: float,
                               mission_duration_s: float) -> dict[str, tuple[float, float]]:
    running_max = (mission_duration_s / config.reference_duration_s) * (
        config.goal_weight + config.view_weight + config.control_weight)
    # Readiness telescopes: the whole episode can pay at most one unit of it.
    running_max += config.readiness_weight + 2*config.potential_weight
    positive_progress_bound = config.readiness_weight + 2*config.potential_weight
    terminal_discount = math.exp(-mission_duration_s / discount_time_constant_s)
    result: dict[str, tuple[float, float]] = {}
    for name, bonus in config.terminal_bonus:
        discounted = bonus * terminal_discount
        if bonus >= 0.0:
            result[name] = (discounted - running_max, bonus + positive_progress_bound)
        else:
            result[name] = (bonus - running_max, discounted + positive_progress_bound)
    return result


def terminal_ordering_report(config: RewardConfig,
                             discount_time_constant_s: float,
                             mission_duration_s: float) -> dict[str, float]:
    """Check the ordering that actually governs the policy's choice.

    All terminal payments at a common termination time carry the same discount
    factor, so the discounted ordering at any time is the undiscounted ordering
    of the table. What can still invert it is the running cost: if the whole
    episode's running magnitude exceeds the gap between two adjacent outcome
    tiers, a policy can buy a better-ranked outcome by paying running cost, or
    escape running cost by taking a worse one. Earlier the gap was negative —
    SAFE_ABORT paid -3 against a -12 timeout, which made inducing a visual loss
    the cheapest way out of a hard episode.

    The returned ``margin`` is the smallest adjacent gap minus that running
    magnitude; the contract requires it to stay positive.
    """
    table = dict(config.terminal_bonus)
    unsafe = max(table[name] for name in config.unsafe_reasons)
    tiers = (("SUCCESS", table["SUCCESS"]),
             ("TASK_TIMEOUT", table["TASK_TIMEOUT"]),
             ("SAFE_ABORT", table["SAFE_ABORT"]),
             ("UNSAFE", unsafe))
    gaps = {f"{left}_over_{right}": lvalue - rvalue
            for (left, lvalue), (right, rvalue) in zip(tiers, tiers[1:])}
    running_magnitude = (mission_duration_s / config.reference_duration_s) * (
        config.goal_weight + config.view_weight + config.control_weight
    ) + config.readiness_weight
    # A late abort is discounted towards zero; an abort must stay worse than a
    # timeout even when it happens at the very deadline.
    late_abort = table["SAFE_ABORT"] * math.exp(
        -mission_duration_s / discount_time_constant_s)
    late_timeout = table["TASK_TIMEOUT"] * math.exp(
        -mission_duration_s / discount_time_constant_s)
    return {**gaps, "running_magnitude": float(running_magnitude),
            "margin": float(min(gaps.values()) - running_magnitude),
            "late_abort": float(late_abort), "late_timeout": float(late_timeout),
            "late_abort_still_worse": float(late_timeout - late_abort)}
