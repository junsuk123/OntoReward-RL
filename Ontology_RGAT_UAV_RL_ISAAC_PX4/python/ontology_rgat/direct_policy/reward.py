"""Pure reward-v5 and terminal semantics, shared by both dimensions/backends."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from .contracts import RewardSpec, TerminationSpec


@dataclass(frozen=True)
class TruthState:
    relative_position: tuple[float, ...]       # pad - vehicle horizontally, height last
    relative_velocity: tuple[float, ...]       # pad - vehicle horizontally, own vz last
    attitude: tuple[float, ...]                # pitch, or roll/pitch
    angular_rate: tuple[float, ...]


@dataclass(frozen=True)
class ViewMeasurement:
    detected: bool
    bearing_rad: tuple[float, ...] = ()
    fov_rad: tuple[float, ...] = (math.radians(60),)


@dataclass(frozen=True)
class TerminalEvent:
    occurred: bool = False
    reason: str = ""
    time_s: float = 0.0
    physical_contact: bool = False
    mechanically_safe: bool = False


def _bounded_square(value):
    square = np.square(value)
    return square / (1.0 + square)


def _goal_cost(truth: TruthState, reward: RewardSpec, aim_slope: float) -> float:
    r = np.asarray(truth.relative_position, dtype=float)
    height = max(float(r[-1]), 0.0)
    horizontal = r[:-1]
    if len(horizontal) == 1:
        horizontal_error = horizontal[0] - aim_slope * height
    else:
        horizontal_error = float(np.linalg.norm(horizontal))
    x_cost = _bounded_square(horizontal_error / reward.goal_length_x_m)
    h_cost = _bounded_square(height / reward.goal_length_h_m)
    return float(reward.goal_horizontal_share * x_cost
                 + (1-reward.goal_horizontal_share) * h_cost)


def _running_goal_cost(truth: TruthState, reward: RewardSpec, aim_slope: float) -> float:
    r = np.asarray(truth.relative_position, dtype=float)
    height = max(float(r[-1]), 0.0)
    horizontal = r[:-1]
    horizontal_error = (horizontal[0] - aim_slope * height if len(horizontal) == 1
                        else float(np.linalg.norm(horizontal)))
    qx, qh, delta = (horizontal_error / reward.goal_length_x_m,
                     height / reward.goal_length_h_m, reward.goal_huber_delta)
    rho = lambda q: 2 * delta * delta * (math.sqrt(1 + (q / delta) ** 2) - 1)
    return reward.goal_horizontal_share * rho(qx) + (1-reward.goal_horizontal_share) * rho(qh)


def _tracking_cost(truth: TruthState, reward: RewardSpec, aim_slope: float) -> float:
    r = np.asarray(truth.relative_position, dtype=float)
    v = np.asarray(truth.relative_velocity, dtype=float)
    height, vertical_speed = max(float(r[-1]), 0), float(v[-1])
    horizontal, horizontal_velocity = r[:-1], v[:-1]
    if len(horizontal) == 1:
        cross = horizontal[0] - aim_slope * height
        target = aim_slope * vertical_speed - reward.approach_position_rate * cross
        target = float(np.clip(target, -reward.target_relative_speed_m_s,
                               reward.target_relative_speed_m_s))
        error = horizontal_velocity[0] - target
        return float(_bounded_square(error / reward.velocity_length_m_s))
    distance = float(np.linalg.norm(horizontal))
    direction = horizontal / max(distance, 1e-12)
    target = -reward.approach_position_rate * horizontal
    magnitude = np.linalg.norm(target)
    if magnitude > reward.target_relative_speed_m_s:
        target = direction * -reward.target_relative_speed_m_s
    return float(_bounded_square(np.linalg.norm(horizontal_velocity-target)
                                 / reward.velocity_length_m_s))


def _vertical_cost(truth: TruthState, reward: RewardSpec) -> float:
    height = max(float(truth.relative_position[-1]), 0)
    vertical_speed = float(truth.relative_velocity[-1])
    target = -min(reward.target_descent_speed_m_s,
                  reward.vertical_position_rate * height)
    return float(_bounded_square((vertical_speed-target) / reward.vertical_speed_length_m_s))


def _readiness(truth: TruthState, reward: RewardSpec, terminal: TerminationSpec) -> float:
    r = np.asarray(truth.relative_position, dtype=float)
    v = np.asarray(truth.relative_velocity, dtype=float)
    horizontal, velocity = r[:-1], v[:-1]
    distance = float(np.linalg.norm(horizontal))
    closing = min(terminal.touchdown_xy_speed_m_s, 0.6 * distance)
    desired_xy = -horizontal / max(distance, 1e-12) * closing
    desired_z = -min(0.8 * terminal.touchdown_z_speed_m_s,
                     reward.vertical_position_rate * max(float(r[-1]), 0))
    footprint = (terminal.pad_half_length_m if len(horizontal) == 1 else
                 terminal.pad_half_width_m)
    risk = float(np.sum(np.square(horizontal / footprint)))
    risk += (max(float(r[-1]), 0) / reward.readiness_height_m) ** 2
    risk += float(np.sum(np.square((velocity-desired_xy)
                                   / terminal.touchdown_xy_speed_m_s)))
    risk += ((float(v[-1])-desired_z) / terminal.touchdown_z_speed_m_s) ** 2
    risk += float(np.sum(np.square(np.asarray(truth.attitude)
                                   / terminal.touchdown_tilt_rad)))
    risk += float(np.sum(np.square(np.asarray(truth.angular_rate)
                                   / terminal.touchdown_rate_rad_s)))
    return math.exp(-0.5 * min(risk, 100.0))


def reward_v5(previous: TruthState, current: TruthState,
              measurement: ViewMeasurement, normalized_action,
              dt_s: float, event: TerminalEvent,
              reward: RewardSpec = RewardSpec(),
              terminal: TerminationSpec = TerminationSpec(),
              *, camera_pitch_offset_rad: float = -math.pi/6) -> tuple[float, dict]:
    aim_slope = math.tan(-camera_pitch_offset_rad)
    goal = _goal_cost(current, reward, aim_slope)
    old_goal = _goal_cost(previous, reward, aim_slope)
    running_goal = _running_goal_cost(current, reward, aim_slope)
    track, old_track = (_tracking_cost(current, reward, aim_slope),
                        _tracking_cost(previous, reward, aim_slope))
    vertical, old_vertical = (_vertical_cost(current, reward),
                              _vertical_cost(previous, reward))
    if measurement.detected and measurement.bearing_rad:
        bearing = np.asarray(measurement.bearing_rad)
        fov = np.broadcast_to(np.asarray(measurement.fov_rad), bearing.shape)
        view = min(1.0, float(np.sum(np.square(bearing / (fov/2)))))
    else:
        view = 1.0
    control = 0.5 * float(np.sum(np.square(normalized_action)))
    running = dt_s / reward.reference_time_s * (
        reward.goal_weight * running_goal + reward.view_weight * view
        + reward.control_weight * control)
    ready, old_ready = (_readiness(current, reward, terminal),
                        _readiness(previous, reward, terminal))
    readiness_reward = reward.readiness_weight * (ready-old_ready)
    old_phi = (-reward.potential_weight * old_goal
               - reward.velocity_potential_weight * old_track
               - reward.vertical_potential_weight * old_vertical)
    new_phi = 0.0 if event.occurred else (
        -reward.potential_weight * goal
        -reward.velocity_potential_weight * track
        -reward.vertical_potential_weight * vertical)
    shaping = math.exp(-dt_s/reward.discount_tau_s) * new_phi - old_phi
    terminal_bonus = dict(reward.terminal).get(event.reason, 0.0) if event.occurred else 0.0
    total = terminal_bonus-running+readiness_reward+shaping
    return float(total), {
        "goal_cost": goal, "running_goal_cost": running_goal, "view_cost": view,
        "control_cost": control, "scaled_running_cost": running,
        "landing_readiness": ready, "previous_landing_readiness": old_ready,
        "readiness_reward": readiness_reward, "track_cost": track,
        "vertical_approach_cost": vertical, "potential_shaping": shaping,
        "terminal_bonus": terminal_bonus, "dt": dt_s,
    }


def classify_terminal(truth: TruthState, *, elapsed_s: float, contact: bool,
                      hard_envelope_violation: bool = False,
                      terminal: TerminationSpec = TerminationSpec()) -> TerminalEvent:
    """Direct-mode terminal classifier: no perception authorization gate."""
    r = np.asarray(truth.relative_position, dtype=float)
    v = np.asarray(truth.relative_velocity, dtype=float)
    horizontal = r[:-1]
    if contact:
        in_footprint = (abs(horizontal[0]) <= terminal.pad_half_length_m
                        if len(horizontal) == 1 else
                        (abs(horizontal[0]) <= terminal.pad_half_length_m
                         and abs(horizontal[1]) <= terminal.pad_half_width_m))
        safe = (float(np.linalg.norm(v[:-1])) <= terminal.touchdown_xy_speed_m_s
                and abs(float(v[-1])) <= terminal.touchdown_z_speed_m_s
                and float(np.linalg.norm(truth.attitude)) <= terminal.touchdown_tilt_rad
                and float(np.linalg.norm(truth.angular_rate)) <= terminal.touchdown_rate_rad_s)
        reason = "SUCCESS" if in_footprint and safe else (
            "MISSED_PAD_CONTACT" if not in_footprint else "UNSAFE_CONTACT")
        return TerminalEvent(True, reason, elapsed_s, True, in_footprint and safe)
    height = float(r[-1])
    if hard_envelope_violation or height > terminal.ceiling_m or height < terminal.floor_m-1e-9:
        return TerminalEvent(True, "SAFETY_ENVELOPE_VIOLATION", elapsed_s)
    if elapsed_s >= terminal.mission_time_s-1e-12:
        return TerminalEvent(True, "TASK_TIMEOUT", terminal.mission_time_s)
    return TerminalEvent(time_s=elapsed_s)
