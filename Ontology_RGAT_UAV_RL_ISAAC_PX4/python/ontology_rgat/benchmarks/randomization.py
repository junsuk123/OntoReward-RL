"""Seeded Table-II domain randomization for paired benchmark episodes."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..initialization import curriculum_camera_entry


@dataclass(frozen=True)
class DomainRandomizationSample:
    velocity_gain_xy: float
    velocity_gain_z: float
    attitude_gain_roll_pitch: float
    attitude_gain_yaw: float
    external_force_n: np.ndarray
    external_torque_nm: np.ndarray
    initial_velocity_m_s: np.ndarray
    initial_angular_rate_rad_s: np.ndarray
    ground_texture_id: int
    ground_texture_scale: float
    brightness: float
    rgb_scale: np.ndarray
    light_direction_deg: float


def sample_domain_randomization(seed: int) -> DomainRandomizationSample:
    rng = np.random.default_rng(int(seed))
    uniform = rng.uniform
    return DomainRandomizationSample(
        velocity_gain_xy=float(uniform(2.7, 3.3)),
        velocity_gain_z=float(uniform(1.3, 1.7)),
        attitude_gain_roll_pitch=float(uniform(1.6, 1.85)),
        attitude_gain_yaw=float(uniform(0.25, 0.4)),
        external_force_n=uniform(-0.75, 0.75, 3),
        external_torque_nm=uniform(-4e-3, 4e-3, 3),
        initial_velocity_m_s=uniform(-1.0, 1.0, 3),
        initial_angular_rate_rad_s=np.deg2rad(uniform(-10.0, 10.0, 3)),
        ground_texture_id=int(rng.integers(1, 51)),
        ground_texture_scale=float(uniform(0.4, 1.2)),
        brightness=float(uniform(0.5, 1.0)),
        rgb_scale=uniform(0.5, 1.0, 3),
        light_direction_deg=float(uniform(45.0, 135.0)),
    )


def sample_initial_condition(seed: int, curriculum: float = 1.0) -> dict:
    """Seeded reset draw; ``c=1`` is the exact Table-I distribution."""
    rng = np.random.default_rng(int(seed))
    c = float(np.clip(curriculum, 0.0, 1.0))
    raw_position = np.array([
        rng.uniform(-3.0, 3.0), rng.uniform(-3.0, 3.0), rng.uniform(2.0, 8.0)])
    raw_yaw_deg = float(rng.uniform(-60.0, 60.0))
    position, yaw_deg = curriculum_camera_entry(
        raw_position, raw_yaw_deg, c)
    return {
        "relative_position_m": position,
        "platform_yaw_misalignment_rad": math.radians(yaw_deg),
        "platform_speed_m_s": c * rng.uniform(0.0, 8.0),
        "platform_yaw_rate_rad_s": 0.0,
        "speed_step_m_s": c * rng.uniform(-0.5, 0.5),
        "yaw_rate_step_rad_s": math.radians(c * rng.uniform(-3.0, 3.0)),
    }
