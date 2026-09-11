"""Safe mapping from normalized actions to PX4 velocity/yaw-rate setpoints."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class VelocityCommand:
    velocity_body_heading_m_s: np.ndarray
    yaw_rate_rad_s: float

    def as_array(self) -> np.ndarray:
        return np.r_[self.velocity_body_heading_m_s, self.yaw_rate_rad_s]


class VelocityYawRateController:
    """Saturating and rate-limited common low-level command adapter.

    PX4 owns attitude and motor stabilization.  This object is deterministic;
    changing reward modes cannot change its gains or limits.
    """

    def __init__(self, max_velocity=(2.0, 2.0, 1.0),
                 max_yaw_rate_rad_s=math.radians(60.0),
                 max_acceleration=(1.5, 1.5, 1.0),
                 max_yaw_acceleration_rad_s2=math.radians(90.0), dt=0.1,
                 curriculum_min_action_scale=0.35):
        self.max_velocity = np.asarray(max_velocity, dtype=float)
        self.max_acceleration = np.asarray(max_acceleration, dtype=float)
        if self.max_velocity.shape != (3,) or self.max_acceleration.shape != (3,):
            raise ValueError("velocity and acceleration limits must have three values")
        self.max_yaw_rate = float(max_yaw_rate_rad_s)
        self.max_yaw_acceleration = float(max_yaw_acceleration_rad_s2)
        self.dt = float(dt)
        self.curriculum_min_action_scale = float(curriculum_min_action_scale)
        if self.dt <= 0 or np.any(self.max_velocity <= 0) or np.any(self.max_acceleration <= 0):
            raise ValueError("controller period and limits must be positive")
        if not 0.0 < self.curriculum_min_action_scale <= 1.0:
            raise ValueError("minimum action-envelope scale must be in (0, 1]")
        self.set_curriculum(1.0)
        self.reset()

    @classmethod
    def from_mapping(cls, control=None, *, dt=None):
        """Build the shared controller from the experiment control section."""
        control = control or {}
        period = float(control.get("dt_seconds", 0.1) if dt is None else dt)
        return cls(
            max_velocity=control.get("max_velocity_m_s", (2.0, 2.0, 1.0)),
            max_yaw_rate_rad_s=math.radians(float(
                control.get("max_yaw_rate_deg_s", 60.0))),
            max_acceleration=control.get(
                "max_acceleration_m_s2", (1.5, 1.5, 1.0)),
            max_yaw_acceleration_rad_s2=math.radians(float(
                control.get("max_yaw_acceleration_deg_s2", 90.0))),
            dt=period,
            curriculum_min_action_scale=float(
                control.get("curriculum_min_action_scale", 0.35)),
        )

    def set_curriculum(self, curriculum: float) -> None:
        """Scale the safe envelope from hover/slow-follow to its final limits."""
        value = float(curriculum)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("curriculum must be finite and in [0, 1]")
        self.curriculum = value
        self.action_scale = (self.curriculum_min_action_scale
                             + (1.0 - self.curriculum_min_action_scale) * value)

    def reset(self) -> None:
        self._last = np.zeros(4, dtype=float)

    def command(self, normalized_action) -> VelocityCommand:
        action = np.asarray(normalized_action, dtype=float).reshape(-1)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError("velocity/yaw-rate action must contain four finite values")
        action = np.clip(action, -1.0, 1.0)
        velocity_limit = self.max_velocity * self.action_scale
        yaw_rate_limit = self.max_yaw_rate * self.action_scale
        acceleration_limit = self.max_acceleration * self.action_scale
        yaw_acceleration_limit = self.max_yaw_acceleration * self.action_scale
        target = np.r_[action[:3] * velocity_limit,
                       action[3] * yaw_rate_limit]
        delta_limit = np.r_[acceleration_limit * self.dt,
                            yaw_acceleration_limit * self.dt]
        command = self._last + np.clip(target - self._last, -delta_limit, delta_limit)
        command[:3] = np.clip(command[:3], -velocity_limit, velocity_limit)
        command[3] = float(np.clip(command[3], -yaw_rate_limit, yaw_rate_limit))
        self._last = command
        return VelocityCommand(command[:3].copy(), float(command[3]))
