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

    def __init__(self, max_velocity=(8.0, 8.0, 3.0),
                 max_yaw_rate_rad_s=math.radians(90.0),
                 max_acceleration=(6.0, 6.0, 4.0),
                 max_yaw_acceleration_rad_s2=math.radians(180.0), dt=0.1):
        self.max_velocity = np.asarray(max_velocity, dtype=float)
        self.max_acceleration = np.asarray(max_acceleration, dtype=float)
        if self.max_velocity.shape != (3,) or self.max_acceleration.shape != (3,):
            raise ValueError("velocity and acceleration limits must have three values")
        self.max_yaw_rate = float(max_yaw_rate_rad_s)
        self.max_yaw_acceleration = float(max_yaw_acceleration_rad_s2)
        self.dt = float(dt)
        if self.dt <= 0 or np.any(self.max_velocity <= 0) or np.any(self.max_acceleration <= 0):
            raise ValueError("controller period and limits must be positive")
        self.reset()

    def reset(self) -> None:
        self._last = np.zeros(4, dtype=float)

    def command(self, normalized_action) -> VelocityCommand:
        action = np.asarray(normalized_action, dtype=float).reshape(-1)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError("velocity/yaw-rate action must contain four finite values")
        action = np.clip(action, -1.0, 1.0)
        target = np.r_[action[:3] * self.max_velocity,
                       action[3] * self.max_yaw_rate]
        delta_limit = np.r_[self.max_acceleration * self.dt,
                            self.max_yaw_acceleration * self.dt]
        command = self._last + np.clip(target - self._last, -delta_limit, delta_limit)
        command[:3] = np.clip(command[:3], -self.max_velocity, self.max_velocity)
        command[3] = float(np.clip(command[3], -self.max_yaw_rate, self.max_yaw_rate))
        self._last = command
        return VelocityCommand(command[:3].copy(), float(command[3]))
