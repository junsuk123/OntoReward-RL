"""True spatial ENU acceleration boundary; not the legacy planar tilt action.

The policy has three translations, no learned attitude/yaw action. PX4 owns
attitude stabilization. Derived roll/pitch are diagnostics, never extra inputs.
This controller does not provide a landing supervisor or a 3D policy backend.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .planar_controller import STANDARD_GRAVITY


SPATIAL_ACTION_CONTRACT = "spatial-enu-net-acceleration-v1"
SPATIAL_DIRECT_ACTION_CONTRACT = "spatial-enu-direct-net-acceleration-v2"
SPATIAL_ACTION_NAMES = ("east_acceleration", "north_acceleration", "up_acceleration")


@dataclass(frozen=True)
class SpatialCommand:
    velocity_enu_m_s: np.ndarray
    acceleration_enu_m_s2: np.ndarray
    normalized_action: np.ndarray
    yaw_enu_rad: float
    derived_roll_pitch_rad: tuple[float, float]
    thrust_weight_ratio: float
    constrained: bool
    acceleration_only: bool = False

    def wire_payload(self):
        if self.acceleration_only:
            return {"action_contract": SPATIAL_DIRECT_ACTION_CONTRACT, "frame": "enu",
                    "acceleration_enu_m_s2": self.acceleration_enu_m_s2.tolist(),
                    "yaw_rad": self.yaw_enu_rad}
        return {"action_contract": SPATIAL_ACTION_CONTRACT, "frame": "enu",
                "velocity_enu_m_s": self.velocity_enu_m_s.tolist(),
                "acceleration_enu_m_s2": self.acceleration_enu_m_s2.tolist(),
                "yaw_rad": self.yaw_enu_rad}


class SpatialAccelerationController:
    """Shared bounded acceleration→velocity/acceleration setpoint adapter.

    Joint tilt/thrust constraints prevent independently valid x/y commands
    from exceeding resultant attitude authority. Limits cannot be relaxed by
    a policy arm. Call reset with measured own velocity at every handover.
    """
    def __init__(self, *, max_velocity=(3., 3., 1.5),
                 max_acceleration=(2., 2., 1.), max_tilt_deg=25.,
                 max_thrust_weight_ratio=1.5, dt=.1, acceleration_only=False):
        self.acceleration_only = bool(acceleration_only)
        self.max_velocity = np.asarray(max_velocity, dtype=float)
        self.max_acceleration = np.asarray(max_acceleration, dtype=float)
        self.dt = float(dt)
        self.max_tilt = math.radians(float(max_tilt_deg))
        self.max_thrust_weight_ratio = float(max_thrust_weight_ratio)
        for values in (self.max_velocity, self.max_acceleration):
            if values.shape != (3,) or not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError("spatial limits must contain three positive finite values")
        if (not math.isfinite(self.dt) or self.dt <= 0
                or not math.isfinite(self.max_tilt) or not 0 < self.max_tilt <= math.radians(25.)
                or not math.isfinite(self.max_thrust_weight_ratio)
                or not 1 < self.max_thrust_weight_ratio <= 2.):
            raise ValueError("invalid spatial period, tilt or thrust envelope")
        if (np.any(self.max_velocity > (10., 10., 5.))
                or np.any(self.max_acceleration > (8., 8., 4.))):
            raise ValueError("controller limits exceed spatial gateway bounds")
        self._velocity = None
        self._yaw = None

    @classmethod
    def from_mapping(cls, config):
        if config.get("action_contract") != SPATIAL_ACTION_CONTRACT:
            raise ValueError("spatial controller requires its explicit action contract")
        if config.get("frame") != "enu":
            raise ValueError("spatial controller requires ENU, not body-heading coordinates")
        return cls(max_velocity=config["max_velocity_m_s"],
                   max_acceleration=config["max_acceleration_m_s2"],
                   max_tilt_deg=config["max_tilt_deg"],
                   max_thrust_weight_ratio=config["max_thrust_weight_ratio"],
                   dt=config["dt_seconds"])

    def reset(self, *, own_velocity_enu_m_s, yaw_enu_rad):
        velocity = np.asarray(own_velocity_enu_m_s, dtype=float)
        yaw = float(yaw_enu_rad)
        if (velocity.shape != (3,) or not np.isfinite(velocity).all()
                or np.any(np.abs(velocity) > self.max_velocity)
                or not math.isfinite(yaw) or abs(yaw) > 4*math.pi):
            raise ValueError("handover requires own ENU velocity within the envelope and finite yaw")
        self._velocity = velocity.copy()
        self._yaw = yaw

    def command(self, normalized_action, *, own_velocity_enu_m_s=None):
        if self._velocity is None:
            raise RuntimeError("reset from own telemetry before spatial policy handover")
        raw = np.asarray(normalized_action, dtype=float)
        if raw.shape != (3,) or not np.isfinite(raw).all():
            raise ValueError("spatial action must be finite [ax, ay, az]")
        action = np.clip(raw, -1., 1.)
        requested = action * self.max_acceleration
        if self.acceleration_only:
            measured = np.asarray(own_velocity_enu_m_s, dtype=float)
            if measured.shape != (3,) or not np.isfinite(measured).all():
                raise ValueError("direct acceleration requires current measured own velocity")
            # No hidden integrated velocity reference in the direct contract.
            # A physical overspeed cannot be projected away in one timestep.
            self._velocity = measured.copy()
            lower = np.clip((-self.max_velocity-measured)/self.dt,
                            -self.max_acceleration, self.max_acceleration)
            upper = np.clip((self.max_velocity-measured)/self.dt,
                            -self.max_acceleration, self.max_acceleration)
            acceleration = np.clip(requested, lower, upper)
        else:
            acceleration = np.clip(requested,
                (-self.max_velocity-self._velocity)/self.dt,
                (self.max_velocity-self._velocity)/self.dt)
        max_force = STANDARD_GRAVITY * self.max_thrust_weight_ratio
        acceleration[2] = min(acceleration[2], max_force-STANDARD_GRAVITY)
        fz = STANDARD_GRAVITY + acceleration[2]
        horizontal_limit = min(fz*math.tan(self.max_tilt),
                               math.sqrt(max(0., max_force**2-fz**2)))
        xy = float(np.linalg.norm(acceleration[:2]))
        if xy > horizontal_limit:
            acceleration[:2] *= horizontal_limit / xy
        velocity = self._velocity + acceleration*self.dt
        self._velocity = velocity.copy()
        force_norm = float(np.linalg.norm(np.r_[acceleration[:2], fz]))
        c, s = math.cos(self._yaw), math.sin(self._yaw)
        fx = c*acceleration[0] + s*acceleration[1]
        fy = -s*acceleration[0] + c*acceleration[1]
        attitude = (-math.asin(float(np.clip(fy/force_norm, -1., 1.))), math.atan2(fx, fz))
        return SpatialCommand(velocity, acceleration.copy(), action.copy(), self._yaw,
                              attitude, force_norm/STANDARD_GRAVITY,
                              bool(not np.array_equal(raw, action)
                                   or not np.allclose(acceleration, requested, atol=1e-12, rtol=0)),
                              self.acceleration_only)
