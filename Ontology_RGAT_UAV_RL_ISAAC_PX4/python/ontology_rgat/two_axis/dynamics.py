"""Planar pitch/thrust surrogate used by every comparison arm."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .config import DynamicsConfig


@dataclass(frozen=True)
class PlanarState:
    x_m: float
    z_m: float
    vx_m_s: float
    vz_m_s: float
    theta_rad: float
    pitch_rate_rad_s: float
    thrust_n: float
    time_s: float = 0.0

    @property
    def y_m(self) -> float:
        return 0.0

    @property
    def vy_m_s(self) -> float:
        return 0.0

    @property
    def roll_rad(self) -> float:
        return 0.0

    @property
    def yaw_rad(self) -> float:
        return 0.0


@dataclass(frozen=True)
class InnerLoopSetpoint:
    theta_rad: float
    thrust_n: float


def acceleration_to_thrust_pitch(request_m_s2: np.ndarray,
                                 config: DynamicsConfig) -> InnerLoopSetpoint:
    request = np.asarray(request_m_s2, dtype=float).reshape(2)
    ax = float(np.clip(request[0], -config.ax_max_m_s2, config.ax_max_m_s2))
    az = float(np.clip(request[1], -config.az_max_m_s2, config.az_max_m_s2))
    theta = math.atan2(ax, config.gravity_m_s2 + az)
    theta = float(np.clip(theta, -config.pitch_limit_rad, config.pitch_limit_rad))
    thrust = config.mass_kg * math.hypot(ax, config.gravity_m_s2 + az)
    thrust = float(np.clip(thrust, 0.0,
                           config.max_thrust_weight_ratio * config.mass_kg
                           * config.gravity_m_s2))
    return InnerLoopSetpoint(theta, thrust)


def realized_acceleration(state: PlanarState,
                          config: DynamicsConfig) -> np.ndarray:
    """World-frame net acceleration; gravity is subtracted exactly once."""
    return np.array([
        state.thrust_n * math.sin(state.theta_rad) / config.mass_kg,
        state.thrust_n * math.cos(state.theta_rad) / config.mass_kg
        - config.gravity_m_s2,
    ], dtype=float)


def step_planar(state: PlanarState, request_m_s2: np.ndarray, dt_s: float,
                config: DynamicsConfig, *, pitch_disturbance_rad_s2: float = 0.0
                ) -> PlanarState:
    """Advance actual attitude, thrust and translation by one physics step."""
    dt = float(dt_s)
    if dt <= 0.0:
        raise ValueError("physics step must be positive")
    setpoint = acceleration_to_thrust_pitch(request_m_s2, config)
    omega = config.pitch_natural_frequency_rad_s
    pitch_accel = (omega ** 2 * (setpoint.theta_rad - state.theta_rad)
                   - 2.0 * config.pitch_damping_ratio * omega
                   * state.pitch_rate_rad_s + float(pitch_disturbance_rad_s2))
    pitch_rate = float(np.clip(state.pitch_rate_rad_s + pitch_accel * dt,
                               -config.pitch_rate_limit_rad_s,
                               config.pitch_rate_limit_rad_s))
    theta = state.theta_rad + pitch_rate * dt
    if abs(theta) >= config.pitch_limit_rad:
        theta = float(np.clip(theta, -config.pitch_limit_rad, config.pitch_limit_rad))
        if np.sign(pitch_rate) == np.sign(theta):
            pitch_rate = 0.0
    alpha = 1.0 - math.exp(-dt / config.thrust_time_constant_s)
    thrust = state.thrust_n + alpha * (setpoint.thrust_n - state.thrust_n)
    provisional = replace(state, theta_rad=theta,
                          pitch_rate_rad_s=pitch_rate, thrust_n=thrust)
    accel = realized_acceleration(provisional, config)
    x = state.x_m + state.vx_m_s * dt + 0.5 * accel[0] * dt ** 2
    z = state.z_m + state.vz_m_s * dt + 0.5 * accel[1] * dt ** 2
    return PlanarState(
        x_m=float(x), z_m=float(z),
        vx_m_s=float(state.vx_m_s + accel[0] * dt),
        vz_m_s=float(state.vz_m_s + accel[1] * dt),
        theta_rad=theta, pitch_rate_rad_s=pitch_rate,
        thrust_n=float(thrust), time_s=state.time_s + dt)


def reachable_pitch_envelope(config: DynamicsConfig) -> tuple[float, float]:
    """Steady setpoint envelope reachable from the configured acceleration box."""
    candidates = [
        acceleration_to_thrust_pitch(np.array([ax, az]), config).theta_rad
        for ax in (-config.ax_max_m_s2, config.ax_max_m_s2)
        for az in (-config.az_max_m_s2, config.az_max_m_s2)
    ]
    return min(candidates), max(candidates)
