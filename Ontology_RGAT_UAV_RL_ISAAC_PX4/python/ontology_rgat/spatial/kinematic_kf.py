"""Constant-acceleration Kalman filter on the WMON sensor set (comparison baseline).

Same inputs as ``spatial.wmon``: per axis a position (intermittent, e.g. GNSS),
a velocity and an acceleration measurement every step.  State [p, v, a] per
axis, white-jerk process noise with spectral density ``q``:

    F = [[1, dt, dt^2/2], [0, 1, dt], [0, 0, 1]]
    Q = q [[dt^5/20, dt^4/8, dt^3/6], [dt^4/8, dt^3/3, dt^2/2], [dt^3/6, dt^2/2, dt]]

A missing position drops its measurement row.  Axes are independent and
share the noise model, so every axis runs in one batched update.  The output
at each step is the causal posterior, the quantity WMON's atoms estimate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class KalmanNoise:
    sigma_position: float
    sigma_velocity: float
    sigma_acceleration: float
    jerk_psd: float          # q, (m/s^3)^2 / Hz

    def __post_init__(self) -> None:
        for name in ("sigma_position", "sigma_velocity", "sigma_acceleration", "jerk_psd"):
            if not getattr(self, name) > 0.0:
                raise ValueError(f"{name} must be positive")

    def to_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}


def transition(dt: float) -> np.ndarray:
    return np.array([[1.0, dt, 0.5 * dt * dt], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])


def process_noise(dt: float, q: float) -> np.ndarray:
    return q * np.array([[dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
                         [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2],
                         [dt ** 3 / 6, dt ** 2 / 2, dt]])


class KinematicKalman:
    """Streaming filter over ``axes`` independent axes (state rows p, v, a)."""

    def __init__(self, noise: KalmanNoise, axes: int = 3) -> None:
        self.noise = noise
        self.axes = axes
        self.reset()

    def reset(self) -> None:
        self.x = np.zeros((self.axes, 3))
        self.P = np.zeros((self.axes, 3, 3))
        self.initialized = False

    def _r(self, valid: bool) -> tuple[np.ndarray, np.ndarray]:
        n = self.noise
        sig = np.array([n.sigma_position, n.sigma_velocity, n.sigma_acceleration])
        rows = [0, 1, 2] if valid else [1, 2]
        return np.eye(3)[rows], np.diag(sig[rows] ** 2)

    def step(self, position: np.ndarray, velocity: np.ndarray, acceleration: np.ndarray,
             position_valid: bool, dt: float) -> np.ndarray:
        """One causal step; returns the posterior state (axes, 3)."""
        z_all = np.stack([np.asarray(position, float), np.asarray(velocity, float),
                          np.asarray(acceleration, float)], axis=1)          # (axes, 3)
        if not self.initialized:
            if not position_valid:
                raise ValueError("the first step needs a position measurement")
            self.x = z_all.copy()
            _, R = self._r(True)
            self.P = np.broadcast_to(R, (self.axes, 3, 3)).copy()
            self.initialized = True
            return self.x.copy()
        F, Q = transition(dt), process_noise(dt, self.noise.jerk_psd)
        self.x = self.x @ F.T
        self.P = F @ self.P @ F.T + Q
        H, R = self._r(bool(position_valid))
        z = z_all[:, [0, 1, 2] if position_valid else [1, 2]]
        S = H @ self.P @ H.T + R                                            # (axes, m, m)
        K = self.P @ H.T @ np.linalg.inv(S)                                 # (axes, 3, m)
        innovation = z - self.x @ H.T
        self.x = self.x + np.einsum("aij,aj->ai", K, innovation)
        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ np.swapaxes(I_KH, -1, -2) + K @ R @ np.swapaxes(K, -1, -2)
        return self.x.copy()


def kalman_episode(noise: KalmanNoise, observation: np.ndarray, position_valid: Sequence[bool],
                   dt: float) -> np.ndarray:
    """Episode layout of ``wmon.infer_episode``: (3D, T) rows [p(D), v(D), a(D)] -> state."""
    obs = np.asarray(observation, dtype=np.float64)
    axes = obs.shape[0] // 3
    valid = np.asarray(position_valid, dtype=bool)
    kf = KinematicKalman(noise, axes)
    state = np.zeros_like(obs)
    for k in range(obs.shape[1]):
        x = kf.step(obs[:axes, k], obs[axes:2 * axes, k], obs[2 * axes:, k], valid[k], dt)
        state[:, k] = x.T.reshape(-1)
    return state
