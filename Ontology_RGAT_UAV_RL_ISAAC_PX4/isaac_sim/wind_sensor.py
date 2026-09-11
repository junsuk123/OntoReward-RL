"""A small, deterministic UAV wind-sensor model.

The wind field is simulator truth and is used by physics.  Learning must see a
measurement instead, so this module applies a per-episode bias, white noise and
a first-order sensor response before the value crosses ROS 2.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _vector(config: dict[str, Any], name: str, default: tuple[float, float, float]) -> np.ndarray:
    value = np.asarray(config.get(name, default), dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all() or np.any(value < 0.0):
        raise ValueError(f"wind.sensor.{name} must contain three finite non-negative values")
    return value


class WindSensor:
    """Three-axis sonic-anemometer approximation in the world ENU frame."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", True))
        self.noise_std = _vector(self.config, "noise_std_m_s", (0.12, 0.12, 0.08))
        self.bias_std = _vector(self.config, "bias_std_m_s", (0.06, 0.06, 0.04))
        self.time_constant = float(self.config.get("time_constant_s", 0.15))
        self.max_speed = float(self.config.get("max_speed_m_s", 25.0))
        if not np.isfinite(self.time_constant) or self.time_constant < 0.0:
            raise ValueError("wind.sensor.time_constant_s must be finite and non-negative")
        if not np.isfinite(self.max_speed) or self.max_speed <= 0.0:
            raise ValueError("wind.sensor.max_speed_m_s must be finite and positive")
        self.rng = np.random.default_rng(0)
        self.bias = np.zeros(3)
        self.filtered = np.zeros(3)
        self.last_time: float | None = None

    def reset(self, seed: int, sim_time: float) -> None:
        sensor_seed = int(self.config.get("seed", 83))
        self.rng = np.random.default_rng(int(seed) + sensor_seed)
        self.bias = self.rng.normal(0.0, self.bias_std)
        self.filtered = np.zeros(3)
        self.last_time = float(sim_time)

    def measure(self, true_wind_enu: np.ndarray, sim_time: float) -> np.ndarray:
        truth = np.asarray(true_wind_enu, dtype=float)
        if truth.shape != (3,) or not np.isfinite(truth).all():
            raise ValueError("true wind must contain three finite ENU values")
        if not self.enabled:
            return np.zeros(3)

        now = float(sim_time)
        dt = 0.0 if self.last_time is None else max(0.0, now - self.last_time)
        self.last_time = now
        if self.time_constant == 0.0 or dt == 0.0 and not np.any(self.filtered):
            self.filtered = truth.copy()
        else:
            alpha = 1.0 - np.exp(-dt / self.time_constant)
            self.filtered += alpha * (truth - self.filtered)

        measured = self.filtered + self.bias + self.rng.normal(0.0, self.noise_std)
        magnitude = float(np.linalg.norm(measured))
        if magnitude > self.max_speed:
            measured *= self.max_speed / magnitude
        return measured
