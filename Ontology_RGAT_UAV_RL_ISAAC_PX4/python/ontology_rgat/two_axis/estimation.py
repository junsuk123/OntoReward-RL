"""Small causal constant-acceleration tracker shared by every policy arm."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config import EstimatorConfig
from .sensing import PadMeasurement


def average_acceleration(previous_speed_m_s: float, current_speed_m_s: float,
                         previous_time_s: float, current_time_s: float) -> float:
    dt = float(current_time_s) - float(previous_time_s)
    if dt <= 0.0:
        raise ValueError("finite difference requires increasing measurement time")
    return (float(current_speed_m_s) - float(previous_speed_m_s)) / dt


@dataclass(frozen=True)
class TrackEstimate:
    timestamp_s: float
    initialized: bool
    pad_x_m: float
    pad_vx_m_s: float
    pad_ax_m_s2: float
    position_std_m: float
    velocity_std_m_s: float
    acceleration_std_m_s2: float
    time_since_detection_s: float
    update_accepted: bool


class CausalPadEstimator:
    """Three-state Kalman filter whose only target input is a measurement."""

    def __init__(self, config: EstimatorConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._x = np.zeros(3, dtype=float)
        self._P = np.diag([100.0, 16.0, 4.0])
        self._timestamp: float | None = None
        self._last_detection: float | None = None
        self._initialized = False
        self._last_result: TrackEstimate | None = None

    def _predict(self, dt: float) -> None:
        F = np.array([[1.0, dt, 0.5 * dt * dt],
                      [0.0, 1.0, dt],
                      [0.0, 0.0, math.exp(-dt / 2.0)]])
        q = self.config.process_acceleration_std_m_s2 ** 2
        G = np.array([0.5 * dt * dt, dt, 1.0 - math.exp(-dt)])
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + q * np.outer(G, G)

    def update(self, measurement: PadMeasurement, *, own_x_m: float,
               own_vx_m_s: float = 0.0) -> TrackEstimate:
        if self.config.model == "alpha_beta_gamma_v28":
            return self._update_reference(measurement, own_x_m, own_vx_m_s)
        t = float(measurement.timestamp_s)
        if self._timestamp is not None:
            if t < self._timestamp - 1e-12:
                raise ValueError("estimator timestamps must be monotonic")
            if abs(t - self._timestamp) <= 1e-12:
                assert self._last_result is not None
                return self._last_result
            self._predict(t - self._timestamp)
        self._timestamp = t
        accepted = False
        if measurement.relative_x_valid:
            z = float(own_x_m + measurement.relative_x_m)
            if not self._initialized:
                self._x[:] = (z, 0.0, 0.0)
                self._P = np.diag([
                    self.config.measurement_position_std_m ** 2, 4.0, 1.0])
                self._initialized = True
                accepted = True
            else:
                H = np.array([1.0, 0.0, 0.0])
                R = self.config.measurement_position_std_m ** 2
                innovation = z - float(H @ self._x)
                S = float(H @ self._P @ H + R)
                if innovation * innovation <= 16.0 * S:
                    K = self._P @ H / S
                    self._x = self._x + K * innovation
                    self._P = (np.eye(3) - np.outer(K, H)) @ self._P
                    accepted = True
            if accepted:
                self._last_detection = t
        age = (float("inf") if self._last_detection is None
               else max(0.0, t - self._last_detection))
        diag = np.sqrt(np.maximum(np.diag(self._P), 0.0))
        result = TrackEstimate(
            timestamp_s=t, initialized=self._initialized,
            pad_x_m=float(self._x[0]), pad_vx_m_s=float(self._x[1]),
            pad_ax_m_s2=float(self._x[2]), position_std_m=float(diag[0]),
            velocity_std_m_s=float(diag[1]), acceleration_std_m_s2=float(diag[2]),
            time_since_detection_s=age, update_accepted=accepted)
        self._last_result = result
        return result

    def _update_reference(self, measurement, own_x, own_vx):
        """Upstream updatePadTrack.m: ABG gains calibrated for 100 Hz.

        Gains (.20, .02, .00005), decay (1.5 s), initial stds (2,3,2)
        and acceleration cap (3 m/s²) are part of this named model version.
        Own velocity supplies the reset's speed-matched prior, never pad truth.
        """
        t = float(measurement.timestamp_s)
        if self._timestamp is not None and t < self._timestamp-1e-12:
            raise ValueError("estimator timestamps must be monotonic")
        if self._timestamp is not None and abs(t-self._timestamp) <= 1e-12:
            return self._last_result
        dt = 0.0 if self._timestamp is None else t-self._timestamp
        std = (np.array([2.,3.,2.]) if self._last_result is None else np.array([
            self._last_result.position_std_m, self._last_result.velocity_std_m_s,
            self._last_result.acceleration_std_m_s2]))
        if self._initialized:
            self._x[0] += self._x[1]*dt + .5*self._x[2]*dt*dt
            self._x[1] += self._x[2]*dt
            self._x[2] *= math.exp(-dt/1.5)
            q = self.config.process_acceleration_std_m_s2
            std = np.hypot(std, [0.5*q*dt*dt, q*dt, q*math.sqrt(dt)])
        self._timestamp = t
        accepted = False
        if measurement.detected and measurement.relative_x_valid:
            measured_x = own_x + measurement.relative_x_m
            if not self._initialized:
                self._x[:] = (measured_x, own_vx, 0.0)
                self._initialized = accepted = True
            else:
                innovation = measured_x-self._x[0]
                noise = self.config.measurement_position_std_m
                if abs(innovation) <= max(4*max(std[0],noise), .25):
                    gap = max(t-self._last_detection, 1e-12)
                    self._x += np.array([.20, .02/gap, .0001/(gap*gap)])*innovation
                    self._x[2] = np.clip(self._x[2], -3., 3.)
                    std = np.maximum([noise, noise/gap*.02,
                        self.config.process_acceleration_std_m_s2*.25], std*np.array([.5,.8,.85]))
                    accepted = True
            if accepted:
                self._last_detection = t
        age = math.inf if self._last_detection is None else t-self._last_detection
        result = TrackEstimate(t,self._initialized,*map(float,self._x),*map(float,std),age,accepted)
        self._last_result = result
        return result
