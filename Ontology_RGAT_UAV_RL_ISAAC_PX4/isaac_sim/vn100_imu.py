"""Pegasus-compatible VectorNav VN-100 measurement model."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from pegasus.simulator.logic.rotations import rot_ENU_to_NED, rot_FLU_to_FRD
from pegasus.simulator.logic.sensors import Sensor
from pegasus.simulator.logic.sensors.geo_mag_utils import GRAVITY_VECTOR


class Vn100Imu(Sensor):
    """IMU with VN-100 noise, bias, range and deterministic turn-on state."""

    def __init__(self, config):
        super().__init__(sensor_type="IMU", update_rate=float(config["update_rate"]))
        self._rng = np.random.default_rng(int(config.get("seed", 101)))
        gyro = config["gyroscope"]
        accel = config["accelerometer"]
        self._gyro_noise = float(gyro["noise_density"])
        self._gyro_walk = float(gyro["random_walk"])
        self._gyro_tau = float(gyro["bias_correlation_time"])
        self._gyro_range = float(gyro["measurement_range"])
        self._gyro_bias = self._rng.normal(
            0.0, float(gyro["turn_on_bias_sigma"]), size=3)
        self._accel_noise = float(accel["noise_density"])
        self._accel_walk = float(accel["random_walk"])
        self._accel_tau = float(accel["bias_correlation_time"])
        self._accel_range = float(accel["measurement_range"])
        self._accel_bias = self._rng.normal(
            0.0, float(accel["turn_on_bias_sigma"]), size=3)
        self._previous_velocity = np.zeros(3)
        self._state = {
            "orientation": np.array([1.0, 0.0, 0.0, 0.0]),
            "angular_velocity": np.zeros(3),
            "linear_acceleration": np.zeros(3),
        }

    @property
    def state(self):
        return self._state

    def _bias_step(self, bias, walk: float, tau: float, dt: float):
        phi = np.exp(-dt / tau)
        sigma = np.sqrt(-walk * walk * tau / 2.0 * (np.exp(-2.0 * dt / tau) - 1.0))
        return phi * bias + sigma * self._rng.standard_normal(3)

    @Sensor.update_at_rate
    def update(self, state, dt: float):
        self._gyro_bias = self._bias_step(
            self._gyro_bias, self._gyro_walk, self._gyro_tau, dt)
        gyro = (np.asarray(state.angular_velocity, dtype=float)
                + self._gyro_noise / np.sqrt(dt) * self._rng.standard_normal(3)
                + self._gyro_bias)
        gyro = np.clip(gyro, -self._gyro_range, self._gyro_range)

        inertial_accel = ((np.asarray(state.linear_velocity, dtype=float)
                           - self._previous_velocity) / dt - GRAVITY_VECTOR)
        self._previous_velocity = np.asarray(state.linear_velocity, dtype=float).copy()
        accel = Rotation.from_quat(state.attitude).inv().apply(inertial_accel)
        self._accel_bias = self._bias_step(
            self._accel_bias, self._accel_walk, self._accel_tau, dt)
        accel = (accel + self._accel_noise / np.sqrt(dt) * self._rng.standard_normal(3)
                 + self._accel_bias)
        accel = np.clip(accel, -self._accel_range, self._accel_range)

        attitude_frd_ned = (
            rot_ENU_to_NED * (Rotation.from_quat(state.attitude) * rot_FLU_to_FRD))
        self._state = {
            "orientation": attitude_frd_ned.as_quat(),
            "angular_velocity": rot_FLU_to_FRD.apply(gyro),
            "linear_acceleration": rot_FLU_to_FRD.apply(accel),
        }
        return self._state
