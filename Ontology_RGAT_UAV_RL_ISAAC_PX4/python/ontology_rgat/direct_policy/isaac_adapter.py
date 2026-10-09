"""Isaac/PX4 boundary helpers for the direct-policy profiles.

Imports stay pure: constructing a payload never imports ROS, starts Isaac, or
acquires a gateway.  The explicit runner performs those operations only after
``--allow-isaac``.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np

from .observation import (ConstantVelocityKalman2D, PlanarCalibration,
                          PlanarEstimate, SpatialEstimate, planar_vector,
                          spatial_vector)


def enu_to_ned(vector):
    east, north, up = np.asarray(vector, dtype=float)
    return np.array([north, east, -up], dtype=float)


def ned_to_enu(vector):
    north, east, down = np.asarray(vector, dtype=float)
    return np.array([east, north, -down], dtype=float)


def embed_planar_action(action_xz):
    action = np.asarray(action_xz, dtype=float)
    if action.shape != (2,) or not np.all(np.isfinite(action)):
        raise ValueError("planar action must be finite [a_x,a_z]")
    return np.array([action[0], 0.0, action[1]], dtype=float)


def direct_acceleration_payload(acceleration_enu, *, heading_rad: float,
                                decision_id: int, episode_id: str,
                                policy_version: int, stamp_s: float):
    acceleration = np.asarray(acceleration_enu, dtype=float)
    if acceleration.shape != (3,) or not np.all(np.isfinite(acceleration)):
        raise ValueError("direct acceleration must be a finite ENU vector")
    if not math.isfinite(heading_rad) or not math.isfinite(stamp_s):
        raise ValueError("heading/stamp must be finite")
    return {
        "contract": "spatial-enu-direct-net-acceleration-v2",
        "acceleration_enu_m_s2": acceleration.tolist(),
        "heading_rad": float(heading_rad),
        "decision_id": int(decision_id), "episode_id": str(episode_id),
        "policy_version": int(policy_version), "decision_stamp_s": float(stamp_s),
    }


def trajectory_setpoint_fields(acceleration_enu, heading_rad: float):
    """ROS2 TrajectorySetpoint values: only acceleration and yaw are active."""
    return {"position": [math.nan]*3, "velocity": [math.nan]*3,
            "acceleration": enu_to_ned(acceleration_enu).tolist(),
            "jerk": [math.nan]*3, "yaw": float(math.pi/2-heading_rad),
            "yawspeed": math.nan}


@dataclass(frozen=True)
class AuthorityRecord:
    episode_id: str
    decision_id: int
    policy_version: int
    capture_stamp_s: float | None
    receive_stamp_s: float | None
    decision_stamp_s: float
    command_stamp_s: float
    requested_acceleration_enu_m_s2: tuple[float, float, float]
    limited_acceleration_enu_m_s2: tuple[float, float, float]
    gateway_acceleration_ned_m_s2: tuple[float, float, float]
    measured_acceleration_enu_m_s2: tuple[float, float, float] | None
    control_profile: str
    emergency_intervention: bool

    def to_dict(self):
        return asdict(self)


class CommandOwnership:
    """Reject double writers and stale decisions before a command is emitted."""
    def __init__(self, owner: str = "matlab-port"):
        self.owner = owner
        self.episode_id = None
        self.last_decision = -1

    def reset(self, episode_id: str):
        self.episode_id, self.last_decision = str(episode_id), -1

    def validate(self, payload, *, writer: str):
        if writer != self.owner:
            raise RuntimeError(f"command writer {writer!r} does not own this route")
        if payload["episode_id"] != self.episode_id:
            raise RuntimeError("stale command belongs to another episode")
        if payload["decision_id"] <= self.last_decision:
            raise RuntimeError("duplicate or out-of-order decision")
        self.last_decision = payload["decision_id"]


class _ConstantVelocityKalman3D:
    """Small causal [position, velocity] filter for the 3-D extension."""
    def __init__(self, process_acceleration_std=(0.5, 0.5, 0.05),
                 initial_velocity_std=0.5):
        self.process_std = np.asarray(process_acceleration_std, dtype=float)
        self.initial_velocity_std = float(initial_velocity_std)
        self.reset()

    def reset(self):
        self.initialized = False
        self.state = np.zeros(6)
        self.covariance = np.zeros((6, 6))
        self.time = 0.0
        self.last_update_time = -math.inf
        self.updated = False

    def update(self, stamp, measurement, covariance, navigation_velocity):
        if self.initialized:
            dt = float(stamp)-self.time
            if dt < -1e-12:
                raise ValueError("pad estimator time moved backwards")
            if dt > 0:
                eye = np.eye(3)
                transition = np.block([[eye, dt*eye], [np.zeros((3, 3)), eye]])
                acceleration = np.diag(np.square(self.process_std))
                process = np.block([
                    [dt**3/3*acceleration, dt**2/2*acceleration],
                    [dt**2/2*acceleration, dt*acceleration],
                ])
                self.state = transition@self.state
                self.covariance = transition@self.covariance@transition.T+process
            self.time = float(stamp)
            self.updated = measurement is not None
            if measurement is not None:
                select = np.c_[np.eye(3), np.zeros((3, 3))]
                innovation = select@self.covariance@select.T+covariance
                gain = self.covariance@select.T@np.linalg.inv(innovation)
                self.state += gain@(measurement-select@self.state)
                reduction = np.eye(6)-gain@select
                self.covariance = (reduction@self.covariance@reduction.T
                                   + gain@covariance@gain.T)
        elif measurement is not None:
            self.state = np.r_[measurement, np.asarray(navigation_velocity, dtype=float)]
            self.covariance = np.block([
                [covariance, np.zeros((3, 3))],
                [np.zeros((3, 3)), self.initial_velocity_std**2*np.eye(3)],
            ])
            self.initialized = self.updated = True
            self.time = float(stamp)
        else:
            self.updated = False
        if self.updated:
            self.last_update_time = float(stamp)


class CausalIsaacObservation:
    """Adapt the deployed detector/EKF stream to the direct-policy vectors.

    PnP stays in the existing Isaac detector.  The adapter uses only its
    capture-aligned relative solve and own navigation; simulator truth never
    enters this object.
    """
    def __init__(self, dimension, *, pad_offset_z_m=0.6):
        if dimension not in (2, 3):
            raise ValueError("dimension must be 2 or 3")
        self.dimension = int(dimension)
        self.pad_offset_z_m = float(pad_offset_z_m)
        self.filter = (ConstantVelocityKalman2D(PlanarCalibration.source_default())
                       if dimension == 2 else _ConstantVelocityKalman3D())
        self.reset(None)

    def reset(self, episode_id):
        self.filter.reset()
        self.episode_id = episode_id
        self.last_sample_id = -1
        self.last_measurement = None
        self.last_capture_stamp = None
        self.last_receive_stamp = None
        self.updated_since_decision = False

    @staticmethod
    def _angles(measurement):
        from scipy.spatial.transform import Rotation
        q = np.asarray(measurement.quaternion, dtype=float)
        return Rotation.from_quat(q[[1, 2, 3, 0]]).as_euler("xyz")

    def ingest(self, measurement):
        if self.last_measurement is not None \
                and measurement.time_s < self.last_measurement.time_s-1e-12:
            raise ValueError("Isaac navigation time moved backwards")
        fresh = (measurement.optical_position is not None
                 and measurement.sample_id > self.last_sample_id)
        pad_deck = covariance = None
        if fresh:
            capture = (measurement.optical_own_position
                       if measurement.optical_own_position is not None
                       else measurement.own_position)
            pad_deck = np.asarray(capture)-np.asarray(measurement.optical_position)
            sigma = max(0.005, 0.02/max(float(measurement.confidence), 0.1))
            if self.dimension == 2:
                pad_deck = pad_deck[[0, 2]].copy()
                # planar_vector adds the source's deck offset back.
                pad_deck[1] -= self.pad_offset_z_m
                covariance = np.eye(2)*sigma*sigma
            else:
                covariance = np.eye(3)*sigma*sigma
            self.last_sample_id = measurement.sample_id
            self.last_capture_stamp = (measurement.optical_time_s
                                       if measurement.optical_time_s is not None
                                       else measurement.time_s)
            self.last_receive_stamp = measurement.time_s
            self.updated_since_decision = True
        velocity = (measurement.own_velocity[[0, 2]] if self.dimension == 2
                    else measurement.own_velocity)
        self.filter.update(measurement.time_s, pad_deck, covariance, velocity)
        self.last_measurement = measurement

    def vector(self):
        measurement = self.last_measurement
        if measurement is None:
            raise RuntimeError("an Isaac measurement is required before observation")
        angles = self._angles(measurement)
        age = (measurement.time_s-self.filter.last_update_time
               if self.filter.initialized else math.inf)
        updated = self.updated_since_decision
        self.updated_since_decision = False
        if self.dimension == 2:
            estimate = PlanarEstimate(
                pad_position_xz=tuple(self.filter.state[:2]),
                pad_velocity_xz=tuple(self.filter.state[2:]),
                own_position_xz=tuple(measurement.own_position[[0, 2]]),
                own_velocity_xz=tuple(measurement.own_velocity[[0, 2]]),
                pitch_rad=float(angles[1]),
                pitch_rate_rad_s=float(measurement.angular_rate[1]),
                vision_updated=updated, vision_age_s=age,
                navigation_valid=True, navigation_age_s=0.0,
                estimate_initialized=self.filter.initialized,
                pad_offset_z_m=self.pad_offset_z_m)
            vector = planar_vector(estimate)
        else:
            estimate = SpatialEstimate(
                pad_position_enu=tuple(self.filter.state[:3]),
                pad_velocity_enu=tuple(self.filter.state[3:]),
                own_position_enu=tuple(measurement.own_position),
                own_velocity_enu=tuple(measurement.own_velocity),
                roll_pitch_yaw_rad=tuple(angles),
                body_rate_rad_s=tuple(measurement.angular_rate),
                vision_updated=updated, vision_age_s=age,
                navigation_valid=True, navigation_age_s=0.0,
                estimate_initialized=self.filter.initialized)
            vector = spatial_vector(estimate)
        return vector, {
            "episode_id": self.episode_id,
            "sample_id": self.last_sample_id,
            "capture_stamp_s": self.last_capture_stamp,
            "receive_stamp_s": self.last_receive_stamp,
            "decision_stamp_s": measurement.time_s,
            "frame_used": updated,
            "estimate_initialized": self.filter.initialized,
        }
