"""Causal observation assembly for the direct-policy port.

Only timestamped marker corners and fused own navigation enter the estimator.
The public vector functions are intentionally small and deterministic so their
MATLAB golden fixtures can be compared without Isaac or ROS installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import numpy as np

from .contracts import PLANAR_FIELDS, SPATIAL_FIELDS


PLANAR_NAMES = tuple(item.name for item in PLANAR_FIELDS)
SPATIAL_NAMES = tuple(item.name for item in SPATIAL_FIELDS)


def signed_scale(value, scale):
    value = np.asarray(value, dtype=np.float64)
    return value / (np.abs(value) + max(float(scale), np.finfo(float).eps))


def age_scale(value, scale=3.0):
    value = float(value)
    if not math.isfinite(value):
        return 1.0
    value = max(value, 0.0)
    return value / (value + max(float(scale), np.finfo(float).eps))


@dataclass(frozen=True)
class PlanarEstimate:
    pad_position_xz: tuple[float, float]
    pad_velocity_xz: tuple[float, float]
    own_position_xz: tuple[float, float]
    own_velocity_xz: tuple[float, float]
    pitch_rad: float
    pitch_rate_rad_s: float
    vision_updated: bool
    vision_age_s: float
    navigation_valid: bool
    navigation_age_s: float
    estimate_initialized: bool = True
    pad_offset_z_m: float = 0.6


@dataclass(frozen=True)
class SpatialEstimate:
    pad_position_enu: tuple[float, float, float]
    pad_velocity_enu: tuple[float, float, float]
    own_position_enu: tuple[float, float, float]
    own_velocity_enu: tuple[float, float, float]
    roll_pitch_yaw_rad: tuple[float, float, float]
    body_rate_rad_s: tuple[float, float, float]
    vision_updated: bool
    vision_age_s: float
    navigation_valid: bool
    navigation_age_s: float
    estimate_initialized: bool = True


def planar_vector(state: PlanarEstimate) -> np.ndarray:
    """The pinned MATLAB ``toVector`` operation in its registered field order."""
    pad_p = np.asarray(state.pad_position_xz, dtype=np.float64)
    pad_v = np.asarray(state.pad_velocity_xz, dtype=np.float64)
    own_p = np.asarray(state.own_position_xz, dtype=np.float64)
    own_v = np.asarray(state.own_velocity_xz, dtype=np.float64)
    relative_x = pad_p[0] - own_p[0]
    relative_height = own_p[1] - (pad_p[1] + state.pad_offset_z_m)
    relative_vx = pad_v[0] - own_v[0]
    out = np.array([
        signed_scale(relative_x, 3.0),
        signed_scale(relative_height, 8.0),
        signed_scale(relative_vx, 10.0),
        signed_scale(pad_v[0], 10.0),
        signed_scale(own_v[1], 1.5),
        math.sin(state.pitch_rad), math.cos(state.pitch_rad),
        signed_scale(state.pitch_rate_rad_s, math.radians(90.0)),
        float(state.vision_updated), age_scale(state.vision_age_s),
        float(state.navigation_valid), age_scale(state.navigation_age_s),
    ], dtype=np.float64)
    if not state.estimate_initialized:
        out[:4] = 0.0
    if out.shape != (12,) or not np.all(np.isfinite(out)):
        raise ValueError("malformed planar observation")
    return out


def spatial_vector(state: SpatialEstimate) -> np.ndarray:
    """Fixed 21D extension; values remain estimates, never simulator truth."""
    pad_p = np.asarray(state.pad_position_enu, dtype=np.float64)
    pad_v = np.asarray(state.pad_velocity_enu, dtype=np.float64)
    own_p = np.asarray(state.own_position_enu, dtype=np.float64)
    own_v = np.asarray(state.own_velocity_enu, dtype=np.float64)
    rel_p, rel_v = pad_p - own_p, pad_v - own_v
    roll, pitch, yaw = state.roll_pitch_yaw_rad
    p, q, r = state.body_rate_rad_s
    out = np.array([
        signed_scale(rel_p[0], 3.0), signed_scale(rel_p[1], 3.0),
        signed_scale(-rel_p[2], 8.0),
        signed_scale(rel_v[0], 10.0), signed_scale(rel_v[1], 10.0),
        signed_scale(pad_v[0], 10.0), signed_scale(pad_v[1], 10.0),
        signed_scale(own_v[2], 1.5),
        math.sin(roll), math.cos(roll), math.sin(pitch), math.cos(pitch),
        math.sin(yaw), math.cos(yaw),
        signed_scale(p, math.radians(90.0)),
        signed_scale(q, math.radians(90.0)),
        signed_scale(r, math.radians(90.0)),
        float(state.vision_updated), age_scale(state.vision_age_s),
        float(state.navigation_valid), age_scale(state.navigation_age_s),
    ], dtype=np.float64)
    if not state.estimate_initialized:
        out[:7] = 0.0
    if out.shape != (21,) or not np.all(np.isfinite(out)):
        raise ValueError("malformed spatial observation")
    return out


def spatial_to_planar_named(vector: Sequence[float]) -> np.ndarray:
    """Project the y/roll/yaw extension away without positional slicing lore."""
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape != (21,):
        raise ValueError("spatial observation must contain 21 values")
    by_name = dict(zip(SPATIAL_NAMES, vector))
    mapping = {
        "relative_x": "relative_x", "relative_height": "relative_height",
        "relative_vx": "relative_vx", "ugv_vx": "ugv_vx",
        "drone_vz": "drone_vz", "drone_sinTheta": "drone_sinTheta",
        "drone_cosTheta": "drone_cosTheta", "drone_pitchRate": "drone_pitchRate",
        "ugv_visionUpdated": "ugv_visionUpdated", "ugv_visionAge": "ugv_visionAge",
        "drone_navigationValid": "drone_navigationValid",
        "drone_navigationAge": "drone_navigationAge",
    }
    return np.asarray([by_name[mapping[name]] for name in PLANAR_NAMES], dtype=np.float64)


@dataclass(frozen=True)
class MarkerFrame:
    capture_stamp: float
    receive_stamp: float
    image_size: tuple[int, int]
    corners_by_id: Mapping[int, np.ndarray]
    frame_valid: bool = True


@dataclass(frozen=True)
class Navigation2D:
    stamp: float
    position_xz: tuple[float, float]
    velocity_xz: tuple[float, float]
    pitch_rad: float
    pitch_rate_rad_s: float
    valid: bool = True


@dataclass(frozen=True)
class PlanarCalibration:
    camera_matrix: np.ndarray
    distortion: np.ndarray
    body_to_camera: np.ndarray
    marker_centers_xy: Mapping[int, tuple[float, float]]
    marker_sides_m: Mapping[int, float]
    pad_offset_xz: tuple[float, float] = (0.0, 0.6)
    process_acceleration_std: tuple[float, float] = (0.5, 0.05)
    initial_velocity_std: float = 0.5
    corner_noise_std_px: float = 1.0
    maximum_reprojection_error_px: float = 2.0

    @classmethod
    def source_default(cls) -> "PlanarCalibration":
        width, height = 512, 320
        focal = (width - 1) / 2.0
        alpha = math.radians(60.0)
        c, s = math.cos(alpha), math.sin(alpha)
        body_to_camera = np.array([
            [0, -s, c, 0], [-1, 0, 0, 0], [0, -c, -s, 0], [0, 0, 0, 1]
        ], dtype=np.float64)
        return cls(
            camera_matrix=np.array([[focal, 0, (width - 1) / 2],
                                    [0, focal, (height - 1) / 2], [0, 0, 1]],
                                   dtype=np.float64),
            distortion=np.zeros(5, dtype=np.float64),
            body_to_camera=body_to_camera,
            marker_centers_xy={10: (0, 0), 11: (0.375, 0.375),
                               12: (0.375, -0.375), 13: (-0.375, -0.375),
                               14: (-0.375, 0.375)},
            marker_sides_m={10: 0.50, 11: 0.15, 12: 0.15, 13: 0.15, 14: 0.15})


class ConstantVelocityKalman2D:
    """The pinned source's [x,z,vx,vz] CV estimator with Joseph update."""
    def __init__(self, calibration: PlanarCalibration):
        self.calibration = calibration
        self.reset()

    def reset(self) -> None:
        self.initialized = False
        self.state = np.zeros(4, dtype=np.float64)
        self.covariance = np.zeros((4, 4), dtype=np.float64)
        self.time = 0.0
        self.last_update_time = -math.inf
        self.updated = False

    def update(self, stamp: float, measurement: np.ndarray | None,
               covariance: np.ndarray | None, navigation_velocity: Sequence[float]) -> None:
        if self.initialized:
            dt = float(stamp) - self.time
            if dt < -1e-12:
                raise ValueError("UGV estimator time moved backwards")
            if dt > 0:
                eye = np.eye(2)
                transition = np.block([[eye, dt * eye], [np.zeros((2, 2)), eye]])
                acceleration = np.diag(np.square(self.calibration.process_acceleration_std))
                process = np.block([[dt ** 3 / 3 * acceleration, dt ** 2 / 2 * acceleration],
                                    [dt ** 2 / 2 * acceleration, dt * acceleration]])
                self.state = transition @ self.state
                self.covariance = transition @ self.covariance @ transition.T + process
            self.time = float(stamp)
            self.updated = measurement is not None
            if measurement is not None:
                select = np.c_[np.eye(2), np.zeros((2, 2))]
                innovation = select @ self.covariance @ select.T + covariance
                gain = self.covariance @ select.T @ np.linalg.inv(innovation)
                self.state += gain @ (measurement - select @ self.state)
                reduction = np.eye(4) - gain @ select
                self.covariance = (reduction @ self.covariance @ reduction.T
                                   + gain @ covariance @ gain.T)
        elif measurement is not None:
            self.state = np.r_[measurement, np.asarray(navigation_velocity, dtype=np.float64)]
            self.covariance = np.block([
                [covariance, np.zeros((2, 2))],
                [np.zeros((2, 2)), self.calibration.initial_velocity_std ** 2 * np.eye(2)],
            ])
            self.initialized, self.updated, self.time = True, True, float(stamp)
        else:
            self.updated = False
        if self.updated:
            self.last_update_time = float(stamp)


class PlanarObservationAssembler:
    """Frame-aware PnP -> capture-time transform -> CV-KF -> 12D vector."""
    def __init__(self, calibration: PlanarCalibration | None = None):
        self.calibration = calibration or PlanarCalibration.source_default()
        self.filter = ConstantVelocityKalman2D(self.calibration)
        self.last_frame_stamp = -math.inf
        self.episode_id = None

    def reset(self, episode_id=None) -> None:
        self.filter.reset()
        self.last_frame_stamp = -math.inf
        self.episode_id = episode_id

    def _object_and_image_points(self, frame: MarkerFrame):
        objects, pixels = [], []
        for marker_id, corners in frame.corners_by_id.items():
            if marker_id not in self.calibration.marker_centers_xy:
                continue
            corners = np.asarray(corners, dtype=np.float64)
            if corners.shape != (4, 2):
                raise ValueError("each marker must have four image corners")
            cx, cy = self.calibration.marker_centers_xy[marker_id]
            half = self.calibration.marker_sides_m[marker_id] / 2.0
            # OpenCV ArUco order: TL, TR, BR, BL in the pad x/y plane.
            objects.extend(((cx-half, cy+half, 0), (cx+half, cy+half, 0),
                            (cx+half, cy-half, 0), (cx-half, cy-half, 0)))
            pixels.extend(corners)
        return np.asarray(objects, np.float64), np.asarray(pixels, np.float64)

    def _pad_measurement(self, frame: MarkerFrame, nav: Navigation2D):
        if not frame.frame_valid or not frame.corners_by_id:
            return None, None
        object_points, image_points = self._object_and_image_points(frame)
        if len(object_points) < 4:
            return None, None
        try:
            import cv2
        except ImportError:
            return None, None
        try:
            ok, rvec, tvec = cv2.solvePnP(
                object_points, image_points, self.calibration.camera_matrix,
                self.calibration.distortion, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                return None, None
            projected, _ = cv2.projectPoints(object_points, rvec, tvec,
                                             self.calibration.camera_matrix,
                                             self.calibration.distortion)
        except cv2.error:
            return None, None
        rms = float(np.sqrt(np.mean(np.sum(
            (projected.reshape(-1, 2) - image_points) ** 2, axis=1))))
        translation_camera = tvec.reshape(3)
        if rms > self.calibration.maximum_reprojection_error_px or translation_camera[2] <= 0:
            return None, None
        pitch = nav.pitch_rad
        body_to_world = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0],
                                  [-math.sin(pitch), 0, math.cos(pitch)]])
        mount = self.calibration.body_to_camera
        pad_body = mount[:3, 3] + mount[:3, :3] @ translation_camera
        pad_world = np.array([nav.position_xz[0], 0, nav.position_xz[1]]) \
            + body_to_world @ pad_body
        measurement = pad_world[[0, 2]] - np.asarray(self.calibration.pad_offset_xz)
        # First-order source covariance is geometry-dependent.  This compact
        # port uses the same configured corner sigma and keeps it explicit.
        depth = max(float(translation_camera[2]), 1e-6)
        focal = float(self.calibration.camera_matrix[0, 0])
        sigma = self.calibration.corner_noise_std_px * depth / focal
        return measurement, np.eye(2) * sigma * sigma

    def assemble(self, frame: MarkerFrame | None, capture_navigation: Navigation2D,
                 decision_navigation: Navigation2D, decision_stamp: float,
                 *, episode_id=None) -> tuple[np.ndarray, dict]:
        if episode_id is not None and self.episode_id not in (None, episode_id):
            raise ValueError("buffered message crossed an episode reset")
        if decision_stamp < decision_navigation.stamp - 1e-12:
            raise ValueError("navigation is newer than the decision")
        measurement = covariance = None
        used_frame = False
        if frame is not None:
            if frame.capture_stamp > frame.receive_stamp + 1e-12 \
                    or frame.receive_stamp > decision_stamp + 1e-12:
                raise ValueError("non-causal frame timestamps")
            if frame.capture_stamp < self.last_frame_stamp - 1e-12:
                raise ValueError("out-of-order marker frame")
            if frame.capture_stamp > self.last_frame_stamp + 1e-12:
                if abs(capture_navigation.stamp - frame.capture_stamp) > 1e-6:
                    raise ValueError("PnP transform needs capture-time navigation")
                measurement, covariance = self._pad_measurement(frame, capture_navigation)
                self.last_frame_stamp = frame.capture_stamp
                used_frame = measurement is not None
        self.filter.update(decision_stamp, measurement, covariance,
                           decision_navigation.velocity_xz)
        vision_age = (decision_stamp - self.filter.last_update_time
                      if self.filter.initialized else math.inf)
        state = PlanarEstimate(
            pad_position_xz=tuple(self.filter.state[:2]),
            pad_velocity_xz=tuple(self.filter.state[2:]),
            own_position_xz=decision_navigation.position_xz,
            own_velocity_xz=decision_navigation.velocity_xz,
            pitch_rad=decision_navigation.pitch_rad,
            pitch_rate_rad_s=decision_navigation.pitch_rate_rad_s,
            vision_updated=used_frame,
            vision_age_s=vision_age,
            navigation_valid=decision_navigation.valid,
            navigation_age_s=(max(decision_stamp - decision_navigation.stamp, 0.0)
                              if decision_navigation.valid else math.inf),
            estimate_initialized=self.filter.initialized,
            pad_offset_z_m=self.calibration.pad_offset_xz[1])
        provenance = {"capture_stamp": None if frame is None else frame.capture_stamp,
                      "decision_stamp": decision_stamp, "frame_used": used_frame,
                      "estimate_initialized": self.filter.initialized,
                      "episode_id": episode_id}
        return planar_vector(state), provenance
