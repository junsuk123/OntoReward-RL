#!/usr/bin/env python3
"""Simulator-side geometry of the six-keypoint landing pad.

Isaac is **not** imported here, so every quantity in this module is testable
without a simulator and is shared verbatim by the learner, the pretraining
label generator and the stage builder.

Provenance contract
-------------------
Everything this module returns is *simulator truth*.  It is admissible only as

* ``training-label-only``  -- supervised six-keypoint targets,
* ``initialization-only``  -- the episode entry-visibility gate,
* ``evaluation-only``      -- geometric FOV metrics and R-GAT labels.

It must never be concatenated into the PPO actor observation, and it must
never enter the online R-GAT graph (see ``rgat/fov_graph.py``, whose only
constructor argument is a visual-only :class:`FOVSemanticObservation`).

Frames
------
``pad``      ENU at the pad centre: +X east, +Y north, +Z up.
``body``     FLU on the vehicle: +X forward, +Y left, +Z up.
``optical``  OpenCV/ROS camera: +X right, +Y down, +Z along the view direction.

Landmark layout
---------------
Six landmarks on the vertices of a pad-centred regular hexagon of radius
``PAD_LANDMARK_RADIUS_M``, first vertex at 30 degrees so that no landmark
lies on the pad's own +X/+Y axes and the layout is unambiguous under the
90-degree deck symmetry.  This is a documented **PACMAN-compatible
approximation**: the PACMAN asset, its landmark table and its weights are not
public, so nothing here should be described as reproducing PACMAN.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


PAD_LANDMARK_COUNT = 6
PAD_LANDMARK_RADIUS_M = 0.52
PAD_LANDMARK_FIRST_ANGLE_RAD = math.pi / 6.0
PAD_CENTER_PAD_M = np.zeros(3)
KEYPOINT_LAYOUT_ID = "hexagonal-six-landmark-r0.52m/1"

DEFAULT_CAMERA = {
    "resolution": (512, 320),
    "horizontal_fov_deg": 90.0,
    "pitch_down_deg": 60.0,
    "mount_translation_flu_m": (0.0, 0.0, -0.16),
}


def pad_landmarks(radius_m: float = PAD_LANDMARK_RADIUS_M) -> np.ndarray:
    """The six landmark positions in the pad frame, shape ``(6, 3)``."""
    radius = float(radius_m)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("landmark radius must be positive and finite")
    angles = (np.arange(PAD_LANDMARK_COUNT, dtype=float) * (math.pi / 3.0)
              + PAD_LANDMARK_FIRST_ANGLE_RAD)
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles),
                            np.zeros(PAD_LANDMARK_COUNT)))


def pad_center() -> np.ndarray:
    """The landing-pad centre in the pad frame."""
    return PAD_CENTER_PAD_M.copy()


def quat_wxyz_to_matrix(quaternion) -> np.ndarray:
    q = np.asarray(quaternion, dtype=float).reshape(-1)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("quaternion must contain four finite values")
    norm = float(np.linalg.norm(q))
    if norm < 1e-9:
        raise ValueError("quaternion must have non-zero norm")
    w, x, y, z = q / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def body_from_optical(pitch_down_deg: float) -> np.ndarray:
    """Optical axes expressed in body FLU for a forward/down pitched camera.

    ``pitch_down_deg = 90`` is nadir; smaller values tilt the view forward.
    """
    angle = math.radians(float(pitch_down_deg) - 90.0)
    c, s = math.cos(angle), math.sin(angle)
    pitch = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    nadir = np.diag([1.0, -1.0, -1.0])
    return pitch @ nadir


def focal_length_px(width: int, horizontal_fov_deg: float) -> float:
    fov = math.radians(float(horizontal_fov_deg))
    if not 0.0 < fov < math.pi:
        raise ValueError("horizontal_fov_deg must be in (0, 180)")
    return (int(width) / 2.0) / math.tan(fov / 2.0)


@dataclass(frozen=True)
class CameraModel:
    """The rendered landing camera, as the simulator configures it."""

    width: int = 512
    height: int = 320
    horizontal_fov_deg: float = 90.0
    pitch_down_deg: float = 60.0
    mount_translation_flu_m: tuple[float, float, float] = (0.0, 0.0, -0.16)

    @classmethod
    def from_mapping(cls, mapping=None) -> "CameraModel":
        values = dict(DEFAULT_CAMERA)
        values.update({k: v for k, v in dict(mapping or {}).items()
                       if k in DEFAULT_CAMERA})
        width, height = (int(v) for v in values["resolution"])
        return cls(width=width, height=height,
                   horizontal_fov_deg=float(values["horizontal_fov_deg"]),
                   pitch_down_deg=float(values["pitch_down_deg"]),
                   mount_translation_flu_m=tuple(
                       float(v) for v in values["mount_translation_flu_m"]))

    def __post_init__(self) -> None:
        if self.width < 2 or self.height < 2:
            raise ValueError("camera resolution must be at least 2x2")
        mount = np.asarray(self.mount_translation_flu_m, dtype=float)
        if mount.shape != (3,) or not np.isfinite(mount).all():
            raise ValueError("mount translation must be a finite FLU triple")
        object.__setattr__(self, "mount_translation_flu_m",
                           tuple(float(v) for v in mount))
        # Validates the field of view.
        focal_length_px(self.width, self.horizontal_fov_deg)

    @property
    def focal_px(self) -> float:
        return focal_length_px(self.width, self.horizontal_fov_deg)

    @property
    def tan_half_horizontal(self) -> float:
        return math.tan(math.radians(self.horizontal_fov_deg) / 2.0)

    @property
    def tan_half_vertical(self) -> float:
        # Square pixels: the vertical half-angle follows from the same focal
        # length, not from the aspect ratio of the horizontal FOV.
        return self.tan_half_horizontal * self.height / self.width


@dataclass(frozen=True)
class PadKeypointProjection:
    """Exact simulator projection of the pad landmarks and the pad centre."""

    keypoint_pixels: np.ndarray          # (6, 2) pixel coordinates
    keypoint_normalized: np.ndarray      # (6, 2) encoder convention, 2p/(n-1)-1
    keypoint_depth_m: np.ndarray         # (6,) optical +Z
    keypoint_visible: np.ndarray         # (6,) bool, inside the rendered frame
    pad_center_pixels: np.ndarray        # (2,)
    pad_center_normalized: np.ndarray    # (2,) frustum convention, +-1 = edge
    pad_center_depth_m: float
    geometric_pad_center_in_fov: bool

    @property
    def visible_keypoint_fraction(self) -> float:
        return float(np.count_nonzero(self.keypoint_visible)
                     / PAD_LANDMARK_COUNT)


def camera_pose_in_pad(uav_position_pad_enu, uav_quaternion_wxyz,
                       camera: CameraModel):
    """Camera origin in the pad frame and pad-from-optical rotation."""
    position = np.asarray(uav_position_pad_enu, dtype=float).reshape(-1)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("uav position must be a finite pad-frame ENU triple")
    rotation = quat_wxyz_to_matrix(uav_quaternion_wxyz)
    mount = np.asarray(camera.mount_translation_flu_m, dtype=float)
    origin = position + rotation @ mount
    pad_from_optical = rotation @ body_from_optical(camera.pitch_down_deg)
    return origin, pad_from_optical


def project_pad_points(points_pad, uav_position_pad_enu, uav_quaternion_wxyz,
                       camera: CameraModel):
    """Project pad-frame points to pixels; returns ``(pixels, depth)``."""
    points = np.asarray(points_pad, dtype=float).reshape(-1, 3)
    origin, pad_from_optical = camera_pose_in_pad(
        uav_position_pad_enu, uav_quaternion_wxyz, camera)
    optical = (pad_from_optical.T @ (points - origin).T).T
    depth = optical[:, 2]
    safe = np.where(depth > 1e-9, depth, 1.0)
    focal = camera.focal_px
    pixels = np.column_stack((
        focal * optical[:, 0] / safe + camera.width / 2.0,
        focal * optical[:, 1] / safe + camera.height / 2.0,
    ))
    return pixels, depth, optical


def project_landing_pad(uav_position_pad_enu, uav_quaternion_wxyz, *,
                        camera=None,
                        landmark_radius_m: float = PAD_LANDMARK_RADIUS_M,
                        pad_position_pad_enu=PAD_CENTER_PAD_M,
                        ) -> PadKeypointProjection:
    """Training/evaluation-only projection of the pad landmarks and centre.

    ``uav_position_pad_enu`` is the vehicle body origin expressed in the
    gravity-aligned pad frame, i.e. exactly the simulator truth the gateway
    publishes under ``state['truth']['position']``.
    """
    model = camera if isinstance(camera, CameraModel) else CameraModel.from_mapping(camera)
    pad = np.asarray(pad_position_pad_enu, dtype=float).reshape(3)
    points = np.vstack((pad_landmarks(landmark_radius_m) + pad, pad))
    pixels, depth, optical = project_pad_points(
        points, uav_position_pad_enu, uav_quaternion_wxyz, model)

    keypoint_pixels = pixels[:PAD_LANDMARK_COUNT]
    keypoint_depth = depth[:PAD_LANDMARK_COUNT]
    keypoint_visible = (
        (keypoint_depth > 0.0)
        & np.isfinite(keypoint_pixels).all(axis=1)
        & (keypoint_pixels[:, 0] >= 0.0)
        & (keypoint_pixels[:, 0] <= model.width - 1.0)
        & (keypoint_pixels[:, 1] >= 0.0)
        & (keypoint_pixels[:, 1] <= model.height - 1.0))
    keypoint_normalized = np.column_stack((
        2.0 * keypoint_pixels[:, 0] / (model.width - 1.0) - 1.0,
        2.0 * keypoint_pixels[:, 1] / (model.height - 1.0) - 1.0))

    center_depth = float(depth[-1])
    if center_depth > 1e-9:
        center_normalized = np.array([
            float(optical[-1, 0]) / center_depth / model.tan_half_horizontal,
            float(optical[-1, 1]) / center_depth / model.tan_half_vertical])
        in_fov = bool(np.all(np.abs(center_normalized) <= 1.0))
    else:
        # Behind the camera: no image coordinate exists.
        center_normalized = np.array([math.inf, math.inf])
        in_fov = False

    return PadKeypointProjection(
        keypoint_pixels=keypoint_pixels,
        keypoint_normalized=keypoint_normalized,
        keypoint_depth_m=keypoint_depth,
        keypoint_visible=keypoint_visible,
        pad_center_pixels=pixels[-1],
        pad_center_normalized=center_normalized,
        pad_center_depth_m=center_depth,
        geometric_pad_center_in_fov=in_fov)


def geometric_pad_center_in_fov(uav_position_pad_enu, uav_quaternion_wxyz, *,
                                camera=None,
                                pad_position_pad_enu=PAD_CENTER_PAD_M) -> bool:
    """The one geometric definition of landing-pad field-of-view retention.

    True only when the pad centre has positive camera depth **and** both of
    its normalized image coordinates lie inside ``[-1, 1]``.  Marker decoding
    success and learned keypoint confidence are deliberately not consulted:
    a blurred, undetectable pad that still projects inside the frame has not
    left the field of view.
    """
    return project_landing_pad(
        uav_position_pad_enu, uav_quaternion_wxyz, camera=camera,
        pad_position_pad_enu=pad_position_pad_enu).geometric_pad_center_in_fov


def nadir_footprint_m(camera: CameraModel, altitude_m: float) -> tuple[float, float]:
    """Half-extents of what the camera sees on the ground, in metres."""
    altitude = max(float(altitude_m), 0.0)
    return (altitude * camera.tan_half_horizontal,
            altitude * camera.tan_half_vertical)
