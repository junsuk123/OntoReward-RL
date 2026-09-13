"""Curriculum helpers shared by the learner and Isaac reset implementation."""
from __future__ import annotations

import math

import numpy as np


def camera_centered_hover_offset(
        altitude_m: float, pitch_down_deg: float = 60.0,
        mount_translation_flu_m=(0.0, 0.0, -0.16)) -> np.ndarray:
    """Return the UAV-in-pad offset that centres the pad in its camera.

    At handover the pad and UAV headings are aligned. Positive body X is
    forward and Z is up, so a forward/downward camera requires the aircraft to
    hover behind the pad instead of directly over it.
    """
    altitude = float(altitude_m)
    pitch = math.radians(float(pitch_down_deg))
    mount = np.asarray(mount_translation_flu_m, dtype=float).reshape(-1)
    if (mount.shape != (3,) or not np.isfinite(mount).all()
            or not math.isfinite(altitude) or altitude <= 0.0
            or not 0.0 < pitch < math.pi / 2.0):
        raise ValueError("camera-centred hover needs finite mount, altitude and 0--90 deg pitch")
    camera_height = altitude + float(mount[2])
    if camera_height <= 0.0:
        raise ValueError("landing camera must remain above the pad")
    forward_reach = camera_height / math.tan(pitch)
    return np.array([-forward_reach - mount[0], -mount[1], altitude], dtype=float)


def yaw_aligned_hover_offset(offset_flu, yaw_enu_rad: float) -> np.ndarray:
    """Rotate a camera-centred body/FLU hover offset into translated ENU.

    The live pad state uses gravity-aligned ENU axes translated to the deck,
    while the forward/down camera turns with vehicle yaw. Leaving the camera
    offset in unrotated ENU centres the target only when yaw happens to be zero.
    """
    offset = np.asarray(offset_flu, dtype=float).reshape(-1)
    yaw = float(yaw_enu_rad)
    if offset.shape != (3,) or not np.isfinite(offset).all() or not math.isfinite(yaw):
        raise ValueError("hover offset and yaw must be finite")
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * offset[0] - s * offset[1],
                     s * offset[0] + c * offset[1], offset[2]], dtype=float)


def constrain_camera_visible_entry(
        raw_offset, yaw_enu_rad: float, *, image_size=(512, 320),
        horizontal_fov_deg: float = 90.0, pitch_down_deg: float = 60.0,
        mount_translation_flu_m=(0.0, 0.0, -0.16),
        footprint_fraction: float = 0.65,
        target_radius_m: float = 0.0) -> np.ndarray:
    """Condition a seeded entry draw on the deck being inside the camera FOV.

    The paper starts every landing episode with the platform visible.  This
    keeps the sampled altitude and yaw, then shortens only the horizontal
    displacement from the camera-axis centre.  The admissible radius is the
    conservative inscribed circle of the pitched camera's ground footprint,
    so the subsequent detector gate verifies a physically achievable pose
    instead of repeatedly timing out on an impossible draw.
    """
    offset = np.asarray(raw_offset, dtype=float).reshape(-1)
    mount = np.asarray(mount_translation_flu_m, dtype=float).reshape(-1)
    yaw = float(yaw_enu_rad)
    width, height = (int(value) for value in image_size)
    hfov = math.radians(float(horizontal_fov_deg))
    pitch = math.radians(float(pitch_down_deg))
    fraction = float(footprint_fraction)
    target_radius = float(target_radius_m)
    if (offset.shape != (3,) or mount.shape != (3,)
            or not np.isfinite(offset).all() or not np.isfinite(mount).all()
            or not math.isfinite(yaw) or width < 2 or height < 2
            or not 0.0 < hfov < math.pi or not 0.0 < pitch < math.pi / 2.0
            or not 0.0 < fraction <= 1.0
            or not math.isfinite(target_radius) or target_radius < 0.0):
        raise ValueError("camera-visible entry configuration is invalid")
    camera_height = float(offset[2] + mount[2])
    if camera_height <= 0.0:
        raise ValueError("camera-visible entry must remain above the deck")
    vfov = 2.0 * math.atan(math.tan(hfov / 2.0) * height / width)
    centre_distance = camera_height / math.tan(pitch)
    near_angle = min(pitch + vfov / 2.0, math.radians(89.0))
    far_angle = max(pitch - vfov / 2.0, math.radians(1.0))
    near_distance = camera_height / math.tan(near_angle)
    far_distance = camera_height / math.tan(far_angle)
    slant = camera_height / math.sin(pitch)
    lateral_half = slant * math.tan(hfov / 2.0)
    # Bounding only the target centre can legally crop every marker on a
    # low-altitude 1.5 m board. Erode the admissible footprint by the board's
    # enclosing radius so the platform itself begins in view.
    footprint_radius = fraction * min(
        lateral_half, abs(centre_distance - near_distance),
        abs(far_distance - centre_distance)) - target_radius
    footprint_radius = max(0.05, float(footprint_radius))

    centre_body = camera_centered_hover_offset(
        float(offset[2]), math.degrees(pitch), mount)
    centre = yaw_aligned_hover_offset(centre_body, yaw)
    delta = offset[:2] - centre[:2]
    reach = float(np.linalg.norm(delta))
    result = offset.copy()
    if reach > footprint_radius:
        result[:2] = centre[:2] + delta * (footprint_radius / reach)
    return result


def curriculum_motion_scale(curriculum: float, minimum_scale: float) -> float:
    """Keep the platform moving slowly while its difficulty ramps to full."""
    c = float(curriculum)
    minimum = float(minimum_scale)
    if not math.isfinite(c) or not math.isfinite(minimum):
        raise ValueError("motion curriculum values must be finite")
    if not 0.0 <= c <= 1.0 or not 0.0 <= minimum <= 1.0:
        raise ValueError("motion curriculum values must be in [0, 1]")
    return minimum + (1.0 - minimum) * c


def curriculum_camera_entry(raw_offset, raw_yaw_deg: float, curriculum: float,
                            *, minimum_scale: float = 0.0,
                            hover_offset_pad_m=None,
                            minimum_altitude_m: float | None = None,
                            camera_pitch_down_deg: float | None = None,
                            camera_mount_translation_flu_m=(0.0, 0.0, -0.16),
                            ) -> tuple[np.ndarray, float]:
    """Blend the stable airborne spawn into the full Table-I draw.

    At ``c=1`` this is an identity operation. At ``c=0`` the entry is the
    simulator's already-stationary hover support, so PX4 does not perform a
    large unmeasured repositioning maneuver before the first policy action.
    ``minimum_scale`` can retain a small seeded variation when explicitly
    requested. Coordinates are in the gravity-aligned pad frame.

    The final two keyword arguments are retained for API compatibility with
    older callers; hover anchoring supersedes the previous camera-axis
    calculation.
    """
    offset = np.asarray(raw_offset, dtype=float).reshape(-1)
    if hover_offset_pad_m is None:
        hover_offset_pad_m = camera_centered_hover_offset(
            4.5,
            60.0 if camera_pitch_down_deg is None else camera_pitch_down_deg,
            camera_mount_translation_flu_m)
    hover = np.asarray(hover_offset_pad_m, dtype=float).reshape(-1)
    yaw_deg = float(raw_yaw_deg)
    c = float(curriculum)
    minimum_scale = float(minimum_scale)
    if (offset.shape != (3,) or hover.shape != (3,)
            or not np.isfinite(offset).all() or not np.isfinite(hover).all()):
        raise ValueError("entry offset must contain three finite values")
    if not all(math.isfinite(value) for value in (yaw_deg, c, minimum_scale)):
        raise ValueError("entry curriculum values must be finite")
    if not 0.0 <= c <= 1.0:
        raise ValueError("entry curriculum must be in [0, 1]")
    if not 0.0 <= minimum_scale <= 1.0:
        raise ValueError("minimum entry scale must be in [0, 1]")
    if hover[2] <= 0.0:
        raise ValueError("hover entry altitude must be positive")
    if c == 1.0:
        return offset.copy(), yaw_deg

    del minimum_altitude_m
    easy = hover + minimum_scale * (offset - hover)
    easy_yaw_deg = minimum_scale * yaw_deg
    blended = (1.0 - c) * easy + c * offset
    blended_yaw = (1.0 - c) * easy_yaw_deg + c * yaw_deg
    return blended, float(blended_yaw)
