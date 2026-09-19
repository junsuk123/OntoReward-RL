"""Curriculum helpers shared by the learner and Isaac reset implementation."""
from __future__ import annotations

import math

import numpy as np

from .mathx import quat_to_rotm


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


# The rendered landing camera's optical frame (OpenCV: +X right, +Y down, +Z
# view) expressed in body FLU axes when it looks straight down. This is the
# same constant as ``isaac_sim.marker_vision.R_BODY_FROM_OPTICAL``; Isaac's
# ``aim_at_nadir`` forces the camera prim onto exactly this frame after the
# configured down-pitch, so image +X spans the vehicle's pitch plane and image
# +Y spans left/right. ``tests/test_shin2026_integrity.py`` asserts equality.
R_BODY_FROM_OPTICAL_NADIR = np.array([
    [1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
])


def body_from_optical(pitch_down_deg: float) -> np.ndarray:
    """Optical-to-body rotation of the forward/down landing camera."""
    pitch_down = float(pitch_down_deg)
    if not math.isfinite(pitch_down) or not 0.0 < pitch_down <= 90.0:
        raise ValueError("camera pitch-down must be finite and in (0, 90] deg")
    angle = math.radians(pitch_down - 90.0)
    c, s = math.cos(angle), math.sin(angle)
    rotate_y = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return rotate_y @ R_BODY_FROM_OPTICAL_NADIR


def pad_view_margin(uav_position_pad_enu, uav_quaternion_wxyz, *,
                    image_size=(512, 320), horizontal_fov_deg: float = 90.0,
                    pitch_down_deg: float = 60.0,
                    mount_translation_flu_m=(0.0, 0.0, -0.16),
                    pad_position_enu=(0.0, 0.0, 0.0), signed: bool = False):
    """How deep inside the camera frustum the pad centre projects.

    With ``signed`` this returns the ``[column, row]`` coordinates themselves
    rather than their magnitude; ``pad_image_position`` is that spelling.

    Returns the pad centre's normalised image coordinate: the larger of
    ``|x| / tan(hfov/2)`` and ``|y| / tan(vfov/2)`` in the optical frame, so a
    value below 1 lies inside the frame, 0 is the optical axis and ``inf`` is
    behind the camera. The pose is the gravity-aligned pad-relative ENU
    position the entry climb is flown on together with the ENU/FLU attitude.
    """
    position = np.asarray(uav_position_pad_enu, dtype=float).reshape(-1)
    pad = np.asarray(pad_position_enu, dtype=float).reshape(-1)
    mount = np.asarray(mount_translation_flu_m, dtype=float).reshape(-1)
    quaternion = np.asarray(uav_quaternion_wxyz, dtype=float).reshape(-1)
    width, height = (int(value) for value in image_size)
    hfov = math.radians(float(horizontal_fov_deg))
    if (position.shape != (3,) or pad.shape != (3,) or mount.shape != (3,)
            or quaternion.shape != (4,) or not np.isfinite(position).all()
            or not np.isfinite(pad).all() or not np.isfinite(mount).all()
            or not np.isfinite(quaternion).all() or width < 2 or height < 2
            or not 0.0 < hfov < math.pi):
        raise ValueError("pad view geometry configuration is invalid")
    rotation = quat_to_rotm(quaternion)
    camera_position = position + rotation @ mount
    optical_in_enu = rotation @ body_from_optical(pitch_down_deg)
    ray = optical_in_enu.T @ (pad - camera_position)
    depth = float(ray[2])
    if depth <= 1e-9:
        return None if signed else math.inf
    tan_half_h = math.tan(hfov / 2.0)
    tan_half_v = tan_half_h * height / width
    column = float(ray[0]) / depth / tan_half_h
    row = float(ray[1]) / depth / tan_half_v
    if signed:
        return np.array([column, row], dtype=float)
    return max(abs(column), abs(row))


def pad_image_position(uav_position_pad_enu, uav_quaternion_wxyz, **camera):
    """Where the pad centre projects, signed, in the encoder's convention.

    ``pad_view_margin`` answers "is it in frame" and therefore throws the sign
    away. A servo needs the direction: this returns ``[column, row]`` with the
    same normalization the keypoint encoder is trained against
    (``2 * pixel / (size - 1) - 1``, so +-1 is the frame edge), or ``None``
    when the pad is behind the camera.

    This camera's mounting rotates the frame: ``R_BODY_FROM_OPTICAL_NADIR``
    sends optical x to body forward and optical y to body right, so the image
    *column* moves with the pad's fore/aft position and the image *row* moves
    with its lateral position. Both signs are measured here rather than
    assumed, which is what makes them safe to build a controller on.
    """
    return pad_view_margin(uav_position_pad_enu, uav_quaternion_wxyz,
                           signed=True, **camera)


def nadir_image_setpoint(horizontal_fov_deg: float = 90.0,
                         pitch_down_deg: float = 60.0) -> np.ndarray:
    """The image point a vehicle directly above the pad sees the pad at.

    The landing camera looks ``pitch_down_deg`` below horizontal, so the image
    centre is *not* the landing point: holding the pad there parks the vehicle
    ``camera_height / tan(pitch)`` behind the deck, which is exactly what
    ``camera_centered_hover_offset`` computes for the entry pose. A servo that
    drives the pad to the image centre therefore never arrives.

    The point to drive it to is the nadir direction, ``90 - pitch_down``
    degrees off the optical axis along the column axis. Two properties make it
    usable as a fixed setpoint: it is a direction, so it does not move with
    altitude -- the same normalized column holds at 8 m and at 0.5 m -- and
    for this camera it is well inside the frame, at
    ``tan(30 deg) / tan(45 deg) = 0.577`` of the half width.
    """
    hfov = math.radians(float(horizontal_fov_deg))
    pitch = float(pitch_down_deg)
    if (not math.isfinite(hfov) or not 0.0 < hfov < math.pi
            or not math.isfinite(pitch) or not 0.0 < pitch <= 90.0):
        raise ValueError("nadir setpoint needs a valid camera geometry")
    return np.array([
        -math.tan(math.radians(90.0 - pitch)) / math.tan(hfov / 2.0), 0.0])


def pad_in_camera_view(uav_position_pad_enu, uav_quaternion_wxyz, *,
                       margin_fraction: float = 0.85, **camera) -> bool:
    """Is the pad centre inside the frame, clear of its edges?

    ``margin_fraction`` is the admissible fraction of the half field of view,
    so 0.85 keeps the centre out of the outer 15 % of the image on each side.
    """
    fraction = float(margin_fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("view margin fraction must be in (0, 1]")
    return pad_view_margin(
        uav_position_pad_enu, uav_quaternion_wxyz, **camera) <= fraction


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
