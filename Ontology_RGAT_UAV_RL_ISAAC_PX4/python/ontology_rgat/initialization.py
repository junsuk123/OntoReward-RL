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
