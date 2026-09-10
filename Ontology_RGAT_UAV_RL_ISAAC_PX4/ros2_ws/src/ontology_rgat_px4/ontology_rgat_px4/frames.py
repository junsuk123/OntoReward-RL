from __future__ import annotations

import math
from typing import Iterable

import numpy as np


R_ENU_FROM_NED = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
R_FLU_FROM_FRD = np.diag([1.0, -1.0, -1.0])


def _vec3(value: Iterable[float]) -> np.ndarray:
    out = np.asarray(tuple(value), dtype=float)
    if out.shape != (3,) or not np.isfinite(out).all():
        raise ValueError("expected three finite values")
    return out


def ned_to_enu(value: Iterable[float]) -> np.ndarray:
    return R_ENU_FROM_NED @ _vec3(value)


def enu_to_ned(value: Iterable[float]) -> np.ndarray:
    return R_ENU_FROM_NED.T @ _vec3(value)


def frd_to_flu(value: Iterable[float]) -> np.ndarray:
    return R_FLU_FROM_FRD @ _vec3(value)


def flu_to_frd(value: Iterable[float]) -> np.ndarray:
    return R_FLU_FROM_FRD.T @ _vec3(value)


def quat_wxyz_to_matrix(q: Iterable[float]) -> np.ndarray:
    qv = np.asarray(tuple(q), dtype=float)
    if qv.shape != (4,) or not np.isfinite(qv).all():
        raise ValueError("expected four finite quaternion values")
    norm = np.linalg.norm(qv)
    if norm < 1e-12:
        raise ValueError("zero quaternion")
    w, x, y, z = qv / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def matrix_to_quat_wxyz(r: np.ndarray) -> np.ndarray:
    m = np.asarray(r, dtype=float)
    if m.shape != (3, 3) or not np.isfinite(m).all():
        raise ValueError("expected a finite 3x3 rotation matrix")
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s,
                      (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                          (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
        elif i == 1:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                          0.25 * s, (m[1, 2] + m[2, 1]) / s])
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                          (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    q /= np.linalg.norm(q)
    return q if q[0] >= 0.0 else -q


def quat_ned_frd_to_enu_flu(q_wxyz: Iterable[float]) -> np.ndarray:
    r_ned_from_frd = quat_wxyz_to_matrix(q_wxyz)
    r_enu_from_flu = R_ENU_FROM_NED @ r_ned_from_frd @ R_FLU_FROM_FRD.T
    return matrix_to_quat_wxyz(r_enu_from_flu)


def quat_enu_flu_to_ned_frd(q_wxyz: Iterable[float]) -> np.ndarray:
    r_enu_from_flu = quat_wxyz_to_matrix(q_wxyz)
    r_ned_from_frd = R_ENU_FROM_NED.T @ r_enu_from_flu @ R_FLU_FROM_FRD
    return matrix_to_quat_wxyz(r_ned_from_frd)


def euler_zyx_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def yaw_from_quat_wxyz(q: Iterable[float]) -> float:
    w, x, y, z = np.asarray(tuple(q), dtype=float)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def yaw_enu_to_ned(yaw_enu: float) -> float:
    """Heading of the same attitude expressed in PX4's NED/FRD convention.

    ENU/FLU measures yaw from east counter-clockwise, NED/FRD from north
    clockwise, so the two differ by a reflection about the 45 degree axis.
    """
    return math.atan2(math.cos(yaw_enu), math.sin(yaw_enu))


# --------------------------------------------------------------- geodesy
# WGS84 semi-major axis. The projection below is a local tangent plane, so the
# flattening term would be noise against the metre-scale errors this
# environment models. Deliberately the same formula as
# isaac_sim/osm_city.project_to_local -- the simulator lays the city out with
# it and the gateway reads PX4's origin back with it, so the two have to agree
# to the metre or the deck lands in the wrong street. tests/test_frames.py
# pins them against each other.
WGS84_A = 6378137.0


def geodetic_to_enu(latitude_deg: float, longitude_deg: float,
                    lat0_deg: float, lon0_deg: float) -> np.ndarray:
    """Where a lat/lon sits in the ENU frame pinned at (LAT0, LON0), in metres."""
    east = math.radians(longitude_deg - lon0_deg) * WGS84_A * math.cos(
        math.radians(lat0_deg))
    north = math.radians(latitude_deg - lat0_deg) * WGS84_A
    return np.array([east, north, 0.0], dtype=float)
