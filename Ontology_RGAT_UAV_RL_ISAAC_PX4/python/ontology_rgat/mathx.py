"""Quaternion and angle helpers.

Ported from the retired MATLAB ``+mathx`` package. The convention is the one
the whole workspace uses: quaternions are ``[w x y z]`` in ENU/FLU, and Euler
angles are intrinsic Z-Y-X (yaw, pitch, roll) returned as ``[roll pitch yaw]``.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "quat_normalize",
    "quat_to_rotm",
    "quat_to_euler_zyx",
    "euler_to_quat",
    "wrap_pi",
    "skew",
]


def quat_normalize(q) -> np.ndarray:
    """Unit quaternion, defaulting to identity for a degenerate input."""
    qv = np.asarray(q, dtype=float).reshape(4)
    n = float(np.linalg.norm(qv))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return qv / n


def quat_to_rotm(q) -> np.ndarray:
    """Body-to-inertial rotation matrix for a ``[w x y z]`` quaternion."""
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_to_euler_zyx(q) -> np.ndarray:
    """``[roll pitch yaw]`` in radians, with the pitch clamped at the poles."""
    w, x, y, z = quat_normalize(q)
    sin_pitch = 2.0 * (w * y - z * x)
    sin_pitch = min(1.0, max(-1.0, sin_pitch))
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(sin_pitch)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


def euler_to_quat(rpy) -> np.ndarray:
    """``[w x y z]`` from intrinsic Z-Y-X ``[roll pitch yaw]``."""
    roll, pitch, yaw = (float(v) for v in np.asarray(rpy, dtype=float).reshape(3))
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return quat_normalize([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def wrap_pi(angle):
    """Wrap an angle (scalar or array) into ``(-pi, pi]``."""
    return np.mod(np.asarray(angle, dtype=float) + np.pi, 2 * np.pi) - np.pi


def skew(v) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
