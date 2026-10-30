"""Camera-frame kinematics of the pad estimate, from the current state alone (guide 7).

W is ENU, B is FLU, C is the optical frame (+X right, +Y down, +Z forward).
The rates are the instantaneous kinematics of the current estimate, not a
time-series derivative: they do not explain estimator jumps or predict.
Nothing here reads the pad attitude, so it says where the pad centre
projects, not whether the board is detectable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import CameraCalibration
from .observation import CurrentPhysical


MARGIN_NAMES = ("left", "right", "top", "bottom")
NEUTRAL = 0.0


def rotation_world_from_body(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """R_WB = Rz(yaw) Ry(pitch) Rx(roll), the scipy extrinsic ``"xyz"`` convention."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)


@dataclass(frozen=True)
class CameraKinematics:
    """Pad centre in the camera and its image projection.

    Projection values are the registered neutral 0.0 whenever ``front_valid``
    is False; ``p_c``/``p_c_dot`` are also 0.0 when there is no estimate at
    all, because the zeroed estimate would otherwise project a fictitious
    point 0.16 m behind the lens.
    """
    p_c: tuple[float, float, float]
    p_c_dot: tuple[float, float, float]
    u: float
    v: float
    u_dot: float
    v_dot: float
    xi: float
    eta: float
    xi_dot: float
    eta_dot: float
    margins: tuple[float, float, float, float]
    front_valid: bool

    def margin(self, name: str) -> float:
        return self.margins[MARGIN_NAMES.index(name)]


def _neutral(p_c, p_c_dot) -> CameraKinematics:
    return CameraKinematics(p_c=p_c, p_c_dot=p_c_dot, u=NEUTRAL, v=NEUTRAL,
                            u_dot=NEUTRAL, v_dot=NEUTRAL, xi=NEUTRAL, eta=NEUTRAL,
                            xi_dot=NEUTRAL, eta_dot=NEUTRAL, margins=(NEUTRAL,) * 4,
                            front_valid=False)


def camera_kinematics(phys: CurrentPhysical, calibration: CameraCalibration) -> CameraKinematics:
    """p_C = R_CB (b - t_BC) and p_C_dot = R_CB (R_BW v_rel - omega x b), b = R_BW r_W.

    The rotational term uses the body-frame ``b``, not ``p_C``: with the lens
    0.16 m below the body origin, ``omega x p_C`` would drop the lever arm.
    """
    if calibration.has_distortion:
        raise NotImplementedError("only the zero-distortion nominal pinhole is registered")
    if not phys.estimate_valid:
        return _neutral((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    r_bw = rotation_world_from_body(phys.roll, phys.pitch, phys.yaw).T
    r_cb = calibration.r_cb_matrix
    r_w = np.array([phys.ex, phys.ey, -phys.h], dtype=np.float64)
    v_rel_w = np.array([phys.vrx, phys.vry, -phys.vz], dtype=np.float64)
    omega_b = np.array([phys.p, phys.q, phys.r], dtype=np.float64)
    b = r_bw @ r_w
    p_c = r_cb @ (b - calibration.t_bc_vector)
    p_c_dot = r_cb @ (r_bw @ v_rel_w - np.cross(omega_b, b))
    position, rate = tuple(p_c.tolist()), tuple(p_c_dot.tolist())
    x, y, z = position
    x_dot, y_dot, z_dot = rate
    if not z > calibration.z_near_m:
        return _neutral(position, rate)
    u = calibration.fx * x / z + calibration.cx
    v = calibration.fy * y / z + calibration.cy
    u_dot = calibration.fx * (x_dot * z - x * z_dot) / (z * z)
    v_dot = calibration.fy * (y_dot * z - y * z_dot) / (z * z)
    width, height = calibration.width - 1, calibration.height - 1
    xi, eta = 2.0 * u / width - 1.0, 2.0 * v / height - 1.0
    return CameraKinematics(p_c=position, p_c_dot=rate, u=u, v=v, u_dot=u_dot, v_dot=v_dot,
                            xi=xi, eta=eta, xi_dot=2.0 * u_dot / width,
                            eta_dot=2.0 * v_dot / height,
                            margins=(1.0 + xi, 1.0 - xi, 1.0 + eta, 1.0 - eta),
                            front_valid=True)
