"""Causal relative-ray frame conversion, using PnP and own measured attitude."""
import numpy as np

from .frames import quat_wxyz_to_matrix


def imu_rotated_relative_position(position_in_board, body_in_board_wxyz, own_enu_wxyz):
    """Resolve the measured body-to-board vector in ENU without deck truth.

    Planar PnP's small roll/pitch ambiguity can couple a false body tilt into
    large lateral position error. Its relative vector in the optical/body
    frame is much better constrained by the image. Undo PnP's board-frame
    rotation, then use the independent own IMU/EKF attitude for ENU.

    The input position is the BODY origin, already corrected for the camera
    mount. Consequently the mount is neither omitted nor subtracted twice.
    This does not replace independent PnP/IMU consistency or innovation gates.
    """
    p = np.asarray(position_in_board, dtype=float)
    q = np.asarray(body_in_board_wxyz, dtype=float)
    own = np.asarray(own_enu_wxyz, dtype=float)
    if p.shape != (3,) or q.shape != (4,) or own.shape != (4,):
        raise ValueError('finite relative position and two attitudes required')
    if (not np.isfinite(p).all() or not np.isfinite(q).all()
            or not np.isfinite(own).all() or np.linalg.norm(q) < 1e-9
            or np.linalg.norm(own) < 1e-9):
        raise ValueError('finite relative position and nonzero attitudes required')
    q, own = q/np.linalg.norm(q), own/np.linalg.norm(own)
    return quat_wxyz_to_matrix(own) @ (quat_wxyz_to_matrix(q).T @ p)
