"""Physical attitude support for the disarmed spatial setup, not an RL aid."""
import numpy as np
from scipy.spatial.transform import Rotation


def leveling_torque(attitude_xyzw, angular_velocity_body, *, controlled):
    """Bounded body-frame PD torque; exactly zero under autopilot control.

    Pinning angular velocity to zero preserves a post-contact tilted pose and
    hides the rotation from the IMU. A real torque levels the body continuously
    so IMU and GPS describe one physical trajectory. No pose is teleported.
    """
    if controlled:
        return np.zeros(3)
    attitude = np.asarray(attitude_xyzw, dtype=float)
    rate = np.asarray(angular_velocity_body, dtype=float)
    if (
        attitude.shape != (4,)
        or rate.shape != (3,)
        or not (np.isfinite(attitude).all() and np.isfinite(rate).all())
    ):
        raise ValueError("finite physical prearm attitude and body rate required")
    rotation = Rotation.from_quat(attitude)
    matrix = rotation.as_matrix()
    yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    target = Rotation.from_euler("z", yaw)
    error = (rotation.inv() * target).as_rotvec()
    torque = 0.25 * error - 0.15 * rate
    norm = np.linalg.norm(torque)
    return torque * min(1.0, 0.4 / max(norm, 1e-12))
