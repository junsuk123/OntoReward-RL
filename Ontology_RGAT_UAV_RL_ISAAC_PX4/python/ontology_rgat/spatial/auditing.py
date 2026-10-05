"""Post-policy terminal audits; never observations, rewards or a controller.

SAFE_ABORT is the reference's bounded-recovery timeout label. It does not
itself certify a stationary hold. Preserve that outcome for reproducibility
and require separate evidence before calling an actual integration safe.
"""
import math

import numpy as np
from scipy.spatial.transform import Rotation


def terminal_hold_audit(info):
    """Conservative terminal snapshot check, NOT a sustained-stability proof.

The planar reference's abort_complete limits are extended by horizontal
norm and thrust-axis tilt. Truth is used only here, in isolated evaluation,
to verify noncontact and deck clearance; no audit output feeds a policy.
Missing historical telemetry fails closed without relabeling its outcome.
"""
    result = dict(version="terminal-hold-snapshot/1", evaluated=False,
                  hold_verified=False, sustained_stability_claim=False)
    try:
        velocity = np.asarray(info["own_velocity_enu_m_s"], dtype=float)
        quaternion = np.asarray(info["own_quaternion_wxyz"], dtype=float)
        position = np.asarray(info["truth_relative_position"], dtype=float)
        if (velocity.shape != (3,) or quaternion.shape != (4,)
                or position.shape != (3,) or not np.isfinite(
                    np.r_[velocity, quaternion, position]).all()
                or np.linalg.norm(quaternion) < 1e-9
                or not isinstance(info["truth_contact"], (bool, np.bool_))):
            return result
        thrust_axis = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).apply([0, 0, 1])
        tilt = math.acos(float(np.clip(thrust_axis[2], -1., 1.)))
    except (KeyError, TypeError, ValueError):
        return result
    checks = dict(no_contact=not bool(info["truth_contact"]),
                  clearance=float(position[2]) >= .5,
                  horizontal_speed=float(np.linalg.norm(velocity[:2])) <= .2,
                  vertical_speed=abs(float(velocity[2])) <= .1,
                  tilt=tilt <= math.radians(5))
    result.update(evaluated=True, hold_verified=all(checks.values()), checks=checks,
                  horizontal_speed_m_s=float(np.linalg.norm(velocity[:2])),
                  vertical_speed_m_s=float(velocity[2]), tilt_rad=tilt,
                  deck_clearance_m=float(position[2]))
    return result
