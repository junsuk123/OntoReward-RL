"""Compatibility adapter for the existing three-field PX4 gateway."""
from __future__ import annotations

import numpy as np

from .config import DynamicsConfig
from .dynamics import acceleration_to_thrust_pitch


def px4_gateway_command(normalized_two_axis_action: np.ndarray,
                        dynamics: DynamicsConfig) -> np.ndarray:
    """Return legacy ``[a_fwd, a_z, tilt]`` with tilt derived, never learned.

    The gateway wire shape remains intact while the policy retains exactly two
    degrees of freedom.  The third field is an inner-loop setpoint consequence
    of the requested net acceleration and is not stored as a policy action.
    """
    action = np.clip(np.asarray(normalized_two_axis_action, dtype=float).reshape(2),
                     -1.0, 1.0)
    request = np.array([action[0] * dynamics.ax_max_m_s2,
                        action[1] * dynamics.az_max_m_s2])
    setpoint = acceleration_to_thrust_pitch(request, dynamics)
    normalized_pitch = setpoint.theta_rad / dynamics.pitch_limit_rad
    return np.array([action[0], action[1],
                     float(np.clip(normalized_pitch, -1.0, 1.0))])
