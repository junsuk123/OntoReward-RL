"""Compatibility adapter for the existing three-field PX4 gateway."""
from __future__ import annotations

import numpy as np

from .config import DynamicsConfig


def px4_gateway_command(normalized_two_axis_action: np.ndarray,
                        dynamics: DynamicsConfig, *, controller) -> np.ndarray:
    """Map into an explicitly supplied legacy controller envelope.

    Returns normalized controller input, NOT gateway wire bytes. The legacy
    adapter integrates velocity and sends the existing velocity/tilt protocol.
    Its tilt means g*tan(theta) feed-forward, not a thrust/pitch setpoint.
    This conversion alone does not establish closed-loop plant equivalence.
    """
    action = np.asarray(normalized_two_axis_action, dtype=float).reshape(2)
    if not np.isfinite(action).all():
        raise ValueError("action must be finite")
    action = np.clip(action, -1.0, 1.0)
    request = np.array([action[0] * dynamics.ax_max_m_s2,
                        action[1] * dynamics.az_max_m_s2])
    limits = np.asarray(controller.max_acceleration) * controller.action_scale
    tilt_limit = float(controller.max_longitudinal_tilt * controller.action_scale)
    if limits.shape != (2,) or not np.isfinite(limits).all() or min(limits) <= 0 or tilt_limit <= 0:
        raise ValueError("invalid legacy controller envelope")
    from ..controllers.planar_controller import STANDARD_GRAVITY
    pitch = np.arctan2(request[0], STANDARD_GRAVITY)
    command = np.r_[request/limits, pitch/tilt_limit]
    if np.max(np.abs(command)) > 1 + 1e-12:
        raise ValueError("legacy envelope cannot realize this two-axis request; no silent clipping")
    if not controller.tilt_channel_enabled and abs(pitch) > 1e-12:
        raise ValueError("legacy tilt feed-forward is disabled")
    return command
