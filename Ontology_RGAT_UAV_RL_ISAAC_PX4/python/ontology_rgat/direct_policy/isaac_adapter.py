"""Isaac/PX4 boundary helpers for the direct-policy profiles.

Imports stay pure: constructing a payload never imports ROS, starts Isaac, or
acquires a gateway.  The explicit runner performs those operations only after
``--allow-isaac``.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np


def enu_to_ned(vector):
    east, north, up = np.asarray(vector, dtype=float)
    return np.array([north, east, -up], dtype=float)


def ned_to_enu(vector):
    north, east, down = np.asarray(vector, dtype=float)
    return np.array([east, north, -down], dtype=float)


def embed_planar_action(action_xz):
    action = np.asarray(action_xz, dtype=float)
    if action.shape != (2,) or not np.all(np.isfinite(action)):
        raise ValueError("planar action must be finite [a_x,a_z]")
    return np.array([action[0], 0.0, action[1]], dtype=float)


def direct_acceleration_payload(acceleration_enu, *, heading_rad: float,
                                decision_id: int, episode_id: str,
                                policy_version: int, stamp_s: float):
    acceleration = np.asarray(acceleration_enu, dtype=float)
    if acceleration.shape != (3,) or not np.all(np.isfinite(acceleration)):
        raise ValueError("direct acceleration must be a finite ENU vector")
    if not math.isfinite(heading_rad) or not math.isfinite(stamp_s):
        raise ValueError("heading/stamp must be finite")
    return {
        "contract": "spatial-enu-direct-net-acceleration-v2",
        "acceleration_enu_m_s2": acceleration.tolist(),
        "heading_rad": float(heading_rad),
        "decision_id": int(decision_id), "episode_id": str(episode_id),
        "policy_version": int(policy_version), "decision_stamp_s": float(stamp_s),
    }


def trajectory_setpoint_fields(acceleration_enu, heading_rad: float):
    """ROS2 TrajectorySetpoint values: only acceleration and yaw are active."""
    return {"position": [math.nan]*3, "velocity": [math.nan]*3,
            "acceleration": enu_to_ned(acceleration_enu).tolist(),
            "jerk": [math.nan]*3, "yaw": float(math.pi/2-heading_rad),
            "yawspeed": math.nan}


@dataclass(frozen=True)
class AuthorityRecord:
    episode_id: str
    decision_id: int
    policy_version: int
    capture_stamp_s: float | None
    receive_stamp_s: float | None
    decision_stamp_s: float
    command_stamp_s: float
    requested_acceleration_enu_m_s2: tuple[float, float, float]
    limited_acceleration_enu_m_s2: tuple[float, float, float]
    gateway_acceleration_ned_m_s2: tuple[float, float, float]
    measured_acceleration_enu_m_s2: tuple[float, float, float] | None
    control_profile: str
    emergency_intervention: bool

    def to_dict(self):
        return asdict(self)


class CommandOwnership:
    """Reject double writers and stale decisions before a command is emitted."""
    def __init__(self, owner: str = "matlab-port"):
        self.owner = owner
        self.episode_id = None
        self.last_decision = -1

    def reset(self, episode_id: str):
        self.episode_id, self.last_decision = str(episode_id), -1

    def validate(self, payload, *, writer: str):
        if writer != self.owner:
            raise RuntimeError(f"command writer {writer!r} does not own this route")
        if payload["episode_id"] != self.episode_id:
            raise RuntimeError("stale command belongs to another episode")
        if payload["decision_id"] <= self.last_decision:
            raise RuntimeError("duplicate or out-of-order decision")
        self.last_decision = payload["decision_id"]
