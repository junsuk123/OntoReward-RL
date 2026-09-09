from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterable


MAX_DATAGRAM = 32_768


class ProtocolError(ValueError):
    pass


def now_ns() -> int:
    return time.monotonic_ns()


def encode(message: dict[str, Any]) -> bytes:
    raw = json.dumps(message, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(raw) > MAX_DATAGRAM:
        raise ProtocolError("message exceeds UDP datagram limit")
    return raw


def decode(raw: bytes, expected_version: int = 1) -> dict[str, Any]:
    if len(raw) > MAX_DATAGRAM:
        raise ProtocolError("oversized datagram")
    try:
        msg = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("invalid JSON") from exc
    if not isinstance(msg, dict):
        raise ProtocolError("message must be an object")
    if msg.get("v") != expected_version:
        raise ProtocolError("protocol version mismatch")
    if msg.get("type") not in {
        "hello", "state", "action", "reset", "arm", "disarm",
        "goto", "enable_offboard", "disable_offboard", "error", "ack",
    }:
        raise ProtocolError("unknown message type")
    if not isinstance(msg.get("seq"), int) or msg["seq"] < 0:
        raise ProtocolError("invalid sequence")
    return msg


def finite_vector(value: Iterable[Any], length: int, name: str) -> tuple[float, ...]:
    try:
        out = tuple(float(x) for x in value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{name} must be numeric") from exc
    if len(out) != length or not all(math.isfinite(x) for x in out):
        raise ProtocolError(f"{name} must contain {length} finite values")
    return out


def validate_action(msg: dict[str, Any]) -> tuple[float, float, float, float]:
    action = finite_vector(msg.get("action", ()), 4, "action")
    if any(abs(x) > 1.0 for x in action):
        raise ProtocolError("normalized action outside [-1, 1]")
    return action  # collective, roll, pitch, yaw-rate


# Guard rails for the pre-episode climb. The gateway must never be talked into
# flying the vehicle outside the landing arena or holding position forever.
GOTO_MAX_RADIUS_M = 20.0
GOTO_MAX_ALTITUDE_M = 15.0
# A freshly booted PX4 refuses to arm for roughly 40 s, and the hold has to
# outlast that plus the climb.
GOTO_MAX_HOLD_S = 120.0


@dataclass(frozen=True)
class GotoRequest:
    position_enu: tuple[float, float, float]
    yaw_enu_rad: float
    hold_s: float


def validate_goto(msg: dict[str, Any]) -> GotoRequest:
    """Validate a position-hold request used to fly the episode entry pose.

    The climb is flown by PX4's position controller, so the only thing the
    protocol has to guarantee is that the requested point is inside the arena.
    """
    position = finite_vector(msg.get("position", ()), 3, "position")
    if math.hypot(position[0], position[1]) > GOTO_MAX_RADIUS_M:
        raise ProtocolError("goto position outside the permitted arena radius")
    if not 0.0 < position[2] <= GOTO_MAX_ALTITUDE_M:
        raise ProtocolError("goto altitude must be above ground and inside the arena")
    yaw = msg.get("yaw", 0.0)
    try:
        yaw = float(yaw)
    except (TypeError, ValueError) as exc:
        raise ProtocolError("yaw must be numeric") from exc
    if not math.isfinite(yaw):
        raise ProtocolError("yaw must be finite")
    hold_s = msg.get("hold_s", GOTO_MAX_HOLD_S)
    try:
        hold_s = float(hold_s)
    except (TypeError, ValueError) as exc:
        raise ProtocolError("hold_s must be numeric") from exc
    if not math.isfinite(hold_s) or not 0.0 < hold_s <= GOTO_MAX_HOLD_S:
        raise ProtocolError(f"hold_s must be in (0, {GOTO_MAX_HOLD_S}]")
    return GotoRequest(position, yaw, hold_s)


@dataclass
class VehicleSample:
    timestamp_ns: int = 0
    # PX4's own clock. Under lockstep SITL this is simulated time, which is the
    # only clock an episode may be paced by.
    px4_time_us: int = 0
    position_enu: tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity_enu: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quaternion_enu_flu_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    angular_velocity_flu: tuple[float, float, float] = (0.0, 0.0, 0.0)
    acceleration_enu: tuple[float, float, float] = (0.0, 0.0, 0.0)
    wind_enu: tuple[float, float, float] = (0.0, 0.0, 0.0)
    aero_force_enu: tuple[float, float, float] = (0.0, 0.0, 0.0)
    marker_quality: float = 0.0
    armed: bool = False
    nav_state: int = 0
    landed: bool = True
    estimator_valid: bool = False
    source: str = "px4"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_message(self, version: int, seq: int, ack_seq: int) -> dict[str, Any]:
        return {
            "v": version,
            "type": "state",
            "seq": seq,
            "ack_seq": ack_seq,
            "time_ns": now_ns(),
            "sample_time_ns": self.timestamp_ns,
            "px4_time_us": self.px4_time_us,
            "frame": "ENU_FLU",
            "position": list(self.position_enu),
            "velocity": list(self.velocity_enu),
            "quaternion_wxyz": list(self.quaternion_enu_flu_wxyz),
            "angular_velocity": list(self.angular_velocity_flu),
            "acceleration": list(self.acceleration_enu),
            "wind": list(self.wind_enu),
            "aero_force": list(self.aero_force_enu),
            "marker_quality": float(self.marker_quality),
            "armed": self.armed,
            "nav_state": self.nav_state,
            "landed": self.landed,
            "estimator_valid": self.estimator_valid,
            "source": self.source,
            "extra": self.extra,
        }
