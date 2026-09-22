from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterable


MAX_DATAGRAM = 32_768
BENCHMARK_SCENARIOS = {
    "training_random_walk", "straight_8mps", "linear_acceleration_wave",
    "circle", "zigzag", "u_turn", "vertical_heave_boat",
    # The training walk with one straight dash to the carrier's peak speed,
    # for behaviour-cloning demonstrations that lose and recover the pad
    # (isaac_sim/pad_motion.py, 2026-09-21).
    "training_random_walk_escape_burst",
    # The same dash laid over a constant-velocity straight run instead of the
    # walk: the deck cruises, the vehicle settles into following it, and then
    # it doubles its speed and leaves the frame. The CICS2026 three-arm
    # comparison deck (2026-09-22).
    "straight_escape_burst",
    # The same deck closed into an oval: two straights joined by constant-speed
    # semicircles, driven forever. Bounded by construction, so the deck needs
    # neither the arena clamp nor the inward heading steering that an open
    # straight run does (isaac_sim/pad_motion.py, 2026-09-22).
    "straight_escape_burst_track",
}


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
        "hello", "state", "action", "velocity_action", "reset", "arm", "disarm",
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


def validate_velocity_action(msg: dict[str, Any]) -> tuple[float, float, float, float]:
    """Validate physical [vx, vy, vz, yaw-rate] in body-heading/FLU units."""
    command = finite_vector(msg.get("command", ()), 4, "velocity command")
    limits = (10.0, 10.0, 5.0, math.radians(180.0))
    if any(abs(value) > limit for value, limit in zip(command, limits)):
        raise ProtocolError("velocity command exceeds protocol safety bounds")
    return command


# Guard rails for the pre-episode climb. The gateway must never be talked into
# flying the vehicle outside the city or holding position forever.
# The deck drives a lap of a city block rather than circling a small arena, so
# a world-frame request has to be allowed to reach the far side of it; the
# gateway clamps the derived target as well, against the block it was told
# about rather than against this constant.
GOTO_MAX_RADIUS_M = 140.0
GOTO_MAX_ALTITUDE_M = 25.0
# A pad-frame request is an offset from a deck that is itself moving, so it gets
# a tighter budget than an absolute point; the gateway additionally clamps the
# world target it derives from it.
GOTO_MAX_PAD_RADIUS_M = 10.0
GOTO_FRAMES = ("world", "pad")
# A freshly booted PX4 refuses to arm for roughly 40 s, and the hold has to
# outlast that plus the climb.
# Long enough to cover a PPO/estimator update between measured episodes on
# a rendered two-pair stage; a learner that dies mid-hold still lands.
GOTO_MAX_HOLD_S = 900.0


@dataclass(frozen=True)
class GotoRequest:
    position_enu: tuple[float, float, float]
    yaw_enu_rad: float
    hold_s: float
    frame: str = "world"

    @property
    def is_pad_relative(self) -> bool:
        return self.frame == "pad"


def validate_goto(msg: dict[str, Any]) -> GotoRequest:
    """Validate a position-hold request used to fly the episode entry pose.

    The climb is flown by PX4's position controller, so the only thing the
    protocol has to guarantee is that the requested point is inside the arena.
    With a moving pad the request is an offset in the pad frame and the gateway
    re-streams it against the live deck pose, so the offset is what is bounded.
    """
    frame = str(msg.get("frame", "world")).lower()
    if frame not in GOTO_FRAMES:
        raise ProtocolError(f"goto frame must be one of {GOTO_FRAMES}")
    position = finite_vector(msg.get("position", ()), 3, "position")
    radius = GOTO_MAX_PAD_RADIUS_M if frame == "pad" else GOTO_MAX_RADIUS_M
    if math.hypot(position[0], position[1]) > radius:
        raise ProtocolError("goto position outside the permitted arena radius")
    if not 0.0 < position[2] <= GOTO_MAX_ALTITUDE_M:
        raise ProtocolError("goto altitude must be above the pad and inside the arena")
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
    return GotoRequest(position, yaw, hold_s, frame)


def static_pad_state() -> dict[str, Any]:
    """A pad that never moves, which is what the fixed-pad experiment is."""
    return {
        "valid": True,
        "source": "static",
        "position": [0.0, 0.0, 0.0],
        "velocity": [0.0, 0.0, 0.0],
        "yaw": 0.0,
        "yaw_rate": 0.0,
        "speed": 0.0,
        "sigma_xy_m": 0.0,
    }


def open_sky_gnss_state() -> dict[str, Any]:
    """What a link that models no GNSS degradation must report.

    Every field is an observable a receiver publishes, and every number is
    finite: a consumer that is handed no GNSS information has to behave like
    the open-sky control condition, not read an absent field as a total
    outage. Note what is deliberately absent -- the true position error, the
    true NLOS count and the true sky view. None of those is something a
    receiver knows, so none of them crosses this wire.
    """
    return {
        "enabled": False,
        "source": "unavailable",
        "valid": True,
        "fix_type": 3,
        "satellites_tracked": 12,
        "nlos_detected_fraction": 0.0,
        "cn0_mean_db": 45.0,
        "hdop": 1.0,
        "vdop": 1.6,
        "residual_rms_m": 0.0,
        "sigma_xy_m": 0.0,
        "quality": 1.0,
        # The lorry publishes its own integrity too: the pose it broadcasts is
        # only as good as its receiver, and that receiver is in the same street.
        "deck_quality": 1.0,
        "deck_sigma_xy_m": 0.0,
    }


def unavailable_battery_state() -> dict[str, Any]:
    """Battery fields for a link that cannot report energy at all.

    Every number is finite because the wire format refuses NaN and Infinity on
    purpose. ``enabled: false`` is the only field a consumer may act on: it means
    these are not measurements, so the reserve must not be read as "full" and an
    episode must never be terminated on it.
    """
    return {
        "enabled": False,
        "source": "unavailable",
        "remaining_j": 0.0,
        "initial_j": 0.0,
        "capacity_j": 0.0,
        "energy_used_j": 0.0,
        "power_w": 0.0,
        "hover_power_w": 0.0,
        "hover_seconds_remaining": 0.0,
        "reserve": 1.0,
        "landing_reserve_s": 0.0,
        "state_of_charge": 1.0,
        "voltage_v": 0.0,
        "depleted": False,
    }


@dataclass
class VehicleSample:
    timestamp_ns: int = 0
    # PX4's own clock. Under lockstep SITL this is simulated time, which is the
    # only clock an episode may be paced by.
    px4_time_us: int = 0
    # Pad-relative, because the pad is the target and it moves. The world pose
    # PX4 estimates is carried alongside for telemetry only.
    position_enu: tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity_enu: tuple[float, float, float] = (0.0, 0.0, 0.0)
    position_world_enu: tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity_world_enu: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quaternion_enu_flu_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    angular_velocity_flu: tuple[float, float, float] = (0.0, 0.0, 0.0)
    acceleration_enu: tuple[float, float, float] = (0.0, 0.0, 0.0)
    wind_enu: tuple[float, float, float] = (0.0, 0.0, 0.0)
    aero_force_enu: tuple[float, float, float] = (0.0, 0.0, 0.0)
    marker_quality: float = 0.0
    # Pad-relative position and velocity as the simulator knows them, for
    # scoring the episode only. Under GNSS degradation the pad-relative state
    # the policy flies on is wrong by metres, so grading the landing on it
    # would grade the receiver's mistake instead of the landing.
    truth_position_enu: tuple[float, float, float] | None = None
    truth_velocity_enu: tuple[float, float, float] | None = None
    # Simulator attitude truth, carried so geometric landing-pad field-of-view
    # can be evaluated from simulator geometry alone rather than from the
    # estimator's attitude. Scoring/labelling only, like the two above.
    truth_quaternion_enu_flu_wxyz: tuple[float, float, float, float] | None = None
    armed: bool = False
    nav_state: int = 0
    landed: bool = True
    estimator_valid: bool = False
    source: str = "px4"
    pad: dict[str, Any] = field(default_factory=static_pad_state)
    battery: dict[str, Any] = field(default_factory=unavailable_battery_state)
    gnss: dict[str, Any] = field(default_factory=open_sky_gnss_state)
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
            # The target moves, so the contract is explicit about which frame
            # position/velocity are expressed in rather than leaving it implied.
            "position_frame": "pad",
            "position": list(self.position_enu),
            "velocity": list(self.velocity_enu),
            "world": {
                "position": list(self.position_world_enu),
                "velocity": list(self.velocity_world_enu),
            },
            "pad": dict(self.pad),
            "battery": dict(self.battery),
            "gnss": dict(self.gnss),
            "truth": {
                "valid": self.truth_position_enu is not None,
                "position": list(self.truth_position_enu or self.position_enu),
                "velocity": list(self.truth_velocity_enu or self.velocity_enu),
                "quaternion_wxyz": list(
                    self.truth_quaternion_enu_flu_wxyz
                    or self.quaternion_enu_flu_wxyz),
                "attitude_valid": self.truth_quaternion_enu_flu_wxyz is not None,
            },
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
