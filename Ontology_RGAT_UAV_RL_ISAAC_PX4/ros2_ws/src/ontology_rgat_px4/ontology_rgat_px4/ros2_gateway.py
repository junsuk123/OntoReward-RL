from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import threading
from typing import Any

import numpy as np

from .battery import BatteryModel
from .config import GatewayConfig, load_gateway_config
from .frames import (
    enu_to_ned,
    geodetic_to_enu,
    frd_to_flu,
    ned_to_enu,
    quat_enu_flu_to_ned_frd,
    quat_ned_frd_to_enu_flu,
    quat_wxyz_to_matrix,
    yaw_enu_to_ned,
    yaw_from_quat_wxyz,
    euler_zyx_to_quat_wxyz,
)
from .protocol import (
    BENCHMARK_SCENARIOS,
    PLANAR_TILT_LIMIT_RAD,
    ProtocolError,
    VehicleSample,
    now_ns,
    open_sky_gnss_state,
    static_pad_state,
    validate_action,
    validate_velocity_action,
    validate_velocity_extras,
    validate_goto,
)
from .safety import SafetyGate
from .udp_server import DatagramServer

# Standard gravity, used to turn a commanded longitudinal tilt into the
# acceleration feed-forward PX4 realises it with.
GRAVITY_M_S2 = 9.80665

try:
    import rclpy
except ImportError:
    # Imports are completed below. Keeping the error local gives useful --help
    # and permits unit tests on machines without ROS/px4_msgs.
    rclpy = None


def _load_ros_types():
    global rclpy
    if rclpy is None:
        import rclpy as imported_rclpy
        rclpy = imported_rclpy
    from geometry_msgs.msg import PoseStamped, Vector3Stamped
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from px4_msgs.msg import BatteryStatus, FailsafeFlags
    from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint
    from px4_msgs.msg import EstimatorGpsStatus, EstimatorStatusFlags, SensorGps
    from px4_msgs.msg import VehicleAttitudeSetpoint, VehicleCommand, VehicleCommandAck
    from px4_msgs.msg import VehicleLandDetected, VehicleLocalPosition
    from px4_msgs.msg import VehicleOdometry, VehicleStatus, VehicleThrustSetpoint
    from std_msgs.msg import Bool, Float32, String
    return (Node, PoseStamped, Vector3Stamped, Odometry, BatteryStatus, FailsafeFlags,
            EstimatorGpsStatus, EstimatorStatusFlags, SensorGps,
            OffboardControlMode, TrajectorySetpoint,
            VehicleAttitudeSetpoint, VehicleCommand, VehicleCommandAck, VehicleLandDetected,
            VehicleLocalPosition, VehicleOdometry, VehicleStatus, VehicleThrustSetpoint,
            Bool, Float32, String)


def _set_if_present(message: Any, name: str, value: Any) -> None:
    if hasattr(message, name):
        setattr(message, name, value)


def _finite_or(value: Any, fallback: float = 0.0) -> float:
    """Keep optional PX4 diagnostics safe for the strict JSON wire format."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return result if math.isfinite(result) else float(fallback)


FAILSAFE_BOOLEAN_FIELDS = (
    "angular_velocity_invalid", "attitude_invalid", "local_altitude_invalid",
    "local_position_invalid", "local_position_invalid_relaxed",
    "local_velocity_invalid", "global_position_invalid",
    "auto_mission_missing", "offboard_control_signal_lost",
    "home_position_invalid", "manual_control_signal_lost",
    "gcs_connection_lost", "battery_low_remaining_time", "battery_unhealthy",
    "primary_geofence_breached", "mission_failure",
    "vtol_fixed_wing_system_failure", "wind_limit_exceeded",
    "flight_time_limit_exceeded", "local_position_accuracy_low",
    "fd_critical_failure", "fd_esc_arming_failure",
    "fd_imbalanced_prop", "fd_motor_failure",
)

# These conditions can independently represent a physical, estimator or policy
# failure. Expected absent RC/GCS/mission flags are deliberately omitted: PX4
# publishes them in autonomous SITL even when they are irrelevant to OFFBOARD.
FAILSAFE_HARD_FIELDS = frozenset({
    "angular_velocity_invalid", "attitude_invalid", "local_altitude_invalid",
    "local_position_invalid", "local_position_invalid_relaxed",
    "local_velocity_invalid", "global_position_invalid",
    "home_position_invalid", "battery_low_remaining_time", "battery_unhealthy",
    "primary_geofence_breached", "mission_failure",
    "vtol_fixed_wing_system_failure", "wind_limit_exceeded",
    "flight_time_limit_exceeded", "local_position_accuracy_low",
    "fd_critical_failure", "fd_esc_arming_failure",
    "fd_imbalanced_prop", "fd_motor_failure",
})

# These inputs are expected to be absent in autonomous SITL. Immediately after
# a real OFFBOARD heartbeat loss clears, ``failsafe_flags`` can publish the
# cleared offboard bit before ``vehicle_status.failsafe`` clears. In that one
# callback-ordering window only these benign inputs remain. Treating the
# snapshot as a hard vehicle fault aborts a resumable multi-hour run.
FAILSAFE_SITL_INFRASTRUCTURE_FIELDS = frozenset({
    "auto_mission_missing", "offboard_control_signal_lost",
    "manual_control_signal_lost", "gcs_connection_lost",
})

# ``fd_critical_failure`` is PX4's attitude failure detector: roll or pitch
# beyond FD_FAIL_R/FD_FAIL_P for FD_FAIL_R_TTRI seconds. On this airframe
# nothing else can raise it -- FAILURE_ALT is unused and FAILURE_EXT needs an
# ATS receiver SITL does not have. It therefore means one thing: the vehicle
# tipped over. That is the task failing, not the infrastructure, and the
# learner already classifies it (``LiveShinEnvironment`` crash_tilt). Naming it
# separately lets the bridge end the episode as a crash instead of ending the
# run, which is what a hard-failsafe abort did on 2026-09-23: PX4 asserts the
# bit at 60 deg while the learner's own crash verdict sits at 75 deg, so every
# genuine tip-over was intercepted before it could ever be scored.
FAILSAFE_ATTITUDE_FIELDS = frozenset({"fd_critical_failure"})


def failsafe_detail(message: Any, *, target: str = "sitl") -> dict[str, Any]:
    """Expose PX4 failsafe inputs and classify safe automatic recovery.

    An OFFBOARD signal loss in SITL is a ROS 2/uXRCE transport failure and may
    be retried after cycling the simulator. PX4 can clear that bit one callback
    before ``VehicleStatus.failsafe``; the remaining mission/RC/GCS-missing
    inputs are therefore the same recoverable transition. Any hard input or
    any hardware failsafe remains non-recoverable.
    """
    active = [name for name in FAILSAFE_BOOLEAN_FIELDS
              if bool(getattr(message, name, False))]
    battery_warning = int(getattr(message, "battery_warning", 0))
    if battery_warning:
        active.append(f"battery_warning_{battery_warning}")
    hard_active = FAILSAFE_HARD_FIELDS.intersection(active)
    hard = bool(hard_active or battery_warning)
    recoverable = bool(
        str(target).lower() == "sitl"
        and active
        and set(active).issubset(FAILSAFE_SITL_INFRASTRUCTURE_FIELDS)
        and not hard)
    # A tip-over and nothing else. Any second hard input (estimator, battery,
    # geofence, motor) keeps the snapshot a non-recoverable abort, because then
    # the attitude is a symptom and the learner's crash verdict would hide the
    # cause.
    attitude_failure = bool(hard_active == FAILSAFE_ATTITUDE_FIELDS
                            and not battery_warning)
    return {
        "reasons": active,
        "battery_warning": battery_warning,
        "recoverable_infrastructure": recoverable,
        "attitude_failure": attitude_failure,
    }


def offboard_recovery_allowed(failsafe_active: bool,
                              detail: Any) -> bool:
    """Decide whether to keep asking PX4 for OFFBOARD during a failsafe.

    A momentarily starved heartbeat drops PX4 into its offboard-loss reaction,
    and ``VehicleStatus.failsafe`` stays latched there until something commands
    the vehicle back: PX4 does not return to OFFBOARD on its own. Gating the
    mode request on that same latch made a sub-second transport blip cost the
    whole entry budget -- the stream resumes, ``offboard_control_signal_lost``
    clears, and nothing ever asks for OFFBOARD again, so the vehicle drifts
    under AUTO until the entry gate gives up and rebuilds the shared simulator.

    Recovery is offered only once the cause has cleared: a recoverable SITL
    classification whose remaining inputs are the mission/RC/GCS flags PX4
    publishes throughout autonomous SITL anyway. While the offboard signal is
    still lost PX4 would reject the request, and six consecutive rejections
    abort the episode, so that window stays silent -- as does any hard input.
    """
    if not failsafe_active:
        return True
    detail = detail if isinstance(detail, dict) else {}
    if not bool(detail.get("recoverable_infrastructure", False)):
        return False
    reasons = detail.get("reasons", ())
    if not isinstance(reasons, (list, tuple)):
        return False
    return "offboard_control_signal_lost" not in reasons


def advance_pad_contact_latch(latched: bool, armed_clear: bool,
                              armed: bool, raw_contact: bool,
                              px4_landed: bool = False,
                              airborne_clearance: bool = True) -> tuple[bool, bool]:
    """Latch only after both PX4 and the contact switch confirm takeoff."""
    if not armed:
        return bool(latched), bool(armed_clear)
    if not raw_contact and not px4_landed and airborne_clearance:
        armed_clear = True
    # PX4 can report ``landed`` while an externally supported airborne start
    # is hovering with its motors idle. It is useful for confirming that the
    # vehicle has cleared the launch surface, but it must never masquerade as
    # physical roof contact. The two signals are fused later in
    # ``_refresh_land_detector`` without relabelling their source.
    elif armed_clear and raw_contact:
        latched = True
    return bool(latched), bool(armed_clear)


def effective_px4_landed(raw_landed: bool, armed: bool, truth_position,
                         airborne_clearance_m: float = 0.5) -> bool:
    """Reject a delayed PX4 landed bit while SITL truth is clearly airborne."""
    truth = np.asarray(truth_position if truth_position is not None else (),
                       dtype=float).reshape(-1)
    clearly_airborne = bool(
        armed and truth.size == 3 and np.isfinite(truth).all()
        and float(truth[2]) >= float(airborne_clearance_m))
    return bool(raw_landed and not clearly_airborne)


def action_age_seconds(target: str, wall_now_ns: int, last_wall_ns: int,
                       px4_time_us: int, last_px4_time_us: int) -> float:
    """Use the safe common age across lockstep and DDS delivery in SITL.

    Slow rendering makes wall age much larger than simulated age, while a DDS
    backlog can make the newest delivered PX4 stamp jump far ahead of the stamp
    present when an action was received. Either is a false timeout on its own;
    a genuinely missing client advances both clocks past the limit.
    """
    wall_age = max(0.0, (wall_now_ns - last_wall_ns) * 1e-9)
    if (target == "sitl" and px4_time_us > 0 and last_px4_time_us > 0
            and px4_time_us >= last_px4_time_us):
        sim_age = (px4_time_us - last_px4_time_us) * 1e-6
        return min(wall_age, sim_age)
    return wall_age


class ContinuousPx4Clock:
    """Preserve simulated-time deltas across XRCE-DDS clock rebases.

    PX4 timestamps exported through XRCE-DDS include the agent time offset.
    When the timesync filter resets, that offset can disappear and later return,
    making the raw timestamp jump between Unix-epoch and boot-time domains.
    Neither jump is simulated progress. Accumulating only credible consecutive
    deltas gives the learner a continuous lockstep clock without replacing it
    with wall time.
    """

    def __init__(self, max_forward_gap_us: int = 10_000_000):
        self.max_forward_gap_us = int(max_forward_gap_us)
        if self.max_forward_gap_us <= 0:
            raise ValueError("maximum PX4 clock gap must be positive")
        self.raw_time_us: int | None = None
        self.logical_time_us: int | None = None
        self.last_delta_us = 0
        self.discontinuities = 0

    def update(self, raw_time_us: int) -> tuple[int, bool]:
        raw = int(raw_time_us)
        if raw <= 0:
            return int(self.logical_time_us or 0), False
        if self.raw_time_us is None:
            self.raw_time_us = raw
            self.logical_time_us = raw
            return raw, False

        delta = raw - self.raw_time_us
        self.raw_time_us = raw
        self.last_delta_us = delta
        discontinuity = delta < 0 or delta > self.max_forward_gap_us
        if discontinuity:
            self.discontinuities += 1
            return int(self.logical_time_us), True

        self.logical_time_us = int(self.logical_time_us) + delta
        return int(self.logical_time_us), False


def bounded_position_update(reference, measurement, max_correction_m: float) -> np.ndarray:
    """Apply a finite optical correction without permitting a pose jump."""
    result = np.asarray(reference, dtype=float).copy()
    delta = np.asarray(measurement, dtype=float) - result
    distance = float(np.linalg.norm(delta))
    limit = max(0.0, float(max_correction_m))
    if distance > limit and distance > 1e-9:
        delta *= limit / distance
    return result + delta


# A deck twist far above the configured rover ceiling is a corrupt sample,
# not a fast rover. Feeding that forward would fly the vehicle away from the
# deck it is supposed to be waiting over.
ENTRY_FEEDFORWARD_MAX_SPEED_M_S = 3.0


def entry_feedforward_velocity(deck_velocity,
                               max_speed_m_s: float = ENTRY_FEEDFORWARD_MAX_SPEED_M_S):
    """The deck velocity to feed forward on a pad-relative entry setpoint.

    The entry setpoint is re-aimed at the live deck every control tick, but it
    used to carry position alone. PX4's position loop then has to manufacture
    the whole chase velocity out of position error, so it settles at a standing
    lag of roughly ``v_deck / MPC_XY_P``: at the configured 0.60 m/s rover
    ceiling and the stock 0.95 1/s gain that is ~0.63 m of permanent offset,
    against a 0.90 m entry tolerance -- before the rover turns or the seeded
    speed perturbation moves it at all. The gate then expires with the vehicle
    trailing the deck, which reads as an infrastructure failure and restarts a
    simulator that was never at fault.

    Handing PX4 the deck's own velocity leaves the position loop to correct
    only the residual. This is setup, not the experiment: it decides where the
    vehicle waits before handover, it is identical for both arms, and nothing
    from it reaches the policy, the reward or the log.

    ``None`` means "publish no feed-forward", which is the previous behaviour.
    """
    velocity = np.asarray(deck_velocity, dtype=float).reshape(-1)
    if velocity.shape != (3,) or not np.isfinite(velocity).all():
        return None
    limit = float(max_speed_m_s)
    if not math.isfinite(limit) or limit <= 0.0:
        return None
    speed = float(np.linalg.norm(velocity))
    if speed > limit:
        # A sample this far out of range is not trustworthy enough to scale
        # down and fly; fall back to position-only control.
        return None
    return velocity


def advance_velocity_position_target(reference, velocity_enu, dt_s: float, *,
                                     floor_z_m: float, ceiling_z_m: float,
                                     world_radius_m: float) -> np.ndarray:
    """Integrate one finite velocity command into a bounded world pose.

    PX4 accepts position setpoints with velocity feed-forward.  Keeping a
    continuous position target through the entry-hold/policy handover prevents
    a near-zero, untrained action from turning a stable hover into an open
    velocity-mode drop.  The helper is pure so the safety bounds can be tested
    without ROS or PX4 message packages.
    """
    target = np.asarray(reference, dtype=float).reshape(-1)
    velocity = np.asarray(velocity_enu, dtype=float).reshape(-1)
    values = (float(dt_s), float(floor_z_m), float(ceiling_z_m),
              float(world_radius_m))
    if (target.size != 3 or velocity.size != 3
            or not np.isfinite(target).all()
            or not np.isfinite(velocity).all()
            or not all(math.isfinite(value) for value in values)):
        raise ValueError("position target update requires finite 3-vectors and bounds")
    if dt_s < 0.0:
        raise ValueError("position target dt must be non-negative")
    if ceiling_z_m < floor_z_m:
        raise ValueError("position target ceiling must not be below its floor")
    if world_radius_m <= 0.0:
        raise ValueError("position target world radius must be positive")
    result = target + velocity * dt_s
    radial = float(np.linalg.norm(result[:2]))
    if radial > world_radius_m:
        result[:2] *= world_radius_m / radial
    result[2] = float(np.clip(result[2], floor_z_m, ceiling_z_m))
    return result


def _topic(cfg: GatewayConfig, direction: str, name: str) -> str:
    return f"{cfg.namespace}/{direction}/{name}"


def _landing_topic(cfg: GatewayConfig, legacy: str, relative: str) -> str:
    """Return a pair-local simulator topic without changing legacy runs."""
    if not cfg.topic_root:
        return legacy
    return f"{cfg.topic_root}/{relative.lstrip('/')}"


def parallel_gateway_config(cfg: GatewayConfig, pair_index: int,
                            pair_count: int, gateway_port: int | None = None):
    """Resolve one PX4/gateway/topic identity for a shared-world pair."""
    index, count = int(pair_index), int(pair_count)
    if count < 1 or index not in range(count):
        raise ValueError("pair-index must be in [0, parallel-pairs)")
    if count == 1:
        return (cfg if gateway_port is None else
                replace(cfg, gateway_port=int(gateway_port)))
    return replace(
        cfg,
        pair_index=index,
        pair_count=count,
        topic_root=f"/landing_pair_{index}",
        namespace=("/fmu" if index == 0 else f"/px4_{index}/fmu"),
        target_system=index + 1,
        source_system=int(cfg.source_system) + index,
        gateway_port=(int(gateway_port) if gateway_port is not None
                      else int(cfg.gateway_port) + 2 * index),
        matlab_port=int(cfg.matlab_port) + 2 * index,
    )


class Px4GatewayNode:
    """Composition wrapper so importing this module does not require ROS."""

    def __new__(cls, cfg: GatewayConfig, allow_arm: bool = False, allow_offboard: bool = False):
        types = _load_ros_types()
        return _Px4GatewayNode(cfg, SafetyGate(cfg.target, allow_arm, allow_offboard), types)


def _make_qos(rclpy_module):
    return rclpy_module.qos.QoSProfile(
        reliability=rclpy_module.qos.ReliabilityPolicy.BEST_EFFORT,
        durability=rclpy_module.qos.DurabilityPolicy.TRANSIENT_LOCAL,
        history=rclpy_module.qos.HistoryPolicy.KEEP_LAST,
        depth=1,
    )


def _Px4GatewayNode(cfg: GatewayConfig, safety: SafetyGate, types):
    (Node, PoseStamped, Vector3Stamped, Odometry, BatteryStatus, FailsafeFlags,
     EstimatorGpsStatus, EstimatorStatusFlags, SensorGps,
     OffboardControlMode, TrajectorySetpoint,
     VehicleAttitudeSetpoint, VehicleCommand, VehicleCommandAck, VehicleLandDetected,
     VehicleLocalPosition, VehicleOdometry, VehicleStatus, VehicleThrustSetpoint,
     Bool, Float32, String) = types

    class NodeImpl(Node):
        def __init__(self):
            super().__init__(f"ontology_rgat_px4_gateway_{cfg.pair_index}")
            # Sensor topics can arrive hundreds of times per second.  Keep the
            # OFFBOARD heartbeat in its own callback group so those callbacks
            # cannot starve it past PX4's 500 ms loss threshold.  The lock
            # serializes the small set of command fields shared with UDP.
            from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
            self.control_callback_group = MutuallyExclusiveCallbackGroup()
            self.control_lock = threading.RLock()
            self.cfg = cfg
            self.safety = safety
            self.sample = VehicleSample()
            self.action = (0.0, 0.0, 0.0, 0.0)
            self.velocity_action = (0.0, 0.0, 0.0, 0.0)
            # Planar envelope: the commanded longitudinal tilt and the absolute
            # heading to hold. ``None`` restores the free-yaw behaviour every
            # earlier run had.
            self.velocity_tilt_rad = 0.0
            self.velocity_yaw_hold_rad: float | None = None
            self.velocity_position_target_enu: np.ndarray | None = None
            self.velocity_position_time_us = 0
            self.command_interface = "attitude"
            self.last_action_ns = 0
            self.last_action_px4_time_us = 0
            self.last_command_seq = -1
            self.pending_state_ack = -1
            # Action/reset replies are asynchronous: odometry and Isaac's
            # reset acknowledgement arrive after the UDP receive callback has
            # returned.  Remember the requesting endpoint so a concurrent
            # read-only protocol probe cannot steal the flight controller's
            # next reply by becoming DatagramServer.peer in the meantime.
            self.pending_state_peer: tuple[str, int] | None = None
            self.tx_seq = 0
            self.start_ns = now_ns()
            self.prestream = 0
            self.offboard_requested = False
            self.offboard_enabled = cfg.target == "sitl"
            self.last_velocity: np.ndarray | None = None
            self.last_velocity_ns = 0
            self.px4_clock = ContinuousPx4Clock()
            self.pending_reset_seq = -1
            self.pending_reset_peer: tuple[str, int] | None = None
            self.pad_position_enu: np.ndarray | None = None
            self.pad_pose_time_ns = 0
            self.marker_pose_rejections = 0
            # Continuity across a camera/PX4 position handover; see
            # _blend_position_source.
            self._last_source_was_optical: bool | None = None
            self._last_policy_position = np.zeros(3)
            self._source_offset = np.zeros(3)
            self._source_offset_ns = 0
            # The deck the pad rides on. A static pad is the same code path with
            # a zero twist, so there is no second branch to keep in step.
            # PX4's local frame expressed in the world frame the city and the
            # deck live in; see _update_world_origin.
            self.world_from_px4 = np.zeros(3)
            self.world_origin_known = False
            # PX4's own world pose, before the canyon error is injected.
            self.px4_world_position: np.ndarray | None = None
            self.deck_position_enu = np.zeros(3)
            # A filtered copy of the above, used only to aim the entry climb;
            # see _track_deck.
            self.deck_track_position = np.zeros(3)
            self.deck_track_ns = 0
            self.pad_track_position = np.zeros(3)
            self.pad_track_ns = 0
            self.deck_velocity_enu = np.zeros(3)
            self.deck_yaw = 0.0
            self.deck_yaw_rate = 0.0
            self.deck_time_ns = 0
            self.deck_sigma_xy_m = 0.0
            self.warned_missing_deck = False
            # The simulator's own deck pose, for scoring the episode only. It
            # never reaches the policy: see docs/ARCHITECTURE.md, "GNSS".
            self.deck_truth_position_enu: np.ndarray | None = None
            self.deck_truth_velocity_enu = np.zeros(3)
            self.uav_truth_position_enu: np.ndarray | None = None
            self.uav_truth_velocity_enu = np.zeros(3)
            self.uav_truth_quaternion_wxyz: np.ndarray | None = None
            # The drone receiver is normally injected upstream through HIL_GPS.
            # The offset members remain only for legacy telemetry-only runs;
            # `gnss_injected_into_px4` prevents accidental double injection.
            self.gnss = open_sky_gnss_state()
            self.gnss_offset = np.zeros(3)
            self.gnss_velocity_offset = np.zeros(3)
            self.gnss_injected_into_px4 = False
            self.gnss_time_ns = 0
            self.warned_missing_gnss = False
            # Pre-episode climb flown by PX4's own position controller.
            self.goto_target_enu: np.ndarray | None = None
            self.goto_pad_relative = False
            self.goto_yaw_enu = 0.0
            self.goto_deadline_ns = 0
            self.battery = BatteryModel(cfg.battery)
            self.battery_armed = False
            self.pending_battery_hover_s: float | None = None
            self.battery_time_us = 0
            self.px4_battery_time_ns = 0
            self.estimator_health_ns = 0
            self.estimator_healthy = False
            self.land_detected_ns = 0
            self.px4_landed = False
            self.pad_contact_raw = False
            self.pad_contact_latched = False
            # The vehicle begins each ordinary episode on the roof. A contact
            # can only become touchdown after it has armed and cleared the roof.
            self.pad_contact_armed_clear = False
            self.pad_contact_ns = 0
            self.warned_missing_land_detector = False
            self.offboard_nav_state = int(getattr(VehicleStatus, "NAVIGATION_STATE_OFFBOARD", 14))
            # The policy remains at the paper's 10 Hz; this only repeats the
            # latest setpoint and OffboardControlMode heartbeat at a robust
            # transport rate.
            self.heartbeat_hz = max(20.0, float(cfg.control_hz))
            self.mode_request_period = max(1, int(round(self.heartbeat_hz / 2.0)))
            self.last_mode_request_tick = -self.mode_request_period
            self.offboard_mode_rejections = 0
            self.px4_failsafe = False
            self.clock_skew_baseline_us: int | None = None
            self.px4_failsafe_detail = {
                "reasons": [], "battery_warning": 0,
                "recoverable_infrastructure": False,
                "attitude_failure": False,
            }
            self.reported_failsafe_signature: tuple[str, ...] | None = None

            qos = _make_qos(rclpy)
            self.offboard_pub = self.create_publisher(
                OffboardControlMode, _topic(cfg, "in", "offboard_control_mode"), 10)
            self.attitude_pub = self.create_publisher(
                VehicleAttitudeSetpoint, _topic(cfg, "in", "vehicle_attitude_setpoint"), 10)
            self.trajectory_pub = self.create_publisher(
                TrajectorySetpoint, _topic(cfg, "in", "trajectory_setpoint"), 10)
            self.command_pub = self.create_publisher(
                VehicleCommand, _topic(cfg, "in", "vehicle_command"), 10)
            self.reset_pub = self.create_publisher(
                String, _landing_topic(cfg, "/landing_sim/reset", "sim/reset"), 10)
            # Isaac holds the vehicle at its hover start until the autopilot is
            # actually flying it; this is how it learns that. Latched depth so a
            # simulator that comes up late still gets the current answer.
            self.flight_pub = self.create_publisher(
                String, _landing_topic(
                    cfg, "/landing_sim/flight_state", "sim/flight_state"), 10)
            self.published_flight_state: tuple[bool, bool, bool, int] | None = None

            self.create_subscription(VehicleOdometry, _topic(cfg, "out", "vehicle_odometry"),
                                     self._on_odometry, qos)
            self.create_subscription(VehicleStatus, _topic(cfg, "out", "vehicle_status"),
                                     self._on_status, qos)
            self.create_subscription(FailsafeFlags, _topic(cfg, "out", "failsafe_flags"),
                                     self._on_failsafe_flags, qos)
            self.create_subscription(VehicleLandDetected, _topic(cfg, "out", "vehicle_land_detected"),
                                     self._on_land, qos)
            self.create_subscription(VehicleLocalPosition, _topic(cfg, "out", "vehicle_local_position"),
                                     self._on_local_position, qos)
            self.create_subscription(SensorGps, _topic(cfg, "out", "vehicle_gps_position"),
                                     self._on_sensor_gps, qos)
            self.create_subscription(EstimatorGpsStatus,
                                     _topic(cfg, "out", "estimator_gps_status"),
                                     self._on_estimator_gps, qos)
            self.create_subscription(EstimatorStatusFlags,
                                     _topic(cfg, "out", "estimator_status_flags"),
                                     self._on_estimator_flags, qos)
            self.create_subscription(VehicleCommandAck, _topic(cfg, "out", "vehicle_command_ack"),
                                     self._on_command_ack, qos)
            self.create_subscription(VehicleThrustSetpoint,
                                     _topic(cfg, "out", "vehicle_thrust_setpoint"),
                                     self._on_thrust_setpoint, qos)
            sensor_qos = rclpy.qos.qos_profile_sensor_data
            # Policy-safe measurement. /environment/wind is simulator truth
            # and deliberately has no subscriber on the control path.
            self.create_subscription(Vector3Stamped, _landing_topic(
                                     cfg, "/landing_uav0/sensors/wind", "uav/sensors/wind"),
                                     self._on_wind, sensor_qos)
            self.create_subscription(Vector3Stamped, _landing_topic(
                                     cfg, "/landing_uav0/environment/aero_force", "uav/environment/aero_force"),
                                     self._on_aero_force, sensor_qos)
            self.create_subscription(Float32, _landing_topic(
                                     cfg, "/landing_uav0/perception/marker_quality", "uav/perception/marker_quality"),
                                     self._on_marker_quality, sensor_qos)
            self.create_subscription(Bool, _landing_topic(
                                     cfg, "/landing_uav0/perception/pad_contact", "uav/perception/pad_contact"),
                                     self._on_pad_contact, sensor_qos)
            self.create_subscription(PoseStamped, _landing_topic(
                                     cfg, "/landing_uav0/perception/uav_pose_in_pad", "uav/perception/uav_pose_in_pad"),
                                     self._on_pad_pose, sensor_qos)
            # The deck broadcasts its own state, the way a cooperative ground
            # vehicle would over a V2V link. The drone's own estimate of where
            # the pad is still comes from the camera.
            self.create_subscription(Odometry, _landing_topic(
                                     cfg, "/landing_pad/state/odom", "pad/state/odom"),
                                     self._on_deck_odom, sensor_qos)
            self.create_subscription(Odometry, _landing_topic(
                                     cfg, "/landing_pad/state/odom_truth", "pad/state/odom_truth"),
                                     self._on_deck_truth, sensor_qos)
            self.create_subscription(Odometry, _landing_topic(
                                     cfg, "/landing_uav0/state/odom_truth", "uav/state/odom_truth"),
                                     self._on_uav_truth, sensor_qos)
            # Receiver observables plus simulator-only truth used for scoring.
            # The navigation error itself has already crossed HIL_GPS.
            self.create_subscription(String, _landing_topic(
                                     cfg, "/landing_uav0/gnss/status", "uav/gnss/status"),
                                     self._on_gnss_status, sensor_qos)
            self.create_subscription(BatteryStatus, _topic(cfg, "out", "battery_status"),
                                     self._on_battery_status, qos)
            self.create_subscription(String, _landing_topic(
                                     cfg, "/landing_sim/reset_ack", "sim/reset_ack"),
                                     self._on_reset_ack, 10)

            self.udp = DatagramServer(
                cfg.bind_host, cfg.gateway_port, cfg.protocol_version,
                self._on_udp_synchronized)
            self.create_timer(0.005, self.udp.poll)
            self.create_timer(
                1.0 / self.heartbeat_hz,
                self._control_tick_synchronized,
                callback_group=self.control_callback_group)
            self.get_logger().info(
                f"gateway target={cfg.target}, UDP={cfg.bind_host}:{cfg.gateway_port}, "
                f"PX4 namespace={cfg.namespace}, arm_allowed={safety.may_arm()}, "
                f"offboard_allowed={safety.may_enable_offboard()}, "
                f"heartbeat={self.heartbeat_hz:g} Hz, "
                f"pad_motion={cfg.pad_motion}, gnss={'on' if cfg.gnss_enabled else 'off'}, "
                f"battery={'on' if cfg.battery.enabled else 'off'} "
                f"(hover {self.battery.hover_power_w:.0f} W)"
            )

        def destroy_node(self):
            self.udp.close()
            return super().destroy_node()

        def _timestamp_us(self) -> int:
            return self.get_clock().now().nanoseconds // 1000

        def _on_udp_synchronized(self, msg: dict[str, Any]) -> None:
            with self.control_lock:
                self._on_udp(msg)

        def _control_tick_synchronized(self) -> None:
            with self.control_lock:
                self._control_tick()

        def _on_udp(self, msg: dict[str, Any]) -> None:
            kind = msg["type"]
            seq = msg["seq"]
            if seq <= self.last_command_seq and kind not in {"hello", "state"}:
                self._send_ack(seq, "duplicate")
                return
            if kind == "action":
                self.action = validate_action(msg)
                self.command_interface = "attitude"
                self.velocity_position_target_enu = None
                self.velocity_position_time_us = 0
                self.last_action_ns = now_ns()
                self.last_action_px4_time_us = int(self.sample.px4_time_us)
                self.last_command_seq = seq
                self.pending_state_ack = seq
                self.pending_state_peer = self.udp.peer
                self.goto_target_enu = None
                if not self.battery_armed:
                    self._arm_battery()
                self._publish_flight_state()
            elif kind == "velocity_action":
                first_policy_action = not bool(self.last_action_ns)
                self.velocity_action = validate_velocity_action(msg)
                (self.velocity_tilt_rad,
                 self.velocity_yaw_hold_rad) = validate_velocity_extras(msg)
                new_velocity_handover = (
                    self.command_interface != "velocity_yaw_rate"
                    or self.velocity_position_target_enu is None)
                self.command_interface = "velocity_yaw_rate"
                self.last_action_ns = now_ns()
                self.last_action_px4_time_us = int(self.sample.px4_time_us)
                self.last_command_seq = seq
                self.pending_state_ack = seq
                self.pending_state_peer = self.udp.peer
                # The first policy action ends the pre-episode position hold and
                # starts the energy budget: the seeded reserve is the reserve at
                # handover, so the climb PX4 flew to get here is not charged to
                # the policy.
                self.goto_target_enu = None
                if new_velocity_handover:
                    here = self.px4_world_position
                    if here is None:
                        candidate = np.asarray(
                            self.sample.position_world_enu, dtype=float)
                        if candidate.shape == (3,) and np.isfinite(candidate).all():
                            here = candidate
                    if here is None:
                        # This fallback is only reachable before the first PX4
                        # world pose.  It preserves the current pad-relative
                        # state rather than creating a target at world zero.
                        here = self.deck_position_enu + np.asarray(
                            self.sample.position_enu, dtype=float)
                    self.velocity_position_target_enu = np.asarray(
                        here, dtype=float).copy()
                    self.velocity_position_time_us = int(self.sample.px4_time_us)
                # Only rejections occurring after policy handover belong to
                # this measured episode. Pre-arm transient rejections are not
                # allowed to poison its health status.
                if first_policy_action:
                    self.offboard_mode_rejections = 0
                    self.sample.extra["offboard_mode_rejections"] = 0
                if not self.battery_armed:
                    self._arm_battery()
                self._publish_flight_state()
            elif kind == "reset":
                self.safety.require_reset()
                wind_scale = float(msg.get("wind_scale", 1.0))
                if not math.isfinite(wind_scale) or not 0.0 <= wind_scale <= 4.0:
                    raise ProtocolError("wind_scale must be finite and in [0,4]")
                # Scales the deck speed Isaac draws for the episode, so a sweep
                # can ask "how fast a rover can this policy still land on".
                pad_scale = float(msg.get("pad_scale", 1.0))
                if not math.isfinite(pad_scale) or not 0.0 <= pad_scale <= 4.0:
                    raise ProtocolError("pad_scale must be finite and in [0,4]")
                initial_condition_scale = float(msg.get(
                    "initial_condition_scale", min(pad_scale, 1.0)))
                if (not math.isfinite(initial_condition_scale)
                        or not 0.0 <= initial_condition_scale <= 1.0):
                    raise ProtocolError(
                        "initial_condition_scale must be finite and in [0,1]")
                # Scales the canyon's error mechanisms without moving a
                # building, so a sweep can ask how much GNSS degradation a
                # policy survives; 0.0 is the open-sky control condition.
                gnss_scale = float(msg.get("gnss_scale", 1.0))
                if not math.isfinite(gnss_scale) or not 0.0 <= gnss_scale <= 4.0:
                    raise ProtocolError("gnss_scale must be finite and in [0,4]")
                scenario = str(msg.get("scenario", "training_random_walk"))
                if scenario not in BENCHMARK_SCENARIOS:
                    raise ProtocolError(f"unsupported benchmark scenario {scenario!r}")
                self.last_command_seq = seq
                self.pending_reset_seq = seq
                self.pending_reset_peer = self.udp.peer
                # ``finish_episode`` installs this bounded pad-relative hold.
                # Keep it streaming until the freshly seeded entry goto
                # replaces it.  Clearing the target while the photoreal scene
                # prepares its reset acknowledgement leaves no setpoint for up
                # to a second and trips PX4's OFFBOARD-loss failsafe between
                # otherwise continuous airborne episodes.
                keep_airborne_hold = bool(
                    cfg.start_airborne and self.sample.armed
                    and self.goto_target_enu is not None)
                # A disarmed vehicle is on the ground whatever the land detector
                # says -- and before its first sample arrives it says "airborne"
                # by design (see _refresh_land_detector), so without this the
                # first reset after every gateway start put a parked PX4 into
                # AUTO.LAND straight out of "Ready for takeoff".
                if self.sample.landed or not self.sample.armed:
                    self._vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)
                elif cfg.start_airborne:
                    # The controlled benchmark ends one airborne episode by
                    # flying a bounded position hold. Keep that continuous
                    # flight state: AUTO.LAND cannot be cancelled reliably
                    # enough to establish the next entry pose inside its reset
                    # deadline, and every measured episode begins only after
                    # the following goto has settled at an airborne hover.
                    self.get_logger().info(
                        "reset while airborne; retaining flight for next entry hover")
                else:
                    # Cutting power to an airborne vehicle used to be harmless
                    # because Isaac teleported it anyway. It no longer does, so
                    # a reset mid-flight has to be a commanded landing; the
                    # climb that follows simply takes control back.
                    self._vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                    self.get_logger().info(
                        "reset while airborne; commanded landing instead of disarm")
                req = String()
                req.data = json.dumps({"v": cfg.protocol_version, "seq": seq,
                                       "seed": int(msg.get("seed", 0)),
                                       "wind_scale": wind_scale,
                                       "pad_scale": pad_scale,
                                       "initial_condition_scale":
                                           initial_condition_scale,
                                       "gnss_scale": gnss_scale,
                                       "scenario": scenario})
                self.reset_pub.publish(req)
                self.action = (0.0, 0.0, 0.0, 0.0)
                self.last_action_ns = 0
                self.last_action_px4_time_us = 0
                self.velocity_position_target_enu = None
                self.velocity_position_time_us = 0
                # A previous terminal outcome deliberately disabled offboard.
                # Reset starts a new, independently authorised SITL episode.
                self.offboard_enabled = cfg.target == "sitl"
                if not keep_airborne_hold:
                    self.prestream = 0
                    self.offboard_requested = False
                    self.last_mode_request_tick = -self.mode_request_period
                    self.goto_target_enu = None
                    self.goto_pad_relative = False
                self.deck_track_ns = 0
                self.pad_track_ns = 0
                self.battery_armed = False
                self.pending_battery_hover_s = None
                self.pad_contact_raw = False
                self.pad_contact_latched = False
                self.pad_contact_armed_clear = False
                self.pad_contact_ns = 0
                self.sample.extra["pad_contact"] = False
                self._publish_flight_state()
            elif kind == "goto":
                self.safety.require_autonomous_climb()
                request = validate_goto(msg)
                if request.is_pad_relative and not self._deck_is_fresh(now_ns()):
                    raise ProtocolError(
                        "pad-relative goto needs a fresh /landing_pad/state/odom")
                if request.is_pad_relative and not self.world_origin_known:
                    # Flying it anyway would aim at a world-frame point in
                    # PX4's local frame and put the vehicle a block away.
                    raise ProtocolError(
                        "pad-relative goto needs PX4's global origin; "
                        "vehicle_local_position has not reported xy_global yet")
                self.last_command_seq = seq
                self.goto_target_enu = np.asarray(request.position_enu, dtype=float)
                self.velocity_position_target_enu = None
                self.velocity_position_time_us = 0
                self.goto_pad_relative = request.is_pad_relative
                self.goto_yaw_enu = request.yaw_enu_rad
                self.goto_deadline_ns = now_ns() + int(request.hold_s * 1e9)
                # An action deadman must not cancel the climb that precedes it.
                self.action = (0.0, 0.0, 0.0, 0.0)
                self.last_action_ns = 0
                self.last_action_px4_time_us = 0
                self._send_ack(seq, "goto_started",
                               {"position": list(request.position_enu),
                                "frame": request.frame,
                                "yaw": request.yaw_enu_rad,
                                "hold_s": request.hold_s})
            elif kind == "arm":
                self.safety.require_arm()
                self.last_command_seq = seq
                self._vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                self._send_ack(seq, "arm_requested")
            elif kind == "disarm":
                self.last_command_seq = seq
                if self.sample.landed:
                    # On a moving deck PX4's world-frame land detector can
                    # remain airborne even though the physical roof-contact
                    # latch has authoritatively ended the SITL episode.  The
                    # callback's first forced command can be overwritten in
                    # the uORB queue by this immediate learner command, so the
                    # learner command must carry the same force token too.
                    force = (21196.0 if cfg.target == "sitl"
                             and self.pad_contact_latched else 0.0)
                    self._vehicle_command(
                        VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                        0.0, force)
                    self._send_ack(seq, "disarm_requested")
                else:
                    # PX4 refuses to disarm in flight, and it is right to. End
                    # the episode with a landing so the vehicle is not left to
                    # the offboard-loss failsafe.
                    self.goto_target_enu = None
                    self._vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                    self._send_ack(seq, "landing_requested")
            elif kind == "enable_offboard":
                self.safety.require_offboard()
                self.last_command_seq = seq
                self.offboard_enabled = True
                self._send_ack(seq, "offboard_enabled")
            elif kind == "disable_offboard":
                self.last_command_seq = seq
                self.offboard_enabled = False
                self.offboard_requested = False
                self.last_action_ns = 0
                self.last_action_px4_time_us = 0
                self.velocity_position_target_enu = None
                self.velocity_position_time_us = 0
                self._send_ack(seq, "offboard_disabled")
            elif kind in {"hello", "state"}:
                if kind == "hello":
                    # One localhost controller is permitted. A hello opens a
                    # session whose command sequence starts again at one.
                    self.last_command_seq = -1
                self._send_state(seq)
            else:
                raise ProtocolError(f"unsupported command: {kind}")

        def _control_tick(self) -> None:
            stamp = now_ns()
            self._integrate_battery()
            if self.goto_target_enu is not None and stamp >= self.goto_deadline_ns:
                self.goto_target_enu = None
                # The climb only ever expires when handover never came, because
                # the first policy action clears the target. Simply going quiet
                # here leaves an armed vehicle in the air with nothing
                # commanding it, which is the fall the learner then reports as a
                # failed reset. Put it down under PX4's own controller instead
                # and let the reset retry from a vehicle that is on the deck.
                self.offboard_requested = False
                self.prestream = 0
                if self.sample.armed:
                    self._vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                    self.get_logger().warning(
                        "goto hold expired without handover; commanding a landing")
                else:
                    self.get_logger().warning(
                        "goto hold expired; releasing the setpoint stream")
            if self.last_action_ns:
                age_s = action_age_seconds(
                    cfg.target, stamp, self.last_action_ns,
                    int(self.sample.px4_time_us), self.last_action_px4_time_us)
                timeout_s = (cfg.sitl_action_timeout_s
                             if cfg.target == "sitl" else cfg.action_timeout_s)
                if age_s > timeout_s:
                    keep_sitl_hover = bool(
                        cfg.target == "sitl" and cfg.start_airborne
                        and self.sample.armed and self.px4_world_position is not None)
                    if keep_sitl_hover:
                        # A rendered frame or learner update can exceed the
                        # action deadline between episodes. Convert the last
                        # command to an immediate world-position hover without
                        # dropping one OFFBOARD heartbeat; stale velocity must
                        # not continue, but a 12 ms signal gap must not trigger
                        # AUTO.LAND either.
                        self.goto_target_enu = self.px4_world_position.copy()
                        self.goto_pad_relative = False
                        self.goto_yaw_enu = yaw_from_quat_wxyz(
                            self.sample.quaternion_enu_flu_wxyz)
                        self.goto_deadline_ns = stamp + int(120.0e9)
                        self.get_logger().warning(
                            f"action deadman expired after {age_s:.3f}s; "
                            "holding current SITL position between episodes")
                    else:
                        if self.prestream:
                            self.get_logger().warning(
                                f"action deadman expired after {age_s:.3f}s "
                                f"({cfg.target} limit {timeout_s:.3f}s); yielding "
                                "to PX4 offboard-loss failsafe")
                        self.prestream = 0
                        self.offboard_requested = False
                        self.last_mode_request_tick = -self.mode_request_period
                    self.last_action_ns = 0
                    self.last_action_px4_time_us = 0
                    self.velocity_position_target_enu = None
                    self.velocity_position_time_us = 0
            if self.last_action_ns:
                if self.command_interface == "velocity_yaw_rate":
                    self._publish_velocity_setpoint()
                else:
                    self._publish_attitude_setpoint()
            elif self.goto_target_enu is not None:
                self._publish_position_setpoint()
            else:
                return

            self.prestream += 1
            if (self.offboard_enabled and self.sample.armed
                    and offboard_recovery_allowed(
                        self.px4_failsafe, self.px4_failsafe_detail)
                    and self.prestream >= cfg.offboard_prestream_count):
                self.offboard_requested = self.sample.nav_state == self.offboard_nav_state
                # PX4 only accepts the mode switch once it has seen a steady
                # setpoint stream, and rejects it outright in some pre-arm
                # states, so a single latched request is not enough.
                if (not self.offboard_requested
                        and self.prestream - self.last_mode_request_tick >= self.mode_request_period):
                    self._vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                    self.last_mode_request_tick = self.prestream

        def _publish_offboard_mode(self, **active: bool) -> None:
            mode = OffboardControlMode()
            mode.timestamp = self._timestamp_us()
            for name in ("position", "velocity", "acceleration", "attitude", "body_rate",
                         "thrust_and_torque", "direct_actuator"):
                _set_if_present(mode, name, bool(active.get(name, False)))
            self.offboard_pub.publish(mode)

        def _publish_position_setpoint(self) -> None:
            # PX4 flies the episode entry pose itself: no teleport, so the
            # estimator never sees a jump it cannot explain.
            # Chasing a moving deck with a position-only setpoint costs a
            # standing lag of v_deck / MPC_XY_P, which is most of the entry
            # tolerance at full platform speed. Feed the deck's own velocity
            # forward so the position loop only has to close the residual.
            feedforward = (
                entry_feedforward_velocity(
                    self.deck_truth_velocity_enu
                    if self.deck_truth_position_enu is not None
                    else self.deck_velocity_enu)
                if self.goto_pad_relative else None)
            self._publish_offboard_mode(position=True,
                                        velocity=feedforward is not None)
            # Back out of the world frame: this is published as a *local*
            # setpoint, so a world-frame target would be off by the distance
            # between the two origins -- which is what sent the vehicle
            # sideways the moment it lifted off the pad.
            target_ned = enu_to_ned(self._goto_world_target() - self.world_from_px4)
            sp = TrajectorySetpoint()
            sp.timestamp = self._timestamp_us()
            sp.position = [float(x) for x in target_ned]
            sp.velocity = ([float("nan")] * 3 if feedforward is None else
                           [float(x) for x in enu_to_ned(feedforward)])
            sp.acceleration = [float("nan")] * 3
            sp.yaw = float(yaw_enu_to_ned(self.goto_yaw_enu))
            sp.yawspeed = 0.0
            self.trajectory_pub.publish(sp)

        def _publish_attitude_setpoint(self) -> None:
            self._publish_offboard_mode(attitude=True)
            a0, a_roll, a_pitch, a_yaw_rate = self.action
            yaw_enu = yaw_from_quat_wxyz(self.sample.quaternion_enu_flu_wxyz)
            q_enu = euler_zyx_to_quat_wxyz(
                cfg.max_roll_pitch_rad * a_roll,
                cfg.max_roll_pitch_rad * a_pitch,
                yaw_enu,
            )
            q_ned = quat_enu_flu_to_ned_frd(q_enu)
            thrust = float(np.clip(cfg.hover_thrust * (1.0 + cfg.collective_span * a0), 0.05, 0.90))
            sp = VehicleAttitudeSetpoint()
            sp.timestamp = self._timestamp_us()
            sp.q_d = [float(x) for x in q_ned]
            sp.thrust_body = [0.0, 0.0, -thrust]
            _set_if_present(sp, "yaw_sp_move_rate", float(-cfg.max_yaw_rate_rad_s * a_yaw_rate))
            _set_if_present(sp, "reset_integral", False)
            self.attitude_pub.publish(sp)

        def _publish_velocity_setpoint(self) -> None:
            """Track velocity through a continuous, position-backed setpoint."""
            self._publish_offboard_mode(position=True, velocity=True)
            vx, vy, vz, yaw_rate = self.velocity_action
            yaw = yaw_from_quat_wxyz(self.sample.quaternion_enu_flu_wxyz)
            c, s = math.cos(yaw), math.sin(yaw)
            velocity_enu = np.array([c * vx - s * vy, s * vx + c * vy, vz])
            now_us = int(self.sample.px4_time_us)
            if self.velocity_position_target_enu is None:
                here = (self.px4_world_position if self.px4_world_position is not None
                        else np.asarray(self.sample.position_world_enu, dtype=float))
                self.velocity_position_target_enu = np.asarray(here, dtype=float).copy()
                self.velocity_position_time_us = now_us
            dt = 0.0
            if now_us > 0 and self.velocity_position_time_us > 0:
                elapsed = (now_us - self.velocity_position_time_us) * 1e-6
                if elapsed > 0.0:
                    # DDS may deliver a batch after a slow rendered frame. Do
                    # not let that backlog become a multi-metre setpoint jump.
                    dt = min(elapsed, 2.0 / cfg.control_hz)
            self.velocity_position_time_us = now_us
            deck_z = float(self.deck_position_enu[2])
            self.velocity_position_target_enu = advance_velocity_position_target(
                self.velocity_position_target_enu, velocity_enu, dt,
                floor_z_m=deck_z,
                ceiling_z_m=deck_z + cfg.max_altitude_m,
                world_radius_m=cfg.world_radius_m)
            target_ned = enu_to_ned(
                self.velocity_position_target_enu - self.world_from_px4)
            self.sample.extra["velocity_position_target_world_enu_m"] = [
                float(value) for value in self.velocity_position_target_enu]
            sp = TrajectorySetpoint()
            sp.timestamp = self._timestamp_us()
            sp.position = [float(value) for value in target_ned]
            sp.velocity = [float(value) for value in enu_to_ned(velocity_enu)]
            # A tilt cannot be asked for directly through a velocity setpoint.
            # What PX4 does accept is an acceleration feed-forward, and its
            # position controller turns a horizontal acceleration into exactly
            # the attitude that produces it -- so ``a = g tan(theta)`` along the
            # vehicle's heading IS the tilt request. Bounded twice: by the
            # learner's own envelope and by the protocol limit, so a malformed
            # command cannot ask the airframe to invert itself.
            tilt = float(np.clip(self.velocity_tilt_rad,
                                 -PLANAR_TILT_LIMIT_RAD, PLANAR_TILT_LIMIT_RAD))
            if abs(tilt) > 1e-9:
                forward = GRAVITY_M_S2 * math.tan(tilt)
                acceleration_enu = np.array([c * forward, s * forward, 0.0])
                sp.acceleration = [float(value)
                                   for value in enu_to_ned(acceleration_enu)]
                self.sample.extra["commanded_longitudinal_tilt_rad"] = tilt
            else:
                sp.acceleration = [float("nan")] * 3
                self.sample.extra["commanded_longitudinal_tilt_rad"] = 0.0
            # Heading. The planar envelope holds one absolute heading for the
            # whole episode; without it PX4 free-runs the yaw under a zero rate,
            # which drifts and breaks the "always aligned with the deck"
            # constraint the whole comparison rests on.
            if self.velocity_yaw_hold_rad is None:
                sp.yaw = float("nan")
                sp.yawspeed = float(-yaw_rate)
            else:
                sp.yaw = float(yaw_enu_to_ned(self.velocity_yaw_hold_rad))
                sp.yawspeed = 0.0
                self.sample.extra["commanded_yaw_hold_enu_rad"] = float(
                    self.velocity_yaw_hold_rad)
            self.trajectory_pub.publish(sp)

        def _vehicle_command(self, command: int, param1: float = 0.0, param2: float = 0.0) -> None:
            msg = VehicleCommand()
            msg.timestamp = self._timestamp_us()
            msg.param1 = float(param1)
            msg.param2 = float(param2)
            msg.command = int(command)
            msg.target_system = cfg.target_system
            msg.target_component = cfg.target_component
            msg.source_system = cfg.source_system
            msg.source_component = cfg.source_component
            msg.from_external = True
            self.command_pub.publish(msg)

        def _on_odometry(self, msg) -> None:
            try:
                # Into the world frame the deck is broadcast in. Everything
                # downstream differences the two, so this has to happen before
                # any of it -- see _update_world_origin.
                position = ned_to_enu(msg.position) + self.world_from_px4
                # Kept before the canyon error is added: the entry setpoint is
                # published into PX4's own frame, so it has to be built from
                # the pose PX4 actually holds, not from the degraded copy the
                # policy is handed.
                self.px4_world_position = position.copy()
                q_enu = quat_ned_frd_to_enu_flu(msg.q)
                velocity = np.asarray(msg.velocity, dtype=float)
                if velocity.shape != (3,) or not np.isfinite(velocity).all():
                    raise ValueError("invalid velocity")
            except ValueError:
                self.sample.estimator_valid = False
                return
            body_frame = getattr(msg, "VELOCITY_FRAME_BODY_FRD", 3)
            if getattr(msg, "velocity_frame", 0) == body_frame:
                velocity = quat_wxyz_to_matrix(msg.q) @ velocity
            velocity = ned_to_enu(velocity)
            omega = frd_to_flu(msg.angular_velocity)
            stamp = now_ns()
            accel = np.zeros(3)
            if self.last_velocity is not None and stamp > self.last_velocity_ns:
                dt = (stamp - self.last_velocity_ns) * 1e-9
                if 1e-4 < dt < 0.2:
                    accel = (velocity - self.last_velocity) / dt
            self.last_velocity = velocity
            self.last_velocity_ns = stamp
            self.sample.timestamp_ns = stamp
            raw_px4_time_us = int(msg.timestamp)
            px4_time_us, clock_rebased = self.px4_clock.update(raw_px4_time_us)
            self.sample.px4_time_us = px4_time_us
            self.sample.extra["px4_clock_discontinuities"] = int(
                self.px4_clock.discontinuities)
            if clock_rebased:
                self.get_logger().warning(
                    "XRCE-DDS rebased the PX4 timestamp "
                    f"(raw delta {self.px4_clock.last_delta_us / 1e6:.3f} s); "
                    "continuing on the monotonic simulated-time axis")
            pad_pose_fresh = (stamp - self.pad_pose_time_ns) * 1e-9 <= cfg.state_timeout_s
            deck_fresh = self._deck_is_fresh(stamp)
            # Hardware always flies on the pad-relative pose. In SITL it is a
            # choice: with the camera enabled the policy lands on the marker it
            # can actually see, and falls back to the PX4 estimate the moment
            # the pad leaves the frame.
            # The detector reports a miss on the same frame it loses the tags,
            # a whole state_timeout_s before the pose it last solved goes
            # stale. Waiting for staleness serves that frozen solve as if it
            # were live, so the policy flies half a second of a pad-relative
            # position that stopped tracking the moment the pad left the frame.
            # Legacy ArUco path only. The primary keypoint benchmark runs with
            # ``vision.mode: keypoint_fiducial`` and ``pose_source_for_policy:
            # false``, so the simulator publishes neither marker quality nor a
            # detector-solved pose: ``marker_live`` stays False, no marker pose
            # ever reaches navigation, and field-of-view truth is the separate
            # geometric quantity computed from pad-relative truth.
            marker_live = self.sample.marker_quality > 0.0
            use_pad_pose = pad_pose_fresh and marker_live and (
                cfg.target == "hardware" or cfg.marker_pose_drives_policy)
            # Everything the policy sees is relative to the deck, because the
            # deck is the target and it moves. The camera already solves in the
            # pad frame; the PX4 fallback is the deck pose subtracted from the
            # world estimate. Both express the pad frame as ENU axes translated
            # to the deck origin -- deliberately not rotated with the deck, so
            # the axes stay gravity-aligned and no Coriolis term appears in the
            # relative velocity.
            # With upstream injection this is already EKF2's covariance-
            # weighted GNSS/inertial estimate.  The legacy downstream path is
            # retained for recorded fixtures and configurations that explicitly
            # disable HIL_GPS injection, but the error must never be added twice.
            estimated_position = position + (
                np.zeros(3) if self.gnss_injected_into_px4 else self.gnss_offset)
            estimated_velocity = velocity + (
                np.zeros(3) if self.gnss_injected_into_px4 else self.gnss_velocity_offset)
            fallback_position = estimated_position - self.deck_position_enu
            # Velocity always comes from the flight stack's estimator, whatever
            # is providing position: the marker solve is not differentiated
            # here, so there is no optical velocity to prefer.
            policy_velocity = estimated_velocity - self.deck_velocity_enu
            raw_policy_position = self._blend_position_source(
                use_pad_pose, self.pad_position_enu, fallback_position, stamp)
            self._track_pad_relative(
                raw_policy_position, policy_velocity,
                stamp_ns=int(msg.timestamp) * 1000, optical=use_pad_pose)
            # When the marker is absent, propagate its last reliable solve with
            # relative velocity and let GNSS correct it only at a rate justified
            # by the two receivers' covariance. This is the actual navigation
            # estimate, not merely an entry-setpoint smoother.
            policy_position = self.pad_track_position.copy()
            self._last_policy_position = policy_position.copy()
            self.sample.position_enu = tuple(float(x) for x in policy_position)
            self.sample.velocity_enu = tuple(float(x) for x in policy_velocity)
            self.sample.position_world_enu = tuple(float(x) for x in estimated_position)
            self.sample.velocity_world_enu = tuple(float(x) for x in estimated_velocity)
            self._set_truth(position, velocity)
            self.sample.quaternion_enu_flu_wxyz = tuple(float(x) for x in q_enu)
            self.sample.angular_velocity_flu = tuple(float(x) for x in omega)
            self.sample.acceleration_enu = tuple(float(x) for x in accel)
            self.sample.pad = self._pad_state(deck_fresh)
            self.sample.battery = self.battery.sample()
            self.sample.gnss = self._gnss_state(stamp)
            navigation_valid = bool(np.isfinite(position).all() and np.isfinite(q_enu).all())
            self.sample.estimator_valid = (
                navigation_valid
                and self._estimator_is_healthy(stamp)
                and (cfg.target == "sitl" or pad_pose_fresh)
                # A moving deck whose pose has gone quiet makes every
                # pad-relative number a guess, so say the state is invalid
                # instead of reporting a stale target as if it were live.
                and (cfg.pad_is_static or deck_fresh)
            )
            self.sample.extra["position_source"] = (
                "vision_imu_fused" if use_pad_pose else "imu_dr_gnss_bounded")
            self.sample.extra["position_fusion_residual_m"] = float(
                np.linalg.norm(raw_policy_position - policy_position))
            # How much of the last source change has not yet been faded out. A
            # consumer reading position_source alone would think the handover
            # was instantaneous; it is not, and this says by how much.
            self.sample.extra["source_handover_offset_m"] = float(
                np.linalg.norm(self._decayed_offset(stamp)))
            self.sample.extra["marker_live"] = bool(marker_live)
            self.sample.extra["control_source"] = (
                "action" if self.last_action_ns
                else "goto" if self.goto_target_enu is not None
                else "idle"
            )
            self.sample.extra["offboard_active"] = bool(self.offboard_requested)
            self.sample.extra["control_mapping"] = {
                "interface": self.command_interface,
                "hover_thrust": cfg.hover_thrust,
                "collective_span": cfg.collective_span,
                "max_roll_pitch_rad": cfg.max_roll_pitch_rad,
                "max_yaw_rate_rad_s": cfg.max_yaw_rate_rad_s,
            }
            if self.pending_state_ack >= 0:
                seq = self.pending_state_ack
                peer = self.pending_state_peer
                self.pending_state_ack = -1
                self.pending_state_peer = None
                self._send_state(seq, peer=peer)

        def _on_status(self, msg) -> None:
            self.sample.nav_state = int(msg.nav_state)
            armed_value = getattr(msg, "ARMING_STATE_ARMED", 2)
            self.sample.armed = int(msg.arming_state) == int(armed_value)
            self.sample.extra["pre_flight_checks_pass"] = bool(
                getattr(msg, "pre_flight_checks_pass", False))
            self.px4_failsafe = bool(getattr(msg, "failsafe", False))
            self.sample.extra["px4_failsafe"] = self.px4_failsafe
            self.sample.extra["px4_failsafe_detail"] = dict(
                self.px4_failsafe_detail)
            self._report_failsafe()
            self._publish_flight_state()

        def _on_failsafe_flags(self, msg) -> None:
            self.px4_failsafe_detail = failsafe_detail(msg, target=cfg.target)
            self.sample.extra["px4_failsafe_detail"] = dict(
                self.px4_failsafe_detail)
            self._report_failsafe()

        def _clock_skew_us(self) -> int | None:
            """Wall-clock minus PX4 time, the quantity uxrce_dds_client tracks.

            The gateway stamps every setpoint from the wall clock, and PX4's
            uxrce_dds_client rewrites that into simulated time with the
            Timesync filter's offset before OffboardChecks compares it against
            hrt_absolute_time(). While the offset is right this difference is a
            constant. It is the drift in it, not a gap in the 20 Hz stream,
            that expires COM_OF_LOSS_T, so record it whenever PX4 says the
            offboard signal is lost.
            """
            px4_time_us = int(self.sample.px4_time_us)
            if px4_time_us <= 0:
                return None
            return self._timestamp_us() - px4_time_us

        def _report_failsafe(self) -> None:
            if not self.px4_failsafe:
                self.reported_failsafe_signature = None
                # Healthy flight is the only honest baseline for the drift
                # reported below.
                skew = self._clock_skew_us()
                if skew is not None:
                    self.clock_skew_baseline_us = skew
                return
            reasons = tuple(self.px4_failsafe_detail.get("reasons", ()))
            if reasons == self.reported_failsafe_signature:
                return
            self.reported_failsafe_signature = reasons
            detail = ", ".join(reasons) if reasons else "unknown"
            recovery = ("recoverable SITL infrastructure fault"
                        if self.px4_failsafe_detail.get(
                            "recoverable_infrastructure", False)
                        else "tip-over; the learner scores it as a crash"
                        if self.px4_failsafe_detail.get(
                            "attitude_failure", False)
                        else "non-recoverable vehicle/task fault")
            drift = ""
            skew = self._clock_skew_us()
            if ("offboard_control_signal_lost" in reasons
                    and skew is not None
                    and self.clock_skew_baseline_us is not None):
                drift = (f"; wall-vs-PX4 clock drift "
                         f"{(skew - self.clock_skew_baseline_us) * 1e-6:+.1f} s "
                         f"against COM_OF_LOSS_T")
            self.get_logger().warning(
                f"PX4 failsafe active: {detail} ({recovery}){drift}")

        def _on_land(self, msg) -> None:
            self.px4_landed = bool(msg.landed)
            self.sample.extra["px4_landed"] = self.px4_landed
            self.land_detected_ns = now_ns()

        def _on_pad_contact(self, msg) -> None:
            was_latched = self.pad_contact_latched
            self.pad_contact_raw = bool(msg.data)
            self.pad_contact_ns = now_ns()
            # Sticky until reset: contact can bounce for one frame exactly when
            # motors stop, but touchdown has already physically occurred.
            truth = self.sample.truth_position_enu
            clearance = bool(truth is not None and float(truth[2]) >= 0.5)
            px4_landed = effective_px4_landed(
                self.px4_landed, self.sample.armed, truth)
            self.pad_contact_latched, self.pad_contact_armed_clear = (
                advance_pad_contact_latch(
                    self.pad_contact_latched, self.pad_contact_armed_clear,
                    self.sample.armed, self.pad_contact_raw, px4_landed,
                    clearance))
            if self.pad_contact_latched and not was_latched:
                # Do not wait for the next learner sample to stop the motors.
                # PX4's land detector observes world motion and can reject a
                # normal disarm while the lorry is driving, so physical deck
                # contact is the SITL-only authority for a forced disarm.
                self.sample.landed = True
                self.sample.extra["pad_contact"] = True
                self.sample.extra["touchdown_source"] = "pad_contact"
                if cfg.target == "sitl":
                    # Keep an OFFBOARD position heartbeat until PX4 confirms
                    # the forced disarm. On a moving deck the land detector can
                    # reject the first request; stopping the heartbeat first
                    # guarantees an avoidable OFFBOARD-loss failsafe.
                    if self.px4_world_position is not None:
                        self.goto_target_enu = self.px4_world_position.copy()
                        self.goto_pad_relative = False
                        self.goto_yaw_enu = yaw_from_quat_wxyz(
                            self.sample.quaternion_enu_flu_wxyz)
                        self.goto_deadline_ns = now_ns() + int(30.0e9)
                        self.last_action_ns = 0
                        self.last_action_px4_time_us = 0
                        self.velocity_position_target_enu = None
                        self.velocity_position_time_us = 0
                    else:
                        self.velocity_action = (0.0, 0.0, 0.0, 0.0)
                        self.velocity_tilt_rad = 0.0
                    self._vehicle_command(
                        VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                        0.0, 21196.0)
                    relative = self.sample.truth_position_enu
                    where = ("unknown" if relative is None else
                             ",".join(f"{float(value):.2f}" for value in relative))
                    self.get_logger().info(
                        "physical pad touchdown: OFFBOARD hold retained until "
                        "SITL force-disarm confirmation "
                        f"(truth pad xyz={where})")

        def _on_thrust_setpoint(self, msg) -> None:
            # PX4's own normalised body thrust. While its position controller
            # holds a hover this is the hover thrust the gateway mapping must
            # be centred on; see tools/calibrate_hover_thrust.py.
            self.sample.extra["px4_thrust"] = float(-msg.xyz[2])
            self._publish_flight_state()

        def _on_command_ack(self, msg) -> None:
            # A silently rejected arm used to look exactly like a vehicle that
            # simply refused to climb, so say so out loud.
            accepted = int(getattr(VehicleCommandAck, "VEHICLE_CMD_RESULT_ACCEPTED", 0))
            in_progress = int(getattr(VehicleCommandAck, "VEHICLE_CMD_RESULT_IN_PROGRESS", 5))
            result = int(msg.result)
            self.sample.extra["last_command"] = [int(msg.command), result]
            mode_command = int(getattr(
                VehicleCommand, "VEHICLE_CMD_DO_SET_MODE", 176))
            if int(msg.command) == mode_command:
                if result in (accepted, in_progress):
                    self.offboard_mode_rejections = 0
                else:
                    self.offboard_mode_rejections += 1
                self.sample.extra["offboard_mode_rejections"] = int(
                    self.offboard_mode_rejections)
            if result not in (accepted, in_progress):
                self.get_logger().warning(
                    f"PX4 rejected command {int(msg.command)} with result {result} "
                    f"(reason {int(msg.result_param1)})"
                )

        def _on_local_position(self, msg) -> None:
            # PX4 is the only component that knows whether its EKF has actually
            # converged; a finite odometry sample on its own proves nothing.
            # heading_good_for_control is deliberately not part of the gate: it
            # is normally false on a stationary disarmed vehicle, which is
            # exactly the state every episode starts from.
            self.estimator_healthy = bool(
                msg.xy_valid and msg.z_valid and msg.v_xy_valid and msg.v_z_valid
            )
            self.estimator_health_ns = now_ns()
            self.sample.extra["heading_good_for_control"] = bool(
                getattr(msg, "heading_good_for_control", False)
            )
            dead_reckoning = bool(getattr(msg, "dead_reckoning", False))
            self.sample.extra["navigation_mode"] = (
                "inertial_dead_reckoning" if dead_reckoning else "gnss_aided")
            self.sample.extra["local_position_accuracy"] = {
                "eph_m": _finite_or(getattr(msg, "eph", 0.0), 99.9),
                "epv_m": _finite_or(getattr(msg, "epv", 0.0), 99.9),
                "evh_m_s": _finite_or(getattr(msg, "evh", 0.0), 99.9),
                "evv_m_s": _finite_or(getattr(msg, "evv", 0.0), 99.9),
            }
            self._update_world_origin(msg)

        def _on_sensor_gps(self, msg) -> None:
            self.sample.extra["sensor_gps"] = {
                "fix_type": int(getattr(msg, "fix_type", 0)),
                "satellites_used": int(getattr(msg, "satellites_used", 0)),
                "eph_m": _finite_or(getattr(msg, "eph", 0.0), 99.9),
                "epv_m": _finite_or(getattr(msg, "epv", 0.0), 99.9),
                "speed_accuracy_m_s": _finite_or(
                    getattr(msg, "s_variance_m_s", 0.0), 99.9),
            }

        def _on_estimator_gps(self, msg) -> None:
            failures = [name.removeprefix("check_fail_") for name in (
                "check_fail_gps_fix", "check_fail_min_sat_count", "check_fail_max_pdop",
                "check_fail_max_horz_err", "check_fail_max_vert_err",
                "check_fail_max_spd_err", "check_fail_max_horz_drift",
                "check_fail_max_vert_drift", "check_fail_max_horz_spd_err",
                "check_fail_max_vert_spd_err") if bool(getattr(msg, name, False))]
            self.sample.extra["ekf_gps_checks"] = {
                "passed": bool(getattr(msg, "checks_passed", False)),
                "failed": failures,
                "horizontal_drift_m_s": _finite_or(getattr(
                    msg, "position_drift_rate_horizontal_m_s", 0.0), 99.9),
                "vertical_drift_m_s": _finite_or(getattr(
                    msg, "position_drift_rate_vertical_m_s", 0.0), 99.9),
            }

        def _on_estimator_flags(self, msg) -> None:
            inertial_dr = bool(getattr(msg, "cs_inertial_dead_reckoning", False))
            gps_fused = bool(getattr(msg, "cs_gps", False))
            self.sample.extra["ekf_fusion"] = {
                "gps": gps_fused,
                "inertial_dead_reckoning": inertial_dr,
                "horizontal_position_rejected": bool(
                    getattr(msg, "reject_hor_pos", False)),
                "horizontal_velocity_rejected": bool(
                    getattr(msg, "reject_hor_vel", False)),
                "accelerometer_bias_fault": bool(
                    getattr(msg, "fs_bad_acc_bias", False)),
            }
            # EstimatorStatusFlags is more explicit than VehicleLocalPosition
            # on the transition edge; let it refine the telemetry label.
            if inertial_dr:
                self.sample.extra["navigation_mode"] = "inertial_dead_reckoning"
            elif gps_fused:
                self.sample.extra["navigation_mode"] = "gnss_aided"

        def _update_world_origin(self, msg) -> None:
            """Where PX4's local frame sits in the simulator's world frame.

            PX4 pins its local frame wherever the EKF initialised, which is the
            spawn point -- on the roof of a lorry parked some way down the
            block. The deck broadcasts its pose in the world frame the city is
            laid out in, whose origin is the map datum. Those are two different
            origins, and until this was worked out the gateway simply mixed
            them: a goto was published as a *local* setpoint carrying a *world*
            target, so the vehicle darted off by the distance between the two
            the moment it left the pad, and the pad-relative state the policy
            flew on carried the same constant error.

            PX4 publishes the geodetic coordinates of its own origin, and the
            map datum is configuration, so the offset is recoverable without
            any privileged knowledge -- which is also how a real vehicle does
            it: a cooperative lorry broadcasts a geodetic position and the
            drone brings it into its own local frame.
            """
            if not bool(getattr(msg, "xy_global", False)):
                return
            latitude = float(getattr(msg, "ref_lat", float("nan")))
            longitude = float(getattr(msg, "ref_lon", float("nan")))
            if not (math.isfinite(latitude) and math.isfinite(longitude)):
                return
            if not (cfg.map_latitude_deg or cfg.map_longitude_deg):
                # No datum configured: the world frame is PX4's own, which is
                # what a single-vehicle setup with a pad at the origin had.
                self.world_from_px4 = np.zeros(3)
                self.world_origin_known = True
                return
            altitude = float(getattr(msg, "ref_alt", float("nan")))
            if not math.isfinite(altitude):
                altitude = cfg.map_altitude_m
            offset = geodetic_to_enu(latitude, longitude,
                                     cfg.map_latitude_deg, cfg.map_longitude_deg,
                                     altitude, cfg.map_altitude_m)
            if not self.world_origin_known:
                self.get_logger().info(
                    f"PX4 local frame is at world ENU ({offset[0]:.1f}, "
                    f"{offset[1]:.1f}, {offset[2]:.2f}) m; deck poses will be "
                    "brought into it.")
            self.world_from_px4 = offset
            self.world_origin_known = True

        def _estimator_is_healthy(self, stamp: int) -> bool:
            fresh = (stamp - self.estimator_health_ns) * 1e-9 <= cfg.state_timeout_s
            return bool(self.estimator_healthy and fresh)

        def _on_wind(self, msg) -> None:
            self.sample.wind_enu = (float(msg.vector.x), float(msg.vector.y), float(msg.vector.z))
            self.sample.extra["wind_source"] = "uav_anemometer"

        def _on_aero_force(self, msg) -> None:
            self.sample.aero_force_enu = (float(msg.vector.x), float(msg.vector.y), float(msg.vector.z))
            self.sample.extra["aero_force_source"] = "simulator_truth"

        def _on_marker_quality(self, msg) -> None:
            self.sample.marker_quality = float(np.clip(msg.data, 0.0, 1.0))

        def _blend_position_source(self, use_pad_pose: bool,
                                   optical, fallback: np.ndarray,
                                   stamp: int) -> np.ndarray:
            """Hand over between the camera and the PX4 estimate continuously.

            The two sources do not agree: the marker solve is a direct optical
            measurement of the pad-relative pose, the fallback is a difference
            of two GNSS-derived positions, and in the canyon they can be metres
            apart. Switching between them is therefore a step in the state the
            policy flies on -- and it lands exactly when the pad leaves the
            frame, which is when the controller can least afford to be kicked.
            The policy would read that step as the deck jumping sideways and
            haul the vehicle after it.

            So the difference at the moment of the handover is carried and
            faded out over ``vision_handover_tau_s``. The estimate stays
            continuous, converges to whichever source is now authoritative, and
            no step is ever presented as motion. Errors are not invented here:
            the offset only ever shrinks, and it is reported so a consumer can
            see when the state is still mid-handover.
            """
            source = np.asarray(optical if use_pad_pose else fallback, dtype=float)
            tau = float(cfg.vision_handover_tau_s)
            previous = self._last_source_was_optical
            self._last_source_was_optical = bool(use_pad_pose)
            if tau <= 0.0 or previous is None:
                self._source_offset = np.zeros(3)
                self._source_offset_ns = stamp
                self._last_policy_position = source
                return source
            if previous != bool(use_pad_pose):
                # Re-anchor on the state actually being flown, so the first
                # sample after the handover equals the last one before it.
                self._source_offset = self._decayed_offset(stamp) + (
                    np.asarray(self._last_policy_position, dtype=float) - source)
                self._source_offset_ns = stamp
            blended = source + self._decayed_offset(stamp)
            self._last_policy_position = blended
            return blended

        def _decayed_offset(self, stamp: int) -> np.ndarray:
            """What is left of the handover step, decayed toward zero."""
            offset = np.asarray(self._source_offset, dtype=float)
            if not offset.any():
                return np.zeros(3)
            tau = float(cfg.vision_handover_tau_s)
            age = max(0.0, (stamp - self._source_offset_ns) * 1e-9)
            return offset * math.exp(-age / max(tau, 1e-6))

        def _publish_flight_state(self) -> None:
            """Tell Isaac when the autopilot takes the vehicle over.

            Arming alone is not control. PX4 needs time to accept OFFBOARD
            after the setpoint prestream; dropping Isaac's physical hover hold
            at ARM makes the unpowered interval end on the deck. ``controlled``
            releases that hold only after PX4 reports OFFBOARD.
            """
            armed = bool(self.sample.armed)
            # Handover is the moment the policy takes the vehicle: the entry
            # climb is over and the episode has begun. That is when the deck is
            # allowed to pull away, so the climb happens over a lorry standing
            # still and only the landing has to chase one.
            handover = bool(self.last_action_ns)
            commanded = bool(self.goto_target_enu is not None or handover)
            controlled = bool(
                armed and commanded and self.sample.nav_state == self.offboard_nav_state)
            px4_thrust = float(np.clip(
                self.sample.extra.get("px4_thrust", 0.0), 0.0, 1.0))
            # Two-percent buckets are much finer than the handoff needs and
            # avoid turning this state latch into a second high-rate telemetry
            # stream at PX4's thrust-setpoint publication rate.
            thrust_bucket = int(round(50.0 * px4_thrust))
            state = (armed, handover, controlled, thrust_bucket)
            if state == self.published_flight_state:
                return
            self.published_flight_state = state
            message = String()
            message.data = json.dumps({"v": cfg.protocol_version,
                                       "armed": armed, "handover": handover,
                                       "controlled": controlled,
                                       "px4_thrust": px4_thrust})
            self.flight_pub.publish(message)

        def _on_pad_pose(self, msg) -> None:
            position = np.array(
                [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float
            )
            if np.isfinite(position).all():
                stamp = now_ns()
                # Planar PnP can occasionally choose a mirrored/remote branch
                # with a finite, low-reprojection solution. During continuous
                # tracking the vehicle cannot move metres between camera
                # frames, so reject that innovation instead of re-anchoring DR
                # to a catastrophic but numerically valid pose. After a genuine
                # outage the freshness window expires and reacquisition remains
                # possible from any position.
                previous_fresh = (
                    self.pad_position_enu is not None
                    and (stamp - self.pad_pose_time_ns) * 1e-9 <= cfg.state_timeout_s)
                if previous_fresh:
                    innovation = float(np.linalg.norm(position - self.pad_position_enu))
                    innovation_limit = cfg.marker_pose_max_step_m
                elif self.pad_track_ns:
                    innovation = float(np.linalg.norm(position - self.pad_track_position))
                    innovation_limit = cfg.marker_pose_reacquire_error_m
                else:
                    innovation = 0.0
                    innovation_limit = float("inf")
                # A 90-degree downward camera cannot see a target whose lateral
                # displacement is many times its solved height. This image-
                # geometry gate also protects first acquisition, where there is
                # no previous pose for an innovation check.
                z = float(position[2])
                radial = float(np.linalg.norm(position[:2]))
                geometrically_possible = (
                    -0.25 <= z <= cfg.max_altitude_m
                    and radial <= 1.8 * max(z, 0.5) + 0.5)
                # A planar board can also return a low-reprojection pose on
                # the wrong branch by tilting the camera and translating it
                # sideways.  Attitude comes independently from PX4's IMU, so
                # this is a real sensor-consistency check rather than access to
                # simulator truth.  Compare quaternions without assuming a
                # sign, since q and -q represent the same rotation.
                o = msg.pose.orientation
                marker_q = np.array([o.w, o.x, o.y, o.z], dtype=float)
                imu_q = np.asarray(self.sample.quaternion_enu_flu_wxyz, dtype=float)
                attitude_innovation_deg = 180.0
                if (np.isfinite(marker_q).all() and np.isfinite(imu_q).all()
                        and np.linalg.norm(marker_q) > 1e-9
                        and np.linalg.norm(imu_q) > 1e-9):
                    marker_q /= np.linalg.norm(marker_q)
                    imu_q /= np.linalg.norm(imu_q)
                    attitude_innovation_deg = math.degrees(2.0 * math.acos(
                        float(np.clip(abs(np.dot(marker_q, imu_q)), 0.0, 1.0))))
                attitude_consistent = attitude_innovation_deg <= 12.0
                if (not geometrically_possible
                        or not attitude_consistent
                        or innovation > innovation_limit):
                    self.marker_pose_rejections += 1
                    # Detection confidence and PnP pose validity are different
                    # observables. A planar board can be clearly visible while
                    # solvePnP chooses a mirrored/remote branch. Keep the visual
                    # quality for the actor, entry-FOV gate and ontology, while
                    # rejecting only this pose from position fusion.
                    self.sample.extra["marker_pose_valid"] = False
                    self.sample.extra["marker_pose_rejected"] = True
                    self.sample.extra["marker_pose_innovation_m"] = innovation
                    self.sample.extra["marker_attitude_innovation_deg"] = (
                        attitude_innovation_deg)
                    return
                self.pad_position_enu = position
                self.pad_pose_time_ns = stamp
                self.sample.extra["marker_pose_valid"] = True
                self.sample.extra["marker_pose_rejected"] = False
                self.sample.extra["marker_pose_rejections"] = self.marker_pose_rejections
                self.sample.extra["marker_attitude_innovation_deg"] = (
                    attitude_innovation_deg)

        def _on_deck_odom(self, msg) -> None:
            """What the lorry broadcasts about itself over its V2V link.

            Its own receiver is in the same canyon, so this pose carries the
            lorry's GNSS error and its covariance says how big the lorry thinks
            that error is. It is the only deck pose the policy may see.
            """
            p = msg.pose.pose.position
            v = msg.twist.twist.linear
            w = msg.twist.twist.angular
            position = np.array([p.x, p.y, p.z], dtype=float)
            velocity = np.array([v.x, v.y, v.z], dtype=float)
            if not (np.isfinite(position).all() and np.isfinite(velocity).all()):
                return
            o = msg.pose.pose.orientation
            self.deck_position_enu = position
            self.deck_velocity_enu = velocity
            self.deck_yaw = float(yaw_from_quat_wxyz((o.w, o.x, o.y, o.z)))
            self.deck_yaw_rate = float(w.z)
            self._track_deck(position, velocity)
            covariance = getattr(msg.pose, "covariance", None)
            if covariance is not None and len(covariance) >= 8:
                variance = float(covariance[0]) + float(covariance[7])
                self.deck_sigma_xy_m = math.sqrt(variance) if variance > 0.0 else 0.0
            self.deck_time_ns = now_ns()

        def _track_pad_relative(self, position: np.ndarray,
                                velocity: np.ndarray, *, stamp_ns: int,
                                optical: bool = False) -> None:
            """Fuse marker, inertial propagation and relative GNSS position.

            The measured pad-relative position is the difference of two
            independently map-mitigated fixes.  It is metre-scale in the
            nominal canyon, but can still become a large, correlated outlier in
            deeply shadowed regions. Aim a position controller straight at it
            and the vehicle physically chases receiver error.

            A live marker solve is authoritative and also re-anchors the DR
            state. Without it, relative velocity predicts the pose and the
            difference of the two GNSS positions only bounds drift. Its update
            time constant grows with combined receiver variance, so a 20 m
            canyon fix cannot pull the controller tens of metres during a short
            marker outage while a good fix still recentres it promptly.
            """
            # Velocity is metres per *simulated* second. Isaac/PX4 run slower
            # than wall time under lockstep rendering, so integrating with
            # now_ns() over-propagates DR by the real-time-factor inverse.
            stamp = int(stamp_ns)
            if self.pad_track_ns == 0:
                self.pad_track_position = np.asarray(position, dtype=float).copy()
                self.pad_track_ns = stamp
                return
            dt = (stamp - self.pad_track_ns) * 1e-9
            self.pad_track_ns = stamp
            if not 0.0 < dt < 1.0:
                # A renderer/DDS stall is not evidence that a noisy absolute
                # fix suddenly became correct. Keep the last DR anchor; a live
                # optical sample may still correct it, but never jump it.
                if optical:
                    self.pad_track_position = bounded_position_update(
                        self.pad_track_position, position,
                        cfg.marker_fusion_max_correction_m)
                return
            predicted = self.pad_track_position + np.asarray(velocity, dtype=float) * dt
            if optical:
                self.pad_track_position = bounded_position_update(
                    predicted, position, cfg.marker_fusion_max_correction_m)
                return
            uav_sigma = _finite_or(self.gnss.get("sigma_xy_m", 99.9), 99.9)
            combined_sigma = math.hypot(uav_sigma, max(self.deck_sigma_xy_m, 0.0))
            quality = _finite_or(self.gnss.get("quality", 0.0), 0.0)
            hil_mode = str(self.sample.extra.get("hil_gps_mode", "legacy"))
            if (not bool(self.gnss.get("valid", False))
                    or quality <= cfg.gnss_dr_enter_quality
                    or hil_mode in {"imu_dominant", "inertial_dead_reckoning"}):
                # High-uncertainty canyon regime: pure short-term DR. Applying
                # even a tiny gain to a 20--70 m outlier over every high-rate
                # callback caused metres of systematic pull during one landing.
                # Use the sensor adapter's hysteretic mode too: otherwise one
                # marginal-quality epoch re-enables the correction while the
                # upstream estimator is deliberately still IMU-dominant.
                gain = 0.0
            else:
                reference_sigma = 1.5
                tau = max(cfg.pad_track_tau_s, 1e-3) * max(
                    1.0, (combined_sigma / reference_sigma) ** 2)
                tau = min(tau, 600.0)
                gain = min(dt / tau, 1.0)
            self.pad_track_position = predicted + gain * (position - predicted)

        def _track_deck(self, position: np.ndarray, velocity: np.ndarray) -> None:
            """A steadier deck pose, for the entry setpoint only.

            The broadcast carries the lorry's own canyon GNSS error, which is a
            correlated process metres across that wanders on the order of a
            metre a second. Handed straight to PX4's position controller at the
            control rate, that is a setpoint which never stands still: the
            vehicle chases the lorry's receiver noise and never settles inside
            the entry tolerance, which reads as a drone that will not stop
            drifting over a stationary lorry.

            So the entry hold flies a constant-velocity track of the broadcast
            rather than the broadcast itself -- predict on the reported twist,
            correct gently toward the reported position. Because the prediction
            carries the velocity, a lorry at a steady 8 m/s is followed with no
            lag; only the noise is attenuated. This is what any real consumer of
            a cooperative broadcast does, and it is deliberately *not* on the
            path the policy sees: `sample.pad` and the pad-relative state still
            carry the raw broadcast, because coping with it is the experiment.
            """
            stamp = now_ns()
            if self.deck_track_ns == 0:
                self.deck_track_position = position.copy()
                self.deck_track_ns = stamp
                return
            dt = (stamp - self.deck_track_ns) * 1e-9
            self.deck_track_ns = stamp
            if not 0.0 < dt < 1.0:
                # A gap this long means the track is stale rather than smooth.
                self.deck_track_position = position.copy()
                return
            predicted = self.deck_track_position + velocity * dt
            gain = min(dt / max(cfg.deck_track_tau_s, 1e-3), 1.0)
            self.deck_track_position = predicted + gain * (position - predicted)

        def _on_deck_truth(self, msg) -> None:
            """The simulator's own deck pose. Scoring only, never control."""
            p = msg.pose.pose.position
            v = msg.twist.twist.linear
            position = np.array([p.x, p.y, p.z], dtype=float)
            velocity = np.array([v.x, v.y, v.z], dtype=float)
            if np.isfinite(position).all() and np.isfinite(velocity).all():
                self.deck_truth_position_enu = position
                self.deck_truth_velocity_enu = velocity

        def _on_uav_truth(self, msg) -> None:
            """Simulator geometry for scoring only, never policy or control."""
            p = msg.pose.pose.position
            v = msg.twist.twist.linear
            o = msg.pose.pose.orientation
            position = np.array([p.x, p.y, p.z], dtype=float)
            velocity = np.array([v.x, v.y, v.z], dtype=float)
            if np.isfinite(position).all() and np.isfinite(velocity).all():
                self.uav_truth_position_enu = position
                self.uav_truth_velocity_enu = velocity
            # The attitude half is what makes geometric pad-centre FOV a pure
            # simulator quantity instead of an estimator-dependent one.
            quaternion = np.array([o.w, o.x, o.y, o.z], dtype=float)
            if np.isfinite(quaternion).all() and np.linalg.norm(quaternion) > 1e-9:
                self.uav_truth_quaternion_wxyz = (
                    quaternion / np.linalg.norm(quaternion))

        def _on_gnss_status(self, msg) -> None:
            """Receive policy-safe observables and scoring-only GNSS truth.

            HIL_GPS normally applies the measurement upstream of PX4. The
            downstream offset path below remains only for explicit legacy runs
            with ``inject_into_px4: false``.
            """
            try:
                payload = json.loads(msg.data)
                uav = payload["uav"]
                deck = payload.get("deck") or {}
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                self.get_logger().warning("ignored malformed GNSS status")
                return
            truth = uav.get("truth") or {}
            offset = np.asarray(truth.get("error_enu_m", (0.0, 0.0, 0.0)), dtype=float)
            if offset.shape != (3,) or not np.isfinite(offset).all():
                return
            velocity_offset = np.asarray(
                truth.get("velocity_error_enu_m_s", (0.0, 0.0, 0.0)), dtype=float)
            if velocity_offset.shape != (3,) or not np.isfinite(velocity_offset).all():
                velocity_offset = np.zeros(3)
            self.gnss_injected_into_px4 = bool(payload.get("injected_into_px4", False))
            self.sample.extra["hil_gps_mode"] = str(payload.get("hil_gps_mode", "legacy"))
            self.gnss_offset = np.zeros(3) if self.gnss_injected_into_px4 else offset
            self.gnss_velocity_offset = (
                np.zeros(3) if self.gnss_injected_into_px4 else velocity_offset)
            self.gnss = {
                "enabled": True,
                "source": "isaac",
                "fusion_path": ("px4_ekf2_hil_gps" if self.gnss_injected_into_px4
                                else "gateway_legacy"),
                "valid": bool(uav.get("valid", True)),
                "fix_type": int(uav.get("fix_type", 3)),
                "satellites_tracked": int(uav.get("satellites_tracked", 0)),
                # The receiver's own C/N0 test, not the simulator's count: a
                # detector with false alarms and misses is what a real vehicle
                # has, and it is what the ontology has to reason on.
                "nlos_detected_fraction": float(uav.get("nlos_detected_fraction", 0.0)),
                "cn0_mean_db": float(uav.get("cn0_mean_db", 0.0)),
                "hdop": float(uav.get("hdop", 99.9)),
                "vdop": float(uav.get("vdop", 99.9)),
                "residual_rms_m": float(uav.get("residual_rms_m", 0.0)),
                "sigma_xy_m": float(uav.get("sigma_xy_m", 0.0)),
                "quality": float(uav.get("quality", 1.0)),
                # The lorry publishes its own integrity too, because the pose
                # it broadcasts is only as good as its receiver.
                "deck_quality": float(deck.get("quality", 1.0)),
                "deck_sigma_xy_m": float(deck.get("sigma_xy_m", 0.0)),
            }
            self.gnss_time_ns = now_ns()

        def _on_battery_status(self, msg) -> None:
            """PX4's own pack state. Authoritative on hardware, ignored in SITL.

            SITL's simulated battery is the documented cause of the stale
            arming failure and cannot be seeded per episode, so the model owns
            SITL and this is telemetry there.
            """
            remaining = float(getattr(msg, "remaining", float("nan")))
            voltage = float(getattr(msg, "voltage_v", 0.0))
            self.px4_battery_time_ns = now_ns()
            self.sample.extra["px4_battery"] = [remaining, voltage]
            if (cfg.target == "hardware" and cfg.battery.enabled
                    and cfg.battery.prefer_px4_telemetry_on_hardware
                    and math.isfinite(remaining)):
                self.battery.adopt_px4(remaining, voltage)

        def _deck_is_fresh(self, stamp: int) -> bool:
            if cfg.pad_is_static:
                return True
            if self.deck_time_ns == 0:
                if ((stamp - self.start_ns) * 1e-9 >= 2.0
                        and not self.warned_missing_deck):
                    self.warned_missing_deck = True
                    self.get_logger().error(
                        "pad.motion is %r but no /landing_pad/state/odom has arrived; "
                        "every pad-relative number would be measured against a "
                        "stationary deck" % cfg.pad_motion)
                return False
            return (stamp - self.deck_time_ns) * 1e-9 <= cfg.state_timeout_s

        def _pad_state(self, fresh: bool) -> dict[str, Any]:
            if cfg.pad_is_static and self.deck_time_ns == 0:
                return static_pad_state()
            return {
                "valid": bool(fresh),
                "source": "v2v" if fresh else "stale",
                "position": [float(x) for x in self.deck_position_enu],
                "velocity": [float(x) for x in self.deck_velocity_enu],
                "yaw": float(self.deck_yaw),
                "yaw_rate": float(self.deck_yaw_rate),
                "speed": float(np.linalg.norm(self.deck_velocity_enu[:2])),
                # How well the lorry says it knows where it is. A consumer that
                # ignores this is treating a canyon fix as a survey mark.
                "sigma_xy_m": float(self.deck_sigma_xy_m),
            }

        def _gnss_state(self, stamp: int) -> dict[str, Any]:
            """The receiver's own report, or the open-sky default.

            A configured-on GNSS model whose topic has gone quiet is reported
            as stale rather than as a good fix: silently substituting open sky
            would make a degraded run look like the control condition.
            """
            if not cfg.gnss_enabled:
                return open_sky_gnss_state()
            if self.gnss_time_ns == 0:
                if ((stamp - self.start_ns) * 1e-9 >= 2.0
                        and not self.warned_missing_gnss):
                    self.warned_missing_gnss = True
                    self.get_logger().error(
                        "gnss.enabled is true but no /landing_uav0/gnss/status has "
                        "arrived; navigation integrity is being marked invalid")
                state = open_sky_gnss_state()
                state.update({"enabled": True, "source": "missing", "valid": False,
                              "fix_type": 0, "satellites_tracked": 0,
                              "sigma_xy_m": 99.9, "quality": 0.0})
                return state
            state = dict(self.gnss)
            if (stamp - self.gnss_time_ns) * 1e-9 > cfg.state_timeout_s:
                state["source"] = "stale"
                state["valid"] = False
                state["quality"] = 0.0
            return state

        def _set_truth(self, position: np.ndarray, velocity: np.ndarray) -> None:
            """Pad-relative state as the simulator knows it, for scoring only.

            With HIL_GPS injection PX4's world pose is deliberately fallible,
            so scoring uses the separate simulator UAV and deck odometry. A
            legacy downstream-injection run may still use the clean PX4 pose.
            Absent either valid truth path the field says so rather than
            guessing.
            """
            if self.deck_truth_position_enu is None:
                self.sample.truth_position_enu = None
                self.sample.truth_velocity_enu = None
                self.sample.truth_quaternion_enu_flu_wxyz = None
                return
            self.sample.truth_quaternion_enu_flu_wxyz = (
                None if self.uav_truth_quaternion_wxyz is None else
                tuple(float(x) for x in self.uav_truth_quaternion_wxyz))
            if self.uav_truth_position_enu is not None:
                position = self.uav_truth_position_enu
                velocity = self.uav_truth_velocity_enu
            elif self.gnss_injected_into_px4:
                # Once the urban fix is fused upstream, PX4 odometry is no
                # longer simulator truth.  Never grade on it as if it were.
                self.sample.truth_position_enu = None
                self.sample.truth_velocity_enu = None
                self.sample.truth_quaternion_enu_flu_wxyz = None
                return
            self.sample.truth_position_enu = tuple(
                float(x) for x in position - self.deck_truth_position_enu)
            self.sample.truth_velocity_enu = tuple(
                float(x) for x in velocity - self.deck_truth_velocity_enu)

        def _goto_world_target(self) -> np.ndarray:
            """Where PX4 is told to fly, in world ENU.

            A pad-relative climb is re-aimed at the live deck on every control
            tick, so PX4 chases a moving entry point instead of holding a stale
            one. The result is clamped to the arena the drone is allowed in:
            the protocol bounds the offset, and this bounds where the offset
            plus a moving deck can put the vehicle.
            """
            target = np.asarray(self.goto_target_enu, dtype=float).copy()
            deck = (self.deck_track_position if self.deck_track_ns
                    else self.deck_position_enu)
            if self.goto_pad_relative:
                # The offset is bounded against the pad-relative arena, because
                # that is the frame it is expressed in. Clamping the sum against
                # the world origin instead would drag the vehicle back to the
                # middle of the block every time the lorry drove away from it.
                offset = float(math.hypot(target[0], target[1]))
                if offset > cfg.world_xy_limit_m > 0.0:
                    target[:2] *= cfg.world_xy_limit_m / offset
                target[2] = float(min(max(target[2], 0.2), cfg.max_altitude_m))
                # Fly the gap that was *measured*, not an absolute point.
                #
                # In this canyon the drone's own fix and the lorry's are each
                # about twenty metres out, but they share a constellation, so
                # better than eighty per cent of that is common-mode and their
                # difference is good to a few metres. Adding the offset to the
                # lorry's absolute broadcast throws that away: the setpoint
                # inherits the lorry's full error and wanders with it, so the
                # vehicle chases a point that never stands still -- while the
                # entry tolerance, which is checked on the pad-relative state,
                # reports it as a metre away and settling. That is the drift
                # that keeps a handover from ever completing.
                #
                # Commanding the remaining pad-relative error instead cancels
                # both absolute fixes exactly the way the state contract
                # already does, and leaves the setpoint as steady as the
                # differential measurement is.
                here_world = (self.px4_world_position
                              if self.px4_world_position is not None
                              else np.asarray(self.sample.position_world_enu, dtype=float))
                # Fly the climb on the simulator's own pad-relative pose when
                # it has one. This is setup, not the experiment: it decides
                # where the vehicle waits before an episode begins, and nothing
                # on it reaches the policy, the reward or the log -- exactly
                # like re-seating the vehicle between episodes.
                #
                # It has to be the true pose because of a circularity. The
                # episode is supposed to open with the deck in frame, and the
                # camera reaches three or four metres across at entry altitude.
                # Flown on the GNSS fallback instead -- thirty metres out in
                # this canyon -- the vehicle parks a block from the deck, never
                # sees a marker, and so never gets the fix that would have let
                # it fly there accurately. Truth breaks the loop; the policy
                # still takes over on the raw measurement it will have to land
                # on.
                truth = self.sample.truth_position_enu
                if truth is not None:
                    here_pad = np.asarray(truth, dtype=float)
                else:
                    here_pad = (self.pad_track_position if self.pad_track_ns
                                else np.asarray(self.sample.position_enu, dtype=float))
                if np.isfinite(here_world).all() and np.isfinite(here_pad).all():
                    target = here_world + (target - here_pad)
                else:
                    target = target + deck
            # And the result is bounded against the city, so no combination of
            # a legal offset and a moving deck can put the vehicle outside it.
            radial = float(math.hypot(target[0], target[1]))
            if radial > cfg.world_radius_m > 0.0:
                target[:2] *= cfg.world_radius_m / radial
            ceiling = cfg.max_altitude_m + (
                deck[2] if self.goto_pad_relative else 0.0)
            target[2] = float(min(max(target[2], 0.2), max(ceiling, 0.2)))
            return target

        def _arm_battery(self) -> None:
            """Start the episode's energy budget at policy handover."""
            self.battery.reset(self.pending_battery_hover_s)
            self.battery_armed = True
            self.battery_time_us = int(self.sample.px4_time_us)
            self.sample.battery = self.battery.sample()

        def _integrate_battery(self) -> None:
            """Charge elapsed *simulated* seconds against the pack.

            Wall time is the wrong clock: PX4 and Isaac run in lockstep, so a
            slow frame would otherwise bill the policy for the simulator's
            stall. PX4's own timestamp is the simulator's clock.
            """
            if not cfg.battery.enabled or not self.battery_armed:
                return
            now_us = int(self.sample.px4_time_us)
            if now_us <= 0:
                return
            if self.battery_time_us <= 0:
                self.battery_time_us = now_us
                return
            dt = (now_us - self.battery_time_us) * 1e-6
            self.battery_time_us = now_us
            # Guard a clock reset and a long stall; the pacing loop already
            # refuses to fake progress, so a huge step is not ours to charge.
            if not 0.0 < dt < 10.0 / cfg.control_hz:
                return
            if self.battery.source == "px4":
                return
            thrust = self.sample.extra.get("px4_thrust")
            if thrust is None:
                # No patched thrust feedback: price the collective we commanded.
                thrust = cfg.hover_thrust * (1.0 + cfg.collective_span * self.action[0])
            self.battery.integrate(float(thrust), cfg.hover_thrust, dt)

        def _on_reset_ack(self, msg) -> None:
            try:
                payload = json.loads(msg.data)
                seq = int(payload["seq"])
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                self.get_logger().warning("ignored malformed reset acknowledgement")
                return
            if seq == self.pending_reset_seq:
                peer = self.pending_reset_peer
                self.pending_reset_seq = -1
                self.pending_reset_peer = None
                # Isaac draws the episode's starting reserve from the same seed
                # as the entry pose, so energy is reproducible with the rest of
                # the initial condition rather than being a second RNG.
                hover_s = payload.get("battery_hover_seconds")
                if hover_s is not None and math.isfinite(float(hover_s)):
                    self.pending_battery_hover_s = float(hover_s)
                self.battery.reset(self.pending_battery_hover_s)
                self.battery_armed = False
                self.battery_time_us = 0
                self.sample.battery = self.battery.sample()
                self._send_ack(seq, "reset_complete", payload, peer=peer)

        def _send_ack(self, ack_seq: int, status: str, detail: Any = None,
                      peer: tuple[str, int] | None = None) -> None:
            self.tx_seq += 1
            self.udp.send({"v": cfg.protocol_version, "type": "ack", "seq": self.tx_seq,
                           "ack_seq": ack_seq, "time_ns": now_ns(), "status": status,
                           "detail": detail}, peer=peer)

        def _send_state(self, ack_seq: int,
                        peer: tuple[str, int] | None = None) -> None:
            self.tx_seq += 1
            stamp = now_ns()
            if ((stamp - self.sample.timestamp_ns) * 1e-9 > cfg.state_timeout_s):
                self.sample.estimator_valid = False
            self._refresh_land_detector(stamp)
            # Pad and battery are refreshed here as well as on odometry, so a
            # state query that arrives before the first PX4 sample still gets
            # the truth about the deck and the pack instead of a default.
            self.sample.pad = self._pad_state(self._deck_is_fresh(stamp))
            self.sample.battery = self.battery.sample()
            self.sample.gnss = self._gnss_state(stamp)
            # Report the commanded fusion regime as well as PX4's raw flags.
            # A valid low-quality fix remains weakly fused by EKF2 to bound
            # drift, so cs_gps alone must not label that interval fully aided.
            hil_mode = str(self.sample.extra.get("hil_gps_mode", "legacy"))
            ekf = self.sample.extra.get("ekf_fusion") or {}
            if hil_mode == "inertial_dead_reckoning" or bool(
                    ekf.get("inertial_dead_reckoning", False)):
                self.sample.extra["navigation_mode"] = "inertial_dead_reckoning"
            elif hil_mode == "imu_dominant":
                self.sample.extra["navigation_mode"] = "imu_dr_gnss_bounded"
            elif bool(ekf.get("gps", False)):
                self.sample.extra["navigation_mode"] = "gnss_aided"
            self.udp.send(self.sample.to_message(
                cfg.protocol_version, self.tx_seq, ack_seq), peer=peer)

        def _refresh_land_detector(self, stamp: int) -> None:
            """Fuse PX4 landed state with the physical pad-contact switch.

            PX4 has to be built with vehicle_land_detected in dds_topics.yaml
            (see patches/px4-v1.14-publish-land-detected.patch). A default of
            "landed" silently disables touchdown detection for a whole run, so
            report the outage instead of guessing. On a driving deck PX4's
            world-frame motion test is insufficient, while physical roof
            contact remains authoritative.
            """
            if self.land_detected_ns == 0:
                self.sample.extra["land_detector"] = "missing"
                # The PX4 writer is change-driven and may legitimately stay
                # quiet through the externally supported pre-arm hover. Give
                # it through the normal arm/takeoff transition before warning.
                if ((stamp - self.start_ns) * 1e-9 >= 10.0
                        and not self.warned_missing_land_detector):
                    self.warned_missing_land_detector = True
                    self.get_logger().warning(
                        f"no {_topic(cfg, 'out', 'vehicle_land_detected')} sample "
                        "received yet; physical pad contact remains the SITL "
                        "touchdown authority"
                    )
            elif (stamp - self.land_detected_ns) * 1e-9 > cfg.state_timeout_s:
                self.sample.extra["land_detector"] = "stale"
            else:
                self.sample.extra["land_detector"] = "live"

            contact_live = (self.pad_contact_ns > 0
                            and (stamp - self.pad_contact_ns) * 1e-9
                            <= cfg.state_timeout_s)
            self.sample.extra["pad_contact_raw"] = bool(
                contact_live and self.pad_contact_raw)
            self.sample.extra["pad_contact"] = bool(self.pad_contact_latched)
            px4_landed = effective_px4_landed(
                self.px4_landed, self.sample.armed,
                self.sample.truth_position_enu)
            self.sample.extra["px4_landed_raw"] = bool(self.px4_landed)
            self.sample.extra["px4_landed"] = bool(px4_landed)
            self.sample.landed = bool(px4_landed or self.pad_contact_latched)

            px4_authoritative = bool(
                cfg.pad_is_static and self.sample.extra["land_detector"] == "live")
            self.sample.extra["land_detector_authoritative"] = bool(
                self.pad_contact_latched or px4_authoritative)
            self.sample.extra["touchdown_source"] = (
                "pad_contact" if self.pad_contact_latched
                else "px4_land_detector" if px4_landed
                else "none")

    return NodeImpl()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MATLAB to PX4 uXRCE-DDS gateway")
    parser.add_argument("--config", required=True)
    parser.add_argument("--target", choices=("sitl", "hardware"))
    parser.add_argument("--allow-arm", action="store_true")
    parser.add_argument("--allow-offboard", action="store_true")
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--parallel-pairs", type=int, default=1)
    parser.add_argument("--gateway-port", type=int)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_gateway_config(Path(args.config), args.target)
    cfg = parallel_gateway_config(
        cfg, args.pair_index, args.parallel_pairs, args.gateway_port)
    _load_ros_types()
    rclpy.init(args=None)
    node = Px4GatewayNode(cfg, allow_arm=args.allow_arm, allow_offboard=args.allow_offboard)
    # The node's default callback group handles sensor traffic while the
    # dedicated control group keeps publishing PX4's OFFBOARD heartbeat.
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    except Exception:
        # Some Humble builds surface the same launch-driven context shutdown as
        # RCLError while constructing the executor wait set. Preserve genuine
        # runtime failures, but do not print a traceback for an already-closed
        # ROS context during an otherwise clean stack restart.
        if rclpy.ok():
            raise
    finally:
        executor.shutdown()
        node.destroy_node()
        # A launch service can shut the shared context down before spin exits.
        # Calling shutdown twice used to turn every clean stack restart into a
        # misleading traceback in gateway.log.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
