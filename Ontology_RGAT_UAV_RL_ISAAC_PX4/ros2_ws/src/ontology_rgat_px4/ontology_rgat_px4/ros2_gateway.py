from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
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
    ProtocolError,
    VehicleSample,
    now_ns,
    open_sky_gnss_state,
    static_pad_state,
    validate_action,
    validate_goto,
)
from .safety import SafetyGate
from .udp_server import DatagramServer

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
    from px4_msgs.msg import BatteryStatus, OffboardControlMode, TrajectorySetpoint
    from px4_msgs.msg import EstimatorGpsStatus, EstimatorStatusFlags, SensorGps
    from px4_msgs.msg import VehicleAttitudeSetpoint, VehicleCommand, VehicleCommandAck
    from px4_msgs.msg import VehicleLandDetected, VehicleLocalPosition
    from px4_msgs.msg import VehicleOdometry, VehicleStatus, VehicleThrustSetpoint
    from std_msgs.msg import Float32, String
    return (Node, PoseStamped, Vector3Stamped, Odometry, BatteryStatus,
            EstimatorGpsStatus, EstimatorStatusFlags, SensorGps,
            OffboardControlMode, TrajectorySetpoint,
            VehicleAttitudeSetpoint, VehicleCommand, VehicleCommandAck, VehicleLandDetected,
            VehicleLocalPosition, VehicleOdometry, VehicleStatus, VehicleThrustSetpoint,
            Float32, String)


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


def _topic(cfg: GatewayConfig, direction: str, name: str) -> str:
    return f"{cfg.namespace}/{direction}/{name}"


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
    (Node, PoseStamped, Vector3Stamped, Odometry, BatteryStatus,
     EstimatorGpsStatus, EstimatorStatusFlags, SensorGps,
     OffboardControlMode, TrajectorySetpoint,
     VehicleAttitudeSetpoint, VehicleCommand, VehicleCommandAck, VehicleLandDetected,
     VehicleLocalPosition, VehicleOdometry, VehicleStatus, VehicleThrustSetpoint,
     Float32, String) = types

    class NodeImpl(Node):
        def __init__(self):
            super().__init__("ontology_rgat_px4_gateway")
            self.cfg = cfg
            self.safety = safety
            self.sample = VehicleSample()
            self.action = (0.0, 0.0, 0.0, 0.0)
            self.last_action_ns = 0
            self.last_command_seq = -1
            self.pending_state_ack = -1
            self.tx_seq = 0
            self.prestream = 0
            self.offboard_requested = False
            self.offboard_enabled = cfg.target == "sitl"
            self.last_velocity: np.ndarray | None = None
            self.last_velocity_ns = 0
            self.pending_reset_seq = -1
            self.pad_position_enu: np.ndarray | None = None
            self.pad_pose_time_ns = 0
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
            self.warned_missing_land_detector = False
            self.offboard_nav_state = int(getattr(VehicleStatus, "NAVIGATION_STATE_OFFBOARD", 14))
            self.mode_request_period = max(1, int(round(cfg.control_hz / 2.0)))
            self.last_mode_request_tick = -self.mode_request_period

            qos = _make_qos(rclpy)
            self.offboard_pub = self.create_publisher(
                OffboardControlMode, _topic(cfg, "in", "offboard_control_mode"), 10)
            self.attitude_pub = self.create_publisher(
                VehicleAttitudeSetpoint, _topic(cfg, "in", "vehicle_attitude_setpoint"), 10)
            self.trajectory_pub = self.create_publisher(
                TrajectorySetpoint, _topic(cfg, "in", "trajectory_setpoint"), 10)
            self.command_pub = self.create_publisher(
                VehicleCommand, _topic(cfg, "in", "vehicle_command"), 10)
            self.reset_pub = self.create_publisher(String, "/landing_sim/reset", 10)
            # Isaac holds the vehicle at its hover start until the autopilot is
            # actually flying it; this is how it learns that. Latched depth so a
            # simulator that comes up late still gets the current answer.
            self.flight_pub = self.create_publisher(String, "/landing_sim/flight_state", 10)
            self.published_armed: tuple[bool, bool] | None = None

            self.create_subscription(VehicleOdometry, _topic(cfg, "out", "vehicle_odometry"),
                                     self._on_odometry, qos)
            self.create_subscription(VehicleStatus, _topic(cfg, "out", "vehicle_status"),
                                     self._on_status, qos)
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
            self.create_subscription(Vector3Stamped, "/landing_uav0/environment/wind",
                                     self._on_wind, sensor_qos)
            self.create_subscription(Vector3Stamped, "/landing_uav0/environment/aero_force",
                                     self._on_aero_force, sensor_qos)
            self.create_subscription(Float32, "/landing_uav0/perception/marker_quality",
                                     self._on_marker_quality, sensor_qos)
            self.create_subscription(PoseStamped, "/landing_uav0/perception/uav_pose_in_pad",
                                     self._on_pad_pose, sensor_qos)
            # The deck broadcasts its own state, the way a cooperative ground
            # vehicle would over a V2V link. The drone's own estimate of where
            # the pad is still comes from the camera.
            self.create_subscription(Odometry, "/landing_pad/state/odom",
                                     self._on_deck_odom, sensor_qos)
            self.create_subscription(Odometry, "/landing_pad/state/odom_truth",
                                     self._on_deck_truth, sensor_qos)
            self.create_subscription(Odometry, "/landing_uav0/state/odom_truth",
                                     self._on_uav_truth, sensor_qos)
            # The drone's own receiver, as a receiver reports it, plus the error
            # the simulator wants injected downstream of PX4's estimator.
            self.create_subscription(String, "/landing_uav0/gnss/status",
                                     self._on_gnss_status, sensor_qos)
            self.create_subscription(BatteryStatus, _topic(cfg, "out", "battery_status"),
                                     self._on_battery_status, qos)
            self.create_subscription(String, "/landing_sim/reset_ack", self._on_reset_ack, 10)

            self.udp = DatagramServer(cfg.bind_host, cfg.gateway_port,
                                      cfg.protocol_version, self._on_udp)
            self.create_timer(0.005, self.udp.poll)
            self.create_timer(1.0 / cfg.control_hz, self._control_tick)
            self.get_logger().info(
                f"gateway target={cfg.target}, UDP={cfg.bind_host}:{cfg.gateway_port}, "
                f"PX4 namespace={cfg.namespace}, arm_allowed={safety.may_arm()}, "
                f"offboard_allowed={safety.may_enable_offboard()}, "
                f"pad_motion={cfg.pad_motion}, gnss={'on' if cfg.gnss_enabled else 'off'}, "
                f"battery={'on' if cfg.battery.enabled else 'off'} "
                f"(hover {self.battery.hover_power_w:.0f} W)"
            )

        def destroy_node(self):
            self.udp.close()
            return super().destroy_node()

        def _timestamp_us(self) -> int:
            return self.get_clock().now().nanoseconds // 1000

        def _on_udp(self, msg: dict[str, Any]) -> None:
            kind = msg["type"]
            seq = msg["seq"]
            if seq <= self.last_command_seq and kind not in {"hello", "state"}:
                self._send_ack(seq, "duplicate")
                return
            if kind == "action":
                self.action = validate_action(msg)
                self.last_action_ns = now_ns()
                self.last_command_seq = seq
                self.pending_state_ack = seq
                # The first policy action ends the pre-episode position hold and
                # starts the energy budget: the seeded reserve is the reserve at
                # handover, so the climb PX4 flew to get here is not charged to
                # the policy.
                self.goto_target_enu = None
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
                # Scales the canyon's error mechanisms without moving a
                # building, so a sweep can ask how much GNSS degradation a
                # policy survives; 0.0 is the open-sky control condition.
                gnss_scale = float(msg.get("gnss_scale", 1.0))
                if not math.isfinite(gnss_scale) or not 0.0 <= gnss_scale <= 4.0:
                    raise ProtocolError("gnss_scale must be finite and in [0,4]")
                self.last_command_seq = seq
                self.pending_reset_seq = seq
                # A disarmed vehicle is on the ground whatever the land detector
                # says -- and before its first sample arrives it says "airborne"
                # by design (see _refresh_land_detector), so without this the
                # first reset after every gateway start put a parked PX4 into
                # AUTO.LAND straight out of "Ready for takeoff".
                if self.sample.landed or not self.sample.armed:
                    self._vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)
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
                                       "gnss_scale": gnss_scale})
                self.reset_pub.publish(req)
                self.action = (0.0, 0.0, 0.0, 0.0)
                self.last_action_ns = 0
                self.prestream = 0
                self.offboard_requested = False
                self.last_mode_request_tick = -self.mode_request_period
                self.goto_target_enu = None
                self.goto_pad_relative = False
                self.deck_track_ns = 0
                self.pad_track_ns = 0
                self.battery_armed = False
                self.pending_battery_hover_s = None
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
                self.goto_pad_relative = request.is_pad_relative
                self.goto_yaw_enu = request.yaw_enu_rad
                self.goto_deadline_ns = now_ns() + int(request.hold_s * 1e9)
                # An action deadman must not cancel the climb that precedes it.
                self.action = (0.0, 0.0, 0.0, 0.0)
                self.last_action_ns = 0
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
                    self._vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)
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
                age_s = (stamp - self.last_action_ns) * 1e-9
                if age_s > cfg.action_timeout_s:
                    if self.prestream:
                        self.get_logger().warning(
                            "action deadman expired; yielding to PX4 offboard-loss failsafe")
                    self.prestream = 0
                    self.offboard_requested = False
                    self.last_mode_request_tick = -self.mode_request_period
                    self.last_action_ns = 0
            if self.last_action_ns:
                self._publish_attitude_setpoint()
            elif self.goto_target_enu is not None:
                self._publish_position_setpoint()
            else:
                return

            self.prestream += 1
            if self.offboard_enabled and self.prestream >= cfg.offboard_prestream_count:
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
            self._publish_offboard_mode(position=True)
            # Back out of the world frame: this is published as a *local*
            # setpoint, so a world-frame target would be off by the distance
            # between the two origins -- which is what sent the vehicle
            # sideways the moment it lifted off the pad.
            target_ned = enu_to_ned(self._goto_world_target() - self.world_from_px4)
            sp = TrajectorySetpoint()
            sp.timestamp = self._timestamp_us()
            sp.position = [float(x) for x in target_ned]
            sp.velocity = [float("nan")] * 3
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
            self.sample.px4_time_us = int(msg.timestamp)
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
            policy_position = self._blend_position_source(
                use_pad_pose, self.pad_position_enu, fallback_position, stamp)
            # Velocity always comes from the flight stack's estimator, whatever
            # is providing position: the marker solve is not differentiated
            # here, so there is no optical velocity to prefer.
            policy_velocity = estimated_velocity - self.deck_velocity_enu
            self._track_pad_relative(policy_position, policy_velocity)
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
                "uav_pose_in_pad" if use_pad_pose else "px4_ekf_minus_deck_gnss")
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
                "hover_thrust": cfg.hover_thrust,
                "collective_span": cfg.collective_span,
                "max_roll_pitch_rad": cfg.max_roll_pitch_rad,
                "max_yaw_rate_rad_s": cfg.max_yaw_rate_rad_s,
            }
            if self.pending_state_ack >= 0:
                seq = self.pending_state_ack
                self.pending_state_ack = -1
                self._send_state(seq)

        def _on_status(self, msg) -> None:
            self.sample.nav_state = int(msg.nav_state)
            armed_value = getattr(msg, "ARMING_STATE_ARMED", 2)
            self.sample.armed = int(msg.arming_state) == int(armed_value)
            self.sample.extra["pre_flight_checks_pass"] = bool(
                getattr(msg, "pre_flight_checks_pass", False))
            self.sample.extra["px4_failsafe"] = bool(getattr(msg, "failsafe", False))
            self._publish_flight_state()

        def _on_land(self, msg) -> None:
            self.sample.landed = bool(msg.landed)
            self.land_detected_ns = now_ns()

        def _on_thrust_setpoint(self, msg) -> None:
            # PX4's own normalised body thrust. While its position controller
            # holds a hover this is the hover thrust the gateway mapping must
            # be centred on; see tools/calibrate_hover_thrust.py.
            self.sample.extra["px4_thrust"] = float(-msg.xyz[2])

        def _on_command_ack(self, msg) -> None:
            # A silently rejected arm used to look exactly like a vehicle that
            # simply refused to climb, so say so out loud.
            accepted = int(getattr(VehicleCommandAck, "VEHICLE_CMD_RESULT_ACCEPTED", 0))
            in_progress = int(getattr(VehicleCommandAck, "VEHICLE_CMD_RESULT_IN_PROGRESS", 5))
            result = int(msg.result)
            self.sample.extra["last_command"] = [int(msg.command), result]
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

        def _on_aero_force(self, msg) -> None:
            self.sample.aero_force_enu = (float(msg.vector.x), float(msg.vector.y), float(msg.vector.z))

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

            Only on a change: this is a latch for the simulator's hover hold,
            not telemetry, and the state sample already carries `armed` for
            anything that wants it every tick.
            """
            armed = bool(self.sample.armed)
            # Handover is the moment the policy takes the vehicle: the entry
            # climb is over and the episode has begun. That is when the deck is
            # allowed to pull away, so the climb happens over a lorry standing
            # still and only the landing has to chase one.
            handover = bool(self.last_action_ns)
            state = (armed, handover)
            if state == self.published_armed:
                return
            self.published_armed = state
            message = String()
            message.data = json.dumps({"v": cfg.protocol_version,
                                       "armed": armed, "handover": handover})
            self.flight_pub.publish(message)

        def _on_pad_pose(self, msg) -> None:
            position = np.array(
                [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float
            )
            if np.isfinite(position).all():
                self.pad_position_enu = position
                self.pad_pose_time_ns = now_ns()

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
                                velocity: np.ndarray) -> None:
            """A steadier pad-relative pose, for the entry setpoint only.

            The measured pad-relative position is the difference of two fixes
            in the same canyon, so most of the error is common-mode and what is
            left is metres, wandering on the multipath correlation time. Aim a
            position controller straight at it and the vehicle physically
            chases that wander -- it was flying at 1.8 m/s while trying to hold
            station, which is real motion in pursuit of noise.

            Predicting on the measured relative velocity and correcting gently
            toward the measured position leaves a deck moving at a steady speed
            tracked without lag, and takes the wander out of the setpoint. Only
            the entry climb reads this; `sample.position_enu` is untouched,
            because flying the raw differential is the experiment.
            """
            stamp = now_ns()
            if self.pad_track_ns == 0:
                self.pad_track_position = np.asarray(position, dtype=float).copy()
                self.pad_track_ns = stamp
                return
            dt = (stamp - self.pad_track_ns) * 1e-9
            self.pad_track_ns = stamp
            if not 0.0 < dt < 1.0:
                self.pad_track_position = np.asarray(position, dtype=float).copy()
                return
            predicted = self.pad_track_position + np.asarray(velocity, dtype=float) * dt
            gain = min(dt / max(cfg.pad_track_tau_s, 1e-3), 1.0)
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
            position = np.array([p.x, p.y, p.z], dtype=float)
            velocity = np.array([v.x, v.y, v.z], dtype=float)
            if np.isfinite(position).all() and np.isfinite(velocity).all():
                self.uav_truth_position_enu = position
                self.uav_truth_velocity_enu = velocity

        def _on_gnss_status(self, msg) -> None:
            """The drone's receiver, and the error to inject downstream of it.

            The observables are forwarded to the policy; the error vector is
            applied to PX4's estimate here and never leaves the gateway, since
            no receiver knows its own error and no consumer may act as if it
            did.
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
                if not self.warned_missing_deck:
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
                if not self.warned_missing_gnss:
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
                return
            if self.uav_truth_position_enu is not None:
                position = self.uav_truth_position_enu
                velocity = self.uav_truth_velocity_enu
            elif self.gnss_injected_into_px4:
                # Once the urban fix is fused upstream, PX4 odometry is no
                # longer simulator truth.  Never grade on it as if it were.
                self.sample.truth_position_enu = None
                self.sample.truth_velocity_enu = None
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
            if not cfg.battery.enabled:
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
                self.pending_reset_seq = -1
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
                self._send_ack(seq, "reset_complete", payload)

        def _send_ack(self, ack_seq: int, status: str, detail: Any = None) -> None:
            self.tx_seq += 1
            self.udp.send({"v": cfg.protocol_version, "type": "ack", "seq": self.tx_seq,
                           "ack_seq": ack_seq, "time_ns": now_ns(), "status": status,
                           "detail": detail})

        def _send_state(self, ack_seq: int) -> None:
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
            self.udp.send(self.sample.to_message(cfg.protocol_version, self.tx_seq, ack_seq))

        def _refresh_land_detector(self, stamp: int) -> None:
            """Never let a silent land detector read as a landed vehicle.

            PX4 has to be built with vehicle_land_detected in dds_topics.yaml
            (see patches/px4-v1.14-publish-land-detected.patch). A default of
            "landed" silently disables touchdown detection for a whole run, so
            report the outage instead of guessing.
            """
            if self.land_detected_ns == 0:
                self.sample.landed = False
                self.sample.extra["land_detector"] = "missing"
                if not self.warned_missing_land_detector:
                    self.warned_missing_land_detector = True
                    self.get_logger().error(
                        f"no {_topic(cfg, 'out', 'vehicle_land_detected')} received; "
                        "rebuild PX4 with patches/px4-v1.14-publish-land-detected.patch"
                    )
                return
            if (stamp - self.land_detected_ns) * 1e-9 > cfg.state_timeout_s:
                self.sample.extra["land_detector"] = "stale"
            else:
                self.sample.extra["land_detector"] = "live"
            # PX4 decides "landed" from world-frame motion. A vehicle sitting on
            # a driving deck is moving, so the detector under-reports touchdown
            # and the client must fall back to pad-relative altitude and closing
            # speed. Say so rather than letting it look authoritative.
            self.sample.extra["land_detector_authoritative"] = bool(cfg.pad_is_static)

    return NodeImpl()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MATLAB to PX4 uXRCE-DDS gateway")
    parser.add_argument("--config", required=True)
    parser.add_argument("--target", choices=("sitl", "hardware"))
    parser.add_argument("--allow-arm", action="store_true")
    parser.add_argument("--allow-offboard", action="store_true")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_gateway_config(Path(args.config), args.target)
    _load_ros_types()
    rclpy.init(args=None)
    node = Px4GatewayNode(cfg, allow_arm=args.allow_arm, allow_offboard=args.allow_offboard)
    try:
        rclpy.spin(node)
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
        node.destroy_node()
        # A launch service can shut the shared context down before spin exits.
        # Calling shutdown twice used to turn every clean stack restart into a
        # misleading traceback in gateway.log.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
