from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .config import GatewayConfig, load_gateway_config
from .frames import (
    enu_to_ned,
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
    from rclpy.node import Node
    from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint
    from px4_msgs.msg import VehicleAttitudeSetpoint, VehicleCommand, VehicleCommandAck
    from px4_msgs.msg import VehicleLandDetected, VehicleLocalPosition
    from px4_msgs.msg import VehicleOdometry, VehicleStatus, VehicleThrustSetpoint
    from std_msgs.msg import Float32, String
    return (Node, PoseStamped, Vector3Stamped, OffboardControlMode, TrajectorySetpoint,
            VehicleAttitudeSetpoint, VehicleCommand, VehicleCommandAck, VehicleLandDetected,
            VehicleLocalPosition, VehicleOdometry, VehicleStatus, VehicleThrustSetpoint,
            Float32, String)


def _set_if_present(message: Any, name: str, value: Any) -> None:
    if hasattr(message, name):
        setattr(message, name, value)


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
    (Node, PoseStamped, Vector3Stamped, OffboardControlMode, TrajectorySetpoint,
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
            # Pre-episode climb flown by PX4's own position controller.
            self.goto_target_enu: np.ndarray | None = None
            self.goto_yaw_enu = 0.0
            self.goto_deadline_ns = 0
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

            self.create_subscription(VehicleOdometry, _topic(cfg, "out", "vehicle_odometry"),
                                     self._on_odometry, qos)
            self.create_subscription(VehicleStatus, _topic(cfg, "out", "vehicle_status"),
                                     self._on_status, qos)
            self.create_subscription(VehicleLandDetected, _topic(cfg, "out", "vehicle_land_detected"),
                                     self._on_land, qos)
            self.create_subscription(VehicleLocalPosition, _topic(cfg, "out", "vehicle_local_position"),
                                     self._on_local_position, qos)
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
            self.create_subscription(String, "/landing_sim/reset_ack", self._on_reset_ack, 10)

            self.udp = DatagramServer(cfg.bind_host, cfg.gateway_port,
                                      cfg.protocol_version, self._on_udp)
            self.create_timer(0.005, self.udp.poll)
            self.create_timer(1.0 / cfg.control_hz, self._control_tick)
            self.get_logger().info(
                f"gateway target={cfg.target}, UDP={cfg.bind_host}:{cfg.gateway_port}, "
                f"PX4 namespace={cfg.namespace}, arm_allowed={safety.may_arm()}, "
                f"offboard_allowed={safety.may_enable_offboard()}"
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
                # The first policy action ends the pre-episode position hold.
                self.goto_target_enu = None
            elif kind == "reset":
                self.safety.require_reset()
                wind_scale = float(msg.get("wind_scale", 1.0))
                if not math.isfinite(wind_scale) or not 0.0 <= wind_scale <= 4.0:
                    raise ProtocolError("wind_scale must be finite and in [0,4]")
                self.last_command_seq = seq
                self.pending_reset_seq = seq
                if self.sample.landed:
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
                                       "wind_scale": wind_scale})
                self.reset_pub.publish(req)
                self.action = (0.0, 0.0, 0.0, 0.0)
                self.last_action_ns = 0
                self.prestream = 0
                self.offboard_requested = False
                self.last_mode_request_tick = -self.mode_request_period
                self.goto_target_enu = None
            elif kind == "goto":
                self.safety.require_autonomous_climb()
                request = validate_goto(msg)
                self.last_command_seq = seq
                self.goto_target_enu = np.asarray(request.position_enu, dtype=float)
                self.goto_yaw_enu = request.yaw_enu_rad
                self.goto_deadline_ns = now_ns() + int(request.hold_s * 1e9)
                # An action deadman must not cancel the climb that precedes it.
                self.action = (0.0, 0.0, 0.0, 0.0)
                self.last_action_ns = 0
                self._send_ack(seq, "goto_started",
                               {"position": list(request.position_enu),
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
            if self.goto_target_enu is not None and stamp >= self.goto_deadline_ns:
                self.goto_target_enu = None
                self.get_logger().warning("goto hold expired; releasing the setpoint stream")
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
            target_ned = enu_to_ned(self.goto_target_enu)
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
                position = ned_to_enu(msg.position)
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
            # Hardware always flies on the pad-relative pose. In SITL it is a
            # choice: with the camera enabled the policy lands on the marker it
            # can actually see, and falls back to the PX4 estimate the moment
            # the pad leaves the frame.
            use_pad_pose = pad_pose_fresh and (
                cfg.target == "hardware" or cfg.marker_pose_drives_policy)
            policy_position = self.pad_position_enu if use_pad_pose else position
            self.sample.position_enu = tuple(float(x) for x in policy_position)
            self.sample.velocity_enu = tuple(float(x) for x in velocity)
            self.sample.quaternion_enu_flu_wxyz = tuple(float(x) for x in q_enu)
            self.sample.angular_velocity_flu = tuple(float(x) for x in omega)
            self.sample.acceleration_enu = tuple(float(x) for x in accel)
            navigation_valid = bool(np.isfinite(position).all() and np.isfinite(q_enu).all())
            self.sample.estimator_valid = (
                navigation_valid
                and self._estimator_is_healthy(stamp)
                and (cfg.target == "sitl" or pad_pose_fresh)
            )
            self.sample.extra["position_source"] = (
                "uav_pose_in_pad" if use_pad_pose else "px4_local")
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

        def _estimator_is_healthy(self, stamp: int) -> bool:
            fresh = (stamp - self.estimator_health_ns) * 1e-9 <= cfg.state_timeout_s
            return bool(self.estimator_healthy and fresh)

        def _on_wind(self, msg) -> None:
            self.sample.wind_enu = (float(msg.vector.x), float(msg.vector.y), float(msg.vector.z))

        def _on_aero_force(self, msg) -> None:
            self.sample.aero_force_enu = (float(msg.vector.x), float(msg.vector.y), float(msg.vector.z))

        def _on_marker_quality(self, msg) -> None:
            self.sample.marker_quality = float(np.clip(msg.data, 0.0, 1.0))

        def _on_pad_pose(self, msg) -> None:
            position = np.array(
                [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float
            )
            if np.isfinite(position).all():
                self.pad_position_enu = position
                self.pad_pose_time_ns = now_ns()

        def _on_reset_ack(self, msg) -> None:
            try:
                payload = json.loads(msg.data)
                seq = int(payload["seq"])
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                self.get_logger().warning("ignored malformed reset acknowledgement")
                return
            if seq == self.pending_reset_seq:
                self.pending_reset_seq = -1
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
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
