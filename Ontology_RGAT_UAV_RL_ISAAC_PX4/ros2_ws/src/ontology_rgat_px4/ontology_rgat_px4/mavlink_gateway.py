from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .config import GatewayConfig, load_gateway_config
from .frames import (
    frd_to_flu,
    ned_to_enu,
    quat_enu_flu_to_ned_frd,
    quat_ned_frd_to_enu_flu,
    quat_wxyz_to_matrix,
    yaw_from_quat_wxyz,
    euler_zyx_to_quat_wxyz,
)
from .protocol import ProtocolError, VehicleSample, now_ns, validate_action
from .safety import SafetyGate
from .udp_server import DatagramServer


class MavlinkGateway:
    def __init__(self, cfg: GatewayConfig, connection: str, baud: int,
                 allow_arm: bool, allow_offboard: bool):
        try:
            from pymavlink import mavutil
        except ImportError as exc:
            raise SystemExit("pymavlink is required: python3 -m pip install pymavlink") from exc
        self.mavutil = mavutil
        self.cfg = cfg
        self.safety = SafetyGate(cfg.target, allow_arm, allow_offboard)
        self.master = mavutil.mavlink_connection(connection, baud=baud,
                                                  source_system=cfg.source_system,
                                                  source_component=cfg.source_component)
        self.udp = DatagramServer(cfg.bind_host, cfg.gateway_port, cfg.protocol_version, self._on_udp)
        self.sample = VehicleSample(source="mavlink")
        self.action = (0.0, 0.0, 0.0, 0.0)
        self.last_action_ns = 0
        self.pending_state_ack = -1
        self.last_command_seq = -1
        self.tx_seq = 0
        self.prestream = 0
        self.offboard_requested = False
        self.offboard_enabled = cfg.target == "sitl"
        self.last_control = 0.0
        self.q_ned_frd = np.array([1.0, 0.0, 0.0, 0.0])

    def run(self) -> None:
        print("Waiting for PX4 MAVLink heartbeat...")
        self.master.wait_heartbeat(timeout=15)
        print(f"MAVLink connected: system={self.master.target_system}, component={self.master.target_component}")
        self.master.mav.request_data_stream_send(
            self.master.target_system, self.master.target_component,
            self.mavutil.mavlink.MAV_DATA_STREAM_ALL, int(self.cfg.control_hz), 1,
        )
        period = 1.0 / self.cfg.control_hz
        try:
            while True:
                self.udp.poll()
                for _ in range(100):
                    msg = self.master.recv_match(blocking=False)
                    if msg is None:
                        break
                    self._on_mavlink(msg)
                now = time.monotonic()
                if now - self.last_control >= period:
                    self.last_control = now
                    self._control_tick()
                time.sleep(0.001)
        except KeyboardInterrupt:
            pass
        finally:
            self.udp.close()
            self.master.close()

    def _on_udp(self, msg) -> None:
        kind, seq = msg["type"], msg["seq"]
        if seq <= self.last_command_seq and kind not in {"hello", "state"}:
            self._ack(seq, "duplicate")
            return
        if kind == "action":
            self.action = validate_action(msg)
            self.last_action_ns = now_ns()
            self.last_command_seq = seq
            self.pending_state_ack = seq
        elif kind == "arm":
            self.safety.require_arm()
            self.last_command_seq = seq
            self._arm(True)
            self._ack(seq, "arm_requested")
        elif kind == "disarm":
            self.last_command_seq = seq
            self._arm(False)
            self._ack(seq, "disarm_requested")
        elif kind == "reset":
            raise PermissionError("MAVLink gateway cannot reset Isaac; use the ROS 2 gateway for SITL episodes")
        elif kind == "enable_offboard":
            self.safety.require_offboard()
            self.last_command_seq = seq
            self.offboard_enabled = True
            self._ack(seq, "offboard_enabled")
        elif kind == "disable_offboard":
            self.last_command_seq = seq
            self.offboard_enabled = False
            self.offboard_requested = False
            self.last_action_ns = 0
            self._ack(seq, "offboard_disabled")
        elif kind in {"hello", "state"}:
            if kind == "hello":
                self.last_command_seq = -1
            self._state(seq)
        else:
            raise ProtocolError(f"unsupported command: {kind}")

    def _on_mavlink(self, msg) -> None:
        kind = msg.get_type()
        if kind == "LOCAL_POSITION_NED":
            self.sample.position_enu = tuple(ned_to_enu((msg.x, msg.y, msg.z)))
            self.sample.velocity_enu = tuple(ned_to_enu((msg.vx, msg.vy, msg.vz)))
            self.sample.timestamp_ns = now_ns()
            self.sample.estimator_valid = True
            if self.pending_state_ack >= 0:
                seq, self.pending_state_ack = self.pending_state_ack, -1
                self._state(seq)
        elif kind == "ATTITUDE_QUATERNION":
            q = (msg.q1, msg.q2, msg.q3, msg.q4)
            self.q_ned_frd = np.asarray(q, dtype=float)
            self.sample.quaternion_enu_flu_wxyz = tuple(quat_ned_frd_to_enu_flu(q))
            self.sample.angular_velocity_flu = tuple(frd_to_flu((msg.rollspeed, msg.pitchspeed, msg.yawspeed)))
        elif kind == "HIGHRES_IMU":
            accel_ned = quat_wxyz_to_matrix(self.q_ned_frd) @ np.array(
                [msg.xacc, msg.yacc, msg.zacc]
            )
            self.sample.acceleration_enu = tuple(ned_to_enu(accel_ned))
        elif kind == "HEARTBEAT":
            self.sample.armed = bool(msg.base_mode & self.mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            self.sample.nav_state = int(msg.custom_mode)
        elif kind == "EXTENDED_SYS_STATE":
            landed_state = getattr(msg, "landed_state", 0)
            self.sample.landed = landed_state == self.mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND

    def _control_tick(self) -> None:
        if not self.last_action_ns:
            return
        if (now_ns() - self.last_action_ns) * 1e-9 > self.cfg.action_timeout_s:
            print("Action deadman expired; yielding to PX4 offboard-loss failsafe")
            self.last_action_ns = 0
            self.prestream = 0
            self.offboard_requested = False
            return
        a0, ar, ap, ay = self.action
        yaw = yaw_from_quat_wxyz(self.sample.quaternion_enu_flu_wxyz)
        q_enu = euler_zyx_to_quat_wxyz(self.cfg.max_roll_pitch_rad * ar,
                                       self.cfg.max_roll_pitch_rad * ap, yaw)
        q_ned = quat_enu_flu_to_ned_frd(q_enu)
        thrust = float(np.clip(self.cfg.hover_thrust * (1 + self.cfg.collective_span * a0), 0.05, 0.90))
        mask = (self.mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_ROLL_RATE_IGNORE |
                self.mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_PITCH_RATE_IGNORE)
        self.master.mav.set_attitude_target_send(
            int(time.monotonic() * 1000) & 0xFFFFFFFF,
            self.master.target_system,
            self.master.target_component,
            mask,
            [float(x) for x in q_ned],
            0.0, 0.0, float(-self.cfg.max_yaw_rate_rad_s * ay), thrust,
        )
        self.prestream += 1
        if (self.offboard_enabled and self.prestream >= self.cfg.offboard_prestream_count
                and not self.offboard_requested):
            self.master.set_mode(self.master.mode_mapping().get("OFFBOARD", 6))
            self.offboard_requested = True

    def _arm(self, arm: bool) -> None:
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            self.mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
            1.0 if arm else 0.0, 0, 0, 0, 0, 0, 0,
        )

    def _ack(self, ack_seq: int, status: str) -> None:
        self.tx_seq += 1
        self.udp.send({"v": self.cfg.protocol_version, "type": "ack", "seq": self.tx_seq,
                       "ack_seq": ack_seq, "time_ns": now_ns(), "status": status})

    def _state(self, ack_seq: int) -> None:
        self.tx_seq += 1
        if ((now_ns() - self.sample.timestamp_ns) * 1e-9 > self.cfg.state_timeout_s):
            self.sample.estimator_valid = False
        self.udp.send(self.sample.to_message(self.cfg.protocol_version, self.tx_seq, ack_seq))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MATLAB to PX4 MAVLink gateway")
    parser.add_argument("--config", required=True)
    parser.add_argument("--target", choices=("sitl", "hardware"), default="hardware")
    parser.add_argument("--device", required=True, help="serial device or pymavlink URL, e.g. udp:127.0.0.1:14550")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--allow-arm", action="store_true")
    parser.add_argument("--allow-offboard", action="store_true")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_gateway_config(Path(args.config), target=args.target)
    MavlinkGateway(cfg, args.device, args.baud, args.allow_arm, args.allow_offboard).run()


if __name__ == "__main__":
    main()
