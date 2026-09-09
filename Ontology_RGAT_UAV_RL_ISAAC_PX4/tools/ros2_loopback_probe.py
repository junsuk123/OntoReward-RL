#!/usr/bin/env python3
"""Publish mock PX4 telemetry and verify UDP state, attitude and climb setpoints."""

from __future__ import annotations

import json
import socket
import time

import numpy as np
import rclpy
from px4_msgs.msg import (
    TrajectorySetpoint,
    VehicleAttitudeSetpoint,
    VehicleLocalPosition,
    VehicleOdometry,
)
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


class Probe(Node):
    def __init__(self):
        super().__init__("ontology_rgat_loopback_probe")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.odom_pub = self.create_publisher(VehicleOdometry, "/fmu/out/vehicle_odometry", qos)
        self.local_pub = self.create_publisher(
            VehicleLocalPosition, "/fmu/out/vehicle_local_position", qos)
        self.setpoint = None
        self.trajectory = None
        self.create_subscription(VehicleAttitudeSetpoint, "/fmu/in/vehicle_attitude_setpoint",
                                 self._on_setpoint, 10)
        self.create_subscription(TrajectorySetpoint, "/fmu/in/trajectory_setpoint",
                                 self._on_trajectory, 10)

    def _on_setpoint(self, msg):
        self.setpoint = msg

    def _on_trajectory(self, msg):
        self.trajectory = msg

    def publish_odometry(self):
        msg = VehicleOdometry()
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        msg.pose_frame = VehicleOdometry.POSE_FRAME_NED
        msg.position = [1.0, 2.0, -3.0]
        msg.q = [1.0, 0.0, 0.0, 0.0]
        msg.velocity_frame = VehicleOdometry.VELOCITY_FRAME_NED
        msg.velocity = [0.1, 0.2, 0.3]
        msg.angular_velocity = [0.01, 0.02, 0.03]
        self.odom_pub.publish(msg)
        # The gateway only trusts a sample once PX4 says its EKF has converged.
        local = VehicleLocalPosition()
        local.timestamp = msg.timestamp
        local.xy_valid = True
        local.z_valid = True
        local.v_xy_valid = True
        local.v_z_valid = True
        local.heading_good_for_control = True
        self.local_pub.publish(local)


def main():
    rclpy.init()
    node = Probe()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.setblocking(False)
    destination = ("127.0.0.1", 14650)
    hello = {"v": 1, "type": "hello", "seq": 1, "time_ns": time.monotonic_ns()}
    goto = {"v": 1, "type": "goto", "seq": 2, "time_ns": time.monotonic_ns(),
            "position": [1.0, 2.0, 4.0], "yaw": 0.0, "hold_s": 10.0}
    action = {"v": 1, "type": "action", "seq": 3, "time_ns": time.monotonic_ns(),
              "action": [0.1, -0.2, 0.3, -0.4]}
    sock.sendto(json.dumps(hello).encode(), destination)
    sent = set()
    goto_acked = False
    action_state = None
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        node.publish_odometry()
        rclpy.spin_once(node, timeout_sec=0.02)
        try:
            reply = json.loads(sock.recvfrom(32768)[0].decode())
            if reply.get("ack_seq") == 1 and 2 not in sent:
                sock.sendto(json.dumps(goto).encode(), destination)
                sent.add(2)
            elif reply.get("ack_seq") == 2:
                goto_acked = True
            elif reply.get("ack_seq") == 3 and reply.get("type") == "state":
                action_state = reply
        except BlockingIOError:
            pass
        # The climb setpoint only appears on the next control tick, so hand
        # over to the action phase from the loop rather than from a datagram.
        if goto_acked and node.trajectory is not None and 3 not in sent:
            sock.sendto(json.dumps(action).encode(), destination)
            sent.add(3)
        if action_state is not None and node.setpoint is not None:
            break
    try:
        assert action_state is not None, "no action-correlated state reply"
        assert action_state["position"] == [2.0, 1.0, 3.0]
        assert action_state["estimator_valid"] is True
        assert node.trajectory is not None, "no TrajectorySetpoint during the climb"
        # ENU (1, 2, 4) is NED (2, 1, -4).
        np.testing.assert_allclose(node.trajectory.position, [2.0, 1.0, -4.0], atol=1e-5)
        assert node.setpoint is not None, "no VehicleAttitudeSetpoint"
        assert -0.9 <= node.setpoint.thrust_body[2] <= -0.05
        print("ROS2_LOOPBACK=PASS")
    finally:
        sock.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
