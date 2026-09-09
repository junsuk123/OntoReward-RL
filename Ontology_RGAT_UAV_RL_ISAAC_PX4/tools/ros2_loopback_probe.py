#!/usr/bin/env python3
"""Publish mock PX4 telemetry and a mock deck, then verify what the gateway makes of it.

Checks the pad-relative contract end to end without Isaac: the state the client
receives must be the PX4 estimate minus the deck pose, a pad-frame goto must be
re-aimed at the live deck, and the energy budget must start charging once the
policy takes over.
"""

from __future__ import annotations

import json
import socket
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from px4_msgs.msg import (
    TrajectorySetpoint,
    VehicleAttitudeSetpoint,
    VehicleLocalPosition,
    VehicleOdometry,
)
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


# The deck the pad rides on, in world ENU, and the constant twist it drives with.
DECK_POSITION = np.array([5.0, -3.0, 0.32])
DECK_VELOCITY = np.array([0.5, 0.25, 0.0])
# PX4 NED (1, 2, -3) is world ENU (2, 1, 3); NED velocity (0.1, 0.2, 0.3) is
# world ENU (0.2, 0.1, -0.3).
VEHICLE_WORLD_ENU = np.array([2.0, 1.0, 3.0])
VEHICLE_WORLD_VELOCITY_ENU = np.array([0.2, 0.1, -0.3])
GOTO_PAD_OFFSET = np.array([1.0, 2.0, 4.0])


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
        # Without this the gateway has a moving-pad configuration and no deck to
        # measure against, and correctly refuses to call the state valid.
        self.deck_pub = self.create_publisher(Odometry, "/landing_pad/state/odom", 10)
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
        deck = Odometry()
        deck.header.stamp = self.get_clock().now().to_msg()
        deck.header.frame_id = "map"
        deck.child_frame_id = "landing_pad"
        deck.pose.pose.position.x = float(DECK_POSITION[0])
        deck.pose.pose.position.y = float(DECK_POSITION[1])
        deck.pose.pose.position.z = float(DECK_POSITION[2])
        deck.pose.pose.orientation.w = 1.0
        deck.twist.twist.linear.x = float(DECK_VELOCITY[0])
        deck.twist.twist.linear.y = float(DECK_VELOCITY[1])
        deck.twist.twist.linear.z = float(DECK_VELOCITY[2])
        self.deck_pub.publish(deck)


def main():
    rclpy.init()
    node = Probe()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.setblocking(False)
    destination = ("127.0.0.1", 14650)
    hello = {"v": 1, "type": "hello", "seq": 1, "time_ns": time.monotonic_ns()}
    goto = {"v": 1, "type": "goto", "seq": 2, "time_ns": time.monotonic_ns(),
            "position": GOTO_PAD_OFFSET.tolist(), "yaw": 0.0, "hold_s": 10.0,
            "frame": "pad"}
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
        assert action_state["position_frame"] == "pad", "gateway did not declare the pad frame"
        assert action_state["estimator_valid"] is True
        # The policy sees the deck subtracted out of the PX4 estimate.
        np.testing.assert_allclose(
            action_state["position"], VEHICLE_WORLD_ENU - DECK_POSITION, atol=1e-5)
        np.testing.assert_allclose(
            action_state["velocity"], VEHICLE_WORLD_VELOCITY_ENU - DECK_VELOCITY, atol=1e-5)
        np.testing.assert_allclose(
            action_state["world"]["position"], VEHICLE_WORLD_ENU, atol=1e-5)
        assert action_state["pad"]["valid"] is True, "deck state was not accepted"
        np.testing.assert_allclose(action_state["pad"]["velocity"], DECK_VELOCITY, atol=1e-5)
        assert node.trajectory is not None, "no TrajectorySetpoint during the climb"
        # A pad-frame goto is re-aimed at the live deck: world ENU
        # (5, -3, 0.32) + (1, 2, 4) = (6, -1, 4.32), which is NED (-1, 6, -4.32).
        world_target = DECK_POSITION + GOTO_PAD_OFFSET
        expected_ned = [world_target[1], world_target[0], -world_target[2]]
        np.testing.assert_allclose(node.trajectory.position, expected_ned, atol=1e-5)
        assert node.setpoint is not None, "no VehicleAttitudeSetpoint"
        assert -0.9 <= node.setpoint.thrust_body[2] <= -0.05
        # The seeded reserve is armed at handover, so by now energy is being spent.
        battery = action_state["battery"]
        assert battery["enabled"] is True, "battery model is not running"
        assert battery["hover_power_w"] > 0.0
        assert battery["depleted"] is False
        print("ROS2_LOOPBACK=PASS")
    finally:
        sock.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
