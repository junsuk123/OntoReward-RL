#!/usr/bin/env python3
"""Record the minimal-observation ROS chain while something else flies.

Subscribes to every topic of the chain and writes one JSON line per message
(stamp, receipt time and the fields needed to compare against the in-process
computation), until SIGINT/SIGTERM. Read-only: it publishes nothing.

Run in a ROS 2 environment that can see the stack (rmw_fastrtps_cpp, px4_msgs
and ontology_rgat_interfaces sourced).
"""
from __future__ import annotations

import json
import signal
import sys


def main() -> int:
    out_path = sys.argv[1]
    ns = sys.argv[2] if len(sys.argv) > 2 else "/landing_uav0"
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy
    from geometry_msgs.msg import Vector3Stamped
    from nav_msgs.msg import Odometry
    from rosgraph_msgs.msg import Clock
    from std_msgs.msg import Float32
    from ontology_rgat_interfaces.msg import (LandingObservation, PadDetection,
                                              SafetyStatus, SemanticGraph)
    reliable = QoSProfile(depth=10, history=HistoryPolicy.KEEP_LAST,
                          reliability=ReliabilityPolicy.RELIABLE)
    out = open(out_path, "w", buffering=1)

    def stamp(header):
        return header.stamp.sec + 1e-9 * header.stamp.nanosec

    class Recorder(Node):
        def __init__(self):
            super().__init__("minimal_chain_recorder")
            self.sim_now = None
            spec = [
                (Clock, "/clock", qos_profile_sensor_data, self.on_clock),
                (PadDetection, f"{ns}/perception/pad_detection", reliable,
                 lambda m: self.write("pad_detection", stamp(m.header), detected=m.detected,
                                      position=[m.position.x, m.position.y, m.position.z],
                                      markers=m.marker_count)),
                (Odometry, f"{ns}/localization/odometry", qos_profile_sensor_data,
                 lambda m: self.write("odometry", stamp(m.header),
                                      valid=bool(m.pose.covariance[0] >= 0))),
                (LandingObservation, f"{ns}/observation", reliable,
                 lambda m: self.write("observation", stamp(m.header), vector=list(m.vector))),
                (SemanticGraph, f"{ns}/ontology/graph", reliable,
                 lambda m: self.write("graph", stamp(m.header), schema=m.schema_hash[:12],
                                      edge_weight_min=min(m.edge_weight) if m.edge_weight else None)),
                (Vector3Stamped, f"{ns}/policy/acceleration_command", reliable,
                 lambda m: self.write("command", stamp(m.header),
                                      vector=[m.vector.x, m.vector.y, m.vector.z])),
                (SafetyStatus, f"{ns}/safety/status", reliable,
                 lambda m: self.write("status", stamp(m.header), mode=m.mode,
                                      pad_class=m.pad_loss_class, reason=m.pad_lost_reason)),
                (Float32, f"{ns}/policy/relational_health", reliable,
                 lambda m: self.write("relational_health", None, value=m.data)),
            ]
            for msg_type, name, qos, callback in spec:
                self.create_subscription(msg_type, name, callback, qos)

        def on_clock(self, msg):
            self.sim_now = msg.clock.sec + 1e-9 * msg.clock.nanosec

        def write(self, kind, msg_stamp, **fields):
            out.write(json.dumps({"kind": kind, "stamp": msg_stamp,
                                  "received_sim": self.sim_now, **fields},
                                 default=lambda v: v.item() if hasattr(v, "item") else str(v)) + "\n")

    rclpy.init()
    node = Recorder()
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("flag", True))
    try:
        while rclpy.ok() and not stop["flag"]:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        out.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
