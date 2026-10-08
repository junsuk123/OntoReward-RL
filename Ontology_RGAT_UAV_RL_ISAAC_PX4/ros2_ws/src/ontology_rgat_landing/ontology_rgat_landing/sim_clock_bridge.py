"""Isaac ``simulation/clock`` (JSON String, sim seconds) -> ``/clock``.

The Isaac stage does not publish ``/clock``; its simulator time goes out as
``{"sim_time_s": ...}`` on ``<ns>/simulation/clock``, and the camera's capture
stamps are in that same timebase. The pipeline nodes run with
``use_sim_time:=true`` on this clock so that capture stamps, EKF receipt
stamps and decision times are all sim seconds. Time never goes backwards and
is never invented: a message older than the last one is dropped.
"""
from __future__ import annotations

import json
import math

from .common import run, topic


def main() -> None:
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rosgraph_msgs.msg import Clock
    from std_msgs.msg import String

    class SimClockBridge(Node):
        def __init__(self):
            super().__init__("sim_clock_bridge")
            ns = self.declare_parameter("namespace", "/landing_uav0").value
            self.last = -math.inf
            self.pub = self.create_publisher(Clock, "/clock", 10)
            self.create_subscription(String, topic(ns, "simulation/clock"),
                                     self.on_clock, qos_profile_sensor_data)

        def on_clock(self, msg):
            try:
                value = float(json.loads(msg.data)["sim_time_s"])
            except (ValueError, KeyError, TypeError):
                return
            if not math.isfinite(value) or value < 0 or value <= self.last:
                return
            self.last = value
            out = Clock()
            ns = int(round(value * 1e9))
            out.clock.sec, out.clock.nanosec = ns // 1_000_000_000, ns % 1_000_000_000
            self.pub.publish(out)

    run(SimClockBridge)


if __name__ == "__main__":
    main()
