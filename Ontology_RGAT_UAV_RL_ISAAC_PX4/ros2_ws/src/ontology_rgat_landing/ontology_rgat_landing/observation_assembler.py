"""PadDetection + EKF odometry -> LandingObservation at the decision rate."""
from __future__ import annotations

import numpy as np

from .common import (add_repo_paths, add_reset_service, observation_to_msg,
                     reliable_latest_qos, run, stamp_to_float, topic)


def main() -> None:
    add_repo_paths()
    from builtin_interfaces.msg import Time
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from nav_msgs.msg import Odometry
    from ontology_rgat_interfaces.msg import LandingObservation as ObservationMsg, PadDetection
    from ontology_rgat.minimal.constants import DEFAULT_CONSTANTS
    from ontology_rgat.minimal.observation import (ObservationAssembler, OwnState,
                                                    PadDetectionSample)

    class ObservationAssemblerNode(Node):
        def __init__(self):
            super().__init__("observation_assembler")
            ns = self.declare_parameter("namespace", "/landing_uav0").value
            rate = self.declare_parameter("rate_hz", 1.0 / DEFAULT_CONSTANTS.policy_dt_s).value
            self.assembler = ObservationAssembler()
            self.pub = self.create_publisher(
                ObservationMsg, topic(ns, "observation"), reliable_latest_qos())
            self.create_subscription(Odometry, topic(ns, "localization/odometry"),
                                     self.on_odometry, qos_profile_sensor_data)
            self.create_subscription(PadDetection, topic(ns, "perception/pad_detection"),
                                     self.on_pad, reliable_latest_qos())
            self.create_timer(1.0 / float(rate), self.on_timer)
            add_reset_service(self, ns, self.reset)

        def reset(self):
            # Keep the odometry stream; drop the previous episode's pad.
            self.assembler = ObservationAssembler(odometry=self.assembler.odometry)

        def on_odometry(self, msg):
            p, v, q = msg.pose.pose.position, msg.twist.twist.linear, msg.pose.pose.orientation
            self.assembler.on_odometry(OwnState(
                stamp_to_float(msg.header.stamp), np.array([p.x, p.y, p.z]),
                np.array([v.x, v.y, v.z]), np.array([q.w, q.x, q.y, q.z]),
                msg.pose.covariance[0] >= 0.0))

        def on_pad(self, msg):
            self.assembler.on_pad_detection(PadDetectionSample(
                stamp_to_float(msg.header.stamp), bool(msg.detected),
                np.array([msg.position.x, msg.position.y, msg.position.z])))

        def on_timer(self):
            now = self.get_clock().now().nanoseconds * 1e-9
            obs = self.assembler.assemble(now)
            if obs is not None:
                self.pub.publish(observation_to_msg(obs, ObservationMsg, Time))

    run(ObservationAssemblerNode)


if __name__ == "__main__":
    main()
