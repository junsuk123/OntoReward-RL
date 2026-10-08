"""Policy command + LandingObservation -> supervised ENU acceleration + SafetyStatus.

Reads the minimal observation only, never the graph. Each command is paired
with the observation of the SAME decision time (the policy copies the
observation stamp through the graph), so the supervisor judges the state the
policy saw. An observation whose command has not arrived within
``max_command_age_s`` is stepped with a zero request (the supervisor's laws
still apply), so a stalled policy never leaves a stale acceleration in force.
Every observation is stepped exactly once, which the pad memory's dead
reckoning relies on.

This node publishes the supervised setpoint on ``safety/acceleration_setpoint``.
It does not talk to PX4 itself: ``tools/minimal_isaac_flight.py --control ros``
forwards that setpoint through the existing gateway, which keeps entry,
OFFBOARD handover and confirmed cleanup.
"""
from __future__ import annotations

import numpy as np

from .common import (add_repo_paths, add_reset_service, float_to_stamp,
                     msg_to_observation, reliable_latest_qos, run, stamp_to_float,
                     topic)


def main() -> None:
    add_repo_paths()
    from builtin_interfaces.msg import Time
    from rclpy.node import Node
    from geometry_msgs.msg import Vector3Stamped
    from ontology_rgat_interfaces.msg import LandingObservation as ObservationMsg, SafetyStatus
    from ontology_rgat.minimal.supervisor import MinimalSafetySupervisor

    class SafetySupervisorNode(Node):
        def __init__(self):
            super().__init__("safety_supervisor")
            ns = self.declare_parameter("namespace", "/landing_uav0").value
            self.max_age = float(self.declare_parameter("max_command_age_s", 0.25).value)
            self.supervisor = MinimalSafetySupervisor()
            self.pending = {}                # observation stamp -> observation
            self.setpoint_pub = self.create_publisher(
                Vector3Stamped, topic(ns, "safety/acceleration_setpoint"), reliable_latest_qos())
            self.status_pub = self.create_publisher(
                SafetyStatus, topic(ns, "safety/status"), reliable_latest_qos())
            self.create_subscription(Vector3Stamped, topic(ns, "policy/acceleration_command"),
                                     self.on_command, reliable_latest_qos())
            self.create_subscription(ObservationMsg, topic(ns, "observation"),
                                     self.on_observation, reliable_latest_qos())
            self.create_timer(0.02, self.on_watchdog)
            add_reset_service(self, ns, self.reset)

        def reset(self):
            self.supervisor.reset()
            self.pending.clear()

        def on_observation(self, msg):
            obs = msg_to_observation(msg)
            self.pending[obs.stamp] = obs

        def on_command(self, msg):
            stamp = stamp_to_float(msg.header.stamp)
            match = min(self.pending, key=lambda t: abs(t - stamp), default=None)
            if match is None or abs(match - stamp) > 1e-6:
                return                       # command for an observation already handled
            self.flush_before(match)
            self.decide(self.pending.pop(match),
                        np.array([msg.vector.x, msg.vector.y, msg.vector.z]))

        def on_watchdog(self):
            now = self.get_clock().now().nanoseconds * 1e-9
            for stamp in sorted(self.pending):
                if now - stamp > self.max_age:
                    self.decide(self.pending.pop(stamp), np.zeros(3))

        def flush_before(self, stamp):
            for older in sorted(t for t in self.pending if t < stamp):
                self.decide(self.pending.pop(older), np.zeros(3))

        def decide(self, obs, requested):
            decision = self.supervisor.step(obs, requested)
            stamp = float_to_stamp(decision.stamp, Time)
            setpoint = Vector3Stamped()
            setpoint.header.stamp, setpoint.header.frame_id = stamp, "map"
            setpoint.vector.x, setpoint.vector.y, setpoint.vector.z = map(float, decision.applied)
            self.setpoint_pub.publish(setpoint)
            status = SafetyStatus()
            status.header.stamp, status.header.frame_id = stamp, "map"
            status.mode = decision.mode
            status.pad_loss_class = decision.assessment.pad_class
            status.pad_lost_reason = decision.assessment.reason
            r, a = status.requested_acceleration, status.applied_acceleration
            r.x, r.y, r.z = map(float, decision.requested)
            a.x, a.y, a.z = map(float, decision.applied)
            status.intervened = list(decision.intervened)
            self.status_pub.publish(status)

    run(SafetySupervisorNode)


if __name__ == "__main__":
    main()
