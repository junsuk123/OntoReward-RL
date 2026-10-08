"""PX4 EKF2 (IMU + GNSS fused) -> nav_msgs/Odometry in ENU / FLU.

PX4 already fuses the IMU and GNSS; this node only changes frames. Position
and velocity come from ``vehicle_local_position`` (the EKF state, never a raw
fix), attitude from ``vehicle_attitude``. ``header.stamp`` is the ROS clock at
receipt, the clock every other node in the pipeline uses.
"""
from __future__ import annotations

import numpy as np

from .common import add_repo_paths, run, topic


def ned_to_enu(x, y, z) -> tuple[float, float, float]:
    return float(y), float(x), -float(z)


def quat_ned_frd_to_enu_flu(q_wxyz) -> np.ndarray:
    from ontology_rgat.minimal.observation import quat_wxyz_to_matrix
    r_ned_frd = quat_wxyz_to_matrix(q_wxyz)
    ned_to_enu_m = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float)
    frd_to_flu = np.diag([1.0, -1.0, -1.0])
    r = ned_to_enu_m @ r_ned_frd @ frd_to_flu
    return _matrix_to_quat(r)


def _matrix_to_quat(r: np.ndarray) -> np.ndarray:
    t = np.trace(r)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k]) * 2
        q = [0.0, 0.0, 0.0, 0.0]
        q[0] = (r[k, j] - r[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (r[j, i] + r[i, j]) / s
        q[1 + k] = (r[k, i] + r[i, k]) / s
    q = np.asarray(q, dtype=float)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def main() -> None:
    add_repo_paths()
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from nav_msgs.msg import Odometry
    from px4_msgs.msg import VehicleAttitude, VehicleLocalPosition

    class Px4LocalizationBridge(Node):
        def __init__(self):
            super().__init__("px4_localization_bridge")
            ns = self.declare_parameter("namespace", "/landing_uav0").value
            fmu = self.declare_parameter("px4_namespace", "/fmu").value
            self.quat = None
            self.pub = self.create_publisher(
                Odometry, topic(ns, "localization/odometry"), qos_profile_sensor_data)
            self.create_subscription(VehicleAttitude, topic(fmu, "out/vehicle_attitude"),
                                     self.on_attitude, qos_profile_sensor_data)
            self.create_subscription(VehicleLocalPosition,
                                     topic(fmu, "out/vehicle_local_position"),
                                     self.on_position, qos_profile_sensor_data)

        def on_attitude(self, msg):
            self.quat = quat_ned_frd_to_enu_flu(msg.q)

        def on_position(self, msg):
            if self.quat is None:
                return
            out = Odometry()
            out.header.stamp = self.get_clock().now().to_msg()
            out.header.frame_id, out.child_frame_id = "map", "base_link"
            p, v = out.pose.pose.position, out.twist.twist.linear
            p.x, p.y, p.z = ned_to_enu(msg.x, msg.y, msg.z)
            v.x, v.y, v.z = ned_to_enu(msg.vx, msg.vy, msg.vz)
            o = out.pose.pose.orientation
            o.w, o.x, o.y, o.z = map(float, self.quat)
            valid = msg.xy_valid and msg.z_valid and msg.v_xy_valid and msg.v_z_valid
            # Covariance carries validity: 0 on the diagonal is a valid
            # solution, -1 (ROS convention for "unknown") is not.
            out.pose.covariance[0] = 0.0 if valid else -1.0
            self.pub.publish(out)

    run(Px4LocalizationBridge)


if __name__ == "__main__":
    main()
