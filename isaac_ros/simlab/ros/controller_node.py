"""ROS 2 node that runs a simlab algorithm against the Isaac Sim scene.

Subscribes
    /odom       nav_msgs/Odometry       UGV pose and velocity
    /tf         via tf2                 person_NN frames

Publishes
    /cmd_vel            geometry_msgs/Twist            drive command
    /simlab/markers     visualization_msgs/MarkerArray people + robot, for RViz

Run it with the system ROS 2 Python:

    python3 -m simlab.ros.controller_node --ros-args -p config:=configs/default.yaml
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import ColorRGBA
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from simlab.algorithms import DriveCommand, Observation, build_controller
from simlab.config import load_config


def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class ControllerNode(Node):
    """Bridges ROS 2 topics to a :class:`simlab.algorithms.DriveController`."""

    def __init__(self) -> None:
        super().__init__("simlab_controller")

        self.declare_parameter("config", "configs/default.yaml")
        self.declare_parameter("rate_hz", 30.0)
        self.declare_parameter("publish_markers", True)

        config_path = self.get_parameter("config").get_parameter_value().string_value
        cfg = load_config(config_path)
        self.ros_cfg = cfg.ros2
        self.people_count = len(cfg.people.active_agents())

        self.controller = build_controller(
            cfg.ugv.controller.name, cfg.ugv.controller.params
        )
        self.controller.reset()

        self._odom: Optional[Odometry] = None
        self._start_time: Optional[float] = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(Odometry, self.ros_cfg.odom_topic, self._on_odom, sensor_qos)
        self.cmd_pub = self.create_publisher(Twist, self.ros_cfg.cmd_vel_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, "simlab/markers", 1)

        rate = self.get_parameter("rate_hz").get_parameter_value().double_value
        self.create_timer(1.0 / rate, self._on_tick)

        self.get_logger().info(
            f"controller='{self.controller.name}' people={self.people_count} "
            f"sub={self.ros_cfg.odom_topic} pub={self.ros_cfg.cmd_vel_topic} @{rate:g}Hz"
        )

    # -- callbacks ---------------------------------------------------------
    def _on_odom(self, msg: Odometry) -> None:
        self._odom = msg

    def _elapsed(self) -> float:
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._start_time is None:
            self._start_time = now
        return now - self._start_time

    def _people_positions(self) -> List[Tuple[float, float]]:
        """Look up each person's frame in the odom frame. Missing frames are skipped."""
        positions: List[Tuple[float, float]] = []
        for index in range(self.people_count):
            frame = self.ros_cfg.person_frame(index)
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.ros_cfg.odom_frame, frame, rclpy.time.Time()
                )
            except Exception:
                continue
            positions.append(
                (tf.transform.translation.x, tf.transform.translation.y)
            )
        return positions

    def _on_tick(self) -> None:
        if self._odom is None:
            return  # nothing from the simulator yet

        pose = self._odom.pose.pose
        orientation = pose.orientation
        people = self._people_positions()
        obs = Observation(
            t=self._elapsed(),
            robot_xy=(pose.position.x, pose.position.y),
            robot_yaw=_yaw_from_quat(
                orientation.x, orientation.y, orientation.z, orientation.w
            ),
            people_xy=tuple(people),
        )

        command: DriveCommand = self.controller.step(obs)
        twist = Twist()
        twist.linear.x = float(command.linear)
        twist.angular.z = float(command.angular)
        self.cmd_pub.publish(twist)

        if self.get_parameter("publish_markers").get_parameter_value().bool_value:
            self.marker_pub.publish(self._markers(obs))

    # -- visualization -----------------------------------------------------
    def _markers(self, obs: Observation) -> MarkerArray:
        array = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        for index, (x, y) in enumerate(obs.people_xy):
            marker = Marker()
            marker.header.frame_id = self.ros_cfg.odom_frame
            marker.header.stamp = stamp
            marker.ns = "people"
            marker.id = index
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.pose.position.z = 0.85
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = 0.5
            marker.scale.z = 1.7
            marker.color = ColorRGBA(r=0.95, g=0.55, b=0.15, a=0.85)
            array.markers.append(marker)

        goal = Marker()
        goal.header.frame_id = self.ros_cfg.base_frame
        goal.header.stamp = stamp
        goal.ns = "robot"
        goal.id = 0
        goal.type = Marker.ARROW
        goal.action = Marker.ADD
        goal.pose.orientation.w = 1.0
        goal.scale.x = 0.8
        goal.scale.y = 0.12
        goal.scale.z = 0.12
        goal.color = ColorRGBA(r=0.2, g=0.7, b=1.0, a=0.9)
        array.markers.append(goal)
        return array

    def stop_robot(self) -> None:
        self.cmd_pub.publish(Twist())


def main(argv: Optional[List[str]] = None) -> int:
    rclpy.init(args=argv)
    node = ControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
