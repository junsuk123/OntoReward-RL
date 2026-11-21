# Copyright (c) 2026 Jun Suk Kim (김준석)
# Maintainer: Jun Suk Kim (김준석) <junsuk55944587@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Demo-only RViz helper: turns the simulator's truth odometry into TF frames (map->uav,
map->pad) and two trajectory paths, so an RViz follower camera can stay on the vehicle.
It only subscribes to the evaluator-side truth topics and publishes /demo/* for RViz;
nothing on the control path reads anything it publishes.
"""
import rclpy
import rclpy.executors
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from tf2_ros import TransformBroadcaster

TOPICS = {"uav": "/landing_uav0/state/odom_truth", "pad": "/landing_pad/state/odom_truth"}
KEEP = 1500


class Relay(Node):
    def __init__(self):
        super().__init__("demo_tf_relay")
        self.tf = TransformBroadcaster(self)
        self.paths = {k: Path() for k in TOPICS}
        self.pubs = {k: self.create_publisher(Path, f"/demo/{k}_path", 10) for k in TOPICS}
        for name, topic in TOPICS.items():
            self.create_subscription(Odometry, topic, lambda m, n=name: self.on(n, m), 10)

    def on(self, name, msg):
        t = TransformStamped()
        t.header = msg.header
        t.header.frame_id = "map"
        t.child_frame_id = name
        p = msg.pose.pose
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = (
            p.position.x, p.position.y, p.position.z)
        t.transform.rotation = p.orientation
        self.tf.sendTransform(t)
        path = self.paths[name]
        path.header = t.header
        pose = PoseStamped(); pose.header = t.header; pose.pose = p
        path.poses.append(pose)
        del path.poses[:-KEEP]
        self.pubs[name].publish(path)


def main():
    rclpy.init()
    node = Relay()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
