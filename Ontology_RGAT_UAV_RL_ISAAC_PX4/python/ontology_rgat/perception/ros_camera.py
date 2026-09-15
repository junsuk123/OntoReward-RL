"""Thread-safe ROS image boundary for the deployed benchmark actor.

The image topic is the whole deployed actor boundary.  The separate
pad-relative truth-pose topic subscribed below is ``training-label-only``
simulator geometry: it exists so keypoint supervision can project the known
pad landmarks for a rendered frame, and it is never routed into
:class:`~ontology_rgat.benchmarks.shin2026.ActorObservation`.
"""
from __future__ import annotations

from collections import deque
import threading
import time

import numpy as np


# The simulator publishes the rendered frame and its truth pose from the same
# tick with the same stamp, so the match is exact. Only DDS delivery order is
# unguaranteed, and that is a matter of waiting briefly rather than of
# accepting a neighbouring pose: at 2 m/s one 30 Hz frame of skew is 6.6 cm,
# which is several pixels of keypoint-label error near touchdown.
POSE_WAIT_TIMEOUT_S = 0.5


class LatestGrayscaleFrame:
    def __init__(self, expected_shape=(320, 512)):
        self.expected_shape = tuple(expected_shape)
        self._condition = threading.Condition()
        self._frame = None
        self._stamp_ns = 0
        self.subscription = None

    def attach(self, node, topic="/landing_uav0/perception/landing_camera/image_raw"):
        from sensor_msgs.msg import Image
        self.subscription = node.create_subscription(Image, topic, self.callback, 1)
        return self

    def callback(self, message) -> None:
        if str(message.encoding).lower() != "mono8":
            raise ValueError("benchmark actor camera must publish mono8")
        image = np.frombuffer(message.data, dtype=np.uint8).reshape(
            int(message.height), int(message.step))[:, :int(message.width)].copy()
        if image.shape != self.expected_shape:
            raise ValueError(f"actor camera expected {self.expected_shape}, got {image.shape}")
        stamp = getattr(message.header, "stamp", None)
        stamp_ns = (int(getattr(stamp, "sec", 0)) * 1_000_000_000
                    + int(getattr(stamp, "nanosec", 0)))
        with self._condition:
            self._frame, self._stamp_ns = image, stamp_ns
            self._condition.notify_all()

    def latest(self, timeout_s=1.0, after_stamp_ns: int | None = None):
        deadline = time.monotonic() + float(timeout_s)
        with self._condition:
            while (self._frame is None
                   or (after_stamp_ns is not None
                       and self._stamp_ns <= int(after_stamp_ns))):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("no Shin benchmark camera frame received")
                self._condition.wait(remaining)
            return self._frame.copy(), self._stamp_ns


class LatestPadRelativeTruthPose:
    """Training-label-only buffer of the simulator's pad-relative UAV pose.

    Never consumed by the actor, the reward or the online R-GAT: the only
    caller is keypoint-label generation, which projects the known landmarks
    through this pose.
    """

    def __init__(self, depth: int = 128):
        self._condition = threading.Condition()
        self._poses = deque(maxlen=int(depth))
        self.subscription = None

    def attach(self, node, topic="/landing_uav0/perception/pad_relative_truth_pose"):
        from geometry_msgs.msg import PoseStamped
        self.subscription = node.create_subscription(
            PoseStamped, topic, self.callback, 10)
        return self

    def callback(self, message) -> None:
        stamp = getattr(message.header, "stamp", None)
        stamp_ns = (int(getattr(stamp, "sec", 0)) * 1_000_000_000
                    + int(getattr(stamp, "nanosec", 0)))
        position = message.pose.position
        orientation = message.pose.orientation
        pose = np.array([position.x, position.y, position.z, orientation.w,
                         orientation.x, orientation.y, orientation.z],
                        dtype=float)
        if not np.isfinite(pose).all():
            return
        with self._condition:
            self._poses.append((stamp_ns, pose))
            self._condition.notify_all()

    def at(self, stamp_ns: int):
        """The pose stamped exactly ``stamp_ns``, or ``None``."""
        stamp_ns = int(stamp_ns)
        with self._condition:
            for stamp, pose in reversed(self._poses):
                if stamp == stamp_ns:
                    return pose.copy()
        return None

    def wait_for(self, stamp_ns: int, timeout_s: float = POSE_WAIT_TIMEOUT_S):
        """Block briefly for the pose belonging to exactly this frame.

        Returns ``None`` on timeout so the caller drops that frame rather than
        labelling it with a neighbouring tick's pose.
        """
        stamp_ns = int(stamp_ns)
        deadline = time.monotonic() + float(timeout_s)
        with self._condition:
            while True:
                for stamp, pose in reversed(self._poses):
                    if stamp == stamp_ns:
                        return pose.copy()
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(remaining)


class RosGrayscaleSource:
    """Own a ROS node/thread and return one fresh mono frame per control step.

    ``truth_pose_topic`` is optional and strictly separate: subscribing to it
    adds :meth:`labelled`, used only by offline keypoint calibration.  The
    deployed ``__call__`` boundary keeps returning pixels and nothing else.
    """

    def __init__(self, topic="/landing_uav0/perception/landing_camera/image_raw",
                 expected_shape=(320, 512), timeout_s=2.0,
                 node_name="shin2026_actor_camera",
                 truth_pose_topic=None):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor

        self.rclpy = rclpy
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init(args=None)
        self.node = rclpy.create_node(str(node_name))
        # rclpy.spin_once(node) uses a shared global executor. Calling it from
        # three camera threads races the executor's callback generator. Each
        # pair owns a small executor instead, while all nodes still share the
        # same ROS context and DDS participant.
        self.executor = SingleThreadedExecutor(context=self.node.context)
        self.executor.add_node(self.node)
        self.frames = LatestGrayscaleFrame(expected_shape).attach(self.node, topic)
        self.truth_poses = (
            LatestPadRelativeTruthPose().attach(self.node, truth_pose_topic)
            if truth_pose_topic else None)
        self.timeout_s = float(timeout_s)
        self.last_stamp_ns = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        from rclpy.executors import ExternalShutdownException

        while not self._stop.is_set() and self.rclpy.ok():
            try:
                self.executor.spin_once(timeout_sec=0.05)
            except ExternalShutdownException:
                # Another ROS owner (normally the gateway during Ctrl-C) may
                # close the shared context before this camera source.  That is
                # an orderly shutdown, not a camera-thread failure.
                return

    def __call__(self):
        frame, stamp = self.frames.latest(
            self.timeout_s, after_stamp_ns=self.last_stamp_ns)
        self.last_stamp_ns = stamp
        return frame

    def labelled(self):
        """One fresh frame with its training-only pad-relative truth pose.

        Returns ``(image, pose)`` or ``(image, None)`` when no pose sample is
        close enough in time; the label generator drops the unmatched frame
        rather than guessing a pose for it.
        """
        if self.truth_poses is None:
            raise RuntimeError(
                "this camera source was built without a truth-pose topic, so "
                "geometric keypoint labels cannot be produced")
        frame, stamp = self.frames.latest(
            self.timeout_s, after_stamp_ns=self.last_stamp_ns)
        self.last_stamp_ns = stamp
        return frame, self.truth_poses.wait_for(stamp)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.executor.remove_node(self.node)
        self.executor.shutdown(timeout_sec=1.0)
        self.node.destroy_node()
        if self._owns_context and self.rclpy.ok():
            self.rclpy.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
