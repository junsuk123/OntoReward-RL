"""Thread-safe ROS image boundary for the deployed benchmark actor."""
from __future__ import annotations

import threading
import time

import numpy as np


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


class RosGrayscaleSource:
    """Own a ROS node/thread and return one fresh mono frame per control step."""

    def __init__(self, topic="/landing_uav0/perception/landing_camera/image_raw",
                 expected_shape=(320, 512), timeout_s=2.0):
        import rclpy

        self.rclpy = rclpy
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init(args=None)
        self.node = rclpy.create_node("shin2026_actor_camera")
        self.frames = LatestGrayscaleFrame(expected_shape).attach(self.node, topic)
        self.timeout_s = float(timeout_s)
        self.last_stamp_ns = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        while not self._stop.is_set() and self.rclpy.ok():
            self.rclpy.spin_once(self.node, timeout_sec=0.05)

    def __call__(self):
        frame, stamp = self.frames.latest(
            self.timeout_s, after_stamp_ns=self.last_stamp_ns)
        self.last_stamp_ns = stamp
        return frame

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.node.destroy_node()
        if self._owns_context and self.rclpy.ok():
            self.rclpy.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
