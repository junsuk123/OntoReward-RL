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

    def latest(self, timeout_s=1.0):
        deadline = time.monotonic() + float(timeout_s)
        with self._condition:
            while self._frame is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("no Shin benchmark camera frame received")
                self._condition.wait(remaining)
            return self._frame.copy(), self._stamp_ns
