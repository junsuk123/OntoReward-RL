"""camera -> ArUco -> PnP -> PadDetection (board origin in body FLU).

Reuses ``isaac_sim/marker_vision.MarkerPoseEstimator`` (OpenCV only, no Isaac
import), the same solver the Isaac stage publishes ``uav_pose_in_pad`` with.
Intrinsics come from ``camera_info`` when ``use_camera_info`` is true (real
cameras). The Isaac stage publishes no ``camera_info``, so by default they are
built from the profile's resolution and horizontal FOV (square pixels, centred
principal point) -- the same ``intrinsics_from_fov`` the stage renders with.
The board, dictionary and mount always come from the deployment profile.
"""
from __future__ import annotations

import numpy as np

from .common import add_repo_paths, reliable_latest_qos, run, topic

ENCODINGS = {"mono8": (1, None), "rgb8": (3, None), "bgr8": (3, "bgr"),
             "rgba8": (4, None), "bgra8": (4, "bgr")}


def image_to_array(msg) -> np.ndarray:
    if msg.encoding not in ENCODINGS:
        raise ValueError(f"unsupported image encoding {msg.encoding!r}")
    channels, order = ENCODINGS[msg.encoding]
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    rows = data.reshape(msg.height, msg.step)[:, :msg.width * channels]
    image = rows.reshape(msg.height, msg.width, channels) if channels > 1 else rows
    if order == "bgr":
        image = image[:, :, [2, 1, 0]]
    return np.ascontiguousarray(image)


def board_origin_in_body(observation) -> np.ndarray:
    """Invert the solver's body-in-pad pose: pad origin seen from the body."""
    from ontology_rgat.minimal.observation import quat_wxyz_to_matrix
    r_pad_from_body = quat_wxyz_to_matrix(observation.quaternion_pad_flu_wxyz)
    return r_pad_from_body.T @ (-np.asarray(observation.position_pad_enu, float))


def main() -> None:
    root = add_repo_paths()
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image
    from ontology_rgat_interfaces.msg import PadDetection
    from config_loader import load_config
    from marker_vision import MarkerBoard, MarkerPoseEstimator, intrinsics_from_fov

    class ArucoPadDetector(Node):
        def __init__(self):
            super().__init__("aruco_pad_detector")
            ns = self.declare_parameter("namespace", "/landing_uav0").value
            profile = (self.declare_parameter("profile", "").value
                       or str(root / "config/spatial-isaac-system-v12.yaml"))
            vision = load_config(profile)["vision"]
            self.board = MarkerBoard.from_config(vision["board"])
            self.dictionary = vision.get("dictionary", "DICT_4X4_50")
            self.mount = vision["camera"]["mount_translation_flu_m"]
            self.quality = (float(vision.get("quality_reprojection_px", 3.0)),
                            float(vision.get("quality_full_scale_px", 120.0)))
            self.estimator = None
            if not self.declare_parameter("use_camera_info", False).value:
                width, height = vision["camera"]["resolution"]
                self.build(intrinsics_from_fov(
                    int(width), int(height), float(vision["camera"]["horizontal_fov_deg"])), None)
            self.pub = self.create_publisher(
                PadDetection, topic(ns, "perception/pad_detection"), reliable_latest_qos())
            self.create_subscription(
                CameraInfo, topic(ns, "perception/landing_camera/camera_info"),
                self.on_info, qos_profile_sensor_data)
            self.create_subscription(
                Image, topic(ns, "perception/landing_camera/image_raw"),
                self.on_image, qos_profile_sensor_data)
            self.get_logger().info(
                f"board {len(self.board)} markers ({self.dictionary}) from {profile}")

        def on_info(self, msg):
            if self.estimator is not None:
                return
            self.build(np.array(msg.k, dtype=float).reshape(3, 3),
                       np.array(msg.d, dtype=float) if len(msg.d) else None)

        def build(self, k, d):
            self.estimator = MarkerPoseEstimator(
                self.board, k, self.mount, distortion=d, dictionary=self.dictionary,
                quality_reprojection_px=self.quality[0],
                quality_full_scale_px=self.quality[1])
            self.get_logger().info(f"intrinsics fx={k[0, 0]:.1f}; detector ready")

        def on_image(self, msg):
            if self.estimator is None:
                return
            observation = self.estimator.detect(image_to_array(msg))
            out = PadDetection()
            out.header.stamp = msg.header.stamp          # capture time
            out.header.frame_id = "base_link"
            out.detected = bool(observation.detected)
            if observation.detected:
                p = board_origin_in_body(observation)
                out.position.x, out.position.y, out.position.z = map(float, p)
                out.marker_count = len(observation.marker_ids)
                out.reprojection_error_px = float(observation.reprojection_px)
            self.pub.publish(out)

    run(ArucoPadDetector)


if __name__ == "__main__":
    main()
