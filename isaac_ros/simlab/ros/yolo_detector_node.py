"""Real RGB YOLO inference and annotated ROS image publication."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import torch
import yaml
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
from ultralytics import YOLO

from simlab.config import load_config
from simlab.ml.lineage import active_model_for_signature, drone_model_signature
from simlab.sim.assets import drone_spec


def image_to_bgr(msg: Image) -> np.ndarray:
    channels = 4 if msg.encoding.lower() in {"rgba8", "bgra8"} else 3
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    image = raw[:, : msg.width * channels].reshape(msg.height, msg.width, channels)
    encoding = msg.encoding.lower()
    if encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image[:, :, :3].copy()


def bgr_message(image: np.ndarray, source: Image) -> Image:
    output = Image()
    output.header = source.header
    output.height, output.width = image.shape[:2]
    output.encoding = "bgr8"
    output.is_bigendian = False
    output.step = output.width * 3
    output.data = np.ascontiguousarray(image).tobytes()
    return output


class YoloDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("simlab_yolo_detector")
        self.declare_parameter("perception_config", "configs/perception.yaml")
        self.declare_parameter("scene_config", "configs/default.yaml")
        self.declare_parameter("model", "")
        config_path = Path(str(self.get_parameter("perception_config").value))
        self.cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.scene = load_config(str(self.get_parameter("scene_config").value))
        detector = self.cfg["detector"]
        requested = str(self.get_parameter("model").value).strip()
        signature = drone_model_signature(self.scene)
        active = active_model_for_signature(Path(self.cfg["artifacts_root"]), signature) or ""
        self.model_name = requested or active or detector["base_model"]
        self.model = YOLO(self.model_name)
        model_names = self.model.names
        if isinstance(model_names, list):
            model_names = dict(enumerate(model_names))
        self.drone_class_ids = {
            int(class_id)
            for class_id, name in model_names.items()
            if str(name).strip().lower() == "drone"
        }
        self.learned_drone_detector = bool(self.drone_class_ids)
        self.conf = float(detector["confidence"])
        self.iou = float(detector["iou"])
        self.imgsz = int(detector["image_size"])
        self.device = detector.get("device", 0) if torch.cuda.is_available() else "cpu"
        self.quantize = 16 if detector.get("half", True) and torch.cuda.is_available() else None
        self.period = 1.0 / float(detector.get("max_rate_hz", 10.0))
        self.last_run = {"front_near": 0.0, "front_far": 0.0}
        self.busy = False
        self.positions = {}
        self.roster = {}
        for team_name, team in (
            ("friendly", self.scene.drones.friendly),
            ("enemy", self.scene.drones.enemy),
        ):
            prefix = "".join(ch if ch.isalnum() else "_" for ch in team.label).strip("_")
            diameter = drone_spec(team.model).nominal_width_m * team.asset_scale
            for index in range(team.count):
                self.roster[f"{prefix}_{index + 1:02d}"] = (team_name, diameter)

        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(TFMessage, "tf", self.on_tf, 20)
        self.annotated_publishers = {}
        for camera in self.last_run:
            self.annotated_publishers[camera] = self.create_publisher(
                Image, f"simlab/{camera}/detections", qos
            )
            self.create_subscription(
                Image,
                f"simlab/{camera}/image",
                lambda msg, name=camera: self.on_image(name, msg),
                qos,
            )
        self.detection_pub = self.create_publisher(String, "simlab/yolo/detections", 10)
        self.status_pub = self.create_publisher(String, "simlab/yolo/status", 10)
        self.status_pub.publish(
            String(
                data=json.dumps(
                    {
                        "state": "ready",
                        "model": self.model_name,
                        "mode": (
                            "learned_drone_only"
                            if self.learned_drone_detector
                            else "simulator_gt_bootstrap"
                        ),
                    }
                )
            )
        )
        self.get_logger().info(
            f"YOLO ready: model={self.model_name}, device={self.device}, imgsz={self.imgsz}"
        )

    def on_tf(self, msg: TFMessage) -> None:
        for transform in msg.transforms:
            if transform.child_frame_id in self.roster:
                p = transform.transform.translation
                self.positions[transform.child_frame_id] = (p.x, p.y, p.z)

    def bootstrap_boxes(self, camera: str, width: int, height: int):
        """Project simulator truth while no learned drone detector is accepted."""
        front = self.scene.cameras.front
        forward = np.asarray(front.view_direction, dtype=float)
        forward /= np.linalg.norm(forward)
        position = np.asarray(front.near_position, dtype=float)
        if camera == "front_far":
            baseline = np.asarray(front.baseline_direction, dtype=float)
            baseline /= np.linalg.norm(baseline)
            position = position + baseline * front.separation_m
        right0 = np.cross(forward, np.asarray((0.0, 0.0, 1.0)))
        right0 /= np.linalg.norm(right0)
        up0 = np.cross(right0, forward)
        angle = math.radians(front.image_rotation_clockwise_deg - 90.0)
        right = math.cos(angle) * right0 + math.sin(angle) * up0
        up = -math.sin(angle) * right0 + math.cos(angle) * up0
        optics = front.optics
        fx = width * optics.focal_length_mm / optics.horizontal_aperture_mm
        fy = fx
        boxes = []
        for drone_id, point in self.positions.items():
            relative = np.asarray(point, dtype=float) - position
            depth = float(np.dot(relative, forward))
            if depth <= optics.clipping_range[0]:
                continue
            u = width * 0.5 + fx * float(np.dot(relative, right)) / depth
            v = height * 0.5 - fy * float(np.dot(relative, up)) / depth
            team, diameter = self.roster[drone_id]
            half = max(7.0, fx * diameter * 0.70 / depth)
            xyxy = [max(0.0, u - half), max(0.0, v - half), min(width - 1.0, u + half), min(height - 1.0, v + half)]
            if 0 <= u < width and 0 <= v < height and xyxy[2] > xyxy[0] and xyxy[3] > xyxy[1]:
                boxes.append({
                    "xyxy": [round(value, 2) for value in xyxy],
                    "confidence": 1.0,
                    "class_id": 0,
                    "class_name": "drone",
                    "track_id": drone_id,
                    "team": team,
                    "source": "simulator_gt_bootstrap",
                })
        return boxes

    def on_image(self, camera: str, msg: Image) -> None:
        now = time.monotonic()
        if self.busy or now - self.last_run[camera] < self.period:
            return
        self.last_run[camera] = now
        self.busy = True
        try:
            frame = image_to_bgr(msg)
            boxes = []
            if self.learned_drone_detector:
                result = self.model.predict(
                    frame,
                    conf=self.conf,
                    iou=self.iou,
                    imgsz=self.imgsz,
                    device=self.device,
                    quantize=self.quantize,
                    classes=sorted(self.drone_class_ids),
                    verbose=False,
                )[0]
            else:
                result = None
            if result is not None and result.boxes is not None:
                for xyxy, score, class_id in zip(
                    result.boxes.xyxy.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(),
                    result.boxes.cls.cpu().tolist(),
                ):
                    class_id = int(class_id)
                    if class_id not in self.drone_class_ids:
                        continue
                    boxes.append(
                        {
                            "xyxy": [round(float(v), 2) for v in xyxy],
                            "confidence": round(float(score), 4),
                            "class_id": class_id,
                            "class_name": "drone",
                            "source": "learned_yolo",
                        }
                    )
            annotated = frame.copy()
            if not self.learned_drone_detector:
                boxes = self.bootstrap_boxes(camera, msg.width, msg.height)
            for box in boxes:
                x1, y1, x2, y2 = (int(value) for value in box["xyxy"])
                team = box.get("team")
                color = (255, 100, 20) if team == "friendly" else (
                    (20, 20, 255) if team == "enemy" else (40, 220, 40)
                )
                label = box.get("track_id") or "drone"
                if box.get("source") == "learned_yolo":
                    label = f"drone {box['confidence']:.2f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            payload = {
                "camera": camera,
                "stamp": {"sec": msg.header.stamp.sec, "nanosec": msg.header.stamp.nanosec},
                "model": self.model_name,
                "boxes": boxes,
            }
            self.detection_pub.publish(String(data=json.dumps(payload, sort_keys=True)))
            self.annotated_publishers[camera].publish(bgr_message(annotated, msg))
        except Exception as exc:
            self.get_logger().error(f"{camera} inference failed: {exc}")
        finally:
            self.busy = False


def main() -> None:
    rclpy.init()
    node = YoloDetectorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
