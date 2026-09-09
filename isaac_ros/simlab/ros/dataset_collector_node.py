"""Collect camera frames, occlusion-aware box prompts, and tracking ground truth.

Two streams come out of one episode:

``prompts.jsonl``
    One record per saved image: the boxes a labeller should refine. A box is
    only written for an aircraft the camera can actually see. Projecting truth
    without asking what stands in front of it is how a detector learns to fire
    on empty facades.

``tracking_truth.jsonl``
    A time-based stream, independent of whether a frame arrived, holding what
    each scenario is about: heading, speed, depth order and formation for the
    mutual-occlusion case; which structure hides an aircraft and where it is
    predicted to re-emerge for the structural case; how long the observation has
    been missing and whether the track should still be alive for the dropout
    case.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from tf2_msgs.msg import TFMessage

from simlab.config import load_config
from simlab.ros.yolo_detector_node import image_to_bgr
from simlab.scenarios.projection import Projection, SceneProjector, classify
from simlab.scenarios.timeline import ScenarioTimeline

Vec3 = Tuple[float, float, float]
CAMERAS = ("front_near", "front_far", "satellite_nadir")


@dataclass
class DroneTrack:
    """Where an aircraft is and how it is moving, from the simulator's TF."""

    position: Vec3 = (0.0, 0.0, 0.0)
    velocity: Vec3 = (0.0, 0.0, 0.0)
    stamp_s: float = 0.0
    #: Per camera: (time, position, velocity) of the last unobstructed sighting.
    last_seen: Dict[str, Tuple[float, Vec3, Vec3]] = field(default_factory=dict)

    def update(self, position: Vec3, stamp_s: float) -> None:
        dt = stamp_s - self.stamp_s
        if 1e-3 < dt < 1.0:
            self.velocity = tuple(
                (position[axis] - self.position[axis]) / dt for axis in range(3)
            )  # type: ignore[assignment]
        self.position = position
        self.stamp_s = stamp_s

    @property
    def speed(self) -> float:
        return math.dist((0.0, 0.0, 0.0), self.velocity)


class DatasetCollectorNode(Node):
    def __init__(self) -> None:
        super().__init__("simlab_dataset_collector")
        self.declare_parameter("config", "configs/default.yaml")
        self.declare_parameter("output", "artifacts/perception/dataset_raw")
        self.declare_parameter("sample_period_s", 1.0)
        self.declare_parameter("truth_rate_hz", 5.0)
        self.declare_parameter("min_visibility", 0.35)
        self.cfg = load_config(str(self.get_parameter("config").value))
        self.output = Path(str(self.get_parameter("output").value)).resolve()
        (self.output / "images").mkdir(parents=True, exist_ok=True)
        self.manifest = self.output / "prompts.jsonl"
        self.truth_log = self.output / "tracking_truth.jsonl"
        self.period_ns = int(float(self.get_parameter("sample_period_s").value) * 1e9)
        self.min_visibility = float(self.get_parameter("min_visibility").value)

        self.projector = SceneProjector(self.cfg, CAMERAS)
        self.roster = self.projector.roster
        self.timeline = (
            ScenarioTimeline.from_scene(self.cfg) if self.cfg.scenarios.enabled else None
        )
        self.tracks: Dict[str, DroneTrack] = {}
        self.last_stamp: Dict[str, int] = {}
        self.frame_size: Dict[str, Tuple[int, int]] = {}
        self.last_image_s: Dict[str, float] = {}
        self.saved_images = 0
        self.saved_objects = 0
        self.hidden_objects = 0

        qos = QoSProfile(depth=2, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(TFMessage, "tf", self.on_tf, 20)
        self.sample_publishers = {}
        for camera in CAMERAS:
            width, height = self._default_frame_size(camera)
            self.frame_size[camera] = (width, height)
            self.sample_publishers[camera] = self.create_publisher(
                Image, f"simlab/dataset/{camera}/image", qos
            )
            self.create_subscription(
                Image,
                f"simlab/{camera}/image",
                lambda msg, name=camera: self.on_image(name, msg),
                qos,
            )
        rate = max(0.5, float(self.get_parameter("truth_rate_hz").value))
        self.create_timer(1.0 / rate, self.on_truth_tick)
        self.write_session()
        episode = self.timeline.episode_id if self.timeline else "no-scenario"
        self.get_logger().info(
            f"dataset collector ready: episode={episode} output={self.output} "
            f"sight-line boxes={sum(1 for box in self.projector.boxes if box.occluding)} "
            f"min_visibility={self.min_visibility}"
        )

    # -- geometry ----------------------------------------------------------
    def _default_frame_size(self, camera: str) -> Tuple[int, int]:
        """Image size before the first frame arrives, for the truth stream."""
        optics = (
            self.cfg.cameras.front.optics
            if camera.startswith("front")
            else self.cfg.cameras.satellite.optics
        )
        scale = self.cfg.ros2.swarm_image_scale
        return (
            max(1, round(optics.resolution[0] * scale)),
            max(1, round(optics.resolution[1] * scale)),
        )

    def projections(self, camera: str, width: int, height: int) -> List[Projection]:
        positions = {drone_id: track.position for drone_id, track in self.tracks.items()}
        return self.projector.observe(camera, positions, width, height)

    # -- subscriptions -----------------------------------------------------
    def on_tf(self, msg: TFMessage) -> None:
        for transform in msg.transforms:
            drone_id = transform.child_frame_id
            if drone_id not in self.roster:
                continue
            stamp = transform.header.stamp
            stamp_s = float(stamp.sec) + float(stamp.nanosec) * 1e-9
            translation = transform.transform.translation
            track = self.tracks.setdefault(drone_id, DroneTrack(stamp_s=stamp_s))
            track.update((translation.x, translation.y, translation.z), stamp_s)

    def on_image(self, camera: str, msg: Image) -> None:
        stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        stamp_s = stamp_ns * 1e-9
        self.frame_size[camera] = (msg.width, msg.height)
        if self.timeline is not None and not self.timeline.observing(camera, stamp_s):
            # The simulator normally stops publishing during a blackout. On a Kit
            # build without a camera enable input the frame still arrives, and
            # saving it would contradict the gap this episode is meant to contain.
            return
        self.last_image_s[camera] = stamp_s
        if stamp_ns - self.last_stamp.get(camera, -self.period_ns) < self.period_ns:
            return
        if not self.tracks:
            return
        self.last_stamp[camera] = stamp_ns
        # The rosbag records this sampled stream, not the 60 Hz render stream,
        # which keeps a default collection run to a practical size.
        self.sample_publishers[camera].publish(msg)
        frame = image_to_bgr(msg)
        name = f"{camera}_{stamp_ns:019d}.jpg"
        path = self.output / "images" / name
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        projections = self.projections(camera, msg.width, msg.height)
        visible, hidden = [], []
        for projection in projections:
            entry = self.roster[projection.drone_id]
            payload = {
                "drone_id": projection.drone_id,
                "team": entry.team,
                "model": entry.model,
                "formation": entry.formation,
                "bbox_xyxy": projection.bbox,
                "visibility": round(projection.visibility, 4),
                "depth_m": round(projection.depth_m, 3),
                "depth_rank": projection.depth_rank,
                "state": classify(projection, self.min_visibility),
            }
            if projection.blocker:
                payload["blocked_by"] = projection.blocker
            if projection.covered_by and projection.coverage > 0.0:
                payload["covered_by"] = projection.covered_by
                payload["coverage"] = round(projection.coverage, 4)
            if projection.inside_frame and projection.visibility >= self.min_visibility:
                visible.append(payload)
                self.tracks[projection.drone_id].last_seen[camera] = (
                    stamp_s,
                    self.tracks[projection.drone_id].position,
                    self.tracks[projection.drone_id].velocity,
                )
            else:
                hidden.append(payload)

        self.saved_images += 1
        self.saved_objects += len(visible)
        self.hidden_objects += len(hidden)
        record = {
            "image": str(path.relative_to(self.output)),
            "camera": camera,
            "stamp_ns": stamp_ns,
            "width": msg.width,
            "height": msg.height,
            "scenario": self.timeline.scenario if self.timeline else None,
            "environment": self.cfg.world.environment,
            "episode_id": self.timeline.episode_id if self.timeline else "",
            # Boxes to train on: only what the camera could really see.
            "objects": visible,
            # Everything the projection produced but the camera could not see.
            # Kept out of the labels and kept in the record, so an occluded frame
            # is a usable negative rather than a silently dropped one.
            "hidden": hidden,
        }
        self._append(self.manifest, record)

    # -- truth stream ------------------------------------------------------
    def on_truth_tick(self) -> None:
        """Emit tracking truth on a clock, so blackouts are recorded as events.

        An image-driven log has nothing to say exactly when the interesting
        thing happens -- the frames stop arriving. This runs regardless.
        """
        if not self.tracks:
            return
        now = self.get_clock().now()
        t = now.nanoseconds * 1e-9
        for camera in CAMERAS:
            width, height = self.frame_size[camera]
            gap = self.timeline.blackout(camera, t) if self.timeline else None
            last_frame = self.last_image_s.get(camera)
            frame_age = None if last_frame is None else round(max(0.0, t - last_frame), 3)
            projections = {p.drone_id: p for p in self.projections(camera, width, height)}
            entries = []
            for drone_id, track in self.tracks.items():
                projection = projections.get(drone_id)
                entry = self.roster[drone_id]
                state = "sensor_blackout" if gap else (
                    classify(projection, self.min_visibility) if projection else "out_of_view"
                )
                if state == "visible":
                    # "Last seen" has to follow the camera, not the sampler: a
                    # 0.4 s blackout is invisible to a 1 Hz image sample, and the
                    # gap length is the whole point of the dropout scenario.
                    track.last_seen[camera] = (t, track.position, track.velocity)
                seen_t, seen_position, seen_velocity = track.last_seen.get(
                    camera, (None, track.position, track.velocity)
                )
                since = None if seen_t is None else round(t - seen_t, 3)
                payload = {
                    "drone_id": drone_id,
                    "team": entry.team,
                    "model": entry.model,
                    "formation": entry.formation,
                    "position": [round(v, 3) for v in track.position],
                    "velocity": [round(v, 3) for v in track.velocity],
                    "speed_mps": round(track.speed, 3),
                    "heading_rad": round(math.atan2(track.velocity[1], track.velocity[0]), 4),
                    "state": state,
                    # The aircraft never stops flying, so the truth track never
                    # ends -- an ID that dies during a gap is the tracker's
                    # error, not a change in the world.
                    "track_alive": True,
                    "seconds_since_observation": since,
                    "seconds_since_frame": frame_age,
                }
                if projection is not None:
                    payload.update(
                        {
                            "bbox_xyxy": projection.bbox,
                            "visibility": round(projection.visibility, 4),
                            "depth_m": round(projection.depth_m, 3),
                            "depth_rank": projection.depth_rank,
                        }
                    )
                    if projection.blocker:
                        payload["blocked_by"] = projection.blocker
                    if projection.covered_by and projection.coverage > 0.0:
                        payload["covered_by"] = projection.covered_by
                if state != "visible":
                    payload["predicted"] = self.prediction(
                        camera, seen_position, seen_velocity, since or 0.0, entry.radius_m,
                        width, height,
                    )
                entries.append(payload)
            record = {
                "stamp_s": round(t, 3),
                "camera": camera,
                "scenario": self.timeline.scenario if self.timeline else None,
                "environment": self.cfg.world.environment,
                "episode_id": self.timeline.episode_id if self.timeline else "",
                "sensor_state": "blackout" if gap else "observing",
                "gap": gap.as_dict() if gap else None,
                "seconds_since_frame": frame_age,
                "drones": entries,
            }
            self._append(self.truth_log, record)

    def prediction(
        self,
        camera: str,
        position: Vec3,
        velocity: Vec3,
        horizon_s: float,
        radius: float,
        width: int,
        height: int,
    ) -> Dict[str, object]:
        """Where a lost aircraft should be, and where it should come back.

        Constant-velocity dead reckoning from the last clean sighting: the same
        prediction a tracker has to make, recorded so a re-identification model
        can be scored against what actually happened.
        """
        predicted = tuple(position[axis] + velocity[axis] * horizon_s for axis in range(3))
        projected = self.projector.camera(camera).project(predicted, radius, width, height)
        payload: Dict[str, object] = {
            "position": [round(v, 3) for v in predicted],
            "horizon_s": round(horizon_s, 3),
        }
        if projected is not None:
            payload["bbox_xyxy"] = projected.bbox
            payload["depth_m"] = round(projected.depth_m, 3)
            payload["inside_frame"] = projected.inside_frame
        return payload

    # -- output ------------------------------------------------------------
    @staticmethod
    def _append(path: Path, record: Dict[str, object]) -> None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    def write_session(self) -> None:
        """Episode metadata beside the frames, so a session is self-describing."""
        session = {
            "session": self.output.name,
            "scenario": self.timeline.scenario if self.timeline else None,
            "environment": self.cfg.world.environment,
            "episode_id": self.timeline.episode_id if self.timeline else "",
            "seed": self.timeline.seed if self.timeline else None,
            "min_visibility": self.min_visibility,
            "roster": {
                drone_id: {"team": entry.team, "model": entry.model, "formation": entry.formation}
                for drone_id, entry in self.roster.items()
            },
            "timeline": self.timeline.describe() if self.timeline else None,
            "images": self.saved_images,
            "labelled_objects": self.saved_objects,
            "occluded_objects": self.hidden_objects,
        }
        (self.output / "session.json").write_text(
            json.dumps(session, indent=2), encoding="utf-8"
        )


def main() -> None:
    rclpy.init()
    node = DatasetCollectorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # The counts are only final at shutdown; rewrite the session file so the
        # training side can trust it without re-reading every record.
        node.write_session()
        node.get_logger().info(
            f"collected {node.saved_images} images, {node.saved_objects} labelled objects, "
            f"{node.hidden_objects} occluded"
        )
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
