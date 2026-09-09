"""Prototype satellite-label -> ontology-gated front tracking -> RViz pipeline.

The current harness uses simulator TF as the detector position measurement while
it verifies real camera-topic delivery, ontology gating, ID continuity, and RViz
visualization. A learned image feature extractor can later replace
``_detections`` without changing the track lifecycle or visualization contract.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import ColorRGBA, String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from simlab.config import load_config

Point3 = Tuple[float, float, float]


@dataclass
class Detection:
    truth_id: str  # evaluation only; hidden from the association cost
    ontology_key: Tuple[str, str]
    position: Point3


@dataclass
class Track:
    track_id: str
    ontology_key: Tuple[str, str]
    position: Point3
    trail: List[Point3] = field(default_factory=list)
    missed: int = 0


class SwarmPipelineNode(Node):
    def __init__(self) -> None:
        super().__init__("simlab_swarm_pipeline")
        self.declare_parameter("config", "configs/default.yaml")
        self.declare_parameter("rate_hz", 10.0)
        config_path = self.get_parameter("config").value
        self.cfg = load_config(config_path)
        self.map_frame = self.cfg.ros2.map_frame

        self.roster: Dict[str, Tuple[str, str]] = {}
        for team_name, team in (
            ("friendly", self.cfg.drones.friendly),
            ("enemy", self.cfg.drones.enemy),
        ):
            for index in range(team.count):
                prefix = "".join(ch if ch.isalnum() else "_" for ch in team.label).strip("_")
                prefix = prefix or ("Ally" if team_name == "friendly" else "Enemy")
                self.roster[f"{prefix}_{index + 1:02d}"] = (team_name, team.model)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tracks: Dict[str, Track] = {}
        self.id_switches = 0
        self.associations = 0
        self.tick_count = 0
        self.image_frames = {name: 0 for name in ("front_near", "front_far", "satellite_nadir")}

        qos = QoSProfile(depth=2, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        for sensor_id in self.image_frames:
            self.create_subscription(
                Image,
                f"simlab/{sensor_id}/image",
                lambda msg, name=sensor_id: self._on_image(name, msg),
                qos,
            )
        self.marker_pub = self.create_publisher(MarkerArray, "simlab/tracking/markers", 10)
        self.status_pub = self.create_publisher(String, "simlab/tracking/status", 10)
        rate = float(self.get_parameter("rate_hz").value)
        self.create_timer(1.0 / rate, self._on_tick)
        self.get_logger().info(
            f"pipeline ready: satellite labels -> ontology+motion association; roster={len(self.roster)}"
        )

    def _on_image(self, sensor_id: str, _msg: Image) -> None:
        self.image_frames[sensor_id] += 1

    def _detections(self) -> List[Detection]:
        detections = []
        for track_id, ontology_key in self.roster.items():
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame, track_id, rclpy.time.Time()
                )
            except Exception:
                continue
            translation = transform.transform.translation
            detections.append(
                Detection(
                    truth_id=track_id,
                    ontology_key=ontology_key,
                    position=(translation.x, translation.y, translation.z),
                )
            )
        # Detector ordering must not carry identity between frames.
        if self.tick_count % 2:
            detections.reverse()
        return detections

    def _satellite_initialize(self, detections: List[Detection]) -> None:
        """Satellite labels provide the immutable identity supervision."""
        for detection in detections:
            if detection.truth_id not in self.tracks:
                self.tracks[detection.truth_id] = Track(
                    track_id=detection.truth_id,
                    ontology_key=detection.ontology_key,
                    position=detection.position,
                    trail=[detection.position],
                )

    def _associate_front(self, detections: List[Detection]) -> None:
        """Greedy global assignment with ontology hard-gating and motion distance."""
        candidates = []
        tracks = list(self.tracks.values())
        for track_index, track in enumerate(tracks):
            for detection_index, detection in enumerate(detections):
                if track.ontology_key != detection.ontology_key:
                    continue
                distance = math.dist(track.position, detection.position)
                if distance <= 2.0:
                    candidates.append((distance, track_index, detection_index))
        used_tracks, used_detections = set(), set()
        for _distance, track_index, detection_index in sorted(candidates):
            if track_index in used_tracks or detection_index in used_detections:
                continue
            track = tracks[track_index]
            detection = detections[detection_index]
            used_tracks.add(track_index)
            used_detections.add(detection_index)
            self.associations += 1
            if track.track_id != detection.truth_id:
                self.id_switches += 1
            track.position = detection.position
            track.trail.append(detection.position)
            track.trail = track.trail[-120:]
            track.missed = 0
        for index, track in enumerate(tracks):
            if index not in used_tracks:
                track.missed += 1

    def _on_tick(self) -> None:
        self.tick_count += 1
        detections = self._detections()
        if not detections:
            return
        self._satellite_initialize(detections)
        self._associate_front(detections)
        self.marker_pub.publish(self._markers())
        status = {
            "active_tracks": len(self.tracks),
            "id_switches": self.id_switches,
            "associations": self.associations,
            "camera_frames": self.image_frames,
            "association": "ontology(team,model)+motion; TF-backed detector prototype",
        }
        self.status_pub.publish(String(data=json.dumps(status, sort_keys=True)))
        if self.tick_count % 20 == 0:
            self.get_logger().info(json.dumps(status, sort_keys=True))

    def _markers(self) -> MarkerArray:
        result = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for index, track in enumerate(self.tracks.values()):
            ally = track.ontology_key[0] == "friendly"
            color = ColorRGBA(r=0.08 if ally else 1.0, g=0.35 if ally else 0.08, b=1.0 if ally else 0.03, a=0.95)
            body = Marker()
            body.header.frame_id = self.map_frame
            body.header.stamp = stamp
            body.ns = "tracked_drones"
            body.id = index
            body.type = Marker.SPHERE
            body.action = Marker.ADD
            body.pose.position.x, body.pose.position.y, body.pose.position.z = track.position
            body.pose.orientation.w = 1.0
            body.scale.x = body.scale.y = body.scale.z = 0.45 if ally else 0.65
            body.color = color
            result.markers.append(body)

            label = Marker()
            label.header = body.header
            label.ns = "track_ids"
            label.id = index
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = track.position[0]
            label.pose.position.y = track.position[1]
            label.pose.position.z = track.position[2] + 0.6
            label.pose.orientation.w = 1.0
            label.scale.z = 0.38
            label.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            label.text = f"{track.track_id} [{track.ontology_key[1]}]"
            result.markers.append(label)

            trail = Marker()
            trail.header = body.header
            trail.ns = "track_trails"
            trail.id = index
            trail.type = Marker.LINE_STRIP
            trail.action = Marker.ADD
            trail.scale.x = 0.06
            trail.color = color
            from geometry_msgs.msg import Point

            trail.points = [Point(x=x, y=y, z=z) for x, y, z in track.trail]
            result.markers.append(trail)
        return result


def main(args: Optional[List[str]] = None) -> int:
    rclpy.init(args=args)
    node = SwarmPipelineNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
