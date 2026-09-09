"""Pinhole projection of the simulator's truth into each camera image.

This is the one place that knows how a world point becomes a box, so the
dataset collector, the detector's bootstrap boxes and the tests all agree. It
carries no ROS or Isaac import, which is what makes the occlusion reasoning
testable without a simulator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from simlab.config.schema import CameraOpticsConfig, SceneConfig
from simlab.scenarios.occlusion import Box, box_coverage, structure_visibility
from simlab.scenarios.scene import RosterEntry, camera_positions, drone_roster, scene_boxes

Vec3 = Tuple[float, float, float]


def _sub(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _unit(value: Sequence[float]) -> Vec3:
    length = math.sqrt(_dot(value, value))
    if length < 1e-12:
        raise ValueError("cannot normalize a zero-length direction")
    return (value[0] / length, value[1] / length, value[2] / length)


@dataclass(frozen=True)
class ImageBox:
    """A projected sphere: its clamped image box and its depth from the lens."""

    bbox: List[float]
    depth_m: float
    center: Tuple[float, float]
    inside_frame: bool


@dataclass
class Projection:
    """One aircraft as one camera sees it, with occlusion resolved."""

    drone_id: str
    bbox: List[float]
    depth_m: float
    inside_frame: bool
    structure_visibility: float
    blocker: Optional[str] = None
    covered_by: Optional[str] = None
    coverage: float = 0.0
    depth_rank: int = 0

    @property
    def visibility(self) -> float:
        """How much of the aircraft survives structures and nearer aircraft."""
        return max(0.0, self.structure_visibility * (1.0 - self.coverage))


class CameraProjector:
    """The pose and optics of one camera, ready to project world points."""

    def __init__(
        self,
        position: Vec3,
        forward: Vec3,
        right: Vec3,
        up: Vec3,
        optics: CameraOpticsConfig,
    ) -> None:
        self.position = position
        self.forward = forward
        self.right = right
        self.up = up
        self.optics = optics

    @classmethod
    def from_scene(cls, scene: SceneConfig, camera: str) -> "CameraProjector":
        position = camera_positions(scene)[camera]
        if camera.startswith("front"):
            front = scene.cameras.front
            forward = _unit(front.view_direction)
            optics = front.optics
            # Gf's minimum aim rotation starts 90 degrees counter-clockwise from
            # a world-up camera in this rig, so the configured correction is
            # applied relative to that, not to the raw world-up basis.
            clockwise = math.radians(front.image_rotation_clockwise_deg - 90.0)
            up_reference: Vec3 = (0.0, 0.0, 1.0)
        else:
            satellite = scene.cameras.satellite
            forward = _unit(_sub(satellite.target, position))
            optics = satellite.optics
            clockwise = 0.0
            up_reference = (0.0, 1.0, 0.0)
        right0 = _unit(_cross(forward, up_reference))
        up0 = _unit(_cross(right0, forward))
        cosine, sine = math.cos(clockwise), math.sin(clockwise)
        right = tuple(cosine * right0[i] + sine * up0[i] for i in range(3))
        up = tuple(-sine * right0[i] + cosine * up0[i] for i in range(3))
        return cls(position, forward, right, up, optics)  # type: ignore[arg-type]

    def project(self, point: Vec3, radius: float, width: int, height: int) -> ImageBox | None:
        """Image box of a sphere at ``point``; None when it is behind the lens."""
        relative = _sub(point, self.position)
        depth = _dot(relative, self.forward)
        if depth <= self.optics.clipping_range[0]:
            return None
        focal = width * self.optics.focal_length_mm / self.optics.horizontal_aperture_mm
        u = width * 0.5 + focal * _dot(relative, self.right) / depth
        v = height * 0.5 - focal * _dot(relative, self.up) / depth
        # A floor of a few pixels keeps a distant aircraft from collapsing into a
        # degenerate box that no labeller or metric can use.
        half = max(6.0, focal * radius * 1.40 / depth)
        inside = 0 <= u < width and 0 <= v < height
        if inside:
            bbox = [
                max(0.0, u - half), max(0.0, v - half),
                min(width - 1.0, u + half), min(height - 1.0, v + half),
            ]
        else:
            # Left unclamped on purpose: an aircraft two hundred pixels past the
            # right edge is a usable prediction of where it comes back, while a
            # box clamped onto the border would silently claim it is at the edge.
            bbox = [u - half, v - half, u + half, v + half]
        return ImageBox(bbox=[round(value, 2) for value in bbox], depth_m=depth,
                        center=(u, v), inside_frame=inside)


class SceneProjector:
    """Every camera of a scene, plus the geometry that blocks their sight lines."""

    def __init__(self, scene: SceneConfig, cameras: Sequence[str]) -> None:
        self.scene = scene
        self.boxes: List[Box] = scene_boxes(scene)
        self.roster: Dict[str, RosterEntry] = drone_roster(scene)
        self.projectors = {name: CameraProjector.from_scene(scene, name) for name in cameras}

    def camera(self, name: str) -> CameraProjector:
        return self.projectors[name]

    def observe(
        self, camera: str, positions: Mapping[str, Vec3], width: int, height: int
    ) -> List[Projection]:
        """Project every known aircraft, then resolve what hides what.

        Two independent effects, kept separate because the scenarios ask
        different questions of them: a structure between the lens and the
        aircraft (a world-space ray test) and a nearer aircraft painted over a
        farther one (image-space coverage, in depth order).
        """
        projector = self.projectors[camera]
        results: List[Projection] = []
        for drone_id, point in positions.items():
            entry = self.roster.get(drone_id)
            if entry is None:
                continue
            image_box = projector.project(point, entry.radius_m, width, height)
            if image_box is None:
                continue
            visibility = structure_visibility(
                projector.position, point, entry.radius_m, self.boxes
            )
            results.append(
                Projection(
                    drone_id=drone_id,
                    bbox=image_box.bbox,
                    depth_m=image_box.depth_m,
                    inside_frame=image_box.inside_frame,
                    structure_visibility=visibility.ratio,
                    blocker=visibility.blocker,
                )
            )

        results.sort(key=lambda item: item.depth_m)
        for rank, projection in enumerate(results):
            projection.depth_rank = rank
            for nearer in results[:rank]:
                coverage = box_coverage(projection.bbox, nearer.bbox)
                if coverage > projection.coverage:
                    projection.coverage = coverage
                    projection.covered_by = nearer.drone_id
        return results


def classify(projection: Projection | None, minimum_visibility: float) -> str:
    """The track state a detector-plus-tracker would have to explain."""
    if projection is None or not projection.inside_frame:
        return "out_of_view"
    if projection.structure_visibility <= minimum_visibility:
        return "occluded_by_structure"
    if projection.visibility < minimum_visibility:
        return "occluded_by_drone"
    return "visible"
