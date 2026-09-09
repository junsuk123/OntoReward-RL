"""Axis-aligned box geometry and camera-visibility ground truth.

The same box list builds the stage (:mod:`simlab.sim.world`) and answers "could
this camera see that drone" while labelling
(:mod:`simlab.ros.dataset_collector_node`), so a frame where a building hides a
drone never produces a training box for it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

Vec3 = Tuple[float, float, float]

#: Kinds that block a line of sight. Roads and lane markings are flat scenery
#: that a camera above the ground never has to see through.
OCCLUDING_KINDS = frozenset({"building", "occluder"})


@dataclass(frozen=True)
class Box:
    """One axis-aligned box: scenery for the stage, geometry for the ray test."""

    name: str
    center: Vec3
    size: Vec3  # full extents, not half extents
    color: Vec3 = (0.5, 0.5, 0.5)
    kind: str = "occluder"

    @property
    def minimum(self) -> Vec3:
        return tuple(c - s * 0.5 for c, s in zip(self.center, self.size))  # type: ignore[return-value]

    @property
    def maximum(self) -> Vec3:
        return tuple(c + s * 0.5 for c, s in zip(self.center, self.size))  # type: ignore[return-value]

    @property
    def occluding(self) -> bool:
        return self.kind in OCCLUDING_KINDS


@dataclass(frozen=True)
class Visibility:
    """How much of a target one camera can see, and what is in the way."""

    ratio: float
    blocker: str | None

    @property
    def hidden(self) -> bool:
        return self.ratio <= 0.0


def segment_hits_box(start: Vec3, end: Vec3, box: Box) -> bool:
    """Slab test: does the segment ``start -> end`` enter ``box``?"""
    low, high = box.minimum, box.maximum
    near_t, far_t = 0.0, 1.0
    for axis in range(3):
        origin = start[axis]
        direction = end[axis] - origin
        if abs(direction) < 1e-12:
            if origin < low[axis] or origin > high[axis]:
                return False
            continue
        first = (low[axis] - origin) / direction
        second = (high[axis] - origin) / direction
        if first > second:
            first, second = second, first
        near_t = max(near_t, first)
        far_t = min(far_t, second)
        if near_t > far_t:
            return False
    return True


def sample_points(center: Vec3, radius: float) -> List[Vec3]:
    """Centre plus six axial points -- a cheap stand-in for the airframe hull."""
    offsets = (
        (0.0, 0.0, 0.0),
        (radius, 0.0, 0.0), (-radius, 0.0, 0.0),
        (0.0, radius, 0.0), (0.0, -radius, 0.0),
        (0.0, 0.0, radius), (0.0, 0.0, -radius),
    )
    return [(center[0] + dx, center[1] + dy, center[2] + dz) for dx, dy, dz in offsets]


def structure_visibility(
    camera_position: Vec3, center: Vec3, radius: float, boxes: Sequence[Box]
) -> Visibility:
    """Fraction of the sampled hull the camera can reach without crossing a box."""
    blockers = [box for box in boxes if box.occluding]
    if not blockers:
        return Visibility(1.0, None)
    points = sample_points(center, radius)
    clear = 0
    blocker_name: str | None = None
    for index, point in enumerate(points):
        hit = next((box for box in blockers if segment_hits_box(camera_position, point, box)), None)
        if hit is None:
            clear += 1
        elif index == 0 or blocker_name is None:
            # The centre ray names the blocker; it is the one a predictor would
            # reason about ("which structure is it behind right now").
            blocker_name = hit.name
    return Visibility(clear / len(points), blocker_name if clear < len(points) else None)


def box_coverage(target: Sequence[float], cover: Sequence[float]) -> float:
    """Fraction of image-space box ``target`` that ``cover`` paints over."""
    tx1, ty1, tx2, ty2 = target
    cx1, cy1, cx2, cy2 = cover
    area = max(0.0, tx2 - tx1) * max(0.0, ty2 - ty1)
    if area <= 0.0:
        return 0.0
    overlap = max(0.0, min(tx2, cx2) - max(tx1, cx1)) * max(0.0, min(ty2, cy2) - max(ty1, cy1))
    return min(1.0, overlap / area)


# -- occluder loadouts ------------------------------------------------------
# (name, x, y, extent_x, extent_y, height, colour). The front cameras sit at
# x=+12 looking down -x and the drones fly near x=0, so a box at x in [4, 9]
# casts its visual shadow across a metre-scale window of the flight line -- long
# enough for a track to die, short enough to reappear inside one episode.
_OccluderSpec = Tuple[str, float, float, float, float, float, Vec3]

_OCCLUDER_SETS: Dict[str, Tuple[_OccluderSpec, ...]] = {
    "none": (),
    "urban_screens": (
        ("Screen_A", 6.0, 1.1, 0.6, 0.9, 5.5, (0.42, 0.44, 0.47)),
        ("Screen_B", 8.0, -2.0, 0.6, 0.5, 5.0, (0.5, 0.47, 0.42)),
        ("Tower_C", 4.2, -3.6, 1.3, 1.6, 7.5, (0.36, 0.4, 0.45)),
        ("Billboard_D", 7.0, 3.5, 0.5, 1.2, 6.0, (0.55, 0.5, 0.36)),
    ),
    "container_yard": (
        # Two-high stacks; a single container is too low to cross the sight line.
        ("Stack_A", 5.0, 2.2, 2.4, 0.8, 5.2, (0.62, 0.32, 0.24)),
        ("Stack_B", 5.0, -1.4, 2.4, 0.8, 5.2, (0.22, 0.45, 0.55)),
        ("Stack_C", 8.2, 0.4, 2.4, 0.5, 5.2, (0.3, 0.52, 0.34)),
        ("Gantry_Leg_L", 6.6, 4.4, 0.5, 0.5, 8.5, (0.7, 0.66, 0.2)),
        ("Gantry_Leg_R", 6.6, -4.4, 0.5, 0.5, 8.5, (0.7, 0.66, 0.2)),
        ("Gantry_Beam", 6.6, 0.0, 0.5, 9.3, 0.7, (0.7, 0.66, 0.2)),
    ),
    "gantry_wall": (
        ("Wall_North", 6.5, 2.6, 0.5, 2.2, 6.5, (0.46, 0.46, 0.44)),
        ("Wall_South", 6.5, -2.6, 0.5, 2.2, 6.5, (0.46, 0.46, 0.44)),
        ("Pylon_Mid", 9.0, 0.0, 0.7, 0.7, 7.0, (0.38, 0.38, 0.4)),
    ),
    "sparse_masts": (
        ("Mast_A", 5.5, 0.6, 0.35, 0.35, 7.0, (0.5, 0.5, 0.52)),
        ("Mast_B", 7.5, -1.9, 0.35, 0.35, 7.0, (0.5, 0.5, 0.52)),
        ("Mast_C", 4.5, 3.0, 0.35, 0.35, 7.0, (0.5, 0.5, 0.52)),
    ),
}

OCCLUDER_SET_KEYS = tuple(sorted(_OCCLUDER_SETS))


def _jitter(seed: int, salt: int, span: float) -> float:
    """Small deterministic offset so repeated episodes are not pixel-identical."""
    value = (seed * 6364136223846793005 + salt * 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
    return ((value >> 33) / float(1 << 31) - 1.0) * span


def occluder_boxes(set_key: str, seed: int = 0) -> List[Box]:
    """Boxes for one loadout, nudged by ``seed`` but never into the flight lanes."""
    try:
        specs = _OCCLUDER_SETS[set_key]
    except KeyError:
        raise ValueError(
            f"unknown occluder set {set_key!r}; available: {list(OCCLUDER_SET_KEYS)}"
        ) from None
    boxes: List[Box] = []
    for index, (name, x, y, extent_x, extent_y, height, color) in enumerate(specs):
        shifted_x = x + _jitter(seed, index * 2 + 1, 0.4)
        shifted_y = y + _jitter(seed, index * 2 + 2, 0.5)
        boxes.append(
            Box(
                name=name,
                center=(shifted_x, shifted_y, height * 0.5),
                size=(extent_x, extent_y, height),
                color=color,
                kind="occluder",
            )
        )
    return boxes


def shadow_window(
    camera_position: Vec3, box: Box, lane_x: float, altitude: float
) -> Tuple[float, float] | None:
    """The ``y`` interval of a lane that ``box`` hides from ``camera_position``.

    Used to report, per episode, that a structural-occlusion run really does put
    each aircraft out of sight -- and by how much -- instead of trusting that the
    hand-placed geometry lines up.
    """
    depth = camera_position[0] - lane_x
    span = camera_position[0] - box.center[0]
    if depth <= 1e-6 or span <= 1e-6 or span >= depth:
        return None
    fraction = span / depth
    low, high = box.minimum, box.maximum
    height = camera_position[2] + (altitude - camera_position[2]) * fraction
    if not low[2] <= height <= high[2]:
        return None
    camera_y = camera_position[1]
    return (
        camera_y + (low[1] - camera_y) / fraction,
        camera_y + (high[1] - camera_y) / fraction,
    )


def lane_occlusion_windows(
    camera_position: Vec3, boxes: Iterable[Box], lane_x: float, altitude: float
) -> List[Tuple[str, float, float]]:
    """Every ``(blocker, y_start, y_end)`` a lane passes through, sorted by y."""
    windows = []
    for box in boxes:
        if not box.occluding:
            continue
        window = shadow_window(camera_position, box, lane_x, altitude)
        if window is None:
            continue
        start, end = sorted(window)
        if math.isfinite(start) and math.isfinite(end):
            windows.append((box.name, start, end))
    return sorted(windows, key=lambda item: item[1])
