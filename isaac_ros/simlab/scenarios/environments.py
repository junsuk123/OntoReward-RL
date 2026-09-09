"""Named flight environments.

Detector accuracy collapses when every training frame shares one sky, one
ground albedo and one skyline. Each preset here changes the lighting, the
palette, the block layout and the set of structures standing between the
cameras and the flight line, while leaving the arena itself -- the corridor the
aircraft fly along and the calibrated camera rig -- untouched. The perception
stage therefore sees the same task under different appearances instead of a new
task.

Only knobs that actually reach the render are listed: dome light scale and
colour, per-surface display colours, building layout, and the occluder loadout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from simlab.scenarios.occlusion import Box, occluder_boxes

Vec3 = Tuple[float, float, float]

#: Half-width of the corridor kept free of scenery, in metres. Buildings start
#: outside it so the shuttle lanes and the camera stem stay clear.
ARENA_CLEARANCE_M = 9.5


@dataclass(frozen=True)
class EnvironmentPreset:
    """One look: lighting, palette, block layout, and what stands in the way."""

    key: str
    description: str
    light_scale: float  # multiplies world.dome_light_intensity
    light_color: Vec3
    ground_color: Vec3
    road_color: Vec3
    marking_color: Vec3
    building_palette: Tuple[Vec3, ...]
    building_columns: Tuple[float, ...]  # x centres of the block grid
    building_rows: Tuple[float, ...]  # y centres of the block grid
    building_height_range: Tuple[float, float]
    building_footprint: Tuple[float, float] = (9.2, 7.7)
    window_bands: bool = True
    window_color: Vec3 = (0.16, 0.38, 0.52)
    occluder_set: str = "urban_screens"
    roads: bool = True


ENVIRONMENTS: Dict[str, EnvironmentPreset] = {
    "urban_day": EnvironmentPreset(
        key="urban_day",
        description="midday city block, neutral light, mixed concrete facades",
        light_scale=1.0,
        light_color=(1.0, 0.99, 0.96),
        ground_color=(0.16, 0.18, 0.16),
        road_color=(0.055, 0.06, 0.065),
        marking_color=(0.95, 0.78, 0.12),
        building_palette=(
            (0.38, 0.42, 0.48), (0.52, 0.44, 0.37), (0.34, 0.39, 0.46),
            (0.48, 0.39, 0.34), (0.36, 0.43, 0.40), (0.43, 0.42, 0.48),
        ),
        building_columns=(-16.0, 0.0, 16.0),
        building_rows=(-15.0, 15.0),
        building_height_range=(7.0, 15.0),
        occluder_set="urban_screens",
    ),
    "urban_overcast": EnvironmentPreset(
        key="urban_overcast",
        description="flat overcast light, low contrast, desaturated facades",
        light_scale=0.62,
        light_color=(0.9, 0.93, 1.0),
        ground_color=(0.21, 0.22, 0.23),
        road_color=(0.1, 0.11, 0.12),
        marking_color=(0.72, 0.7, 0.6),
        building_palette=(
            (0.44, 0.45, 0.47), (0.4, 0.41, 0.43), (0.48, 0.48, 0.5),
            (0.37, 0.39, 0.41), (0.46, 0.44, 0.43), (0.41, 0.43, 0.46),
        ),
        building_columns=(-16.0, 0.0, 16.0),
        building_rows=(-15.0, 15.0),
        building_height_range=(8.0, 17.0),
        window_color=(0.3, 0.34, 0.38),
        occluder_set="gantry_wall",
    ),
    "urban_dusk": EnvironmentPreset(
        key="urban_dusk",
        description="low warm sun, long shadows, lit window bands",
        light_scale=0.34,
        light_color=(1.0, 0.72, 0.45),
        ground_color=(0.13, 0.12, 0.13),
        road_color=(0.05, 0.048, 0.055),
        marking_color=(0.85, 0.62, 0.2),
        building_palette=(
            (0.35, 0.28, 0.26), (0.42, 0.31, 0.24), (0.3, 0.27, 0.3),
            (0.38, 0.3, 0.25), (0.28, 0.28, 0.32), (0.4, 0.33, 0.28),
        ),
        building_columns=(-16.0, 0.0, 16.0),
        building_rows=(-15.0, 15.0),
        building_height_range=(9.0, 18.0),
        window_color=(0.95, 0.78, 0.42),
        occluder_set="urban_screens",
    ),
    "urban_night": EnvironmentPreset(
        key="urban_night",
        description="dark sky, cool ambient, bright window bands as the only cues",
        light_scale=0.1,
        light_color=(0.55, 0.62, 0.9),
        ground_color=(0.05, 0.055, 0.07),
        road_color=(0.02, 0.022, 0.03),
        marking_color=(0.6, 0.55, 0.3),
        building_palette=(
            (0.1, 0.11, 0.14), (0.13, 0.12, 0.14), (0.09, 0.1, 0.13),
            (0.12, 0.13, 0.16), (0.1, 0.12, 0.15), (0.11, 0.11, 0.13),
        ),
        building_columns=(-16.0, 0.0, 16.0),
        building_rows=(-15.0, 15.0),
        building_height_range=(9.0, 18.0),
        window_color=(1.0, 0.93, 0.6),
        occluder_set="sparse_masts",
    ),
    "industrial_yard": EnvironmentPreset(
        key="industrial_yard",
        description="low dense sheds and a container yard; the busiest occluders",
        light_scale=0.85,
        light_color=(1.0, 0.97, 0.9),
        ground_color=(0.24, 0.23, 0.21),
        road_color=(0.15, 0.145, 0.14),
        marking_color=(0.9, 0.85, 0.3),
        building_palette=(
            (0.5, 0.48, 0.44), (0.45, 0.3, 0.26), (0.3, 0.42, 0.46),
            (0.52, 0.5, 0.3), (0.36, 0.38, 0.4), (0.46, 0.42, 0.36),
        ),
        building_columns=(-18.0, -6.0, 6.0, 18.0),
        building_rows=(-13.0, 13.0),
        building_height_range=(4.0, 9.0),
        building_footprint=(9.5, 6.5),
        window_bands=False,
        occluder_set="container_yard",
    ),
    "desert_outpost": EnvironmentPreset(
        key="desert_outpost",
        description="bright sand, sparse low structures, wide empty sky",
        light_scale=1.25,
        light_color=(1.0, 0.96, 0.86),
        ground_color=(0.62, 0.53, 0.36),
        road_color=(0.5, 0.44, 0.32),
        marking_color=(0.75, 0.7, 0.55),
        building_palette=(
            (0.66, 0.58, 0.44), (0.6, 0.52, 0.4), (0.7, 0.62, 0.48),
        ),
        building_columns=(-20.0, 20.0),
        building_rows=(-18.0, 18.0),
        building_height_range=(3.5, 6.5),
        building_footprint=(8.0, 7.0),
        window_bands=False,
        occluder_set="sparse_masts",
        roads=False,
    ),
    "coastal_flats": EnvironmentPreset(
        key="coastal_flats",
        description="hazy bright sky over pale flats, few tall structures",
        light_scale=1.1,
        light_color=(0.92, 0.96, 1.0),
        ground_color=(0.4, 0.45, 0.44),
        road_color=(0.3, 0.34, 0.35),
        marking_color=(0.85, 0.85, 0.82),
        building_palette=(
            (0.72, 0.74, 0.76), (0.64, 0.68, 0.72), (0.78, 0.78, 0.76),
        ),
        building_columns=(-19.0, 19.0),
        building_rows=(-16.0, 16.0),
        building_height_range=(5.0, 11.0),
        occluder_set="gantry_wall",
    ),
}

ENVIRONMENT_KEYS = tuple(sorted(ENVIRONMENTS))


def environment(key: str) -> EnvironmentPreset:
    try:
        return ENVIRONMENTS[key]
    except KeyError:
        raise ValueError(
            f"unknown environment {key!r}; available: {list(ENVIRONMENT_KEYS)}"
        ) from None


def _height(preset: EnvironmentPreset, seed: int, index: int) -> float:
    low, high = preset.building_height_range
    value = (seed * 2654435761 + index * 40503 + 12345) & 0xFFFFFFFF
    return low + (high - low) * (value / float(0xFFFFFFFF))


def environment_boxes(
    preset: EnvironmentPreset, seed: int = 0, occluder_set: str | None = None
) -> List[Box]:
    """Every static box of one environment: ground, roads, blocks, occluders.

    Ordered ground-up so the stage builder can create prims in one pass, and
    kind-tagged so the visibility test only ray-casts against what is solid.
    """
    boxes: List[Box] = [
        Box("Ground", (0.0, 0.0, -0.15), (80.0, 80.0, 0.3), preset.ground_color, "ground"),
    ]
    if preset.roads:
        boxes += [
            Box("RoadEastWest", (0.0, 0.0, 0.015), (80.0, 9.0, 0.03), preset.road_color, "road"),
            Box("RoadNorthSouth", (0.0, 0.0, 0.02), (9.0, 80.0, 0.04), preset.road_color, "road"),
        ]
        for axis in ("x", "y"):
            for index, offset in enumerate(range(-36, 37, 6)):
                position = (float(offset), 0.0, 0.055) if axis == "x" else (0.0, float(offset), 0.055)
                size = (3.0, 0.12, 0.03) if axis == "x" else (0.12, 3.0, 0.03)
                boxes.append(
                    Box(f"Lane_{axis}_{index:02d}", position, size, preset.marking_color, "marking")
                )

    footprint_x, footprint_y = preset.building_footprint
    index = 0
    for column in preset.building_columns:
        for row in preset.building_rows:
            if abs(column) < ARENA_CLEARANCE_M and abs(row) < ARENA_CLEARANCE_M:
                continue  # never build inside the flight corridor
            height = _height(preset, seed, index)
            color = preset.building_palette[index % len(preset.building_palette)]
            prefix = f"Building_{index:02d}"
            boxes.append(
                Box(
                    f"{prefix}/Podium", (column, row, 0.12),
                    (footprint_x + 1.8, footprint_y + 1.8, 0.24), (0.45, 0.45, 0.43), "building",
                )
            )
            boxes.append(
                Box(
                    f"{prefix}/Tower", (column, row, height * 0.5 + 0.24),
                    (footprint_x, footprint_y, height), color, "building",
                )
            )
            boxes.append(
                Box(
                    f"{prefix}/Roof", (column, row, height + 0.65),
                    (footprint_x * 0.54, footprint_y * 0.45, 0.8), (0.18, 0.2, 0.22), "building",
                )
            )
            if preset.window_bands:
                facade_y = row - footprint_y * 0.503 if row > 0 else row + footprint_y * 0.503
                for floor in range(2, int(height), 2):
                    boxes.append(
                        Box(
                            f"{prefix}/WindowBand_{floor:02d}", (column, facade_y, float(floor)),
                            (footprint_x * 0.85, 0.04, 0.65), preset.window_color, "marking",
                        )
                    )
            index += 1

    boxes += occluder_boxes(occluder_set or preset.occluder_set, seed)
    return boxes
