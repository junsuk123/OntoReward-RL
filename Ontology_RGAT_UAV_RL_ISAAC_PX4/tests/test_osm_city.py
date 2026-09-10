"""The real-city import: projection, orientation, and what it hands the GNSS model."""
from __future__ import annotations

import copy
import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "isaac_sim"))

from osm_city import (  # noqa: E402
    buildings_from_extract, load_extract, minimum_area_rectangle, project_to_local)
from urban_scene import Building, UrbanConfig, UrbanLayout  # noqa: E402


def _sorted_rows(points: np.ndarray) -> np.ndarray:
    return np.asarray(points)[np.lexsort((np.asarray(points)[:, 1],
                                          np.asarray(points)[:, 0]))]


def _corners(centre, size, yaw: float) -> np.ndarray:
    """The four corners of an oriented rectangle, in a comparable order."""
    half = 0.5 * np.asarray(size, dtype=float)
    unit = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]) * half
    c, s = math.cos(yaw), math.sin(yaw)
    turned = np.stack([unit[:, 0] * c - unit[:, 1] * s,
                       unit[:, 0] * s + unit[:, 1] * c], axis=1)
    return _sorted_rows(turned + np.asarray(centre, dtype=float))


def _layout_of(*buildings: Building) -> UrbanLayout:
    """An UrbanLayout over exactly these buildings, bypassing the generator."""
    class _Fixed(UrbanLayout):
        def _generate(self):
            return iter(buildings)

    return _Fixed(UrbanConfig.from_mapping({"urban": {}}))


@pytest.fixture(scope="module")
def system_config():
    with (ROOT / "config" / "system.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


@pytest.fixture(scope="module")
def extract(system_config):
    name = system_config["urban"]["extract"]
    return load_extract(ROOT / "assets" / "city" / f"{name}.json")


def test_the_projection_puts_the_origin_at_the_origin(extract):
    local = project_to_local([extract.latitude], [extract.longitude],
                             extract.latitude, extract.longitude)
    assert local[0] == pytest.approx([0.0, 0.0], abs=1e-9)


def test_the_projection_is_metric_in_both_axes(extract):
    """A degree of latitude is ~111 km; a degree of longitude shrinks with it."""
    north = project_to_local([extract.latitude + 0.001], [extract.longitude],
                             extract.latitude, extract.longitude)[0]
    east = project_to_local([extract.latitude], [extract.longitude + 0.001],
                            extract.latitude, extract.longitude)[0]
    assert north[1] == pytest.approx(111.19, rel=0.01)
    assert north[0] == pytest.approx(0.0, abs=1e-6)
    assert east[0] == pytest.approx(111.19 * math.cos(math.radians(extract.latitude)),
                                    rel=0.01)


def test_heading_turns_the_map_and_not_the_scale(extract):
    """The city rotates about the origin, so distances are preserved."""
    plain = project_to_local([extract.latitude + 0.001], [extract.longitude + 0.001],
                             extract.latitude, extract.longitude)[0]
    turned = project_to_local([extract.latitude + 0.001], [extract.longitude + 0.001],
                              extract.latitude, extract.longitude,
                              heading_rad=math.radians(90.0))[0]
    assert np.linalg.norm(turned) == pytest.approx(np.linalg.norm(plain), rel=1e-9)
    assert turned[0] == pytest.approx(plain[1], abs=1e-6)


def test_minimum_area_rectangle_recovers_a_rotated_box():
    """A 20x6 box turned 30 degrees comes back as a 20x6 box turned 30 degrees."""
    angle = math.radians(30.0)
    half = np.array([10.0, 3.0])
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float) * half
    c, s = math.cos(angle), math.sin(angle)
    turned = np.stack([corners[:, 0] * c - corners[:, 1] * s,
                       corners[:, 0] * s + corners[:, 1] * c], axis=1)
    turned = turned + np.array([12.0, -5.0])

    centre, size, yaw = minimum_area_rectangle(turned)

    assert centre == pytest.approx([12.0, -5.0], abs=1e-6)
    assert sorted(size) == pytest.approx([6.0, 20.0], abs=1e-6)
    # The rectangle is invariant under a quarter turn with its extents swapped,
    # so the yaw is only meaningful through the corners it reproduces.
    assert _corners(centre, size, yaw) == pytest.approx(_sorted_rows(turned), abs=1e-6)


def test_a_bounding_box_would_have_lost_the_bearing():
    """The reason min-area is worth the rotating calipers at all."""
    angle = math.radians(40.0)
    half = np.array([15.0, 2.5])
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float) * half
    c, s = math.cos(angle), math.sin(angle)
    turned = np.stack([corners[:, 0] * c - corners[:, 1] * s,
                       corners[:, 0] * s + corners[:, 1] * c], axis=1)

    _, size, _ = minimum_area_rectangle(turned)
    aabb = turned.max(axis=0) - turned.min(axis=0)

    assert sorted(size) == pytest.approx([5.0, 30.0], abs=1e-6)
    # The axis-aligned box is nearly square: it has thrown the facade away.
    assert min(aabb) / max(aabb) > 0.6


def test_real_footprints_carry_real_bearings(extract):
    """A real city is not a grid, and the import must not flatten it into one."""
    buildings = list(buildings_from_extract(extract, keep_within_m=200.0))
    assert len(buildings) > 30
    yaws = np.array([b.yaw_rad for b in buildings])
    assert np.std(yaws) > 0.2
    assert not np.allclose(yaws, 0.0)


def test_untagged_heights_are_drawn_from_the_configured_range(extract):
    """OSM rarely tags height; a single default would flatten the skyline."""
    lo, hi = 12.0, 36.0
    buildings = list(buildings_from_extract(
        extract, height_range_m=(lo, hi), seed=3, keep_within_m=200.0))
    heights = np.array([b.height_m for b in buildings])
    drawn = heights[(heights >= lo) & (heights <= hi)]

    assert heights.min() > 0.0
    assert np.std(drawn) > 1.0                       # a skyline, not a wall
    # Seeding is what keeps a run reproducible from the extract alone.
    again = list(buildings_from_extract(
        extract, height_range_m=(lo, hi), seed=3, keep_within_m=200.0))
    assert [b.height_m for b in again] == pytest.approx(list(heights))


def test_the_route_is_cut_clear_of_the_masonry(system_config):
    """The lap is a rectangle imposed on a real map, so it has to be cleared."""
    layout = UrbanLayout(UrbanConfig.from_mapping(system_config))
    assert layout.cfg.source == "osm"
    assert len(layout.buildings) > 20

    route = layout._route_polyline()
    assert not any(layout.contains([x, y, 2.0]) for x, y in route)
    # And clear at the height the deck's roof rides at, not just at the road.
    assert not any(layout.contains([x, y, layout.cfg.height_range_m[0] * 0.0 + 3.2])
                   for x, y in route)


def test_the_real_city_still_makes_a_canyon(system_config):
    """Real geometry has to keep the thing the experiment measures.

    The sky must close in over the carriageway and open again -- a lap that
    reports one constant sky view would make the GNSS sweep meaningless.
    """
    layout = UrbanLayout(UrbanConfig.from_mapping(system_config))
    route = layout._route_polyline()[::30]
    sky = np.array([layout.sky_view_fraction([x, y, 4.0]) for x, y in route])

    assert sky.max() < 0.98                          # never open country
    assert sky.max() - sky.min() > 0.10              # and it varies around the lap


def test_a_rotated_building_blocks_across_its_facade_and_not_along_it():
    """The occlusion test must respect the yaw, not the bounding box.

    The slab's long axis starts along north and the yaw turns it to 135 deg,
    so it runs north-west. Standing off it to the north-east, the sky is cut
    across the facade and open along it -- which an axis-aligned bounding box,
    being nearly square here, could not tell apart.
    """
    slab = Building(x0=-2.0, x1=2.0, y0=-30.0, y1=30.0, height_m=40.0,
                    yaw_rad=math.radians(45.0))
    layout = _layout_of(slab)

    eye = [11.0, 11.0, 2.0]                       # off the facade, to the NE
    across = layout.blocked(eye, math.radians(225.0), math.radians(20.0))[0]
    along = layout.blocked(eye, math.radians(135.0), math.radians(20.0))[0]

    assert across and not along
    # The same footprint left upright runs north-south instead, so the sky it
    # leaves open is a different one. That the answer changes at all is what
    # proves the yaw is being used rather than a bounding box.
    upright = _layout_of(dataclasses.replace(slab, yaw_rad=0.0))
    assert upright.blocked(eye, math.radians(135.0), math.radians(20.0))[0]
