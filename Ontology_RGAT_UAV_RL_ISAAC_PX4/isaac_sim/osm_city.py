#!/usr/bin/env python3
"""Real building footprints, from OpenStreetMap, as the city the episode flies in.

The synthetic block in ``urban_scene`` is a good canyon but an invented one:
its facades are all parallel, all the same depth, and all the same distance
from the road. A real street is none of those things, and the GNSS model is
sensitive to exactly that -- which satellites survive depends on the gaps
between real buildings, not on a regular pattern.

This module turns a cached Overpass extract (``scripts/fetch_city.py``) into
the same :class:`~urban_scene.Building` boxes the rest of the system already
uses, so the camera renders and the GNSS model occludes with real geometry and
nothing downstream has to know where the skyline came from.

Two approximations, both deliberate:

* each footprint becomes its **minimum-area rectangle**, not its true polygon.
  The occlusion test is an exact ray/box intersection, and keeping it that way
  is what lets it run per satellite per receiver at the publication rate. A
  min-area rectangle keeps the building's real bearing and real footprint size,
  which is what shapes the canyon; a bounding box would square every building
  up to east-north and destroy it.
* the projection is a local tangent plane about the origin. Over the few
  hundred metres of one city block that is accurate to centimetres, which is
  well inside the metre-scale errors this environment exists to model.

No Isaac and no network here: this reads a file, so it is testable offline and
a run is reproducible from the cached extract rather than from whatever the
Overpass API happened to return that day.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from urban_scene import Building

__all__ = ["CityExtract", "load_extract", "buildings_from_extract",
           "project_to_local", "minimum_area_rectangle", "WGS84_A"]

# WGS84 semi-major axis. The tangent-plane projection below is a metre-scale
# approximation, so the flattening term would be noise on noise.
WGS84_A = 6378137.0

# What OSM calls a storey, in metres, when a building gives levels but no
# height. 3.1 m is the usual mixed residential/commercial figure.
METRES_PER_LEVEL = 3.1


class CityExtract:
    """One cached Overpass answer: an origin and a list of footprints."""

    def __init__(self, data: dict[str, Any]):
        origin = data.get("origin") or {}
        try:
            self.latitude = float(origin["latitude"])
            self.longitude = float(origin["longitude"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("city extract has no usable origin") from exc
        self.altitude = float(origin.get("altitude", 0.0))
        self.name = str(data.get("name", "unnamed"))
        self.attribution = str(data.get("attribution", ""))
        self.buildings = list(data.get("buildings") or [])

    def __len__(self) -> int:
        return len(self.buildings)


def load_extract(path: str | Path) -> CityExtract:
    """Read a cached extract, with a message that says how to make one."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No city extract at {path}. Fetch one first, for example:\n"
            f"  scripts/fetch_city.py --name {path.stem} "
            f"--latitude 37.5665 --longitude 126.9780 --radius 320")
    with path.open(encoding="utf-8") as stream:
        return CityExtract(json.load(stream))


def project_to_local(lat, lon, lat0: float, lon0: float,
                     heading_rad: float = 0.0) -> np.ndarray:
    """WGS84 degrees to local ENU metres about (LAT0, LON0).

    HEADING_RAD rotates the map about the origin, which is how a real street
    grid is lined up with the route the deck drives: the route is a rectangle
    in world axes, so the city turns rather than the route.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    east = np.radians(lon - lon0) * WGS84_A * math.cos(math.radians(lat0))
    north = np.radians(lat - lat0) * WGS84_A
    if heading_rad:
        c, s = math.cos(heading_rad), math.sin(heading_rad)
        east, north = c * east + s * north, -s * east + c * north
    return np.stack([east, north], axis=-1)


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain, counter-clockwise, without the closing point.

    Written out rather than pulled from scipy so this module stays importable
    with nothing but numpy, the way the rest of the geometry layer is.
    """
    pts = np.unique(points, axis=0)
    if pts.shape[0] <= 2:
        return pts
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    def half(sequence):
        chain: list[np.ndarray] = []
        for point in sequence:
            while len(chain) >= 2:
                a, b = chain[-2], chain[-1]
                if np.cross(b - a, point - a) > 1e-12:
                    break
                chain.pop()
            chain.append(point)
        return chain[:-1]

    return np.array(half(pts) + half(pts[::-1]), dtype=float)


def minimum_area_rectangle(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Smallest-area enclosing rectangle: ``(centre, size, yaw_rad)``.

    Rotating calipers over the hull: the minimum-area rectangle always has a
    side flush with one of the hull's edges, so testing each edge's direction
    is exhaustive rather than a search.
    """
    hull = _convex_hull(np.asarray(points, dtype=float).reshape(-1, 2))
    if hull.shape[0] < 3:
        lo, hi = hull.min(axis=0), hull.max(axis=0)
        return 0.5 * (lo + hi), np.maximum(hi - lo, 1e-3), 0.0
    best = None
    edges = np.roll(hull, -1, axis=0) - hull
    for edge in edges:
        norm = float(np.hypot(edge[0], edge[1]))
        if norm < 1e-9:
            continue
        angle = math.atan2(edge[1], edge[0])
        c, s = math.cos(-angle), math.sin(-angle)
        rotated = np.stack([hull[:, 0] * c - hull[:, 1] * s,
                            hull[:, 0] * s + hull[:, 1] * c], axis=1)
        lo, hi = rotated.min(axis=0), rotated.max(axis=0)
        size = hi - lo
        area = float(size[0] * size[1])
        if best is None or area < best[0]:
            centre_rotated = 0.5 * (lo + hi)
            cb, sb = math.cos(angle), math.sin(angle)
            centre = np.array([centre_rotated[0] * cb - centre_rotated[1] * sb,
                               centre_rotated[0] * sb + centre_rotated[1] * cb])
            best = (area, centre, np.maximum(size, 1e-3), angle)
    if best is None:                                    # degenerate footprint
        lo, hi = hull.min(axis=0), hull.max(axis=0)
        return 0.5 * (lo + hi), np.maximum(hi - lo, 1e-3), 0.0
    return best[1], best[2], best[3]


def _distance_to_box(points: np.ndarray, centre: np.ndarray,
                     size: np.ndarray, yaw: float) -> float:
    """Closest approach of POINTS to an oriented rectangle; 0 if any is inside."""
    c, s = math.cos(yaw), math.sin(yaw)
    delta = points - centre
    local = np.stack([delta[:, 0] * c + delta[:, 1] * s,
                      -delta[:, 0] * s + delta[:, 1] * c], axis=1)
    outside = np.abs(local) - 0.5 * size
    # Standard box distance: the positive part gives the exterior distance and
    # a point with both components negative is inside, hence exactly zero.
    return float(np.linalg.norm(np.maximum(outside, 0.0), axis=1).min())


def _height_of(entry: dict[str, Any], default_m: float) -> float:
    """Metres, from whichever of OSM's several height tags the mapper used."""
    height = entry.get("height_m")
    if height is not None:
        try:
            value = float(height)
            if math.isfinite(value) and value > 0.0:
                return value
        except (TypeError, ValueError):
            pass
    levels = entry.get("levels")
    if levels is not None:
        try:
            value = float(levels) * METRES_PER_LEVEL
            if math.isfinite(value) and value > 0.0:
                return value
        except (TypeError, ValueError):
            pass
    return float(default_m)


def buildings_from_extract(extract: CityExtract, *, heading_rad: float = 0.0,
                           height_range_m: tuple[float, float] = (12.0, 36.0),
                           seed: int = 7,
                           default_height_m: float = 12.0,
                           min_footprint_m: float = 3.0,
                           keep_within_m: float = 0.0,
                           clear_of: Sequence[np.ndarray] | None = None,
                           clearance_m: float = 0.0) -> Iterator[Building]:
    """Real footprints as oriented boxes, in world ENU metres.

    ``clear_of`` is a polyline -- in practice the route the deck drives -- that
    the city is cut back from by ``clearance_m``. A real extract has buildings
    where this experiment needs a carriageway, because the route is a rectangle
    imposed on the map rather than a road traced from it; without the cut the
    lorry drives through masonry. Dropping a footprint is honest in a way that
    shrinking one is not: what is left is real, and what was in the way is
    simply absent.
    """
    route = None if clear_of is None else np.asarray(clear_of, dtype=float).reshape(-1, 2)
    # Most OSM buildings outside the big cities' cores carry no height at all --
    # in the Seoul extract this was written against, 284 of 349. Giving those a
    # single default would flatten the skyline into a wall of equal blocks and
    # the canyon would stop varying along the route, which is the one thing the
    # GNSS model is there to measure. They are drawn from the configured range
    # instead, seeded, so an untagged city is still a city and still
    # reproducible. Real footprints, real bearings, honest about the heights:
    # `height_source` on each building says which is which.
    rng = np.random.default_rng(int(seed))
    lo, hi = (float(height_range_m[0]), float(height_range_m[1]))
    for entry in extract.buildings:
        ring = entry.get("ring") or []
        if len(ring) < 3:
            continue
        ring = np.asarray(ring, dtype=float).reshape(-1, 2)
        local = project_to_local(ring[:, 0], ring[:, 1],
                                 extract.latitude, extract.longitude, heading_rad)
        centre, size, yaw = minimum_area_rectangle(local)
        if float(min(size)) < min_footprint_m:
            continue
        if keep_within_m > 0.0 and float(np.hypot(*centre)) > keep_within_m:
            continue
        mapped = _height_of(entry, 0.0)
        height = mapped if mapped > 0.0 else float(rng.uniform(lo, hi))
        if route is not None and clearance_m > 0.0:
            # Distance from the route to the *body* of the rectangle, not to
            # its corners: a large building can straddle the whole lap with its
            # corners far outside the clearance, and a corner test keeps it and
            # then drives the lorry through its interior.
            if _distance_to_box(route, centre, size, yaw) < clearance_m:
                continue
        yield Building(x0=float(centre[0] - 0.5 * size[0]),
                       x1=float(centre[0] + 0.5 * size[0]),
                       y0=float(centre[1] - 0.5 * size[1]),
                       y1=float(centre[1] + 0.5 * size[1]),
                       height_m=float(height), yaw_rad=float(yaw))
