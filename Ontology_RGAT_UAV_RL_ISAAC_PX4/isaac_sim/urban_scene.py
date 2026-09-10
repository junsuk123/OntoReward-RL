#!/usr/bin/env python3
"""The city block the episode is flown in: geometry first, USD second.

Isaac is not imported at module scope, so ``UrbanLayout`` is testable without a
simulator -- which matters more here than for the pad, because the *same*
geometry has two consumers that must not be allowed to disagree:

* the stage the camera renders, so the drone sees the buildings it flies
  between;
* the sky mask the GNSS model occludes satellites with (``gnss.py``).

A skyline drawn from one set of boxes and an occlusion mask computed from
another would produce an urban-canyon experiment whose GNSS outages have
nothing to do with the visible city, so both read this class.

Layout
------
One city block, driven around. The route (``pad_motion.RoadRoute``) is a
rounded rectangle in the middle of the streets that surround the block; the
buildings are the block itself and the facades on the far side of each of the
four streets, cut by cross-street gaps. Driving the loop therefore takes the
deck through four canyons at two orientations plus four open intersections,
which is what makes satellite visibility change during an episode instead of
being one static number per run.

Frames: world ENU, origin at the block centre, +X east, +Y north, +Z up.
Buildings are axis-aligned boxes, which is what keeps the occlusion test exact
and cheap enough to run per satellite per receiver at the physics rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np


@dataclass(frozen=True)
class Building:
    """One block of a building, in world ENU metres.

    ``x0..y1`` is the footprint before rotation and ``yaw_rad`` turns it about
    its own centre. The synthetic block leaves the yaw at zero and is therefore
    axis-aligned exactly as before; real footprints imported from OpenStreetMap
    carry the bearing their street actually runs at, which a bounding box would
    throw away -- and with it the canyon, because a facade that has been
    squared up to east-north no longer channels the sky or the wind the way the
    real one does.
    """
    x0: float
    x1: float
    y0: float
    y1: float
    height_m: float
    yaw_rad: float = 0.0

    @property
    def center(self) -> tuple[float, float]:
        return (0.5 * (self.x0 + self.x1), 0.5 * (self.y0 + self.y1))

    @property
    def size(self) -> tuple[float, float]:
        return (self.x1 - self.x0, self.y1 - self.y0)


@dataclass(frozen=True)
class UrbanConfig:
    enabled: bool
    seed: int
    # Route rectangle: the straight-line extents of the loop the deck drives.
    block_size_m: tuple[float, float]
    corner_radius_m: float
    # Street cross-section.
    road_half_width_m: float
    sidewalk_m: float
    lane_width_m: float
    # Buildings.
    height_range_m: tuple[float, float]
    facade_depth_m: float
    facade_segment_m: float
    cross_street_gap_m: float
    outer_rows: int
    row_spacing_m: float
    mid_block_gap: bool
    # Where the skyline comes from: 'synthetic' is the generated block below,
    # 'osm' is a cached extract of a real place (see isaac_sim/osm_city.py).
    source: str = "synthetic"
    extract: str = ""
    # The real world is not aligned to the route rectangle, so the map turns
    # under it: this is the bearing of the street the deck should drive along.
    map_heading_deg: float = 0.0
    import_radius_m: float = 160.0
    route_clearance_m: float = 11.0
    # Where on Earth the world origin is. PX4's home and the GNSS geometry are
    # set from this, so 'a real country' means these numbers and not the
    # buildings alone: the constellation's elevations depend on latitude.
    latitude_deg: float = 0.0
    longitude_deg: float = 0.0
    altitude_m: float = 0.0

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "UrbanConfig":
        urban = (data.get("urban") or {}) if isinstance(data, dict) else {}
        origin = (urban.get("origin") or {}) if isinstance(urban, dict) else {}
        size = tuple(float(v) for v in urban.get("block_size_m", (90.0, 60.0)))
        if len(size) != 2 or min(size) <= 0.0:
            raise ValueError("urban.block_size_m must be two positive lengths")
        heights = tuple(float(v) for v in urban.get("height_range_m", (12.0, 34.0)))
        if len(heights) != 2 or not 0.0 < heights[0] <= heights[1]:
            raise ValueError("urban.height_range_m must be two ordered positive heights")
        return cls(
            enabled=bool(urban.get("enabled", True)),
            seed=int(urban.get("seed", 7)),
            block_size_m=size,
            corner_radius_m=float(urban.get("corner_radius_m", 12.0)),
            road_half_width_m=float(urban.get("road_half_width_m", 7.0)),
            sidewalk_m=float(urban.get("sidewalk_m", 2.5)),
            lane_width_m=float(urban.get("lane_width_m", 3.3)),
            height_range_m=heights,
            facade_depth_m=float(urban.get("facade_depth_m", 14.0)),
            facade_segment_m=float(urban.get("facade_segment_m", 18.0)),
            cross_street_gap_m=float(urban.get("cross_street_gap_m", 16.0)),
            outer_rows=int(urban.get("outer_rows", 1)),
            row_spacing_m=float(urban.get("row_spacing_m", 4.0)),
            mid_block_gap=bool(urban.get("mid_block_gap", True)),
            source=str(urban.get("source", "synthetic")).lower(),
            extract=str(urban.get("extract", "")),
            map_heading_deg=float(urban.get("map_heading_deg", 0.0)),
            import_radius_m=float(urban.get("import_radius_m", 160.0)),
            route_clearance_m=float(urban.get("route_clearance_m", 11.0)),
            latitude_deg=float(origin.get("latitude", 0.0)),
            longitude_deg=float(origin.get("longitude", 0.0)),
            altitude_m=float(origin.get("altitude", 0.0)),
        )

    @property
    def street_half_span_m(self) -> float:
        """Route centre to the near facade: carriageway plus footway."""
        return self.road_half_width_m + self.sidewalk_m


class UrbanLayout:
    """The buildings, and what they hide.

    ``blocked`` and ``sky_mask`` are the only things the GNSS model needs, and
    ``buildings`` is the only thing the stage needs, so the two views cannot
    drift apart.
    """

    def __init__(self, cfg: UrbanConfig):
        self.cfg = cfg
        self.extract_name = ""
        self.attribution = ""
        self._buildings: tuple[Building, ...] = tuple(self._generate())
        # The occlusion test runs once per satellite per receiver per update,
        # so the boxes are held as arrays and tested all at once.
        box = np.array([[b.x0, b.x1, b.y0, b.y1, b.height_m, b.yaw_rad]
                        for b in self._buildings], dtype=float).reshape(-1, 6)
        self._lo = box[:, (0, 2)]
        self._hi = box[:, (1, 3)]
        self._height = box[:, 4]
        # Oriented form, which is what the ray test actually uses: a box is
        # tested in its own frame, so a rotated footprint costs one rotation of
        # the ray rather than a separate code path.
        self._centre = 0.5 * (self._lo + self._hi)
        self._half = 0.5 * (self._hi - self._lo)
        self._cos = np.cos(box[:, 5])
        self._sin = np.sin(box[:, 5])

    # ------------------------------------------------------------ geometry
    @property
    def buildings(self) -> tuple[Building, ...]:
        return self._buildings

    def _generate(self) -> Iterator[Building]:
        cfg = self.cfg
        if not cfg.enabled:
            return
        if cfg.source == "osm":
            yield from self._from_extract()
            return
        yield from self._synthetic()

    def _route_polyline(self, samples: int = 720) -> np.ndarray:
        """The centreline the deck drives, as points, in world ENU.

        Sampled from ``pad_motion.RoadRoute`` itself rather than rebuilt from
        the same numbers here: the route the city is cut back from has to be
        the route the lorry actually drives, and two copies of a rounded
        rectangle are two things that can disagree.
        """
        from pad_motion import RoadRoute

        route = RoadRoute(self.cfg.block_size_m[0], self.cfg.block_size_m[1],
                          self.cfg.corner_radius_m)
        arc = np.linspace(0.0, route.perimeter, int(samples), endpoint=False)
        return np.array([route.at(float(s))[0][:2] for s in arc], dtype=float)

    def _from_extract(self) -> Iterator[Building]:
        """Real footprints, cut back from the carriageway the deck needs."""
        from osm_city import buildings_from_extract, load_extract

        root = Path(__file__).resolve().parents[1]
        name = self.cfg.extract or "city"
        extract = load_extract(root / "assets" / "city" / f"{name}.json")
        self.extract_name = extract.name
        self.attribution = extract.attribution
        yield from buildings_from_extract(
            extract,
            heading_rad=math.radians(self.cfg.map_heading_deg),
            height_range_m=self.cfg.height_range_m,
            seed=self.cfg.seed,
            keep_within_m=self.cfg.import_radius_m,
            clear_of=self._route_polyline(),
            clearance_m=self.cfg.route_clearance_m)

    def _synthetic(self) -> Iterator[Building]:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        lo, hi = cfg.height_range_m
        half_x, half_y = (0.5 * cfg.block_size_m[0], 0.5 * cfg.block_size_m[1])
        inner = cfg.street_half_span_m
        depth = cfg.facade_depth_m
        # --- the block the route encircles ------------------------------
        # Split into slabs of different height, so the inner facade has the
        # broken skyline a real block does rather than one flat wall.
        bx0, bx1 = -half_x + inner, half_x - inner
        by0, by1 = -half_y + inner, half_y - inner
        if bx1 - bx0 > 1.0 and by1 - by0 > 1.0:
            count = max(1, int(round((bx1 - bx0) / cfg.facade_segment_m)))
            for x0, x1 in zip(np.linspace(bx0, bx1, count + 1)[:-1],
                              np.linspace(bx0, bx1, count + 1)[1:]):
                yield Building(float(x0), float(x1), by0, by1,
                               float(rng.uniform(lo, hi)))
        # --- the far side of each of the four streets --------------------
        # The gaps are the cross streets, so they are cut where the cross
        # streets actually are -- at the corners the route turns through, plus
        # a mid-block one. Aligning them with the geometry rather than spacing
        # them evenly is what makes the sky open up at the intersections, which
        # is the only reason satellite visibility changes during a lap.
        gaps_x = self._gap_centers(half_x)
        gaps_y = self._gap_centers(half_y)
        for row in range(max(0, cfg.outer_rows)):
            offset = row * (depth + cfg.row_spacing_m)
            near_x = half_x + inner + offset
            near_y = half_y + inner + offset
            for sign in (-1.0, 1.0):
                y0 = near_y if sign > 0 else -near_y - depth
                for x0, x1 in self._segments(-near_x - depth, near_x + depth, gaps_x):
                    yield Building(x0, x1, float(y0), float(y0 + depth),
                                   float(rng.uniform(lo, hi)))
                x0 = near_x if sign > 0 else -near_x - depth
                for y0, y1 in self._segments(-near_y, near_y, gaps_y):
                    yield Building(float(x0), float(x0 + depth), y0, y1,
                                   float(rng.uniform(lo, hi)))

    def _gap_centers(self, half: float) -> tuple[float, ...]:
        """Where the cross streets cut a facade row: the corners, and midblock."""
        centers = (-half, half)
        return centers + (0.0,) if self.cfg.mid_block_gap else centers

    def _segments(self, start: float, stop: float,
                  gap_centers: tuple[float, ...]) -> Iterator[tuple[float, float]]:
        """Facade slabs along one street, with the cross-street gaps cut out."""
        half_gap = 0.5 * self.cfg.cross_street_gap_m
        edges = [start]
        for center in sorted(gap_centers):
            if start < center < stop:
                edges.extend((center - half_gap, center + half_gap))
        edges.append(stop)
        for a, b in zip(edges[0::2], edges[1::2]):
            if b - a > 1.0:
                yield (float(a), float(b))

    # ----------------------------------------------------------- occlusion
    def blocked_batch(self, position, azimuth_rad, elevation_rad):
        """Vectorised :meth:`blocked` over a whole constellation.

        Returns ``(blocked[S], distance_m[S])``. One call per receiver per
        update tests every satellite against every box, which is what keeps the
        model affordable at the publication rate.
        """
        p = np.asarray(position, dtype=float).reshape(3)
        az = np.atleast_1d(np.asarray(azimuth_rad, dtype=float))
        el = np.atleast_1d(np.asarray(elevation_rad, dtype=float))
        ce = np.cos(el)
        d = np.stack([ce * np.cos(az), ce * np.sin(az)], axis=1)        # [S, 2]
        if self._height.size == 0:
            return np.zeros(az.shape, dtype=bool), np.zeros(az.shape)
        # Slab intersection in each box's own frame, per satellite per box.
        # Rotating the ray costs two multiplies and keeps one exact test for
        # both the axis-aligned synthetic block and a real, rotated skyline.
        d_local = self._to_box_frame(d[:, None, :])                     # [S, B, 2]
        p_local = self._to_box_frame(
            (p[:2][None, :] - self._centre)[None, :, :])                # [1, B, 2]
        half = self._half[None, :, :]
        safe = np.where(np.abs(d_local) < 1e-12, 1e-12, d_local)
        t_lo = (-half - p_local) / safe
        t_hi = (half - p_local) / safe
        enter = np.maximum(np.minimum(t_lo, t_hi).max(axis=2), 0.0)     # [S, B]
        exit_ = np.maximum(t_lo, t_hi).min(axis=2)
        # A ray parallel to an axis never leaves that slab, so it only crosses
        # the box when it is already inside it.
        parallel = np.abs(d_local) < 1e-12                              # [S, B, 2]
        inside = np.abs(p_local) <= half                                # [1, B, 2]
        keep = np.where(parallel, inside, True).all(axis=2)
        crosses = keep & (exit_ > enter)
        # The ray only climbs, so its lowest point inside the footprint is at
        # the entry: testing that one point is exact, not a sample.
        under = p[2] + enter * np.sin(el)[:, None] <= self._height[None, :]
        hit = crosses & under
        distance = np.where(hit, enter, np.inf).min(axis=1)
        return np.any(hit, axis=1), np.where(np.isfinite(distance), distance, 0.0)

    def _to_box_frame(self, vector):
        """Rotate an ENU horizontal vector into every box's own frame.

        VECTOR broadcasts against ``[.., B, 2]``; the result is that vector
        expressed in each building's footprint axes.
        """
        x, y = vector[..., 0], vector[..., 1]
        return np.stack([x * self._cos + y * self._sin,
                         -x * self._sin + y * self._cos], axis=-1)

    def blocked(self, position, azimuth_rad: float, elevation_rad: float
                ) -> tuple[bool, float]:
        """Is that line of sight cut, and how far away is the obstruction?

        AZIMUTH is measured from east toward north (ENU), ELEVATION up from the
        horizon. Returns ``(blocked, distance_m)``; the distance is to the
        nearest blocking facade, which is what the reflected-path model in
        ``gnss.py`` prices its excess delay with.

        The ray only climbs, so within the horizontal span where it crosses a
        box its lowest point is at the entry -- testing that one point is
        therefore exact, not a sample.
        """
        hit, distance = self.blocked_batch(position, [azimuth_rad], [elevation_rad])
        return bool(hit[0]), float(distance[0])

    def sky_mask(self, position, azimuth_rad: float,
                 resolution_deg: float = 2.0) -> float:
        """Lowest elevation with a clear view, in radians, at one azimuth.

        A bisection would be wrong on a non-convex skyline, so this is a scan.
        It exists for diagnostics and tests; the per-satellite path uses
        :meth:`blocked` directly and never pays for it.
        """
        step = math.radians(max(resolution_deg, 0.25))
        elevation = 0.0
        while elevation < 0.5 * math.pi:
            if not self.blocked(position, azimuth_rad, elevation)[0]:
                return elevation
            elevation += step
        return 0.5 * math.pi

    def contains(self, position) -> bool:
        """Is that point inside a building?

        The GUI camera needs this: a chase offset that fits an open field does
        not fit a nineteen-metre street, and a camera inside a facade shows a
        black screen with nothing to explain it.
        """
        p = np.asarray(position, dtype=float).reshape(3)
        if self._height.size == 0:
            return False
        local = self._to_box_frame((p[:2][None, :] - self._centre))
        inside = np.all(np.abs(local) <= self._half, axis=1)
        return bool(np.any(inside & (p[2] <= self._height) & (p[2] >= 0.0)))

    def clear_of_buildings(self, point, toward, fallback_height_m: float = 3.0):
        """Pull a point in along its own ray until it is out of the masonry.

        Walking the ray rather than nudging sideways keeps whatever the caller
        was expressing: a point offset from the lorry stays offset in the same
        direction, only less far. Two callers, and each of them was a real
        failure:

        * the GUI chase camera, where an offset that fits an open field does
          not fit a 19 m street, and a camera inside a facade renders a black
          screen with nothing in it to explain why;
        * the episode entry pose, where roughly one seed in three thousand puts
          the point PX4 is told to fly to inside a building -- and PX4 flies
          into it, misses the entry pose, and times out the whole reset.
        """
        point = np.asarray(point, dtype=float).reshape(3)
        toward = np.asarray(toward, dtype=float).reshape(3)
        if not self.contains(point):
            return point
        direction = point - toward
        for fraction in (0.8, 0.65, 0.5, 0.38, 0.28, 0.2, 0.14, 0.1):
            candidate = toward + fraction * direction
            if not self.contains(candidate):
                return candidate
        # Nothing on the ray is clear, which means the reference point itself is
        # inside a building. Sit above it rather than in the wall.
        return toward + np.array([0.0, 0.0, float(fallback_height_m)])

    def openness(self, position, samples: int = 16,
                 elevation_deg: float = 30.0) -> float:
        """A cheap sky-view proxy: how many probe rays at one elevation get out.

        :meth:`sky_view_fraction` is the honest solid-angle measure and costs a
        full elevation scan per azimuth, which is far too much to run every
        frame. This is one batched occlusion test, so it can drive the wind
        channeling at the publication rate whether or not the GNSS model is on.
        """
        azimuth = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
        elevation = np.full(samples, math.radians(elevation_deg))
        blocked, _ = self.blocked_batch(position, azimuth, elevation)
        return float(np.mean(~blocked))

    def sky_view_fraction(self, position, samples: int = 72) -> float:
        """Fraction of the hemisphere that is sky, weighted by solid angle.

        ``1 - cos(mask)`` is the solid angle a mask of that elevation hides, so
        this is the number a fisheye sky-view-factor measurement would report.
        """
        total = 0.0
        for i in range(samples):
            az = 2.0 * math.pi * i / samples
            total += math.cos(self.sky_mask(position, az))
        return float(total / samples)


def _slab_interval(origin: float, direction: float, lo: float, hi: float
                   ) -> tuple[float, float]:
    """Ray-parameter interval inside one axis slab, or an empty one."""
    if abs(direction) < 1e-12:
        return (-math.inf, math.inf) if lo <= origin <= hi else (math.inf, -math.inf)
    a = (lo - origin) / direction
    b = (hi - origin) / direction
    return (min(a, b), max(a, b))


# --------------------------------------------------------------------- USD
class UrbanScene:
    """The same layout, built into the Isaac stage.

    ``pxr`` is imported inside :meth:`spawn` so that importing this module --
    which the GNSS model and the tests do -- never needs Omniverse.
    """

    ROOT = "/World/city"

    def __init__(self, layout: UrbanLayout):
        self.layout = layout

    def spawn(self, world) -> None:
        from pxr import Gf, UsdGeom, UsdPhysics

        cfg = self.layout.cfg
        if not cfg.enabled:
            return
        stage = world.stage
        UsdGeom.Xform.Define(stage, self.ROOT)
        self._road(stage, Gf, UsdGeom)
        for index, building in enumerate(self.layout.buildings):
            path = f"{self.ROOT}/building_{index:03d}"
            cube = UsdGeom.Cube.Define(stage, path)
            cube.CreateSizeAttr(1.0)
            width, depth = building.size
            api = UsdGeom.XformCommonAPI(cube)
            api.SetScale(Gf.Vec3f(float(width), float(depth), float(building.height_m)))
            cx, cy = building.center
            api.SetTranslate(Gf.Vec3d(float(cx), float(cy), 0.5 * building.height_m))
            # Real footprints carry the bearing of the street they stand on, so
            # the stage has to turn them the same way the occlusion test does.
            api.SetRotate(Gf.Vec3f(0.0, 0.0, float(math.degrees(building.yaw_rad))))
            # Enough tonal variation that the facades read as separate buildings
            # from the air; a uniform grey city gives the camera no edges to
            # resolve and the viewport no depth.
            shade = 0.42 + 0.34 * ((index * 0.6180339887) % 1.0)
            cube.CreateDisplayColorAttr(
                [Gf.Vec3f(float(shade), float(shade * 0.97), float(shade * 0.92))])
            # Collidable: a policy that flies into a facade must hit it rather
            # than pass through and land as if the city were a painting.
            UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(path))

    def _road(self, stage, Gf, UsdGeom) -> None:
        """Two crossing carriageways, drawn flat on the ground plane.

        Visual only: the ground plane already carries the collider, and the
        deck is driven kinematically rather than by tyre friction.
        """
        cfg = self.layout.cfg
        half_x, half_y = (0.5 * cfg.block_size_m[0], 0.5 * cfg.block_size_m[1])
        outer = cfg.street_half_span_m + cfg.road_half_width_m
        spans = (
            ("road_ew", 2.0 * (half_x + outer), 2.0 * (half_y + cfg.road_half_width_m)),
            ("road_ns", 2.0 * (half_x + cfg.road_half_width_m), 2.0 * (half_y + outer)),
        )
        for name, size_x, size_y in spans:
            path = f"{self.ROOT}/{name}"
            slab = UsdGeom.Cube.Define(stage, path)
            slab.CreateSizeAttr(1.0)
            api = UsdGeom.XformCommonAPI(slab)
            api.SetScale(Gf.Vec3f(float(size_x), float(size_y), 0.02))
            # Just above the ground plane, so it reads as asphalt and does not
            # z-fight with it.
            api.SetTranslate(Gf.Vec3d(0.0, 0.0, 0.011))
