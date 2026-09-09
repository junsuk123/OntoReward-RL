#!/usr/bin/env python3
"""Kinematic trajectory of the road vehicle carrying the landing pad.

Isaac is not imported here, so the trajectory is testable without a simulator
and the gateway can reuse the same configuration parsing.

The deck is driven kinematically rather than as a physics-simulated vehicle.
Pegasus ships no ground-vehicle asset, and a driven UGV would put a second
controller between the seed and the trajectory the drone has to chase -- the
experiment wants a reproducible moving target, not a rover-control study. The
consequence to keep in mind: the deck is infinitely stiff and never reacts to
the drone landing on it.

Position and velocity are analytic, so the pad twist the policy feeds forward is
exact rather than finite-differenced. Heading is integrated under a rate limit,
because a real deck steers rather than snapping to its velocity, and because
every closed-form profile has cusps where the velocity direction reverses.

``road`` is the mode the urban experiment runs: a lorry driving a lap of the
city block, in lane, through traffic. It is built as a rounded rectangle
parameterised by arc length, so the route is exactly the road drawn in
``urban_scene.UrbanScene`` and both the position and the velocity stay closed
form through the corners. The three older profiles are kept because ``static``
is the fixed-pad control condition the earlier results were measured against.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


_erf = np.vectorize(math.erf)

MODES = ("static", "constant", "circular", "lissajous", "road")


@dataclass(frozen=True)
class PadMotionConfig:
    mode: str
    start_position_enu_m: tuple[float, float, float]
    deck_height_m: float
    deck_size_m: tuple[float, float]
    speed_min_m_s: float
    speed_max_m_s: float
    circular_radius_m: float
    lissajous_amplitude_m: tuple[float, float]
    lissajous_frequency_hz: tuple[float, float]
    heading_follows_velocity: bool
    yaw_rate_limit_rad_s: float
    arena_radius_m: float
    # --- road mode -------------------------------------------------------
    # The route rectangle is the street the city was built around, so these
    # default to the urban block rather than to numbers of their own: a deck
    # driving a road that is not where the buildings are would give the GNSS
    # model obstructions the camera never sees.
    route_size_m: tuple[float, float]
    route_corner_radius_m: float
    # Where on the lap an episode begins. 'continue' leaves the lorry where it
    # is and only reseeds how it drives from there; 'seeded' draws a fresh
    # point on the route, which makes the whole initial condition a function of
    # the seed but teleports the deck up to half a lap away from the vehicle
    # parked on it.
    route_start: str
    # Lane centres on the vehicle's own side of the carriageway, as offsets
    # from the route centreline. Right-hand traffic and a counter-clockwise lap
    # put those on the negative side; a lane change moves between them and
    # never into the oncoming lanes.
    lane_centres_m: tuple[float, ...]
    lane_wander_m: float
    lane_wander_hz: float
    lane_change_probability: float
    stop_interval_s: float
    stop_sigma_s: float
    stop_depth_range: tuple[float, float]

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "PadMotionConfig":
        pad = data.get("pad", {}) or {}
        mode = str(pad.get("motion", "static")).lower()
        if mode not in MODES:
            raise ValueError(f"pad.motion must be one of {MODES}, got {mode!r}")
        low, high = (float(v) for v in pad.get("speed_range_m_s", (0.0, 0.0)))
        if not 0.0 <= low <= high:
            raise ValueError("pad.speed_range_m_s must be non-negative and ordered")
        start = tuple(float(v) for v in pad.get("start_position_enu_m", (0.0, 0.0, 0.0)))
        if len(start) != 3:
            raise ValueError("pad.start_position_enu_m must have three components")
        urban = (data.get("urban") or {}) if isinstance(data, dict) else {}
        lane_width = float(urban.get("lane_width_m", 3.3))
        route = tuple(float(v) for v in pad.get(
            "route_size_m", urban.get("block_size_m", (90.0, 60.0))))
        if len(route) != 2 or min(route) <= 0.0:
            raise ValueError("pad.route_size_m must be two positive lengths")
        stops = tuple(float(v) for v in pad.get("stop_depth_range", (0.35, 1.0)))
        if len(stops) != 2 or not 0.0 <= stops[0] <= stops[1] <= 1.0:
            raise ValueError("pad.stop_depth_range must be ordered inside [0,1]")
        # One lane centre per lane that fits on this side of the carriageway,
        # counted out from the centreline. An explicit pad.lane_offset_m pins
        # the lorry to one lane instead.
        half_road = float(urban.get("road_half_width_m", 7.0))
        deck_width = float(pad.get("deck_size_m", (6.2, 2.45))[1])
        if "lane_offset_m" in pad:
            lanes = (float(pad["lane_offset_m"]),)
        else:
            count = max(1, int(half_road // lane_width))
            lanes = tuple(-(k + 0.5) * lane_width for k in range(count))
        # A lane the vehicle does not fit in is a configuration error, not
        # something to discover as a lorry clipping a parked car.
        if lanes and abs(min(lanes)) + 0.5 * deck_width > half_road + 1e-9:
            raise ValueError(
                "pad.deck_size_m does not fit the outer lane implied by "
                "urban.road_half_width_m and urban.lane_width_m")
        route_start = str(pad.get("route_start", "continue")).lower()
        if route_start not in ("continue", "seeded"):
            raise ValueError("pad.route_start must be 'continue' or 'seeded'")
        interval = float(pad.get("stop_interval_s", 11.0))
        sigma = float(pad.get("stop_sigma_s", 1.7))
        if mode == "road" and interval < 4.0 * sigma:
            # Two overlapping dips could sum past one and drive the profile
            # backwards, which is not traffic, it is reverse gear.
            raise ValueError("pad.stop_interval_s must be at least 4x pad.stop_sigma_s")
        return cls(
            mode=mode,
            start_position_enu_m=start,
            deck_height_m=float(pad.get("deck_height_m", 0.0)),
            deck_size_m=tuple(float(v) for v in pad.get("deck_size_m", (1.3, 0.9))),
            speed_min_m_s=low,
            speed_max_m_s=high,
            circular_radius_m=float(pad.get("circular_radius_m", 4.0)),
            lissajous_amplitude_m=tuple(
                float(v) for v in pad.get("lissajous_amplitude_m", (4.5, 3.0))),
            lissajous_frequency_hz=tuple(
                float(v) for v in pad.get("lissajous_frequency_hz", (0.045, 0.07))),
            heading_follows_velocity=bool(pad.get("heading_follows_velocity", True)),
            yaw_rate_limit_rad_s=math.radians(float(pad.get("yaw_rate_limit_deg_s", 60.0))),
            arena_radius_m=float(pad.get("arena_radius_m", 8.0)),
            route_size_m=route,
            route_start=route_start,
            route_corner_radius_m=float(pad.get("route_corner_radius_m",
                                                urban.get("corner_radius_m", 12.0))),
            lane_centres_m=lanes,
            lane_wander_m=float(pad.get("lane_wander_m", 0.18)),
            lane_wander_hz=float(pad.get("lane_wander_hz", 0.09)),
            lane_change_probability=float(pad.get("lane_change_probability", 0.5)),
            stop_interval_s=float(pad.get("stop_interval_s", 11.0)),
            stop_sigma_s=float(pad.get("stop_sigma_s", 1.7)),
            stop_depth_range=stops,
        )

    @property
    def is_static(self) -> bool:
        return self.mode == "static" or self.speed_max_m_s <= 0.0


class RoadRoute:
    """The lap of the block, parameterised by arc length.

    A rounded rectangle: four straights joined by quarter circles, traversed
    counter-clockwise, which for right-hand traffic is a lap made of left turns
    with the block on the inside. Arc-length parameterisation is what keeps the
    velocity exact: position, unit tangent and curvature are all closed form at
    every ``s``, so the twist the policy feeds forward is differentiated
    analytically rather than sampled.
    """

    def __init__(self, size_x: float, size_y: float, radius: float):
        half_x, half_y = 0.5 * float(size_x), 0.5 * float(size_y)
        radius = float(np.clip(radius, 0.0, min(half_x, half_y) - 1e-6))
        self.half = (half_x, half_y)
        self.radius = radius
        self.straight = (size_x - 2.0 * radius, size_y - 2.0 * radius)
        self.arc = 0.5 * math.pi * radius
        sx, sy = self.straight
        # Cumulative arc length at the end of each of the eight segments.
        self.edges = np.cumsum([sx, self.arc, sy, self.arc, sx, self.arc, sy, self.arc])
        self.perimeter = float(self.edges[-1])

    def at(self, s: float) -> tuple[np.ndarray, np.ndarray, float]:
        """``(point, unit tangent, signed curvature)`` at arc length ``s``."""
        half_x, half_y = self.half
        r = self.radius
        sx, sy = self.straight
        u = float(s) % self.perimeter if self.perimeter > 0.0 else 0.0
        index = int(np.searchsorted(self.edges, u, side="right"))
        index = min(index, 7)
        local = u - (0.0 if index == 0 else float(self.edges[index - 1]))
        if index == 0:                       # south straight, heading east
            return np.array([-0.5 * sx + local, -half_y]), np.array([1.0, 0.0]), 0.0
        if index == 2:                       # east straight, heading north
            return np.array([half_x, -0.5 * sy + local]), np.array([0.0, 1.0]), 0.0
        if index == 4:                       # north straight, heading west
            return np.array([0.5 * sx - local, half_y]), np.array([-1.0, 0.0]), 0.0
        if index == 6:                       # west straight, heading south
            return np.array([-half_x, 0.5 * sy - local]), np.array([0.0, -1.0]), 0.0
        centers = {1: (half_x - r, -half_y + r), 3: (half_x - r, half_y - r),
                   5: (-half_x + r, half_y - r), 7: (-half_x + r, -half_y + r)}
        start_angle = {1: -0.5 * math.pi, 3: 0.0, 5: 0.5 * math.pi, 7: math.pi}[index]
        theta = start_angle + (local / r if r > 1e-9 else 0.0)
        center = np.asarray(centers[index], dtype=float)
        point = center + r * np.array([math.cos(theta), math.sin(theta)])
        tangent = np.array([-math.sin(theta), math.cos(theta)])
        return point, tangent, (1.0 / r if r > 1e-9 else 0.0)

    @staticmethod
    def left_normal(tangent: np.ndarray) -> np.ndarray:
        return np.array([-tangent[1], tangent[0]])


class PadTrajectory:
    """Where the deck is, how fast, and which way it is pointing."""

    def __init__(self, cfg: PadMotionConfig):
        self.cfg = cfg
        self.start = np.asarray(cfg.start_position_enu_m, dtype=float)
        self.speed = 0.0
        self.heading0 = 0.0
        self.phase = np.zeros(2)
        self.t0 = 0.0
        self.yaw = 0.0
        self._yaw_initialised = False
        self.route = RoadRoute(cfg.route_size_m[0], cfg.route_size_m[1],
                               cfg.route_corner_radius_m)
        # Road-mode episode draw: where on the lap the lorry starts, when the
        # traffic ahead of it stops, how hard, and when it changes lane.
        self.s0 = 0.0
        self.stop_phase = 0.0
        self.stop_depths = np.zeros(0)
        self.lane_base = cfg.lane_centres_m[0] if cfg.lane_centres_m else 0.0
        self.lane_change_at = math.inf
        self.lane_change_to = 0.0
        self.wander_phase = 0.0
        self._driven = False

    def reset(self, seed: int, sim_time: float, speed_scale: float = 1.0) -> dict[str, Any]:
        """Draw this episode's deck motion from the episode seed.

        SPEED_SCALE multiplies the drawn speed, which is what an evaluation
        sweep varies to ask how fast a deck a policy can still land on. It is
        applied after the draw so the seed still picks the same point in the
        distribution at every scale, which is what makes the sweep paired.
        """
        rng = np.random.default_rng(int(seed) + 977)
        cfg = self.cfg
        # Where the lorry has got to so far -- along the road and across it --
        # read before the clock and the lane draw are overwritten.
        carried = self.arc_length(sim_time)
        carried_lane = self.lane_now(sim_time)
        self.speed = float(rng.uniform(cfg.speed_min_m_s, cfg.speed_max_m_s))
        self.speed *= max(float(speed_scale), 0.0)
        self.heading0 = float(rng.uniform(0.0, 2.0 * math.pi))
        self.phase = rng.uniform(0.0, 2.0 * math.pi, size=2)
        # Drawn either way, so the two modes consume the seed identically and a
        # sweep stays paired across them.
        seeded_s0 = float(rng.uniform(0.0, self.route.perimeter))
        self.s0 = carried if cfg.route_start == "continue" else seeded_s0
        self.stop_phase = float(rng.uniform(0.0, cfg.stop_interval_s))
        # A short cycle of stop depths, indexed modulo its length: the lorry
        # meets a different light each time without the profile needing an
        # unbounded draw.
        self.stop_depths = rng.uniform(cfg.stop_depth_range[0],
                                       cfg.stop_depth_range[1], size=16)
        # Which lane the lorry is in, and whether it changes to another one on
        # the same side. Both are drawn from the lanes that exist, so a lane
        # change can never put it into oncoming traffic.
        lanes = cfg.lane_centres_m or (0.0,)
        start = int(rng.integers(len(lanes)))
        seeded_lane = float(lanes[start])
        # A lorry that is being left where it stands stays in the lane it is
        # in -- including a lane change it has already made. Redrawing this
        # would slide the deck sideways by up to a full lane width under
        # whatever is parked on its roof.
        self.lane_base = (carried_lane if cfg.route_start == "continue"
                          and self._driven else seeded_lane)
        # Late enough that the smooth step is still flat at t = 0, so the
        # lateral position is continuous across the reset as well.
        self.lane_change_at = float(rng.uniform(6.0, 30.0))
        others = [v for v in lanes if abs(v - self.lane_base) > 1e-9]
        self.lane_change_to = (
            float(rng.choice(others)) - self.lane_base
            if others and rng.random() < cfg.lane_change_probability else 0.0)
        seeded_phase = float(rng.uniform(0.0, 2.0 * math.pi))
        if cfg.route_start == "continue" and self._driven:
            # Advance the phase by the time that has passed, so the wander picks
            # up exactly where it was rather than snapping back to its own t=0.
            self.wander_phase = float(
                (self.wander_phase
                 + 2.0 * math.pi * cfg.lane_wander_hz * (float(sim_time) - self.t0))
                % (2.0 * math.pi))
        else:
            self.wander_phase = seeded_phase
        self.t0 = float(sim_time)
        self._driven = True
        self._yaw_initialised = False
        position, velocity = self.pose(sim_time)
        self.yaw = self._velocity_heading(velocity, fallback=self.heading0)
        self._yaw_initialised = True
        return {
            "mode": cfg.mode,
            "speed_m_s": self.speed,
            "speed_scale": float(speed_scale),
            "heading_rad": self.heading0,
            "position_enu_m": position.tolist(),
            "velocity_enu_m_s": velocity.tolist(),
            "yaw_rad": self.yaw,
            "route_start": cfg.route_start,
            "arc_length_m": self.s0,
            "lane_offset_m": self.lane_base,
        }

    def lane_now(self, sim_time: float) -> float:
        """Where across the carriageway the lorry is, excluding its wander.

        The wander is carried by its own phase, so this is the part a reset has
        to fold in: a lane change the lorry has already completed is where it
        *is*, and dropping it would put the deck back in the lane it left.
        """
        cfg = self.cfg
        if cfg.mode != "road" or not cfg.lane_centres_m:
            return self.lane_base
        offset, _ = self._lane_offset(float(sim_time) - self.t0)
        omega = 2.0 * math.pi * cfg.lane_wander_hz
        if self.speed > 0.0:
            offset -= cfg.lane_wander_m * math.sin(
                omega * (float(sim_time) - self.t0) + self.wander_phase)
        return float(np.clip(offset, min(cfg.lane_centres_m), max(cfg.lane_centres_m)))

    def arc_length(self, sim_time: float) -> float:
        """How far along the route the lorry has driven, in metres.

        This is what ``route_start: continue`` carries across a reset, so the
        deck picks up where it left off instead of teleporting out from under
        the vehicle parked on it. ``_traffic`` covers zero distance at ``t = 0``
        by construction, so the carry is exact and the pose is continuous.
        """
        if self.cfg.mode != "road":
            return 0.0
        return self.s0 + self._traffic(float(sim_time) - self.t0)[0]

    def pose(self, sim_time: float) -> tuple[np.ndarray, np.ndarray]:
        """Analytic deck position and velocity in world ENU."""
        cfg = self.cfg
        t = float(sim_time) - self.t0
        offset = np.zeros(3)
        velocity = np.zeros(3)
        if not cfg.is_static:
            if cfg.mode == "constant":
                # A straight shuttle: a rover cannot drive out of the arena, and
                # a reversal is the honest bounded version of "constant speed".
                amplitude = max(0.5 * cfg.arena_radius_m, 1e-3)
                omega = self.speed / amplitude
                direction = np.array([math.cos(self.heading0), math.sin(self.heading0), 0.0])
                offset = direction * amplitude * math.sin(omega * t)
                velocity = direction * self.speed * math.cos(omega * t)
            elif cfg.mode == "circular":
                radius = max(cfg.circular_radius_m, 1e-3)
                omega = self.speed / radius
                angle = omega * t + self.heading0
                offset = np.array([radius * math.cos(angle), radius * math.sin(angle), 0.0])
                offset -= np.array([radius * math.cos(self.heading0),
                                    radius * math.sin(self.heading0), 0.0])
                velocity = np.array([-radius * omega * math.sin(angle),
                                     radius * omega * math.cos(angle), 0.0])
            elif cfg.mode == "road":
                point, velocity_xy = self._road_pose(t)
                position = np.array([self.start[0] + point[0],
                                     self.start[1] + point[1],
                                     self.start[2] + cfg.deck_height_m])
                # The route is bounded by construction, so the arena clamp that
                # keeps the open-field profiles inside their circle would only
                # be able to do harm here.
                return position, np.array([velocity_xy[0], velocity_xy[1], 0.0])
            elif cfg.mode == "lissajous":
                ax, ay = cfg.lissajous_amplitude_m
                fx, fy = cfg.lissajous_frequency_hz
                wx, wy = 2.0 * math.pi * fx, 2.0 * math.pi * fy
                # Scale both frequencies together so the drawn speed is the
                # profile's peak speed rather than an unrelated amplitude.
                peak = math.hypot(ax * wx, ay * wy)
                scale = (self.speed / peak) if peak > 1e-9 else 0.0
                wx, wy = wx * scale, wy * scale
                offset = np.array([ax * math.sin(wx * t + self.phase[0]),
                                   ay * math.sin(wy * t + self.phase[1]), 0.0])
                offset -= np.array([ax * math.sin(self.phase[0]),
                                    ay * math.sin(self.phase[1]), 0.0])
                velocity = np.array([ax * wx * math.cos(wx * t + self.phase[0]),
                                     ay * wy * math.cos(wy * t + self.phase[1]), 0.0])
        position = self.start + offset
        position[2] = self.start[2] + cfg.deck_height_m
        # Never let a profile drive the deck out of the arena the drone is
        # allowed to follow it into.
        radial = math.hypot(position[0], position[1])
        if radial > cfg.arena_radius_m > 0.0:
            position[0] *= cfg.arena_radius_m / radial
            position[1] *= cfg.arena_radius_m / radial
            velocity = np.zeros(3)
        return position, velocity

    # ------------------------------------------------------------ road mode
    def _road_pose(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """In-lane position and velocity on the block lap, both closed form."""
        cfg = self.cfg
        distance, speed = self._traffic(t)
        point, tangent, curvature = self.route.at(self.s0 + distance)
        normal = RoadRoute.left_normal(tangent)
        offset, offset_rate = self._lane_offset(t)
        # A curve offset by a constant e advances at (1 - kappa e) times the
        # centreline rate; the remaining term is the driver crossing the lane.
        position = point + offset * normal
        velocity = speed * (1.0 - curvature * offset) * tangent + offset_rate * normal
        return position, velocity

    def _traffic(self, t: float) -> tuple[float, float]:
        """Distance covered and current speed under stop-and-go traffic.

        The speed is the drawn cruise speed with a Gaussian dip at each light,
        deep enough to be a full stop when the draw says so. Both the dip and
        its integral are closed form -- ``erf`` is the integral of the Gaussian
        -- so the deck never has to be numerically integrated and the twist the
        policy feeds forward stays exact.
        """
        cfg = self.cfg
        if self.speed <= 0.0 or cfg.stop_interval_s <= 0.0:
            return 0.0, 0.0
        sigma = max(cfg.stop_sigma_s, 1e-3)
        last = int(math.floor((max(t, 0.0) + 6.0 * sigma - self.stop_phase)
                              / cfg.stop_interval_s))
        if last < 0 or self.stop_depths.size == 0:
            return self.speed * t, self.speed
        k = np.arange(0, min(last, 4000) + 1)
        centres = self.stop_phase + k * cfg.stop_interval_s
        depth = self.stop_depths[k % self.stop_depths.size]
        # Speed: cruise minus the dips. Distance: cruise time minus their
        # integrals, referenced to t = 0 so the lap starts where it is drawn.
        dip = depth * np.exp(-0.5 * ((t - centres) / sigma) ** 2)
        root2 = math.sqrt(2.0)
        area = depth * sigma * math.sqrt(0.5 * math.pi) * (
            _erf((t - centres) / (sigma * root2)) - _erf(-centres / (sigma * root2)))
        speed = self.speed * float(max(1.0 - dip.sum(), 0.0))
        return self.speed * (t - float(area.sum())), speed

    def _lane_offset(self, t: float) -> tuple[float, float]:
        """Where in the lane the lorry is, and how fast it is crossing it."""
        cfg = self.cfg
        if self.speed <= 0.0:
            # A parked lorry holds its lane. Without this the zero-speed sweep
            # point -- the static control condition -- would still creep
            # sideways, and it would no longer be a control condition.
            return self.lane_base, 0.0
        omega = 2.0 * math.pi * cfg.lane_wander_hz
        offset = self.lane_base + cfg.lane_wander_m * math.sin(omega * t + self.wander_phase)
        rate = cfg.lane_wander_m * omega * math.cos(omega * t + self.wander_phase)
        if self.lane_change_to != 0.0 and math.isfinite(self.lane_change_at):
            # A lane change is a smooth step, so the lateral velocity it adds is
            # a bump rather than an impulse.
            tau = 1.6
            u = (t - self.lane_change_at) / tau
            offset += self.lane_change_to * 0.5 * (1.0 + math.tanh(u))
            rate += self.lane_change_to * 0.5 / (tau * math.cosh(u) ** 2)
        return offset, rate

    def step_heading(self, velocity: np.ndarray, dt: float) -> tuple[float, float]:
        """Rate-limited deck heading, and the yaw rate actually applied."""
        cfg = self.cfg
        if not cfg.heading_follows_velocity or cfg.is_static:
            return self.yaw, 0.0
        target = self._velocity_heading(velocity, fallback=self.yaw)
        error = math.atan2(math.sin(target - self.yaw), math.cos(target - self.yaw))
        if dt <= 0.0:
            return self.yaw, 0.0
        limit = cfg.yaw_rate_limit_rad_s * dt
        applied = max(-limit, min(limit, error))
        self.yaw = math.atan2(math.sin(self.yaw + applied), math.cos(self.yaw + applied))
        return self.yaw, applied / dt

    @staticmethod
    def _velocity_heading(velocity: np.ndarray, fallback: float) -> float:
        # Below this the direction is numerically meaningless, and a cusp would
        # otherwise spin the deck through 180 degrees in one step.
        if math.hypot(float(velocity[0]), float(velocity[1])) < 0.05:
            return float(fallback)
        return math.atan2(float(velocity[1]), float(velocity[0]))
