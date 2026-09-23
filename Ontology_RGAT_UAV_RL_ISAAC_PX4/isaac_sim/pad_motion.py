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

``road`` is the mode the generated urban experiment runs: a lorry driving a
rounded-rectangle lap. ``waypoints`` follows a surveyed 3-D polyline; it either
reverses smoothly at its ends or continuously follows a configured closed loop.
Both modes report an analytic velocity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


_erf = np.vectorize(math.erf)

MODES = ("static", "constant", "circular", "lissajous", "road", "waypoints",
         "random_walk")
CARRIERS = ("lorry", "ugv")
# The training random walk with one straight dash laid over it: the deck
# accelerates to the carrier's scenario peak speed, holds it for
# ESCAPE_BURST_DISTANCE_M and eases back onto the walk. Written for the
# behaviour-cloning demonstrations (2026-09-21): the teacher follows the pad,
# the pad leaves the camera frame, the teacher climbs, re-acquires it and
# follows again, so the student sees a recovery it will need rather than only
# approaches that never lose the deck.
ESCAPE_BURST_SCENARIO = "training_random_walk_escape_burst"
# The same pull-away laid over a constant-velocity straight run instead of the
# random walk. This is the CICS2026 comparison scenario: the deck cruises in a
# straight line, the vehicle settles into following it, and then it accelerates
# hard enough to leave a downward camera's frame, which is the event the three
# arms are compared on. The walk-based variant above stays for the behaviour
# cloning demonstrations that need a varied approach before the same event.
STRAIGHT_ESCAPE_BURST_SCENARIO = "straight_escape_burst"
# The same deck on a closed oval instead of an open straight line: two straight
# segments joined by two constant-speed semicircles, driven forever.
#
# Why a closed track at all. ``straight_escape_burst`` drives a genuine
# straight line, but only within one episode. ``route_start: continue`` leaves
# the deck where the episode ended and the next one draws a fresh heading, so
# the episode *sequence* is a 2-D random walk whose distance from the origin
# grows without bound -- which is why ``reset`` has to rotate the drawn heading
# back toward the origin past half the arena, and why ``pose`` has to clamp at
# its edge. Both are corrections applied to the deck because it was going
# somewhere it must not go.
#
# A closed loop removes the reason for either. The deck never leaves a region
# of roughly ``straight + 2 * radius`` by ``2 * radius``, every episode
# continues the previous one along the same path rather than choosing a new
# direction, and the straight segments are still straight lines -- which is
# what the scenario is for. The turns are constant-speed circular arcs of a
# fixed radius, so the deck's speed never changes except for the dash itself.
STRAIGHT_ESCAPE_BURST_TRACK_SCENARIO = "straight_escape_burst_track"
# The three CICS2026 comparison decks. Each one is a straight line at a
# constant heading whose speed is piecewise constant: three segments, switching
# at SEGMENTED_CRUISE_SWITCH_TIMES_S, with a bounded acceleration between them
# so the change is a real vehicle changing speed and not a step in velocity.
#
# This is the scenario family of the reduced 2-D study, on this carrier. Its
# speeds are quoted in the study's units (1.0-7.0 m/s) and scaled by
# ``pad.segmented_cruise_speed_scale`` onto the RANGER MINI's 1.0 m/s ceiling,
# so the ratios between segments and between scenarios are exactly the study's
# while the fastest segment maps onto the fastest speed this carrier can drive.
#
# Why a straight line at a constant heading matters here and not merely as a
# simplification: the reduced control envelope pins the vehicle's lateral
# velocity to zero and holds its yaw, which keeps it centred on the deck's
# track *because the track does not turn*. A curved deck would make those two
# constraints wrong rather than merely restrictive. See docs/PLANAR_ENVELOPE.md.
SEGMENTED_CRUISE_SCENARIOS = (
    "segmented_cruise_slow", "segmented_cruise_medium", "segmented_cruise_fast",
)
# Segment boundaries, in seconds from the start of the episode. The first is
# early enough that no controller can land before it and skip the disturbance
# the scenario exists to apply.
SEGMENTED_CRUISE_SWITCH_TIMES_S = (3.0, 15.0)
# Per-scenario segment speeds, in the reduced study's units, before
# ``pad.segmented_cruise_speed_scale``.
SEGMENTED_CRUISE_SPEEDS_M_S = {
    "segmented_cruise_slow": (1.0, 4.0, 1.5),
    "segmented_cruise_medium": (1.5, 5.5, 2.0),
    "segmented_cruise_fast": (2.0, 7.0, 2.5),
}
# Maps the fastest segment of the fastest scenario (7.0) onto the carrier's
# 1.0 m/s ceiling. Named separately from ``benchmark_speed_scale`` so the
# retired decks, which are written against an 8.0 m/s peak, keep their own.
SEGMENTED_CRUISE_SPEED_SCALE = 1.0 / 7.0
# Bound on the speed change at a segment boundary, in the study's units
# (its ``ugvAccelMax``), scaled with the speeds.
SEGMENTED_CRUISE_ACCELERATION_M_S2 = 4.0
# The lane every segmented-cruise deck drives, in world ENU. One heading for
# every pair, so the pairs spawn in a row across the lane and set off in the
# same direction (``parallel.pair_offsets_enu_m`` places the row).
#
# Fixing it replaces the per-episode heading draw, which is what the open
# straight decks used and what made the episode sequence a 2-D random walk.
SEGMENTED_CRUISE_LANE_HEADING_DEG = 0.0
# How far down the lane a deck may get before the NEXT episode sends it back.
#
# A fixed heading cannot use the inward steering that bounds the random walk --
# there is no heading left to rotate -- so the lane is a shuttle instead: the
# deck drives one way until it has run out its leg, and the following episode
# starts it back the other way. The reversal happens BETWEEN episodes, so no
# episode ever contains one and every episode is a straight line at a constant
# heading, which is what the reduced control envelope depends on.
#
# 150 m is about 8 episodes of the fastest deck (18 m per 30 s episode), so the
# pairs stay pointed the same way for long stretches and the excursion is
# bounded at leg + one episode. It is not a physical constant: raise it for
# longer unbroken runs, at the cost of a wider arena.
SEGMENTED_CRUISE_LEG_M = 150.0

BENCHMARK_SCENARIOS = (
    "training_random_walk", "straight_8mps", "linear_acceleration_wave",
    "circle", "zigzag", "u_turn", "vertical_heave_boat",
    ESCAPE_BURST_SCENARIO, STRAIGHT_ESCAPE_BURST_SCENARIO,
    STRAIGHT_ESCAPE_BURST_TRACK_SCENARIO,
) + SEGMENTED_CRUISE_SCENARIOS
# Scenarios whose shape is a fixed closed path rather than a fresh heading per
# episode. They carry their phase across a reset instead of their heading, and
# the arena's inward steering never applies to them: the path is bounded by
# construction, so pulling it inward would only bend a straight into a curve.
CLOSED_TRACK_SCENARIOS = (STRAIGHT_ESCAPE_BURST_TRACK_SCENARIO,)
# How the dash is started (``pad.escape_burst_trigger``).
#   following -- Isaac fires it the moment the vehicle is actually following
#                the deck: within ESCAPE_BURST_FOLLOW_LATERAL_M and inside the
#                ESCAPE_BURST_FOLLOW_ALTITUDE_M band after policy handover,
#                with the deck heading straight out behind the camera. A
#                seed-timed dash (first version, 2026-09-21) fired at 4-7 s,
#                when the privileged PD teacher was still 3.6-4.4 m up on its
#                slow approach descent, and from there a 60-deg camera keeps a
#                deck 2.4 m away comfortably in frame: three flights landed
#                without ever losing the pad, which was the whole point.
#   timed     -- the seed-drawn start below, kept for closed-form tests.
ESCAPE_BURST_TRIGGERS = ("following", "timed")
# The vehicle counts as following when it is this close laterally and this
# low: from 2.5 m the frame reaches only ~4.7 m ahead and nothing behind, so a
# dash out behind the camera leaves it within a second or two of relative
# motion. Below 0.5 m the flare is committed and a loss there is the expected
# end of a landing, not a recovery to demonstrate.
ESCAPE_BURST_FOLLOW_LATERAL_M = 0.9
ESCAPE_BURST_FOLLOW_ALTITUDE_M = (0.5, 2.5)
# Never in the first moments after handover, and always by this long after it
# even if the teacher never closes: a demonstration without the event is the
# walk-only flight the scenario exists to replace.
ESCAPE_BURST_MIN_FOLLOW_S = 2.0
ESCAPE_BURST_FALLBACK_S = 25.0
# ``timed`` only: when the dash begins, drawn per episode from the seed.
ESCAPE_BURST_START_WINDOW_S = (4.0, 7.0)
# Acceleration and deceleration time of the dash. 0.3 -> 1.0 m/s in 1 s is a
# brisk pull-away for a small UGV, not a teleport.
ESCAPE_BURST_RAMP_S = 1.0
# Ground covered at the peak before the deck eases off. A 60-deg camera with
# a 32-deg vertical half-angle sees about 1.9 x altitude ahead, 1 x altitude
# to the side and nothing behind, so against a 0.6 m/s pursuer this is enough
# to leave the frame from every direction below about 1.5 m of altitude.
ESCAPE_BURST_DISTANCE_M = 5.0

# Fastest ground speed any named scenario asks for, before
# ``pad.benchmark_speed_scale``. straight_8mps and the peak of
# linear_acceleration_wave both reach it.
BENCHMARK_PEAK_SPEED_M_S = 8.0
# Cruise speed of ``straight_escape_burst`` before its dash, pre-scale like
# every other scenario literal here. Half of BENCHMARK_PEAK_SPEED_M_S, so the
# dash is a clean doubling (0.50 -> 1.00 m/s on this carrier at the profile's
# 0.125 scale) rather than a change the vehicle could absorb without ever
# losing the deck. It also leaves the cruise phase comfortably inside the
# command envelope, so the vehicle can actually settle into following before
# the event the scenario exists to produce.
STRAIGHT_ESCAPE_CRUISE_SPEED_M_S = 4.0
# Radius the ``circle`` scenario turns at. Held fixed under scaling so a slower
# deck drives the same circle more slowly, rather than shrinking it onto a
# radius smaller than the landing pad itself.
BENCHMARK_CIRCLE_RADIUS_M = 8.0
# Geometry of the closed oval, in metres and held fixed under
# ``benchmark_speed_scale`` for the same reason the circle radius is: scaling
# the deck's speed must slow it down, not shrink the ground it drives on.
#
# 20 m of straight is long enough that a 30 s episode at the profile's 0.50 m/s
# cruise (15 m) is usually straight from end to end, which is the condition the
# escape dash is supposed to interrupt. 6 m of turn radius is four times the
# 1.5 m deck, so the arc is a drive rather than a pirouette, and the whole loop
# fits in 32 m x 12 m -- comfortably inside a 60 m half-arena and the drone's
# own world limit.
BENCHMARK_TRACK_STRAIGHT_M = 20.0
BENCHMARK_TRACK_RADIUS_M = 6.0


def segmented_cruise_speeds(time_axis, scenario: str, *,
                            speed_scale: float = SEGMENTED_CRUISE_SPEED_SCALE,
                            switch_times_s=SEGMENTED_CRUISE_SWITCH_TIMES_S,
                            acceleration_m_s2=SEGMENTED_CRUISE_ACCELERATION_M_S2):
    """Ground speed of one segmented-cruise deck over ``time_axis``.

    The profile is three constant speeds joined by bounded ramps. A step in
    velocity is not something a vehicle does, and the drone would see the
    resulting infinite acceleration as a target that teleports; the ramp is
    the reduced study's ``ugvAccelMax``, scaled with the speeds so a slower
    carrier takes proportionally as long to change speed.
    """
    if scenario not in SEGMENTED_CRUISE_SPEEDS_M_S:
        raise ValueError(f"{scenario!r} is not a segmented-cruise scenario")
    scale = float(speed_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("the segmented-cruise speed scale must be positive")
    times = np.asarray(time_axis, dtype=float).reshape(-1)
    targets = [value * scale for value in SEGMENTED_CRUISE_SPEEDS_M_S[scenario]]
    switches = [float(value) for value in switch_times_s]
    if len(switches) + 1 != len(targets):
        raise ValueError("a three-segment deck needs exactly two switching times")
    rate = abs(float(acceleration_m_s2)) * scale
    if rate <= 0.0:
        raise ValueError("the segment acceleration limit must be positive")

    # Piecewise-constant demand, then a rate limit applied forward in time.
    demand = np.full(times.shape, targets[0], dtype=float)
    for index, boundary in enumerate(switches):
        demand[times >= boundary] = targets[index + 1]
    speeds = np.empty_like(demand)
    speeds[0] = demand[0]
    steps = np.diff(times, prepend=times[0])
    for index in range(1, speeds.size):
        allowed = rate * max(float(steps[index]), 0.0)
        speeds[index] = speeds[index - 1] + float(np.clip(
            demand[index] - speeds[index - 1], -allowed, allowed))
    return speeds


def stadium_track(arc_length, straight_m: float, radius_m: float):
    """Heading and turn rate at ``arc_length`` along a closed oval.

    The loop is a stadium: straight, semicircle, straight back, semicircle.
    Returned as ``(heading_offset, curvature)`` -- what the deck is pointing at
    relative to the loop's own orientation, and ``1/radius`` while it is in a
    turn, zero on the straights. The caller multiplies the curvature by the
    deck's speed to get its yaw rate, which is what makes the turns
    constant-speed circular motion rather than a slowdown.

    Arc length is taken modulo the perimeter, so a deck may be handed any phase
    -- including one carried across an episode boundary -- and stays on the loop.
    """
    straight = max(float(straight_m), 0.0)
    radius = max(float(radius_m), 1e-6)
    turn = math.pi * radius
    perimeter = 2.0 * (straight + turn)
    s = np.asarray(arc_length, dtype=float) % perimeter
    heading = np.zeros_like(s)
    curvature = np.zeros_like(s)
    # 1. first straight, heading 0
    # 2. first half turn, heading sweeping 0 -> pi
    on_first_turn = (s >= straight) & (s < straight + turn)
    heading[on_first_turn] = (s[on_first_turn] - straight) / radius
    curvature[on_first_turn] = 1.0 / radius
    # 3. return straight, heading pi
    on_return = (s >= straight + turn) & (s < 2.0 * straight + turn)
    heading[on_return] = math.pi
    # 4. second half turn, heading sweeping pi -> 2pi
    on_second_turn = s >= 2.0 * straight + turn
    heading[on_second_turn] = math.pi + (
        s[on_second_turn] - 2.0 * straight - turn) / radius
    curvature[on_second_turn] = 1.0 / radius
    return heading, curvature


def escape_burst_due(*, lateral_m: float, altitude_m: float,
                     elapsed_s: float) -> bool:
    """Whether the simulator should start the escape dash now.

    ``elapsed_s`` is simulated time since the policy took the vehicle over.
    True once the vehicle has been following -- close and low -- for at least
    ``ESCAPE_BURST_MIN_FOLLOW_S``, and unconditionally after
    ``ESCAPE_BURST_FALLBACK_S`` so every demonstration carries the event.
    """
    if not all(math.isfinite(float(v)) for v in (lateral_m, altitude_m, elapsed_s)):
        return False
    if float(elapsed_s) >= ESCAPE_BURST_FALLBACK_S:
        return True
    if float(elapsed_s) < ESCAPE_BURST_MIN_FOLLOW_S:
        return False
    low, high = ESCAPE_BURST_FOLLOW_ALTITUDE_M
    return (float(lateral_m) <= ESCAPE_BURST_FOLLOW_LATERAL_M
            and low <= float(altitude_m) <= high)


@dataclass(frozen=True)
class PadMotionConfig:
    mode: str
    carrier: str
    vehicle_model: str
    vehicle_visual_usd: str
    vehicle_dimensions_m: tuple[float, float, float]
    vehicle_visual_origin_from_road_m: tuple[float, float, float]
    vehicle_mass_kg: float
    vehicle_payload_kg: float
    vehicle_max_speed_m_s: float
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
    # Multiplies the closed-form ground speed of the six named benchmark
    # scenarios. They are written at the paper's 4-8 m/s, which a small UGV
    # cannot drive and which would leave the arena inside one episode; this
    # scales them onto the configured carrier without touching their timing.
    benchmark_speed_scale: float
    # The same idea for the three segmented-cruise decks, which are written in
    # the reduced study's 1.0-7.0 m/s rather than against an 8.0 m/s peak, so
    # they need their own factor to land on the carrier's ceiling.
    segmented_cruise_speed_scale: float
    # The lane those decks drive, and how far along it they may get before the
    # next episode turns them round. See the constants above.
    segmented_cruise_heading_rad: float
    segmented_cruise_leg_m: float
    # Radius the ``circle`` scenario turns at. Exposed because scaling the deck
    # down without shrinking this turns the episode into a shallow arc: at
    # scale 0.125 the stock 8 m circle completes under half a lap in 30 s.
    benchmark_circle_radius_m: float
    # --- road mode -------------------------------------------------------
    # The route rectangle is the street the city was built around, so these
    # default to the urban block rather than to numbers of their own: a deck
    # driving a road that is not where the buildings are would give the GNSS
    # model obstructions the camera never sees.
    route_size_m: tuple[float, float]
    route_corner_radius_m: float
    route_waypoints_enu_m: tuple[tuple[float, float, float], ...]
    # A broad campus road can be a closed loop.  In that case the last point
    # joins the first and the UGV keeps driving forward instead of stopping at
    # the seam and reversing around the entire route.
    waypoint_loop: bool
    # Fixed-time acceleration/deceleration at each end of a waypoint shuttle.
    # Tying the ramp to total route length made a campus-scale route need
    # several minutes merely to become visibly mobile.
    waypoint_ramp_s: float
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
    speed_step_range_m_s: tuple[float, float]
    yaw_rate_step_range_rad_s: tuple[float, float]
    motion_update_dt_s: float
    # How the escape-burst scenario starts its dash, see ESCAPE_BURST_TRIGGERS.
    escape_burst_trigger: str = "following"
    # Closed-oval geometry for STRAIGHT_ESCAPE_BURST_TRACK_SCENARIO.
    benchmark_track_straight_m: float = BENCHMARK_TRACK_STRAIGHT_M
    benchmark_track_radius_m: float = BENCHMARK_TRACK_RADIUS_M

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "PadMotionConfig":
        pad = data.get("pad", {}) or {}
        mode = str(pad.get("motion", "static")).lower()
        if mode not in MODES:
            raise ValueError(f"pad.motion must be one of {MODES}, got {mode!r}")
        carrier = str(pad.get("carrier", "lorry")).lower()
        if carrier not in CARRIERS:
            raise ValueError(f"pad.carrier must be one of {CARRIERS}, got {carrier!r}")
        low, high = (float(v) for v in pad.get("speed_range_m_s", (0.0, 0.0)))
        if not 0.0 <= low <= high:
            raise ValueError("pad.speed_range_m_s must be non-negative and ordered")
        vehicle_dimensions = tuple(float(v) for v in pad.get(
            "vehicle_dimensions_m", (0.720, 0.500, 0.345)))
        if len(vehicle_dimensions) != 3 or min(vehicle_dimensions) <= 0.0:
            raise ValueError("pad.vehicle_dimensions_m must be three positive lengths")
        visual_origin = tuple(float(v) for v in pad.get(
            "vehicle_visual_origin_from_road_m", (0.0, 0.0, 0.31)))
        if len(visual_origin) != 3:
            raise ValueError("pad.vehicle_visual_origin_from_road_m must have three components")
        vehicle_model = str(pad.get(
            "vehicle_model", "agilex_ranger_mini_v3" if carrier == "ugv" else "box_lorry"))
        vehicle_mass = float(pad.get("vehicle_mass_kg", 75.0 if carrier == "ugv" else 120.0))
        vehicle_payload = float(pad.get("vehicle_payload_kg", 120.0))
        vehicle_max_speed = float(pad.get("vehicle_max_speed_m_s", 2.0))
        if min(vehicle_mass, vehicle_payload, vehicle_max_speed) <= 0.0:
            raise ValueError("pad vehicle mass, payload and max speed must be positive")
        if carrier == "ugv" and vehicle_model != "agilex_ranger_mini_v3":
            raise ValueError("the supported UGV is 'agilex_ranger_mini_v3'")
        if carrier == "ugv" and high > vehicle_max_speed + 1e-9:
            raise ValueError("pad.speed_range_m_s exceeds the RANGER MINI 3.0 limit")
        benchmark_speed_scale = float(pad.get("benchmark_speed_scale", 1.0))
        if not math.isfinite(benchmark_speed_scale) or benchmark_speed_scale <= 0.0:
            raise ValueError("pad.benchmark_speed_scale must be positive and finite")
        segmented_cruise_speed_scale = float(pad.get(
            "segmented_cruise_speed_scale", SEGMENTED_CRUISE_SPEED_SCALE))
        if (not math.isfinite(segmented_cruise_speed_scale)
                or segmented_cruise_speed_scale <= 0.0):
            raise ValueError(
                "pad.segmented_cruise_speed_scale must be positive and finite")
        segmented_cruise_heading_deg = float(pad.get(
            "segmented_cruise_heading_deg", SEGMENTED_CRUISE_LANE_HEADING_DEG))
        if not math.isfinite(segmented_cruise_heading_deg):
            raise ValueError("pad.segmented_cruise_heading_deg must be finite")
        segmented_cruise_leg_m = float(pad.get(
            "segmented_cruise_leg_m", SEGMENTED_CRUISE_LEG_M))
        if not math.isfinite(segmented_cruise_leg_m) or segmented_cruise_leg_m <= 0.0:
            raise ValueError("pad.segmented_cruise_leg_m must be positive and finite")
        benchmark_track_straight_m = float(pad.get(
            "benchmark_track_straight_m", BENCHMARK_TRACK_STRAIGHT_M))
        benchmark_track_radius_m = float(pad.get(
            "benchmark_track_radius_m", BENCHMARK_TRACK_RADIUS_M))
        if (not math.isfinite(benchmark_track_straight_m)
                or benchmark_track_straight_m < 0.0):
            raise ValueError(
                "pad.benchmark_track_straight_m must be non-negative and finite")
        if (not math.isfinite(benchmark_track_radius_m)
                or benchmark_track_radius_m <= 0.0):
            raise ValueError(
                "pad.benchmark_track_radius_m must be positive and finite")
        # A turn tighter than the deck it carries is a pirouette, not a drive:
        # the instantaneous centre of rotation falls *inside* the deck, so its
        # inner corner travels backwards while its outer one travels forwards.
        # The threshold is therefore the deck's half-diagonal -- the radius at
        # which that centre reaches the deck's furthest corner -- not its full
        # diagonal, which was this check's first form and rejected every
        # shipped lorry profile (6.2 x 2.45 m deck, 6.67 m diagonal) against
        # the 6 m default radius.
        #
        # Read straight from the mapping: ``deck_size`` itself is parsed
        # further down, and this check belongs with the radius it constrains.
        deck_half_diagonal = 0.5 * math.hypot(
            *(float(v) for v in pad.get("deck_size_m", (6.2, 2.45))[:2]))
        if benchmark_track_radius_m <= deck_half_diagonal:
            raise ValueError(
                "pad.benchmark_track_radius_m must exceed the deck's "
                f"half-diagonal ({deck_half_diagonal:.2f} m), or the turn "
                "pivots inside the deck instead of driving around it")
        benchmark_circle_radius_m = float(pad.get(
            "benchmark_circle_radius_m", BENCHMARK_CIRCLE_RADIUS_M))
        if (not math.isfinite(benchmark_circle_radius_m)
                or benchmark_circle_radius_m <= 0.0):
            raise ValueError(
                "pad.benchmark_circle_radius_m must be positive and finite")
        # The named scenarios set their own speed and bypass speed_range_m_s
        # entirely, so the carrier limit has to be enforced against them too --
        # otherwise a 1 m/s rover is silently driven at the scenario's 8 m/s.
        # Only ``random_walk`` reaches that code: under every other mode
        # ``pose`` ignores the scenario, so the campus waypoint profile is not
        # inconsistent for leaving the scale at 1.0.
        if (mode == "random_walk" and carrier == "ugv"
                and BENCHMARK_PEAK_SPEED_M_S * benchmark_speed_scale
                > vehicle_max_speed + 1e-9):
            raise ValueError(
                f"pad.benchmark_speed_scale {benchmark_speed_scale:g} drives the "
                f"named scenarios at "
                f"{BENCHMARK_PEAK_SPEED_M_S * benchmark_speed_scale:.2f} m/s, "
                f"past this carrier's {vehicle_max_speed:g} m/s limit")
        start = tuple(float(v) for v in pad.get("start_position_enu_m", (0.0, 0.0, 0.0)))
        if len(start) != 3:
            raise ValueError("pad.start_position_enu_m must have three components")
        urban = (data.get("urban") or {}) if isinstance(data, dict) else {}
        lane_width = float(urban.get("lane_width_m", 3.3))
        route = tuple(float(v) for v in pad.get(
            "route_size_m", urban.get("block_size_m", (90.0, 60.0))))
        if len(route) != 2 or min(route) <= 0.0:
            raise ValueError("pad.route_size_m must be two positive lengths")
        # ``null`` is how an overlay profile clears an inherited key here (see
        # the vision block of config/shin2026-system.yaml), so a profile that
        # drops the campus route must not fall over on None.
        waypoints = tuple(
            tuple(float(component) for component in point)
            for point in (pad.get("route_waypoints_enu_m") or ())
        )
        if mode == "waypoints":
            if len(waypoints) < 2 or any(len(point) != 3 for point in waypoints):
                raise ValueError(
                    "pad.route_waypoints_enu_m must contain at least two XYZ points"
                )
            if any(np.linalg.norm(np.subtract(a, b)) < 1e-6
                   for a, b in zip(waypoints[:-1], waypoints[1:])):
                raise ValueError("consecutive route waypoints must be distinct")
        waypoint_loop = bool(pad.get("waypoint_loop", False))
        if (mode == "waypoints" and waypoint_loop
                and np.linalg.norm(np.subtract(waypoints[0], waypoints[-1])) > 1e-6):
            raise ValueError(
                "a waypoint loop must repeat its first point as its last point")
        waypoint_ramp_s = float(pad.get("waypoint_ramp_s", 4.0))
        if not math.isfinite(waypoint_ramp_s) or waypoint_ramp_s <= 0.0:
            raise ValueError("pad.waypoint_ramp_s must be positive and finite")
        stops = tuple(float(v) for v in pad.get("stop_depth_range", (0.35, 1.0)))
        if len(stops) != 2 or not 0.0 <= stops[0] <= stops[1] <= 1.0:
            raise ValueError("pad.stop_depth_range must be ordered inside [0,1]")
        # One lane centre per lane that fits on this side of the carriageway,
        # counted out from the centreline. An explicit pad.lane_offset_m pins
        # the lorry to one lane instead.
        half_road = float(urban.get("road_half_width_m", 7.0))
        deck_size = tuple(float(v) for v in pad.get("deck_size_m", (6.2, 2.45)))
        deck_width = float(deck_size[1])
        if len(deck_size) != 2 or min(deck_size) <= 0.0:
            raise ValueError("pad.deck_size_m must be two positive lengths")
        if carrier == "ugv" and (deck_size[0] + 1e-9 < vehicle_dimensions[0]
                                 or deck_size[1] + 1e-9 < vehicle_dimensions[1]):
            raise ValueError("landing deck must cover the RANGER MINI 3.0 footprint")
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
        speed_steps = tuple(float(v) for v in pad.get(
            "speed_step_perturbation_m_s", (-0.5, 0.5)))
        yaw_steps = tuple(math.radians(float(v)) for v in pad.get(
            "yaw_rate_step_perturbation_deg_s", (-3.0, 3.0)))
        motion_dt = float(pad.get("motion_update_dt_s", 0.1))
        if (len(speed_steps) != 2 or speed_steps[0] > speed_steps[1]
                or len(yaw_steps) != 2 or yaw_steps[0] > yaw_steps[1]):
            raise ValueError("pad random-walk perturbation ranges must be ordered pairs")
        if not math.isfinite(motion_dt) or motion_dt <= 0.0:
            raise ValueError("pad.motion_update_dt_s must be positive and finite")
        escape_burst_trigger = str(pad.get("escape_burst_trigger", "following")).lower()
        if escape_burst_trigger not in ESCAPE_BURST_TRIGGERS:
            raise ValueError(
                f"pad.escape_burst_trigger must be one of {ESCAPE_BURST_TRIGGERS}, "
                f"got {escape_burst_trigger!r}")
        return cls(
            mode=mode,
            carrier=carrier,
            vehicle_model=vehicle_model,
            vehicle_visual_usd=str(pad.get("vehicle_visual_usd", "")),
            vehicle_dimensions_m=vehicle_dimensions,
            vehicle_visual_origin_from_road_m=visual_origin,
            vehicle_mass_kg=vehicle_mass,
            vehicle_payload_kg=vehicle_payload,
            vehicle_max_speed_m_s=vehicle_max_speed,
            start_position_enu_m=start,
            deck_height_m=float(pad.get("deck_height_m", 0.0)),
            deck_size_m=deck_size,
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
            benchmark_speed_scale=benchmark_speed_scale,
            segmented_cruise_speed_scale=segmented_cruise_speed_scale,
            segmented_cruise_heading_rad=math.radians(segmented_cruise_heading_deg),
            segmented_cruise_leg_m=segmented_cruise_leg_m,
            benchmark_circle_radius_m=benchmark_circle_radius_m,
            benchmark_track_straight_m=benchmark_track_straight_m,
            benchmark_track_radius_m=benchmark_track_radius_m,
            route_size_m=route,
            route_start=route_start,
            route_corner_radius_m=float(pad.get("route_corner_radius_m",
                                                urban.get("corner_radius_m", 12.0))),
            route_waypoints_enu_m=waypoints,
            waypoint_loop=waypoint_loop,
            waypoint_ramp_s=waypoint_ramp_s,
            lane_centres_m=lanes,
            lane_wander_m=float(pad.get("lane_wander_m", 0.18)),
            lane_wander_hz=float(pad.get("lane_wander_hz", 0.09)),
            lane_change_probability=float(pad.get("lane_change_probability", 0.5)),
            stop_interval_s=float(pad.get("stop_interval_s", 11.0)),
            stop_sigma_s=float(pad.get("stop_sigma_s", 1.7)),
            stop_depth_range=stops,
            speed_step_range_m_s=speed_steps,
            yaw_rate_step_range_rad_s=yaw_steps,
            motion_update_dt_s=motion_dt,
            escape_burst_trigger=escape_burst_trigger,
        )

    @property
    def is_static(self) -> bool:
        return self.mode == "static" or self.speed_max_m_s <= 0.0


@dataclass(frozen=True)
class LorryPart:
    """One piece of the vehicle the pad is painted on, in the pad frame.

    Pad frame: origin at the centre of the marker plane, +X the way the lorry
    points, +Z up, so the road is at ``-deck_height_m`` and every part hangs
    between the two. ``kind`` is ``'box'`` or ``'wheel'``; a wheel's ``size``
    is ``(diameter, width, diameter)`` so one number pair describes both.
    """
    name: str
    kind: str
    size: tuple[float, float, float]
    centre: tuple[float, float, float]
    colour: tuple[float, float, float]
    collider: bool = False


def lorry_parts(deck_size_m, deck_height_m: float) -> tuple[LorryPart, ...]:
    """The box lorry under the landing deck.

    Kept here, next to the motion that drives it, rather than inline in the
    USD: the shape has to agree with ``deck_size_m`` and ``deck_height_m`` --
    the roof is the deck, the wheels have to reach the road -- and geometry
    that only exists inside a draw call cannot be checked without a simulator.

    Wheel radius is fixed at 0.5 m and the chassis at 1.05 m, which is an
    ordinary 7.5 t box lorry; the deck height and footprint come from config
    and everything else is placed off them.
    """
    length, width = (float(v) for v in deck_size_m)
    road_z = -float(deck_height_m)
    wheel_r, chassis_top = 0.50, road_z + 1.05
    box_h = -chassis_top                       # rails up to the roof at z = 0
    # Derived from the box, not fixed: at a hard 2.45 m the cab stood proud of
    # the landing deck on the configured 3.20 m body, which would put a
    # windscreen in the way of an approach that clipped the front of the roof.
    cab_len, cab_h = 1.90, max(box_h - 0.35, 0.9 * box_h)
    parts = [
        # The cargo box: the landing deck is its lid, so it is exactly the
        # deck footprint and a policy cannot fly through the lorry side-on.
        LorryPart("cargo_box", "box", (length, width, box_h),
                  (0.0, 0.0, chassis_top + 0.5 * box_h), (0.86, 0.87, 0.89), True),
        # Lower than the box and ahead of it, so from the air the silhouette
        # reads as a box lorry rather than as one long container.
        LorryPart("cab", "box", (cab_len, width * 0.98, cab_h),
                  (0.5 * (length + cab_len), 0.0, chassis_top + 0.5 * cab_h),
                  (0.20, 0.34, 0.56)),
        LorryPart("chassis", "box", (length + cab_len, width * 0.80, 0.22),
                  (0.5 * cab_len, 0.0, chassis_top - 0.11), (0.12, 0.12, 0.14)),
    ]
    # Steering pair under the cab, two drive pairs under the box.
    axles = (0.5 * length + 0.6, -0.10 * length, -0.34 * length)
    for index, x in enumerate(axles):
        for side, tag in ((-1.0, "l"), (1.0, "r")):
            parts.append(LorryPart(
                f"wheel_{index}_{tag}", "wheel",
                (2.0 * wheel_r, 0.30, 2.0 * wheel_r),
                (float(x), float(side * 0.5 * width), road_z + wheel_r),
                (0.06, 0.06, 0.07)))
    return tuple(parts)


def ugv_parts(deck_size_m, deck_height_m: float,
              vehicle_dimensions_m=(0.720, 0.500, 0.345)) -> tuple[LorryPart, ...]:
    """RANGER MINI 3.0 fallback geometry in the landing-deck frame.

    The official mesh is preferred at runtime. These dimensions keep collision
    and a no-asset fallback faithful to the 720 x 500 x 345 mm chassis and to
    the wheel positions/collision sizes in AgileX's public V3 URDF.
    """
    deck_length, deck_width = (float(v) for v in deck_size_m)
    length, width, overall_height = (float(v) for v in vehicle_dimensions_m)
    if length > deck_length + 1e-9 or width > deck_width + 1e-9:
        raise ValueError("RANGER MINI 3.0 does not fit under the configured deck")
    road_z = -float(deck_height_m)
    wheel_r, wheel_width = 0.09, 0.08
    parts = [
        LorryPart(
            "ranger_collision", "box", (0.50, 0.35, 0.20),
            (0.0, 0.0, road_z + 0.22), (0.12, 0.16, 0.19), True,
        ),
        LorryPart(
            "ranger_body", "box", (length, 0.42, 0.15),
            (0.0, 0.0, road_z + 0.255), (0.16, 0.23, 0.30), False,
        ),
        LorryPart(
            "ranger_top", "box", (0.48, 0.32, 0.075),
            (-0.025, 0.0, road_z + overall_height - 0.0375),
            (0.30, 0.38, 0.44), False,
        ),
    ]
    # AgileX URDF steering joint origins: x=+/-0.25, y=+/-0.19 m;
    # its wheel collision is a radius-0.09, width-0.08 m cylinder.
    for index, x in enumerate((-0.25, 0.25)):
        for side, tag in ((-1.0, "l"), (1.0, "r")):
            parts.append(LorryPart(
                f"ranger_wheel_{index}_{tag}", "wheel",
                (2.0 * wheel_r, wheel_width, 2.0 * wheel_r),
                (x, side * 0.19, road_z + wheel_r),
                (0.04, 0.04, 0.05),
            ))
    return tuple(parts)


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


class WaypointRoute:
    """Arc-length parameterisation of an absolute 3-D waypoint polyline."""

    def __init__(self, points):
        self.points = np.asarray(points, dtype=float)
        if (self.points.ndim != 2 or self.points.shape[0] < 2
                or self.points.shape[1] != 3):
            raise ValueError("waypoint route requires at least two XYZ points")
        delta = np.diff(self.points, axis=0)
        self.segment_lengths = np.linalg.norm(delta, axis=1)
        if np.any(self.segment_lengths < 1e-6):
            raise ValueError("waypoint route contains a zero-length segment")
        self.tangents = delta / self.segment_lengths[:, None]
        self.edges = np.cumsum(self.segment_lengths)
        self.length = float(self.edges[-1])

    def at(self, distance: float) -> tuple[np.ndarray, np.ndarray]:
        s = float(np.clip(distance, 0.0, self.length))
        index = min(int(np.searchsorted(self.edges, s, side="right")),
                    len(self.segment_lengths) - 1)
        start = 0.0 if index == 0 else float(self.edges[index - 1])
        fraction = (s - start) / float(self.segment_lengths[index])
        point = self.points[index] + fraction * (
            self.points[index + 1] - self.points[index])
        return point, self.tangents[index]


class PadTrajectory:
    """Where the deck is, how fast, and which way it is pointing."""

    def __init__(self, cfg: PadMotionConfig, *, initial_route_fraction: float = 0.0):
        self.cfg = cfg
        self.initial_route_fraction = float(initial_route_fraction)
        if (not math.isfinite(self.initial_route_fraction)
                or not 0.0 <= self.initial_route_fraction < 1.0):
            raise ValueError("initial_route_fraction must be finite in [0,1)")
        self.start = np.asarray(cfg.start_position_enu_m, dtype=float)
        self.speed = 0.0
        self.heading0 = 0.0
        self.phase = np.zeros(2)
        self.t0 = 0.0
        self.yaw = 0.0
        self._yaw_initialised = False
        self.route = RoadRoute(cfg.route_size_m[0], cfg.route_size_m[1],
                               cfg.route_corner_radius_m)
        self.waypoint_route = (
            WaypointRoute(cfg.route_waypoints_enu_m)
            if cfg.route_waypoints_enu_m else None
        )
        self.initial_waypoint_distance = (
            self.initial_route_fraction * self.waypoint_route.length
            if self.waypoint_route is not None else 0.0)
        # Seconds into the forward-and-reverse shuttle cycle.
        self.waypoint_phase = 0.0
        self._waypoint_tangent = np.array([1.0, 0.0, 0.0])
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
        self.random_walk_position = np.zeros((1, 3))
        # Where the analytic track is anchored. ``_prepare_benchmark_motion``
        # always rebuilds the track from zero, so without this the deck would
        # snap back to ``start_position_enu_m`` at every reset while the drone,
        # which is deliberately left flying between episodes, stayed where the
        # last episode ended. That gap is the whole distance the deck covered:
        # 30 m for straight_8mps, which no entry budget can close against a
        # deck that then drives away again.
        self.random_walk_origin = np.zeros(3)
        self.random_walk_velocity = np.zeros((1, 3))
        self.benchmark_scenario = "training_random_walk"
        # Timing of the escape dash of the current episode, or None until it
        # has happened; reported so a flight trace can be read against it.
        self.escape_burst: dict[str, float] | None = None
        # True while the escape-burst scenario is waiting for Isaac to fire the
        # dash (``following`` trigger); the drawn track is kept so the dash can
        # be laid over it at whatever instant that turns out to be.
        self.escape_burst_armed = False
        # Which way along the fixed lane a segmented-cruise deck is currently
        # driving. Flipped between episodes by ``_segmented_cruise_heading``;
        # never inside one.
        self._segmented_cruise_reversed = False
        self._track_speeds = np.zeros(0)
        # How far along a closed track the deck has driven. Zero for every
        # other scenario, and the thing ``route_start: continue`` carries for
        # this one instead of a position offset and a fresh heading.
        self.track_phase_m = 0.0
        self._track_yaw_rates = np.zeros(0)
        self._track_headings = np.zeros(0)

    def initial_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the parked pose before the first seeded reset.

        Parallel campus vehicles are staged at different arc-lengths of the
        *same* surveyed route.  This keeps every UGV on asphalt and gives the
        three UAVs independent airspace, unlike translating the entire route
        sideways.  The first reset below starts from exactly this point, so the
        deck never teleports out from under its airborne vehicle during boot.
        """
        if self.cfg.mode != "waypoints" or self.waypoint_route is None:
            return self.pose(0.0)
        point, tangent = self.waypoint_route.at(self.initial_waypoint_distance)
        position = np.asarray(point, dtype=float).copy()
        position[2] += self.cfg.deck_height_m
        return position, np.zeros(3, dtype=float)

    def reset(self, seed: int, sim_time: float, speed_scale: float = 1.0,
              scenario: str = "training_random_walk") -> dict[str, Any]:
        """Draw this episode's deck motion from the episode seed.

        SPEED_SCALE multiplies the drawn speed, which is what an evaluation
        sweep varies to ask how fast a deck a policy can still land on. It is
        applied after the draw so the seed still picks the same point in the
        distribution at every scale, which is what makes the sweep paired.
        """
        rng = np.random.default_rng(int(seed) + 977)
        cfg = self.cfg
        if scenario not in BENCHMARK_SCENARIOS:
            raise ValueError(f"unknown benchmark platform scenario {scenario!r}")
        self.benchmark_scenario = scenario
        carried_waypoint = self._waypoint_progress(sim_time)
        # Where the lorry has got to so far -- along the road and across it --
        # read before the clock and the lane draw are overwritten.
        carried = self.arc_length(sim_time)
        carried_lane = self.lane_now(sim_time)
        self.speed = float(rng.uniform(cfg.speed_min_m_s, cfg.speed_max_m_s))
        self.speed *= max(float(speed_scale), 0.0)
        drawn_heading = float(rng.uniform(0.0, 2.0 * math.pi))
        # A closed track is one fixed path, so it carries its PHASE across a
        # reset where an open scenario carries its position and redraws its
        # heading. Redrawing the heading here would rotate the whole loop under
        # the vehicle parked on it; carrying the phase is what makes the next
        # episode continue the same lap. The draw still happens either way, so
        # the seed consumes identically and a paired sweep stays paired.
        on_closed_track = scenario in CLOSED_TRACK_SCENARIOS
        carried_track = cfg.route_start == "continue" and self._driven
        if on_closed_track and carried_track:
            self.track_phase_m = self.track_phase_at(sim_time)
        else:
            self.heading0 = drawn_heading
            self.track_phase_m = 0.0
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
        if cfg.mode == "waypoints":
            self.waypoint_phase = (
                self._waypoint_phase_for(*carried_waypoint)
                if cfg.route_start == "continue" and self._driven
                else self._waypoint_phase_for(
                    self.initial_waypoint_distance, 1.0)
            )
        if cfg.route_start == "continue" and self._driven:
            # Advance the phase by the time that has passed, so the wander picks
            # up exactly where it was rather than snapping back to its own t=0.
            self.wander_phase = float(
                (self.wander_phase
                 + 2.0 * math.pi * cfg.lane_wander_hz * (float(sim_time) - self.t0))
                % (2.0 * math.pi))
        else:
            self.wander_phase = seeded_phase
        # Read before t0 moves and before the track is rebuilt: this is where
        # the deck actually stands right now.
        carried_offset = (np.asarray(self.pose(sim_time)[0], dtype=float)
                          - self.start if self._driven else np.zeros(3))
        self.t0 = float(sim_time)
        self._driven = True
        self._yaw_initialised = False
        if cfg.mode == "random_walk":
            if scenario in SEGMENTED_CRUISE_SCENARIOS:
                # Replaces the drawn heading entirely: every pair drives the
                # same lane, so the row spawned by
                # ``parallel.pair_offsets_enu_m`` sets off together.
                self.heading0 = self._segmented_cruise_heading(carried_offset)
            self._prepare_benchmark_motion(rng, max(float(speed_scale), 0.0))
            # ``continue`` means the same thing it means for the surveyed
            # route: leave the deck where it stands rather than teleporting it
            # under whatever is parked on its roof. Only the horizontal anchor
            # carries; height is re-derived from the start and the deck height.
            if cfg.route_start == "continue":
                self.random_walk_origin = np.array(
                    [carried_offset[0], carried_offset[1], 0.0])
                # Leaving the deck where it stands turns the episode sequence
                # into a 2-D random walk, because each episode draws its own
                # heading. Expected distance from the origin grows without
                # bound, and the arena clamp does not push back -- it pins the
                # deck on the boundary with zero velocity, which would quietly
                # turn the benchmark into a stationary target.
                #
                # So steer rather than teleport: past half the arena the drawn
                # heading is rotated toward the origin, fully so at the edge.
                # The draw still comes from the episode seed, no position ever
                # jumps, and inside the half-radius nothing is changed at all.
                radius = float(np.linalg.norm(self.random_walk_origin[:2]))
                limit = float(cfg.arena_radius_m)
                # A closed track keeps this too, and it costs it nothing: the
                # pull rotates ``heading0``, which for a closed path turns the
                # whole loop rather than bending any of its straights. The loop
                # alone never reaches the half-arena -- it is 32 m by 12 m --
                # but the escape dash leaves it by ~5 m each time it fires, and
                # those excursions accumulate as a random walk of the loop's
                # position. Measured here with an adversarial dash that always
                # points the same way, 30 episodes reached the arena edge. The
                # rotation is what turns that walk back.
                # A segmented-cruise deck has no heading left to rotate --
                # its lane is fixed so the pairs stay in a row pointing the
                # same way -- so it is bounded by the shuttle above instead.
                # Rotating it here would bend the straight the envelope needs.
                if scenario in SEGMENTED_CRUISE_SCENARIOS:
                    pass
                elif limit > 0.0 and radius > 0.5 * limit:
                    inward = math.atan2(-self.random_walk_origin[1],
                                        -self.random_walk_origin[0])
                    pull = min(1.0, (radius - 0.5 * limit) / (0.5 * limit))
                    delta = (inward - self.heading0 + math.pi) % (2.0 * math.pi) - math.pi
                    self.heading0 = self.heading0 + pull * delta
                    self._prepare_benchmark_motion(
                        rng, max(float(speed_scale), 0.0))
            else:
                self.random_walk_origin = np.zeros(3)
        position, velocity = self.pose(sim_time)
        fallback = self.heading0
        if cfg.mode == "waypoints":
            fallback = math.atan2(self._waypoint_tangent[1], self._waypoint_tangent[0])
        self.yaw = self._velocity_heading(velocity, fallback=fallback)
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
            "benchmark_scenario": self.benchmark_scenario,
            "escape_burst": self.escape_burst,
        }

    def _segmented_cruise_heading(self, carried_offset) -> float:
        """Lane heading for the episode about to start.

        One fixed lane for every pair, so the row of pairs sets off in the same
        direction; reversed once this pair's deck has run its leg out, so the
        sequence is bounded without the inward steering a drawn heading gets.

        The decision is taken HERE, at the reset, from where the deck ended the
        previous episode. Inside an episode the heading is a constant, which is
        what lets the reduced envelope hold zero lateral velocity and one yaw
        and still stay on the deck's track (docs/PLANAR_ENVELOPE.md).
        """
        cfg = self.cfg
        lane = float(cfg.segmented_cruise_heading_rad)
        if cfg.route_start != "continue":
            # A reseeded deck starts at its own lane origin every episode, so
            # there is nothing to shuttle.
            self._segmented_cruise_reversed = False
            return lane
        offset = np.asarray(carried_offset, dtype=float).reshape(-1)
        along = (math.cos(lane) * float(offset[0])
                 + math.sin(lane) * float(offset[1]))
        leg = float(cfg.segmented_cruise_leg_m)
        if along >= leg:
            self._segmented_cruise_reversed = True
        elif along <= -leg:
            self._segmented_cruise_reversed = False
        return lane + math.pi if self._segmented_cruise_reversed else lane

    def _prepare_benchmark_motion(self, rng, scale: float,
                                  samples: int = 6001) -> None:
        """Precompute paper-defined random walk or documented test approximation."""
        cfg = self.cfg
        dt = cfg.motion_update_dt_s
        if dt <= 0.0:
            raise ValueError("pad.motion_update_dt_s must be positive")
        speeds = np.empty(samples)
        yaw_rates = np.empty(samples)
        headings = np.empty(samples)
        scenario = self.benchmark_scenario
        time_axis = np.arange(samples) * dt
        headings[:] = self.heading0
        yaw_rates[:] = 0.0
        self.escape_burst = None
        self.escape_burst_armed = False
        if scenario in ("training_random_walk", ESCAPE_BURST_SCENARIO):
            dv = scale * rng.uniform(*cfg.speed_step_range_m_s, size=samples - 1)
            dw = scale * rng.uniform(*cfg.yaw_rate_step_range_rad_s, size=samples - 1)
            speeds[0], yaw_rates[0] = self.speed, 0.0
            for index in range(1, samples):
                speeds[index] = np.clip(speeds[index - 1] + dv[index - 1],
                                        cfg.speed_min_m_s, scale * cfg.speed_max_m_s)
                yaw_rates[index] = np.clip(yaw_rates[index - 1] + dw[index - 1],
                                           -cfg.yaw_rate_limit_rad_s,
                                           cfg.yaw_rate_limit_rad_s)
                headings[index] = headings[index - 1] + yaw_rates[index] * dt
            if scenario == ESCAPE_BURST_SCENARIO:
                if cfg.escape_burst_trigger == "timed":
                    # Drawn after the walk so the walk itself is the one the
                    # same seed gives under training_random_walk.
                    begin = min(int(round(float(rng.uniform(
                        *ESCAPE_BURST_START_WINDOW_S)) / dt)), samples - 1)
                    self._overlay_escape_burst(speeds, yaw_rates, headings, dt,
                                               begin=begin)
                else:
                    self.escape_burst_armed = True
        # ``benchmark_speed_scale`` slows the deck onto the configured carrier
        # without redefining any scenario: every switching time below is left
        # alone, so a scaled zigzag still reverses every 3 s and a scaled u-turn
        # still turns through pi between t=5 s and t=10 s. Only the distance
        # covered in that time shrinks.
        speed_scale = cfg.benchmark_speed_scale
        if scenario == "straight_8mps":
            speeds[:] = 8.0 * speed_scale
        elif scenario == "linear_acceleration_wave":
            speeds[:] = 4.0 * speed_scale * (1.0 + np.sin(0.5 * time_axis))
        elif scenario == "circle":
            speeds[:] = 6.0 * speed_scale
            # Derived from the radius rather than scaled with the speed: at
            # scale 0.125 a scaled yaw rate would turn a 1 m circle, which is
            # smaller than the 1.5 m deck driving it.
            yaw_rates[:] = speeds / cfg.benchmark_circle_radius_m
            headings[:] = self.heading0 + yaw_rates * time_axis
        elif scenario == "zigzag":
            speeds[:] = 6.0 * speed_scale
            headings[:] = self.heading0 + np.where(
                (np.floor(time_axis / 3.0).astype(int) % 2) == 0,
                math.radians(45.0), -math.radians(45.0))
        elif scenario == "u_turn":
            speeds[:] = 6.0 * speed_scale
            turn_time = np.clip(time_axis - 5.0, 0.0, 5.0)
            headings[:] = self.heading0 + math.pi * turn_time / 5.0
            yaw_rates[(time_axis >= 5.0) & (time_axis <= 10.0)] = math.pi / 5.0
        elif scenario == STRAIGHT_ESCAPE_BURST_TRACK_SCENARIO:
            # The straight run, closed into an oval. Speed is constant for the
            # whole loop -- the turns are constant-speed circular motion, so
            # the deck never slows to corner -- and the heading comes from the
            # arc length the deck has driven, continued across episodes.
            speeds[:] = STRAIGHT_ESCAPE_CRUISE_SPEED_M_S * speed_scale
            if cfg.escape_burst_trigger == "timed":
                begin = min(int(round(float(rng.uniform(
                    *ESCAPE_BURST_START_WINDOW_S)) / dt)), samples - 1)
                peak, begin, hold_end, end, distance = self._escape_burst_speeds(
                    speeds, dt, begin=begin)
                track_burst = {
                    "start_s": begin * dt, "peak_m_s": peak,
                    "hold_end_s": hold_end * dt, "end_s": end * dt,
                    "distance_m": float(distance)}
            else:
                track_burst = None
                self.escape_burst_armed = True
            headings[:], yaw_rates[:] = self._track_schedule(speeds, dt)
            if track_burst is not None:
                track_burst["heading_rad"] = float(headings[begin])
                self.escape_burst = track_burst
        elif scenario == STRAIGHT_ESCAPE_BURST_SCENARIO:
            # Constant-velocity straight run; the dash is laid over it either
            # here (``timed``) or by the simulator once the vehicle is actually
            # following (``following``), exactly as for the walk-based variant.
            speeds[:] = STRAIGHT_ESCAPE_CRUISE_SPEED_M_S * speed_scale
            yaw_rates[:] = 0.0
            headings[:] = self.heading0
            if cfg.escape_burst_trigger == "timed":
                begin = min(int(round(float(rng.uniform(
                    *ESCAPE_BURST_START_WINDOW_S)) / dt)), samples - 1)
                self._overlay_escape_burst(speeds, yaw_rates, headings, dt,
                                           begin=begin)
            else:
                self.escape_burst_armed = True
        elif scenario == "vertical_heave_boat":
            speeds[:] = 4.0 * speed_scale
        elif scenario in SEGMENTED_CRUISE_SCENARIOS:
            # Straight line, constant heading, piecewise-constant speed. The
            # heading is whatever the episode drew and then never changes, so
            # the deck's track is a fixed line and the reduced envelope's
            # lateral and yaw constraints stay correct for the whole episode.
            speeds[:] = segmented_cruise_speeds(
                time_axis, scenario,
                speed_scale=cfg.segmented_cruise_speed_scale)
            yaw_rates[:] = 0.0
            headings[:] = self.heading0
        velocity = np.c_[speeds * np.cos(headings), speeds * np.sin(headings),
                         np.zeros(samples)]
        if scenario == "vertical_heave_boat":
            velocity[:, 2] = 0.4 * np.cos(0.8 * time_axis)
        self.speed = float(speeds[0])
        position = np.zeros((samples, 3))
        position[1:] = np.cumsum(velocity[:-1] * dt, axis=0)
        self.random_walk_position = position
        self.random_walk_velocity = velocity
        self._track_speeds = speeds
        self._track_yaw_rates = yaw_rates
        self._track_headings = headings

    def track_phase_at(self, sim_time: float) -> float:
        """Arc length driven along a closed track, for carrying across a reset.

        Integrates the speeds the track actually ran at, so the escape dash
        counts for the ground it covered rather than for the cruise it
        replaced. Zero for every scenario that is not a closed track.
        """
        if self.benchmark_scenario not in CLOSED_TRACK_SCENARIOS:
            return 0.0
        speeds = np.asarray(self._track_speeds, dtype=float)
        if speeds.size == 0:
            return float(self.track_phase_m)
        dt = self.cfg.motion_update_dt_s
        u = max(0.0, float(sim_time) - self.t0) / dt
        index = min(int(math.floor(u)), speeds.size - 1)
        fraction = min(max(u - index, 0.0), 1.0)
        driven = float(speeds[:index].sum() + fraction * speeds[index]) * dt
        return float(self.track_phase_m + driven)

    def trigger_escape_burst(self, sim_time: float,
                             heading: float | None = None) -> dict[str, float] | None:
        """Start the escape dash now, from wherever the deck is on its walk.

        Called by the simulator once the vehicle is following the deck (see
        ``ESCAPE_BURST_FOLLOW_*``), with ``heading`` the direction to dash in
        -- straight out behind the camera, so the pad leaves the frame within
        a second or two rather than after metres of forward footprint. The
        track is rewritten from the next sample onward, so the deck's position
        and the interpolation of the sample it is currently between are left
        untouched: nothing jumps. The deck's yaw still turns toward the new
        velocity under its rate limit, as it does for every other manoeuvre.

        Returns the dash's timing, or None when this episode's scenario has no
        dash waiting (any other scenario, ``timed`` trigger, or already fired).
        """
        if not self.escape_burst_armed or self.cfg.mode != "random_walk":
            return None
        dt = self.cfg.motion_update_dt_s
        samples = len(self._track_speeds)
        if samples < 3:
            return None
        u = max(0.0, float(sim_time) - self.t0) / dt
        begin = min(int(math.floor(u)) + 1, samples - 1)
        if begin >= samples - 2:
            return None
        if self.benchmark_scenario in CLOSED_TRACK_SCENARIOS:
            # A closed track keeps its path: the dash is the deck covering the
            # same oval at twice the speed. ``heading`` -- "straight out behind
            # the camera" -- is what an open scenario uses to guarantee the pad
            # leaves the frame, and honouring it here would take the deck off
            # its loop by the dash distance every time, which accumulates into
            # exactly the unbounded walk the closed track exists to prevent
            # (measured: 30 dashes in one direction reached the arena edge).
            # The frame-exit event survives, because it comes from the speed
            # doubling on a straight, which is what the open scenario is.
            self._overlay_track_escape_burst(dt, begin=begin)
        else:
            self._overlay_escape_burst(self._track_speeds, self._track_yaw_rates,
                                       self._track_headings, dt, begin=begin,
                                       heading=heading)
        speeds, headings = self._track_speeds, self._track_headings
        velocity = self.random_walk_velocity
        velocity[begin:, 0] = speeds[begin:] * np.cos(headings[begin:])
        velocity[begin:, 1] = speeds[begin:] * np.sin(headings[begin:])
        position = self.random_walk_position
        position[begin + 1:] = position[begin] + np.cumsum(
            velocity[begin:-1] * dt, axis=0)
        self.escape_burst_armed = False
        return dict(self.escape_burst)

    def _overlay_escape_burst(self, speeds, yaw_rates, headings, dt: float, *,
                              begin: int, heading: float | None = None) -> None:
        """Lay one straight dash over the drawn random walk, in place.

        From sample ``begin`` the deck accelerates over ``ESCAPE_BURST_RAMP_S``
        from whatever the walk was doing to the carrier's scenario peak (the
        speed ``straight_8mps`` drives at on this carrier), holds it until it
        has covered ``ESCAPE_BURST_DISTANCE_M`` since the dash began, then
        eases back onto the walk's own speed over another ramp. The heading is
        held for the dash -- the walk's own heading at ``begin``, or
        ``heading`` when the caller chooses the direction -- and the walk's
        yaw increments resume from it afterwards, so the track has no jump.

        The dash deliberately ignores the curriculum ``scale``: it exists so
        the demonstration shows the pad leaving the frame, and a dash that
        shrank with the entry curriculum would not leave it at all.
        """
        peak, begin, hold_end, end, distance = self._escape_burst_speeds(
            speeds, dt, begin=begin)
        samples = len(speeds)
        yaw_rates[begin:end + 1] = 0.0
        if heading is not None and begin >= 1:
            # The caller's direction: written as the heading at ``begin`` so
            # the recomputation below carries it through the dash and the
            # walk's own yaw increments continue from it afterwards.
            headings[begin - 1] = float(heading)
        for index in range(max(begin, 1), samples):
            headings[index] = headings[index - 1] + yaw_rates[index] * dt
        self.escape_burst = {
            "start_s": begin * dt, "peak_m_s": peak,
            "hold_end_s": hold_end * dt, "end_s": end * dt,
            "distance_m": float(distance),
            "heading_rad": float(headings[begin])}

    def _escape_burst_speeds(self, speeds, dt: float, *, begin: int):
        """Write the dash's speed ramp into ``speeds``; return its timing.

        Speed only. What the deck does with that speed -- hold a straight
        heading, or stay on a closed track and simply cover it faster -- is the
        caller's, because the two scenarios want different things from the same
        acceleration.
        """
        cfg = self.cfg
        samples = len(speeds)
        peak = float(BENCHMARK_PEAK_SPEED_M_S * cfg.benchmark_speed_scale)
        ramp = max(1, int(round(ESCAPE_BURST_RAMP_S / dt)))
        begin = int(min(max(begin, 0), samples - 1))
        base = float(speeds[begin])
        top = min(begin + ramp, samples - 1)
        hold_end = top
        distance = 0.0
        for index in range(begin, samples):
            speed = (peak if index >= top
                     else base + (peak - base) * (index - begin) / ramp)
            distance += speed * dt
            hold_end = index
            if index >= top and distance >= ESCAPE_BURST_DISTANCE_M:
                break
        end = min(hold_end + ramp, samples - 1)
        resume = float(speeds[end])
        for index in range(begin, end + 1):
            if index < top:
                speeds[index] = base + (peak - base) * (index - begin) / ramp
            elif index <= hold_end:
                speeds[index] = peak
            else:
                speeds[index] = peak + (resume - peak) * (
                    (index - hold_end) / max(1, end - hold_end))
        return peak, begin, hold_end, end, distance

    def _track_schedule(self, speeds, dt: float):
        """Heading and yaw rate for a closed track driven at ``speeds``.

        The path is fixed; the speed decides how far along it the deck has got.
        Integrating the speed into an arc length and reading the loop at that
        arc length is what lets the escape dash be a pure *speed* event: the
        deck covers the oval faster without leaving it, so the pull-away is
        still a straight line whenever it fires on a straight, and the deck's
        position stays bounded by the loop for as many episodes as it drives.
        """
        cfg = self.cfg
        arc = self.track_phase_m + np.cumsum(
            np.concatenate(([0.0], np.asarray(speeds[:-1], dtype=float)))) * dt
        offsets, curvature = stadium_track(
            arc, cfg.benchmark_track_straight_m, cfg.benchmark_track_radius_m)
        return self.heading0 + offsets, np.asarray(speeds) * curvature

    def _overlay_track_escape_burst(self, dt: float, *, begin: int) -> None:
        """The dash on a closed track: the deck speeds up, the path does not move."""
        speeds = self._track_speeds
        peak, begin, hold_end, end, distance = self._escape_burst_speeds(
            speeds, dt, begin=begin)
        headings, yaw_rates = self._track_schedule(speeds, dt)
        self._track_headings[:] = headings
        self._track_yaw_rates[:] = yaw_rates
        self.escape_burst = {
            "start_s": begin * dt, "peak_m_s": peak,
            "hold_end_s": hold_end * dt, "end_s": end * dt,
            "distance_m": float(distance),
            "heading_rad": float(headings[begin])}

    def pull_away(self, sim_time: float) -> None:
        """Move off from a standing start, as if from a light.

        A lorry that steps from rest to cruise in one physics tick shears the
        vehicle parked on its roof and hands PX4 a deck twist that no real
        traffic produces. ``_traffic`` already models a light as a Gaussian dip
        in the cruise speed with a closed-form integral, so a full-depth stop
        centred on this instant *is* the pull-away: speed rises from zero over
        roughly ``2 * stop_sigma_s`` and the distance stays exact.

        Only the phase is re-anchored. The seed still draws the cruise speed,
        the lane, the wander and every light after this one.
        """
        if self.cfg.mode != "road" or self.speed <= 0.0:
            return
        # Carry the lap to where it stands before the clock is re-anchored,
        # exactly as reset does, so the lorry pulls away from where it waited.
        self.s0 = self.arc_length(sim_time)
        self.t0 = float(sim_time)
        self.stop_phase = 0.0
        if self.stop_depths.size == 0:
            self.stop_depths = np.ones(1)
        else:
            self.stop_depths = self.stop_depths.copy()
            self.stop_depths[0] = 1.0

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
            elif cfg.mode == "waypoints":
                point, tangent, speed = self._waypoint_pose(t)
                position = np.asarray(point, dtype=float).copy()
                position[2] += cfg.deck_height_m
                self._waypoint_tangent = np.asarray(tangent, dtype=float)
                return position, speed * self._waypoint_tangent
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
            elif cfg.mode == "random_walk":
                u = max(0.0, t) / cfg.motion_update_dt_s
                index = min(int(math.floor(u)), len(self.random_walk_position) - 1)
                fraction = min(max(u - index, 0.0), 1.0)
                # Interpolate along the TRACK, then anchor. Both terms of the
                # lerp have to be track coordinates: folding
                # ``random_walk_origin`` in before it (as this did until
                # 2026-09-22) makes the target ``track[index+1] - origin``, so
                # the reported deck slides the whole anchor vector backwards as
                # ``fraction`` sweeps 0 -> 1 and snaps back at the next sample.
                #
                # It is silent on the first episode, because ``route_start:
                # continue`` only makes the anchor non-zero once a previous
                # episode has left the deck somewhere. After that it is the
                # dominant term: measured on the minimal profile, one 0.1 s
                # motion-update interval swept 15 m of reported deck position
                # against the 0.0125 m the deck actually covers in a quarter of
                # it. Every consumer saw that sawtooth -- the FOV graph, the
                # relative-state supervision, the reward and the dashboard
                # trajectory panel, which is where it was finally noticed.
                track = self.random_walk_position[index]
                if index + 1 < len(self.random_walk_position):
                    track = track + fraction * (
                        self.random_walk_position[index + 1] - track)
                offset = track + self.random_walk_origin
                velocity = self.random_walk_velocity[index].copy()
        position = self.start + offset
        if not (cfg.mode == "random_walk"
                and self.benchmark_scenario == "vertical_heave_boat"):
            position[2] = self.start[2] + cfg.deck_height_m
        # Never let a profile drive the deck out of the arena the drone is
        # allowed to follow it into.
        radial = math.hypot(position[0], position[1])
        if radial > cfg.arena_radius_m > 0.0:
            position[0] *= cfg.arena_radius_m / radial
            position[1] *= cfg.arena_radius_m / radial
            velocity = np.zeros(3)
        return position, velocity

    def _waypoint_parameters(self) -> tuple[float, float, float]:
        """Return ``(ramp, acceleration, one-way duration)``."""
        if self.waypoint_route is None or self.speed <= 0.0:
            return 0.0, 0.0, math.inf
        length = self.waypoint_route.length
        ramp = min(float(self.cfg.waypoint_ramp_s), 0.49 * length / self.speed)
        acceleration = self.speed / ramp
        leg_time = length / self.speed + ramp
        return ramp, acceleration, leg_time

    def _waypoint_forward(self, elapsed: float) -> tuple[float, float]:
        """Distance and non-negative speed through one trapezoidal leg."""
        if self.waypoint_route is None or self.speed <= 0.0:
            return 0.0, 0.0
        ramp, acceleration, leg_time = self._waypoint_parameters()
        t = float(np.clip(elapsed, 0.0, leg_time))
        distance_ramp = 0.5 * self.speed * ramp
        if t < ramp:
            return 0.5 * acceleration * t * t, acceleration * t
        if t <= leg_time - ramp:
            return distance_ramp + self.speed * (t - ramp), self.speed
        remaining = leg_time - t
        return (self.waypoint_route.length - 0.5 * acceleration * remaining * remaining,
                acceleration * remaining)

    def _waypoint_progress(self, sim_time: float) -> tuple[float, float]:
        """Current route distance and direction, used to preserve reset pose."""
        if self.waypoint_route is None or self.speed <= 0.0:
            return 0.0, 1.0
        if self.cfg.waypoint_loop:
            distance, _ = self._waypoint_loop_forward(
                self.waypoint_phase + float(sim_time) - self.t0)
            return distance, 1.0
        _, _, leg_time = self._waypoint_parameters()
        phase = (self.waypoint_phase + float(sim_time) - self.t0) % (2.0 * leg_time)
        if phase <= leg_time:
            distance, _ = self._waypoint_forward(phase)
            return distance, 1.0
        distance_from_end, _ = self._waypoint_forward(phase - leg_time)
        return self.waypoint_route.length - distance_from_end, -1.0

    def _waypoint_phase_for(self, distance: float, direction: float) -> float:
        """Invert the trapezoid after a reseed changes the cruise speed."""
        if self.waypoint_route is None or self.speed <= 0.0:
            return 0.0
        ramp, acceleration, leg_time = self._waypoint_parameters()
        length = self.waypoint_route.length
        if self.cfg.waypoint_loop:
            # Resume at cruise speed after an episode boundary.  The modulo
            # term chooses a time whose accumulated distance is exactly the
            # carried point even when the newly seeded cruise speed changed.
            distance_ramp = 0.5 * self.speed * ramp
            return ramp + ((float(distance) - distance_ramp) % length) / self.speed
        d = float(np.clip(distance if direction >= 0.0 else length - distance,
                          0.0, length))
        distance_ramp = 0.5 * self.speed * ramp
        if d < distance_ramp:
            elapsed = math.sqrt(2.0 * d / acceleration)
        elif d <= length - distance_ramp:
            elapsed = ramp + (d - distance_ramp) / self.speed
        else:
            elapsed = leg_time - math.sqrt(2.0 * (length - d) / acceleration)
        return elapsed if direction >= 0.0 else leg_time + elapsed

    def _waypoint_pose(self, t: float) -> tuple[np.ndarray, np.ndarray, float]:
        """Position, forward tangent and signed speed on a smooth shuttle.

        A fixed-time trapezoidal profile eases the carrier at both ends. On the
        return leg the signed speed is negative while the forward tangent is
        retained, so a UGV reverses instead of attempting a narrow-road U-turn.
        """
        if self.waypoint_route is None:
            raise RuntimeError("waypoints mode has no route")
        if self.cfg.waypoint_loop:
            distance, signed_speed = self._waypoint_loop_forward(
                self.waypoint_phase + float(t))
            point, tangent = self.waypoint_route.at(distance)
            return point, tangent, signed_speed
        _, _, leg_time = self._waypoint_parameters()
        phase = (self.waypoint_phase + float(t)) % (2.0 * leg_time)
        if phase <= leg_time:
            distance, signed_speed = self._waypoint_forward(phase)
        else:
            distance_from_end, speed = self._waypoint_forward(phase - leg_time)
            distance = self.waypoint_route.length - distance_from_end
            signed_speed = -speed
        point, tangent = self.waypoint_route.at(distance)
        return point, tangent, signed_speed

    def _waypoint_loop_forward(self, elapsed: float) -> tuple[float, float]:
        """Distance and speed on a closed route with a single smooth launch."""
        if self.waypoint_route is None or self.speed <= 0.0:
            return 0.0, 0.0
        ramp, acceleration, _ = self._waypoint_parameters()
        t = max(0.0, float(elapsed))
        if t < ramp:
            distance = 0.5 * acceleration * t * t
            speed = acceleration * t
        else:
            distance = 0.5 * self.speed * ramp + self.speed * (t - ramp)
            speed = self.speed
        return distance % self.waypoint_route.length, speed

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
        if self.cfg.mode == "waypoints":
            target = math.atan2(float(self._waypoint_tangent[1]),
                                float(self._waypoint_tangent[0]))
        else:
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
