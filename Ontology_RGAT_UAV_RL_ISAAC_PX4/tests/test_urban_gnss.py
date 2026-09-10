"""The urban environment: the city, the lorry on the road, and the GNSS it ruins.

Each of these is a property the experiment depends on and that nothing else
would catch. The city and the satellite mask have to be the same geometry, or
an outage has no cause you can point at in the viewport; the route has to stay
on the carriageway, or the lorry drives through a building; the fix has to be
degraded by the canyon and by nothing else, or the sweep measures the wrong
thing; and the integrity the policy is shown has to be built from observables,
or the whole result is a receiver reading its own error.
"""
from __future__ import annotations

import copy
import dataclasses
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "isaac_sim"))

from gnss import (  # noqa: E402
    MIN_SATELLITES, GnssConfig, GnssFix, UrbanGnss, hil_gps_measurement)
from pad_motion import PadMotionConfig, PadTrajectory, RoadRoute  # noqa: E402
from urban_scene import UrbanConfig, UrbanLayout  # noqa: E402


@pytest.fixture(scope="module")
def system_config():
    with (ROOT / "config" / "system.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


@pytest.fixture(scope="module")
def layout(system_config):
    """The synthetic block, whatever the shipped config is pointed at.

    These tests check the occlusion model against a geometry whose answer is
    known analytically -- facades parallel to the street, gaps exactly where
    the config puts them. A real OpenStreetMap extract has neither, so running
    them against `urban.source: osm` would only assert that Myeongdong happens
    to be shaped like the synthetic block. The real city is covered by
    test_osm_city.py instead.
    """
    synthetic = copy.deepcopy(system_config)
    synthetic.setdefault("urban", {})["source"] = "synthetic"
    return UrbanLayout(UrbanConfig.from_mapping(synthetic))


def _mid_block(cfg: UrbanConfig, height: float = 3.2) -> list[float]:
    """A point on the south carriageway, away from any cross street."""
    return [0.25 * cfg.block_size_m[0], -0.5 * cfg.block_size_m[1], height]


def _corner(cfg: UrbanConfig, height: float = 3.2) -> list[float]:
    return [0.5 * cfg.block_size_m[0], -0.5 * cfg.block_size_m[1], height]


# ------------------------------------------------------------------- the city
def test_a_street_canyon_blocks_across_the_street_and_not_along_it(layout):
    """The defining property of a canyon, and the reason the fix degrades."""
    point = _mid_block(layout.cfg)
    across = max(layout.sky_mask(point, 0.5 * math.pi),
                 layout.sky_mask(point, -0.5 * math.pi))
    along = max(layout.sky_mask(point, 0.0), layout.sky_mask(point, math.pi))
    assert math.degrees(across) > 30.0
    assert math.degrees(along) < 10.0


def test_an_intersection_lets_the_sky_back_in(layout):
    """A lap has to be a sequence of outages and recoveries, not one number."""
    assert (layout.sky_view_fraction(_corner(layout.cfg))
            > layout.sky_view_fraction(_mid_block(layout.cfg)))


def test_there_is_no_canyon_above_the_roofline(layout):
    high = _mid_block(layout.cfg, height=2.0 * layout.cfg.height_range_m[1])
    assert layout.sky_view_fraction(high) == pytest.approx(1.0)


def test_the_batched_occlusion_test_matches_the_scalar_one(layout):
    point = _mid_block(layout.cfg)
    azimuth = np.linspace(0.0, 2.0 * math.pi, 23)
    elevation = np.linspace(0.1, 1.4, 23)
    hit, distance = layout.blocked_batch(point, azimuth, elevation)
    for i, (az, el) in enumerate(zip(azimuth, elevation)):
        scalar_hit, scalar_distance = layout.blocked(point, az, el)
        assert bool(hit[i]) == scalar_hit
        assert distance[i] == pytest.approx(scalar_distance)


# ------------------------------------------------------------------ the route
def test_the_lorry_drives_on_the_carriageway_and_not_through_a_building(
        system_config, layout):
    """The route and the city are configured from the same block."""
    cfg = PadMotionConfig.from_mapping(system_config)
    urban = layout.cfg
    half_width = 0.5 * cfg.deck_size_m[1]
    for seed in (5, 17, 31):
        trajectory = PadTrajectory(cfg)
        trajectory.reset(seed=seed, sim_time=0.0)
        for t in np.linspace(0.0, 240.0, 600):
            position, _ = trajectory.pose(t)
            centre, tangent, _ = trajectory.route.at(
                trajectory.s0 + trajectory._traffic(t)[0])
            normal = RoadRoute.left_normal(tangent)
            lateral = float(np.dot(position[:2] - centre, normal))
            # Right-hand traffic on a counter-clockwise lap: the lorry stays on
            # its own side, and a lane change must not put it into oncoming
            # traffic. Its whole width has to fit the carriageway, not just its
            # centreline.
            assert lateral < 0.0
            assert abs(lateral) + half_width <= urban.road_half_width_m
            for building in layout.buildings:
                inside = (building.x0 <= position[0] <= building.x1
                          and building.y0 <= position[1] <= building.y1)
                assert not inside


def test_a_lane_change_stays_on_the_lorrys_own_side(system_config):
    """Every lane the trajectory can pick is on the right of the centreline."""
    cfg = PadMotionConfig.from_mapping(system_config)
    assert cfg.lane_centres_m
    assert all(lane < 0.0 for lane in cfg.lane_centres_m)
    chosen = set()
    for seed in range(40):
        trajectory = PadTrajectory(cfg)
        trajectory.reset(seed=seed, sim_time=0.0)
        chosen.add(round(trajectory.lane_base, 3))
        chosen.add(round(trajectory.lane_base + trajectory.lane_change_to, 3))
    assert chosen == {round(lane, 3) for lane in cfg.lane_centres_m}


def test_a_lorry_too_wide_for_its_lane_is_a_configuration_error(system_config):
    """Discovered at load, not as a lorry clipping the kerb mid-episode."""
    wide = {"pad": dict(system_config["pad"], deck_size_m=[6.2, 9.0]),
            "urban": system_config["urban"]}
    with pytest.raises(ValueError, match="does not fit"):
        PadMotionConfig.from_mapping(wide)


def test_traffic_stops_the_lorry_and_the_profile_stays_forwards(system_config):
    """A stop is what makes some episodes winnable; reverse gear is not traffic."""
    cfg = PadMotionConfig.from_mapping(system_config)
    trajectory = PadTrajectory(cfg)
    trajectory.reset(seed=3, sim_time=0.0)
    times = np.linspace(0.0, 4.0 * cfg.stop_interval_s, 4000)
    speed = np.array([float(np.linalg.norm(trajectory.pose(t)[1])) for t in times])
    distance = np.array([trajectory._traffic(t)[0] for t in times])

    assert speed.min() < 0.35 * trajectory.speed
    assert speed.max() > 0.9 * trajectory.speed
    assert np.all(np.diff(distance) >= -1e-9)


def test_road_velocity_is_the_derivative_of_position_through_the_corners(
        system_config):
    """Closed form all the way round, so the fed-forward twist is exact."""
    trajectory = PadTrajectory(PadMotionConfig.from_mapping(system_config))
    trajectory.reset(seed=11, sim_time=0.0)
    delta = 1e-5
    for t in np.linspace(0.5, 200.0, 900):
        before, _ = trajectory.pose(t - delta)
        after, _ = trajectory.pose(t + delta)
        _, velocity = trajectory.pose(t)
        assert (after - before) / (2.0 * delta) == pytest.approx(velocity, abs=1e-5)


def test_the_route_is_closed_so_the_lorry_never_runs_out_of_road(system_config):
    cfg = PadMotionConfig.from_mapping(system_config)
    route = RoadRoute(cfg.route_size_m[0], cfg.route_size_m[1],
                      cfg.route_corner_radius_m)
    start, tangent, _ = route.at(0.0)
    wrapped, wrapped_tangent, _ = route.at(route.perimeter)
    assert start == pytest.approx(wrapped)
    assert tangent == pytest.approx(wrapped_tangent)


def test_nothing_is_spawned_inside_a_building(system_config, layout):
    """A regression: the deck's own start is not the route's origin.

    The route is centred on the city block, so its origin is the middle of the
    block -- inside a facade. Taking the configured start position as the deck's
    start put the deck, and the vehicle that spawns on its roof, 33 m deep in
    masonry while the lorry drove away without them.
    """
    cfg = PadMotionConfig.from_mapping(system_config)
    deck, _ = PadTrajectory(cfg).pose(0.0)
    drone = deck + np.asarray(system_config["isaac"]["spawn_position_enu_m"], dtype=float)

    assert not layout.contains(deck), f"deck spawns inside a building at {deck}"
    assert not layout.contains(drone), f"vehicle spawns inside a building at {drone}"
    assert deck[2] == pytest.approx(cfg.deck_height_m)


def test_the_lorry_is_never_left_far_from_the_vehicle_parked_on_it(system_config):
    """``route_start: continue`` has to be continuous, not merely close.

    A reset redraws how the lorry drives but must not move it: the vehicle is
    sitting on its roof, and a deck that jumps out from under it either strands
    it or shears it off. ``seeded`` is the honest alternative and is checked
    here for what it costs, so the trade is a measured one.
    """
    cfg = PadMotionConfig.from_mapping(system_config)
    assert cfg.route_start == "continue"

    def jumps(mode):
        trajectory = PadTrajectory(dataclasses.replace(cfg, route_start=mode))
        clock = 0.0
        trajectory.reset(1, clock)
        out = []
        for episode in range(2, 30):
            clock += float(system_config["landing"]["max_time_s"])
            before, _ = trajectory.pose(clock)
            trajectory.reset(episode, clock)
            after, _ = trajectory.pose(clock)
            out.append(float(np.linalg.norm(after - before)))
        return out

    assert max(jumps("continue")) < 0.01
    # And the reason the default is not the other one.
    assert max(jumps("seeded")) > 10.0


def test_the_entry_pose_is_never_inside_a_building(system_config, layout):
    """The point PX4 is told to fly to before the policy takes over.

    A Gaussian tail puts it inside a facade about once in three thousand seeds,
    and PX4 obligingly flies into the wall, never reaches the pose, and times
    out the whole reset ninety seconds later. Rare enough to survive a smoke
    test and common enough to cost a paper-scale sweep several restarts.
    """
    cfg = PadMotionConfig.from_mapping(system_config)
    trajectory = PadTrajectory(cfg)
    clock = 0.0
    trajectory.reset(0, clock)
    guarded = 0
    for seed in range(3000):
        clock += 20.0
        trajectory.reset(seed, clock)
        deck, _ = trajectory.pose(clock)
        rng = np.random.default_rng(seed)
        raw = np.array([1.8 * rng.normal(), 1.4 * rng.normal(),
                        4.0 + 1.8 * rng.random()])
        entry = layout.clear_of_buildings(deck + raw, deck)
        assert not layout.contains(entry)
        guarded += not np.allclose(entry, deck + raw)
    # The guard must be a tail correction, not a reshaping of the draw.
    assert guarded < 30, f"the entry distribution was clipped {guarded} times in 3000"


def test_the_chase_camera_never_ends_up_inside_a_building(system_config, layout):
    """A regression, and one that looks exactly like a broken simulator.

    The GUI camera offset that framed the vehicle over an open field reaches
    past the facade of a nineteen-metre street. Inside a facade it renders a
    black screen carrying nothing but the unlit debug overlay -- a drone, a
    lorry and a line, hanging in the dark. This checks the shipped offset
    against the shipped city, which is the pair that actually broke.
    """
    view = system_config["isaac"]["viewport_follow"]
    assert view.get("frame") == "street", "a world-frame offset cannot know where the road is"
    offset = np.asarray(view["offset_m"], dtype=float)
    cfg = PadMotionConfig.from_mapping(system_config)
    spawn = float(system_config["isaac"]["spawn_position_enu_m"][2])
    route = RoadRoute(cfg.route_size_m[0], cfg.route_size_m[1], cfg.route_corner_radius_m)

    samples = pulled = 0
    for i in range(240):
        centre, tangent, _ = route.at(route.perimeter * i / 240.0)
        across = RoadRoute.left_normal(tangent)
        for lane in cfg.lane_centres_m:
            deck = centre + lane * across
            drone = np.array([deck[0], deck[1], cfg.deck_height_m + spawn])
            eye = drone + np.array([
                offset[0] * tangent[0] + offset[1] * across[0],
                offset[0] * tangent[1] + offset[1] * across[1],
                offset[2]])
            placed = layout.clear_of_buildings(eye, drone)
            assert not layout.contains(placed), f"camera buried at {placed}"
            samples += 1
            pulled += not np.allclose(placed, eye)
    # The offset itself has to be usable almost everywhere. Recovering it on
    # every frame would mean the shipped framing is wrong and the guard is
    # covering for it -- on the outside of a corner is the one place the
    # instantaneous tangent points into masonry.
    assert pulled < 0.1 * samples, f"pulled in on {pulled}/{samples} of the lap"


def test_a_point_inside_a_facade_is_reported_as_inside(layout):
    building = layout.buildings[0]
    centre = np.array([*building.center, 0.5 * building.height_m])
    assert layout.contains(centre)
    assert not layout.contains(centre + np.array([0.0, 0.0, building.height_m]))
    assert not layout.contains([600.0, 600.0, 1.0])


# ------------------------------------------------------------------- the fix
def _settle(gnss: UrbanGnss, uav, deck, steps: int = 60):
    for _ in range(steps):
        fixes = gnss.update(uav, deck, 0.02)
    return fixes


def test_open_sky_scale_is_the_control_condition(system_config, layout):
    """Scale zero has to leave the city standing and the fix perfect."""
    gnss = UrbanGnss(GnssConfig.from_mapping(system_config), layout)
    gnss.reset(4, scale=0.0)
    uav, deck = _settle(gnss, _mid_block(layout.cfg, 8.0), _mid_block(layout.cfg))
    assert uav.quality == pytest.approx(1.0)
    assert np.allclose(uav.error_enu_m, 0.0)
    assert np.allclose(deck.error_enu_m, 0.0)
    # And the buildings are still there to hide the markers and steer the wind.
    assert layout.sky_view_fraction(_mid_block(layout.cfg)) < 1.0


def test_the_canyon_costs_satellites_accuracy_and_integrity(system_config, layout):
    gnss = UrbanGnss(GnssConfig.from_mapping(system_config), layout)
    canyon, open_sky = [], []
    for seed in range(6):
        gnss.reset(seed, scale=1.0)
        canyon.append(_settle(gnss, _mid_block(layout.cfg, 8.0),
                              _mid_block(layout.cfg))[0])
        gnss.reset(seed, scale=1.0)
        open_sky.append(_settle(gnss, [600.0, 600.0, 8.0], [600.0, 600.0, 3.2])[0])

    assert np.mean([f.satellites_nlos for f in canyon]) > 1.0
    assert np.mean([f.satellites_nlos for f in open_sky]) == 0.0
    assert (np.mean([f.sigma_xy_m for f in canyon])
            > 3.0 * np.mean([f.sigma_xy_m for f in open_sky]))
    assert np.mean([f.quality for f in canyon]) < 0.6
    assert np.mean([f.quality for f in open_sky]) > 0.9
    assert (np.mean([np.linalg.norm(f.error_enu_m[:2]) for f in canyon])
            > 3.0 * np.mean([np.linalg.norm(f.error_enu_m[:2]) for f in open_sky]))


def test_nlos_bias_is_horizontal_because_the_barometer_holds_altitude(
        system_config, layout):
    """A canyon's vertical GNSS error is real and largely rejected; say so."""
    gnss = UrbanGnss(GnssConfig.from_mapping(system_config), layout)
    horizontal, vertical = [], []
    for seed in range(6):
        gnss.reset(seed, scale=1.0)
        fix = _settle(gnss, _mid_block(layout.cfg, 8.0), _mid_block(layout.cfg))[0]
        horizontal.append(float(np.linalg.norm(fix.error_enu_m[:2])))
        vertical.append(abs(float(fix.error_enu_m[2])))
    assert np.mean(horizontal) > np.mean(vertical)


def test_the_two_receivers_share_a_sky_so_the_relative_fix_is_the_better_one(
        system_config, layout):
    """The reason a drone can chase a lorry it cannot absolutely locate."""
    gnss = UrbanGnss(GnssConfig.from_mapping(system_config), layout)
    absolute, relative = [], []
    # Receiver-specific C/N0 noise can make a short seed window misleading:
    # one receiver may catch an NLOS signal the other misses. Over a modest
    # ensemble the shared geometry remains the dominant effect.
    for seed in range(32):
        gnss.reset(seed, scale=1.0)
        uav, deck = _settle(gnss, _mid_block(layout.cfg, 6.0), _mid_block(layout.cfg))
        absolute.append(0.5 * (np.linalg.norm(uav.error_enu_m[:2])
                               + np.linalg.norm(deck.error_enu_m[:2])))
        relative.append(np.linalg.norm((uav.error_enu_m - deck.error_enu_m)[:2]))
    assert np.mean(relative) < np.mean(absolute)


def _walled_in() -> UrbanLayout:
    """A shaft with no sky: every line of sight crosses a facade."""
    return UrbanLayout(UrbanConfig.from_mapping({"urban": {
        "block_size_m": [8.0, 8.0], "corner_radius_m": 1.0,
        "road_half_width_m": 1.0, "sidewalk_m": 0.5,
        "height_range_m": [200.0, 220.0], "facade_depth_m": 40.0,
        "facade_segment_m": 400.0, "cross_street_gap_m": 0.0,
        "mid_block_gap": False, "seed": 1}}))


def test_a_fix_made_entirely_of_reflections_still_collapses_the_integrity(
        system_config):
    """Losing every direct signal does not usually mean losing the fix.

    A receiver keeps tracking the reflections and goes on publishing a
    position, and their biases agree with one another, so no consistency check
    can see anything wrong. What catches it is that the signals are too weak
    for the elevations they claim -- and only partly, because a few reflections
    are strong. The integrity has to collapse anyway.
    """
    gnss = UrbanGnss(GnssConfig.from_mapping(system_config), _walled_in())
    gnss.reset(2, scale=1.0)
    fix = _settle(gnss, [0.0, 0.0, 1.0], [0.0, 0.0, 1.0])[0]

    assert fix.satellites_los == 0
    assert fix.nlos_fraction == pytest.approx(1.0)
    assert fix.sky_view == pytest.approx(0.0)
    # The detector is not an oracle: some reflections come in strong enough to
    # pass for a direct signal, which is why the reported integrity is low
    # rather than zero.
    assert 0.3 < fix.nlos_detected_fraction < 1.0
    assert fix.quality < 0.5


def test_a_tunnel_is_an_outage_and_not_a_confident_fix(system_config):
    """With no reflection strong enough to hold, there is no fix at all."""
    cfg = dataclasses.replace(GnssConfig.from_mapping(system_config),
                              nlos_tracking_probability=0.0)
    gnss = UrbanGnss(cfg, _walled_in())
    gnss.reset(2, scale=1.0)
    fix = _settle(gnss, [0.0, 0.0, 1.0], [0.0, 0.0, 1.0])[0]

    assert fix.satellites_tracked < MIN_SATELLITES
    assert not fix.valid
    assert fix.fix_type == 0
    assert fix.quality == 0.0
    # The estimate coasts rather than freezing, so an outage is visible as
    # drift instead of as a suspiciously steady position.
    assert np.linalg.norm(fix.error_enu_m) > 0.0


def test_hil_gps_carries_the_urban_fix_with_mavlink_units(system_config):
    cfg = GnssConfig.from_mapping(system_config)
    fix = GnssFix(valid=True, fix_type=3, satellites_tracked=7,
                  hdop=2.0, vdop=3.0, sigma_xy_m=4.25,
                  error_enu_m=np.array([8.0, -3.0, 1.5]),
                  velocity_error_enu_m_s=np.array([0.2, -0.1, 0.05]))
    measurement = hil_gps_measurement(
        fix, np.array([100.0, 200.0, 10.0]), np.array([1.0, 2.0, -0.5]),
        37.5636, 126.9850, 25.0, cfg)

    assert measurement["fix_type"] == 3
    assert measurement["sattelites_visible"] == 7
    assert measurement["eph"] == 425  # MAVLink HIL_GPS EPH is centimetres
    assert measurement["epv"] == 638
    assert measurement["altitude"] == pytest.approx(36.5)
    assert measurement["velocity_east"] == pytest.approx(1.2)
    assert measurement["velocity_north"] == pytest.approx(1.9)
    assert measurement["velocity_down"] == pytest.approx(0.45)
    # East changes longitude and north changes latitude, never the reverse.
    assert measurement["longitude"] > measurement["longitude_gt"]
    assert measurement["latitude"] < measurement["latitude_gt"]


def test_hil_gps_outage_forces_px4_to_dead_reckon(system_config):
    cfg = GnssConfig.from_mapping(system_config)
    fix = GnssFix(valid=False, fix_type=0, satellites_tracked=2,
                  error_enu_m=np.zeros(3), velocity_error_enu_m_s=np.zeros(3))
    measurement = hil_gps_measurement(
        fix, np.zeros(3), np.zeros(3), 37.0, 127.0, 20.0, cfg)

    assert measurement["fix_type"] == 0
    assert measurement["sattelites_visible"] == 2
    assert measurement["eph"] == 65535
    assert measurement["epv"] == 65535


def test_a_fix_is_reproducible_in_a_fresh_interpreter(system_config, tmp_path):
    """Every draw here has to be a function of the episode seed and nothing else.

    Python salts string hashes per process, so a seed derived from one makes
    the noise a receiver draws depend on which interpreter drew it. Two
    interpreters with deliberately different hash seeds must agree.
    """
    script = tmp_path / "fix.py"
    script.write_text(
        "import sys, yaml, numpy as np\n"
        f"sys.path.insert(0, {str(ROOT / 'isaac_sim')!r})\n"
        f"cfg = yaml.safe_load(open({str(ROOT / 'config' / 'system.yaml')!r}))\n"
        "from urban_scene import UrbanConfig, UrbanLayout\n"
        "from gnss import GnssConfig, UrbanGnss\n"
        "g = UrbanGnss(GnssConfig.from_mapping(cfg), UrbanLayout(UrbanConfig.from_mapping(cfg)))\n"
        "g.reset(31, 1.0)\n"
        "for _ in range(40):\n"
        "    u, d = g.update([20.0, -31.0, 8.0], [20.0, -31.0, 3.2], 0.02)\n"
        "print(repr((u.quality, u.sigma_xy_m, list(u.error_enu_m))))\n")
    outputs = []
    for hash_seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        outputs.append(subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True,
            env=env, check=True).stdout.strip())
    assert len(set(outputs)) == 1, outputs


def test_the_simulator_truth_is_labelled_as_truth_on_the_wire():
    """It crosses to the gateway; it must be impossible to pick up by accident."""
    wire = GnssFix(error_enu_m=np.array([9.0, -4.0, 1.0]), nlos_fraction=0.8).to_dict()
    assert set(wire) - {"truth"} == set(GnssFix().observables())
    assert wire["truth"]["error_enu_m"] == [9.0, -4.0, 1.0]
    assert wire["truth"]["nlos_fraction"] == pytest.approx(0.8)


def test_the_reported_fix_never_carries_the_true_error_to_the_policy():
    """The receiver's own report is the whole of what the policy may see.

    The error vector does cross the simulator-to-gateway wire, because the
    gateway is what injects it into the estimate. It must stop there.
    """
    sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "ontology_rgat_px4"))
    from ontology_rgat_px4.protocol import VehicleSample
    from ontology_rgat.env import _GNSS_FIELDS

    wire = VehicleSample().to_message(1, 1, 0)["gnss"]
    for hidden in ("error_enu_m", "velocity_error_enu_m_s", "truth",
                   # The true NLOS count and sky view are the simulator's too:
                   # what the receiver has is a detector with false alarms and
                   # misses, and the policy has to live with that one.
                   "nlos_fraction", "satellites_nlos", "sky_view"):
        assert hidden not in wire
        assert hidden not in _GNSS_FIELDS
    assert "nlos_detected_fraction" in wire
