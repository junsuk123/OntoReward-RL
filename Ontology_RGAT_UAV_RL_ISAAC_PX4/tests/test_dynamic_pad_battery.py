import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from ontology_rgat_px4.battery import BatteryConfig, BatteryModel
from ontology_rgat_px4.protocol import ProtocolError, VehicleSample, encode, validate_goto


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "isaac_sim"))

from pad_motion import (  # noqa: E402
    PadMotionConfig, PadTrajectory, lorry_parts, ugv_parts)


@pytest.fixture(scope="module")
def system_config():
    with (ROOT / "config" / "system.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_pad_trajectory_is_seeded_and_scale_is_paired(system_config):
    cfg = PadMotionConfig.from_mapping(system_config)
    nominal = PadTrajectory(cfg)
    doubled = PadTrajectory(cfg)
    a = nominal.reset(seed=73, sim_time=10.0, speed_scale=1.0)
    b = doubled.reset(seed=73, sim_time=10.0, speed_scale=2.0)

    assert b["speed_m_s"] == pytest.approx(2.0 * a["speed_m_s"])
    assert b["heading_rad"] == pytest.approx(a["heading_rad"])
    assert np.asarray(b["position_enu_m"]) == pytest.approx(a["position_enu_m"])


def test_zero_pad_scale_is_the_static_control_condition(system_config):
    trajectory = PadTrajectory(PadMotionConfig.from_mapping(system_config))
    trajectory.reset(seed=9, sim_time=2.0, speed_scale=0.0)
    p0, _ = trajectory.pose(2.0)
    p1, v1 = trajectory.pose(16.0)

    assert p1 == pytest.approx(p0)
    assert v1 == pytest.approx(np.zeros(3))


def test_pad_velocity_is_derivative_of_position(system_config):
    trajectory = PadTrajectory(PadMotionConfig.from_mapping(system_config))
    trajectory.reset(seed=22, sim_time=0.0)
    t, dt = 4.0, 1e-4
    before, _ = trajectory.pose(t - dt)
    after, _ = trajectory.pose(t + dt)
    _, velocity = trajectory.pose(t)

    assert (after - before) / (2.0 * dt) == pytest.approx(velocity, abs=1e-5)


def test_battery_model_prices_hover_and_depletes(system_config):
    cfg = BatteryConfig.from_mapping(system_config)
    battery = BatteryModel(cfg)
    battery.reset(hover_seconds=1.0)
    battery.integrate(thrust_norm=0.58, hover_thrust_norm=0.58, dt=0.25)
    sample = battery.sample()

    assert sample["power_w"] == pytest.approx(battery.hover_power_w)
    assert sample["energy_used_j"] == pytest.approx(0.25 * battery.hover_power_w)
    assert sample["hover_seconds_remaining"] == pytest.approx(0.75)
    battery.integrate(thrust_norm=0.58, hover_thrust_norm=0.58, dt=1.0)
    assert battery.sample()["depleted"] is True


def test_battery_power_obeys_momentum_theory(system_config):
    battery = BatteryModel(BatteryConfig.from_mapping(system_config))
    hover = battery.hover_power_w
    battery.reset(hover_seconds=10.0)
    battery.integrate(thrust_norm=2.0 * 0.58, hover_thrust_norm=0.58, dt=0.02)
    expected = (hover - battery.cfg.avionics_power_w) * 2.0**1.5
    expected += battery.cfg.avionics_power_w

    assert battery.power_w == pytest.approx(expected)


def test_protocol_declares_pad_frame_and_finite_battery():
    message = VehicleSample(estimator_valid=True).to_message(1, 2, 1)

    assert message["position_frame"] == "pad"
    assert message["pad"]["valid"] is True
    assert message["battery"]["enabled"] is False
    encode(message)  # the unavailable-energy sentinel must still be strict JSON


def test_pad_relative_goto_has_tighter_guardrail():
    request = validate_goto({
        "position": [1.0, -2.0, 4.0], "yaw": 0.2, "frame": "pad"
    })
    assert request.is_pad_relative
    with pytest.raises(ProtocolError):
        validate_goto({"position": [11.0, 0.0, 4.0], "frame": "pad"})


def test_configured_pack_capacity_is_3s_3500mah(system_config):
    cfg = BatteryConfig.from_mapping(system_config)
    assert cfg.capacity_j == pytest.approx(3.5 * 11.1 * 3600.0)
    assert math.isfinite(BatteryModel(cfg).sample()["remaining_j"])


def test_pinning_t0_parks_the_lorry_without_moving_it(system_config):
    """``LandingDeck.advance`` parks the lorry by pinning ``t0`` to the clock.

    Every quantity the trajectory reports is a function of ``sim_time - t0``,
    so carrying ``t0`` along with the clock has to hold the pose exactly --
    not merely nearly. A deck that crept while it was held would carry the
    drone parked on its roof, which is the whole reason for holding it.
    """
    trajectory = PadTrajectory(PadMotionConfig.from_mapping(system_config))
    trajectory.reset(seed=31, sim_time=0.0)
    parked, _ = trajectory.pose(0.0)

    for step in range(1, 2001):                     # 8 s at the physics rate
        now = 0.004 * step
        trajectory.t0 = now
        position, _ = trajectory.pose(now)
        assert position == pytest.approx(parked, abs=1e-12)

    # And it pulls away from where it stood rather than stepping to cruise:
    # a kinematic deck that jumps to 8 m/s in one tick shears the vehicle
    # parked on its roof.
    trajectory.pull_away(8.0)
    moved, v0 = trajectory.pose(8.0)
    assert moved == pytest.approx(parked, abs=1e-9)
    # Only the lane wander is left -- centimetres per second across the
    # carriageway, against a cruise draw of metres per second along it.
    assert np.linalg.norm(v0[:2]) < 0.05

    speeds = [float(np.linalg.norm(trajectory.pose(8.0 + dt)[1][:2]))
              for dt in (0.5, 1.5, 3.0, 6.0)]
    assert speeds == sorted(speeds)                      # monotone spool-up
    assert speeds[0] < 0.35 * trajectory.speed           # gentle off the mark
    assert speeds[-1] > 0.9 * trajectory.speed           # and up to cruise


def test_the_lorry_is_a_lorry_and_not_a_floating_plank(system_config):
    """The deck used to be a bare slab with nothing under it.

    It moved like a lorry and occluded like a lorry, but the viewport showed a
    plank hanging three metres over the road. These are the constraints that
    make the drawn shape agree with the numbers the rest of the system uses:
    the roof *is* the landing deck, and the wheels *do* reach the road.
    """
    cfg = PadMotionConfig.from_mapping(system_config)
    parts = {p.name: p for p in lorry_parts(cfg.deck_size_m, cfg.deck_height_m)}
    length, width = cfg.deck_size_m
    road_z = -cfg.deck_height_m

    box = parts["cargo_box"]
    # The cargo box is exactly the deck footprint, and its lid is the marker
    # plane: anything else and the drone lands on a roof that is not the box.
    assert box.size[0] == pytest.approx(length)
    assert box.size[1] == pytest.approx(width)
    assert box.centre[2] + 0.5 * box.size[2] == pytest.approx(0.0, abs=1e-9)
    assert box.collider                       # or a policy flies through it

    wheels = [p for p in parts.values() if p.kind == "wheel"]
    assert len(wheels) == 6
    for wheel in wheels:
        # Resting on the road, not sunk into it and not hovering above it.
        assert wheel.centre[2] - 0.5 * wheel.size[0] == pytest.approx(road_z, abs=1e-9)
        assert abs(wheel.centre[1]) == pytest.approx(0.5 * width)

    cab = parts["cab"]
    # Ahead of the box and lower than it, which is what makes the silhouette
    # read as a box lorry from above rather than as one long container.
    assert cab.centre[0] - 0.5 * cab.size[0] >= 0.5 * length - 1e-9
    assert cab.centre[2] + 0.5 * cab.size[2] < 0.0

    # Nothing pokes up through the landing deck.
    assert max(p.centre[2] + 0.5 * p.size[2] for p in parts.values()) <= 1e-9


def test_the_lorry_follows_the_configured_deck(system_config):
    """A taller deck lifts the body; a longer one lengthens it. No magic numbers."""
    cfg = PadMotionConfig.from_mapping(system_config)
    tall = {p.name: p for p in lorry_parts(cfg.deck_size_m, cfg.deck_height_m + 1.0)}
    base = {p.name: p for p in lorry_parts(cfg.deck_size_m, cfg.deck_height_m)}

    assert tall["cargo_box"].size[2] == pytest.approx(base["cargo_box"].size[2] + 1.0)
    for name, wheel in tall.items():
        if wheel.kind == "wheel":
            assert wheel.centre[2] == pytest.approx(base[name].centre[2] - 1.0)

    longer = {p.name: p for p in lorry_parts((cfg.deck_size_m[0] + 2.0,
                                              cfg.deck_size_m[1]), cfg.deck_height_m)}
    assert longer["cargo_box"].size[0] == pytest.approx(cfg.deck_size_m[0] + 2.0)


def _waypoint_config():
    return PadMotionConfig.from_mapping({
        "pad": {
            "carrier": "ugv",
            "motion": "waypoints",
            "deck_size_m": [1.6, 1.0],
            "deck_height_m": 0.75,
            "speed_range_m_s": [1.0, 1.0],
            "route_start": "continue",
            "route_waypoints_enu_m": [
                [-2.0, 4.0, 1.0],
                [2.0, 4.0, 1.2],
                [2.0, 8.0, 1.4],
            ],
            "arena_radius_m": 0.0,
        }
    })


def test_waypoint_ugv_follows_terrain_and_reverses_smoothly():
    cfg = _waypoint_config()
    trajectory = PadTrajectory(cfg)
    trajectory.reset(seed=3, sim_time=0.0)
    length = trajectory.waypoint_route.length

    start, start_velocity = trajectory.pose(0.0)
    reverse_time = trajectory._waypoint_parameters()[2]
    end, end_velocity = trajectory.pose(reverse_time)
    returning, return_velocity = trajectory.pose(1.5 * reverse_time)

    assert start == pytest.approx([-2.0, 4.0, 1.75])
    assert start_velocity == pytest.approx(np.zeros(3), abs=1e-12)
    assert end == pytest.approx([2.0, 8.0, 2.15])
    assert end_velocity == pytest.approx(np.zeros(3), abs=1e-12)
    # It backs down the same narrow route rather than spinning for a U-turn.
    assert np.dot(return_velocity, trajectory._waypoint_tangent) < 0.0

    t, dt = 2.3, 1e-5
    before, _ = trajectory.pose(t - dt)
    after, _ = trajectory.pose(t + dt)
    _, velocity = trajectory.pose(t)
    assert (after - before) / (2.0 * dt) == pytest.approx(velocity, abs=1e-5)
    assert returning[2] > 1.75  # the deck height follows the sloping road


def test_waypoint_route_is_continuous_across_episode_reset():
    trajectory = PadTrajectory(_waypoint_config())
    trajectory.reset(seed=11, sim_time=0.0)
    before, _ = trajectory.pose(7.25)
    trajectory.reset(seed=12, sim_time=7.25)
    after, _ = trajectory.pose(7.25)

    assert after == pytest.approx(before, abs=1e-12)


def test_ranger_mini_v3_matches_official_dimensions_and_urdf_wheels():
    length, width, height = 1.6, 1.0, 0.75
    parts = ugv_parts((length, width), height)
    wheels = [part for part in parts if part.kind == "wheel"]

    assert len(wheels) == 4
    assert any(part.name == "ranger_collision" and part.collider for part in parts)
    for part in parts:
        assert abs(part.centre[0]) + 0.5 * part.size[0] <= 0.5 * length + 1e-9
        assert abs(part.centre[1]) + 0.5 * part.size[1] <= 0.5 * width + 1e-9
        assert part.centre[2] + 0.5 * part.size[2] <= 1e-9
    for wheel in wheels:
        assert wheel.centre[2] - 0.5 * wheel.size[0] == pytest.approx(-height)
        assert abs(wheel.centre[0]) == pytest.approx(0.25)
        assert abs(wheel.centre[1]) == pytest.approx(0.19)
        assert wheel.size == pytest.approx((0.18, 0.08, 0.18))
