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

from pad_motion import PadMotionConfig, PadTrajectory  # noqa: E402


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
