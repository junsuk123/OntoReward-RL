from pathlib import Path

import pytest

from ontology_rgat.cli import base_parser, config_from_args
from ontology_rgat_px4.config import load_gateway_config
from ontology_rgat_px4.safety import (
    HARDWARE_ARM_PHRASE,
    HARDWARE_OFFBOARD_PHRASE,
    SafetyGate,
)


ROOT = Path(__file__).resolve().parents[1]


def test_configuration_loads():
    cfg = load_gateway_config(ROOT / "config" / "system.yaml")
    assert cfg.target == "sitl"
    assert cfg.control_hz == 50.0
    assert cfg.namespace == "/fmu"
    assert cfg.gnss_dr_enter_quality == 0.45


def test_gateway_loads_metasejong_overlay_with_the_simulator_datum():
    cfg = load_gateway_config(ROOT / "config" / "metasejong-demo.yaml")

    assert cfg.pad_motion == "waypoints"
    assert cfg.pad_deck_height_m == pytest.approx(0.42)
    assert cfg.map_latitude_deg == pytest.approx(37.5503)
    assert cfg.map_longitude_deg == pytest.approx(127.0736)
    assert cfg.map_altitude_m == pytest.approx(30.0)
    farthest_waypoint_m = (151.0**2 + 198.0**2) ** 0.5
    assert cfg.world_radius_m == pytest.approx(farthest_waypoint_m + 30.0)


def test_learner_selects_the_same_metasejong_pipeline_yaml():
    path = ROOT / "config" / "metasejong-pipeline.yaml"
    args = base_parser("test").parse_args(["--system-config", str(path)])

    cfg = config_from_args(args)

    assert Path(cfg.paths.system_yaml) == path.resolve()
    assert cfg.reward.fixed.weight_min == pytest.approx(0.025)
    assert cfg.reward.fixed.weight_max == pytest.approx(0.45)
    assert cfg.eval.acceptance.min_success_rate == pytest.approx(0.60)
    assert cfg.eval.acceptance.max_success_std == pytest.approx(0.15)


def test_sitl_reset_and_arm_gate():
    gate = SafetyGate("sitl", allow_arm=True)
    gate.require_reset()
    gate.require_arm()


def test_hardware_never_resets():
    with pytest.raises(PermissionError):
        SafetyGate("hardware", allow_arm=False).require_reset()


def test_hardware_arm_needs_two_factors(monkeypatch):
    gate = SafetyGate("hardware", allow_arm=True)
    with pytest.raises(PermissionError):
        gate.require_arm()
    monkeypatch.setenv("ONTOLOGY_RGAT_HARDWARE_ARM", HARDWARE_ARM_PHRASE)
    gate.require_arm()


def test_hardware_offboard_needs_two_factors(monkeypatch):
    gate = SafetyGate("hardware", allow_arm=False, allow_offboard=True)
    with pytest.raises(PermissionError):
        gate.require_offboard()
    monkeypatch.setenv("ONTOLOGY_RGAT_HARDWARE_OFFBOARD", HARDWARE_OFFBOARD_PHRASE)
    gate.require_offboard()


def test_sitl_may_fly_the_entry_pose():
    SafetyGate("sitl", allow_arm=True).require_autonomous_climb()


def test_hardware_never_climbs_autonomously(monkeypatch):
    monkeypatch.setenv("ONTOLOGY_RGAT_HARDWARE_OFFBOARD", HARDWARE_OFFBOARD_PHRASE)
    gate = SafetyGate("hardware", allow_arm=True, allow_offboard=True)
    with pytest.raises(PermissionError):
        gate.require_autonomous_climb()
