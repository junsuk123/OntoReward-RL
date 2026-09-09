from pathlib import Path

import pytest

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
