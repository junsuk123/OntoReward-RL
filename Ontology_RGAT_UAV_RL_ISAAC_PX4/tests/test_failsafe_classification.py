"""Which PX4 failsafe inputs may end a multi-hour run, and which may not.

2026-09-24 a run that had already collected its four teacher demonstrations and
trained sixteen PPO episodes died on

    PX4 reports an active failsafe (auto_mission_missing,
    manual_control_signal_lost, gcs_connection_lost, battery_unhealthy)

Three of those four are what PX4 publishes throughout autonomous SITL. The
fourth was classified hard, so ``collect_episode_resilient`` re-raised it
without spending one of its three retries, and the run ended.
"""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2_ws/src/ontology_rgat_px4"))

from ontology_rgat_px4.ros2_gateway import (          # noqa: E402
    FAILSAFE_BOOLEAN_FIELDS, FAILSAFE_HARD_FIELDS,
    FAILSAFE_SITL_INFRASTRUCTURE_FIELDS, failsafe_detail)

OBSERVED = ("auto_mission_missing", "manual_control_signal_lost",
            "gcs_connection_lost", "battery_unhealthy")


def _flags(*active, battery_warning=0):
    message = SimpleNamespace(battery_warning=int(battery_warning))
    for name in FAILSAFE_BOOLEAN_FIELDS:
        setattr(message, name, name in active)
    return message


def test_the_two_classifications_cannot_both_claim_an_input():
    assert not (FAILSAFE_HARD_FIELDS & FAILSAFE_SITL_INFRASTRUCTURE_FIELDS)
    assert (FAILSAFE_HARD_FIELDS | FAILSAFE_SITL_INFRASTRUCTURE_FIELDS
            ).issubset(set(FAILSAFE_BOOLEAN_FIELDS))


def test_the_2026_09_24_snapshot_is_recoverable_in_sitl():
    detail = failsafe_detail(_flags(*OBSERVED), target="sitl")
    assert sorted(detail["reasons"]) == sorted(OBSERVED)
    assert detail["recoverable_infrastructure"], (
        "this snapshot costs a retry, not the run")


def test_an_unhealthy_simulated_battery_is_not_a_hardware_verdict():
    """SITL only. On hardware a pack that reports unhealthy is a real pack.

    In SITL it cannot be a charge state at all: ``SIM_BAT_MIN_PCT`` is pinned
    at 100, and the pack the experiment measures is its own integrated model.
    """
    assert "battery_unhealthy" not in FAILSAFE_HARD_FIELDS
    assert "battery_unhealthy" in FAILSAFE_SITL_INFRASTRUCTURE_FIELDS
    for target in ("hardware", "hitl", "sihl"):
        assert not failsafe_detail(
            _flags(*OBSERVED), target=target)["recoverable_infrastructure"]


def test_a_real_fault_beside_it_still_ends_the_episode():
    """The relaxation must not launder a genuine failure travelling with it."""
    for hard in ("local_position_invalid", "primary_geofence_breached",
                 "fd_motor_failure", "battery_low_remaining_time"):
        detail = failsafe_detail(_flags(*OBSERVED, hard), target="sitl")
        assert not detail["recoverable_infrastructure"], hard
    # A PX4 battery WARNING is a level, not a health flag, and still counts.
    assert not failsafe_detail(
        _flags(*OBSERVED, battery_warning=2),
        target="sitl")["recoverable_infrastructure"]


def test_a_tip_over_is_still_handed_to_the_learner_as_a_crash():
    detail = failsafe_detail(_flags("fd_critical_failure"), target="sitl")
    assert detail["attitude_failure"]
    assert not detail["recoverable_infrastructure"]
