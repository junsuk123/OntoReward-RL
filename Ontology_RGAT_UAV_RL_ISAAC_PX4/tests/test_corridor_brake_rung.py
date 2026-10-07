"""``spatial-reference/9``: the terminal-corridor brake holds the descent at the
reward's sink target and acts within one decision step.

Through /8 the corridor brakes only above 1.5 x ``touchdown_z_speed`` (0.45 m/s)
and spreads the correction over the ~0.35 s response delay, so a policy may
touch at 0.30-0.45 m/s and the contact verdict scores it UNSAFE_CONTACT; on /8
96 of 101 held-out unsafe terminals were exactly that, at a median 0.34 m/s.
Measured on the same 15 /8 checkpoints over 48 held-out seeds with nothing
retrained (2026-10-07): 1.5 x over the delay 76.5 % landing / 14.2 % unsafe;
0.8 x over the delay 81.0 / 10.3; 1.0 x within one step 79.6 / 11.5; 0.8 x
within one step 87.9 / 2.8 with timeouts and aborts unchanged. /9 is /8 plus
that one law; this file holds it to the two constants and to /8 otherwise.
"""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ontology_rgat.spatial.core import REFERENCE_SCHEMA, REFERENCE_SCHEMAS, SpatialConfig
from ontology_rgat.spatial.dynamics import THRUST_TAU
from ontology_rgat.spatial.safety import ReferenceSpatialSupervisor
from ontology_rgat.spatial.training import _graph_planes, _ontology


def _cfg(schema):
    from ontology_rgat.spatial.runtime_contract import deployment_profile
    return replace(SpatialConfig(), schema=schema,
                   isaac_profile_sha256=deployment_profile(schema)["sha256"])


class _Corridor:
    """The supervisor decision under test: inside the terminal corridor."""
    inhibited, abort = False, False


class _Own:
    def __init__(self, vz):
        self.own_velocity = np.array([0.0, 0.0, float(vz)])
        self.own_position = np.zeros(3)


class _Estimate:
    def __init__(self, vz):
        self.own = _Own(vz)


def _brake(schema, vz, requested=(0.0, 0.0, -0.3)):
    """Applied vertical acceleration (m/s^2) for a vehicle descending at |vz|
    inside the corridor while the policy still asks for more descent."""
    cfg = _cfg(schema)
    supervisor = ReferenceSpatialSupervisor()
    supervisor.terminal_descent = True
    normalized = supervisor.action(np.asarray(requested, dtype=float), _Estimate(vz), _Corridor(), cfg)
    return float(normalized[2] * cfg.max_acceleration[2])


def test_the_active_contract_is_the_brake_rung():
    assert REFERENCE_SCHEMA == "spatial-reference/12" == REFERENCE_SCHEMAS[-1]
    for schema in ("spatial-reference/9", "spatial-reference/10",
                   "spatial-reference/11",
                   "spatial-reference/12"):
        cfg = _cfg(schema)
        assert cfg.terminal_descent_speed_factor == pytest.approx(0.8)
        assert cfg.terminal_descent_brake_one_step
    for schema in REFERENCE_SCHEMAS[:-4] + ("spatial-causal-rgat/5", "spatial-causal-rgat/10"):
        cfg = _cfg(schema)
        assert cfg.terminal_descent_speed_factor == pytest.approx(1.5), schema
        assert not cfg.terminal_descent_brake_one_step, schema


def test_the_sink_target_is_the_one_the_reward_already_asks_for():
    """environment.py's readiness term wants desired_z = -0.8 * touchdown_z_speed."""
    import inspect
    from ontology_rgat.spatial import environment
    source = inspect.getsource(environment.Evaluator.terms)
    assert "0.8 * cfg.touchdown_z_speed" in source
    assert _cfg("spatial-reference/9").terminal_descent_speed_factor == pytest.approx(0.8)


def test_a_limit_speed_descent_is_braked_hard_on_nine_and_not_at_all_on_eight():
    """0.34 m/s is the median unsafe contact on /8. Inside the corridor /8 lets
    it through (its brake starts at 0.45); /9 removes the 0.10 m/s excess over
    the 0.24 target within one 0.1 s step: +1.0 m/s^2, against the policy's
    requested -0.6."""
    assert _brake("spatial-reference/8", -0.34) == pytest.approx(-0.6)
    assert _brake("spatial-reference/9", -0.34) == pytest.approx((0.34 - 0.24) / 0.1)
    # Below the target the policy's command passes untouched on both rungs.
    assert _brake("spatial-reference/8", -0.20) == pytest.approx(-0.6)
    assert _brake("spatial-reference/9", -0.20) == pytest.approx(-0.6)
    # /8's own brake above 0.45 is the delay-paced reference law, unchanged.
    cfg8 = _cfg("spatial-reference/8")
    delay = THRUST_TAU + 2 / cfg8.attitude_omega + cfg8.actuation_delay_s
    assert _brake("spatial-reference/8", -0.52) == pytest.approx((0.52 - 0.45) / delay)
    # /9 is bounded by the vertical authority, never beyond it.
    assert _brake("spatial-reference/9", -1.0) == pytest.approx(cfg8.max_acceleration[2])


def test_the_brake_rung_differs_from_eight_in_nothing_but_the_brake():
    eight, nine = _cfg("spatial-reference/8"), _cfg("spatial-reference/9")
    assert eight.packet_fields == nine.packet_fields
    assert eight.registry_hash == nine.registry_hash
    assert _ontology(eight).node_names == _ontology(nine).node_names
    assert _graph_planes(eight) == _graph_planes(nine)
    for capability in ("direct_acceleration", "reference_tracking", "matched_disturbances",
                       "reference_context", "reference_supervisor", "camera_transport_delay",
                       "vertical_inhibit_holds", "actuation_delay_s", "attitude_omega",
                       "attitude_damping", "landing_gear_extension_m", "easy_start_lift_m",
                       "optical_realism", "contact_verdict_by_speed", "touchdown_height",
                       "touchdown_xy_speed", "touchdown_z_speed", "touchdown_tilt",
                       "touchdown_rate", "isaac_profile_sha256"):
        assert getattr(eight, capability) == getattr(nine, capability), capability
    assert eight.terminal_descent_speed_factor != nine.terminal_descent_speed_factor
    assert eight.terminal_descent_brake_one_step != nine.terminal_descent_brake_one_step
    assert eight.signature["packet_registry"] == nine.signature["packet_registry"]
    assert eight.signature["config_sha256"] != nine.signature["config_sha256"]


def test_adding_the_rung_did_not_move_eight():
    plan = (Path(__file__).resolve().parents[2] / "results" / "full_pipeline_20261006_v8_5seeds"
            / "low_sigma_kl" / "plan.json")
    if not plan.exists():
        pytest.skip("the /8 five-seed run is not on this machine")
    recorded = json.loads(plan.read_text())["signature"]["config_sha256"]
    assert _cfg("spatial-reference/8").signature["config_sha256"] == recorded


def test_the_gate_rung_differs_from_nine_in_nothing_but_the_gate_and_commit():
    """/10: corridor gate floored at 0.2 m and commit held 3 s (no legs)."""
    nine, ten = _cfg("spatial-reference/9"), _cfg("spatial-reference/10")
    assert (nine.terminal_gate_width_floor_m, nine.terminal_commit_window_s) == (0.0, 1.5)
    assert (ten.terminal_gate_width_floor_m, ten.terminal_commit_window_s) == (0.2, 3.0)
    for schema in REFERENCE_SCHEMAS[:-3]:
        cfg = _cfg(schema)
        assert (cfg.terminal_gate_width_floor_m, cfg.terminal_commit_window_s) == (0.0, 1.5), schema
    assert nine.packet_fields == ten.packet_fields and nine.registry_hash == ten.registry_hash
    assert _graph_planes(nine) == _graph_planes(ten)
    for capability in ("terminal_descent_speed_factor", "terminal_descent_brake_one_step",
                       "landing_gear_extension_m", "easy_start_lift_m", "contact_verdict_by_speed",
                       "attitude_omega", "attitude_damping", "actuation_delay_s",
                       "optical_realism", "touchdown_height", "isaac_profile_sha256"):
        assert getattr(nine, capability) == getattr(ten, capability), capability
    assert nine.signature["config_sha256"] != ten.signature["config_sha256"]


def test_the_gate_opens_low_and_off_centre_on_ten_and_not_on_nine():
    """The /9 Isaac failure: body 0.24 m, 0.12 m off-centre, fresh track, settled."""
    import math
    from types import SimpleNamespace
    from ontology_rgat.spatial.core import Estimator  # noqa: F401  (type only)
    def status(schema):
        cfg = _cfg(schema)
        own = SimpleNamespace(quaternion=np.array([1.0, 0, 0, 0]), angular_rate=np.zeros(3),
                              own_velocity=np.array([0.0, 0.0, -0.03]), own_position=np.zeros(3))
        est = SimpleNamespace(own=own, initialized=True, std=0.03, velocity_std=0.05, age=0.1,
                              r=np.array([0.12, 0.0, 0.24]), rv=np.zeros(3), last_t=10.0)
        return ReferenceSpatialSupervisor().status(est, cfg)
    assert "terminal_descent_corridor" not in status("spatial-reference/9").reasons
    assert "terminal_descent_corridor" in status("spatial-reference/10").reasons


def test_adding_the_rung_did_not_move_nine():
    plan = (Path(__file__).resolve().parents[2] / "results" / "full_pipeline_20261007_v9_5seeds"
            / "low_sigma_kl" / "plan.json")
    if not plan.exists():
        pytest.skip("the /9 five-seed run is not on this machine")
    recorded = json.loads(plan.read_text())["signature"]["config_sha256"]
    assert _cfg("spatial-reference/9").signature["config_sha256"] == recorded
