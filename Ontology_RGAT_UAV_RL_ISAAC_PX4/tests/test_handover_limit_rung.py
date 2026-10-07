"""``spatial-reference/11``: the lateral command is capped and slew-limited for
the first 2 s after handover.

All seven Isaac SAFETY_ENVELOPE_VIOLATIONs recorded on /2-/10 happened 0.9-1.3 s
after handover, each starting with a near-full lateral command issued on a
not-yet-trusted track; five then reversed it in one decision (-2.2 -> +2.3
m/s^2) and PX4 overshot past 21 deg. On the 15 /10 checkpoints over 48
held-out seeds the peak tilt in the first 2 s fell from 21.0 deg (no margin)
to 14.9 deg with a 2.0 m/s^2 cap and a 1.5 m/s^2-per-step slew limit.
"""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ontology_rgat.spatial.core import REFERENCE_SCHEMA, REFERENCE_SCHEMAS, SpatialConfig
from ontology_rgat.spatial.safety import ReferenceSpatialSupervisor


def _cfg(schema):
    from ontology_rgat.spatial.runtime_contract import deployment_profile
    return replace(SpatialConfig(), schema=schema,
                   isaac_profile_sha256=deployment_profile(schema)["sha256"])


def _est(t):
    own = SimpleNamespace(quaternion=np.array([1.0, 0, 0, 0]), angular_rate=np.zeros(3),
                          own_velocity=np.zeros(3), own_position=np.array([0.0, 0.0, 2.0]))
    return SimpleNamespace(own=own, initialized=True, std=0.03, velocity_std=0.05, age=0.1,
                           r=np.array([0.3, 0.0, 2.0]), rv=np.zeros(3), last_t=t)


def _fly(schema, commands, t0=5.0):
    """The recorded Isaac pattern: hard one way, then a one-step reversal."""
    cfg = _cfg(schema)
    supervisor = ReferenceSpatialSupervisor()
    out = []
    for k, command in enumerate(commands):
        est = _est(t0 + 0.1 * k)
        safety = supervisor.status(est, cfg)
        normalized = supervisor.action(np.asarray(command, dtype=float), est, safety, cfg)
        out.append(np.asarray(normalized)[:2] * np.asarray(cfg.max_acceleration[:2]))
    return np.asarray(out)


#: Normalized policy commands of the /10 Isaac violation (vector seed828/12000).
REVERSAL = [(0.0, -0.58, 0.0), (0.08, -0.58, 0.0), (0.19, -0.5, 0.0),
            (0.14, 0.66, 0.0), (0.02, 0.88, 0.0), (-0.06, 0.91, 0.0)]


def test_the_active_contract_is_the_handover_rung():
    assert REFERENCE_SCHEMA == "spatial-reference/12" == REFERENCE_SCHEMAS[-1]
    eleven = _cfg("spatial-reference/11")
    assert (eleven.handover_lateral_cap_m_s2, eleven.handover_window_s,
            eleven.handover_lateral_slew_m_s2) == (2.0, 2.0, 1.5)
    for schema in REFERENCE_SCHEMAS[:-2]:
        assert _cfg(schema).handover_lateral_cap_m_s2 is None, schema


def test_the_recorded_reversal_is_capped_and_slewed_on_eleven_only():
    ten = _fly("spatial-reference/10", REVERSAL)
    eleven = _fly("spatial-reference/11", REVERSAL)
    assert np.max(np.linalg.norm(ten, axis=1)) > 2.2
    assert np.max(np.linalg.norm(eleven, axis=1)) <= 2.0 + 1e-9
    assert np.max(np.linalg.norm(np.diff(ten, axis=0), axis=1)) > 2.5
    assert np.max(np.linalg.norm(np.diff(eleven, axis=0), axis=1)) <= 1.5 + 1e-9


def test_the_limit_ends_with_the_window():
    late = _fly("spatial-reference/11", [(0.0, 0.0, 0.0)] * 21 + [(0.0, 0.95, 0.0)])
    assert np.linalg.norm(late[-1]) == pytest.approx(0.95 * 2.5)


def test_the_handover_rung_differs_from_ten_in_nothing_but_the_limit():
    ten, eleven = _cfg("spatial-reference/10"), _cfg("spatial-reference/11")
    assert ten.packet_fields == eleven.packet_fields and ten.registry_hash == eleven.registry_hash
    for capability in ("terminal_descent_speed_factor", "terminal_descent_brake_one_step",
                       "terminal_gate_width_floor_m", "terminal_commit_window_s",
                       "landing_gear_extension_m", "easy_start_lift_m", "contact_verdict_by_speed",
                       "attitude_omega", "attitude_damping", "actuation_delay_s",
                       "optical_realism", "touchdown_height", "isaac_profile_sha256"):
        assert getattr(ten, capability) == getattr(eleven, capability), capability
    assert ten.signature["config_sha256"] != eleven.signature["config_sha256"]


def test_adding_the_rung_did_not_move_ten():
    plan = (Path(__file__).resolve().parents[2] / "results" / "full_pipeline_20261007_v10_5seeds"
            / "low_sigma_kl" / "plan.json")
    if not plan.exists():
        pytest.skip("the /10 five-seed run is not on this machine")
    recorded = json.loads(plan.read_text())["signature"]["config_sha256"]
    assert _cfg("spatial-reference/10").signature["config_sha256"] == recorded


def test_the_thrust_floor_rung_keeps_net_vertical_acceleration_non_negative_after_handover():
    """/12: across 209 recorded Isaac episodes, a first-1.3 s mean applied az
    below -0.6 m/s^2 ended in an envelope violation 5 of 5 times, az >= 0
    never (0 of 53). The /11 flight's two violations carried -1.8 m/s^2."""
    descend = [(-0.45, -0.9, -0.9)] * 6
    for schema, floor in (("spatial-reference/11", None), ("spatial-reference/12", 0.0)):
        cfg = _cfg(schema)
        assert cfg.handover_vertical_floor_m_s2 == floor
        supervisor = ReferenceSpatialSupervisor()
        vertical = []
        for k, command in enumerate(descend):
            est = _est(5.0 + 0.1 * k)
            safety = supervisor.status(est, cfg)
            vertical.append(supervisor.action(np.asarray(command), est, safety, cfg)[2]
                            * cfg.max_acceleration[2])
        if floor is None:
            assert min(vertical) < -1.0
        else:
            assert min(vertical) >= 0.0
    for schema in REFERENCE_SCHEMAS[:-1]:
        assert _cfg(schema).handover_vertical_floor_m_s2 is None, schema
    eleven, twelve = _cfg("spatial-reference/11"), _cfg("spatial-reference/12")
    assert (eleven.handover_lateral_cap_m_s2, eleven.handover_lateral_slew_m_s2) == \
           (twelve.handover_lateral_cap_m_s2, twelve.handover_lateral_slew_m_s2)
    assert eleven.signature["config_sha256"] != twelve.signature["config_sha256"]


def test_adding_the_rung_did_not_move_eleven():
    plan = (Path(__file__).resolve().parents[2] / "results" / "full_pipeline_20261007_v11_5seeds"
            / "low_sigma_kl" / "plan.json")
    if not plan.exists():
        pytest.skip("the /11 five-seed run is not on this machine")
    recorded = json.loads(plan.read_text())["signature"]["config_sha256"]
    assert _cfg("spatial-reference/11").signature["config_sha256"] == recorded
