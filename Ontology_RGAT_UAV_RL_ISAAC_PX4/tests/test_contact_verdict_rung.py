"""``spatial-reference/7``: a pad contact is judged by its impact speed.

The five-seed /6 Isaac flight (results/full_pipeline_20261006_v6_5seeds, 30
episodes) ended 23 SUCCESS and six contacts that were not. Read from the
recorded truth at the contact step, four of the six were vehicles sinking onto
their legs from a low hover while the supervisor held the vertical rate
(``vertical_stopping_margin``): |vz| 0.01-0.07 m/s, horizontal speed <= 0.05
m/s, two of them inside every touchdown limit (scored UNAUTHORIZED_CONTACT
because descent was inhibited) and two just outside the attitude band (body
rate 10.47 deg/s; tilt 5.27 deg at 12.51 deg/s, scored UNSAFE_CONTACT). The
other two touched inside the terminal-descent corridor at vz -0.62 and -0.38
m/s against a 0.30 m/s limit. The user ruled on 2026-10-06 that the four are
landings and the two are not. The only one-line rule that draws that line is
the impact speed, so /7 = /6 plus exactly that law; the limit VALUES, the
plant, the packet, the graph, the supervisor and the terminal table do not
move, and the recorded contacts below are the measurement the rule is held to.
"""
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ontology_rgat.spatial.core import REFERENCE_SCHEMA, REFERENCE_SCHEMAS, SpatialConfig
from ontology_rgat.spatial.environment import Evaluator, Truth
from ontology_rgat.spatial.training import _graph_planes, _ontology


def _cfg(schema):
    from ontology_rgat.spatial.runtime_contract import deployment_profile
    return replace(SpatialConfig(), schema=schema,
                   isaac_profile_sha256=deployment_profile(schema)["sha256"])


class _Safety:
    def __init__(self, inhibited=False, abort=False):
        self.inhibited, self.abort = inhibited, abort


def _contact(*, xy_speed, vz, tilt_deg, rate_deg, lateral=0.2):
    return Truth(relative_position=np.array([lateral, 0.0, 0.30]),
                 relative_velocity=np.array([xy_speed, 0.0, vz]),
                 roll_pitch=np.array([math.radians(tilt_deg), 0.0]),
                 angular_rate=np.array([math.radians(rate_deg), 0.0, 0.0]),
                 contact=True)


def _verdict(schema, truth, safety):
    cfg = _cfg(schema)
    evaluator = Evaluator(cfg)
    evaluator.reset(truth)
    _, status, _ = evaluator.evaluate(
        truth, elapsed=8.0, dt=cfg.dt, action=np.zeros(3), safety=safety,
        abort_elapsed=0.0, bearings=np.zeros(2), visible=True)
    return status


#: The six non-landing contacts of the /6 flight, as recorded in the per-run
#: traces (``truth_relative_velocity``, ``truth_roll_pitch``,
#: ``truth_angular_rate``, ``safety_reasons``), and how /6 scored them.
RECORDED = (
    # arm, seed, |v_xy|, vz, tilt deg, |w_xy| deg/s, inhibited, /6 status, user ruling
    ("ppo_semantic_flat", 828, 0.04, -0.03, 2.10, 1.99, True, "UNAUTHORIZED_CONTACT", "SUCCESS"),
    ("ppo_semantic_flat", 829, 0.02, -0.01, 1.96, 1.48, True, "UNAUTHORIZED_CONTACT", "SUCCESS"),
    ("ppo_ontology_rgat", 832, 0.05, -0.07, 2.10, 10.47, True, "UNSAFE_CONTACT", "SUCCESS"),
    ("ppo_semantic_flat", 829, 0.03, -0.03, 5.27, 12.51, True, "UNSAFE_CONTACT", "SUCCESS"),
    ("ppo_vector_canonical", 832, 0.04, -0.62, 2.65, 8.46, False, "UNSAFE_CONTACT", "UNSAFE_CONTACT"),
    ("ppo_vector_canonical", 832, 0.10, -0.38, 2.66, 2.82, False, "UNSAFE_CONTACT", "UNSAFE_CONTACT"),
)


def test_the_active_contract_is_the_verdict_rung():
    assert REFERENCE_SCHEMA == "spatial-reference/7" == REFERENCE_SCHEMAS[-1]
    assert _cfg("spatial-reference/7").contact_verdict_by_speed
    for schema in REFERENCE_SCHEMAS[:-1] + ("spatial-causal-rgat/5", "spatial-causal-rgat/10"):
        assert not _cfg(schema).contact_verdict_by_speed, schema


def test_the_six_recorded_contacts_split_four_to_two_and_only_under_seven():
    for arm, seed, xy_speed, vz, tilt, rate, inhibited, six, ruling in RECORDED:
        truth = _contact(xy_speed=xy_speed, vz=vz, tilt_deg=tilt, rate_deg=rate)
        safety = _Safety(inhibited=inhibited)
        assert _verdict("spatial-reference/6", truth, safety) == six, (arm, seed)
        assert _verdict("spatial-reference/7", truth, safety) == ruling, (arm, seed)
    rulings = [row[-1] for row in RECORDED]
    assert rulings.count("SUCCESS") == 4 and rulings.count("UNSAFE_CONTACT") == 2


def test_the_speed_rule_keeps_the_pad_the_envelope_and_the_abort():
    gentle = dict(xy_speed=0.05, vz=-0.05, tilt_deg=8.0, rate_deg=15.0)
    # Inhibited, tilted and rotating past the touchdown band, but sinking at
    # 0.05 m/s: a landing on /7 and a crash on /6.
    assert _verdict("spatial-reference/7", _contact(**gentle), _Safety(inhibited=True)) == "SUCCESS"
    assert _verdict("spatial-reference/6", _contact(**gentle), _Safety(inhibited=True)) == "UNSAFE_CONTACT"
    # The speed limits are the criterion, at their unchanged values.
    cfg = _cfg("spatial-reference/7")
    assert (cfg.touchdown_xy_speed, cfg.touchdown_z_speed) == (0.35, 0.3)
    assert (cfg.touchdown_tilt, cfg.touchdown_rate) == (math.radians(5), math.radians(10))
    assert _verdict("spatial-reference/7", _contact(xy_speed=0.0, vz=-0.31, tilt_deg=0, rate_deg=0),
                    _Safety()) == "UNSAFE_CONTACT"
    assert _verdict("spatial-reference/7", _contact(xy_speed=0.36, vz=0.0, tilt_deg=0, rate_deg=0),
                    _Safety()) == "UNSAFE_CONTACT"
    # Off the pad, past the 21 deg hard envelope, or during a latched abort,
    # a slow contact is still not a landing.
    assert _verdict("spatial-reference/7", _contact(xy_speed=0.0, vz=-0.05, tilt_deg=0, rate_deg=0,
                                                    lateral=0.6), _Safety()) == "MISSED_PAD_CONTACT"
    assert _verdict("spatial-reference/7", _contact(xy_speed=0.0, vz=-0.05, tilt_deg=22.0, rate_deg=0),
                    _Safety()) == "SAFETY_ENVELOPE_VIOLATION"
    assert _verdict("spatial-reference/7", _contact(xy_speed=0.0, vz=-0.05, tilt_deg=0, rate_deg=0),
                    _Safety(inhibited=True, abort=True)) == "UNAUTHORIZED_CONTACT"
    # An authorized, in-band contact lands on both rungs.
    assert _verdict("spatial-reference/6", _contact(xy_speed=0.1, vz=-0.2, tilt_deg=2, rate_deg=3),
                    _Safety()) == "SUCCESS"
    assert _verdict("spatial-reference/7", _contact(xy_speed=0.1, vz=-0.2, tilt_deg=2, rate_deg=3),
                    _Safety()) == "SUCCESS"


def test_the_verdict_rung_differs_from_six_in_nothing_but_the_contact_rule():
    six, seven = _cfg("spatial-reference/6"), _cfg("spatial-reference/7")
    assert six.packet_fields == seven.packet_fields
    assert six.registry_hash == seven.registry_hash
    assert _ontology(six).node_names == _ontology(seven).node_names
    assert _graph_planes(six) == _graph_planes(seven)
    for capability in ("direct_acceleration", "reference_tracking", "matched_disturbances",
                       "reference_context", "reference_supervisor", "camera_transport_delay",
                       "vertical_inhibit_holds", "actuation_delay_s", "attitude_omega",
                       "attitude_damping", "landing_gear_extension_m", "optical_realism",
                       "touchdown_height", "touchdown_xy_speed", "touchdown_z_speed",
                       "touchdown_tilt", "touchdown_rate", "isaac_profile_sha256"):
        assert getattr(six, capability) == getattr(seven, capability), capability
    assert six.contact_verdict_by_speed != seven.contact_verdict_by_speed
    assert six.signature["packet_registry"] == seven.signature["packet_registry"]
    assert six.signature["config_sha256"] != seven.signature["config_sha256"]


def test_adding_the_rung_did_not_move_six():
    """The /6 clones and cells (results/full_pipeline_20261006_v6_5seeds) must
    keep loading under their own schema."""
    plan = (Path(__file__).resolve().parents[2] / "results" / "full_pipeline_20261006_v6_5seeds"
            / "low_sigma_kl" / "plan.json")
    if not plan.exists():
        pytest.skip("the /6 five-seed run is not on this machine")
    recorded = json.loads(plan.read_text())["signature"]["config_sha256"]
    assert _cfg("spatial-reference/6").signature["config_sha256"] == recorded
