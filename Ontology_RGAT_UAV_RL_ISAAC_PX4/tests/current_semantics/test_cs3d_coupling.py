"""G3 local coupling checks on the spatial-reference/12 local plant (pilot seeds only).

The full six-check report runs once per module on every registered pilot
seed; the remaining tests pin the registration (seeds, constants, hashes)
and that a failing check is reported with its numbers, not raised.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import yaml

from ontology_rgat.current_semantics import coupling
from ontology_rgat.current_semantics.backend import CALIBRATION_GAPS, FLIGHT_SCHEMA
from ontology_rgat.current_semantics.contracts import (AUDITED_SEED_RANGES, DEFAULT_CONFIG_PATH,
                                                       contract_manifest, load_config)
from ontology_rgat.spatial.core import SpatialConfig
from ontology_rgat.spatial.environment import deployed_board


CONFIG = load_config()


@pytest.fixture(scope="module")
def report():
    return coupling.run_local_checks(CONFIG)


def test_report_is_strict_json_with_every_check_and_its_hashes(report):
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert set(report["checks"]) == set(coupling.CHECKS)
    assert report["seeds"] == list(CONFIG.g3.pilot_seeds)
    assert report["scientific_use"] is False and report["difficulty"] == 1.0
    assert report["backend"] == "spatial_local_v12" and "not run" in report["isaac_half"]
    manifest = contract_manifest(CONFIG)
    assert report["hashes"]["g3_hash"] == manifest["g3_hash"]
    assert report["hashes"]["algorithm_hash"] == manifest["algorithm_hash"]
    assert set(report["calibration_gaps"]) == set(CALIBRATION_GAPS)


def test_every_local_check_passes_and_the_blind_zone_is_report_only(report):
    verdicts = {name: check["passed"] for name, check in report["checks"].items()}
    assert verdicts == {"axis_coupling": True, "fov_exit": True, "calibration_identity": True,
                        "low_altitude_blind_zone": None, "quality_channels": True,
                        "observation_purity": True}
    assert report["local_checks_passed"] is True


def test_axis_commands_reach_attitude_and_both_image_axes_with_the_isaac_signs(report):
    check = report["checks"]["axis_coupling"]
    assert check["failures"] == [] and len(check["seeds"]) == len(CONFIG.g3.pilot_seeds)
    tilt = math.sin(math.atan(CONFIG.g3.local.axis_coupling.acceleration_m_s2 / 9.80665))
    for row in check["seeds"]:
        assert all(row["conditions"].values()), row["seed"]
        x, y = row["push_end"]["x"], row["push_end"]["y"]
        # +ax -> sin(pitch) > 0 and +ay -> sin(roll) < 0, at the commanded tilt.
        assert x["d_sin_pitch"] == pytest.approx(tilt, rel=0.1)
        assert y["d_sin_roll"] == pytest.approx(-tilt, rel=0.1)
        assert np.sign(x["truth_d_pitch_rad"]) == 1 and np.sign(y["truth_d_roll_rad"]) == -1
        assert row["adapter_constrained"] is False and row["paired_frames_equal"] is True
    assert check["summary"]["worst_superposition_residual"] < 0.05


def test_detection_stops_at_the_frame_edge_and_the_estimate_keeps_predicting(report):
    check = report["checks"]["fov_exit"]
    assert len(check["episodes"]) == 2 * len(CONFIG.g3.pilot_seeds)
    for row in check["episodes"]:
        assert all(row["conditions"].values()), (row["seed"], row["axis"])
        assert row["k_centre_exit_estimate"] <= row["k_last_update"] + 1
        assert row["truth_single_edge_exit"] is True
        ages = row["vision_age_after_loss_s"]
        assert np.allclose(np.diff(ages), 0.1) and ages[0] > 0.1
    # The projection outlives the detection: front_valid stays 1 with the margin < 0.
    assert check["summary"]["front_valid_with_negative_margin"] > 0
    # Recorded tag-vs-centre difference: the local plant drops tags before the
    # deployed board's pinhole rule does (CALIBRATION_GAPS edge_visibility_rule).
    assert check["summary"]["overshoot_minus_pinhole_m"]["max"] < 0.0
    assert check["summary"]["overshoot_at_last_capture_m"]["min"] > 0.0


def test_pinhole_board_prediction_is_the_trailing_corner_tag():
    board = deployed_board(SpatialConfig(schema=FLIGHT_SCHEMA))
    corner = max(abs(board[:, 0]))
    side = board[np.argmax(abs(board[:, 0])), 2]
    for direction in coupling.EXITS.values():
        assert coupling.board_trailing_extent_m(direction[1], board) == pytest.approx(
            corner - 2.0 * side / 3.0)


def test_local_cameras_are_identical_to_the_registered_calibration(report):
    check = report["checks"]["calibration_identity"]
    assert max(check["constant_abs_error"].values()) <= coupling.IDENTITY_TOLERANCE
    assert max(check["behaviour_max_abs_error"].values()) <= coupling.IDENTITY_TOLERANCE
    assert check["corners_compared"] > 0 and check["poses"] == 64
    assert check["local"]["vfov_rad"] == pytest.approx(2.0 * math.atan(240.0 / 320.0))


def test_blind_zone_measures_last_frame_and_contact_against_the_registered_isaac_numbers(report):
    check = report["checks"]["low_altitude_blind_zone"]
    touchdown = CALIBRATION_GAPS["touchdown_body_height"].numbers
    assert check["isaac"]["numbers"]["isaac_first_contact_body_height_m"] == list(
        touchdown["isaac_first_contact_body_height_m"])
    for row in check["episodes"]:
        assert row["measured"] is True and row["termination"]["reason"] == "truth_contact"
        assert row["body_height_at_contact_m"] == pytest.approx(
            CONFIG.local_plant.contact_height_m)
        assert row["body_height_at_last_frame_m"] > row["body_height_at_contact_m"]
        assert row["blind_interval_s"] > 0.0
        # The calibrated local contact lies inside Isaac's measured range.
        low, high = row["contact_height_minus_isaac_m"]
        assert low <= 0.0 <= high
        assert abs(row["contact_vertical_speed_m_s"]) <= CONFIG.contact.z_speed_m_s


def test_quality_channels_match_track_acceptance_and_capture_stamps(report):
    check = report["checks"]["quality_channels"]
    counts = check["counts"]
    for key in ("flag_mismatch", "estimate_mismatch", "stamp_order_violation",
                "capture_mismatch", "sentinel_mismatch"):
        assert counts.get(key, 0) == 0, key
    assert check["max_age_error_s"] <= CONFIG.g3.local.quality_channels.age_tolerance_s
    assert check["navigation"]["constant"] is True
    # The local reset hands over only after an accepted frame (handover_detection).
    assert counts.get("uninitialised", 0) == 0
    assert sum(check["frames_per_decision"].values()) + counts.get("uninitialised", 0) \
        == counts["decisions"]


def test_features_are_a_pure_function_of_each_recorded_vector(report):
    check = report["checks"]["observation_purity"]
    assert check["rebuild_mismatches"] == 0 and check["batch_equal"] is True
    assert check["decisions"] > 1000


def test_checks_are_deterministic_and_refuse_unregistered_seeds():
    seeds = CONFIG.g3.pilot_seeds[:1]
    first = coupling.check_axis_coupling(CONFIG, seeds)
    assert json.dumps(first, sort_keys=True) == json.dumps(
        coupling.check_axis_coupling(CONFIG, seeds), sort_keys=True)
    for foreign in ([4100], [CONFIG.g3.pilot_seeds[0], 12000], []):
        with pytest.raises(ValueError, match="g3_pilot_seeds"):
            coupling.check_axis_coupling(CONFIG, foreign)


def _variant_path(tmp_path, edit):
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    edit(data)
    path = tmp_path / "variant.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_a_failing_check_is_reported_with_its_numbers_not_raised(tmp_path):
    strict = load_config(_variant_path(tmp_path, lambda d: d["g3_local"]["axis_coupling"].update(
        cross_axis_ratio_max=1e-12)))
    result = coupling.check_axis_coupling(strict, strict.g3.pilot_seeds[:1])
    assert result["passed"] is False
    conditions = result["seeds"][0]["conditions"]
    assert conditions["x_pitch_positive"] and not conditions["x_roll_small"]
    assert result["seeds"][0]["push_end"]["x"]["d_sin_pitch"] > 0


def test_g3_registration_is_hashed_apart_from_the_algorithm(tmp_path):
    base = contract_manifest(CONFIG)
    assert base["g3"]["pilot_seeds"] == list(CONFIG.g3.pilot_seeds)
    variant = contract_manifest(load_config(_variant_path(
        tmp_path, lambda d: d["g3_local"]["fov_exit"].update(exit_speed_m_s=0.5))))
    assert variant["g3_hash"] != base["g3_hash"]
    for key in ("algorithm_hash", "action_hash", "feature_hash", "graph_hash",
                "calibration_hash", "observation_hash"):
        assert variant[key] == base[key], key


def test_pilot_seeds_lie_outside_every_audited_range():
    for seed in CONFIG.g3.pilot_seeds:
        assert not any(first <= seed <= last for _, first, last, _ in AUDITED_SEED_RANGES)


@pytest.mark.parametrize("edit,match", [
    (lambda d: d.update(g3_pilot_seeds=[990001, 4100]), "overlap audited"),
    (lambda d: d.update(g3_pilot_seeds=[990001, 990001]), "distinct"),
    (lambda d: d.update(g3_pilot_seeds=[990001.0]), "positive integer"),
    (lambda d: d.update(g3_pilot_seeds=[]), "non-empty"),
    (lambda d: d.pop("g3_local"), "missing keys"),
    (lambda d: d["g3_local"].update(extra={}), "unknown keys"),
    (lambda d: d["g3_local"]["fov_exit"].pop("overshoot_tolerance_m"), "missing keys"),
    (lambda d: d["g3_local"]["axis_coupling"].update(push_steps=5.0), "positive integer"),
    (lambda d: d["g3_local"]["low_altitude_blind_zone"].update(sink_speed_m_s=0), "positive"),
    (lambda d: d["g3_local"]["quality_channels"].update(age_tolerance_s="1e-9"), "number"),
])
def test_g3_config_is_strict(tmp_path, edit, match):
    with pytest.raises(ValueError, match=match):
        load_config(_variant_path(tmp_path, edit))


def test_isaac_half_logic_runs_on_any_backend_and_refuses_foreign_seeds():
    # The Isaac half is exercised here on the LOCAL plant only; it never builds a stack.
    from ontology_rgat.current_semantics.backend import make_local_backend
    local = make_local_backend(CONFIG)
    try:
        report = coupling.run_isaac_checks(CONFIG, local)
        with pytest.raises(ValueError, match="g3_isaac.seeds"):
            coupling.run_isaac_checks(CONFIG, local, seeds=(990003,))
    finally:
        local.close()
    json.dumps(report, allow_nan=False)
    assert report["backend"] == "spatial_local_v12" and report["isaac_checks_passed"] is True
    assert set(report["checks"]) == {"axis_signs", "fov_exit", "low_altitude_blind_zone",
                                     "quality_channels", "observation_purity", "frame_rate",
                                     "termination_and_cleanup"}
    signs = report["checks"]["axis_signs"]
    assert signs["corr_sin_pitch_vs_ax"] >= CONFIG.g3.isaac.axis_sign_min_abs_correlation
    assert signs["corr_sin_roll_vs_ay"] <= -CONFIG.g3.isaac.axis_sign_min_abs_correlation
    assert report["checks"]["frame_rate"]["passed"] is None
    assert report["hashes"]["task_hash"] == contract_manifest(CONFIG)["task_hash"]
