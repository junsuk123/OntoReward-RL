from pathlib import Path

from ontology_rgat.config import default_config
from ontology_rgat_px4.config import load_gateway_config
from ontology_rgat_px4.ros2_gateway import parallel_gateway_config


ROOT = Path(__file__).resolve().parents[1]


def test_learner_pair_ports_are_disjoint_and_legacy_is_unchanged():
    from run_three_pipeline import _pair_live_config

    base = default_config()
    legacy = _pair_live_config(base, 0, 1)
    assert legacy is base

    pairs = [_pair_live_config(base, index, 3) for index in range(3)]
    assert [pair.external.gateway_port for pair in pairs] == [14650, 14652, 14654]
    assert [pair.external.local_port for pair in pairs] == [14651, 14653, 14655]
    assert [pair.external.pair_index for pair in pairs] == [0, 1, 2]
    assert len({pair.external.local_port for pair in pairs}) == 3
    assert all(pair.external.entry_timeout == 1.25 * base.external.entry_timeout
               for pair in pairs)


def test_gateway_pair_identity_matches_px4_multi_vehicle_contract():
    base = load_gateway_config(ROOT / "config/system.yaml")
    pairs = [parallel_gateway_config(base, index, 3) for index in range(3)]

    assert [pair.namespace for pair in pairs] == [
        "/fmu", "/px4_1/fmu", "/px4_2/fmu"]
    assert [pair.target_system for pair in pairs] == [1, 2, 3]
    assert [pair.gateway_port for pair in pairs] == [14650, 14652, 14654]
    assert [pair.topic_root for pair in pairs] == [
        "/landing_pair_0", "/landing_pair_1", "/landing_pair_2"]


def test_gateway_single_pair_keeps_original_topics_and_port():
    base = load_gateway_config(ROOT / "config/system.yaml")
    resolved = parallel_gateway_config(base, 0, 1)
    assert resolved.topic_root == ""
    assert resolved.namespace == "/fmu"
    assert resolved.gateway_port == 14650


def test_parallel_route_phases_and_marker_ids_fit_declared_dictionary():
    from config_loader import load_config

    config = load_config(ROOT / "config/seminar-fast-system.yaml")
    parallel = config["parallel"]
    assert len(parallel["pair_offsets_enu_m"]) >= 3
    assert all(tuple(offset) == (0.0, 0.0, 0.0)
               for offset in parallel["pair_offsets_enu_m"][:3])
    phases = parallel["route_phase_fractions"][:3]
    assert len(phases) == 3
    assert len(set(float(value) for value in phases)) == 3
    assert all(0.0 <= float(value) < 1.0 for value in phases)
    marker_ids = [int(marker["id"]) for marker in config["vision"]["board"]]
    expanded = {
        marker_id + pair * int(parallel["marker_id_stride"])
        for pair in range(3) for marker_id in marker_ids
    }
    assert len(expanded) == 3 * len(marker_ids)
    assert min(expanded) >= 0 and max(expanded) < 250


def test_keypoint_calibration_uses_the_board_rendered_for_each_parallel_pair():
    from config_loader import load_config
    from run_three_pipeline import _calibration_system_for_pair

    system = load_config(ROOT / "config/shin2026-system.yaml")
    original_dictionary = system["vision"]["dictionary"]
    original_ids = [int(marker["id"]) for marker in system["vision"]["board"]]
    pair0 = _calibration_system_for_pair(system, 0, 2)
    pair1 = _calibration_system_for_pair(system, 1, 2)
    stride = int(system["parallel"]["marker_id_stride"])

    assert pair0["vision"]["dictionary"] == system["parallel"]["marker_dictionary"]
    assert [int(marker["id"]) for marker in pair0["vision"]["board"]] == original_ids
    assert [int(marker["id"]) for marker in pair1["vision"]["board"]] == [
        marker_id + stride for marker_id in original_ids]
    assert system["vision"]["dictionary"] == original_dictionary
    assert [int(marker["id"]) for marker in system["vision"]["board"]] == original_ids


def test_bare_repository_launcher_selects_two_pair_operator_profile():
    launcher = (ROOT.parent / "run.sh").read_text(encoding="utf-8")
    assert "if [[ $# -eq 0 ]]" in launcher
    assert "seminar_fast=true" in launcher
    assert "--parallel-pairs 2" in launcher
    assert "--pipelines shin_se_fixed shin_se_onto_rgat_fov" in launcher
    assert "--stay-open" in launcher
