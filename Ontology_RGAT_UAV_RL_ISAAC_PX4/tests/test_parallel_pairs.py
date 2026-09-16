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


def test_parallel_route_phases_separate_the_two_pairs_without_marker_ids():
    from config_loader import load_config

    config = load_config(ROOT / "config/seminar-fast-system.yaml")
    parallel = config["parallel"]
    assert len(parallel["pair_offsets_enu_m"]) == 2
    assert all(tuple(offset) == (0.0, 0.0, 0.0)
               for offset in parallel["pair_offsets_enu_m"])
    phases = parallel["route_phase_fractions"]
    assert len(phases) == 2
    assert len(set(float(value) for value in phases)) == 2
    assert all(0.0 <= float(value) < 1.0 for value in phases)
    # Pairs are separated along the route, not by marker identity: the primary
    # profile has no dictionary and no board to stride.
    assert parallel["marker_dictionary"] is None
    assert parallel["marker_id_stride"] is None
    assert config["vision"]["mode"] == "keypoint_fiducial"
    assert not config["vision"].get("board")


def test_keypoint_calibration_shares_one_target_across_parallel_pairs():
    from config_loader import load_config
    from run_three_pipeline import _calibration_system_for_pair

    system = load_config(ROOT / "config/shin2026-system.yaml")
    landing_pad = dict(system["vision"]["landing_pad"])

    # Every pair's deck carries the identical six-keypoint target, so nothing
    # is specialised and nothing in the source profile is mutated.
    for pair_index in range(2):
        resolved = _calibration_system_for_pair(system, pair_index, 2)
        assert resolved["vision"]["landing_pad"] == landing_pad
        assert not resolved["vision"].get("dictionary")
        assert not resolved["vision"].get("board")
    assert system["vision"]["landing_pad"] == landing_pad


def test_legacy_aruco_profile_still_separates_parallel_pairs_by_marker_id():
    from config_loader import load_config
    from run_three_pipeline import _calibration_system_for_pair

    system = load_config(ROOT / "config/system.yaml")
    original_ids = [int(marker["id"]) for marker in system["vision"]["board"]]
    stride = int(system["parallel"]["marker_id_stride"])
    pair0 = _calibration_system_for_pair(system, 0, 2)
    pair1 = _calibration_system_for_pair(system, 1, 2)

    assert pair0["vision"]["dictionary"] == system["parallel"]["marker_dictionary"]
    assert [int(marker["id"]) for marker in pair0["vision"]["board"]] == original_ids
    assert [int(marker["id"]) for marker in pair1["vision"]["board"]] == [
        marker_id + stride for marker_id in original_ids]
    assert [int(marker["id"]) for marker in system["vision"]["board"]] == original_ids


def test_bare_repository_launcher_selects_two_pair_operator_profile():
    launcher = (ROOT.parent / "run.sh").read_text(encoding="utf-8")
    assert "if [[ $# -eq 0 ]]" in launcher
    assert "seminar_fast=true" in launcher
    assert "--parallel-pairs 2" in launcher
    assert "--pipelines shin_se_fixed shin_se_onto_rgat_recovery" in launcher
    assert "--stay-open" in launcher


def test_a_headless_flight_still_opens_the_operator_rviz_view():
    # ``--headless`` is Isaac Sim's own window. RViz 2 is a separate process
    # reading ROS topics the run publishes in either mode, so a headless
    # seminar run must still get its live view; only ``--no-rviz`` or a
    # missing DISPLAY turns it off.
    for launcher in ("run_three_pipeline.py", "run_shin2026_pipeline.py"):
        source = (ROOT / "python" / launcher).read_text(encoding="utf-8")
        start = source.index("_start_rviz(\n")
        call = source[start:source.index(")", start)]
        assert "no_rviz" in call, f"{launcher} must still honour --no-rviz"
        assert "headless" not in call, (
            f"{launcher} suppresses RViz when Isaac is headless")


def test_rviz_is_still_skipped_without_a_display(monkeypatch, capsys):
    import sys
    sys.path.insert(0, str(ROOT / "python"))
    from run_shin2026_pipeline import _start_rviz

    monkeypatch.delenv("DISPLAY", raising=False)
    process, stream = _start_rviz(True, parallel_pairs=2)
    assert (process, stream) == (None, None)
    assert "DISPLAY is unset" in capsys.readouterr().out
