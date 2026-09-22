from pathlib import Path

import pytest

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


def test_bare_repository_launcher_runs_the_full_reference_experiment():
    """``./run.sh`` alone must be the experiment, not the preview of it.

    The zero-argument path used to select ``--seminar-fast``, the profile
    marked ``publication_claim_allowed: false``. The command that reads as
    "run the experiment" therefore ran the one whose results may not be
    published, and a budget fix to the experiment config landed on a file
    that path never opens.
    """
    launcher = (ROOT.parent / "run.sh").read_text(encoding="utf-8")

    # Nothing may turn the preview on by itself; it is an explicit flag.
    assert "if [[ $# -eq 0 ]]" not in launcher
    assert "seminar_fast=true" in launcher, "the preview must stay reachable"
    preview = launcher[launcher.index("if [[ \"$seminar_fast\" == true ]]"):]
    assert "--seminar-fast) seminar_fast=true" in launcher
    # The preview profile itself is unchanged, and still opt-in only.
    assert "--parallel-pairs 2" in preview
    assert "--pipelines shin_se_fixed shin_se_onto_rgat_recovery" in preview
    assert "--stay-open" in preview

    # With no arguments the launcher falls through to the primary two-pipeline
    # entry point at the mode the reference budget is declared for.
    assert 'launcher="$project_root/scripts/run_two_pipeline.sh"' in launcher
    assert 'arguments=(--mode full "${arguments[@]}")' in launcher


def test_the_bare_launcher_budget_is_the_one_the_experiment_config_declares():
    """``--mode full`` takes its budgets from the config, so they must be sane.

    The 2026-09-18 run flew a 40,000-episode-per-arm budget that this machine
    would have needed months to finish, against a measured ~20 episodes per
    hour per pair. Nothing in the launcher imposes a budget any more, so the
    config is the only place this can be got wrong.
    """
    from config_loader import load_config
    from conftest import default_experiment_config

    # The budget of the config the bare launcher opens, which is the whole
    # point of this test; naming a file here guarded whatever that file
    # happened to be after the default moved.
    config = load_config(str(default_experiment_config()))
    per_pair_hourly = 20.0
    arms = 2

    training = int(config["training"]["episodes_full"])
    evaluation = sum(int(count) for count in config["evaluation"].values())
    design = int(config["fov_risk_design"]["episodes_full"])

    # Arms train on their own pairs, so training is one arm's wall clock;
    # evaluation flies every seed twice across the same two pairs.
    days = (training / per_pair_hourly
            + evaluation * arms / (arms * per_pair_hourly)
            + design / per_pair_hourly) / 24.0
    assert days < 14.0, f"the declared budget is {days:.0f} days of flying"
    assert int(config["fov_risk_design"]["max_episodes_full"]) >= design


def test_rviz_gives_every_physical_pair_its_own_view_not_every_method():
    """Four spawned pairs are four RViz views, even with two arms.

    Keying the publishers by method collapsed four pairs to two: pairs 2 and 3
    then published their trails, markers and ``map -> landing_pad`` transforms
    into pair 0 and 1's namespaces, while the generated four-pair layout showed
    their own panels empty.
    """
    from types import SimpleNamespace

    from ontology_rgat.viz.rviz import RvizPublisher, RvizPublisherGroup

    class Recording(RvizPublisher):
        made: list = []

        @classmethod
        def create(cls, cfg, potential=None, node_name="ontology_rgat_viz",
                   pair_methods=None):
            if pair_methods:
                return super().create(cfg, potential=potential,
                                      node_name=node_name,
                                      pair_methods=pair_methods)
            stub = SimpleNamespace(
                namespace=str(cfg.viz.rviz.namespace),
                pad_frame=str(cfg.viz.rviz.pad_frame),
                body_frame=str(cfg.viz.rviz.body_frame),
                node_name=str(node_name), potential=None, closed=0,
                cleared=0, steps=[])
            stub.close = lambda: setattr(stub, "closed", stub.closed + 1)
            stub.clear_trails = lambda: setattr(stub, "cleared", stub.cleared + 1)
            stub.publish_benchmark_step = lambda **kwargs: stub.steps.append(kwargs)
            cls.made.append(stub)
            return stub

    methods = ["shin_se_fixed", "shin_se_fixed",
               "shin_se_onto_rgat_recovery", "shin_se_onto_rgat_recovery"]
    group = Recording.create(default_config(), pair_methods=methods)

    assert isinstance(group, RvizPublisherGroup)
    assert len(group.publishers_by_index) == 4
    assert [stub.namespace for stub in Recording.made] == [
        f"/landing_rl/pair_{index}" for index in range(4)]
    assert [stub.pad_frame for stub in Recording.made] == [
        f"landing_pad_{index}" for index in range(4)]
    assert [stub.body_frame for stub in Recording.made] == [
        f"uav_body_{index}" for index in range(4)]

    # Routing follows the physical pair; a method that flies two pairs resolves
    # to its first one only when no pair is named.
    group.publish_benchmark_step(method="shin_se_fixed", pair_index=3, state={},
                                 scenario="x", step=1, dt=.1,
                                 geometric_in_fov=True, status="running")
    assert Recording.made[3].steps and not Recording.made[0].steps
    group.clear_trails(method="shin_se_onto_rgat_recovery")
    assert Recording.made[2].cleared == 1 and Recording.made[3].cleared == 0
    group.clear_trails()
    assert [stub.cleared for stub in Recording.made] == [1, 1, 2, 1]
    group.potential = "frozen"
    assert all(stub.potential == "frozen" for stub in Recording.made)
    group.close()
    assert [stub.closed for stub in Recording.made] == [1, 1, 1, 1]


def test_the_runner_hands_rviz_the_per_pair_method_list():
    import inspect

    import run_three_pipeline

    source = inspect.getsource(run_three_pipeline.main)
    assert "RvizPublisher.create(" in source
    assert "pair_methods=(pair_training_methods" in source, (
        "RViz must be built from the per-pair method list, not args.pipelines")
    methods = ["shin_se_fixed", "shin_se_onto_rgat_recovery"]
    _, pair_methods = run_three_pipeline._balanced_training_pair_assignment(
        methods, 4, 0)
    assert len(pair_methods) == 4


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


def test_a_dead_training_arm_stops_the_reward_design_stage_immediately():
    """The twelve hours the 2026-09-18 run spent after its baseline had died.

    ``training_futures`` is submitted before the reward-design stages and
    joined only after them, so a PPO arm that raises in its first minutes
    leaves its exception unread inside the future while the FOV-risk loop
    keeps flying episodes for a comparison that can no longer be made. The
    loop has to read the futures itself.
    """
    import sys
    from concurrent.futures import ThreadPoolExecutor

    sys.path.insert(0, str(ROOT / "python"))
    from run_three_pipeline import peer_training_health

    with ThreadPoolExecutor(max_workers=2) as pool:
        def die():
            raise RuntimeError("PX4 gateway timeout after 2.00 s")

        futures = {"shin_se_fixed": pool.submit(die)}
        futures["shin_se_fixed"].exception()          # let it settle
        check = peer_training_health(futures)
        with pytest.raises(RuntimeError) as excinfo:
            check()

    assert "shin_se_fixed" in str(excinfo.value)
    # The worker's own failure has to survive as the cause, or the report
    # names the symptom and loses the gateway timeout that produced it.
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "PX4 gateway timeout" in str(excinfo.value.__cause__)


def test_a_healthy_or_absent_training_arm_never_stops_the_reward_design_stage():
    """A finished arm is not a failed one, and a single-pair run has none."""
    import sys
    from concurrent.futures import ThreadPoolExecutor

    sys.path.insert(0, str(ROOT / "python"))
    from run_three_pipeline import peer_training_health

    # ``training_futures`` is an empty dict until a parallel run submits to it.
    peer_training_health({})()

    with ThreadPoolExecutor(max_workers=1) as pool:
        futures = {"shin_se_fixed": pool.submit(lambda: "trained")}
        futures["shin_se_fixed"].result()
        peer_training_health(futures)()


def test_the_entry_travel_budget_is_paid_at_the_closing_speed_not_the_cruise():
    """The deck drives away for the whole trip, so the gap closes slowly.

    Assuming the airframe's own cruise timed out vehicles that were flying
    the transit correctly, and every expiry rebuilt the shared simulator
    under both pairs. The allowance has to cover the longest transit the
    profile can produce at the speed the gap actually closes at.
    """
    from config_loader import load_config

    external = default_config().external
    system = load_config(ROOT / "config/shin2026-minimal-system.yaml")
    deck_ceiling = (8.0 * float(system["pad"]["benchmark_speed_scale"]))

    closing = float(external["entry_travel_speed"])
    assert 0.0 < closing < deck_ceiling, (
        "a stern chase cannot close faster than the deck runs away")

    # The worst transit observed on the four-pair run was 39 m; truncating the
    # allowance below it is the same failure with an extra step.
    longest_transit_m = 39.0
    needed = (longest_transit_m - float(external["entry_tolerance"])) / closing
    assert float(external["entry_travel_budget_max"]) >= needed

    # The wall-clock hang guard must outlast the whole simulated ceiling; the
    # measured multi-pair stage advances roughly one simulated second per wall
    # second, so anything less cuts the budget off before it expires.
    ceiling = (float(external["entry_sim_budget"])
               + float(external["entry_travel_budget_max"]))
    assert float(external["entry_timeout"]) > ceiling


def test_the_entry_hold_never_exceeds_what_the_gateway_accepts():
    """The wall guard and the gateway's autonomous hold are different budgets.

    ``benchmark.entry_timeout_s`` is a wall-clock guard on a stopped simulator
    and is 1200 s on the shipped profile -- 1320 s once the parallel scale is
    applied. The gateway rejects a hold beyond GOTO_MAX_HOLD_S outright, which
    on 2026-09-21 failed the first reset of every stage after the warm start
    (the warm start shortens the guard to 120 s, which is what hid it).
    """
    import sys

    from ontology_rgat.bridge import (GATEWAY_MAX_HOLD_S, entry_hold_seconds)
    from ontology_rgat_px4.protocol import GOTO_MAX_HOLD_S
    from run_shin2026_pipeline import _live_config
    from run_three_pipeline import _pair_live_config

    assert GATEWAY_MAX_HOLD_S == GOTO_MAX_HOLD_S, (
        "the learner's copy of the gateway hold limit has drifted")

    # Short guards are passed through untouched.
    assert entry_hold_seconds(120.0, 2.55) == pytest.approx(122.55)

    cfg = _live_config("full", Path("/tmp/ontology_rgat_entry_hold"),
                       ROOT / "config/shin2026-minimal-system.yaml")
    for pairs in (1, 2, 3, 4):
        live = _pair_live_config(cfg, 0, pairs)
        margin = max(2.0, float(live.external.prestream_count)
                     / float(live.external.control_hz)
                     + float(live.external.reset_settle))
        hold = entry_hold_seconds(live.external.entry_timeout, margin)
        assert 0.0 < hold <= GOTO_MAX_HOLD_S, pairs

    with pytest.raises(Exception):
        entry_hold_seconds(float("nan"), 2.0)


def test_the_rviz_layout_is_told_which_arm_flies_each_pair(monkeypatch, tmp_path):
    """Pair titles name the arms that hold training pairs, in pair order.

    A run may compare more arms than it trains -- the three-arm burst
    comparison flies a non-learned visual servo that takes no training pair --
    so the launcher passes the learned labels and the generated layout titles
    each group with the arm actually flying it.
    """
    import subprocess

    from run_shin2026_pipeline import _start_rviz

    recorded = {}

    class Process:
        def poll(self):
            return None

    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr("run_shin2026_pipeline.shutil.which", lambda _name: "/usr/bin/rviz2")
    monkeypatch.setattr("run_shin2026_pipeline.time.sleep", lambda _s: None)
    monkeypatch.setattr(
        subprocess, "Popen",
        lambda command, **kwargs: recorded.setdefault("command", command) and None
        or Process())

    titles = ["Baseline · Shin SE fixed", "Proposed · Ontology-R-GAT FOV"]
    process, stream = _start_rviz(True, parallel_pairs=4, arm_titles=titles)
    assert process is not None
    command = recorded["command"]
    assert command[1:3] == ["--parallel-pairs", "4"]
    assert command.count("--arm-title") == 2
    assert command[command.index("--arm-title") + 1] == titles[0]
    assert command[-1] == titles[1]
    if stream is not None and not stream.closed:
        stream.close()

    # A single-pair run keeps the shipped layout and passes no titles.
    recorded.clear()
    process, stream = _start_rviz(True, parallel_pairs=1, arm_titles=titles)
    assert "--arm-title" not in recorded["command"]
    if stream is not None and not stream.closed:
        stream.close()


def test_the_rviz_launcher_reads_every_argument_before_it_builds_the_layout():
    """--arm-title may follow --parallel-pairs on the command line.

    Generating the layout inside the parse loop silently dropped any title
    that came after the pair count, which is the order the launcher emits.
    """
    script = (ROOT / "scripts/run_rviz.sh").read_text(encoding="utf-8")
    loop_end = script.index("done\n")
    generation = script.index("make_rviz_layout.py")
    assert generation > loop_end, (
        "the layout must be generated after the argument loop, not inside it")
    assert "--arm-title)" in script[:loop_end], "the launcher must accept the flag"
    assert 'arm_titles+=(--arm-title "$2")' in script
