"""isaac-eval/1: the selected G4 checkpoints flown on a backend the caller owns.

The evaluation logic runs here on the LOCAL plant (any backend works); no test
reaches a flight stack -- the CLI refuses before anything is imported.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess

import pytest
import yaml

from ontology_rgat.current_semantics.backend import make_local_backend
from ontology_rgat.current_semantics.contracts import (DEFAULT_CONFIG_PATH, contract_manifest,
                                                       load_config)
from ontology_rgat.current_semantics.experiment import isaac_eval_state, report_states
from ontology_rgat.current_semantics.isaac_eval import HASH_KEYS, run_isaac_evaluation
from ontology_rgat.current_semantics.models import CurrentActorCritic, save_checkpoint

CONFIG = load_config()
PROJECT = Path(__file__).resolve().parents[2]
RUN_SH = PROJECT.parent / "run.sh"


def checkpoint(tmp_path, name="best.pt", config=CONFIG):
    model = CurrentActorCritic(config.g4.arm, config, seed=0)
    return save_checkpoint(tmp_path / name, model, manifest=contract_manifest(config),
                           extra={"update": 7, "model_seed": 0})


def test_registered_seeds_are_new_and_the_criterion_is_g4s():
    seeds = set(CONFIG.isaac_eval.seeds)
    g4 = CONFIG.g4
    assert not seeds & (set(g4.validation_seeds) | set(g4.test_seeds)
                        | set(CONFIG.g3.pilot_seeds))
    assert not any(996001 <= s <= 996048 or 994001 <= s <= 994030 or 997001 <= s <= 997048
                   or 998001 <= s <= 998048 for s in seeds)
    assert CONFIG.isaac_eval.deterministic_success_min == g4.criteria.deterministic_success_min
    assert contract_manifest(CONFIG)["isaac_eval_hash"]


@pytest.mark.parametrize("seed", [993001, 992001, 990001, 996010, 12000, 994005, 994015,
                                  994025, 997001, 998001])
def test_config_refuses_a_used_seed(tmp_path, seed):
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    data["isaac_eval"]["seeds"] = [seed]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="already used"):
        load_config(path)


def test_local_dry_flight_records_every_episode_and_never_completes_a_subset(tmp_path):
    paths = {"0": checkpoint(tmp_path, "a.pt"), "1": checkpoint(tmp_path, "b.pt")}
    backend = make_local_backend(CONFIG)
    try:
        report = run_isaac_evaluation(CONFIG, backend, paths, tmp_path / "out",
                                      seeds=CONFIG.isaac_eval.seeds[:1], log=lambda _: None)
    finally:
        backend.close()
    rows = [json.loads(line) for line in (tmp_path / "out" / "episodes.jsonl").open()]
    first = CONFIG.isaac_eval.seeds[0]
    assert [(r["model_seed"], r["seed"]) for r in rows] == [("0", first), ("1", first)]
    assert all(r["termination"]["cleanup_confirmed"] for r in rows)
    trace = [json.loads(line) for line in
             (tmp_path / "out" / "traces" / f"model0_seed{first}.jsonl").open()]
    assert len(trace) == rows[0]["decisions"]
    assert set(trace[0]) == {"t", "action", "applied", "measured", "measured_over_s",
                             "estimate", "truth_r", "truth_v", "truth_roll_pitch_deg", "reward"}
    assert report["protocol_complete"] is False and report["isaac_eval_passed"] is False
    assert report["hashes"] == {k: contract_manifest(CONFIG)[k] for k in HASH_KEYS}
    assert report["results"]["0"]["update"] == 7
    assert report["results"]["0"]["deterministic"]["episodes"] == 1
    json.dumps(report, allow_nan=False)


def test_diagnostic_pairs_fly_only_the_named_pairs_and_never_complete(tmp_path):
    paths = {"0": checkpoint(tmp_path, "a.pt"), "1": checkpoint(tmp_path, "b.pt")}
    backend = make_local_backend(CONFIG)
    try:
        report = run_isaac_evaluation(CONFIG, backend, paths, tmp_path / "out",
                                      pairs=[("1", CONFIG.isaac_eval.seeds[1])],
                                      log=lambda _: None)
    finally:
        backend.close()
    rows = [json.loads(line) for line in (tmp_path / "out" / "episodes.jsonl").open()]
    assert [(r["model_seed"], r["seed"]) for r in rows] == [("1", CONFIG.isaac_eval.seeds[1])]
    assert report["protocol_complete"] is False and report["isaac_eval_passed"] is False
    for bad in ([("2", CONFIG.isaac_eval.seeds[1])], [("0", 993001)], [("0", 994051)], []):
        with pytest.raises(ValueError, match="diagnostic pairs"):
            run_isaac_evaluation(CONFIG, None, paths, tmp_path / "x", pairs=bad)
    # A seed already spent on a recorded diagnosis may be re-flown; a fresh one may not.
    from ontology_rgat.current_semantics.isaac_eval import check_pairs
    from ontology_rgat.current_semantics.contracts import DIAGNOSTIC_SEED_RANGES
    spent = DIAGNOSTIC_SEED_RANGES[-2][1]
    assert check_pairs(CONFIG, [(0, spent)], paths) == [("0", spent)]


def test_resume_keeps_every_verdict_and_flies_only_the_missing_pairs(tmp_path):
    paths = {"0": checkpoint(tmp_path, "a.pt"), "1": checkpoint(tmp_path, "b.pt")}
    first = CONFIG.isaac_eval.seeds[0]
    out = tmp_path / "out"
    out.mkdir()
    kept = {"model_seed": "0", "seed": first, "status": "SUCCESS", "termination":
            {"cleanup_confirmed": True}}
    (out / "episodes.jsonl").write_text(json.dumps(kept) + "\n")
    (out / "interruptions.jsonl").write_text(json.dumps(
        {"model_seed": "1", "seed": first, "error": "PX4Failsafe: x"}) + "\n")
    with pytest.raises(ValueError, match="resume=True"):
        run_isaac_evaluation(CONFIG, None, paths, out, seeds=(first,), log=lambda _: None)
    backend = make_local_backend(CONFIG)
    try:
        report = run_isaac_evaluation(CONFIG, backend, paths, out, seeds=(first,),
                                      resume=True, log=lambda _: None)
    finally:
        backend.close()
    rows = [json.loads(line) for line in (out / "episodes.jsonl").open()]
    assert [(r["model_seed"], r["seed"]) for r in rows] == [("0", first), ("1", first)]
    assert rows[0] == kept and report["resumed_with"] == 1
    assert report["results"]["0"]["episodes"][0] == kept
    assert report["interruptions"] == [{"model_seed": "1", "seed": first,
                                        "error": "PX4Failsafe: x"}]
    (out / "episodes.jsonl").write_text(json.dumps({**kept, "seed": 993001}) + "\n")
    with pytest.raises(ValueError, match="does not plan"):
        run_isaac_evaluation(CONFIG, None, paths, out, seeds=(first,), resume=True)


def test_a_stack_failure_stops_the_flight_and_is_reported(tmp_path):
    class Broken:
        name = "isaac_v12"

        def reset(self, seed):
            raise RuntimeError("PX4 refused to arm")

    report = run_isaac_evaluation(CONFIG, Broken(), {"0": checkpoint(tmp_path)},
                                  tmp_path / "out", log=lambda _: None)
    assert report["protocol_complete"] is False
    assert report["stopped"] == {"model_seed": "0", "seed": CONFIG.isaac_eval.seeds[0],
                                 "error": "RuntimeError: PX4 refused to arm"}
    assert report["interruptions"] == [report["stopped"]]


def test_refuses_foreign_seeds_and_stale_checkpoints(tmp_path):
    with pytest.raises(ValueError, match="isaac_eval.seeds"):
        run_isaac_evaluation(CONFIG, None, {"0": checkpoint(tmp_path)}, tmp_path,
                             seeds=(993001,))
    other = replace(CONFIG, task=replace(CONFIG.task, reward=replace(
        CONFIG.task.reward, time_cost_per_s=0.9)))
    stale = checkpoint(tmp_path, "stale.pt", config=other)
    with pytest.raises(ValueError):
        run_isaac_evaluation(CONFIG, None, {"0": stale}, tmp_path)


def test_report_state_reads_only_a_complete_current_flight(tmp_path):
    assert isaac_eval_state(CONFIG, tmp_path)["value"] is None
    hashes = {k: contract_manifest(CONFIG)[k] for k in HASH_KEYS}
    result = {"0": {"deterministic": {"success": 0.9}}}
    base = {"hashes": hashes, "protocol_complete": True, "isaac_eval_passed": True,
            "criterion": {"deterministic_success_min": 0.5}, "results": result}
    for change, expected in (({}, True), ({"isaac_eval_passed": False}, False),
                             ({"protocol_complete": False, "stopped": "x"}, None),
                             ({"hashes": {**hashes, "task_hash": "0" * 64}}, None)):
        (tmp_path / "isaac_eval.json").write_text(json.dumps({**base, **change}))
        state = isaac_eval_state(CONFIG, tmp_path)
        assert state["value"] is expected, change
        assert report_states({}, state)["isaac_integration_passed"] == state


def run(*args):
    return subprocess.run(["bash", str(RUN_SH), "current-semantics-3d", *args],
                          capture_output=True, text=True, timeout=180, cwd=PROJECT)


def test_cli_refuses_without_allow_isaac_and_before_g4(tmp_path):
    done = run("--stage", "isaac-eval", "--run-root", str(tmp_path))
    assert done.returncode == 3 and "no stack was contacted" in done.stdout
    done = run("--stage", "isaac-eval", "--allow-isaac", "--run-root", str(tmp_path))
    assert done.returncode == 3 and "G4 is" in done.stdout
    done = run("--stage", "isaac-eval", "--backend", "isaac", "--run-root", str(tmp_path))
    assert done.returncode == 2
    done = run("--stage", "learnability", "--allow-isaac", "--run-root", str(tmp_path))
    assert done.returncode == 2
    assert list(tmp_path.iterdir()) == []


def test_trace_stage_refuses_without_approval_and_needs_pairs(tmp_path):
    done = run("--stage", "isaac-trace", "--pairs", "1:994031", "--run-root", str(tmp_path))
    assert done.returncode == 3 and "no stack was contacted" in done.stdout
    for args in (("--stage", "isaac-trace", "--allow-isaac"),
                 ("--stage", "isaac-eval", "--pairs", "1:994031")):
        assert run(*args, "--run-root", str(tmp_path)).returncode == 2
    assert list(tmp_path.iterdir()) == []


def test_a_checkpoint_from_another_task_flies_only_as_a_diagnostic(tmp_path):
    other = replace(CONFIG, task=replace(CONFIG.task, reward=replace(
        CONFIG.task.reward, time_cost_per_s=0.9)))
    foreign = checkpoint(tmp_path, "foreign.pt", config=other)
    from ontology_rgat.current_semantics.models import load_checkpoint
    model = CurrentActorCritic(CONFIG.g4.arm, CONFIG, seed=0)
    with pytest.raises(ValueError, match="task_hash"):
        load_checkpoint(foreign, model, manifest=contract_manifest(CONFIG))
    meta = load_checkpoint(foreign, model, manifest=contract_manifest(CONFIG),
                           allow_foreign_task=True)
    assert meta["metadata"]["task_hash"] == contract_manifest(other)["task_hash"]
    with pytest.raises(ValueError, match="diagnostic re-flight"):
        run_isaac_evaluation(CONFIG, None, {"0": foreign}, tmp_path / "x", foreign_task=True)
    # Any other binding still refuses, foreign task or not.
    scaled = replace(CONFIG, feature_scales=replace(CONFIG.feature_scales,
                                                    approach_position_m=2.0))
    elsewhere = checkpoint(tmp_path, "elsewhere.pt", config=scaled)
    with pytest.raises(ValueError, match="feature_hash"):
        load_checkpoint(elsewhere, model, manifest=contract_manifest(CONFIG),
                        allow_foreign_task=True)
    backend = make_local_backend(CONFIG)
    try:
        report = run_isaac_evaluation(CONFIG, backend, {"0": foreign}, tmp_path / "out",
                                      pairs=[("0", CONFIG.isaac_eval.seeds[0])],
                                      foreign_task=True, log=lambda _: None)
    finally:
        backend.close()
    assert report["foreign_task"] is True and report["isaac_eval_passed"] is False
    assert report["results"]["0"]["checkpoint_task_hash"] == contract_manifest(other)["task_hash"]


def test_selected_checkpoints_come_from_the_named_pilot_and_match_its_record(tmp_path):
    import sys
    sys.path.insert(0, str(PROJECT / "python"))
    from run_current_semantics_3d import _selected_checkpoints
    (tmp_path / "g4_pilot" / "seed0").mkdir(parents=True)
    model = CurrentActorCritic(CONFIG.g4.arm, CONFIG, seed=0)
    save_checkpoint(tmp_path / "g4_pilot" / "seed0" / "best.pt", model,
                    manifest=contract_manifest(CONFIG), extra={"update": 12, "model_seed": 0})
    record = {"tests": {"0": {"checkpoint": "/elsewhere/g4_pilot/seed0/best.pt", "update": 12}}}
    (tmp_path / "g4_pilot.json").write_text(json.dumps(record))
    assert _selected_checkpoints(CONFIG, tmp_path) == {
        "0": tmp_path / "g4_pilot" / "seed0" / "best.pt"}
    record["tests"]["0"]["update"] = 13
    (tmp_path / "g4_pilot.json").write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="holds update 12"):
        _selected_checkpoints(CONFIG, tmp_path)


def test_checkpoint_root_is_a_trace_option_only(tmp_path):
    done = run("--stage", "isaac-eval", "--checkpoint-root", str(tmp_path), "--run-root",
               str(tmp_path))
    assert done.returncode == 2 and list(tmp_path.iterdir()) == []
