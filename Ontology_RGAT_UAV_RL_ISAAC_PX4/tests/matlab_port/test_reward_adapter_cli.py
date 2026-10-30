import json
import math
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ontology_rgat.bridge import CleanupStopUnconfirmed, PX4Failsafe
from ontology_rgat.direct_policy import runner as direct_runner
from ontology_rgat.direct_policy.isaac_adapter import (
    CommandOwnership, direct_acceleration_payload, embed_planar_action,
    trajectory_setpoint_fields,
)
from ontology_rgat.direct_policy.reward import (
    TerminalEvent, TruthState, ViewMeasurement, classify_terminal, reward_v5,
)
from ontology_rgat.direct_policy.reporting import authority_metrics, export_matlab


def test_terminal_order_and_reward_components_are_finite():
    old = TruthState((0.2, 1.0), (0.1, -0.2), (0.01,), (0.02,))
    new = TruthState((0.1, 0.9), (0.05, -0.2), (0.01,), (0.02,))
    event = classify_terminal(new, elapsed_s=1, contact=False)
    value, terms = reward_v5(old, new, ViewMeasurement(True, (0.1,), (1.0,)),
                             (0.1, -0.1), 0.1, event)
    assert math.isfinite(value) and all(math.isfinite(float(v)) for v in terms.values())
    landed = classify_terminal(TruthState((0.1, 0.04), (0.1, -0.1), (0,), (0,)),
                               elapsed_s=2, contact=True)
    assert landed.reason == "SUCCESS"
    missed = classify_terminal(TruthState((2, 0.04), (0, -0.1), (0,), (0,)),
                               elapsed_s=2, contact=True)
    assert missed.reason == "MISSED_PAD_CONTACT"


def test_px4_acceleration_boundary_and_command_ownership():
    enu = embed_planar_action((1.0, 0.5))
    payload = direct_acceleration_payload(enu, heading_rad=0.2, decision_id=1,
                                          episode_id="e", policy_version=3, stamp_s=4)
    setpoint = trajectory_setpoint_fields(enu, 0.2)
    np.testing.assert_allclose(setpoint["acceleration"], [0, 1, -0.5])
    assert all(math.isnan(item) for item in setpoint["position"]+setpoint["velocity"])
    owner = CommandOwnership(); owner.reset("e"); owner.validate(payload, writer="matlab-port")
    try:
        owner.validate(payload, writer="matlab-port")
        assert False, "duplicate decision accepted"
    except RuntimeError:
        pass


def test_cli_dry_run_is_non_mutating_and_does_not_require_isaac(tmp_path, request):
    root = request.config.rootpath
    command = [sys.executable, str(root/"python/run_matlab_port.py"), "--stage", "smoke",
               "--backend", "isaac", "--dimension", "2", "--dry-run",
               "--output", str(tmp_path)]
    result = subprocess.run(command, cwd=root, text=True, capture_output=True, check=True)
    assert '"status": "NOT_RUN"' in result.stdout


def test_cli_requires_a_checkpoint_before_actual_isaac(tmp_path, request):
    root = request.config.rootpath
    command = [sys.executable, str(root/"python/run_matlab_port.py"),
               "--stage", "evaluate", "--backend", "isaac", "--dimension", "2",
               "--allow-isaac", "--seed", "10000", "--output", str(tmp_path)]
    result = subprocess.run(command, cwd=root, text=True, capture_output=True)
    assert result.returncode != 0
    assert "--checkpoint-root" in result.stderr


def test_complete_pipeline_is_explicit_and_dry_run_is_non_mutating(tmp_path, request):
    project = request.config.rootpath
    output = tmp_path / "matlab-port"
    result = subprocess.run(
        ["bash", str(project.parent/"run.sh"), "matlab-port-all", "--dry-run",
         "--no-isaac", "--run-root", str(output)],
        cwd=project.parent, text=True, capture_output=True, check=True)
    assert '"status": "NOT_RUN"' in result.stdout
    assert "matlab_port_golden_6082258.json" in result.stdout
    assert not output.exists()


def test_final_isaac_pipeline_omits_non_deployment_stages(tmp_path, request):
    project = request.config.rootpath
    output = tmp_path / "matlab-port-final"
    result = subprocess.run(
        ["bash", str(project.parent/"run.sh"), "matlab-port-final", "--dry-run",
         "--run-root", str(output)],
        cwd=project.parent, text=True, capture_output=True, check=True)
    assert '"workflow": "final-isaac"' in result.stdout
    assert '"dimensions": [\n    3\n  ]' in result.stdout
    for retained in ("train-3d", "isaac-3d"):
        assert f'"name": "{retained}"' in result.stdout
    train_line = next(line for line in result.stdout.splitlines()
                      if '"pipeline_stage": "train-3d"' in line)
    assert '"--backend", "isaac"' in train_line
    assert '"--allow-isaac"' in train_line
    assert '"--control-profile", "direct"' in train_line
    assert '"--training-profile", "stable"' in train_line
    assert '"--episodes-per-update", "12"' in train_line
    isaac_line = next(line for line in result.stdout.splitlines()
                       if '"pipeline_stage": "isaac-3d"' in line)
    assert '"--training-profile", "stable"' in isaac_line
    assert '"--fresh-stack-per-episode"' not in train_line
    for omitted in ("train-2d", "isaac-2d", "audit-", "matlab-parity", "smoke-",
                    "evaluate-deterministic-", "evaluate-sampled-", "report-"):
        assert f'"name": "{omitted}' not in result.stdout
    assert not output.exists()


def test_training_checkpoints_every_update_and_resumes(tmp_path, request):
    project = request.config.rootpath
    runner = project/"python/run_matlab_port.py"
    output = tmp_path/"resume"
    common = [sys.executable, str(runner), "--stage", "train", "--dimension", "2",
              "--output", str(output), "--methods", "ppo",
              "--episodes-per-update", "1", "--validation-episodes", "1",
              "--steps", "2"]
    subprocess.run([*common, "--updates", "1"], cwd=project, check=True,
                   text=True, capture_output=True)
    assert (output/"ppo/checkpoint_last.pt").is_file()
    subprocess.run([*common, "--updates", "2", "--resume-checkpoint-root", str(output)],
                   cwd=project, check=True, text=True, capture_output=True)
    history = json.loads((output/"ppo/history.json").read_text())
    assert [row["update"] for row in history] == [1, 2]


def test_direct_rollout_retries_same_seed_after_recoverable_step_failure(monkeypatch):
    attempts = []
    recoveries = []

    def rollout(_contract, _model, seed, **_kwargs):
        attempts.append(seed)
        if len(attempts) == 1:
            error = PX4Failsafe(["gcs_connection_lost"], recoverable=True)
            error.transitions_collected = 7
            raise error
        return "batch", {"seed": seed}

    def recover(error, *, seed, transitions_collected):
        recoveries.append((seed, transitions_collected, error.recoverable))
        return True

    monkeypatch.setattr(direct_runner, "_rollout", rollout)
    monkeypatch.setattr(
        "ontology_rgat.spatial.lifecycle.recover_interrupted_episode", recover)

    batch, episode = direct_runner._rollout_resilient(
        object(), object(), 100088, deterministic=False, max_steps=3)

    assert batch == "batch" and episode["seed"] == 100088
    assert attempts == [100088, 100088]
    assert recoveries == [(100088, 7, True)]


def test_completed_rollout_survives_failed_cleanup_after_owned_restart(monkeypatch):
    completed = ("batch", {"seed": 100031, "status": "TASK_TIMEOUT"})
    failure = CleanupStopUnconfirmed("stop was not confirmed")
    failure.completed_rollout = completed
    failure.transitions_collected = 670
    calls = []

    def rollout(*_args, **_kwargs):
        calls.append("rollout")
        raise failure

    def recover(error, *, seed, transitions_collected):
        calls.append((error, seed, transitions_collected))
        return True

    monkeypatch.setattr(direct_runner, "_rollout", rollout)
    monkeypatch.setattr(
        "ontology_rgat.spatial.lifecycle.recover_failed_cleanup", recover)

    assert direct_runner._rollout_resilient(
        object(), object(), 100031, deterministic=True, max_steps=700) == completed
    assert calls == ["rollout", (failure, 100031, 670)]


def test_validation_progress_resumes_at_first_unwritten_seed(tmp_path, monkeypatch):
    calls = []

    def interrupted(_contract, _model, seed, **_kwargs):
        calls.append(seed)
        if seed == 12:
            raise RuntimeError("simulated interruption")
        return "batch", {"seed": seed, "status": "SUCCESS", "return": 1.0}

    progress = tmp_path/"validation.json"
    monkeypatch.setattr(direct_runner, "_rollout", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        direct_runner.evaluate(object(), object(), [11, 12, 13],
                               progress_path=progress,
                               progress_metadata={"checkpoint": "abc"})
    saved = json.loads(progress.read_text())
    assert [row["seed"] for row in saved["episodes"]] == [11]
    assert saved["complete"] is False

    def resumed(_contract, _model, seed, **_kwargs):
        calls.append(seed)
        return "batch", {"seed": seed, "status": "SUCCESS", "return": 2.0}

    monkeypatch.setattr(direct_runner, "_rollout", resumed)
    result = direct_runner.evaluate(
        object(), object(), [11, 12, 13], progress_path=progress,
        progress_metadata={"checkpoint": "abc"})
    assert calls == [11, 12, 12, 13]
    assert result["success_rate"] == 1.0
    assert json.loads(progress.read_text())["complete"] is True


def test_training_commits_update_before_validation_and_resumes_it(
        tmp_path, monkeypatch):
    from ontology_rgat.direct_policy.backends import backend_factory
    from ontology_rgat.direct_policy.contracts import make_contract
    from ontology_rgat.direct_policy.models import load_checkpoint

    output = tmp_path/"train"
    contract = make_contract(2, backend="local")
    args = SimpleNamespace(
        output=str(output), dimension=2, methods=["ppo"], updates=1,
        episodes_per_update=1, validation_episodes=1, evaluate_every=25,
        steps=2, training_profile="compatibility", seed=1, graph_seed=0,
        backend="local", resume_checkpoint_root=None)
    validation_calls = []

    def fail_validation(_contract, model, *_args, **_kwargs):
        validation_calls.append("failed")
        metadata = load_checkpoint(output/"ppo/checkpoint_last.pt", model)
        assert metadata["training_step"] == 1
        raise RuntimeError("validation interrupted")

    monkeypatch.setattr(direct_runner, "evaluate", fail_validation)
    with pytest.raises(RuntimeError, match="validation interrupted"):
        direct_runner._train_with_factory(
            args, contract, backend_factory(contract), backend_name="local-reference")
    history = json.loads((output/"ppo/history.json").read_text())
    assert history[0]["update"] == 1 and "validation" not in history[0]

    def finish_validation(*_args, **_kwargs):
        validation_calls.append("resumed")
        return {"deterministic": True,
                "episodes": [{"seed": 2001, "status": "SUCCESS", "return": 3.0}],
                "success_rate": 1.0, "unsafe_rate": 0.0, "mean_return": 3.0}

    monkeypatch.setattr(direct_runner, "evaluate", finish_validation)
    args.resume_checkpoint_root = str(output)
    direct_runner._train_with_factory(
        args, contract, backend_factory(contract), backend_name="local-reference")
    history = json.loads((output/"ppo/history.json").read_text())
    assert validation_calls == ["failed", "resumed"]
    assert history[0]["validation"]["success_rate"] == 1.0
    assert (output/"ppo/checkpoint_best.pt").is_file()


def test_authority_metrics_and_matlab_exports(tmp_path):
    records = [
        {"time_s": 0.0, "horizontal_distance_m": 1.0, "height_m": 2.0,
         "uav_x_m": 0.0, "uav_z_m": 2.0, "pad_x_m": 1.0, "pad_z_m": 0.0,
         "requested_ax": 1.0, "requested_az": 0.0, "applied_ax": 0.8,
         "applied_az": 0.0},
        {"time_s": 0.1, "horizontal_distance_m": 0.9, "height_m": 1.9,
         "uav_x_m": 0.1, "uav_z_m": 1.9, "pad_x_m": 1.0, "pad_z_m": 0.0,
         "requested_ax": 0.5, "requested_az": -0.1, "applied_ax": 0.5,
         "applied_az": -0.1},
    ]
    authority = authority_metrics([
        {"requested": [1, 0], "limited": [0.8, 0], "measured": [0.7, 0],
         "decision_stamp_s": 0, "command_stamp_s": 0.01},
        {"requested": [0.5, -0.1], "limited": [0.5, -0.1], "measured": [0.4, -0.1],
         "decision_stamp_s": 0.1, "command_stamp_s": 0.12},
    ])
    assert authority["intervention_rate"] == 0.5
    outputs = export_matlab(records, authority, tmp_path/"evaluation", {"split": "test"})
    assert all((tmp_path/f"evaluation.{suffix}").exists() for suffix in ("csv", "json", "mat"))
