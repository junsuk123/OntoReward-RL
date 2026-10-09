import json
import math
import subprocess
import sys

import numpy as np

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
    assert '"--fresh-stack-per-episode"' in train_line
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
