import json
import subprocess
import sys

from ontology_rgat.direct_policy.visualization import export_report, snapshot


def _episode(seed, status, value):
    return {"seed": seed, "status": status, "return": value, "steps": 10}


def test_interrupted_training_snapshot_uses_committed_history_only(tmp_path):
    root = tmp_path / "run"
    train = root / "3d" / "train"
    (train / "ppo").mkdir(parents=True)
    (train / "training_manifest.json").write_text(json.dumps({
        "dimension": 3, "updates": 10, "episodes_per_update": 2,
        "target_kl": 0.02,
    }))
    history = [
        {"update": 1, "episodes": [_episode(1, "SUCCESS", 25),
                                    _episode(2, "TASK_TIMEOUT", -12)],
         "samples": 20, "post_step_kl": 0.01, "actor_lr": 2e-5,
         "accepted_steps": 2, "rejected_steps": 0},
        {"update": 2, "episodes": [_episode(3, "SUCCESS", 24),
                                    _episode(4, "UNSAFE_CONTACT", -40)],
         "samples": 20, "post_step_kl": 0.015, "actor_lr": 2e-5,
         "accepted_steps": 1, "rejected_steps": 1,
         "validation": {"success_rate": .75, "unsafe_rate": .0,
                        "mean_return": 14.0, "episodes": []}},
    ]
    (train / "ppo" / "history.json").write_text(json.dumps(history))
    (train / "episode_recovery.jsonl").write_text(
        json.dumps({"attempt": 1, "state": "restarting"}) + "\n" +
        json.dumps({"attempt": 1, "state": "restarted", "seed": 5,
                    "transitions_discarded": 3}) + "\n")
    (train / "cleanup_recovery.jsonl").write_text(
        json.dumps({"attempt": 2, "state": "restarted", "seed": 6,
                    "transitions_discarded": 0,
                    "counted_as_rl_episode": True}) + "\n")

    state = snapshot(root)

    assert state["state"] == "INTERRUPTED"
    arm = state["arms"][0]
    assert arm["updates_completed"] == 2 and arm["planned_updates"] == 10
    assert arm["episodes_complete"] == 4 and arm["success_rate"] == .5
    assert arm["unsafe_rate"] == .25 and arm["rejected_steps"] == 1
    assert arm["validations"][0]["success_rate"] == .75
    assert state["recoveries"]["episode_count"] == 1
    assert state["recoveries"]["cleanup_count"] == 1


def test_export_is_self_contained_and_cli_once_reads_it(tmp_path, request):
    root = tmp_path / "run"
    train = root / "3d" / "train" / "ppo"
    train.mkdir(parents=True)
    (train / "history.json").write_text(json.dumps([
        {"update": 1, "episodes": [_episode(1, "SUCCESS", 25)],
         "post_step_kl": 0.0, "accepted_steps": 1, "rejected_steps": 0}
    ]))
    export_report(root, planned_updates=2, episodes_per_update=1)
    page = (root / "visualization" / "index.html").read_text()
    state = json.loads((root / "visualization" / "state.json").read_text())
    assert "MATLAB-port 학습 · 결과 분석" in page
    assert "https://" not in page and "__EMBEDDED_STATE__" not in page
    assert state["arms"][0]["progress"] == .5

    tool = request.config.rootpath / "tools" / "matlab_port_dashboard.py"
    result = subprocess.run(
        [sys.executable, str(tool), "--run-root", str(root), "--once",
         "--planned-updates", "2", "--episodes-per-update", "1"],
        text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)["arms"][0]["episodes_complete"] == 1
