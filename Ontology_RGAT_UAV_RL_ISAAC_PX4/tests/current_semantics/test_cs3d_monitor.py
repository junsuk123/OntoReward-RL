"""The study monitor (monitor.py): read-only view of runs, processes, tasks and logs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
import urllib.request

from ontology_rgat.current_semantics.experiment import TASK_RUN_ROOTS
from ontology_rgat.current_semantics.monitor import parse_processes, serve, snapshot


def _history(run: Path, updates: int, *, difficulty: float = 1.0, done: bool = False) -> None:
    run.mkdir(parents=True, exist_ok=True)
    with (run / "history.jsonl").open("w") as f:
        for u in range(updates):
            f.write(json.dumps({"update": u, "difficulty": difficulty, "episodes": 12,
                                "success": 0.5, "mean_return": 3.0, "statuses":
                                {"SUCCESS": 6, "MISSED_PAD_CONTACT": 6},
                                "log_std": [-1.0, -1.2, -1.4], "accepted_steps": 30,
                                "rejected_steps": 1, "wall_s": 2.0 * (u + 1)}) + "\n")
    (run / "validation.jsonl").write_text(json.dumps(
        {"update": updates - 1, "deterministic": {"success": 0.75},
         "sampled": {"success": 0.5}}) + "\n")
    if done:
        (run / "final.pt").write_bytes(b"")


def _tree(tmp_path: Path) -> Path:
    v7 = tmp_path / TASK_RUN_ROOTS["spatial-current-task/7"].name
    _history(v7 / "g4_pilot" / "seed0", 100)
    _history(v7 / "development_design" / "reference" / "seed0", 600, done=True)
    _history(v7 / "g4_pilot_v8" / "g4_pilot" / "seed0", 5)          # archived: not a live run
    stalled = v7 / "development_design" / "old" / "seed1"
    _history(stalled, 50)
    old = time.time() - 3600
    os.utime(stalled / "history.jsonl", (old, old))
    (v7 / "g4_pilot10.log").write_text("seed 0 update 99/600\n")
    v5 = tmp_path / TASK_RUN_ROOTS["spatial-current-task/5"].name
    v5.mkdir()
    (v5 / "g4_pilot.json").write_text(json.dumps({
        "protocol": "g4-pilot/7", "learnability_passed": True,
        "tests": {"0": {"deterministic": {"success": 0.9}, "sampled": {"success": 0.8}}}}))
    (v5 / "isaac_handover_fix_2.json").write_text(json.dumps(
        {"protocol": "isaac-handover-fix/2", "complete": True, "fix_verified": True,
         "episodes": 80, "success": {"after": 74, "before": 57, "local": 75}}))
    (v5 / "isaac_handover_fix_2_registration.json").write_text("{}")
    return tmp_path


def _listing(root: Path) -> dict[str, float]:
    return {str(p): p.stat().st_mtime for p in root.rglob("*")}


def test_snapshot_reads_runs_tasks_and_logs_without_writing(tmp_path):
    root = _tree(tmp_path)
    before = _listing(root)
    state = snapshot(root)
    assert _listing(root) == before
    runs = {r["path"].split("/", 1)[1]: r for r in state["runs"]}
    assert set(runs) == {"g4_pilot/seed0", "development_design/reference/seed0",
                         "development_design/old/seed1"}
    live = runs["g4_pilot/seed0"]
    assert live["state"] == "running" and live["update"] == 99 and live["planned"] == 600
    assert abs(live["eta_s"] - 2.0 * 500) < 1e-6
    assert live["best_validation"] == {"deterministic": 0.75, "sampled": 0.5, "update": 99}
    assert abs(live["series"]["unsafe"][-1] - 0.5) < 1e-9 and live["kind"] == "G4 pilot"
    assert runs["development_design/reference/seed0"]["state"] == "done"
    assert runs["development_design/old/seed1"]["state"] == "stalled"
    assert [r["state"] for r in state["runs"]] == ["running", "stalled", "done"]
    tasks = {t["task"]: t for t in state["tasks"]}
    assert tasks["spatial-current-task/5"]["g4"]["passed"] is True
    assert [f["file"] for f in tasks["spatial-current-task/5"]["isaac"]] == \
        ["isaac_handover_fix_2.json"]
    assert tasks["spatial-current-task/6"]["root"] is None
    assert [l["path"].rsplit("/", 1)[-1] for l in state["logs"]] == ["g4_pilot10.log"]
    json.dumps(state, allow_nan=False, default=lambda o: None)


def test_spawned_workers_fold_into_their_job_and_shell_wrappers_are_dropped():
    text = "\n".join([
        "100 1 600 99.0 204800 python3 /x/python/run_current_semantics_3d.py --stage learnability"
        " --config /x/config/current_semantics/spatial_view_contact_v3.yaml",
        "101 100 590 95.0 102400 python3 -c from multiprocessing.spawn import spawn_main; "
        "spawn_main(tracker_fd=5, pipe_handle=7) --multiprocessing-fork",
        "102 100 590 97.0 102400 python3 -c from multiprocessing.spawn import spawn_main; "
        "spawn_main(tracker_fd=5, pipe_handle=9) --multiprocessing-fork",
        "103 1 30 0.0 4096 /bin/bash -c eval 'python3 run_current_semantics_3d.py --stage x'",
        "104 1 10 0.0 4096 python3 /y/other_tool.py"])
    rows = parse_processes(text)
    assert [r["pid"] for r in rows] == [100]
    job = rows[0]
    assert job["workers"] == 2 and abs(job["worker_cpu"] - 192.0) < 1e-9
    assert job["label"] == ("run_current_semantics_3d.py --stage learnability --config "
                            "spatial_view_contact_v3.yaml")


def test_the_server_publishes_the_page_and_the_state(tmp_path):
    root = _tree(tmp_path)
    server, url = serve(port=8840, results=root, open_browser=False)
    try:
        page = urllib.request.urlopen(url, timeout=5).read().decode("utf-8")
        assert "<title>Study Monitor</title>" in page and "api/state" in page
        state = json.loads(urllib.request.urlopen(url + "api/state", timeout=5).read())
        assert {r["state"] for r in state["runs"]} == {"running", "stalled", "done"}
    finally:
        server.shutdown()
        server.server_close()


def test_relation_shuffle_runs_and_their_verdict_are_shown(tmp_path):
    v5 = tmp_path / TASK_RUN_ROOTS["spatial-current-task/5"].name
    _history(v5 / "relation_shuffle" / "seed10" / "shuffle1", 40)
    (v5 / "relation_shuffle_registration.json").write_text(json.dumps(
        {"protocol": "relation-shuffle/1", "models": 10}))
    state = snapshot(tmp_path)
    (run,) = state["runs"]
    assert run["kind"] == "관계 섞기 대조군" and run["update"] == 39
    (task,) = [t for t in state["tasks"] if t["task"] == "spatial-current-task/5"]
    assert task["relation_shuffle"] == {"protocol": "relation-shuffle/1", "models": 10,
                                        "complete": None, "verdict": None, "delta": None,
                                        "ci95": None}
