"""The live four-arm development dashboard (dashboard.py): read-only and live."""
from __future__ import annotations

import json
from pathlib import Path
import urllib.request

from ontology_rgat.current_semantics import development as dev
from ontology_rgat.current_semantics.contracts import (DEFAULT_CONFIG_PATH, contract_manifest,
                                                       load_config)
from ontology_rgat.current_semantics.dashboard import export, serve, snapshot
from ontology_rgat.current_semantics.models import ARMS

CONFIG = load_config()


def _registered(root: Path) -> dict:
    manifest = contract_manifest(CONFIG)
    (root / "g4_pilot.json").write_text(json.dumps({
        "protocol_complete": True, "learnability_passed": True,
        "hashes": {k: manifest[k] for k in ("g4_hash", "algorithm_hash", "task_hash",
                                            "calibration_hash")}}))
    _, payload = dev.register(DEFAULT_CONFIG_PATH, dev.DEFAULT_DEVELOPMENT_PATH, root)
    (root / dev.REGISTRATION).write_text(json.dumps(payload))
    return payload


def _model(root, seed, arm, updates, *, done=False, success=0.5):
    d = root / "development" / f"seed{seed}" / arm
    d.mkdir(parents=True, exist_ok=True)
    with (d / "history.jsonl").open("w") as f:
        for u in range(updates):
            f.write(json.dumps({"update": u, "difficulty": 1.0, "success": success,
                                "mean_return": 10.0, "log_std": [-1.0, -1.0, -1.0],
                                "mean_decisions": 60, "episodes": 12,
                                "statuses": {"SUCCESS": 6, "UNSAFE_CONTACT": 6},
                                "wall_s": float(u)}) + "\n")
    if done:
        cells = {c: {"episodes": 100, "success": success, "unsafe": 0.0, "timeout": 0.0}
                 for c in dev.CELLS}
        (d / "summary.json").write_text(json.dumps({
            "training": {"best_update": updates - 1},
            "validation": {"deterministic_success": 1.0, "sampled_success": 1.0},
            "cells": cells}))
    else:
        (d / "status.json").write_text(json.dumps({"phase": "training"}))


def test_snapshot_before_registration_shows_the_plan_and_writes_nothing(tmp_path):
    state = snapshot(tmp_path)
    assert state["overall"] == "not_registered" and state["arms"] == list(ARMS)
    assert len(state["models"]) == 20 and all(m["state"] == "pending"
                                              for m in state["models"].values())
    assert list(tmp_path.iterdir()) == []


def test_snapshot_follows_training_and_previews_the_gate_on_complete_seeds(tmp_path):
    _registered(tmp_path)
    for arm in ARMS:
        _model(tmp_path, 10, arm, 600, done=True,
               success=0.8 if arm == "ontology_rgat_ppo" else 0.6)
    _model(tmp_path, 11, "raw_ppo", 37)
    state = snapshot(tmp_path)
    assert state["overall"] == "running" and state["progress"]["current"] == "11/raw_ppo"
    assert state["progress"]["models_done"] == 4 and state["progress"]["seeds_complete"] == [10]
    assert state["models"]["11/raw_ppo"]["update"] == 36
    assert state["models"]["11/raw_ppo"]["history"]["unsafe"][0] == 0.5
    assert state["aggregate"]["ontology_rgat_ppo"]["composition_ood"]["success"] == 0.8
    preview = state["gate_preview"]
    assert preview["composition_margins"]["raw"]["mean"] > 0
    assert state["verdict"] is None
    json.dumps(state, allow_nan=False)


def test_server_and_export_serve_the_same_state(tmp_path):
    _registered(tmp_path)
    server, url = serve(tmp_path, port=8840, open_browser=False)
    try:
        page = urllib.request.urlopen(url, timeout=5).read().decode()
        api = json.loads(urllib.request.urlopen(url + "api/state", timeout=5).read())
    finally:
        server.shutdown()
        server.server_close()
    assert "__EMBEDDED_STATE__" not in page and "api/state" in page
    assert api["overall"] == "registered_waiting"
    out = export(tmp_path)
    assert out == tmp_path / "development" / "dashboard.html"
    assert '"overall": "registered_waiting"' in out.read_text()
