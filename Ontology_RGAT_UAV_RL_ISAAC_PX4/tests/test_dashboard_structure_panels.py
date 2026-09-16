"""The three explanatory panels: run pipeline, algorithm, and the two MDPs.

They are built from executable constants, so these tests check the coupling
rather than the prose: rename a graph node or change lambda and the panel has
to follow. A diagram that can drift away from the code is worse than none.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "isaac_sim"))

from ontology_rgat.rgat.fov_graph import (FOV_FEATURE_NAMES,  # noqa: E402
                                          FOV_GOAL_NODE, FOV_GRAPH_EDGES,
                                          FOV_GRAPH_INPUT_DIM, FOV_NODE_NAMES,
                                          FOV_RELATION_NAMES)
from ontology_rgat.viz.contracts import (NON_PRIMARY_STAGES,  # noqa: E402
                                         RUN_PIPELINE_ALIASES,
                                         RUN_PIPELINE_STAGES,
                                         algorithm_pipeline_contract,
                                         mdp_contract, run_pipeline_contract)
from ontology_rgat.viz.dashboard import PAGE  # noqa: E402


def _algorithm(**overrides):
    settings = dict(lambda_fov=0.1, horizon_seconds=1.0, control_hz=10.0)
    settings.update(overrides)
    return algorithm_pipeline_contract(**settings)


def _mdp(**overrides):
    settings = dict(
        control={"dt_seconds": 0.1, "horizon_steps": 300,
                 "max_velocity_m_s": [2.0, 2.0, 1.0], "max_yaw_rate_deg_s": 60.0},
        reward={"terminal": {"success": 10.0, "failure": -10.0},
                "active_perception": {"alpha": 0.1, "beta": 1.0, "tau": 0.01}},
        lambda_fov=0.1, horizon_seconds=1.0, pad_speed_range=(0.08, 0.15))
    settings.update(overrides)
    return mdp_contract(**settings)


# ------------------------------------------------- 1. run/validation pipeline

def test_the_run_pipeline_covers_the_stages_the_runner_actually_publishes():
    spec = run_pipeline_contract()
    ids = [stage["id"] for stage in spec["stages"]]
    assert ids == list(ids)  # ordered, no reshuffling by the builder
    assert len(ids) == len(set(ids))
    for stage in spec["stages"]:
        assert stage["title"] and stage["detail"] and stage["produces"]
        assert stage["kind"] in {"infra", "gate", "train", "data", "eval", "report"}

    # Every stage name the two-pipeline runner can publish must land on a
    # stage, directly or through an alias, or the panel would show the wrong
    # position for part of the run.
    import re

    runner = (ROOT / "python/run_three_pipeline.py").read_text(encoding="utf-8")
    published = {match.group(1) for match in
                 re.finditer(r'\.stage\(\s*"([^"]+)"', runner)}
    assert "stack startup" in published and "FOV-risk data" in published
    resolvable = set(ids) | set(RUN_PIPELINE_ALIASES) | set(NON_PRIMARY_STAGES)
    assert sorted(name for name in published if name not in resolvable) == []
    # Aliases and exclusions must not overlap or contradict the track.
    assert set(RUN_PIPELINE_ALIASES.values()) <= set(ids)
    assert not (set(RUN_PIPELINE_ALIASES) & NON_PRIMARY_STAGES)
    assert not (set(ids) & NON_PRIMARY_STAGES)


def test_the_pipeline_marks_validation_gates_separately_from_training():
    kinds = {stage["id"]: stage["kind"] for stage in RUN_PIPELINE_STAGES}
    assert kinds["keypoint validation"] == "gate"
    assert kinds["deterministic checkpoint validation"] == "gate"
    assert kinds["paired evaluation"] == "eval"
    assert kinds["R-GAT training"] == "train"
    assert kinds["FOV-risk data"] == "data"


# ----------------------------------------------------- 2. algorithm pipeline

def test_the_algorithm_diagram_is_generated_from_the_graph_constants():
    chain = {node["id"]: node for node in _algorithm()["chain"]}
    assert list(chain) == ["sensor", "perception", "features", "graph",
                           "rgat", "scalar", "reward"]
    features = " ".join(chain["features"]["lines"])
    assert f"{len(FOV_FEATURE_NAMES)}개" in features
    graph = " ".join(chain["graph"]["lines"])
    assert f"{len(FOV_NODE_NAMES)} node" in graph
    assert f"{len(FOV_RELATION_NAMES)} relation" in graph
    assert f"edge {len(FOV_GRAPH_EDGES)}" in graph
    rgat = " ".join(chain["rgat"]["lines"])
    assert f"RGAT({FOV_GRAPH_INPUT_DIM} → 24)" in rgat
    assert "RGAT(24 → 1)" in rgat
    assert FOV_GOAL_NODE in " ".join(chain["scalar"]["lines"])
    # A different hidden width must show up rather than stay hardcoded.
    assert "RGAT(19 → 32)" in " ".join(
        _algorithm(hidden_dim=32)["chain"][4]["lines"])


def test_the_algorithm_diagram_follows_lambda_and_the_horizon():
    spec = _algorithm(lambda_fov=0.25, horizon_seconds=2.0, control_hz=10.0)
    reward = " ".join(spec["chain"][6]["lines"])
    assert "λ = 0.25" in reward
    scalar = " ".join(spec["chain"][5]["lines"])
    assert "20 step(2s)" in scalar
    assert "/ 20" in " ".join(spec["offline"]["lines"])


def test_the_offline_supervision_is_drawn_as_a_training_only_branch():
    spec = _algorithm()
    assert spec["offline"]["into"] == "rgat"
    lines = " ".join(spec["offline"]["lines"])
    assert "시뮬레이터 기하" in lines
    assert "mask" in lines and "0으로 채우지 않음" in lines
    assert "학습 시에만 · 추론에는 들어가지 않음" in PAGE


def test_the_diagram_colour_means_deployment_survival():
    """Blue/orange has to say the same thing as the 배포 유지/제거 line."""
    spec = _algorithm()
    kept = spec["deployed_stages"]
    surviving = [node["title"] for node in spec["chain"][:kept]]
    assert surviving == ["센서", "인식"]
    dropped = [node["title"] for node in spec["chain"][kept:]]
    assert "R-GAT" in dropped and "보상" in dropped
    # Everything the deployment note removes must be past the colour boundary.
    for name in ("온톨로지", "R-GAT", "보상", "critic"):
        assert name in spec["deployment"]["drops"]
    assert "spec.deployed_stages" in PAGE
    assert "배포에 남음" in PAGE


def test_the_algorithm_panel_states_what_it_does_not_claim():
    limits = " ".join(_algorithm()["limits"])
    assert "PBRS" in limits and "최적정책 불변" in limits
    assert "이진 손실 확률이 아니다" in limits
    assert "attention 자체는 인과 근거가 아니다" in limits


# --------------------------------------------------------- 3. MDP structure

def test_only_the_reward_row_differs_between_the_two_agents():
    spec = _mdp()
    differing = [row for row in spec["rows"] if not row["identical"]]
    assert [row["key"] for row in differing] == ["reward"]
    assert spec["total_count"] - spec["identical_count"] == 1
    assert [arm["id"] for arm in spec["arms"]] == [
        "shin_se_fixed", "shin_se_onto_rgat_recovery"]
    # Every identical row must still say something concrete.
    for row in spec["rows"]:
        assert row["label"] and row["value"]
    keys = [row["key"] for row in spec["rows"]]
    for required in ("observation", "action", "environment", "reward",
                     "termination", "critic", "deployment"):
        assert required in keys


def test_the_reward_row_carries_the_live_lambda_and_the_zero_lambda_contract():
    row = next(r for r in _mdp(lambda_fov=0.3)["rows"] if r["key"] == "reward")
    assert "λ=0.3" in row["delta"]
    assert "λ=0이면 baseline과 동일" in row["delta"]
    assert "Table III" in row["value"]
    assert "α=0.1" in row["value"] and "τ=0.01" in row["value"]


def test_the_mdp_rows_follow_the_resolved_action_and_environment_limits():
    spec = _mdp(control={"dt_seconds": 0.05, "horizon_steps": 600,
                         "max_velocity_m_s": [8.0, 8.0, 3.0],
                         "max_yaw_rate_deg_s": 120.0},
                pad_speed_range=(0.0, 8.0))
    action = next(r for r in spec["rows"] if r["key"] == "action")
    assert "8/8/3 m/s" in action["note"] and "120°/s" in action["note"]
    environment = next(r for r in spec["rows"] if r["key"] == "environment")
    assert "0–8 m/s" in environment["value"]
    assert "600 step" in environment["note"]


def test_a_temporary_fov_loss_is_documented_as_non_terminal():
    row = next(r for r in _mdp()["rows"] if r["key"] == "termination")
    assert "일시적 FOV 상실은 종료가 아니며" in row["note"]
    assert "LSTM hidden도 초기화하지 않는다" in row["note"]


# ------------------------------------------------------------- page wiring

def test_the_three_panels_are_wired_into_the_page():
    for marker in ("kind:'runpipe'", "kind:'algopipe'", "kind:'mdp'",
                   "runPipelinePanel(state)", "algoPipelinePanel(state)",
                   "mdpPanel(state)", "run-pipeline", "algo-pipeline",
                   "mdp-grid"):
        assert marker in PAGE, marker
    assert "시뮬레이션 · 학습 · 검증 파이프라인" in PAGE
    assert "제안 알고리즘 · 센서 → 온톨로지/R-GAT 그래프 → 보상 함수" in PAGE
    assert "두 강화학습 에이전트의 State · Action · Environment · Reward" in PAGE
    assert "4 · 알고리즘과 MDP 구조" in PAGE
    # The identical rows are merged into one cell on purpose.
    assert 'class="same"' in PAGE and 'grid-column:2/4' in PAGE
