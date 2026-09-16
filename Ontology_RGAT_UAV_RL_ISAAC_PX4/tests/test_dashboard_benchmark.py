import numpy as np
import pytest

from ontology_rgat.viz.dashboard import PAGE
from ontology_rgat.viz.live import BenchmarkMonitor, LiveStore, MATLAB_COLORS


class _FakeRviz:
    def __init__(self):
        self.steps = []
        self.potential = None

    def clear_trails(self, **_kwargs):
        pass

    def publish_benchmark_step(self, **kwargs):
        self.steps.append(kwargs)


def _monitor(methods):
    """One configured monitor per pair, mirroring the two-pair stage layout."""
    store = LiveStore()
    rviz = _FakeRviz()
    monitor = BenchmarkMonitor(store, rviz=rviz)
    monitor.configure(
        methods=list(methods), mode="quick", config_hash="0" * 16,
        training_total=8, evaluation_total=4, reward_design_id="rgat-test",
        reward_design_sha256="abc",
        pair_layout=[{
            "index": index, "method": method,
            "route_phase_fraction": 0.0, "px4_namespace": "/fmu",
            "topic_root": f"/landing_pair_{index}",
            "camera_topic": (f"/landing_pair_{index}/uav/perception"
                             "/landing_camera/annotated"),
            "rviz_namespace": f"/landing_rl/pair_{index}",
            "gateway_port": 14650 + 2 * index,
            "learner_port": 14651 + 2 * index,
        } for index, method in enumerate(methods)])
    return monitor, store, rviz


def test_benchmark_monitor_publishes_refactored_contract_and_progress():
    class FakeRviz:
        def __init__(self):
            self.steps = []
            self.potential = None

        def clear_trails(self, **_kwargs):
            pass

        def publish_benchmark_step(self, **kwargs):
            self.steps.append(kwargs)

    store = LiveStore()
    rviz = FakeRviz()
    monitor = BenchmarkMonitor(store, rviz=rviz)
    monitor.configure(
        methods=["shin_se_fixed", "shin_se_onto_rgat_recovery"], mode="quick",
        config_hash="1234567890abcdef", training_total=16,
        evaluation_total=12, reward_design_id="rgat-7",
        reward_design_sha256="abc", pair_layout=[{
            "index": 0, "method": "shin_se_onto_rgat_recovery",
            "route_phase_fraction": 0.0, "px4_namespace": "/fmu",
            "topic_root": "/landing_pair_0",
            "camera_topic": "/landing_pair_0/uav/perception/landing_camera/annotated",
            "rviz_namespace": "/landing_rl/pair_0",
            "gateway_port": 14650, "learner_port": 14651,
        }])
    monitor.reset_episode(
        method="shin_se_onto_rgat_recovery", phase="training", seed=42,
        scenario="training_random_walk", curriculum=0.25,
        action_scale=0.5125, motion_scale=0.5125)
    monitor.step(
        index=1, dt=0.1, method="shin_se_onto_rgat_recovery", reward=0.3,
        reward_parts={"task": 0.0, "active_perception": -0.01,
                      "ontology_fov_reward": -0.02,
                      "predicted_fov_unavailability": 0.2,
                      "fov_margin": 0.35, "keypoint_confidence": 0.7,
                      "visible_keypoint_fraction": 0.5},
        estimate=np.array([1, 0, -2, 0.5, 0, 0]),
        truth=np.array([0, 0, -2, 0, 0, 0]), geometric_in_fov=False,
        estimation_loss=0.2, state={
            "position": [0.0, 0.0, -2.0],
            "world": {"velocity": [0.3, 0.4, 0.0]},
            "pad": {"velocity": [0.12, 0.0, 0.0]},
            "battery": {
                "enabled": True, "reserve": .4, "state_of_charge": .03,
                "remaining_j": 4200.0, "energy_used_j": 600.0,
                "power_w": 190.0,
            }})
    monitor.training_update("shin_se_onto_rgat_recovery", {
        "method": "shin_se_onto_rgat_recovery", "episode": 1, "episode_return": 3.0,
        "paper_success": 1.0, "position_rmse": 0.2,
        "velocity_rmse": 0.1, "auxiliary_estimation_loss": 0.05,
        "ppo_loss": -0.01, "value_loss": 0.2, "entropy": 2.0,
        "kl_divergence": 0.001, "curriculum": 0.25,
        "action_envelope_scale": 0.5125,
        "effective_learning_rate": 5e-5, "ppo_early_stop": 0.0,
        "optimization_phase": "ppo", "battery_energy_used_j": 600.0,
        "battery_reserve_final": .4, "battery_depleted": 0.0,
    })

    state = store.snapshot()
    assert state["scalars"]["dashboard_profile"] == "parallel_two_pair"
    assert "simulator truth" in state["scalars"]["actor_contract"]["forbidden"]
    assert "active perception" in state["scalars"]["actor_contract"]["reward_side"]
    assert "-lambda_fov" in state["scalars"]["actor_contract"]["reward_side"]
    live = state["series"]["benchmark_step_shin_se_onto_rgat_recovery"][-1]
    assert live["position_error"] == 1.0
    assert live["status"] == "running"
    assert live["active_perception"] == -0.01
    assert live["ontology_fov_reward"] == -0.02
    assert live["predicted_fov_unavailability"] == 0.2
    assert live["battery_reserve"] == .4
    assert live["battery_remaining_j"] == 4200.0
    assert live["uav_speed_m_s"] == .5
    assert live["ugv_speed_m_s"] == .12
    assert state["series"].get("benchmark_step", []) == []
    assert state["series"]["benchmark_train_shin_se_onto_rgat_recovery"][-1]["episode"] == 1
    assert state["scalars"]["current_action_envelope_scale"] == 0.5125
    assert state["scalars"]["current_pad_motion_scale"] == 0.5125
    assert state["scalars"]["current_training_episode"] == 1
    pair = state["scalars"]["parallel_pair_status"][0]
    assert pair["method"] == "shin_se_onto_rgat_recovery"
    assert pair["assigned_method"] == "shin_se_onto_rgat_recovery"
    assert pair["active_method"] == "shin_se_onto_rgat_recovery"
    assert pair["step"] == 1
    assert pair["geometric_pad_center_in_fov"] is False
    assert pair["relative_xyz"] == [0.0, 0.0, -2.0]
    assert pair["active_perception"] == -0.01
    assert pair["fov_margin"] == 0.35
    assert pair["predicted_fov_unavailability"] == 0.2
    assert pair["ontology_fov_reward"] == -0.02
    assert state["series"]["benchmark_step_pair_0"][-1]["reward"] == .3
    assert rviz.steps[-1]["reward"] == .3
    assert rviz.steps[-1]["reward_parts"]["fov_margin"] == .35
    assert rviz.steps[-1]["pair_index"] == 0


def test_benchmark_monitor_restores_csv_rows_and_pairs_evaluation_series():
    store = LiveStore()
    monitor = BenchmarkMonitor(store)
    monitor.configure(
        methods=["shin_se_fixed", "shin_se_onto_rgat_recovery"], mode="full", config_hash="h",
        training_total=20, evaluation_total=4)
    monitor.restore_training("shin_se_fixed", [{
        "episode": "2", "paper_success": "1.0", "episode_return": "4.5"}])
    monitor.restore_evaluation([
        {"method": "shin_se_fixed", "scenario": "circle", "seed": "5",
         "paper_success": "1.0", "position_rmse": "0.2"},
        {"method": "shin_se_onto_rgat_recovery", "scenario": "circle", "seed": "5",
         "paper_success": "0.0", "position_rmse": "0.3"},
    ])

    state = store.snapshot()
    assert state["series"]["benchmark_train_shin_se_fixed"][0]["episode"] == 2
    assert state["series"]["benchmark_eval_shin_se_fixed"][0]["evaluation_index"] == 1
    assert state["series"]["benchmark_eval_shin_se_onto_rgat_recovery"][0]["evaluation_index"] == 1
    assert state["scalars"]["evaluation_completed"] == 2


def test_evaluation_pair_uses_evaluation_index_not_finished_training_episode():
    store = LiveStore()
    monitor = BenchmarkMonitor(store)
    monitor.configure(
        methods=["shin_se_fixed"], mode="full", config_hash="h",
        training_total=32, evaluation_total=3,
        pair_layout=[{"index": 0, "method": "shin_se_fixed"}])
    monitor.restore_training("shin_se_fixed", [
        {"episode": str(index), "paper_success": "0.0"}
        for index in range(1, 33)])
    monitor.restore_evaluation([{
        "method": "shin_se_fixed", "scenario": "circle", "seed": "5",
        "paper_success": "1.0"}])
    restored_pair = store.snapshot()["scalars"]["parallel_pair_status"][0]
    assert restored_pair["episode"] == 1
    assert restored_pair["episode_kind"] == "평가"
    assert restored_pair["status"] == "complete"

    monitor.reset_episode(
        method="shin_se_fixed", phase="evaluation", seed=6,
        scenario="zigzag", curriculum=1.0)

    state = store.snapshot()
    pair = state["scalars"]["parallel_pair_status"][0]
    assert pair["episode"] == 2
    assert pair["episode_kind"] == "평가"
    assert state["scalars"]["current_evaluation_episode"] == 2
    assert state["scalars"]["current_training_episode"] is None


def test_warmup_and_fov_data_publish_independent_episode_progress():
    store = LiveStore()
    monitor = BenchmarkMonitor(store)
    monitor.configure(
        methods=["shin_se_fixed", "shin_se_onto_rgat_recovery"], mode="quick",
        config_hash="h", training_total=6, evaluation_total=1,
        pair_layout=[
            {"index": 0, "method": "shin_se_fixed"},
            {"index": 1, "method": "shin_se_onto_rgat_recovery"},
        ])

    monitor.reset_episode(
        method="shin_se_fixed", phase="perception warm-up", seed=10,
        scenario="warmup", curriculum=0.0, pair_index=0)
    warmup = store.snapshot()["scalars"]
    assert warmup["current_training_episode"] == 1
    assert warmup["current_evaluation_episode"] is None
    assert warmup["parallel_pair_status"][0]["episode_kind"] == "학습 warm-up"

    monitor.reset_episode(
        method="shin_se_fixed", phase="FOV-risk offline data", seed=20,
        scenario="training_random_walk", curriculum=1.0, pair_index=1)
    monitor.reset_episode(
        method="shin_se_fixed", phase="FOV-risk offline data", seed=21,
        scenario="training_random_walk", curriculum=1.0, pair_index=1)
    fov = store.snapshot()["scalars"]
    assert fov["current_design_episode"] == 2
    assert fov["current_training_episode"] is None
    assert fov["current_evaluation_episode"] is None
    assert fov["parallel_pair_status"][0]["episode"] == 1
    assert fov["parallel_pair_status"][1]["episode"] == 2
    assert fov["parallel_pair_status"][1]["episode_kind"] == "FOV 데이터"


def test_entry_hover_is_published_before_the_episode_starts():
    store = LiveStore()
    monitor = BenchmarkMonitor(store)
    monitor.configure(
        methods=["shin_se_fixed", "shin_se_onto_rgat_recovery"], mode="quick",
        config_hash="h", training_total=6, evaluation_total=1,
        pair_layout=[
            {"index": 0, "method": "shin_se_fixed"},
            {"index": 1, "method": "shin_se_onto_rgat_recovery"},
        ])

    monitor.reset_started(
        method="shin_se_onto_rgat_recovery", phase="FOV-risk offline data",
        seed=70000, scenario="training_random_walk")
    pairs = store.snapshot()["scalars"]["parallel_pair_status"]
    assert pairs[1]["status"] == "entry hover"
    assert pairs[1]["phase"] == "FOV-risk offline data"
    assert pairs[1]["episode"] == 0
    assert pairs[0]["status"] == "waiting"

    monitor.reset_episode(
        method="shin_se_onto_rgat_recovery", phase="FOV-risk offline data",
        seed=70000, scenario="training_random_walk", curriculum=1.0)
    pairs = store.snapshot()["scalars"]["parallel_pair_status"]
    assert pairs[1]["status"] == "running"
    assert pairs[1]["episode"] == 1


def test_pair_status_routes_by_physical_index_during_crossover():
    store = LiveStore()
    monitor = BenchmarkMonitor(store)
    monitor.configure(
        methods=["shin_se_fixed", "shin_se_onto_rgat_recovery"], mode="full", config_hash="h",
        training_total=2, evaluation_total=2,
        pair_layout=[
            {"index": 0, "method": "shin_se_fixed"},
            {"index": 1, "method": "shin_se_onto_rgat_recovery"},
        ])

    monitor.reset_episode(
        method="shin_se_onto_rgat_recovery", phase="evaluation", seed=8,
        scenario="circle", curriculum=1.0, pair_index=0)
    monitor.step(
        index=3, dt=.1, method="shin_se_onto_rgat_recovery", reward=.1,
        reward_parts={}, estimate=None, truth=np.zeros(6), geometric_in_fov=True,
        estimation_loss=None, state={"position": [0.0, 0.0, -1.0]},
        pair_index=0)

    pairs = store.snapshot()["scalars"]["parallel_pair_status"]
    assert pairs[0]["index"] == 0
    assert pairs[0]["method"] == "shin_se_fixed"
    assert pairs[0]["assigned_method"] == "shin_se_fixed"
    assert pairs[0]["active_method"] == "shin_se_onto_rgat_recovery"
    assert pairs[0]["step"] == 3
    assert pairs[1]["index"] == 1
    assert pairs[1]["step"] == 0


def test_dashboard_is_organised_as_comparison_then_ontology_then_run_state():
    """Three questions, in order, and nothing that answers none of them."""
    # 1. the two models against each other
    assert "1 · 두 모델 성능" in PAGE
    assert "[평가] 안전 착륙 성공률" in PAGE
    assert "[평가] scenario별 성공률" in PAGE
    assert "[평가] 상대 위치 RMSE (m)" in PAGE
    assert "[평가] 상대 속도 RMSE (m/s)" in PAGE
    assert "[학습] 안전 착륙 성공률" in PAGE
    assert "benchmark_eval_scenario" in PAGE
    # The head-to-head difference is its own tile, not something to subtract
    # by eye from two other tiles.
    assert "Δ 성공률" in PAGE
    assert "Δ FOV 소실률" in PAGE
    assert "Proposed − Baseline · 표본이 작으면 해석 금지" in PAGE
    assert "Proposed − Baseline · 음수가 개선" in PAGE

    # 2. what the ontology branch actually did, and whether it was right
    assert "2 · 온톨로지-R-GAT의 영향력" in PAGE
    assert "FOV 소실 시간 비율 (낮을수록 좋음)" in PAGE
    assert "readout 보정 · 예측 대 실측 FOV 비가용 비율" in PAGE
    assert "fov_predicted_mean" in PAGE and "fov_actual_mean" in PAGE
    assert "fov_prediction_mae" in PAGE and "fov_prediction_bias" in PAGE
    assert "ontology_fov_reward_share" in PAGE
    assert "ontology_fov_reward_sum" in PAGE
    assert "series:PROPOSED_TRAIN" in PAGE
    assert "FOV 온톨로지 → R-GAT 비가용 비율 readout" in PAGE

    # 3. run state, explicitly framed as not being the comparison
    assert "3 · 실행 상태" in PAGE
    assert "비교 대상이 아니라" in PAGE
    assert "두 방법론의 동일 Shin RL 계약과 추가 FOV 분기" in PAGE
    assert "Hard information boundary" in PAGE
    assert "배터리 고갈 종료율 (원문에 없는 이 백엔드의 종료 조건)" in PAGE

    # the live two-pair view survives, the redundant duplicates do not
    assert "pair-plot-grid" in PAGE
    assert "parallel-pair-grid" in PAGE
    assert "benchmark_step_pair_${index}" in PAGE
    assert "물리 Pair ${index+1} · 현재 정책:" in PAGE
    assert "학습 배정:" in PAGE
    assert "Crossover 평가에서는 정책이 물리 pair를 seed마다 순환" in PAGE
    assert "border-top-color:${activeColor}" in PAGE
    assert "graphSnapshots" in PAGE and "state.graphs" in PAGE
    assert "모든 node 값" in PAGE
    assert "모든 edge · attention-head 값" in PAGE
    for color in MATLAB_COLORS:
        assert color in PAGE


def test_the_dashboard_drops_cards_that_answer_none_of_the_three_questions():
    for removed in (
            "parallel_live_flight",            # duplicate of the pair panel
            "parallel_live_method",            # superseded by section 2
            "benchmark_aux",                   # covered by the RMSE cards
            "benchmark_lr",                    # PPO plumbing
            "benchmark_early_stop",
            "benchmark_battery_used",          # battery is a declared deviation
            "benchmark_battery_final",
            "benchmark_action_scale",          # merged into the curriculum card
            "UAV action-envelope curriculum",
            "실시간 비행 상태 · pair별 독립 sensor"):
        assert removed not in PAGE, removed
    # Stale vocabulary from the retired binary readout must not survive either.
    assert "P(loss≤1s)" not in PAGE
    assert "미래 FOV 소실 확률" not in PAGE
    assert "소실 확률" not in PAGE
    assert "view:'urban'" not in PAGE
    assert "rewardPanel" not in PAGE
    assert "prefers-color-scheme:dark" not in PAGE


def test_the_dashboard_calls_the_readout_a_time_fraction_not_a_probability():
    assert "q(1s 비가용 비율)" in PAGE
    assert "예측 q (향후 1s FOV 밖 시간 비율)" in PAGE
    assert "이진 확률이 아니다" in PAGE
    assert "추가 보상 −λ·q" in PAGE


def test_baseline_reports_keypoint_quality_from_the_shared_visual_features():
    """Both arms must be watchable on identical perception terms.

    ``keypoint_confidence``/``visible_keypoint_fraction`` used to be read from
    reward parts, which only the proposed arm populates. The baseline's tiles
    then sat at a constant zero and looked like a dead camera rather than a
    comparable measurement -- while it is precisely the baseline's perception
    behaviour that the comparison is about.
    """
    from ontology_rgat.rgat.fov_graph import FOV_FEATURE_NAMES

    monitor, store, _ = _monitor(["shin_se_fixed", "shin_se_onto_rgat_recovery"])
    features = {name: 0.0 for name in FOV_FEATURE_NAMES}
    features.update({"keypoint_confidence": 0.62,
                     "visible_keypoint_fraction": 0.5,
                     "fov_margin": 0.41})
    monitor.step(
        index=1, dt=0.1, method="shin_se_fixed", reward=0.2,
        # The baseline has no FOV reward terms at all.
        reward_parts={"task": 0.0, "active_perception": -0.01},
        estimate=np.zeros(6), truth=np.zeros(6), geometric_in_fov=True,
        estimation_loss=0.1,
        fov_semantic_features=np.array(
            [features[name] for name in FOV_FEATURE_NAMES]),
        state={"position": [0.0, 0.0, -2.0],
               "world": {"velocity": [0.0, 0.0, 0.0]},
               "pad": {"velocity": [0.0, 0.0, 0.0]},
               "battery": {"enabled": False}})

    point = store.snapshot()["series"]["benchmark_step_shin_se_fixed"][-1]
    assert point["keypoint_confidence"] == pytest.approx(0.62)
    assert point["visible_keypoint_fraction"] == pytest.approx(0.5)
    assert point["fov_margin"] == pytest.approx(0.41)
    # Geometric truth stays a separate series from perception quality.
    assert point["geometric_in_fov"] == 1.0
    assert point["fov_keypoint_confidence"] == pytest.approx(0.62)
    # The proposed arm's reward terms remain absent for the baseline.
    assert point["ontology_fov_reward"] == 0.0
    assert point["predicted_fov_unavailability"] == 0.0

    pair = store.snapshot()["scalars"]["parallel_pair_status"][0]
    assert pair["keypoint_confidence"] == pytest.approx(0.62)
    assert pair["visible_keypoint_fraction"] == pytest.approx(0.5)
    assert pair["geometric_pad_center_in_fov"] is True


def test_rviz_receives_geometric_fov_and_keypoint_quality_separately():
    monitor, _, rviz = _monitor(["shin_se_fixed"])
    from ontology_rgat.rgat.fov_graph import FOV_FEATURE_NAMES

    features = {name: 0.0 for name in FOV_FEATURE_NAMES}
    features["keypoint_confidence"] = 0.8
    features["visible_keypoint_fraction"] = 1.0
    monitor.step(
        index=1, dt=0.1, method="shin_se_fixed", reward=0.1,
        reward_parts={}, estimate=np.zeros(6), truth=np.zeros(6),
        geometric_in_fov=False, estimation_loss=0.1,
        fov_semantic_features=np.array(
            [features[name] for name in FOV_FEATURE_NAMES]),
        state={"position": [0.0, 0.0, -2.0],
               "world": {"velocity": [0.0, 0.0, 0.0]},
               "pad": {"velocity": [0.0, 0.0, 0.0]},
               "battery": {"enabled": False}})

    published = rviz.steps[-1]
    # Out of frame geometrically, yet the encoder is still confident: the two
    # must not be collapsed into one "marker" indicator.
    assert published["geometric_in_fov"] is False
    assert published["keypoint_confidence"] == pytest.approx(0.8)
    assert published["visible_keypoint_fraction"] == pytest.approx(1.0)


def test_every_dashboard_series_key_is_actually_emitted_somewhere():
    """A renamed metric must break a test, not silently blank a card.

    Cards read keys out of the telemetry by name. If a producer renames one,
    the card keeps rendering and just says "no data yet" forever, which is
    indistinguishable from a run that has not started.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "python/ontology_rgat"
    produced = "\n".join(
        (root / name).read_text(encoding="utf-8")
        for name in ("ppo/recurrent_train.py", "viz/live.py"))

    keys = set()
    for match in re.finditer(r"y:'([a-z0-9_]+)'", PAGE):
        keys.add(match.group(1))
    for match in re.finditer(r"y:\[([^\]]+)\]", PAGE):
        keys.update(re.findall(r"'([a-z0-9_]+)'", match.group(1)))
    # Sanity: the parse has to find the cards, not an empty set.
    assert "paper_success" in keys and "fov_prediction_mae" in keys
    assert len(keys) >= 20

    missing = sorted(key for key in keys if f'"{key}"' not in produced)
    assert missing == [], f"dashboard reads keys nothing emits: {missing}"


def test_the_fov_readout_id_is_not_taken_from_a_legacy_reward_design():
    """``Dashboard.start()`` restores a legacy fixed-reward design from disk
    into ``reward_design_id``. Labelling that as the FOV readout would credit
    a PBRS artifact for the proposed method's frozen scalar."""
    monitor, store, _ = _monitor(["shin_se_fixed", "shin_se_onto_rgat_recovery"])
    assert store.snapshot()["scalars"]["fov_risk_design_id"] is None
    monitor.configure(
        methods=["shin_se_fixed", "shin_se_onto_rgat_recovery"], mode="full",
        config_hash="cfg", training_total=4, evaluation_total=2,
        reward_design_id="legacy-pbrs-000", fov_risk_design_id="fov-risk-abc123",
        pair_layout=[{"index": 0, "method": "shin_se_fixed"},
                     {"index": 1, "method": "shin_se_onto_rgat_recovery"}])
    scalars = store.snapshot()["scalars"]
    assert scalars["fov_risk_design_id"] == "fov-risk-abc123"
    assert scalars["reward_design_id"] == "legacy-pbrs-000"
    # The tile reads only the dedicated key.
    assert "s.fov_risk_design_id" in PAGE
    assert "add('FOV readout 설계',String(s.reward_design_id)" not in PAGE
