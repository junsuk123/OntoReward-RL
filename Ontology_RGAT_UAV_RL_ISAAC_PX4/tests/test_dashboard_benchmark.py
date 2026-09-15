import numpy as np

from ontology_rgat.viz.dashboard import PAGE
from ontology_rgat.viz.live import BenchmarkMonitor, LiveStore, MATLAB_COLORS


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
        methods=["shin_se_fixed", "shin_se_onto_rgat_fov"], mode="quick",
        config_hash="1234567890abcdef", training_total=16,
        evaluation_total=12, reward_design_id="rgat-7",
        reward_design_sha256="abc", pair_layout=[{
            "index": 0, "method": "shin_se_onto_rgat_fov",
            "route_phase_fraction": 0.0, "px4_namespace": "/fmu",
            "topic_root": "/landing_pair_0",
            "camera_topic": "/landing_pair_0/uav/perception/landing_camera/annotated",
            "rviz_namespace": "/landing_rl/pair_0",
            "gateway_port": 14650, "learner_port": 14651,
        }])
    monitor.reset_episode(
        method="shin_se_onto_rgat_fov", phase="training", seed=42,
        scenario="training_random_walk", curriculum=0.25,
        action_scale=0.5125, motion_scale=0.5125)
    monitor.step(
        index=1, dt=0.1, method="shin_se_onto_rgat_fov", reward=0.3,
        reward_parts={"task": 0.0, "active_perception": -0.01,
                      "ontology_fov_reward": -0.02,
                      "predicted_fov_loss_probability": 0.2,
                      "fov_margin": 0.35, "keypoint_confidence": 0.7,
                      "visible_keypoint_fraction": 0.5},
        estimate=np.array([1, 0, -2, 0.5, 0, 0]),
        truth=np.array([0, 0, -2, 0, 0, 0]), in_fov=False,
        estimation_loss=0.2, state={
            "position": [0.0, 0.0, -2.0],
            "world": {"velocity": [0.3, 0.4, 0.0]},
            "pad": {"velocity": [0.12, 0.0, 0.0]},
            "battery": {
                "enabled": True, "reserve": .4, "state_of_charge": .03,
                "remaining_j": 4200.0, "energy_used_j": 600.0,
                "power_w": 190.0,
            }})
    monitor.training_update("shin_se_onto_rgat_fov", {
        "method": "shin_se_onto_rgat_fov", "episode": 1, "episode_return": 3.0,
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
    live = state["series"]["benchmark_step_shin_se_onto_rgat_fov"][-1]
    assert live["position_error"] == 1.0
    assert live["status"] == "running"
    assert live["active_perception"] == -0.01
    assert live["ontology_fov_reward"] == -0.02
    assert live["predicted_fov_loss_probability"] == 0.2
    assert live["battery_reserve"] == .4
    assert live["battery_remaining_j"] == 4200.0
    assert live["uav_speed_m_s"] == .5
    assert live["ugv_speed_m_s"] == .12
    assert state["series"].get("benchmark_step", []) == []
    assert state["series"]["benchmark_train_shin_se_onto_rgat_fov"][-1]["episode"] == 1
    assert state["scalars"]["current_action_envelope_scale"] == 0.5125
    assert state["scalars"]["current_pad_motion_scale"] == 0.5125
    assert state["scalars"]["current_training_episode"] == 1
    pair = state["scalars"]["parallel_pair_status"][0]
    assert pair["method"] == "shin_se_onto_rgat_fov"
    assert pair["assigned_method"] == "shin_se_onto_rgat_fov"
    assert pair["active_method"] == "shin_se_onto_rgat_fov"
    assert pair["step"] == 1
    assert pair["marker_visible"] is False
    assert pair["relative_xyz"] == [0.0, 0.0, -2.0]
    assert pair["active_perception"] == -0.01
    assert pair["fov_margin"] == 0.35
    assert pair["predicted_fov_loss_probability"] == 0.2
    assert pair["ontology_fov_reward"] == -0.02
    assert state["series"]["benchmark_step_pair_0"][-1]["reward"] == .3
    assert rviz.steps[-1]["reward"] == .3
    assert rviz.steps[-1]["reward_parts"]["fov_margin"] == .35
    assert rviz.steps[-1]["pair_index"] == 0


def test_benchmark_monitor_restores_csv_rows_and_pairs_evaluation_series():
    store = LiveStore()
    monitor = BenchmarkMonitor(store)
    monitor.configure(
        methods=["shin_se_fixed", "shin_se_onto_rgat_fov"], mode="full", config_hash="h",
        training_total=20, evaluation_total=4)
    monitor.restore_training("shin_se_fixed", [{
        "episode": "2", "paper_success": "1.0", "episode_return": "4.5"}])
    monitor.restore_evaluation([
        {"method": "shin_se_fixed", "scenario": "circle", "seed": "5",
         "paper_success": "1.0", "position_rmse": "0.2"},
        {"method": "shin_se_onto_rgat_fov", "scenario": "circle", "seed": "5",
         "paper_success": "0.0", "position_rmse": "0.3"},
    ])

    state = store.snapshot()
    assert state["series"]["benchmark_train_shin_se_fixed"][0]["episode"] == 2
    assert state["series"]["benchmark_eval_shin_se_fixed"][0]["evaluation_index"] == 1
    assert state["series"]["benchmark_eval_shin_se_onto_rgat_fov"][0]["evaluation_index"] == 1
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
        methods=["shin_se_fixed", "shin_se_onto_rgat_fov"], mode="quick",
        config_hash="h", training_total=6, evaluation_total=1,
        pair_layout=[
            {"index": 0, "method": "shin_se_fixed"},
            {"index": 1, "method": "shin_se_onto_rgat_fov"},
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
        methods=["shin_se_fixed", "shin_se_onto_rgat_fov"], mode="quick",
        config_hash="h", training_total=6, evaluation_total=1,
        pair_layout=[
            {"index": 0, "method": "shin_se_fixed"},
            {"index": 1, "method": "shin_se_onto_rgat_fov"},
        ])

    monitor.reset_started(
        method="shin_se_onto_rgat_fov", phase="FOV-risk offline data",
        seed=70000, scenario="training_random_walk")
    pairs = store.snapshot()["scalars"]["parallel_pair_status"]
    assert pairs[1]["status"] == "entry hover"
    assert pairs[1]["phase"] == "FOV-risk offline data"
    assert pairs[1]["episode"] == 0
    assert pairs[0]["status"] == "waiting"

    monitor.reset_episode(
        method="shin_se_onto_rgat_fov", phase="FOV-risk offline data",
        seed=70000, scenario="training_random_walk", curriculum=1.0)
    pairs = store.snapshot()["scalars"]["parallel_pair_status"]
    assert pairs[1]["status"] == "running"
    assert pairs[1]["episode"] == 1


def test_pair_status_routes_by_physical_index_during_crossover():
    store = LiveStore()
    monitor = BenchmarkMonitor(store)
    monitor.configure(
        methods=["shin_se_fixed", "shin_se_onto_rgat_fov"], mode="full", config_hash="h",
        training_total=2, evaluation_total=2,
        pair_layout=[
            {"index": 0, "method": "shin_se_fixed"},
            {"index": 1, "method": "shin_se_onto_rgat_fov"},
        ])

    monitor.reset_episode(
        method="shin_se_onto_rgat_fov", phase="evaluation", seed=8,
        scenario="circle", curriculum=1.0, pair_index=0)
    monitor.step(
        index=3, dt=.1, method="shin_se_onto_rgat_fov", reward=.1,
        reward_parts={}, estimate=None, truth=np.zeros(6), in_fov=True,
        estimation_loss=None, state={"position": [0.0, 0.0, -1.0]},
        pair_index=0)

    pairs = store.snapshot()["scalars"]["parallel_pair_status"]
    assert pairs[0]["index"] == 0
    assert pairs[0]["method"] == "shin_se_fixed"
    assert pairs[0]["assigned_method"] == "shin_se_fixed"
    assert pairs[0]["active_method"] == "shin_se_onto_rgat_fov"
    assert pairs[0]["step"] == 3
    assert pairs[1]["index"] == 1
    assert pairs[1]["step"] == 0


def test_dashboard_has_self_contained_matlab_style_benchmark_view():
    assert "두 방법론의 동일 Shin RL 계약과 추가 FOV 분기" in PAGE
    assert "benchmark_eval_scenario" in PAGE
    assert "Hard information boundary" in PAGE
    assert "parallel_live_method" in PAGE
    assert "pair-plot-grid" in PAGE
    assert "benchmark_step_pair_${index}" in PAGE
    assert "UAV action-envelope curriculum" in PAGE
    assert "실시간 비행 상태 · pair별 독립 sensor" in PAGE
    assert "배터리 고갈 종료율" in PAGE
    assert "학습 checkpoint" in PAGE
    assert "학습 완료 · 현재 crossover paired evaluation 갱신 중" in PAGE
    assert "물리 Pair ${index+1} · 현재 정책:" in PAGE
    assert "학습 배정:" in PAGE
    assert "Crossover 평가에서는 정책이 물리 pair를 seed마다 순환" in PAGE
    assert "“현재 정책”으로 표시된 방법에 귀속" in PAGE
    assert "add(`${methodLabel(method)} 평가`" in PAGE
    assert "border-top-color:${activeColor}" in PAGE
    assert "[현재 평가] 이동 성공률" in PAGE
    assert "[완료된 학습 기록] 이동 성공률" in PAGE
    assert "rows.length+' completed'" in PAGE
    assert "training_total||0)/" not in PAGE
    assert "parallel-pair-grid" in PAGE
    assert "2쌍 Shin baseline / Ontology-R-GAT FOV 비교" in PAGE
    assert "FOV Ontology → R-GAT 미래 소실 확률 모듈" in PAGE
    assert "공통 Shin active-perception reward" in PAGE
    assert "제안 모델에만 추가된 Ontology-R-GAT FOV-risk reward" in PAGE
    assert "P(loss≤1s)" in PAGE
    assert "series:['benchmark_step_shin_se_onto_rgat_fov']" in PAGE
    assert "모든 node 값" in PAGE
    assert "모든 edge · attention-head 값" in PAGE
    assert "모든 MLP 출력 head 값" in PAGE
    assert "state.graphs" in PAGE
    assert "graphSnapshots" in PAGE
    assert "view:'urban'" not in PAGE
    assert "rewardPanel" not in PAGE
    assert "debug only" not in PAGE
    assert "No SE 행동 정책" not in PAGE
    assert "[0,1,2].map" not in PAGE
    assert "prefers-color-scheme:dark" not in PAGE
    for color in MATLAB_COLORS:
        assert color in PAGE
