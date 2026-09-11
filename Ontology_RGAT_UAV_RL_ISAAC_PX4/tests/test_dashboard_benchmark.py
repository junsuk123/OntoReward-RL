import numpy as np

from ontology_rgat.viz.dashboard import PAGE
from ontology_rgat.viz.live import BenchmarkMonitor, LiveStore, MATLAB_COLORS


def test_benchmark_monitor_publishes_refactored_contract_and_progress():
    store = LiveStore()
    monitor = BenchmarkMonitor(store)
    monitor.configure(
        methods=["shin2026", "ontoreward"], mode="quick",
        config_hash="1234567890abcdef", training_total=16,
        evaluation_total=12, reward_design_id="rgat-7",
        reward_design_sha256="abc")
    monitor.reset_episode(
        method="ontoreward", phase="training", seed=42,
        scenario="training_random_walk", curriculum=0.25,
        action_scale=0.5125)
    monitor.step(
        index=1, dt=0.1, method="ontoreward", reward=0.3,
        reward_parts={"task": 0.0, "shape": 0.3, "phi": -0.5,
                      "phi_next": -0.2},
        estimate=np.array([1, 0, -2, 0.5, 0, 0]),
        truth=np.array([0, 0, -2, 0, 0, 0]), in_fov=False,
        estimation_loss=0.2, state={"battery": {
            "enabled": True, "reserve": .4, "state_of_charge": .03,
            "remaining_j": 4200.0, "energy_used_j": 600.0, "power_w": 190.0,
        }})
    monitor.training_update("ontoreward", {
        "method": "ontoreward", "episode": 1, "episode_return": 3.0,
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
    assert state["scalars"]["dashboard_profile"] == "shin2026"
    assert "simulator truth" in state["scalars"]["actor_contract"]["forbidden"]
    assert "battery reserve" in state["scalars"]["actor_contract"]["reward_side"]
    assert state["series"]["benchmark_step"][-1]["position_error"] == 1.0
    assert state["series"]["benchmark_step"][-1]["status"] == "running"
    assert state["series"]["benchmark_step"][-1]["shape"] == 0.3
    assert state["series"]["benchmark_step"][-1]["battery_reserve"] == .4
    assert state["series"]["benchmark_step"][-1]["battery_remaining_j"] == 4200.0
    assert state["series"]["benchmark_train_ontoreward"][-1]["episode"] == 1
    assert state["scalars"]["current_action_envelope_scale"] == 0.5125


def test_benchmark_monitor_restores_csv_rows_and_pairs_evaluation_series():
    store = LiveStore()
    monitor = BenchmarkMonitor(store)
    monitor.configure(
        methods=["shin2026", "ontoreward"], mode="full", config_hash="h",
        training_total=20, evaluation_total=4)
    monitor.restore_training("shin2026", [{
        "episode": "2", "paper_success": "1.0", "episode_return": "4.5"}])
    monitor.restore_evaluation([
        {"method": "shin2026", "scenario": "circle", "seed": "5",
         "paper_success": "1.0", "position_rmse": "0.2"},
        {"method": "ontoreward", "scenario": "circle", "seed": "5",
         "paper_success": "0.0", "position_rmse": "0.3"},
    ])

    state = store.snapshot()
    assert state["series"]["benchmark_train_shin2026"][0]["episode"] == 2
    assert state["series"]["benchmark_eval_shin2026"][0]["evaluation_index"] == 1
    assert state["series"]["benchmark_eval_ontoreward"][0]["evaluation_index"] == 1
    assert state["scalars"]["evaluation_completed"] == 2


def test_dashboard_has_self_contained_matlab_style_benchmark_view():
    assert "Shin 2026 controlled benchmark" in PAGE
    assert "benchmark_eval_scenario" in PAGE
    assert "Hard information boundary" in PAGE
    assert "recurrent estimate vs truth" in PAGE
    assert "UAV action-envelope curriculum" in PAGE
    assert "Current episode · real 3S battery" in PAGE
    assert "Battery-depletion terminal rate" in PAGE
    assert "prefers-color-scheme:dark" not in PAGE
    for color in MATLAB_COLORS:
        assert color in PAGE
