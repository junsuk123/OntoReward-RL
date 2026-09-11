import pytest

from ontology_rgat.config import default_config
from ontology_rgat.viz.dashboard import PAGE
from ontology_rgat.viz.live import LiveStore, RewardMonitor


def test_reward_monitor_publishes_rgat_pbrs_decomposition():
    store = LiveStore()
    monitor = RewardMonitor(default_config(), store)
    monitor.reset("proposed", 7)
    monitor.update(
        step=15, t=0.3, mode="proposed", reward=0.397,
        parts={"base": -0.003, "shape": 0.4, "phi0": 0.2, "phi1": 0.4004},
        status="running")

    state = store.snapshot()
    assert state["scalars"]["reward_mode"] == "proposed"
    row = state["series"]["reward"][-1]
    assert row["base"] + row["shape"] == pytest.approx(row["reward"])
    assert row["phi"] == pytest.approx(0.2)
    assert row["phi_next"] == pytest.approx(0.4004)
    assert row["shaped"] is True


def test_dashboard_contains_formula_surface_and_live_decomposition():
    assert "r<sub>R-GAT</sub>" in PAGE
    assert "cv-rgat_reward_surface" in PAGE
    assert "Live reward decomposition during PPO" in PAGE
    assert "PBRS shaping" in PAGE
