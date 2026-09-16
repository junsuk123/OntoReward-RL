"""FOV data collection and offline readout training must be watchable.

Both are long, silent phases of the proposed arm's preparation. Without
telemetry the dashboard shows an idle pair and a stage name for many minutes,
which is indistinguishable from a stalled run.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "isaac_sim"))

from ontology_rgat.rgat import (FOV_FEATURE_NAMES,  # noqa: E402
                                FOVSemanticObservation, build_fov_graph,
                                build_fov_risk_dataset, prepare_fov_risk_artifact,
                                save_fov_risk_dataset, train_fov_risk_model)
from ontology_rgat.viz.dashboard import PAGE  # noqa: E402
from ontology_rgat.viz.live import BenchmarkMonitor, LiveStore  # noqa: E402


def _dataset(seed=0, episodes=6, length=30, steps=5):
    rng = np.random.default_rng(seed)
    out = []
    for episode in range(1, episodes + 1):
        visible, samples = True, []
        for _ in range(length):
            visible = bool(rng.random() > (0.1 if visible else 0.45))
            value = 0.8 if visible else 0.1
            graph = build_fov_graph(
                FOVSemanticObservation(*([value] * len(FOV_FEATURE_NAMES))))
            samples.append({"graph_X": graph.X, "geometric_in_fov": visible})
        out.append({"episode_id": episode, "seed": 100 + episode,
                    "samples": samples})
    return build_fov_risk_dataset(out, prediction_steps=steps)


def _monitor():
    store = LiveStore()
    return BenchmarkMonitor(store, rviz=None), store


# ------------------------------------------------------------- collection

def test_dataset_collection_publishes_progress_and_its_real_exit_condition():
    monitor, store = _monitor()
    assert store.snapshot()["scalars"].get("fov_dataset") is None
    monitor.fov_dataset(
        episodes=12, minimum=40, maximum=120, supervised_samples=300,
        masked_samples=60, target_mean=0.11, loss_episodes=3, covered=False,
        environment_steps=360)
    published = store.snapshot()["scalars"]["fov_dataset"]
    assert published["episodes"] == 12
    assert published["minimum"] == 40 and published["maximum"] == 120
    # The loop exits on coverage, not on the episode count; both are shown.
    assert published["covered"] is False
    assert published["supervised_samples"] == 300
    assert published["masked_samples"] == 60
    assert published["target_mean"] == pytest.approx(0.11)
    assert published["cached"] is False

    monitor.fov_dataset(
        episodes=41, minimum=40, maximum=120, supervised_samples=1100,
        masked_samples=205, target_mean=0.13, loss_episodes=9, covered=True,
        environment_steps=1230, cached=True)
    published = store.snapshot()["scalars"]["fov_dataset"]
    assert published["covered"] is True and published["cached"] is True


def test_a_target_mean_of_none_survives_publication():
    """An empty supervised set must read as unmeasured, not as zero."""
    monitor, store = _monitor()
    monitor.fov_dataset(
        episodes=1, minimum=8, maximum=24, supervised_samples=0,
        masked_samples=30, target_mean=None, loss_episodes=0, covered=False,
        environment_steps=30)
    assert store.snapshot()["scalars"]["fov_dataset"]["target_mean"] is None


# --------------------------------------------------------------- training

def test_offline_training_reports_every_epoch_through_the_progress_callback():
    seen = []
    model, history, metrics = train_fov_risk_model(
        _dataset(), seed=3, validation_fraction=0.34, epochs=4, batch_size=32,
        hidden_dim=8, progress=seen.append)
    assert [row["epoch"] for row in seen] == [1, 2, 3, 4]
    assert all(row["total_epochs"] == 4 for row in seen)
    for row in seen:
        for key in ("train_huber", "train_contract", "validation_loss",
                    "validation_huber", "validation_contract",
                    "best_validation_loss"):
            assert np.isfinite(row[key]), key
    # ``best`` is monotone and agrees with the run's own selection.
    best = [row["best_validation_loss"] for row in seen]
    assert best == sorted(best, reverse=True)
    assert best[-1] == pytest.approx(metrics["best_validation_loss"])
    assert any(row["improved"] for row in seen)
    assert len(history) == 4


def test_training_progress_lands_in_the_store_as_a_series_and_a_scalar():
    monitor, store = _monitor()
    for epoch in (1, 2):
        monitor.fov_training({
            "epoch": epoch, "total_epochs": 8, "train_loss": 0.2 / epoch,
            "train_huber": 0.18 / epoch, "train_contract": 0.01,
            "validation_loss": 0.3 / epoch, "validation_huber": 0.28 / epoch,
            "validation_contract": 0.004,
            "best_validation_loss": 0.3 / epoch, "improved": True})
    snapshot = store.snapshot()
    rows = snapshot["series"]["fov_risk_training"]
    assert [row["epoch"] for row in rows] == [1, 2]
    assert rows[-1]["validation_huber"] == pytest.approx(0.14)
    scalar = snapshot["scalars"]["fov_training"]
    assert scalar["epoch"] == 2 and scalar["total_epochs"] == 8
    assert scalar["improved"] is True


def test_the_frozen_readout_is_published_with_its_constant_predictor_baseline(tmp_path):
    dataset = _dataset(seed=5)
    manifest = save_fov_risk_dataset(
        dataset, tmp_path / "rollouts.npz", config_hash="cfg", seed=7,
        horizon_seconds=0.5, control_hz=10.0)
    seen = []
    frozen, metadata = prepare_fov_risk_artifact(
        tmp_path / "model.pt", dataset, dataset_manifest=manifest,
        config_hash="cfg", seed=7,
        settings={"epochs": 3, "batch_size": 32, "hidden_dim": 8,
                  "validation_fraction": 0.34},
        progress=seen.append)
    assert len(seen) == 3

    monitor, store = _monitor()
    monitor.fov_model(design_id=frozen.design_id, metadata=metadata)
    published = store.snapshot()["scalars"]["fov_model"]
    assert published["design_id"] == frozen.design_id
    assert published["frozen"] is True
    assert published["loss"] == "huber_regression_plus_contract_rule_R-04"
    # Without the constant-predictor RMSE an error figure cannot say whether
    # the readout learned anything beyond the mean.
    assert np.isfinite(published["rmse"])
    assert np.isfinite(published["constant_predictor_rmse"])
    assert np.isfinite(published["mae"])
    assert published["train_episodes"] >= 1
    assert published["validation_episodes"] >= 1
    assert published["supervised_samples"] > 0
    assert published["masked_tail_samples"] > 0


# ------------------------------------------------------------ page wiring

def test_the_dashboard_shows_the_three_preparation_stages_of_the_proposed_arm():
    assert "kind:'fovstatus'" in PAGE
    assert "fovStatusPanel(state)" in PAGE
    assert "제안 arm 준비 상태 · FOV 데이터 수집 → readout 학습 → 동결" in PAGE
    assert "1. 데이터 수집" in PAGE
    assert "2. readout 학습" in PAGE
    assert "3. 동결 readout 검증" in PAGE
    # The exit condition and the honest baseline both have to be on screen.
    assert "두 regime 확보" in PAGE
    assert "상수 예측기 RMSE" in PAGE
    assert "상수 예측기보다 나은가" in PAGE
    assert "PPO 중 동결" in PAGE
    # And the per-epoch curve.
    assert "FOV readout 오프라인 학습 (epoch)" in PAGE
    assert "series:['fov_risk_training']" in PAGE
    assert "'train_huber','validation_huber','validation_contract'" in PAGE
