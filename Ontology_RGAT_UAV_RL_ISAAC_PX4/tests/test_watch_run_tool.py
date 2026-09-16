"""The read-only progress CLI used to watch an in-flight run over SSH."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "watch_run", ROOT / "tools/watch_run.py")
watch_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(watch_run)


def _state(**overrides):
    state = {
        "stage": {"name": "FOV-risk data", "detail": "visual-only graph"},
        "scalars": {
            "benchmark_phase": "reward-design",
            "benchmark_methods": ["shin_se_fixed", "shin_se_onto_rgat_recovery"],
            "training_total": 288,
            "parallel_pair_status": [
                {"index": 0, "method": "shin_se_fixed",
                 "active_method": "shin_se_fixed", "phase": "training",
                 "episode": 12, "step": 44, "status": "running"}],
        },
        "series": {
            "benchmark_train_shin_se_fixed": [
                {"paper_success": 1.0}, {"paper_success": 0.0}],
            "benchmark_eval_shin_se_fixed": [{"paper_success": 1.0}],
        },
    }
    state["scalars"].update(overrides.pop("scalars", {}))
    state.update(overrides)
    return state


def test_the_snapshot_names_the_stage_phase_and_both_arms():
    text = watch_run.render(_state())
    assert "FOV-risk data" in text
    assert "reward-design" in text
    assert "shin_se_fixed" in text and "shin_se_onto_rgat_recovery" in text
    assert "2/288 episodes" in text
    assert "pair 0" in text and "[running]" in text


def test_rows_without_a_success_field_are_skipped_not_counted_as_failures():
    state = _state()
    state["series"]["benchmark_train_shin_se_fixed"] = [
        {"paper_success": 1.0}, {"episode": 2}, {"paper_success": 1.0}]
    text = watch_run.render(state)
    # Two scored rows, both successes: a defaulted zero would read 66.7%.
    assert "100.0%" in text


def test_the_fov_lines_appear_only_once_their_phase_has_started():
    assert "FOV데이터" not in watch_run.render(_state())
    text = watch_run.render(_state(scalars={
        "fov_dataset": {"episodes": 17, "minimum": 40, "maximum": 120,
                        "supervised_samples": 4080, "masked_samples": 170,
                        "target_mean": 0.128, "loss_episodes": 6,
                        "covered": False, "environment_steps": 4250,
                        "cached": False},
        "fov_training": {"epoch": 34, "total_epochs": 80,
                         "validation_loss": 0.01643,
                         "best_validation_loss": 0.01628},
    }))
    assert "17/최소 40·상한 120" in text
    assert "두regime=아직" in text
    assert "epoch 34/80" in text


def test_the_model_line_states_whether_it_beat_the_constant_predictor():
    better = watch_run.render(_state(scalars={"fov_model": {
        "design_id": "fov-risk-abc123", "rmse": 0.07,
        "constant_predictor_rmse": 0.09, "frozen": True}}))
    assert "상수예측기보다 나음" in better and "동결=예" in better
    worse = watch_run.render(_state(scalars={"fov_model": {
        "design_id": "fov-risk-abc123", "rmse": 0.11,
        "constant_predictor_rmse": 0.09, "frozen": True}}))
    assert "상수예측기 이하" in worse


def test_missing_numbers_render_as_unmeasured_rather_than_zero():
    text = watch_run.render(_state(scalars={"fov_model": {
        "design_id": None, "rmse": None, "constant_predictor_rmse": None,
        "frozen": False}}))
    assert "RMSE -- / 상수 --" in text
    assert "0.000" not in text


def test_the_progress_bar_never_overflows_its_width():
    assert watch_run._bar(500, 100).count("#") == 24
    assert watch_run._bar(0, 100).count("#") == 0
    assert watch_run._bar(-5, 0) == "[" + "." * 24 + "]"
