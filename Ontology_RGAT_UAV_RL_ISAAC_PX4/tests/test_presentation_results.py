import csv
import json

import pytest

from ontology_rgat.evaluation.presentation import (
    _evaluation_is_complete, _wilson, write_presentation_results)


METHODS = (
    "shin_se_fixed",
    "shin_se_onto_rgat_recovery",
)


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _row(method, success, *, digest=""):
    return {
        "pipeline": method, "optimization_phase": "ppo",
        "strict_success": success, "paper_success": success,
        "pad_contact": 1, "unsafe_pad_contact": 0 if success else 1,
        "touchdown_lateral_error": .2 if success else .5,
        "geometric_fov_loss_fraction": .1, "selected_checkpoint_sha256": digest,
    }


def test_incomplete_run_uses_current_training_and_never_stale_evaluation(tmp_path):
    manifest = {
        "execution_status": "configured; results pending",
        "pipeline_specs": {method: {
            "fov_risk_reward_enabled": method.endswith("onto_rgat_recovery")}
            for method in METHODS},
        "fov_risk_design_id": None,
        "evaluation": {"circle": 1},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    for method in METHODS:
        _write_csv(tmp_path / "models" / method / f"{method}_training.csv",
                   [_row(method, method == "shin_se_fixed")])
    _write_csv(tmp_path / "evaluation/per_episode.csv",
               [_row(method, True, digest="stale") for method in METHODS])
    summary = write_presentation_results(tmp_path)
    assert summary["performance_status"] == "training_preliminary"
    by_method = {row["method"]: row for row in summary["safe_landing_metrics"]}
    assert by_method["shin_se_fixed"]["safe_landings"] == 1
    assert by_method["shin_se_onto_rgat_recovery"]["episodes"] == 0
    assert (tmp_path / "presentation/slide13_safe_landing_performance.png").is_file()
    assert (tmp_path / "presentation/slide14_rgat_reward_validation.png").is_file()
    for name in (
            "page14_success_rate_ci.png",
            "page14_touchdown_lateral_error.png",
            "fov_risk_model_validation.png",
            "page16_training_safe_landing_progress.png"):
        assert (tmp_path / "presentation" / name).is_file()
    text = (tmp_path / "presentation/presentation_results_summary.json").read_text()
    assert "NaN" not in text


def test_final_evaluation_requires_every_selected_checkpoint_digest():
    manifest = {
        "execution_status": "complete real Isaac/Pegasus/PX4 run",
        "evaluation": {"circle": 1, "zigzag": 1},
        "selected_checkpoints": {
            method: {"sha256": f"digest-{index}"}
            for index, method in enumerate(METHODS)},
    }
    rows = []
    for index, method in enumerate(METHODS):
        rows.extend([_row(method, True, digest=f"digest-{index}") for _ in range(2)])
    assert _evaluation_is_complete(rows, manifest)
    rows[-1]["selected_checkpoint_sha256"] = "wrong"
    assert not _evaluation_is_complete(rows, manifest)


def test_wilson_interval_always_contains_the_point_estimate():
    """The errorbar arrays are built as rate-low and high-rate.

    At p=0 the floating-point interval used to land on ~1e-17 instead of 0, so
    every figure refresh of a run without a success died on "'yerr' must not
    contain negative values" and the slides silently stopped updating.
    """
    for count in (1, 3, 8, 12, 25, 40, 48, 100, 400):
        low, high = _wilson(0, count)
        assert low <= 0.0 <= high
        low, high = _wilson(count, count)
        assert low <= 1.0 <= high
        successes = count // 2
        low, high = _wilson(successes, count)
        assert low <= successes / count <= high


def test_a_failed_refresh_does_not_leak_its_figures(tmp_path, monkeypatch):
    """The loop retries this after every episode; a leak per retry is a leak.

    Before the Wilson interval was clamped every refresh raised between
    ``plt.figure()`` and its ``plt.close()``, and matplotlib ended up warning
    about more than twenty open figures on a run that had produced none.
    """
    import matplotlib.pyplot as plt
    from ontology_rgat.evaluation import presentation

    def explode(*_args, **_kwargs):
        plt.figure()
        plt.figure()
        raise ValueError("'yerr' must not contain negative values")

    monkeypatch.setattr(presentation, "_write_presentation_results", explode)
    before = set(plt.get_fignums())
    with pytest.raises(ValueError, match="yerr"):
        write_presentation_results(tmp_path)
    assert set(plt.get_fignums()) == before
