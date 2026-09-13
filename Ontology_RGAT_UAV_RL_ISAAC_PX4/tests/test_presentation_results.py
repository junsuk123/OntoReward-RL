import csv
import json

from ontology_rgat.evaluation.presentation import (
    _evaluation_is_complete, write_presentation_results)


METHODS = (
    "shin_se_fixed",
    "no_se_fixed",
    "onto_rgat_adaptive_weight_no_se",
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
        "fov_loss_fraction": .1, "selected_checkpoint_sha256": digest,
    }


def test_incomplete_run_uses_current_training_and_never_stale_evaluation(tmp_path):
    manifest = {
        "execution_status": "configured; results pending",
        "pipeline_specs": {method: {
            "use_adaptive_reward_weights": method.startswith("onto_")}
            for method in METHODS},
        "adaptive_reward_design_id": None,
        "evaluation": {"circle": 1},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    for method in METHODS[:2]:
        _write_csv(tmp_path / "models" / method / f"{method}_training.csv",
                   [_row(method, method == "shin_se_fixed")])
    _write_csv(tmp_path / "evaluation/per_episode.csv",
               [_row(method, True, digest="stale") for method in METHODS])
    summary = write_presentation_results(tmp_path)
    assert summary["performance_status"] == "training_preliminary"
    by_method = {row["method"]: row for row in summary["safe_landing_metrics"]}
    assert by_method["shin_se_fixed"]["safe_landings"] == 1
    assert by_method["no_se_fixed"]["safe_landings"] == 0
    assert by_method["onto_rgat_adaptive_weight_no_se"]["episodes"] == 0
    assert (tmp_path / "presentation/slide13_safe_landing_performance.png").is_file()
    assert (tmp_path / "presentation/slide14_rgat_reward_validation.png").is_file()
    for name in (
            "page14_success_rate_ci.png",
            "page14_touchdown_lateral_error.png",
            "page15_reward_weights_comparison.png",
            "page15_reward_model_validation.png",
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
