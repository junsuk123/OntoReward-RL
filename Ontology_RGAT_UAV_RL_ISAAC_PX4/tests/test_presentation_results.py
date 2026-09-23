import csv
import json

import pytest

from ontology_rgat.evaluation.presentation import (
    _evaluation_is_complete, _wilson, write_presentation_results)


METHODS = (
    "shin_se_fixed",
    "shin_se_onto_rgat_state",
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
    """An unfinished run reports its own training and never a stale evaluation.

    The withheld-arm rule only applies to a REWARD-side ontology arm whose
    frozen readout does not exist yet: its episodes were flown against a reward
    the run cannot reproduce, so they are not reported. The current method has
    no such artifact -- its encoder is trained by PPO -- so both of its arms
    report the episodes they actually flew.
    """
    manifest = {
        "execution_status": "configured; results pending",
        "pipeline_specs": {method: {"fov_risk_reward_enabled": False}
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
    assert by_method[METHODS[1]]["episodes"] == 1
    assert by_method[METHODS[1]]["safe_landings"] == 0
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


def test_a_reward_side_arm_is_withheld_until_its_frozen_readout_exists(tmp_path):
    """The retired method's rule, kept executable under its own arm id.

    A reward-side ontology arm's episodes were flown against a reward the run
    cannot reproduce until the frozen readout exists, so they are withheld.
    The current method has no such artifact and is never withheld.
    """
    from ontology_rgat.evaluation.presentation import _current_training_rows

    methods = ("shin_se_fixed", "shin_se_onto_rgat_recovery")
    manifest = {
        "pipeline_specs": {method: {
            "fov_risk_reward_enabled": method.endswith("onto_rgat_recovery")}
            for method in methods},
        "fov_risk_design_id": None,
    }
    for method in methods:
        _write_csv(tmp_path / "models" / method / f"{method}_training.csv",
                   [_row(method, True)])
    rows = _current_training_rows(tmp_path, manifest, methods)
    assert len(rows["shin_se_fixed"]) == 1
    assert rows["shin_se_onto_rgat_recovery"] == []
    # With the readout in place the same arm reports normally again.
    ready = {**manifest, "fov_risk_design_id": "fov-risk-abc123"}
    assert len(_current_training_rows(tmp_path, ready, methods)[
        "shin_se_onto_rgat_recovery"]) == 1


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


SERVO = "image_based_visual_servo_v1"


def test_a_control_arm_appears_in_the_report_without_being_declared():
    """The figures report what flew, not a list in the reporting module.

    The 2026-09-22 comparison adds a non-learned control arm beside the two
    PPO arms. Keyed on a hard-coded pair, every figure dropped it silently
    while the tables carried it.
    """
    from ontology_rgat.evaluation.presentation import (METHOD_LABELS,
                                                       _method_color,
                                                       _method_label,
                                                       _resolve_methods)

    manifest = {"pipelines": list(METHODS)}
    rows = [_row(method, True) for method in (*METHODS, SERVO)]
    assert _resolve_methods(manifest, rows) == (*METHODS, SERVO)

    # A declared arm list wins, and sets the display order.
    declared = {"arms": [{"method": SERVO, "label": "Visual servo (non-learned)",
                          "learned": False},
                         {"method": METHODS[0], "learned": True}]}
    assert _resolve_methods(declared, rows)[0] == SERVO
    assert _method_label(SERVO) == "Visual servo (non-learned)"
    # An arm this module has no colour for still gets one.
    assert _method_color("some_new_arm", 0).startswith("#")

    # The run's own label wins, so a name set in config reaches the figures
    # without this file knowing it. A label equal to the method id is what a
    # run writes when its config has none, and must not beat the curated name.
    assert _method_label(METHODS[0], {METHODS[0]: "Reference PPO"}) == "Reference PPO"
    assert _method_label(METHODS[0], {METHODS[0]: METHODS[0]}) == METHOD_LABELS[METHODS[0]]
    assert _method_label("undeclared_arm", {}) == "undeclared_arm"


def test_a_control_arm_without_a_checkpoint_does_not_make_a_run_look_unfinished():
    """`pipelines` is the learned list when the run does not declare arms.

    Treating every arm as learned made a completed three-arm run fail the
    completeness test -- the control condition has no checkpoint -- and every
    figure then silently reported preliminary training data instead of the
    final evaluation.
    """
    manifest = {
        "execution_status": "complete real Isaac/Pegasus/PX4 run",
        "evaluation": {"straight_escape_burst": 2},
        "pipelines": list(METHODS),
        "selected_checkpoints": {
            method: {"sha256": f"digest-{index}"}
            for index, method in enumerate(METHODS)},
    }
    rows = []
    for index, method in enumerate(METHODS):
        rows.extend([_row(method, True, digest=f"digest-{index}") for _ in range(2)])
    rows.extend([_row(SERVO, False) for _ in range(2)])

    from ontology_rgat.evaluation.presentation import (_learned_methods,
                                                       _resolve_methods)
    methods = _resolve_methods(manifest, rows)
    assert SERVO in methods
    assert _learned_methods(manifest, methods) == METHODS
    assert _evaluation_is_complete(rows, manifest, methods)

    # A learned arm that really is missing its checkpoint still fails.
    broken = dict(manifest, selected_checkpoints={
        METHODS[0]: {"sha256": "digest-0"}})
    assert not _evaluation_is_complete(rows, broken, methods)
