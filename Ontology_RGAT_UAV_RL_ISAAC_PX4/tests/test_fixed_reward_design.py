import json

import numpy as np
import pytest

from ontology_rgat.config import default_config
from ontology_rgat.evaluation.acceptance import assess_optimization
from ontology_rgat.rgat.reward_design import (TERM_SPECS, distill_reward_design,
                                               load_reward_design,
                                               save_reward_design)
from ontology_rgat.semantic import SemanticState, build_ontology_graph


class LinearPotential:
    """A deterministic stand-in whose node sensitivity is easy to audit."""

    def predict_batch(self, X):
        X = np.asarray(X)
        coefficients = np.arange(1, X.shape[1] + 1, dtype=float)
        return X[:, :, 0] @ coefficients / coefficients.sum()


def _dataset(cfg):
    sem = SemanticState(
        position_error=0.2, vertical_speed=-0.3, tilt=0.05,
        angular_rate=0.1, wind_risk=0.4, marker_quality=0.7,
        visual_stability=0.7, alignment=0.8, attitude_stability=0.8,
        touchdown_safety=0.3, pad_motion=0.5, battery_reserve=0.6,
        gnss_integrity=0.75)
    graph = build_ontology_graph(sem, cfg)
    X = np.repeat(graph.X.T[None], 12, axis=0).astype(np.float32)
    return {"X": X, "y": np.zeros(12), "meta": np.zeros((12, 4)), "graph": graph}


def test_rgat_is_distilled_to_bounded_frozen_weights(tmp_path):
    cfg = default_config()
    dataset = _dataset(cfg)
    design = distill_reward_design(
        LinearPotential(), dataset, cfg,
        rgat_history={"train_loss": [0.4, 0.2], "val_loss": [0.5, 0.25]})

    assert set(design.weights) == {row[0] for row in TERM_SPECS}
    assert sum(design.weights.values()) == pytest.approx(1.0)
    assert min(design.weights.values()) >= cfg.reward.fixed.weight_min
    assert max(design.weights.values()) <= cfg.reward.fixed.weight_max
    assert -1.0 <= design.predict(dataset["graph"]) <= 0.0

    path = save_reward_design(design, tmp_path / "reward.json")
    loaded = load_reward_design(path)
    assert loaded.design_id == design.design_id
    assert loaded.weights == pytest.approx(design.weights)
    assert json.loads(path.read_text())["frozen"] is True


def test_dual_acceptance_requires_success_and_consistency():
    cfg = default_config().derive(**{
        "eval.acceptance.min_success_rate": 0.6,
        "eval.acceptance.max_success_std": 0.15,
        "eval.acceptance.min_worst_case_success": 0.35,
        "eval.acceptance.max_rgat_val_mse": 0.35,
    })
    design = distill_reward_design(
        LinearPotential(), _dataset(cfg), cfg,
        rgat_history={"train_loss": [0.2], "val_loss": [0.1]})
    summary = [{"Policy": "Ontology-RGAT", "SuccessRate": 0.75,
                "SuccessCount": 75, "Episodes": 100,
                "SuccessCI95Low": 0.65, "SuccessCI95High": 0.82}]
    rows = [{"Policy": "Ontology-RGAT", "WindScale": x, "SuccessRate": rate}
            for x, rate in ((0.5, 0.8), (1.0, 0.7), (1.5, 0.6))]
    report = assess_optimization(
        {"summary": summary, "wind_sweep": rows}, {"val_loss": [0.1]}, design, cfg)

    assert report["reward_optimization"]["pass"] is True
    assert report["rgat_consistency"]["pass"] is True
    assert report["overall_pass"] is True

    rows[-1]["SuccessRate"] = 0.0
    failed = assess_optimization(
        {"summary": summary, "wind_sweep": rows}, {"val_loss": [0.1]}, design, cfg)
    assert failed["reward_optimization"]["pass"] is True
    assert failed["rgat_consistency"]["pass"] is False
    assert failed["overall_pass"] is False
