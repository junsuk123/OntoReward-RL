"""Distil a trained R-GAT into a fixed, auditable reward function.

The R-GAT is deliberately allowed to be context dependent while it learns the
safe-landing potential.  PPO, however, receives one fixed set of coefficients:
we measure how much the trained potential changes when each physical exchange
term is neutralised over the collected dataset, project those sensitivities to
a bounded simplex, and freeze the result for the whole PPO/evaluation run.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Config
from ..semantic import OntologyGraph

__all__ = ["FixedRewardDesign", "distill_reward_design", "save_reward_design",
           "load_reward_design", "TERM_SPECS"]


# name, ontology nodes, neutral node value, physical interpretation
TERM_SPECS = (
    ("position_error", (0,), 0.0, "horizontal error / success radius"),
    ("vertical_speed", (1,), 0.0, "absolute vertical speed / success limit"),
    ("tilt", (2,), 0.0, "tilt / success limit"),
    ("angular_rate", (3,), 0.0, "body rate / success limit"),
    ("wind_risk", (4,), 0.0, "measured aerodynamic wind risk"),
    ("pad_tracking", (10,), 0.0, "deck-motion and closing-speed risk"),
    ("energy_risk", (11,), 1.0, "one minus battery reserve"),
    ("navigation_risk", (5, 12), 1.0, "marker/GNSS joint outage risk"),
)


def _ranges(cfg: Config) -> dict[str, dict[str, Any]]:
    return {
        "position_error": {"min": 0.0, "max": float(cfg.criteria.xy), "unit": "m"},
        "vertical_speed": {"min": 0.0, "max": float(cfg.criteria.vz), "unit": "m/s"},
        "tilt": {"min": 0.0, "max": float(cfg.criteria.tilt), "unit": "rad"},
        "angular_rate": {"min": 0.0, "max": float(cfg.criteria.rate), "unit": "rad/s"},
        "wind_risk": {"min": 0.0, "max": 1.0, "unit": "normalized"},
        "pad_tracking": {"min": 0.0, "max": 1.0, "unit": "normalized"},
        "energy_risk": {"min": 0.0, "max": 1.0, "unit": "normalized"},
        "navigation_risk": {"min": 0.0, "max": 1.0, "unit": "normalized"},
    }


def _project_bounded_simplex(values: np.ndarray, lower: float,
                             upper: float) -> np.ndarray:
    """Euclidean projection onto ``sum(w)=1, lower<=w<=upper``."""
    values = np.asarray(values, dtype=float)
    n = values.size
    if n * lower > 1.0 + 1e-12 or n * upper < 1.0 - 1e-12:
        raise ValueError("reward-weight bounds do not contain a unit simplex")
    values = np.maximum(values, 0.0)
    values = values / values.sum() if values.sum() > 0.0 else np.full(n, 1.0 / n)
    lo = float(np.min(values - upper)) - 1.0
    hi = float(np.max(values - lower)) + 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        projected = np.clip(values - mid, lower, upper)
        if projected.sum() > 1.0:
            lo = mid
        else:
            hi = mid
    out = np.clip(values - 0.5 * (lo + hi), lower, upper)
    # Remove the last few floating-point ulps without changing the bounds.
    residual = 1.0 - float(out.sum())
    room = (upper - out) if residual > 0 else (out - lower)
    index = int(np.argmax(room))
    out[index] += residual
    return out


def _set_node_value(X: np.ndarray, node: int, value: float) -> None:
    """Change only the state-dependent pair; identity/risk/bias stay fixed."""
    X[:, node, 0] = value
    X[:, node, 1] = 1.0 - value


@dataclass(frozen=True)
class FixedRewardDesign:
    """The fixed coefficients used to calculate the proposed PBRS potential."""

    weights: dict[str, float]
    importance: dict[str, float]
    signed_effect: dict[str, float]
    ranges: dict[str, dict[str, Any]]
    task_reward: dict[str, float]
    lambda_: float
    gamma: float
    design_id: str
    dataset_samples: int
    rgat_epochs: int
    rgat_val_mse: float | None = None

    def _costs(self, graph: OntologyGraph) -> dict[str, float]:
        v = np.asarray(graph.X[0], dtype=float)
        return {
            "position_error": float(np.clip(abs(v[0]) / self.ranges["position_error"]["max"], 0, 1)),
            "vertical_speed": float(np.clip(abs(v[1]) / self.ranges["vertical_speed"]["max"], 0, 1)),
            "tilt": float(np.clip(abs(v[2]) / self.ranges["tilt"]["max"], 0, 1)),
            "angular_rate": float(np.clip(abs(v[3]) / self.ranges["angular_rate"]["max"], 0, 1)),
            "wind_risk": float(np.clip(v[4], 0, 1)),
            "pad_tracking": float(np.clip(v[10], 0, 1)),
            "energy_risk": float(np.clip(1.0 - v[11], 0, 1)),
            # nav_confidence = 1-(1-marker)*(1-gnss); its risk is the product.
            "navigation_risk": float(np.clip((1.0 - v[5]) * (1.0 - v[12]), 0, 1)),
        }

    def predict(self, graph: OntologyGraph) -> float:
        """Fixed linear safety potential ``Phi_w(s)=-sum_i w_i c_i(s)``."""
        costs = self._costs(graph)
        return -float(sum(self.weights[name] * costs[name] for name in self.weights))

    def explain_terms(self, graph: OntologyGraph) -> dict[str, float]:
        costs = self._costs(graph)
        return {name: -self.weights[name] * costs[name] for name in self.weights}

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "ontology_rgat.fixed_reward/1",
            "design_id": self.design_id,
            "frozen": True,
            "formula": "r_sparse + lambda*(gamma*Phi_w(s')-Phi_w(s))",
            "potential": "Phi_w(s)=-sum_i weight_i*normalized_cost_i(s)",
            "weights": dict(self.weights),
            "importance": dict(self.importance),
            "signed_effect": dict(self.signed_effect),
            "physical_ranges": self.ranges,
            "sparse_task_constants": dict(self.task_reward),
            "lambda": self.lambda_,
            "gamma": self.gamma,
            "dataset_samples": self.dataset_samples,
            "rgat_epochs": self.rgat_epochs,
            "rgat_val_mse": self.rgat_val_mse,
        }


def distill_reward_design(potential, dataset: dict[str, Any], cfg: Config,
                          *, rgat_history: dict[str, Any] | None = None
                          ) -> FixedRewardDesign:
    """Turn R-GAT counterfactual sensitivity into bounded fixed coefficients."""
    X = np.asarray(dataset["X"], dtype=np.float32)
    if X.ndim != 3 or X.shape[0] == 0:
        raise ValueError("reward design needs a non-empty [samples,nodes,features] dataset")
    limit = int(cfg.reward.fixed.max_attribution_samples)
    if X.shape[0] > limit:
        index = np.linspace(0, X.shape[0] - 1, limit, dtype=int)
        X = X[index]
    baseline = np.asarray(potential.predict_batch(X), dtype=float)
    importance: dict[str, float] = {}
    signed: dict[str, float] = {}
    for name, nodes, neutral, _ in TERM_SPECS:
        counterfactual = X.copy()
        for node in nodes:
            _set_node_value(counterfactual, node, neutral)
        neutral_prediction = np.asarray(potential.predict_batch(counterfactual), dtype=float)
        delta = neutral_prediction - baseline
        signed[name] = float(np.mean(delta))
        importance[name] = float(np.mean(np.abs(delta)))

    raw = np.asarray([importance[name] for name, *_ in TERM_SPECS], dtype=float)
    floor = float(cfg.reward.fixed.importance_floor)
    raw = raw + max(floor, 0.0)
    weights_array = _project_bounded_simplex(
        raw, float(cfg.reward.fixed.weight_min), float(cfg.reward.fixed.weight_max))
    weights = {spec[0]: float(value) for spec, value in zip(TERM_SPECS, weights_array)}
    history = dict(rgat_history or {})
    val = history.get("val_loss", [])
    payload = {
        "weights": weights,
        "ranges": _ranges(cfg),
        "lambda": float(cfg.reward.pbrs["lambda"]),
        "gamma": float(cfg.reward.pbrs.gamma),
        "task_reward": {str(k): float(v) for k, v in cfg.reward.sparse.items()},
        "samples": int(dataset["X"].shape[0]),
        "epochs": len(history.get("train_loss", [])),
    }
    design_id = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return FixedRewardDesign(
        weights=weights, importance=importance, signed_effect=signed,
        ranges=_ranges(cfg),
        task_reward={str(k): float(v) for k, v in cfg.reward.sparse.items()},
        lambda_=float(cfg.reward.pbrs["lambda"]),
        gamma=float(cfg.reward.pbrs.gamma), design_id=design_id,
        dataset_samples=int(dataset["X"].shape[0]),
        rgat_epochs=len(history.get("train_loss", [])),
        rgat_val_mse=float(val[-1]) if val else None)


def save_reward_design(design: FixedRewardDesign, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(design.to_dict(), indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return path


def load_reward_design(path: str | Path) -> FixedRewardDesign:
    blob = json.loads(Path(path).read_text(encoding="utf-8"))
    if blob.get("format") != "ontology_rgat.fixed_reward/1" or not blob.get("frozen"):
        raise ValueError(f"unsupported or non-frozen reward design: {path}")
    weights = {str(k): float(v) for k, v in blob["weights"].items()}
    wanted = {spec[0] for spec in TERM_SPECS}
    if set(weights) != wanted or not np.isclose(sum(weights.values()), 1.0, atol=1e-8):
        raise ValueError("fixed reward weights have the wrong terms or do not sum to one")
    return FixedRewardDesign(
        weights=weights,
        importance={str(k): float(v) for k, v in blob["importance"].items()},
        signed_effect={str(k): float(v) for k, v in blob["signed_effect"].items()},
        ranges=dict(blob["physical_ranges"]),
        task_reward={str(k): float(v) for k, v in
                     blob.get("sparse_task_constants", {}).items()},
        lambda_=float(blob["lambda"]),
        gamma=float(blob["gamma"]), design_id=str(blob["design_id"]),
        dataset_samples=int(blob["dataset_samples"]),
        rgat_epochs=int(blob["rgat_epochs"]),
        rgat_val_mse=(float(blob["rgat_val_mse"])
                      if blob.get("rgat_val_mse") is not None else None))
