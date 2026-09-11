"""Immutable reward-side potential for the controlled Shin benchmark."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


FEATURES = ("lateral_error", "altitude_error", "relative_horizontal_speed",
            "relative_vertical_speed")


class FrozenControlledPotential:
    """Load a controlled-profile R-GAT distillation artifact.

    The legacy urban artifact is rejected because its UGV/GNSS terms are not
    legal in the primary non-cooperative comparison.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        raw = self.path.read_bytes()
        data = json.loads(raw)
        if data.get("profile") != "controlled_landing":
            raise ValueError("reward artifact must use profile=controlled_landing")
        if not bool(data.get("frozen", False)):
            raise ValueError("controlled R-GAT reward artifact must be frozen")
        if str(data.get("provenance", "")).lower() != "rgat_distillation":
            raise ValueError("controlled reward weights must come from R-GAT distillation")
        weights = data.get("weights") or {}
        if set(weights) != set(FEATURES):
            raise ValueError(f"controlled reward weights must be exactly {FEATURES}")
        self.weights = {name: float(weights[name]) for name in FEATURES}
        if any(value < 0.0 or not np.isfinite(value) for value in self.weights.values()):
            raise ValueError("controlled reward weights must be finite and non-negative")
        if not np.isclose(sum(self.weights.values()), 1.0, atol=1e-6):
            raise ValueError("controlled reward weights must sum to one")
        self.design_id = str(data.get("design_id") or hashlib.sha256(raw).hexdigest()[:16])
        self.sha256 = hashlib.sha256(raw).hexdigest()

    def __call__(self, state) -> float:
        estimate = np.asarray(state["estimated_relative_state"], dtype=float).reshape(-1)
        if estimate.shape != (6,) or not np.isfinite(estimate).all():
            raise ValueError("controlled potential requires a finite six-state estimate")
        costs = {
            "lateral_error": np.clip(np.linalg.norm(estimate[:2]) / 4.25, 0.0, 1.0),
            "altitude_error": np.clip(abs(estimate[2]) / 8.0, 0.0, 1.0),
            "relative_horizontal_speed": np.clip(np.linalg.norm(estimate[3:5]) / 8.0,
                                                  0.0, 1.0),
            "relative_vertical_speed": np.clip(abs(estimate[5]) / 3.0, 0.0, 1.0),
        }
        return -float(sum(self.weights[name] * costs[name] for name in FEATURES))
