"""Immutable reward-side potential for the controlled Shin benchmark."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


FEATURES = ("lateral_error", "altitude_error", "relative_horizontal_speed",
            "relative_vertical_speed", "battery_risk")
NORMALIZATION = {
    "lateral_error": 4.25,
    "altitude_error": 8.0,
    "relative_horizontal_speed": 8.0,
    "relative_vertical_speed": 3.0,
    "battery_risk": 1.0,
}
EMPIRICAL_PROVENANCE = "isaac_px4_shin2026_rollouts"


def controlled_costs(estimated_relative_state, battery_reserve: float = 1.0) -> np.ndarray:
    """Map the visual estimate and onboard energy to five bounded costs."""
    estimate = np.asarray(estimated_relative_state, dtype=float).reshape(-1)
    if estimate.shape != (6,) or not np.isfinite(estimate).all():
        raise ValueError("controlled costs require a finite six-state estimate")
    battery_reserve = float(battery_reserve)
    if not np.isfinite(battery_reserve):
        raise ValueError("controlled costs require a finite battery reserve")
    return np.asarray([
        np.linalg.norm(estimate[:2]) / NORMALIZATION["lateral_error"],
        abs(estimate[2]) / NORMALIZATION["altitude_error"],
        np.linalg.norm(estimate[3:5]) / NORMALIZATION["relative_horizontal_speed"],
        abs(estimate[5]) / NORMALIZATION["relative_vertical_speed"],
        1.0 - np.clip(battery_reserve, 0.0, 1.0),
    ]).clip(0.0, 1.0)


class FrozenControlledPotential:
    """Load a controlled-profile R-GAT distillation artifact.

    The legacy urban artifact is rejected because its UGV/GNSS terms are not
    legal in the primary non-cooperative comparison.
    """

    def __init__(self, path: str | Path, *, expected_config_hash: str | None = None):
        self.path = Path(path).resolve()
        raw = self.path.read_bytes()
        data = json.loads(raw)
        if data.get("profile") != "controlled_landing":
            raise ValueError("reward artifact must use profile=controlled_landing")
        if data.get("format") != "ontology_rgat.controlled_reward/3":
            raise ValueError("controlled reward artifact must use battery-aware format version 3")
        if not bool(data.get("frozen", False)):
            raise ValueError("controlled R-GAT reward artifact must be frozen")
        if str(data.get("provenance", "")).lower() != "rgat_distillation":
            raise ValueError("controlled reward weights must come from R-GAT distillation")
        if data.get("dataset_provenance") != EMPIRICAL_PROVENANCE:
            raise ValueError(
                "controlled reward design requires actual Isaac/PX4 Shin rollouts")
        self.config_hash = str(data.get("dataset_config_hash") or "")
        if not self.config_hash:
            raise ValueError("controlled reward artifact has no dataset configuration hash")
        if expected_config_hash is not None and self.config_hash != str(expected_config_hash):
            raise ValueError("controlled reward artifact configuration mismatch")
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
        self.dataset_provenance = data["dataset_provenance"]
        self.dataset_sha256 = str(data.get("dataset_sha256") or "")

    def __call__(self, state) -> float:
        estimate = np.asarray(state["estimated_relative_state"], dtype=float).reshape(-1)
        if estimate.shape != (6,) or not np.isfinite(estimate).all():
            raise ValueError("controlled potential requires a finite six-state estimate")
        costs = controlled_costs(estimate, state.get("battery_reserve", 1.0))
        return -float(sum(self.weights[name] * value
                          for name, value in zip(FEATURES, costs)))
