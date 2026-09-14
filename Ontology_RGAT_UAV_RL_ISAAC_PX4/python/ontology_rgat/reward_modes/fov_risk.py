"""Additive future-FOV-loss reward used only by the proposed pipeline."""
from __future__ import annotations

import math


DEFAULT_FOV_RISK_LAMBDA = 0.1


def ontology_fov_reward(probability: float, lambda_fov: float = DEFAULT_FOV_RISK_LAMBDA) -> float:
    probability = float(probability)
    coefficient = float(lambda_fov)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("predicted FOV-loss probability must be in [0,1]")
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("lambda_fov must be finite and non-negative")
    return -coefficient * probability

