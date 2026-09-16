"""Additive future-FOV-unavailability reward used only by the proposed pipeline.

The non-terminal reward is ``r_paper(t) - lambda * q_theta(G_{t+1})`` where
``q_theta`` is the frozen direct R-GAT scalar readout of the expected fraction
of the next ``H`` steps with the pad centre outside the frustum.  Terminal
rewards are the paper's, untouched.

This is a plain shaped reward.  It is not potential-based shaping, so no
optimal-policy-invariance claim follows from it.
"""
from __future__ import annotations

import math


DEFAULT_FOV_RISK_LAMBDA = 0.1


def ontology_fov_reward(predicted_unavailability: float,
                        lambda_fov: float = DEFAULT_FOV_RISK_LAMBDA) -> float:
    value = float(predicted_unavailability)
    coefficient = float(lambda_fov)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(
            "predicted FOV unavailability must be a time fraction in [0,1]")
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("lambda_fov must be finite and non-negative")
    return -coefficient * value
