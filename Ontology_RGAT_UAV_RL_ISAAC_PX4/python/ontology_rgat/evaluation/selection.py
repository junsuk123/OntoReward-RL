"""Reward-independent held-out checkpoint selection for the refactored arms."""
from __future__ import annotations

import numpy as np


def trajectory_quality_penalty(metrics) -> float:
    rows = list(metrics)
    if not rows:
        raise ValueError("trajectory quality needs held-out episodes")
    mean = lambda key: float(np.mean([float(row.get(key, 0.0)) for row in rows]))
    return float(2.0 * mean("peak_height_m")
                 + 2.0 * mean("cumulative_climb_after_loss_m")
                 + mean("landing_time_s")
                 + 20.0 * mean("command_variation"))


def robust_checkpoint_score(metrics) -> tuple[float, dict]:
    rows = list(metrics)
    if not rows:
        raise ValueError("checkpoint selection requires held-out episodes")
    scenarios = {}
    for row in rows:
        scenarios.setdefault(str(row.get("scenario", "unknown")), []).append(row)
    scenario_rates = [float(np.mean([float(item.get("strict_success", 0.0))
                                     for item in group]))
                      for group in scenarios.values()]
    robust = float(np.mean([float(row.get("strict_success", 0.0)) for row in rows]))
    worst = min(scenario_rates)
    nominal_rows = [row for row in rows if bool(row.get("nominal", False))] or rows
    nominal = float(np.mean([float(row.get("strict_success", 0.0))
                             for row in nominal_rows]))
    all_nominal = float(all(float(row.get("strict_success", 0.0)) > 0.0
                            for row in nominal_rows))
    capture = float(np.mean([float(row.get("capture_rate", 0.0)) for row in rows]))
    unsafe = float(np.mean([float(row.get(
        "unsafe_contact", row.get("unsafe_pad_contact", 0.0))) for row in rows]))
    mean_return = float(np.mean([float(row.get(
        "return", row.get("episode_return", 0.0))) for row in rows]))
    quality = trajectory_quality_penalty(rows)
    score = (10000.0 * robust + 2000.0 * worst + 1000.0 * nominal
             + 500.0 * all_nominal + 20.0 * capture - 100.0 * unsafe
             + 1e-3 * mean_return - quality)
    return float(score), {
        "robust_landing_rate": robust, "worst_scenario_landing_rate": worst,
        "nominal_landing_rate": nominal, "all_nominal_landed": all_nominal,
        "mean_capture_rate": capture, "unsafe_contact_rate": unsafe,
        "mean_return_tiebreak": mean_return,
        "trajectory_quality_penalty": quality,
    }
