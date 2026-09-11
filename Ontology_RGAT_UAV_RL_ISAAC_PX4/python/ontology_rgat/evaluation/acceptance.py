"""Two-gate acceptance for the learned reward and R-GAT robustness."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..config import Config
from .compare import LABELS

__all__ = ["assess_optimization"]


def _scenario_rates(results: dict[str, Any]) -> list[dict[str, Any]]:
    proposed = LABELS[1]
    rows: list[dict[str, Any]] = []
    nominal = next(row for row in results["summary"] if row["Policy"] == proposed)
    rows.append({"scenario": "nominal", "condition": 1.0,
                 "success_rate": float(nominal["SuccessRate"]),
                 "episodes": int(nominal["Episodes"])})
    sources = (
        ("wind", "wind_sweep", "WindScale"),
        ("pad_motion", "pad_sweep", "PadScale"),
        ("gnss", "gnss_sweep", "GnssScale"),
        ("battery", "battery_bins", "ReserveLowS"),
    )
    for name, key, condition_key in sources:
        for row in results.get(key, []):
            rate = float(row.get("SuccessRate", float("nan")))
            episodes = int(row.get("Episodes", 0))
            # Sweep rows use a common configured n and omit the Episodes field.
            if name != "battery":
                episodes = max(episodes, 1)
            if row.get("Policy") == proposed and math.isfinite(rate) and episodes > 0:
                rows.append({"scenario": name,
                             "condition": float(row[condition_key]),
                             "success_rate": rate, "episodes": episodes})
    return rows


def assess_optimization(results: dict[str, Any], rgat_history: dict[str, Any],
                        reward_design, cfg: Config) -> dict[str, Any]:
    """Evaluate reward effectiveness and cross-environment consistency separately."""
    limits = cfg.eval.acceptance
    proposed = next(row for row in results["summary"] if row["Policy"] == LABELS[1])
    success_rate = float(proposed["SuccessRate"])
    scenarios = _scenario_rates(results)
    rates = np.asarray([row["success_rate"] for row in scenarios], dtype=float)
    success_std = float(rates.std(ddof=0)) if rates.size else float("nan")
    success_range = float(rates.max() - rates.min()) if rates.size else float("nan")
    worst = float(rates.min()) if rates.size else float("nan")
    val_history = list(rgat_history.get("val_loss", []))
    val_mse = float(val_history[-1]) if val_history else float("inf")

    reward_pass = success_rate >= float(limits.min_success_rate)
    consistency_pass = (
        math.isfinite(success_std)
        and success_std <= float(limits.max_success_std)
        and worst >= float(limits.min_worst_case_success)
        and val_mse <= float(limits.max_rgat_val_mse)
    )
    return {
        "format": "ontology_rgat.optimization_acceptance/1",
        "reward_design_id": reward_design.design_id,
        "reward_optimization": {
            "metric": "nominal deterministic landing success rate",
            "success_rate": success_rate,
            "success_count": int(proposed["SuccessCount"]),
            "episodes": int(proposed["Episodes"]),
            "ci95": [float(proposed["SuccessCI95Low"]),
                     float(proposed["SuccessCI95High"])],
            "minimum": float(limits.min_success_rate),
            "pass": bool(reward_pass),
        },
        "rgat_consistency": {
            "metric": "success-rate dispersion across physical environment strata",
            "scenario_count": int(rates.size),
            "mean": float(rates.mean()) if rates.size else float("nan"),
            "std": success_std,
            "range": success_range,
            "worst_case": worst,
            "max_std": float(limits.max_success_std),
            "min_worst_case": float(limits.min_worst_case_success),
            "rgat_val_mse": val_mse,
            "max_rgat_val_mse": float(limits.max_rgat_val_mse),
            "pass": bool(consistency_pass),
        },
        "overall_pass": bool(reward_pass and consistency_pass),
        "scenarios": scenarios,
    }
