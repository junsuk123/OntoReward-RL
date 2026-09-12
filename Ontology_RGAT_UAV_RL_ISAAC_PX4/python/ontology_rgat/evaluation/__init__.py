"""Paired evaluation, generalization sweeps and the publication figures."""
from __future__ import annotations

from .acceptance import assess_optimization
from .compare import compare_policies, evaluate_policy
from .plots import make_plots
from .sweeps import battery_sweep, pad_sweep, wind_sweep
from .three_pipeline import (PHYSICAL_METRICS, paired_confidence_intervals,
                             paired_differences, physical_summary,
                             write_three_pipeline_outputs)

__all__ = [
    "PHYSICAL_METRICS", "assess_optimization", "battery_sweep",
    "compare_policies", "evaluate_policy", "make_plots",
    "paired_confidence_intervals", "paired_differences", "pad_sweep",
    "physical_summary", "wind_sweep", "write_three_pipeline_outputs",
]
