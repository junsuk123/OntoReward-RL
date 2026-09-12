"""Paired evaluation, generalization sweeps and the publication figures."""
from __future__ import annotations

from .compare import compare_policies, evaluate_policy
from .acceptance import assess_optimization
from .plots import make_plots
from .sweeps import battery_sweep, pad_sweep, wind_sweep

__all__ = ["assess_optimization", "battery_sweep", "compare_policies", "evaluate_policy", "make_plots",
           "pad_sweep", "wind_sweep"]
from .three_pipeline import (PHYSICAL_METRICS, paired_confidence_intervals,
                             physical_summary, write_three_pipeline_outputs)

__all__ = ["PHYSICAL_METRICS", "paired_confidence_intervals",
           "physical_summary", "write_three_pipeline_outputs"]
