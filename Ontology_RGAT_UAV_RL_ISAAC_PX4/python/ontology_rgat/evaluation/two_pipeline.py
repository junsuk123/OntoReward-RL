"""Primary two-pipeline scientific result aggregation."""
from __future__ import annotations

from .three_pipeline import (PHYSICAL_METRICS, learning_efficiency,
                             paired_confidence_intervals, paired_differences,
                             physical_summary, write_three_pipeline_outputs)


def write_two_pipeline_outputs(*args, **kwargs):
    return write_three_pipeline_outputs(*args, **kwargs)


__all__ = ["PHYSICAL_METRICS", "learning_efficiency",
           "paired_confidence_intervals", "paired_differences",
           "physical_summary", "write_two_pipeline_outputs"]
