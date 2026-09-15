"""Executable specifications for the controlled two-pipeline experiment."""

from .spec import (ADAPTIVE_PIPELINES, ALL_PIPELINES, FORBIDDEN_PRIMARY_VISION_KEYS,
                   GEOMETRIC_FOV_CRITERION, LEGACY_PIPELINES, PIPELINES,
                   PipelineSpec, assert_no_aruco_in_primary_system,
                   assert_primary_baseline_equivalence,
                   available_pipeline_ids, get_pipeline, primary_pipeline_ids,
                   validate_pipeline_configuration)

__all__ = ["PIPELINES", "LEGACY_PIPELINES", "ADAPTIVE_PIPELINES",
           "ALL_PIPELINES", "FORBIDDEN_PRIMARY_VISION_KEYS",
           "GEOMETRIC_FOV_CRITERION", "PipelineSpec",
           "assert_no_aruco_in_primary_system",
           "assert_primary_baseline_equivalence",
           "available_pipeline_ids", "get_pipeline", "primary_pipeline_ids",
           "validate_pipeline_configuration"]
