"""Frozen evaluation ranges stay separate from selection and train collectors."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "spatial_checkpoint_evaluator",
    Path(__file__).resolve().parents[1] / "tools/evaluate_spatial_checkpoint.py")
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


def test_default_evaluation_seeds_are_backward_compatible():
    assert evaluator.evaluation_seeds("validation", "local") == (2000, 2001)
    assert evaluator.evaluation_seeds("test", "local") == (12000, 12001)
    assert evaluator.evaluation_seeds("test", "isaac") == (12000, 12001)


def test_explicit_frozen_local_holdout_is_paired_and_outside_training_ranges():
    assert evaluator.evaluation_seeds("test", "local", 9100, 20) == tuple(range(9100, 9120))


@pytest.mark.parametrize("split,backend,start,count", [
    ("validation", "local", 9100, 2),
    ("validation", "local", None, 20),
    ("test", "local", 2000, 2),
    ("test", "isaac", 9100, 2),
    ("test", "local", 99999, 2),
    ("test", "local", 9100, 0),
    ("test", "local", 9100, 1001),
])
def test_test_range_cannot_mutate_validation_or_enter_training_seed_space(split, backend, start, count):
    with pytest.raises(ValueError):
        evaluator.evaluation_seeds(split, backend, start, count)
