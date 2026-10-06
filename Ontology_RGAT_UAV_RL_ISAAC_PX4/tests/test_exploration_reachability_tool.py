"""The read-only audit that separates "the task is hard" from "exploration never gets there".

A ceiling measured with a tuned controller is a statement about the task. The
quantity PPO actually depends on is whether the distribution the policy samples
from can reach a landing at all, and the two came apart badly on 2026-10-05:
838 training episodes at the easiest curriculum rung produced zero SUCCESS on a
rung where a fixed open-loop descent lands every seed.
"""
from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "audit_exploration_reachability", ROOT / "tools/audit_exploration_reachability.py")
audit_tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit_tool)

from ontology_rgat.spatial.core import SpatialConfig, REFERENCE_SCHEMA


@pytest.fixture(scope="module")
def report():
    cfg = replace(SpatialConfig(), schema=REFERENCE_SCHEMA)
    return audit_tool.audit(cfg, seeds=(5001, 5002, 5003), difficulties=(0.0,),
                            arm="ppo_vector_canonical", policy_seed=828,
                            descent=-0.12)


def test_the_audit_records_its_contract_and_reads_only(report):
    assert report["audit"] == "exploration-reachability/1"
    assert report["contract"]["schema"] == REFERENCE_SCHEMA
    assert report["open_loop_vertical_action"] == -0.12


def test_the_easiest_rung_is_reachable_by_a_fixed_descent(report):
    """If this ever fails the rung itself has become unsolvable, which is a
    different and much worse problem than the one the audit exists to show."""
    rung = report["rungs"]["difficulty_0.0"]["open_loop_descent"]
    assert rung["outcomes"].get("SUCCESS", 0) == rung["episodes"]


def test_the_audit_separates_requested_from_applied_vertical_acceleration(report):
    """The supervisors rewrite only NEGATIVE vertical commands while descent is
    inhibited, so these two means are not interchangeable and the audit must
    report both. Recording the sign relation, not asserting a fixed value."""
    sampled = report["rungs"]["difficulty_0.0"]["sampled_policy"]
    assert "requested_vertical_mean_m_s2" in sampled
    assert "applied_vertical_mean_m_s2" in sampled
    assert 0.0 <= sampled["overridden_step_fraction"] <= 1.0
    assert sampled["episodes"] == 3
