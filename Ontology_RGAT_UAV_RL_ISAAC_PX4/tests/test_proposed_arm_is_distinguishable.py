"""The proposed arm must not be the flat baseline wearing a different label.

``ppo_ontology_rgat`` differs from ``ppo_semantic_flat`` by exactly one term,
``relational_delta``. That term's readout is zero-initialised, and
``set_adaptation`` freezes the whole relational path for the first
``ontology.adaptation_warmup_fraction`` of the planned budget. While it is
frozen at zero the two arms are not merely similar -- they are the same
function, bit for bit.

This was not caught for weeks. Comparing per-iteration ``actor_loss`` and
``entropy`` to twelve decimals across the shipped spatial runs on 2026-10-05:

    results/spatial_long_nominal_20261005      8607/8607 iterations identical
    results/spatial_relation_on_20261005       1800/2000 iterations identical
    results/spatial_fixed_curriculum_20261005  1800/2000 iterations identical

The long run declared 20000 iterations, so its warmup landed at 18000 and the
run ended at 8607 -- the relational path was never trainable for a single
update. Every three-arm spatial result recorded before this file measures two
arms and a copy of one of them, and says nothing about the representation the
study exists to test.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from ontology_rgat.spatial.core import REFERENCE_SCHEMA, SpatialConfig
from ontology_rgat.spatial.environment import SpatialLandingEnv
from ontology_rgat.spatial.training import SpatialAgent, relational_health


@pytest.fixture(scope="module")
def config():
    return replace(SpatialConfig(), schema=REFERENCE_SCHEMA)


def _observations(config, count=24):
    env = SpatialLandingEnv(config)
    try:
        observation, _ = env.reset(seed=3001)
        out = []
        for _ in range(count):
            out.append(observation)
            observation, _r, done, _t, _i = env.step(np.zeros(3))
            if done:
                break
        return out
    finally:
        env.close()


def _actions(agent, observations):
    with torch.no_grad():
        return np.stack([agent.act(o, deterministic=True)[1] for o in observations])


def test_at_initialization_the_two_graph_arms_are_deliberately_identical(config):
    """The zero-initialised readout is intentional, so this is the start state,
    not the defect. The defect is STAYING here."""
    observations = _observations(config)
    flat = SpatialAgent("ppo_semantic_flat", config, 828)
    rgat = SpatialAgent("ppo_ontology_rgat", config, 828)
    np.testing.assert_allclose(_actions(flat, observations),
                               _actions(rgat, observations), atol=0, rtol=0)
    assert not relational_health(rgat, config)["active"]


def test_a_nonzero_readout_makes_the_proposed_arm_a_different_function(config):
    """Once the relational path carries signal the arms must diverge. If this
    fails, `relational_delta` is not reaching the action and no amount of
    training will make the comparison meaningful."""
    observations = _observations(config)
    flat = SpatialAgent("ppo_semantic_flat", config, 828)
    rgat = SpatialAgent("ppo_ontology_rgat", config, 828)
    with torch.no_grad():
        readout = rgat.actor.encoder.readout
        readout.weight.normal_(0.0, 0.2, generator=torch.Generator().manual_seed(7))
        readout.bias.normal_(0.0, 0.2, generator=torch.Generator().manual_seed(8))
    difference = np.abs(_actions(rgat, observations) - _actions(flat, observations))
    assert difference.max() > 1e-6, (
        "the relational path does not reach the action; the proposed arm cannot "
        "differ from ppo_semantic_flat no matter how it is trained")
    health = relational_health(rgat, config)
    assert health["active"] and not health["degenerate_to_semantic_flat"]


def test_relational_health_reports_the_other_arms_as_not_applicable(config):
    for mode in ("ppo_vector_canonical", "ppo_semantic_flat"):
        report = relational_health(SpatialAgent(mode, config, 828), config)
        assert report["applicable"] is False
        assert report["mode"] == mode


@pytest.mark.parametrize("fraction,iterations,expected_trainable", [
    (0.90, 2000, 200),     # shipped default: 10 % of the budget
    (0.90, 20000, 2000),   # and a run cut short of 18000 gets none of it
    (0.0, 2000, 2000),
    (0.5, 2000, 1000),
])
def test_the_staged_schedule_is_a_fraction_of_the_PLANNED_budget(
        fraction, iterations, expected_trainable):
    """`train_arm` computes warmup from `hyper.iterations`, not from what the
    run actually completes. Declaring a large budget and stopping early
    therefore silences the relational path entirely, which is exactly what
    happened to spatial_long_nominal."""
    import math

    warmup = min(iterations - 1, math.ceil(fraction * iterations))
    assert iterations - warmup == expected_trainable
