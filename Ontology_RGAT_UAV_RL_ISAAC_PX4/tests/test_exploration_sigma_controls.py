"""Exploration sigma is the thing that decides whether a batch contains a landing.

Measured 2026-10-06 on the behaviour-cloned checkpoints at difficulty 0.0, the
rung training opens on: evaluated deterministically they land 79-96 %, and
sampled at the shipped `initial_log_std=-1.1` (sigma 0.333) they land 0.0 %.
PPO rollouts are sampled, so no batch contains a success and nothing in the
gradient points at a landing. These are the two controls over that, plus the
clamp interaction that made an earlier measurement of this silently wrong.
"""
import math
from dataclasses import replace

import pytest
import torch

from ontology_rgat.spatial.core import SpatialConfig
from ontology_rgat.spatial.training import (
    SpatialAgent, _graph_planes, _ontology, load_agent, save_agent)
from ontology_rgat.two_axis.models_v28 import ReferenceHead
from ontology_rgat.two_axis.training import PPOHyperparameters


def _head(**kwargs):
    cfg = SpatialConfig()
    return ReferenceHead(
        "ppo_semantic_flat", "actor", cfg.ontology, 5, -1.1,
        packet_dim=len(cfg.packet_fields), action_dim=3, descent_axis=2,
        minimum_log_std=-2.5, graph_planes=_graph_planes(cfg),
        ontology=_ontology(cfg), **kwargs)


def _inputs():
    cfg = SpatialConfig()
    generator = torch.Generator().manual_seed(3)
    return (torch.randn(4, len(cfg.packet_fields), generator=generator),
            torch.randn(4, 2, _ontology(cfg).node_count, 12, generator=generator))


def test_anneal_schedule_is_inert_by_default_and_monotone_when_set():
    assert PPOHyperparameters(iterations=200).log_std_ceiling(50) == 1.0
    hyper = PPOHyperparameters(iterations=200, initial_log_std=-1.1,
                               final_log_std=-3.5, log_std_anneal_fraction=0.5)
    ceilings = [hyper.log_std_ceiling(i) for i in range(1, 201)]
    assert ceilings[0] == pytest.approx(-1.1)
    assert all(b <= a + 1e-12 for a, b in zip(ceilings, ceilings[1:]))
    # Held at the floor for the rest of the run, not overshooting past it.
    assert ceilings[-1] == pytest.approx(-3.5)
    assert min(ceilings) == pytest.approx(-3.5)
    with pytest.raises(ValueError):
        PPOHyperparameters(initial_log_std=-1.1, final_log_std=-0.5)
    with pytest.raises(ValueError):
        PPOHyperparameters(final_log_std=-3.0, log_std_anneal_fraction=0.0)


def test_the_floor_pins_a_low_parameter_but_not_a_low_ceiling():
    """Two different clamp behaviours, one of which invalidated a measurement.

    The spatial actor is built with `minimum_log_std=-2.5` (sigma 0.082) and
    `forward` clamps to it. An earlier sigma ladder wrote -3.86 into the
    PARAMETER and read `log_std.exp()` back, reporting sigma 0.021; the plant
    actually saw 0.082, the same policy as the cell it was meant to be four
    times smaller than, which is why both scored 12.5 %. Reading the parameter
    is not reading the policy.

    The anneal ceiling is the other case: `torch.clamp` returns `max` when
    `min > max`, so a ceiling below the floor IS honoured.
    """
    packets, graphs = _inputs()

    head = _head()
    with torch.no_grad():
        head.log_std.fill_(math.log(0.021))
    assert float(head.log_std.exp().mean()) == pytest.approx(0.021, rel=1e-6)
    assert float(head(packets, graphs)[1].mean()) == pytest.approx(
        math.exp(-2.5), rel=1e-6), "the parameter is not what the plant sees"

    ceiling = _head()
    ceiling.maximum_log_std = -4.0
    assert float(ceiling(packets, graphs)[1].mean()) == pytest.approx(
        math.exp(-4.0), rel=1e-6), "the anneal ceiling must reach below the floor"


def test_state_dependent_sigma_starts_identical_and_then_varies():
    """It must be a strict generalisation, or it is a different experiment."""
    packets, graphs = _inputs()
    fixed, learned = _head(), _head(state_dependent_log_std=True)
    fixed_mu, fixed_std = fixed(packets, graphs)
    learned_mu, learned_std = learned(packets, graphs)
    assert torch.allclose(fixed_mu, learned_mu)
    assert torch.allclose(fixed_std, learned_std)
    # The per-state head is what the global parameter cannot express.
    assert fixed_std.std(0).max() == 0
    with torch.no_grad():
        final = [l for l in learned.log_std_net
                 if isinstance(l, torch.nn.Linear)][-1]
        final.weight.normal_(0.0, 0.5)
    assert learned(packets, graphs)[1].std(0).max() > 1e-6


def test_actor_architecture_travels_with_the_checkpoint(tmp_path):
    """Not in config_sha256: a new config field orphans every checkpoint.

    The state dict gains `log_std_net.*`, so a checkpoint saved from the
    state-dependent actor cannot load into the global-sigma one. The payload
    therefore records the architecture and `load_agent` rebuilds it.
    """
    cfg = SpatialConfig()
    agent = SpatialAgent("ppo_semantic_flat", cfg, 5, -1.1,
                         state_dependent_log_std=True)
    path = tmp_path / "checkpoint.pt"
    save_agent(path, agent, cfg, eligible=True)
    restored, _ = load_agent(path, cfg)
    assert restored.state_dependent_log_std is True
    assert any(key.startswith("actor.log_std_net")
               for key in restored.state_dict())
    # And the default actor is unchanged, so existing checkpoints still load.
    plain = SpatialAgent("ppo_semantic_flat", cfg, 5, -1.1)
    assert plain.state_dependent_log_std is False
    assert not any(key.startswith("actor.log_std_net")
                   for key in plain.state_dict())
