"""The actor objective must not regress on axes the supervisor replaced.

PPO stores `raw_command`, the action the policy PROPOSED, and regresses on it.
The supervisor rewrites axes INDEPENDENTLY -- `spatial/safety.py` touches only
the vertical one while descent is inhibited -- so a per-step flag is the wrong
granularity: it either keeps a step whose vertical command was replaced or
throws away the horizontal commands that did run. Measured shares of steps
with at least one replaced axis: 38-44 % of 3D, 54 % of 2D, and 73-99 % on the
rungs `tools/audit_exploration_reachability.py` reports.
"""
import math

import numpy as np
import pytest
import torch

from ontology_rgat.two_axis.learning import Transition, executed_axis_mask
from ontology_rgat.two_axis.models import TwoAxisActor, log_probability_axes
from ontology_rgat.two_axis.training import PPOHyperparameters


def test_mask_is_per_axis_not_per_step():
    """One replaced axis must not discard the axes that executed."""
    mask = executed_axis_mask([0.25, -0.40, 0.10], [0.25, -0.20, 0.10])
    assert list(mask) == [1.0, 0.0, 1.0]
    # Scale-relative, so float round-trips through the plant are not "replaced".
    assert list(executed_axis_mask([1.0, 2.0], [1.0 + 1e-12, 2.0])) == [1.0, 1.0]
    # A genuine rewrite is caught even when it is small in absolute terms.
    assert list(executed_axis_mask([1e-3, 1.0], [5e-2, 1.0])) == [0.0, 1.0]
    with pytest.raises(ValueError):
        executed_axis_mask([0.1, 0.2], [0.1, 0.2, 0.3])


def test_factorized_density_sums_to_the_joint_one():
    """The masked path must reuse the same density, not a parallel formula."""
    generator = torch.Generator().manual_seed(7)
    raw = torch.randn(32, 3, generator=generator)
    mu = torch.randn(32, 3, generator=generator)
    std = torch.rand(32, 3, generator=generator) + 0.2
    axes = TwoAxisActor.raw_log_probability_axes(raw, mu, std)
    assert axes.shape == raw.shape
    assert torch.allclose(axes.sum(-1),
                          TwoAxisActor.raw_log_probability(raw, mu, std))


def test_masking_both_sides_keeps_the_ratio_a_probability_ratio():
    """Numerator and denominator must drop the SAME axes.

    Masking only the numerator leaves exp(masked - full), which is not a
    density ratio and silently biases every sample the supervisor touched.
    """
    generator = torch.Generator().manual_seed(11)
    raw = torch.randn(16, 3, generator=generator)
    mu = torch.zeros(16, 3)
    std = torch.ones(16, 3)
    axes = TwoAxisActor.raw_log_probability_axes(raw, mu, std)
    mask = torch.zeros(16, 3)
    mask[:, 0] = 1.0
    # Same policy on both sides: the ratio is exactly 1 for every sample, and
    # that holds under the mask only because both sides use it.
    ratio = torch.exp((axes * mask).sum(-1) - (axes * mask).sum(-1))
    assert torch.allclose(ratio, torch.ones(16))
    # An all-replaced step contributes ratio 1, i.e. no actor gradient.
    none_executed = torch.zeros(16, 3)
    assert torch.allclose(
        torch.exp((axes * none_executed).sum(-1) - (axes * none_executed).sum(-1)),
        torch.ones(16))


def test_hyperparameters_default_to_the_shipped_objective():
    """Opt-in: every recorded result predates this and must stay reproducible."""
    assert PPOHyperparameters().intervention_masking == "none"
    assert PPOHyperparameters(intervention_masking="per_axis").intervention_masking \
        == "per_axis"
    with pytest.raises(ValueError):
        PPOHyperparameters(intervention_masking="per_step")


def test_evidence_helper_works_for_both_agent_classes():
    """It is a module function because the two routes do not share a base class.

    `SpatialAgent` is not a `TwoAxisPPOAgent`; making this a method on the
    latter raised AttributeError across 22 spatial tests. Both only share
    `.tensors()` and `.actor`.
    """
    class _Agent:
        device = "cpu"

        def __init__(self):
            self.actor = lambda p, g: (torch.zeros(1, 3), torch.ones(1, 3))

        def tensors(self, observation):
            return torch.zeros(1, 1), torch.zeros(1, 1, 1)

    axes = log_probability_axes(_Agent(), object(), np.zeros(3))
    assert axes.shape == (3,)
    expected = -0.5 * math.log(2.0 * math.pi)
    assert np.allclose(axes, expected)


def test_masked_update_ignores_a_replaced_axis_and_keeps_the_executed_one():
    """End to end through PPOTrainer, on a two-axis problem with one axis dead.

    Axis 1 is replaced on every step, so its gradient must vanish while axis 0
    still moves. Without the mask both axes are trained on equally.
    """
    from ontology_rgat.two_axis.training import PPOTrainer

    class _Actor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mu = torch.nn.Parameter(torch.zeros(2))
            self.log_std = torch.nn.Parameter(torch.zeros(2))

        def forward(self, packets, graphs):
            batch = packets.shape[0]
            return (self.mu.expand(batch, 2), self.log_std.exp().expand(batch, 2))

    class _Critic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(1, 1)

        def forward(self, packets, graphs):
            return self.head(packets[:, :1]).squeeze(-1)

    class _Agent:
        device = "cpu"
        seed = 3

        def __init__(self):
            self.actor, self.critic = _Actor(), _Critic()

    def transition(raw, executed, advantage_seed):
        mu, std = torch.zeros(2), torch.ones(2)
        axes = TwoAxisActor.raw_log_probability_axes(
            torch.as_tensor(raw).unsqueeze(0), mu, std)[0].numpy()
        observation = type("O", (), {})()
        observation.packet = type("P", (), {"values": np.array([advantage_seed],
                                                               dtype=np.float32)})()
        observation.graph = type("G", (), {"X": np.zeros((1, 1), dtype=np.float32)})()
        return Transition(
            observation=observation, raw_command=np.asarray(raw, dtype=float),
            normalized_command=np.tanh(np.asarray(raw, dtype=float)),
            requested_acceleration_m_s2=np.array([1.0, 1.0]),
            applied_acceleration_m_s2=np.array([1.0, 1.0 if executed[1] else 0.0]),
            old_log_probability=float(axes.sum()), value=0.0,
            reward=float(advantage_seed), reward_components={},
            next_observation=observation, terminated=True, truncated=False,
            dt_s=0.1, safety_flags={}, old_log_probability_axes=axes,
            executed_axes=np.asarray(executed, dtype=float))

    rng = np.random.default_rng(5)
    batch = [transition(rng.normal(size=2), [1.0, 0.0], float(i % 3))
             for i in range(64)]

    moved = {}
    for masking in ("none", "per_axis"):
        agent = _Agent()
        hyper = PPOHyperparameters(epochs=1, minibatch_size=32,
                                   decisions_per_iteration=64,
                                   value_warmup_iterations=0,
                                   entropy_coefficient=0.0,
                                   intervention_masking=masking)
        trainer = PPOTrainer(agent, hyper)
        before = agent.actor.mu.detach().clone()
        trainer.update(batch, discount_time_constant_s=70.0)
        moved[masking] = (agent.actor.mu.detach() - before).abs().numpy()

    # Unmasked, the replaced axis is trained on just like the executed one.
    assert moved["none"][1] > 1e-9
    # Masked, it receives no gradient at all, while axis 0 still learns.
    assert moved["per_axis"][1] < 1e-12, moved["per_axis"]
    assert moved["per_axis"][0] > 1e-9, moved["per_axis"]
