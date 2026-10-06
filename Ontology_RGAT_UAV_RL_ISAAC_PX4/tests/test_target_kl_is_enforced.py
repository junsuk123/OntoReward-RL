"""`target_kl` must bound the policy that each actor step actually produces.

The shipped early stop measures `approx_kl` from the ratio BEFORE a
minibatch's own step, so the first actor step of every iteration is taken
unchecked. Its size in action space is set by `actor_lr`, not by sigma: once
sigma is small (the anneal cells) one Adam step moves the mean by several
sigmas. Measured 2026-10-06 with sigma pinned at 0.030 from iteration 5
(results/full_pipeline_20261006_lowsigma_unconstrained): per-iteration KL
0.5-3.4 against the 0.02 target, and both graph arms went from a 62.5 %-landing
clone to 12/12 SAFE_ABORT at 118 steps within two iterations. Pinning sigma
alone therefore cannot work; the trust region has to be enforced where it
applies.
"""
import numpy as np
import pytest
import torch

from ontology_rgat.two_axis.learning import Transition
from ontology_rgat.two_axis.models import TwoAxisActor
from ontology_rgat.two_axis.training import PPOHyperparameters, PPOTrainer


class _Actor(torch.nn.Module):
    """A mean with a large lever: one Adam step moves it by about `lr * fan`."""

    def __init__(self, sigma, fan=4096):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.zeros(fan, 2))
        self.log_std = torch.nn.Parameter(torch.full((2,), float(np.log(sigma))),
                                          requires_grad=False)

    def forward(self, packets, graphs):
        batch = packets.shape[0]
        mu = self.weights.sum(0).expand(batch, 2)
        return mu, self.log_std.exp().expand(batch, 2)


class _Critic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(1, 1)

    def forward(self, packets, graphs):
        return self.head(packets[:, :1]).squeeze(-1)


class _Agent:
    device = "cpu"
    seed = 3

    def __init__(self, sigma, fan=4096):
        self.actor, self.critic = _Actor(sigma, fan), _Critic()

    def mean(self):
        return self.actor.weights.sum(0).detach().clone()


def _batch(agent, size=64, seed=5):
    """A rollout: proposals sampled from the actor's CURRENT policy."""
    rng = np.random.default_rng(seed)
    mu = agent.mean()
    sigma = float(agent.actor.log_std.exp()[0])
    std = torch.full((2,), sigma)
    rows = []
    for i in range(size):
        raw = mu.numpy() + rng.normal(size=2) * sigma
        logp = float(TwoAxisActor.raw_log_probability(
            torch.as_tensor(raw, dtype=torch.float32).unsqueeze(0), mu, std)[0])
        observation = type("O", (), {})()
        observation.packet = type("P", (), {"values": np.array([float(i % 3)],
                                                               dtype=np.float32)})()
        observation.graph = type("G", (), {"X": np.zeros((1, 1), dtype=np.float32)})()
        rows.append(Transition(
            observation=observation, raw_command=np.asarray(raw, dtype=float),
            normalized_command=np.tanh(np.asarray(raw, dtype=float)),
            requested_acceleration_m_s2=np.ones(2), applied_acceleration_m_s2=np.ones(2),
            old_log_probability=logp, value=0.0, reward=float(raw[0] > 0),
            reward_components={}, next_observation=observation, terminated=True,
            truncated=False, dt_s=0.1, safety_flags={}))
    return rows


def _kl_after(agent, sampled_mean):
    """Exact KL between the sampling policy and the updated one."""
    sigma = agent.actor.log_std.exp().detach()
    return float((0.5 * ((agent.mean() - sampled_mean) / sigma) ** 2).sum())


def _hyper(**overrides):
    settings = dict(epochs=4, minibatch_size=32, decisions_per_iteration=64,
                    value_warmup_iterations=0, entropy_coefficient=0.0,
                    actor_lr=5e-4, target_kl=0.02)
    settings.update(overrides)
    return PPOHyperparameters(**settings)


def test_the_shipped_early_stop_lets_the_first_step_through_at_small_sigma():
    """The defect, pinned so it cannot quietly come back as 'the default'."""
    agent = _Agent(0.03)
    trainer = PPOTrainer(agent, _hyper())
    before = agent.mean()
    metrics = trainer.update(_batch(agent), discount_time_constant_s=70.0)
    assert metrics["policy_updated"] == 1.0
    # The step was taken, and the policy it produced is far outside the
    # trust region the hyperparameters declare.
    assert _kl_after(agent, before) > 10 * 0.02
    assert "rejected_steps" in metrics and metrics["rejected_steps"] == 0.0
    assert metrics["actor_lr"] == pytest.approx(5e-4)


def test_enforced_target_kl_reverts_the_step_and_backs_the_rate_off():
    agent = _Agent(0.03)
    trainer = PPOTrainer(agent, _hyper(enforce_target_kl=True))
    before = agent.mean()
    metrics = trainer.update(_batch(agent), discount_time_constant_s=70.0)
    # The first step is several sigmas wide, so it is the one rejected: the
    # iteration accepted nothing and the rate backs off.
    assert metrics["rejected_steps"] == 1.0 and metrics["accepted_steps"] == 0.0
    assert metrics["actor_lr"] == pytest.approx(2.5e-4)
    # A rejected step leaves no trace: weights back at zero, and the optimiser
    # holds no moments from the gradient it never applied.
    assert _kl_after(agent, before) == 0.0
    assert not trainer.actor_optimizer.state_dict()["state"]


def test_enforced_target_kl_converges_to_a_rate_the_region_admits():
    """Repeated updates settle at an lr whose steps are accepted, so the actor
    still learns -- the control is the step size, not a frozen policy."""
    agent = _Agent(0.03)
    trainer = PPOTrainer(agent, _hyper(enforce_target_kl=True))
    accepted, rates = [], []
    for iteration in range(16):
        before = agent.mean()
        metrics = trainer.update(_batch(agent, seed=iteration),
                                 discount_time_constant_s=70.0)
        accepted.append(metrics["accepted_steps"])
        rates.append(metrics["actor_lr"])
        # The trainer bounds the sample estimate on a 32-row minibatch; the
        # exact KL of the whole policy sits within that estimator's noise.
        assert _kl_after(agent, before) <= 2 * 1.5 * 0.02, iteration
    assert any(accepted[8:]), "the rate never found the trust region"
    # Once a step is accepted, a later rejection is the ordinary early stop:
    # the rate must hold there instead of halving every iteration to the floor.
    first = next(i for i, count in enumerate(accepted) if count)
    assert all(accepted[first:]), accepted
    assert all(b >= a for a, b in zip(rates[first:], rates[first + 1:])), rates
    assert trainer.actor_lr == pytest.approx(5e-4 / 2 ** first)
    # Reward favours raw[0] > 0, so the accepted steps must have moved mu[0]
    # up: enforcement bounds the step, it does not stop learning.
    assert float(agent.actor.weights.sum(0)[0]) > 0.0


def test_enforcement_grows_the_rate_back_but_never_past_the_configured_one():
    """A tiny lr after a burst of rejections must recover once steps are safe."""
    # Wide policy, small lever: the configured lr is well inside the region.
    agent = _Agent(1.0, fan=16)
    trainer = PPOTrainer(agent, _hyper(enforce_target_kl=True))
    trainer._set_actor_lr(1e-6)
    for iteration in range(6):
        trainer.update(_batch(agent, seed=iteration), discount_time_constant_s=70.0)
    assert 1e-6 < trainer.actor_lr <= 5e-4
    trainer._set_actor_lr(5.0)
    assert trainer.actor_lr == pytest.approx(5e-4), "never above actor_lr"


def test_enforcement_is_opt_in_and_identical_when_the_step_is_already_small():
    """Off by default (every recorded run predates it); on, it is a no-op for a
    step the shipped objective would also have accepted."""
    assert PPOHyperparameters().enforce_target_kl is False
    moved = {}
    for flag in (False, True):
        agent = _Agent(1.0, fan=16)
        trainer = PPOTrainer(agent, _hyper(enforce_target_kl=flag, epochs=1))
        metrics = trainer.update(_batch(agent), discount_time_constant_s=70.0)
        moved[flag] = agent.mean()
        assert metrics["rejected_steps"] == 0.0
    assert torch.allclose(moved[False], moved[True])
