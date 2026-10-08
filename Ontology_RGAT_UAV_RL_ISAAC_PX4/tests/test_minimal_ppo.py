"""Guards for the minimal-contract PPO (python/ontology_rgat/minimal/ppo.py)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from ontology_rgat.minimal.arms import build_arm
from ontology_rgat.minimal.ppo import MinimalPPO, MinimalPPOConfig, gae


def test_gae_ends_on_a_scored_terminal():
    adv, ret = gae([1.0, 0.0, 10.0], [0.5, 0.5, 0.5], gamma=0.9, lam=1.0)
    # lambda 1: return = discounted sum of rewards, no bootstrap after the end
    np.testing.assert_allclose(ret, [1 + 0.81 * 10, 0.9 * 10, 10.0], rtol=1e-6)
    np.testing.assert_allclose(adv, ret - 0.5, rtol=1e-6)


def _episode(arm, n=40, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, arm.body[0].in_features if hasattr(arm, "body") else 1)).astype(np.float32)
    with torch.no_grad():
        mean, log_std, value = arm(torch.as_tensor(x))
    act = (mean + 0.1 * torch.randn_like(mean)).numpy()
    std = log_std.exp().numpy()
    logp = -0.5 * ((act - mean.numpy()) / std) ** 2 - np.log(std) - 0.5 * np.log(2 * np.pi)
    return {"seed": 0, "status": "SUCCESS", "steps": [
        (x[i], act[i], logp[i].astype(np.float32), float(value[i]), float(rng.normal()),
         np.ones(3, np.float32)) for i in range(n)]}


def test_critic_warmup_does_not_move_the_actor():
    arm = build_arm("ppo_vector_canonical", seed=1)
    trainer = MinimalPPO("ppo_vector_canonical", arm,
                         MinimalPPOConfig(critic_warmup_iterations=1, minibatch_size=16))
    episode = _episode(arm)
    x = torch.as_tensor(np.stack([s[0] for s in episode["steps"]]))
    before = arm(x)[0].detach().clone()
    stats = trainer.update([episode])
    assert stats["warmup"] and stats["accepted"] == 0
    torch.testing.assert_close(arm(x)[0].detach(), before)


def test_actor_steps_respect_the_trust_region():
    arm = build_arm("ppo_vector_canonical", seed=2)
    trainer = MinimalPPO("ppo_vector_canonical", arm,
                         MinimalPPOConfig(critic_warmup_iterations=0, minibatch_size=16,
                                          learning_rate=1e-1))   # deliberately huge
    stats = trainer.update([_episode(arm, seed=3)])
    assert stats["rejected"] >= 1
    assert stats["kl"] <= 1.5 * trainer.cfg.target_kl + 1e-9
    assert trainer.lr < 1e-1                                     # halved after a rejected first step
