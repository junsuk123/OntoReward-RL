"""The common PPO of guide 8.3: GAE on actual transition times, latent ratio, KL guard."""
import math

import numpy as np
import pytest
import torch

from ontology_rgat.current_semantics.contracts import PPOSettings, load_config
from ontology_rgat.current_semantics.features import build_feature_batch
from ontology_rgat.current_semantics.models import CurrentActorCritic
from ontology_rgat.current_semantics.observation import CurrentPhysical, encode
from ontology_rgat.current_semantics.ppo import (PPOTrainer, build_batch, discounts, gae,
                                                 gaussian_kl)


CONFIG = load_config()
SETTINGS = CONFIG.g4.ppo


def test_discounts_use_the_actual_transition_time():
    np.testing.assert_allclose(discounts([0.1, 0.04, 0.116], 70.0),
                               np.exp(-np.array([0.1, 0.04, 0.116]) / 70.0))
    with pytest.raises(ValueError):
        discounts([0.1, 0.0], 70.0)


def test_gae_matches_the_hand_recursion_terminal_and_truncated():
    r, v, g, lam = np.array([1.0, -2.0, 3.0]), np.array([0.5, 0.2, -0.1]), \
        np.array([0.99, 0.98, 0.97]), 0.9
    for terminal, bootstrap in ((True, 7.0), (False, 7.0)):
        nxt = 0.0 if terminal else bootstrap
        d2 = r[2] + g[2] * nxt - v[2]
        d1 = r[1] + g[1] * v[2] - v[1]
        d0 = r[0] + g[0] * v[1] - v[0]
        a2 = d2; a1 = d1 + g[1] * lam * a2; a0 = d0 + g[0] * lam * a1
        adv, ret = gae(r, v, g, terminal=terminal, bootstrap=bootstrap, lam=lam)
        np.testing.assert_allclose(adv, [a0, a1, a2])
        np.testing.assert_allclose(ret, adv + v)


def test_gaussian_kl_equals_the_torch_distribution_kl():
    gen = torch.Generator().manual_seed(0)
    m0, m1 = torch.randn(5, 3, generator=gen), torch.randn(5, 3, generator=gen)
    s0, s1 = torch.randn(3, generator=gen) * 0.3, torch.randn(3, generator=gen) * 0.3
    ref = torch.distributions.kl_divergence(torch.distributions.Normal(m0, s0.exp()),
                                            torch.distributions.Normal(m1, s1.exp())).sum(-1)
    torch.testing.assert_close(gaussian_kl(m0, s0, m1, s1), ref)


def episode(rng, steps=40):
    obs = []
    for _ in range(steps + 1):
        obs.append(encode(CurrentPhysical(
            ex=rng.uniform(-1, 1), ey=rng.uniform(-1, 1), h=rng.uniform(0.3, 2.5),
            vrx=rng.uniform(-1, 1), vry=rng.uniform(-1, 1), vgx=0.4, vgy=0.0,
            vz=rng.uniform(-0.5, 0.5), roll=0.02, pitch=-0.03, yaw=0.0, p=0.0, q=0.0, r=0.0,
            vision_updated=True, vision_age_s=0.1, nav_valid=True, nav_age_s=0.0,
            estimate_valid=True)))
    return {"observations": np.array(obs), "latents": rng.normal(0, 0.4, (steps, 3)),
            "rewards": rng.normal(0, 1, steps), "dt_s": np.full(steps, 0.1), "terminal": True}


def make_batch(model, seed=0):
    rng = np.random.default_rng(seed)
    return build_batch(model, [episode(rng) for _ in range(3)], settings=SETTINGS,
                       tau_s=70.0, feature_batch=lambda v: build_feature_batch(v, CONFIG))


def test_batch_holds_the_behaviour_policy_and_the_three_views():
    model = CurrentActorCritic("ontology_rgat_ppo", CONFIG, seed=0)
    batch, info = make_batch(model)
    assert len(batch) == 120 and info["episodes"] == 3
    with torch.no_grad():
        mean = model(batch.raw, batch.flat, batch.nodes)
    torch.testing.assert_close(batch.old_mean, mean)
    torch.testing.assert_close(batch.old_log_prob, model._latent_log_prob(batch.latent, mean))


@pytest.mark.parametrize("arm", ["raw_ppo", "ontology_rgat_ppo"])
def test_one_update_moves_the_actor_inside_the_kl_bound_and_fits_the_critic(arm):
    model = CurrentActorCritic(arm, CONFIG, seed=0)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    batch, _ = make_batch(model)
    trainer = PPOTrainer(model, SETTINGS)
    with torch.no_grad():
        loss0 = float(((model.value(batch.raw) - batch.returns) ** 2).mean())
    stats = trainer.update(batch, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        loss1 = float(((model.value(batch.raw) - batch.returns) ** 2).mean())
    assert stats["accepted_steps"] > 0 and stats["max_kl"] <= 1.5 * SETTINGS.target_kl
    assert loss1 < loss0
    changed = {k for k, v in model.state_dict().items() if not torch.equal(v, before[k])}
    assert any(k.startswith("actor_head") for k in changed) and any(k.startswith("critic")
                                                                     for k in changed)
    if arm == "ontology_rgat_ppo":
        assert any(k.startswith("encoder") for k in changed)


def test_a_step_over_the_kl_bound_is_reverted_and_the_rate_backs_off():
    model = CurrentActorCritic("raw_ppo", CONFIG, seed=0)
    batch, _ = make_batch(model)
    tight = PPOSettings(**{**SETTINGS.__dict__, "target_kl": 1e-12, "epochs": 1})
    trainer = PPOTrainer(model, tight)
    actor_before = [p.clone() for p in trainer.actor_params]
    stats = trainer.update(batch, generator=torch.Generator().manual_seed(1))
    assert stats["accepted_steps"] == 0 and stats["rejected_steps"] > 0
    assert all(torch.equal(a, b) for a, b in zip(actor_before, trainer.actor_params))
    assert trainer.actor_lr == pytest.approx(SETTINGS.policy_lr * 0.5)
    assert all(g["lr"] == trainer.actor_lr for g in trainer.actor_opt.param_groups)


def test_entropy_term_is_the_only_extra_actor_objective():
    model = CurrentActorCritic("raw_ppo", CONFIG, seed=0)
    assert math.isclose(float(model.entropy()),
                        3 * (0.5 + 0.5 * math.log(2 * math.pi)) + float(model.log_std.sum()),
                        rel_tol=1e-6)
