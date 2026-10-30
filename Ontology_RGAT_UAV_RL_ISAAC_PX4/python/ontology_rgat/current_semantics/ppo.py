"""Common PPO of spatial_current_semantics_v1 (guide 8.3), identical for every arm.

The actor objective is the clipped surrogate on the latent Gaussian's ratio
(plus ``entropy_coefficient`` times its entropy); the encoder, actor head and
global log_std learn from it alone.  The common raw critic learns from its own
squared error.  Nothing else enters: no auxiliary loss, prior or teacher.

Two measured lessons of this repository are built in.  Discounts are
``exp(-dt/tau)`` of each ACTUAL transition, so GAE and the task's potential
term share one time base.  And the KL bound is enforced: each actor minibatch
step is re-measured on its own minibatch after the step and reverted (weights
and Adam moments) above 1.5 x ``target_kl``, because measuring before the step
let the first step of every iteration through unbounded (AGENTS.md, the
``--enforce-target-kl`` finding).
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from .contracts import PPOSettings
from .models import CurrentActorCritic


KL_REVERT_FACTOR = 1.5
LR_BACKOFF, LR_GROWTH = 0.5, 1.5


def discounts(dt_s: Any, tau_s: float) -> np.ndarray:
    """Per-transition discount exp(-dt/tau) of the actual transition times."""
    dt = np.asarray(dt_s, dtype=np.float64)
    if np.any(dt <= 0.0) or not np.all(np.isfinite(dt)):
        raise ValueError("transition times must be finite and positive")
    return np.exp(-dt / float(tau_s))


def gae(rewards: Any, values: Any, gammas: Any, *, terminal: bool, bootstrap: float,
        lam: float) -> tuple[np.ndarray, np.ndarray]:
    """(advantages, returns) of ONE episode.

    ``values`` holds V(s_t) for its T decisions; after the last decision the
    episode either ended in a task terminal (no bootstrap) or was cut short
    (``bootstrap`` = V(s_T)).
    """
    r = np.asarray(rewards, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    g = np.asarray(gammas, dtype=np.float64)
    if not (r.shape == v.shape == g.shape) or r.ndim != 1:
        raise ValueError("rewards, values and discounts must be equal-length vectors")
    advantages = np.zeros_like(r)
    following, carry = (0.0 if terminal else float(bootstrap)), 0.0
    for t in range(len(r) - 1, -1, -1):
        delta = r[t] + g[t] * following - v[t]
        carry = delta + g[t] * lam * carry
        advantages[t] = carry
        following = v[t]
    return advantages, advantages + v


def gaussian_kl(old_mean: torch.Tensor, old_log_std: torch.Tensor, new_mean: torch.Tensor,
                new_log_std: torch.Tensor) -> torch.Tensor:
    """KL(old || new) of diagonal Gaussians, summed over axes, per row."""
    old_var, new_var = (2.0 * old_log_std).exp(), (2.0 * new_log_std).exp()
    return (new_log_std - old_log_std
            + (old_var + (old_mean - new_mean) ** 2) / (2.0 * new_var) - 0.5).sum(-1)


@dataclass(frozen=True, eq=False)
class Batch:
    """One update's decisions, all tensors aligned on the first axis."""
    raw: torch.Tensor
    flat: torch.Tensor
    nodes: torch.Tensor
    latent: torch.Tensor
    old_log_prob: torch.Tensor
    old_mean: torch.Tensor
    old_log_std: torch.Tensor
    advantage: torch.Tensor
    returns: torch.Tensor

    def __len__(self) -> int:
        return int(self.raw.shape[0])


class PPOTrainer:
    """Two Adam optimisers (actor: encoder + head + log_std; critic: raw critic)."""

    def __init__(self, model: CurrentActorCritic, settings: PPOSettings):
        self.model, self.settings = model, settings
        groups = {group["name"]: group["params"] for group in model.optimizer_groups()}
        self.actor_params = [p for name in ("encoder", "actor_head", "log_std")
                             for p in groups.get(name, [])]
        self.critic_params = list(groups["critic"])
        self.actor_opt = torch.optim.Adam(self.actor_params, lr=settings.policy_lr)
        self.critic_opt = torch.optim.Adam(self.critic_params, lr=settings.value_lr)
        self.actor_lr = settings.policy_lr

    def _snapshot(self) -> tuple[list[torch.Tensor], dict[str, Any]]:
        return ([p.detach().clone() for p in self.actor_params],
                deepcopy(self.actor_opt.state_dict()))

    def _restore(self, snapshot: tuple[list[torch.Tensor], dict[str, Any]]) -> None:
        weights, state = snapshot
        with torch.no_grad():
            for p, w in zip(self.actor_params, weights):
                p.copy_(w)
        self.actor_opt.load_state_dict(state)

    def update(self, batch: Batch, *, generator: torch.Generator) -> dict[str, Any]:
        s = self.settings
        n = len(batch)
        if n == 0:
            raise ValueError("empty batch")
        advantage = batch.advantage
        advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
        accepted = rejected = 0
        kls, actor_losses, value_losses, clip_fraction = [], [], [], []
        for _ in range(s.epochs):
            order = torch.randperm(n, generator=generator)
            for start in range(0, n, s.minibatch):
                idx = order[start:start + s.minibatch]
                snapshot = self._snapshot()
                mean = self.model(batch.raw[idx], batch.flat[idx], batch.nodes[idx])
                log_prob = self.model._latent_log_prob(batch.latent[idx], mean)
                ratio = (log_prob - batch.old_log_prob[idx]).exp()
                a = advantage[idx]
                surrogate = torch.min(ratio * a, ratio.clamp(1 - s.clip, 1 + s.clip) * a)
                loss = -surrogate.mean() - s.entropy_coefficient * self.model.entropy()
                self.actor_opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_params, s.max_grad_norm)
                self.actor_opt.step()
                with torch.no_grad():
                    after = self.model(batch.raw[idx], batch.flat[idx], batch.nodes[idx])
                    kl = float(gaussian_kl(batch.old_mean[idx], batch.old_log_std,
                                           after, self.model.log_std).mean())
                if kl > KL_REVERT_FACTOR * s.target_kl:
                    self._restore(snapshot)
                    rejected += 1
                else:
                    accepted += 1
                    kls.append(kl)
                    actor_losses.append(float(loss.detach()))
                    clip_fraction.append(float(((ratio - 1).abs() > s.clip).float().mean()))
                value = self.model.value(batch.raw[idx])
                value_loss = ((value - batch.returns[idx]) ** 2).mean()
                self.critic_opt.zero_grad(set_to_none=True)
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_params, s.max_grad_norm)
                self.critic_opt.step()
                value_losses.append(float(value_loss.detach()))
        # Adapt only within [0, configured]: back off when nothing was accepted,
        # recover after a clean iteration well inside the bound.
        if accepted == 0:
            self.actor_lr *= LR_BACKOFF
        elif max(kls) < 0.5 * s.target_kl:
            self.actor_lr = min(self.actor_lr * LR_GROWTH, s.policy_lr)
        for group in self.actor_opt.param_groups:
            group["lr"] = self.actor_lr
        return {"decisions": n, "accepted_steps": accepted, "rejected_steps": rejected,
                "max_kl": max(kls) if kls else None,
                "mean_kl": float(np.mean(kls)) if kls else None,
                "actor_loss": float(np.mean(actor_losses)) if actor_losses else None,
                "value_loss": float(np.mean(value_losses)),
                "clip_fraction": float(np.mean(clip_fraction)) if clip_fraction else None,
                "actor_lr": self.actor_lr,
                "log_std": self.model.log_std.detach().tolist()}


def build_batch(model: CurrentActorCritic, episodes: Sequence[dict[str, Any]], *,
                settings: PPOSettings, tau_s: float, feature_batch: Any) -> tuple[Batch, dict]:
    """Assemble one update from complete episodes recorded by the behaviour policy.

    ``feature_batch(vectors) -> (flat, nodes)`` is ``features.build_feature_batch``
    bound to the config.  Values and the behaviour log-probabilities are taken
    from the CURRENT model, which is the policy that collected these episodes.
    """
    raws, latents, advantages, returns = [], [], [], []
    model.eval()
    for episode in episodes:
        obs = np.asarray(episode["observations"], dtype=np.float64)
        steps = len(episode["rewards"])
        if obs.shape != (steps + 1, 21) or steps == 0:
            raise ValueError("an episode needs T >= 1 rewards and T + 1 observations")
        flat, nodes = feature_batch(obs)
        raw_t = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            values = model.value(raw_t).double().numpy()
        gammas = discounts(episode["dt_s"], tau_s)
        adv, ret = gae(episode["rewards"], values[:-1], gammas,
                       terminal=bool(episode["terminal"]), bootstrap=float(values[-1]),
                       lam=settings.gae_lambda)
        raws.append(obs[:-1]); latents.append(np.asarray(episode["latents"]))
        advantages.append(adv); returns.append(ret)
    raw = np.concatenate(raws)
    flat, nodes = feature_batch(raw)
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
    raw_t, flat_t, nodes_t, latent_t = t(raw), t(flat), t(nodes), t(np.concatenate(latents))
    with torch.no_grad():
        old_mean = model(raw_t, flat_t, nodes_t)
        old_log_prob = model._latent_log_prob(latent_t, old_mean)
    model.train()
    batch = Batch(raw=raw_t, flat=flat_t, nodes=nodes_t, latent=latent_t,
                  old_log_prob=old_log_prob, old_mean=old_mean,
                  old_log_std=model.log_std.detach().clone(),
                  advantage=t(np.concatenate(advantages)), returns=t(np.concatenate(returns)))
    return batch, {"episodes": len(episodes), "decisions": len(batch),
                   "explained_variance": _explained_variance(batch)}


def _explained_variance(batch: Batch) -> float | None:
    """1 - Var(returns - advantage-implied values)/Var(returns): critic fit diagnostic."""
    values = batch.returns - batch.advantage
    variance = float(batch.returns.var(unbiased=False))
    if not math.isfinite(variance) or variance == 0.0:
        return None
    return 1.0 - float((batch.returns - values).var(unbiased=False)) / variance
