"""PPO in latent-Gaussian coordinates for the direct-policy port."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from .models import DirectActorCritic


@dataclass
class RolloutBatch:
    policy_input: torch.Tensor
    latent_action: torch.Tensor
    old_log_probability: torch.Tensor
    reward: torch.Tensor
    value: torch.Tensor
    discount: torch.Tensor
    terminated: torch.Tensor
    episode_end: torch.Tensor
    bootstrap_value: torch.Tensor | None = None
    policy_version: int = 0

    def validate(self) -> None:
        count = self.reward.shape[0]
        for name in ("policy_input", "latent_action", "old_log_probability", "value",
                     "discount", "terminated", "episode_end"):
            if getattr(self, name).shape[0] != count:
                raise ValueError(f"rollout field {name} has inconsistent length")


def generalized_advantage(batch: RolloutBatch, gae_lambda: float):
    """GAE with terminal vs rollout truncation kept distinct."""
    batch.validate()
    reward = batch.reward.detach().cpu().numpy()
    value = batch.value.detach().cpu().numpy()
    discount = batch.discount.detach().cpu().numpy()
    terminated = batch.terminated.detach().cpu().numpy().astype(bool)
    episode_end = batch.episode_end.detach().cpu().numpy().astype(bool)
    bootstrap = (np.zeros_like(value) if batch.bootstrap_value is None
                 else batch.bootstrap_value.detach().cpu().numpy())
    advantage = np.zeros_like(reward, dtype=np.float32)
    running = 0.0
    for index in range(len(reward)-1, -1, -1):
        if terminated[index]:
            next_value, continuation = 0.0, 0.0
        elif episode_end[index] or index == len(reward)-1:
            next_value, continuation = float(bootstrap[index]), 0.0
        else:
            next_value, continuation = float(value[index+1]), 1.0
        delta = reward[index] + discount[index] * next_value - value[index]
        running = delta + discount[index] * gae_lambda * continuation * running
        advantage[index] = running
    return torch.from_numpy(advantage), torch.from_numpy(advantage + value)


class DirectPPO:
    def __init__(self, model: DirectActorCritic):
        self.model = model
        spec = model.contract.training
        actor_groups = [
            {"params": model.actor.parameters(), "lr": spec.policy_lr},
            {"params": model.actor_encoder.parameters(), "lr": spec.encoder_lr},
            {"params": [model.log_std], "lr": spec.policy_lr},
        ]
        critic_groups = [
            {"params": model.critic.parameters(), "lr": spec.value_lr},
            {"params": model.critic_encoder.parameters(), "lr": spec.encoder_lr},
        ]
        self.actor_optimizer = torch.optim.Adam(actor_groups)
        self.critic_optimizer = torch.optim.Adam(critic_groups)
        self.policy_version = 0

    def update(self, batch: RolloutBatch, *, generator: torch.Generator | None = None):
        cfg = self.model.contract.training
        if batch.policy_version != self.policy_version:
            raise ValueError("rollout was collected by a different policy version")
        advantage, target = generalized_advantage(batch, cfg.gae_lambda)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        count = len(advantage)
        actor_losses, critic_losses, entropies = [], [], []
        for _ in range(cfg.epochs):
            order = torch.randperm(count, generator=generator)
            for start in range(0, count, cfg.minibatch_size):
                index = order[start:start+cfg.minibatch_size]
                if not len(index):
                    continue
                output = self.model(batch.policy_input[index])
                logp = self.model.latent_log_prob(
                    batch.latent_action[index], output.latent_mean, output.log_std)
                ratio = torch.exp(logp-batch.old_log_probability[index])
                unclipped = ratio * advantage[index]
                clipped = ratio.clamp(1-cfg.clip_ratio, 1+cfg.clip_ratio) * advantage[index]
                entropy = (output.log_std + 0.5*math.log(2*math.pi*math.e)).sum(-1).mean()
                actor_loss = -torch.minimum(unclipped, clipped).mean() \
                    - cfg.entropy_weight * entropy
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.actor.parameters())
                    + list(self.model.actor_encoder.parameters()) + [self.model.log_std], 1.0)
                self.actor_optimizer.step()
                with torch.no_grad():
                    self.model.log_std.clamp_(min=cfg.minimum_log_std)

                # Recompute because the actor and critic have independent encoders.
                prediction = self.model(batch.policy_input[index]).value
                critic_loss = ((prediction-target[index])**2).mean()
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.critic.parameters())
                    + list(self.model.critic_encoder.parameters()), 1.0)
                self.critic_optimizer.step()
                actor_losses.append(float(actor_loss.detach()))
                critic_losses.append(float(critic_loss.detach()))
                entropies.append(float(entropy.detach()))
        self.policy_version += 1
        return {"policy_version": self.policy_version, "samples": count,
                "actor_loss": float(np.mean(actor_losses)),
                "critic_loss": float(np.mean(critic_losses)),
                "entropy": float(np.mean(entropies)),
                "actor_encoder_grad": _gradient_norm(self.model.actor_encoder),
                "critic_encoder_grad": _gradient_norm(self.model.critic_encoder)}

    def optimizer_state(self):
        return {"actor": self.actor_optimizer.state_dict(),
                "critic": self.critic_optimizer.state_dict()}

    def load_optimizer_state(self, state, *, policy_version: int):
        if set(state) != {"actor", "critic"}:
            raise ValueError("checkpoint optimizer state is incomplete")
        self.actor_optimizer.load_state_dict(state["actor"])
        self.critic_optimizer.load_state_dict(state["critic"])
        self.policy_version = int(policy_version)


def _gradient_norm(module):
    return math.sqrt(sum(float(torch.sum(parameter.grad.detach()**2))
                         for parameter in module.parameters()
                         if parameter.grad is not None))
