"""Recurrent visual PPO backbone for the controlled Shin benchmark."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn

from ..benchmarks.shin2026 import ActorObservation
from ..estimation import LSTMRelativeStateEstimator
from ..perception import ShinKeypointEncoder, grayscale_image_tensor


LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


@dataclass
class RecurrentOutput:
    action_mean: torch.Tensor
    action_std: torch.Tensor
    value: torch.Tensor | None
    relative_state: torch.Tensor
    latent: torch.Tensor
    hidden: tuple[torch.Tensor, torch.Tensor]


class ShinRecurrentActorCritic(nn.Module):
    """Paper-compatible information flow with an asymmetric training critic.

    The actor head receives only ``y[6:256]`` and UAV proprioception.  True
    relative state is accepted solely by the separately evaluated critic head.
    """

    def __init__(self, image_embedding=512, lstm_hidden=512, latent_dim=256,
                 actor_hidden=256, critic_hidden=256, action_dim=4,
                 init_log_std=-1.5, freeze_keypoint=False):
        super().__init__()
        if action_dim != 4:
            raise ValueError("benchmark action dimension must be four")
        self.encoder = ShinKeypointEncoder(image_embedding, keypoints=6)
        if freeze_keypoint:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)
        self.estimator = LSTMRelativeStateEstimator(
            image_embedding=image_embedding, proprioception=7,
            hidden_size=lstm_hidden, latent_size=latent_dim, output_size=6)
        self.actor = nn.Sequential(
            nn.Linear(latent_dim - 6 + 7, actor_hidden), nn.Tanh(),
            nn.Linear(actor_hidden, actor_hidden), nn.Tanh(),
            nn.Linear(actor_hidden, action_dim))
        # Section III-D: o_priv=[u_t, s_rel_t], exactly 7+6 values.
        self.critic = nn.Sequential(
            nn.Linear(13, critic_hidden), nn.Tanh(),
            nn.Linear(critic_hidden, critic_hidden), nn.Tanh(),
            nn.Linear(critic_hidden, 1))
        self.log_std = nn.Parameter(torch.full((action_dim,), float(init_log_std)))
        self.action_dim = action_dim

    @property
    def device(self):
        return self.log_std.device

    def initial_state(self, batch_size: int):
        return self.estimator.initial_state(
            batch_size, device=self.device, dtype=self.log_std.dtype)

    def forward(self, images: torch.Tensor, proprioception: torch.Tensor,
                *, true_relative_state: torch.Tensor | None = None,
                hidden=None, episode_start: torch.Tensor | None = None) -> RecurrentOutput:
        if images.ndim == 4:
            images = images[:, None]
        if proprioception.ndim == 2:
            proprioception = proprioception[:, None]
        if images.ndim != 5 or images.shape[2] != 1:
            raise ValueError("images must have shape BxTx1xHxW")
        batch, steps = images.shape[:2]
        encoded = self.encoder(images.reshape(batch * steps, *images.shape[2:])).embedding
        encoded = encoded.reshape(batch, steps, -1)
        estimate = self.estimator(encoded, proprioception, hidden, episode_start)
        actor_features = torch.cat((estimate.latent[..., 6:], proprioception), -1)
        mean = self.actor(actor_features)
        std = self.log_std.exp().expand_as(mean)
        value = None
        if true_relative_state is not None:
            if true_relative_state.ndim == 2:
                true_relative_state = true_relative_state[:, None]
            if true_relative_state.shape != (batch, steps, 6):
                raise ValueError("critic truth must have shape BxTx6")
            value = self.critic(torch.cat((proprioception, true_relative_state), -1)).squeeze(-1)
        return RecurrentOutput(mean, std, value, estimate.relative_state,
                               estimate.latent, estimate.hidden)

    @staticmethod
    def log_prob(pre_squash, action, mean, std):
        gaussian = -0.5 * ((pre_squash - mean) / std) ** 2 - std.log() - LOG_SQRT_2PI
        return (gaussian - torch.log(1.0 - action.square() + 1e-6)).sum(-1)


class StatefulShinPolicy:
    """Deployment wrapper; it has no API through which critic truth can enter."""

    def __init__(self, model: ShinRecurrentActorCritic, seed: int = 0):
        self.model = model
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.hidden = None

    def reset(self) -> None:
        self.hidden = None

    @torch.no_grad()
    def action(self, observation: ActorObservation, deterministic=False):
        image = grayscale_image_tensor(observation.image, device=self.model.device)
        proprio = torch.as_tensor(observation.proprioception[None],
                                  dtype=torch.float32, device=self.model.device)
        result = self.model(image, proprio, hidden=self.hidden)
        self.hidden = tuple(value.detach() for value in result.hidden)
        mean = result.action_mean[:, -1]
        if deterministic:
            pre_squash = mean
        else:
            noise = torch.randn(mean.shape, generator=self.generator)
            pre_squash = mean + result.action_std[:, -1] * noise.to(mean.device)
        return torch.tanh(pre_squash).cpu().numpy()[0]


def recurrent_ppo_loss(model: ShinRecurrentActorCritic, batch: dict,
                       *, clip=0.2, value_coef=0.5, entropy_coef=0.003,
                       auxiliary_coef=1.0):
    """One differentiable recurrent minibatch objective and named telemetry."""
    output = model(batch["images"], batch["proprioception"],
                   true_relative_state=batch["true_relative_state"],
                   hidden=batch.get("initial_hidden"),
                   episode_start=batch.get("episode_start"))
    log_prob = model.log_prob(batch["pre_squash_action"], batch["action"],
                              output.action_mean, output.action_std)
    ratio = (log_prob - batch["old_log_prob"]).exp()
    advantage = batch["advantage"]
    surrogate = torch.minimum(ratio * advantage,
                              ratio.clamp(1.0 - clip, 1.0 + clip) * advantage)
    entropy = torch.log(output.action_std * math.sqrt(2.0 * math.pi * math.e)).sum(-1).mean()
    policy_loss = -surrogate.mean()
    value_loss = (output.value - batch["return"]).square().mean()
    auxiliary_loss = model.estimator.auxiliary_loss(
        output.relative_state, batch["true_relative_state"], batch.get("truth_valid"))
    loss = (policy_loss + value_coef * value_loss - entropy_coef * entropy
            + auxiliary_coef * auxiliary_loss)
    with torch.no_grad():
        approximate_kl = (batch["old_log_prob"] - log_prob).mean()
    metrics = {
        "loss": loss.detach(), "ppo_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(), "entropy": entropy.detach(),
        "kl_divergence": approximate_kl.detach(),
        "auxiliary_estimation_loss": auxiliary_loss.detach(),
    }
    return loss, metrics
