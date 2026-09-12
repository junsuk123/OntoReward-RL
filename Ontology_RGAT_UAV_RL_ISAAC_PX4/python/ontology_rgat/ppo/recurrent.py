"""Recurrent visual PPO backbone for the controlled Shin benchmark."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn

from ..benchmarks.shin2026 import ActorObservation
from ..estimation import RelativeStateAuxiliaryHead
from ..perception import ShinKeypointEncoder, grayscale_image_tensor
from ..pipelines import PipelineSpec, get_pipeline
from .temporal_backbone import TemporalVisualBackbone


LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


@dataclass
class RecurrentOutput:
    action_mean: torch.Tensor
    action_std: torch.Tensor
    value: torch.Tensor | None
    relative_state: torch.Tensor | None
    latent: torch.Tensor
    hidden: tuple[torch.Tensor, torch.Tensor]
    keypoints: torch.Tensor
    heatmaps: torch.Tensor


class PipelineActorCritic(nn.Module):
    """Common recurrent actor/critic with optional Shin auxiliary supervision.

    Every pipeline deploys image + UAV proprioception, an identical keypoint
    encoder/LSTM/latent/actor and the literal ``latent[..., 6:]`` actor slice.
    Only ``shin_se`` constructs ``relative_state_head``.  Critic truth is
    accepted solely by the separately evaluated training critic.
    """

    def __init__(self, image_embedding=512, lstm_hidden=512, latent_dim=256,
                 actor_hidden=256, critic_hidden=256, action_dim=4,
                 init_log_std=-1.5, actor_output_gain=0.01,
                 freeze_keypoint=False,
                 relative_state_scale=(3.0, 3.0, 8.0, 3.0, 3.0, 2.0),
                 pipeline: PipelineSpec | str = "shin_se"):
        super().__init__()
        self.pipeline_spec = (get_pipeline(pipeline) if isinstance(pipeline, str)
                              else pipeline)
        if not isinstance(self.pipeline_spec, PipelineSpec):
            raise TypeError("pipeline must be a PipelineSpec or registered pipeline ID")
        if action_dim != 4:
            raise ValueError("benchmark action dimension must be four")
        self.encoder = ShinKeypointEncoder(image_embedding, keypoints=6)
        if freeze_keypoint:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)
        self.temporal_backbone = TemporalVisualBackbone(
            image_embedding=image_embedding, proprioception=7,
            hidden_size=lstm_hidden, latent_size=latent_dim)
        self.relative_state_head = (
            RelativeStateAuxiliaryHead(relative_state_scale)
            if self.pipeline_spec.state_estimation_enabled else None)
        self.actor = nn.Sequential(
            nn.Linear(latent_dim - self.pipeline_spec.reserved_latent_dimensions + 7,
                      actor_hidden), nn.Tanh(),
            nn.Linear(actor_hidden, actor_hidden), nn.Tanh(),
            nn.Linear(actor_hidden, action_dim))
        output_gain = float(actor_output_gain)
        if not math.isfinite(output_gain) or output_gain < 0.0:
            raise ValueError("actor output gain must be finite and non-negative")
        nn.init.orthogonal_(self.actor[-1].weight, gain=output_gain)
        nn.init.zeros_(self.actor[-1].bias)
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
        return self.temporal_backbone.initial_state(
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
        visual = self.encoder(images.reshape(batch * steps, *images.shape[2:]))
        encoded = visual.embedding.reshape(batch, steps, -1)
        temporal = self.temporal_backbone(
            encoded, proprioception, hidden, episode_start)
        relative_state = (None if self.relative_state_head is None
                          else self.relative_state_head(temporal.latent))
        actor_features = torch.cat(
            (temporal.latent[..., self.pipeline_spec.actor_latent_slice],
             proprioception), -1)
        mean = self.actor(actor_features)
        std = self.log_std.exp().expand_as(mean)
        value = None
        if true_relative_state is not None:
            if true_relative_state.ndim == 2:
                true_relative_state = true_relative_state[:, None]
            if true_relative_state.shape != (batch, steps, 6):
                raise ValueError("critic truth must have shape BxTx6")
            value = self.critic(torch.cat((proprioception, true_relative_state), -1)).squeeze(-1)
        return RecurrentOutput(
            mean, std, value, relative_state, temporal.latent, temporal.hidden,
            visual.keypoints.reshape(batch, steps, 6, 2),
            visual.heatmaps.reshape(batch, steps, 6,
                                    visual.heatmaps.shape[-2],
                                    visual.heatmaps.shape[-1]))

    @staticmethod
    def log_prob(pre_squash, action, mean, std):
        gaussian = -0.5 * ((pre_squash - mean) / std) ** 2 - std.log() - LOG_SQRT_2PI
        return (gaussian - torch.log(1.0 - action.square() + 1e-6)).sum(-1)


class StatefulPipelinePolicy:
    """Deployment wrapper; it has no API through which critic truth can enter."""

    def __init__(self, model: PipelineActorCritic, seed: int = 0):
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


def recurrent_ppo_loss(model: PipelineActorCritic, batch: dict,
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
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    auxiliary_loss = None
    if model.pipeline_spec.auxiliary_estimation_loss_enabled:
        if output.relative_state is None:
            raise RuntimeError("supervised pipeline did not produce a relative-state estimate")
        auxiliary_loss = model.relative_state_head.loss(
            output.relative_state, batch["true_relative_state"],
            batch.get("truth_valid"))
        loss = loss + auxiliary_coef * auxiliary_loss
    with torch.no_grad():
        # Non-negative second-order approximation used by PPO early stopping.
        # A signed mean(old-new) can cancel across samples and hid the policy
        # jumps observed in the live flight run.
        log_ratio = log_prob - batch["old_log_prob"]
        approximate_kl = ((log_ratio.exp() - 1.0) - log_ratio).mean()
    metrics = {
        "loss": loss.detach(), "ppo_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(), "entropy": entropy.detach(),
        "kl_divergence": approximate_kl.detach(),
        "state_estimation_enabled": float(
            model.pipeline_spec.state_estimation_enabled),
        "active_perception_enabled": float(
            model.pipeline_spec.active_perception_enabled),
        "ontology_enabled": float(model.pipeline_spec.ontology_enabled),
    }
    if auxiliary_loss is not None:
        metrics["auxiliary_estimation_loss"] = auxiliary_loss.detach()
    return loss, metrics


class ShinRecurrentActorCritic(PipelineActorCritic):
    """Backward-compatible name for the explicit-supervision Shin pipeline."""

    def __init__(self, *args, pipeline: PipelineSpec | str = "shin_se", **kwargs):
        super().__init__(*args, pipeline=pipeline, **kwargs)


StatefulShinPolicy = StatefulPipelinePolicy
