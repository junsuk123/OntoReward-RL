"""Recurrent visual PPO backbone for the controlled Shin benchmark."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn

from ..benchmarks.shin2026 import ActorObservation
from ..controllers import PLANAR_ACTION_DIM
from ..estimation import RelativeStateAuxiliaryHead
from ..perception import ShinKeypointEncoder, grayscale_image_tensor
from ..pipelines import PipelineSpec, get_pipeline
from .graph_state_encoder import GraphStateEncoder, graph_feature_tensor
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
    keypoint_visibility: torch.Tensor
    # ``g_t`` for the actor and for the critic, or ``None`` on an arm whose
    # observation does not carry the ontology situation graph.
    actor_graph_embedding: torch.Tensor | None = None
    critic_graph_embedding: torch.Tensor | None = None


class PipelineActorCritic(nn.Module):
    """Common recurrent actor/critic with optional Shin auxiliary supervision.

    Every pipeline deploys image + UAV proprioception, an identical keypoint
    encoder/LSTM/latent/actor and the literal ``latent[..., 6:]`` actor slice.
    Only ``shin_se`` constructs ``relative_state_head``.  Critic truth is
    accepted solely by the separately evaluated training critic.
    """

    def __init__(self, image_embedding=512, lstm_hidden=512, latent_dim=256,
                 actor_hidden=256, critic_hidden=256,
                 action_dim=PLANAR_ACTION_DIM,
                 init_log_std=-1.5, actor_output_gain=0.01,
                 freeze_keypoint=False,
                 relative_state_scale=(3.0, 3.0, 8.0, 3.0, 3.0, 2.0),
                 pipeline: PipelineSpec | str = "shin_se",
                 graph_hidden_dim=32, graph_dim=32, graph_relation_dim=6,
                 graph_heads=1, graph_seed=42):
        super().__init__()
        self.pipeline_spec = (get_pipeline(pipeline) if isinstance(pipeline, str)
                              else pipeline)
        if not isinstance(self.pipeline_spec, PipelineSpec):
            raise TypeError("pipeline must be a PipelineSpec or registered pipeline ID")
        if action_dim != PLANAR_ACTION_DIM:
            raise ValueError(
                "the reduced planar envelope has exactly three action channels: "
                "[a_fwd, a_z, tilt]")
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

        # Width of the graph branch. Known before the branch is built, so the
        # actor and the critic can be sized for it and still be constructed in
        # the same order -- and therefore with the same draws from the global
        # generator -- as on an arm that has no graph. The two arms' shared
        # parameters are then bit-identical at the same seed, which is what
        # makes the comparison a comparison; ``tests/test_two_pipeline_fov.py``
        # checks it tensor by tensor.
        self.graph_dim = (int(graph_dim)
                          if self.pipeline_spec.graph_state_enabled else 0)

        # Only the INPUT layer of each head changes width with the graph
        # branch, and ``nn.Linear`` draws a number of values that depends on
        # its fan-in. Building the two input layers last therefore leaves every
        # other draw -- the hidden layers, the outputs, the log-std -- taken
        # from the same point in the global generator on both arms, so the two
        # models are bit-identical everywhere the comparison requires it and
        # differ exactly where the factor is.
        actor_hidden_layer = nn.Linear(actor_hidden, actor_hidden)
        actor_output = nn.Linear(actor_hidden, action_dim)
        # Re-initialised here rather than after assembly, for the same reason
        # the input layers are built last: ``orthogonal_`` draws from the
        # global generator too, and running it after a layer whose fan-in
        # differs between arms would make the output layer differ as well.
        output_gain = float(actor_output_gain)
        if not math.isfinite(output_gain) or output_gain < 0.0:
            raise ValueError("actor output gain must be finite and non-negative")
        nn.init.orthogonal_(actor_output.weight, gain=output_gain)
        nn.init.zeros_(actor_output.bias)
        # Section III-D: o_priv=[u_t, s_rel_t], exactly 7+6 values, plus g_t on
        # the arm that has one.
        critic_hidden_layer = nn.Linear(critic_hidden, critic_hidden)
        critic_output = nn.Linear(critic_hidden, 1)
        actor_input = nn.Linear(
            latent_dim - self.pipeline_spec.reserved_latent_dimensions
            + 7 + self.graph_dim, actor_hidden)
        critic_input = nn.Linear(13 + self.graph_dim, critic_hidden)

        self.actor = nn.Sequential(actor_input, nn.Tanh(),
                                   actor_hidden_layer, nn.Tanh(), actor_output)
        self.critic = nn.Sequential(critic_input, nn.Tanh(),
                                    critic_hidden_layer, nn.Tanh(),
                                    critic_output)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(init_log_std)))
        self.action_dim = action_dim

        # Built last, and only here, for the ordering reason above. Actor and
        # critic hold one encoder each: the two networks already have separate
        # learning rates and Adam state, and a shared encoder would have to
        # merge two differently scaled gradients into one parameter. Both see
        # the same G_t, have the same architecture, and are trained by PPO.
        if self.pipeline_spec.graph_state_enabled:
            representation = self.pipeline_spec.graph_state_representation
            self.policy_graph_encoder = GraphStateEncoder(
                hidden_dim=graph_hidden_dim, graph_dim=self.graph_dim,
                relation_dim=graph_relation_dim, heads=graph_heads,
                representation=representation, seed=int(graph_seed))
            self.value_graph_encoder = GraphStateEncoder(
                hidden_dim=graph_hidden_dim, graph_dim=self.graph_dim,
                relation_dim=graph_relation_dim, heads=graph_heads,
                representation=representation, seed=int(graph_seed) + 1)
        else:
            self.policy_graph_encoder = None
            self.value_graph_encoder = None

    @property
    def graph_state_enabled(self) -> bool:
        return self.policy_graph_encoder is not None

    @property
    def device(self):
        return self.log_std.device

    def initial_state(self, batch_size: int):
        return self.temporal_backbone.initial_state(
            batch_size, device=self.device, dtype=self.log_std.dtype)

    @torch.no_grad()
    def perceive(self, images: torch.Tensor):
        """Run the frozen keypoint encoder alone, for one or more frames.

        The ontology situation graph is built from this encoder's output, and
        the actor needs the graph as an *input*, so the encoder has to run
        before the rest of the network rather than inside it. The result is
        handed straight back to :meth:`forward` as ``visual`` so the frame is
        still encoded exactly once per control step.
        """
        if images.ndim == 4:
            images = images[:, None]
        if images.ndim != 5 or images.shape[2] != 1:
            raise ValueError("images must have shape BxTx1xHxW")
        batch, steps = images.shape[:2]
        visual = self.encoder(images.reshape(batch * steps, *images.shape[2:]))
        return visual, (batch, steps)

    def forward(self, images: torch.Tensor, proprioception: torch.Tensor,
                *, true_relative_state: torch.Tensor | None = None,
                graph_features: torch.Tensor | None = None,
                visual=None,
                hidden=None, episode_start: torch.Tensor | None = None) -> RecurrentOutput:
        """One forward pass.

        ``graph_features`` is ``X_t`` for the ontology situation graph, shaped
        ``BxTxNxD`` (or ``BxNxD`` for a single step). It is required exactly
        when the pipeline declares the graph state and refused otherwise, so an
        arm can never be handed a representation its declaration does not carry.
        """
        if images.ndim == 4:
            images = images[:, None]
        if proprioception.ndim == 2:
            proprioception = proprioception[:, None]
        if images.ndim != 5 or images.shape[2] != 1:
            raise ValueError("images must have shape BxTx1xHxW")
        batch, steps = images.shape[:2]
        # ``visual`` is the rollout's already-computed encoder output. It is
        # only ever supplied under ``torch.no_grad``; a training pass leaves it
        # None so the encoder stays inside the autograd graph.
        if visual is None:
            visual = self.encoder(images.reshape(batch * steps, *images.shape[2:]))
        encoded = visual.embedding.reshape(batch, steps, -1)
        temporal = self.temporal_backbone(
            encoded, proprioception, hidden, episode_start)
        relative_state = (None if self.relative_state_head is None
                          else self.relative_state_head(temporal.latent))

        actor_graph = critic_graph = None
        if self.graph_state_enabled:
            if graph_features is None:
                raise ValueError(
                    "the graph state pipeline requires the situation graph X_t")
            if graph_features.ndim == 3:
                graph_features = graph_features[:, None]
            if graph_features.ndim != 4 or graph_features.shape[:2] != (batch, steps):
                raise ValueError("graph features must have shape BxTxNxD")
            actor_graph = self.policy_graph_encoder(graph_features)
            critic_graph = self.value_graph_encoder(graph_features)
        elif graph_features is not None:
            raise ValueError(
                "this pipeline does not declare the ontology graph state and "
                "must not be handed one")

        actor_parts = [temporal.latent[..., self.pipeline_spec.actor_latent_slice],
                       proprioception]
        if actor_graph is not None:
            actor_parts.append(actor_graph)
        mean = self.actor(torch.cat(actor_parts, -1))
        std = self.log_std.exp().expand_as(mean)
        value = None
        if true_relative_state is not None:
            if true_relative_state.ndim == 2:
                true_relative_state = true_relative_state[:, None]
            if true_relative_state.shape != (batch, steps, 6):
                raise ValueError("critic truth must have shape BxTx6")
            critic_parts = [proprioception, true_relative_state]
            if critic_graph is not None:
                critic_parts.append(critic_graph)
            value = self.critic(torch.cat(critic_parts, -1)).squeeze(-1)
        return RecurrentOutput(
            mean, std, value, relative_state, temporal.latent, temporal.hidden,
            visual.keypoints.reshape(batch, steps, 6, 2),
            visual.heatmaps.reshape(batch, steps, 6,
                                    visual.heatmaps.shape[-2],
                                    visual.heatmaps.shape[-1]),
            visual.visibility.reshape(batch, steps, 6),
            actor_graph_embedding=actor_graph,
            critic_graph_embedding=critic_graph)

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
    def action(self, observation: ActorObservation, deterministic=False, *,
               graph=None):
        image = grayscale_image_tensor(observation.image, device=self.model.device)
        proprio = torch.as_tensor(observation.proprioception[None],
                                  dtype=torch.float32, device=self.model.device)
        graph_features = None
        if self.model.graph_state_enabled:
            if graph is None:
                raise ValueError(
                    "deploying the graph state policy requires the situation graph")
            graph_features = graph_feature_tensor(
                graph, device=self.model.device)[:, None]
        result = self.model(image, proprio, graph_features=graph_features,
                            hidden=self.hidden)
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
                   graph_features=batch.get("graph_features"),
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
        "graph_state_enabled": float(model.pipeline_spec.graph_state_enabled),
    }
    if auxiliary_loss is not None:
        metrics["auxiliary_estimation_loss"] = auxiliary_loss.detach()
    return loss, metrics


class ShinRecurrentActorCritic(PipelineActorCritic):
    """Backward-compatible name for the explicit-supervision Shin pipeline."""

    def __init__(self, *args, pipeline: PipelineSpec | str = "shin_se", **kwargs):
        super().__init__(*args, pipeline=pipeline, **kwargs)


StatefulShinPolicy = StatefulPipelinePolicy
