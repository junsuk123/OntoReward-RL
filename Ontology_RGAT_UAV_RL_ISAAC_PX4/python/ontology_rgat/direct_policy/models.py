"""Source-compatible actor/critic architectures for the direct-policy study."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .contracts import ContractBundle, digest, validate_checkpoint_metadata
from .graph import GraphState, shuffled_relations


class SourceRelationalLayer(nn.Module):
    """One MATLAB-style relation layer with incoming-neighbour softmax."""
    def __init__(self, input_dim: int, hidden_dim: int, relations: int,
                 relation_dim: int = 4):
        super().__init__()
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        self.relations, self.relation_dim = relations, relation_dim
        self.weight = nn.Parameter(torch.empty(relations, hidden_dim, input_dim))
        self.attention = nn.Parameter(torch.empty(relations, 2 * hidden_dim + relation_dim))
        self.embedding = nn.Parameter(torch.empty(relations, relation_dim))

    def reset_parameters(self, generator: torch.Generator, scale: float = 0.1):
        with torch.no_grad():
            self.weight.copy_(torch.randn(self.weight.shape, generator=generator) * scale)
            self.attention.copy_(torch.randn(self.attention.shape, generator=generator) * scale)
            self.embedding.copy_(torch.randn(self.embedding.shape, generator=generator) * scale)

    def forward(self, features, source, target, relation):
        features = torch.as_tensor(features, dtype=torch.float32)
        squeeze = features.dim() == 2
        if squeeze:
            features = features.unsqueeze(0)
        source = torch.as_tensor(source, dtype=torch.long, device=features.device)
        target = torch.as_tensor(target, dtype=torch.long, device=features.device)
        relation = torch.as_tensor(relation, dtype=torch.long, device=features.device)
        batch, nodes, _ = features.shape
        projected = torch.einsum("rhi,bni->brnh", self.weight, features)
        src_h = projected[:, relation, source]
        dst_h = projected[:, relation, target]
        rel_e = self.embedding[relation].unsqueeze(0).expand(batch, -1, -1)
        joined = torch.cat((src_h, dst_h, rel_e), dim=-1)
        raw = (joined * self.attention[relation].unsqueeze(0)).sum(-1)
        score = torch.where(raw >= 0, raw, 0.2 * raw)
        target_index = target.unsqueeze(0).expand(batch, -1)
        node_max = torch.full((batch, nodes), -torch.inf, dtype=score.dtype,
                              device=score.device)
        node_max.scatter_reduce_(1, target_index, score, reduce="amax",
                                 include_self=True)
        exponential = torch.exp(score - node_max.gather(1, target_index))
        denominator = torch.zeros_like(node_max)
        denominator.scatter_add_(1, target_index, exponential)
        alpha = exponential / denominator.gather(1, target_index)
        output = torch.zeros(batch, nodes, self.hidden_dim,
                             dtype=features.dtype, device=features.device)
        output.scatter_add_(
            1, target_index.unsqueeze(-1).expand(-1, -1, self.hidden_dim),
            alpha.unsqueeze(-1) * src_h)
        return output[0] if squeeze else output


class SourceGraphEncoder(nn.Module):
    """Single R-GAT layer + local skip + identity grouped readout."""
    def __init__(self, graph: GraphState, hidden_dim: int = 16,
                 output_dim: int = 32, seed: int = 0,
                 relation_override: np.ndarray | None = None):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        channels = graph.features.shape[1]
        self.layer = SourceRelationalLayer(channels, hidden_dim,
                                           len(graph.schema.relations))
        self.layer.reset_parameters(generator)
        self.local = nn.Linear(channels, hidden_dim)
        self.readout_projection = nn.Linear(hidden_dim * len(graph.schema.readout_groups),
                                            output_dim)
        with torch.no_grad():
            self.local.weight.zero_()
            self.local.bias.zero_()
            diagonal = min(hidden_dim, channels)
            self.local.weight[:diagonal, :diagonal] = torch.eye(diagonal)
            self.readout_projection.weight.copy_(
                torch.randn(self.readout_projection.weight.shape, generator=generator) * 0.1)
            self.readout_projection.bias.zero_()
        self.register_buffer("source", torch.as_tensor(graph.source, dtype=torch.long))
        self.register_buffer("target", torch.as_tensor(graph.target, dtype=torch.long))
        relation = graph.relation if relation_override is None else relation_override
        self.register_buffer("relation", torch.as_tensor(relation, dtype=torch.long))
        self.groups = tuple(tuple(group) for group in graph.schema.readout_groups)
        self.output_dim = output_dim

    def node_embeddings(self, features):
        features = torch.as_tensor(features, dtype=torch.float32)
        return torch.tanh(self.layer(features, self.source, self.target, self.relation)
                          + self.local(features))

    def forward(self, features):
        nodes = self.node_embeddings(features)
        grouped = torch.cat(
            [nodes[..., list(group), :].mean(dim=-2) for group in self.groups], dim=-1)
        return torch.tanh(self.readout_projection(grouped))


class IdentityEncoder(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.output_dim = dimension

    def forward(self, value):
        return torch.as_tensor(value, dtype=torch.float32)


def _mlp(input_dim: int, hidden: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, output_dim))


@dataclass(frozen=True)
class PolicyOutput:
    latent_mean: torch.Tensor
    log_std: torch.Tensor
    value: torch.Tensor


class DirectActorCritic(nn.Module):
    """Separate actor/critic encoders; the actor has no raw graph bypass."""
    def __init__(self, contract: ContractBundle, method: str,
                 graph_example: GraphState | None = None, *, seed: int = 0,
                 graph_seed: int = 0):
        super().__init__()
        if method not in ("ppo", "onto_rgat_ppo", "shuffled_rgat_ppo"):
            raise ValueError(f"unknown direct-policy method: {method}")
        self.contract, self.method = contract, method
        self.graph_seed = int(graph_seed)
        torch.manual_seed(seed)
        training = contract.training
        if method == "ppo":
            self.actor_encoder = IdentityEncoder(contract.observation.dimension)
            self.critic_encoder = IdentityEncoder(contract.observation.dimension)
        else:
            if graph_example is None:
                raise ValueError("graph method requires a graph example")
            relation = None
            if method == "shuffled_rgat_ppo":
                relation = shuffled_relations(graph_example.schema, graph_seed)
            self.actor_encoder = SourceGraphEncoder(
                graph_example, training.graph_hidden_size, training.graph_output_size,
                seed=seed + 1101, relation_override=relation)
            self.critic_encoder = SourceGraphEncoder(
                graph_example, training.graph_hidden_size, training.graph_output_size,
                seed=seed + 2202, relation_override=relation)
        self.actor = _mlp(self.actor_encoder.output_dim, training.hidden_size,
                          len(contract.action.names))
        if getattr(training, "zero_last_layer_initialization", False):
            final = self.actor[-1]
            with torch.no_grad():
                final.weight.zero_()
                final.bias.zero_()
        self.critic = _mlp(self.critic_encoder.output_dim, training.hidden_size, 1)
        self.log_std = nn.Parameter(torch.full((len(contract.action.names),),
                                               training.initial_log_std))
        self.register_buffer("action_scale",
                             torch.tensor(contract.action.maximum, dtype=torch.float32))

    @property
    def is_graph(self) -> bool:
        return self.method != "ppo"

    @property
    def method_hash(self) -> str:
        return digest({"algorithm_hash": self.contract.algorithm_hash,
                       "method": self.method,
                       "graph_seed": self.graph_seed if self.method == "shuffled_rgat_ppo" else None})

    def forward(self, policy_input) -> PolicyOutput:
        actor_state = self.actor_encoder(policy_input)
        critic_state = self.critic_encoder(policy_input)
        latent_mean = self.actor(actor_state)
        if getattr(self.contract.training, "analytic_tracking_prior", False):
            latent_mean = latent_mean + self._tracking_prior(policy_input)
        value = self.critic(critic_state).squeeze(-1)
        maximum = getattr(self.contract.training, "maximum_log_std", None)
        log_std = self.log_std.clamp(
            min=self.contract.training.minimum_log_std, max=maximum)
        log_std = log_std.expand_as(latent_mean)
        return PolicyOutput(latent_mean, log_std, value)

    @staticmethod
    def _inverse_signed(value, scale: float):
        value = value.clamp(-1+1e-6, 1-1e-6)
        return scale*value/(1-value.abs())

    def _tracking_prior(self, policy_input):
        """Observation-only PD tracking/descent mean; PPO learns the residual."""
        dimension = int(self.contract.metadata["dimension"])
        if self.method == "ppo":
            height_index = 1 if dimension == 2 else 2
            vz_index = 4 if dimension == 2 else 7
            height_value = policy_input[..., height_index]
            vz_value = policy_input[..., vz_index]
            vision_age_value = policy_input[..., 9 if dimension == 2 else 18]
            relative_x = policy_input[..., 0]
            relative_vx = policy_input[..., 2 if dimension == 2 else 3]
            bias_x = policy_input[..., -2 if dimension == 2 else -3]
            handover_progress = policy_input[..., -1]
            if dimension == 3:
                relative_y = policy_input[..., 1]
                relative_vy = policy_input[..., 4]
                bias_y = policy_input[..., -2]
        else:
            # Reconstruct the registered normalised horizontal values from the
            # graph.  RelativePosition stores cross_x = rel_x-slope*height;
            # RelativeVelocity retains rel_v in its secondary channels.
            vz_value = policy_input[..., 3, 1]
            height_value = policy_input[..., 3, 2]
            vision_age_value = policy_input[..., 5, 4 if dimension == 2 else 6]
            slope = math.tan(math.pi/6)
            relative_x = policy_input[..., 0, 1] + slope*height_value
            relative_vx = policy_input[..., 1, 2]
            bias_x = policy_input[..., 0, -2 if dimension == 2 else -3]
            handover_progress = policy_input[..., 0, -1]
            if dimension == 3:
                relative_y = policy_input[..., 0, 3]
                relative_vy = policy_input[..., 1, 4]
                bias_y = policy_input[..., 0, -2]
        physical_x = self._inverse_signed(relative_x, 3.0)
        physical_vx = self._inverse_signed(relative_vx, 10.0)
        physical_bias_x = self._inverse_signed(bias_x, 3.0)
        # Keep the measured-safe v4 authority cap.  A deliberately small
        # integral term removes the steady following error that held the
        # Isaac vehicle behind the moving pad for a full 70-second episode.
        cfg = self.contract.training
        action_x = (cfg.tracking_kp*physical_x + cfg.tracking_kd*physical_vx
                    + cfg.tracking_ki*physical_bias_x)
        horizontal = [action_x]
        if dimension == 3:
            physical_y = self._inverse_signed(relative_y, 3.0)
            physical_vy = self._inverse_signed(relative_vy, 10.0)
            physical_bias_y = self._inverse_signed(bias_y, 3.0)
            action_y = (cfg.tracking_kp*physical_y + cfg.tracking_kd*physical_vy
                        + cfg.tracking_ki*physical_bias_y)
            horizontal.append(action_y)
            position_norm = torch.sqrt(physical_x.square()+physical_y.square())
        else:
            position_norm = physical_x.abs()
        horizontal = torch.stack(horizontal, dim=-1)
        horizontal_norm = torch.linalg.vector_norm(horizontal, dim=-1, keepdim=True)
        horizontal = horizontal * torch.clamp(
            cfg.tracking_horizontal_cap_m_s2/(horizontal_norm+1e-8), max=1.0)
        height = self._inverse_signed(height_value, 8.0).clamp(min=0.0)
        own_vz = self._inverse_signed(vz_value, 1.5)
        vision_age = self._inverse_signed(vision_age_value, 3.0).clamp(min=0.0)
        landing_vz = torch.where(
            height > 0.04,
            torch.full_like(height, -cfg.tracking_terminal_sink_m_s),
            torch.zeros_like(height))
        # The deployed CV estimate's velocity is intentionally slow/noisy.
        # Gating descent on it held a well-centred vehicle aloft for 70 s.
        # Position plus the bounded horizontal command is the causal gate;
        # velocity and TrackingBias remain available to the learned residual.
        track_fresh = vision_age <= cfg.tracking_max_vision_age_s
        aligned = (position_norm < cfg.tracking_descent_gate_m) & track_fresh
        recovery_vz = torch.minimum(
            torch.full_like(height, 0.20),
            (cfg.tracking_recovery_height_m-height).clamp(min=0.0))
        desired_vz = torch.where(aligned, landing_vz, recovery_vz)
        vertical_action = (3.0*(desired_vz-own_vz)).clamp(
            -0.98*float(self.action_scale[-1]),
            0.98*float(self.action_scale[-1]))
        vertical_action = torch.where(
            handover_progress >= 1.0, vertical_action,
            torch.maximum(vertical_action, torch.full_like(
                vertical_action, cfg.tracking_handover_climb_m_s2)))
        action = torch.cat((horizontal, vertical_action.unsqueeze(-1)), dim=-1)
        limit = 0.98*self.action_scale
        action = torch.maximum(torch.minimum(action, limit), -limit)
        return torch.atanh(action/self.action_scale)

    def action_from_latent(self, latent, policy_input=None):
        action = torch.tanh(latent) * self.action_scale
        if (policy_input is None
                or not getattr(self.contract.training, "analytic_tracking_prior", False)):
            return action
        dimension = int(self.contract.metadata["dimension"])
        if self.method == "ppo":
            age_value = policy_input[..., 9 if dimension == 2 else 18]
            vz_value = policy_input[..., 4 if dimension == 2 else 7]
            handover_progress = policy_input[..., -1]
        else:
            age_value = policy_input[..., 5, 4 if dimension == 2 else 6]
            vz_value = policy_input[..., 3, 1]
            handover_progress = policy_input[..., 0, -1]
        age = self._inverse_signed(age_value, 3.0).clamp(min=0.0)
        own_vz = self._inverse_signed(vz_value, 1.5)
        cfg = self.contract.training
        # Keep enough authority for a short causal re-acquisition manoeuvre.
        # The lower long-stale cap prevents a frozen track from accelerating
        # the vehicle indefinitely, but applying it as soon as descent is
        # inhibited made recoverable 2--8 s camera misses irreversible.
        cap = torch.where(
            age > cfg.tracking_stale_full_authority_s,
            torch.full_like(age, cfg.tracking_stale_horizontal_cap_m_s2),
            torch.full_like(age, cfg.tracking_horizontal_cap_m_s2))
        horizontal = action[..., :dimension-1]
        norm = torch.linalg.vector_norm(horizontal, dim=-1, keepdim=True)
        horizontal = horizontal * torch.clamp(
            cap.unsqueeze(-1)/(norm+1e-8), max=1.0)
        vertical = action[..., -1]
        vertical = torch.where(
            handover_progress >= 1.0, vertical,
            torch.maximum(vertical, torch.full_like(
                vertical, cfg.tracking_handover_climb_m_s2)))
        descending_stale = ((age > cfg.tracking_max_vision_age_s)
                            & (own_vz < 0.0))
        vertical = torch.where(
            ~descending_stale, vertical,
            torch.maximum(vertical, torch.full_like(
                vertical, cfg.tracking_stale_vertical_floor_m_s2)))
        return torch.cat((horizontal, vertical.unsqueeze(-1)), dim=-1)

    @staticmethod
    def latent_log_prob(latent, mean, log_std):
        return (-0.5 * ((latent - mean) / log_std.exp()) ** 2 - log_std
                - 0.5 * math.log(2 * math.pi)).sum(-1)

    def act(self, policy_input, *, deterministic: bool = False,
            generator: torch.Generator | None = None):
        output = self(policy_input)
        if deterministic:
            latent = output.latent_mean
        else:
            noise = torch.randn(output.latent_mean.shape, generator=generator,
                                device=output.latent_mean.device)
            latent = output.latent_mean + output.log_std.exp() * noise
        action = self.action_from_latent(latent, policy_input)
        log_probability = self.latent_log_prob(
            latent, output.latent_mean, output.log_std)
        return action, latent, log_probability, output.value

    def parameter_report(self) -> dict[str, int]:
        return {
            "actor_encoder": sum(p.numel() for p in self.actor_encoder.parameters()),
            "critic_encoder": sum(p.numel() for p in self.critic_encoder.parameters()),
            "actor": sum(p.numel() for p in self.actor.parameters()) + self.log_std.numel(),
            "critic": sum(p.numel() for p in self.critic.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }


def checkpoint_metadata(contract: ContractBundle, *, training_step: int,
                        optimizer_state: dict[str, Any], rng_state: Any,
                        method_hash: str = "unspecified") -> dict[str, Any]:
    manifest = contract.manifest()
    return {
        "source_sha": manifest["source_sha"],
        "target_baseline_sha": manifest["target_baseline_sha"],
        "algorithm_hash": manifest["algorithm_hash"],
        "method_hash": method_hash,
        "task_contract_hash": manifest["task_contract_hash"],
        "execution_hash": manifest["execution_hash"],
        "training_step": int(training_step),
        "optimizer_state": optimizer_state,
        "rng_state": rng_state,
    }


def save_checkpoint(path, model: DirectActorCritic, *, training_step: int,
                    optimizer_state: dict[str, Any], rng_state: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": model.method, "model_state": model.state_dict(),
        "metadata": checkpoint_metadata(model.contract, training_step=training_step,
                                        optimizer_state=optimizer_state,
                                        rng_state=rng_state,
                                        method_hash=model.method_hash),
    }
    torch.save(payload, path)


def load_checkpoint(path, model: DirectActorCritic, *, allow_execution_transfer=False) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("method") != model.method:
        raise ValueError("checkpoint method is incompatible")
    metadata = payload.get("metadata", {})
    checked = dict(metadata)
    if allow_execution_transfer:
        checked["execution_hash"] = model.contract.execution_hash
    validate_checkpoint_metadata(checked, model.contract)
    if metadata.get("method_hash") != model.method_hash:
        raise ValueError("checkpoint method hash is incompatible")
    model.load_state_dict(payload["model_state"])
    return metadata
