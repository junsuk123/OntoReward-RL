"""Information-matched PPO actor/critic representations."""
from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
from torch import nn

from ..rgat.layers import RelationalGraphAttention
from ..rgat.topology import Topology
from .contracts import load_v2_registry
from .config import GraphConfig
from .environment import ObservationBundle
from .ontology import (FEATURE_CHANNELS, GRAPH_EDGES, NODE_NAMES, QUERY_NODES,
                       RELATION_NAMES, semantic_flat_features)


PolicyMode = Literal[
    "ppo_vector_canonical", "ppo_semantic_flat", "ppo_ontology_rgat"]
POLICY_MODES: tuple[str, ...] = (
    "ppo_vector_canonical", "ppo_semantic_flat", "ppo_ontology_rgat")
GRAPH_INPUT_DIM = len(FEATURE_CHANNELS) + len(NODE_NAMES)


def _mlp(input_dim: int, output_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, output_dim))


def context_topology() -> Topology:
    index = {name: i for i, name in enumerate(NODE_NAMES)}
    rel_index = {name: i for i, name in enumerate(RELATION_NAMES)}
    src = np.array([index[s] for s, _, _ in GRAPH_EDGES], dtype=np.int64)
    dst = np.array([index[d] for _, _, d in GRAPH_EDGES], dtype=np.int64)
    rel = np.array([rel_index[r] for _, r, _ in GRAPH_EDGES], dtype=np.int64)
    return Topology(src, dst, rel, len(NODE_NAMES), len(RELATION_NAMES),
                    NODE_NAMES.index("PolicyNode"))


_LAYER_KWARGS = dict(
    head_aggregation="mean", attention_mode="argat", attention_style="sum",
    attention_units=1, leaky_relu_slope=0.2, kernel_basis_size=0,
    attn_kernel_basis_size=0, feature_dropout=0.0, support_dropout=0.0,
    softmax_floor=1e-12, stable_softmax=True)


class ContextRGATEncoder(nn.Module):
    """Two relational layers and a role-specific query-node readout."""

    def __init__(self, query_node: str, *, hidden_dim: int = 32,
                 output_dim: int = 32, relation_dim: int = 6,
                 seed: int = 42):
        super().__init__()
        if query_node not in QUERY_NODES:
            raise ValueError(f"query node must be one of {QUERY_NODES}")
        self.query_node = query_node
        self.query_index = NODE_NAMES.index(query_node)
        self.topology = context_topology()
        self.layer1 = RelationalGraphAttention(
            GRAPH_INPUT_DIM, hidden_dim, self.topology,
            relation_dim=relation_dim, heads=1, **_LAYER_KWARGS)
        self.layer2 = RelationalGraphAttention(
            hidden_dim, hidden_dim, self.topology,
            relation_dim=relation_dim, heads=1, **_LAYER_KWARGS)
        self.readout = nn.Linear(hidden_dim, output_dim)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.layer1.reset_parameters(generator, scheme="glorot")
        self.layer2.reset_parameters(generator, scheme="glorot")
        with torch.no_grad():
            bound = math.sqrt(1.0 / hidden_dim)
            self.readout.weight.uniform_(-bound, bound, generator=generator)
            self.readout.bias.zero_()

    def forward(self, graph_features: torch.Tensor) -> torch.Tensor:
        if graph_features.shape[-2:] != (len(NODE_NAMES), GRAPH_INPUT_DIM):
            raise ValueError("context graph tensor has incompatible shape")
        hidden = torch.tanh(self.layer1(graph_features))
        hidden = torch.tanh(hidden + self.layer2(hidden))
        return torch.tanh(self.readout(hidden[..., self.query_index, :]))


class _Representation(nn.Module):
    def __init__(self, mode: PolicyMode, *, role: str, hidden: int, seed: int):
        super().__init__()
        if mode not in POLICY_MODES:
            raise ValueError(f"unknown two-axis policy mode: {mode}")
        self.mode = mode
        self.role = role
        packet_dim = load_v2_registry().dimension
        flat_dim = len(NODE_NAMES) * GRAPH_INPUT_DIM
        if mode == "ppo_vector_canonical":
            self.encoder = _mlp(packet_dim, hidden, hidden)
        elif mode == "ppo_semantic_flat":
            self.encoder = _mlp(flat_dim, hidden, hidden)
        else:
            query = "PolicyNode" if role == "actor" else "ValueNode"
            self.encoder = ContextRGATEncoder(
                query, hidden_dim=hidden, output_dim=hidden, seed=seed)

    def forward(self, packets: torch.Tensor, graphs: torch.Tensor) -> torch.Tensor:
        if self.mode == "ppo_vector_canonical":
            return self.encoder(packets)
        if self.mode == "ppo_semantic_flat":
            return self.encoder(graphs.flatten(start_dim=-2))
        return self.encoder(graphs)


class TwoAxisActor(nn.Module):
    def __init__(self, mode: PolicyMode, *, hidden: int = 32, seed: int = 42,
                 initial_log_std: float = -0.7):
        super().__init__()
        self.representation = _Representation(
            mode, role="actor", hidden=hidden, seed=seed)
        self.head = nn.Linear(hidden, 2)
        # Shared across arms. This scale decides how far an untrained policy
        # throws the pad out of a frame only 0.7*h wide, so it governs whether
        # early episodes produce any landing signal at all.
        self.log_std = nn.Parameter(torch.full((2,), float(initial_log_std)))

    def forward(self, packets: torch.Tensor,
                graphs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.head(self.representation(packets, graphs))
        std = self.log_std.clamp(-5.0, 1.0).exp().expand_as(mu)
        return mu, std

    @staticmethod
    def raw_log_probability(raw: torch.Tensor, mu: torch.Tensor,
                            std: torch.Tensor) -> torch.Tensor:
        return TwoAxisActor.raw_log_probability_axes(raw, mu, std).sum(-1)

    @staticmethod
    def raw_log_probability_axes(raw: torch.Tensor, mu: torch.Tensor,
                                 std: torch.Tensor) -> torch.Tensor:
        """The same density, kept factorized so single axes can be dropped.

        The policy is a diagonal Gaussian, so the joint log-density is the sum
        of the per-axis terms. Summing early throws away the only structure
        that lets the actor objective ignore one axis the supervisor replaced
        while keeping the axes it actually executed.
        """
        return -0.5 * (((raw - mu) / std) ** 2
                       + 2.0 * torch.log(std) + math.log(2.0 * math.pi))


class TwoAxisCritic(nn.Module):
    def __init__(self, mode: PolicyMode, *, hidden: int = 32, seed: int = 43):
        super().__init__()
        self.representation = _Representation(
            mode, role="critic", hidden=hidden, seed=seed)
        self.head = nn.Linear(hidden, 1)

    def forward(self, packets: torch.Tensor, graphs: torch.Tensor) -> torch.Tensor:
        return self.head(self.representation(packets, graphs)).squeeze(-1)


class TwoAxisPPOAgent(nn.Module):
    def __init__(self, mode: PolicyMode, *, hidden: int = 32, seed: int = 42,
                 device: str | torch.device = "cpu",
                 initial_log_std: float = -0.7,
                 graph_config: GraphConfig | None = None):
        super().__init__()
        torch.manual_seed(int(seed))
        self.seed = int(seed)
        if mode not in POLICY_MODES:
            raise ValueError(f"unknown two-axis policy mode: {mode}")
        self.mode = mode
        self.graph_config = graph_config or GraphConfig()
        if self.graph_config.schema == "compact_context_graph_v3_grouped":
            from .models_v28 import ReferenceHead
            self.actor = ReferenceHead(mode, "actor", self.graph_config, seed, initial_log_std)
            self.critic = ReferenceHead(mode, "critic", self.graph_config, seed)
        else:
            self.actor = TwoAxisActor(mode, hidden=hidden, seed=seed,
                                      initial_log_std=initial_log_std)
            self.critic = TwoAxisCritic(mode, hidden=hidden, seed=seed + 1)
        self.device = torch.device(device)
        self.to(self.device)
        self._generator = torch.Generator(device=self.device).manual_seed(seed + 2)

    def tensors(self, observation: ObservationBundle
                ) -> tuple[torch.Tensor, torch.Tensor]:
        packets = torch.as_tensor(observation.packet.values,
                                  dtype=torch.float32, device=self.device).unsqueeze(0)
        graphs = torch.as_tensor(observation.graph.X,
                                 dtype=torch.float32, device=self.device).unsqueeze(0)
        return packets, graphs

    @torch.no_grad()
    def act(self, observation: ObservationBundle, *, deterministic: bool = False
            ) -> tuple[np.ndarray, np.ndarray, float, float]:
        packets, graphs = self.tensors(observation)
        mu, std = self.actor(packets, graphs)
        raw = mu if deterministic else mu + std * torch.randn(
            mu.shape, generator=self._generator, device=self.device)
        normalized = torch.tanh(raw)
        logp = self.actor.raw_log_probability(raw, mu, std)
        value = self.critic(packets, graphs)
        return (raw[0].cpu().numpy(), normalized[0].cpu().numpy(),
                float(logp[0].cpu()), float(value[0].cpu()))

    def parameter_count(self) -> dict[str, int]:
        actor = sum(p.numel() for p in self.actor.parameters())
        critic = sum(p.numel() for p in self.critic.parameters())
        return {"actor": actor, "critic": critic, "total": actor + critic}


def observation_arrays(observations: list[ObservationBundle]
                       ) -> tuple[np.ndarray, np.ndarray]:
    packets = np.stack([item.packet.values for item in observations]).astype(np.float32)
    graphs = np.stack([item.graph.X for item in observations]).astype(np.float32)
    return packets, graphs


def capacity_matched_mlp_hidden(mode: str, target_parameters: int,
                                *, maximum_hidden: int = 256,
                                graph_config: GraphConfig | None = None) -> tuple[int, int]:
    """Closest actor+critic parameter count for a vector/flat MLP control."""
    if graph_config and graph_config.schema == "compact_context_graph_v3_grouped":
        if mode not in {"ppo_vector_canonical", "ppo_semantic_flat"}:
            raise ValueError("capacity matching requires an MLP arm")
        dimension = 26 if mode == "ppo_vector_canonical" else 108
        count = lambda h: 2*h*h + (2*dimension+7)*h + 5
    elif mode == "ppo_vector_canonical":
        # Two 3-layer representations plus 2-D actor/value heads and log std.
        count = lambda h: 4 * h * h + 61 * h + 5
    elif mode == "ppo_semantic_flat":
        count = lambda h: 4 * h * h + 471 * h + 5
    else:
        raise ValueError("capacity matching is defined for MLP controls only")
    hidden = min(range(1, maximum_hidden + 1),
                 key=lambda h: abs(count(h) - int(target_parameters)))
    return hidden, count(hidden)


def log_probability_axes(agent, observation, raw) -> np.ndarray:
    """Per-axis log-density of `raw` under `agent`'s CURRENT policy.

    A module function, not a method: the spatial route's `SpatialAgent` is a
    separate class from `TwoAxisPPOAgent` and both only share `.tensors()` and
    `.actor`. Collection stores this so a masked importance ratio can drop the
    same axes from numerator and denominator; a ratio with a masked numerator
    over a full-sum denominator is not a probability ratio at all.
    """
    with torch.no_grad():
        packets, graphs = agent.tensors(observation)
        mu, std = agent.actor(packets, graphs)
        axes = TwoAxisActor.raw_log_probability_axes(
            torch.as_tensor(raw, dtype=mu.dtype, device=mu.device).unsqueeze(0),
            mu, std)
    return axes[0].cpu().numpy()
