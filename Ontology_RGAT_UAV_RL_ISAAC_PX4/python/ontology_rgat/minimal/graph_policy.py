"""R-GAT actor/critic over the minimal ontology graph.

Two ``RelationalGraphAttention`` layers (residual around the second), the
per-edge weights from the ontology applied to every message, four grouped
mean readouts concatenated, then a Gaussian actor over ENU net acceleration
and a value head.

``relational_activity`` is the share of the encoder output that comes from
non-self relations: the graph run with every non-self relation gated off is
subtracted from the full one. Zero means the policy is a per-node MLP in
disguise -- check it before reading any arm comparison (see AGENTS.md on the
proposed arm that was the flat baseline).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from ..rgat.layers import RelationalGraphAttention
from ..rgat.topology import Topology
from .constants import DEFAULT_CONSTANTS, LandingConstants
from .ontology import (DECISION_NODE, EDGE_RELATION, EDGE_SOURCE, EDGE_TARGET,
                       FEATURE_DIM, NODES, READOUT_GROUPS, REL, RELATIONS, schema_hash)


@dataclass(frozen=True)
class GraphPolicyConfig:
    hidden_dim: int = 32
    heads: int = 2
    relation_dim: int = 6
    actor_hidden: int = 64
    # sigma 0.082. Measured on the /5 clones, 48 held-out seeds: sampled
    # landing 58-71 % at the old -1.1 (sigma 0.333), 96-100 % at sigma <= 0.135
    # for both graph arms. PPO's rollouts are sampled, so this is where it starts.
    initial_log_std: float = -2.5
    minimum_log_std: float = -4.0
    seed: int = 0


def build_topology() -> Topology:
    return Topology(EDGE_SOURCE, EDGE_TARGET, EDGE_RELATION, len(NODES),
                    len(RELATIONS), DECISION_NODE)


class MinimalGraphPolicy(nn.Module):
    def __init__(self, config: GraphPolicyConfig = GraphPolicyConfig(),
                 constants: LandingConstants = DEFAULT_CONSTANTS):
        super().__init__()
        self.config = config
        self.schema_hash = schema_hash(constants)
        self.register_buffer("action_scale", torch.tensor(
            constants.max_acceleration_m_s2, dtype=torch.float32))
        generator = torch.Generator().manual_seed(config.seed)
        topology = build_topology()
        kw = dict(heads=config.heads, head_aggregation="mean",
                  relation_dim=config.relation_dim)
        self.layer1 = RelationalGraphAttention(FEATURE_DIM, config.hidden_dim, topology, **kw)
        self.layer2 = RelationalGraphAttention(config.hidden_dim, config.hidden_dim, topology, **kw)
        for layer in (self.layer1, self.layer2):
            layer.reset_parameters(generator, scheme="glorot")
        self.input_skip = nn.Linear(FEATURE_DIM, config.hidden_dim)
        readout = len(READOUT_GROUPS) * config.hidden_dim
        self.actor = nn.Sequential(nn.Linear(readout, config.actor_hidden), nn.Tanh(),
                                   nn.Linear(config.actor_hidden, 3))
        self.critic = nn.Sequential(nn.Linear(readout, config.actor_hidden), nn.Tanh(),
                                    nn.Linear(config.actor_hidden, 1))
        self.log_std = nn.Parameter(torch.full((3,), float(config.initial_log_std)))
        self._groups = [torch.tensor(g, dtype=torch.long) for g in READOUT_GROUPS]
        self_only = torch.zeros(len(RELATIONS))
        self_only[REL["self"]] = 1.0
        self.register_buffer("_self_only", self_only, persistent=False)

    # ------------------------------------------------------------ encoding
    def encode(self, features: torch.Tensor, edge_weight: torch.Tensor,
               relation_gates: torch.Tensor | None = None) -> torch.Tensor:
        h1 = torch.tanh(self.layer1(features, edge_weights=edge_weight,
                                    relation_gates=relation_gates)
                        + self.input_skip(features))
        h2 = torch.tanh(self.layer2(h1, edge_weights=edge_weight,
                                    relation_gates=relation_gates) + h1)
        return h2

    def readout(self, nodes: torch.Tensor) -> torch.Tensor:
        return torch.cat([nodes[..., g, :].mean(-2) for g in self._groups], dim=-1)

    def forward(self, features, edge_weight):
        """Returns (mean action in m/s^2, log_std, value)."""
        features = torch.as_tensor(features, dtype=torch.float32)
        edge_weight = torch.as_tensor(edge_weight, dtype=torch.float32)
        squeeze = features.dim() == 2
        if squeeze:
            features, edge_weight = features.unsqueeze(0), edge_weight.unsqueeze(0)
        z = self.readout(self.encode(features, edge_weight))
        mean = torch.tanh(self.actor(z)) * self.action_scale
        value = self.critic(z).squeeze(-1)
        log_std = self.log_std.clamp(min=self.config.minimum_log_std).expand_as(mean)
        if squeeze:
            mean, log_std, value = mean[0], log_std[0], value[0]
        return mean, log_std, value

    @torch.no_grad()
    def act(self, features, edge_weight, deterministic: bool = True) -> np.ndarray:
        mean, log_std, _ = self(features, edge_weight)
        if not deterministic:
            mean = mean + torch.randn_like(mean) * log_std.exp()
        return mean.numpy().astype(float)

    @torch.no_grad()
    def relational_activity(self, features, edge_weight) -> float:
        features = torch.as_tensor(features, dtype=torch.float32).reshape(1, len(NODES), -1)
        edge_weight = torch.as_tensor(edge_weight, dtype=torch.float32).reshape(1, -1)
        full = self.encode(features, edge_weight)
        alone = self.encode(features, edge_weight, relation_gates=self._self_only)
        return float((full - alone).norm() / full.norm().clamp_min(1e-9))

    # ---------------------------------------------------------- checkpoints
    def save(self, path) -> None:
        torch.save({"schema_hash": self.schema_hash, "config": asdict(self.config),
                    "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path, constants: LandingConstants = DEFAULT_CONSTANTS) -> "MinimalGraphPolicy":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        expected = schema_hash(constants)
        if payload.get("schema_hash") != expected:
            raise ValueError(
                f"checkpoint schema {payload.get('schema_hash')} != ontology {expected}")
        policy = cls(GraphPolicyConfig(**payload["config"]), constants)
        policy.load_state_dict(payload["state_dict"])
        return policy
