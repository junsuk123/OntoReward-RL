"""Frozen R-GAT base with bounded PPO-trainable relation gates only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..rgat.layers import RelationalGraphAttention
from ..rgat.selective_state import (
    ADAPTIVE_RELATIONS, INVARIANT_RELATIONS, SELECTIVE_GRAPH_INPUT_DIM,
    SELECTIVE_GRAPH_VERSION, SELECTIVE_NODE_NAMES, SELECTIVE_RELATION_NAMES,
    SELECTIVE_SCHEMA_HASH, RELATION_PARTITION_HASH, build_selective_graph)
from ..rgat.topology import Topology


SELECTIVE_ARTIFACT_FORMAT = "selective-rgat-pretrained-v1"


class RelationGate(nn.Module):
    def __init__(self, observation_dim: int, beta: float = 0.35):
        super().__init__()
        if not 0.0 < float(beta) < 1.0:
            raise ValueError("relation gate beta must be in (0,1)")
        self.beta = float(beta)
        self.linear = nn.Linear(int(observation_dim), len(ADAPTIVE_RELATIONS))
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, normalized_observation: torch.Tensor) -> torch.Tensor:
        adaptive = 1.0 + self.beta * torch.tanh(
            self.linear(normalized_observation))
        invariant = torch.ones(
            *adaptive.shape[:-1], len(INVARIANT_RELATIONS),
            dtype=adaptive.dtype, device=adaptive.device)
        return torch.cat((adaptive, invariant), dim=-1)


def _topology() -> Topology:
    # Topology is data independent; a zero packet is sufficient to instantiate it.
    from ..contracts.observation import CausalObservationPacket, load_observation_registry
    registry = load_observation_registry()
    packet = CausalObservationPacket(
        np.zeros(registry.dimension, dtype=np.float32), registry.sha256, 0.0)
    graph = build_selective_graph(packet, registry=registry)
    return Topology.from_graph(graph, len(SELECTIVE_RELATION_NAMES))


class SelectiveGraphStateEncoder(nn.Module):
    """The base and readout never receive PPO gradients; only gates do."""

    graph_version = SELECTIVE_GRAPH_VERSION

    def __init__(self, *, observation_dim: int, hidden_dim: int = 32,
                 graph_dim: int = 32, relation_dim: int = 6, heads: int = 1,
                 beta: float = 0.35, normalization_mean=None,
                 normalization_std=None, seed: int = 42):
        super().__init__()
        topology = _topology()
        kwargs = dict(head_aggregation="mean", attention_mode="argat",
                      attention_style="sum", attention_units=1,
                      relation_dim=int(relation_dim), heads=int(heads),
                      stable_softmax=True, softmax_floor=1e-12)
        self.layer1 = RelationalGraphAttention(
            SELECTIVE_GRAPH_INPUT_DIM, int(hidden_dim), topology, **kwargs)
        self.layer2 = RelationalGraphAttention(
            self.layer1.out_dim, int(hidden_dim), topology, **kwargs)
        self.readout = nn.Linear(2 * self.layer2.out_dim, int(graph_dim))
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.layer1.reset_parameters(generator, scheme="glorot")
        self.layer2.reset_parameters(generator, scheme="glorot")
        with torch.no_grad():
            bound = float(np.sqrt(1.0 / (2 * self.layer2.out_dim)))
            self.readout.weight.uniform_(-bound, bound, generator=generator)
            self.readout.bias.zero_()
        self.gate = RelationGate(observation_dim, beta=beta)
        mean = np.zeros(observation_dim) if normalization_mean is None else normalization_mean
        std = np.ones(observation_dim) if normalization_std is None else normalization_std
        mean = torch.as_tensor(mean, dtype=torch.float32).reshape(-1)
        std = torch.as_tensor(std, dtype=torch.float32).reshape(-1)
        if mean.numel() != observation_dim or std.numel() != observation_dim:
            raise ValueError("normalization must match canonical observation dimension")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("normalization mean/std must be finite and std positive")
        self.register_buffer("normalization_mean", mean)
        self.register_buffer("normalization_std", std)
        self._freeze_base()

    @property
    def output_dim(self) -> int:
        return self.readout.out_features

    def _freeze_base(self) -> None:
        for module in (self.layer1, self.layer2, self.readout):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.layer1.eval()
        self.layer2.eval()
        self.readout.eval()
        return self

    def relation_gates(self, observation: torch.Tensor) -> torch.Tensor:
        normalized = ((observation - self.normalization_mean)
                      / self.normalization_std)
        return self.gate(normalized)

    def forward(self, X: torch.Tensor, observation: torch.Tensor,
                *, return_gates: bool = False):
        if X.shape[-2:] != (len(SELECTIVE_NODE_NAMES), SELECTIVE_GRAPH_INPUT_DIM):
            raise ValueError("selective graph feature shape mismatch")
        leading = X.shape[:-2]
        flat_x = X.reshape(-1, X.shape[-2], X.shape[-1])
        flat_o = observation.reshape(-1, observation.shape[-1])
        gates = self.relation_gates(flat_o)
        hidden = torch.tanh(self.layer1(flat_x, relation_gates=gates))
        hidden = torch.tanh(hidden + self.layer2(hidden, relation_gates=gates))
        pooled = torch.cat((hidden.mean(-2), hidden.amax(-2)), dim=-1)
        result = torch.tanh(self.readout(pooled)).reshape(*leading, -1)
        gates = gates.reshape(*leading, -1)
        return (result, gates) if return_gates else result

    def gate_parameters(self):
        return self.gate.parameters()

    def assert_optimizer_safe(self, optimizer: torch.optim.Optimizer) -> None:
        allowed = {id(parameter) for parameter in self.gate.parameters()}
        forbidden = {id(parameter) for module in (self.layer1, self.layer2, self.readout)
                     for parameter in module.parameters()}
        present = {id(parameter) for group in optimizer.param_groups
                   for parameter in group["params"]}
        if present & forbidden:
            raise ValueError("optimizer contains frozen R-GAT/readout parameters")
        if not allowed <= present:
            raise ValueError("optimizer omits one or more relation-gate parameters")

    def load_pretrained_artifact(self, path: str | Path,
                                 *, expected_registry_hash: str) -> dict:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        metadata = dict(payload.get("metadata") or {})
        expected = {
            "format": SELECTIVE_ARTIFACT_FORMAT,
            "graph_schema_hash": SELECTIVE_SCHEMA_HASH,
            "relation_partition_hash": RELATION_PARTITION_HASH,
            "observation_registry_hash": expected_registry_hash,
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"pretrained R-GAT artifact {key} mismatch")
        if not bool(metadata.get("control_sufficiency_passed", False)):
            raise ValueError("pretrained R-GAT failed the control-sufficiency gate")
        self.load_state_dict(payload["base_state"], strict=False)
        self._freeze_base()
        return metadata

    def describe(self) -> dict:
        return {
            "graph_version": self.graph_version,
            "graph_schema_hash": SELECTIVE_SCHEMA_HASH,
            "relation_partition_hash": RELATION_PARTITION_HASH,
            "adaptive_relations": list(ADAPTIVE_RELATIONS),
            "invariant_relations": list(INVARIANT_RELATIONS),
            "gate_formula": "1 + 0.35*tanh(W_r*normalize(O_t)+b_r)",
            "base_frozen": True,
            "ppo_trainable": ["gate.linear.weight", "gate.linear.bias"],
        }


def artifact_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
