"""The three compared arms on the minimal contract, one interface.

``ppo_ontology_rgat``    R-GAT over the ontology graph (``graph_policy``)
``ppo_semantic_flat``    the SAME graph, flattened: node features + edge weights -> MLP
``ppo_vector_canonical`` the raw observation, the last K=3 vectors stacked -> MLP

All three see the same ``LandingObservation`` stream, the same supervisor and
the same plant. The graph arms read the ontology, whose ``PadMemory`` and
``TrackingBias`` carry the past; the vector arm gets the last K observations
AND the same three TrackingBias integrals, so the comparison is representation
against representation, not memory against none (design doc section 8.1).
Without the integrals no memoryless arm can land (bias.py). The flat arm differs from the R-GAT arm only in how the graph is
consumed, which is the comparison the ontology claim rests on.

Every arm returns (mean action m/s^2, log_std, value) and exposes
``inputs(history, graph)`` which builds its own input from the shared stream.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from .constants import DEFAULT_CONSTANTS, LandingConstants
from .graph_policy import GraphPolicyConfig, MinimalGraphPolicy
from .observation import VECTOR_SIZE, LandingObservation
from .ontology import EDGES, FEATURE_DIM, NODES, GraphInstance, schema_hash

ARMS = ("ppo_ontology_rgat", "ppo_semantic_flat", "ppo_vector_canonical")
VECTOR_STACK = 3

# Fixed normalization of the raw vector (physical units in the message).
_VECTOR_SCALE = np.array([10, 10, 10, 3, 3, 3, 3, 3, 8, 1, 1, 1, 5], dtype=np.float32)


def normalize_vector(obs: LandingObservation) -> np.ndarray:
    """Own position is dropped (all three axes), as in the ontology.

    Its origin is the EKF's, not the task's: in the Isaac flight the same
    instant read own z 1.25 m in the gateway's world frame and 0.07 m in PX4's
    local frame. Landing needs the height above the PAD, which is rel_z; an
    origin-dependent input would be an untrained direction the moment the
    origin moves.
    """
    v = obs.vector() / _VECTOR_SCALE
    v[0:3] = 0.0
    return np.clip(v, -1.0, 1.0)


class ObservationStack:
    """The last K normalized observation vectors, oldest first, zero-padded."""

    def __init__(self, k: int = VECTOR_STACK):
        self.k = k
        self.buffer: deque[np.ndarray] = deque(maxlen=k)

    def reset(self) -> None:
        self.buffer.clear()

    def push(self, obs: LandingObservation) -> np.ndarray:
        self.buffer.append(normalize_vector(obs))
        pad = [np.zeros(VECTOR_SIZE, np.float32)] * (self.k - len(self.buffer))
        return np.concatenate(pad + list(self.buffer)).astype(np.float32)


@dataclass(frozen=True)
class MlpConfig:
    hidden: int = 128
    # sigma 0.082. Measured on the /5 clones, 48 held-out seeds: sampled
    # landing 58-71 % at the old -1.1 (sigma 0.333), 96-100 % at sigma <= 0.135
    # for both graph arms. PPO's rollouts are sampled, so this is where it starts.
    initial_log_std: float = -2.5
    minimum_log_std: float = -4.0
    seed: int = 0


class _MlpArm(nn.Module):
    input_kind = ""

    def __init__(self, in_dim: int, config: MlpConfig = MlpConfig(),
                 constants: LandingConstants = DEFAULT_CONSTANTS):
        super().__init__()
        torch.manual_seed(config.seed)
        self.config = config
        self.schema_hash = schema_hash(constants)
        self.register_buffer("action_scale", torch.tensor(
            constants.max_acceleration_m_s2, dtype=torch.float32))
        self.body = nn.Sequential(nn.Linear(in_dim, config.hidden), nn.Tanh(),
                                  nn.Linear(config.hidden, config.hidden), nn.Tanh())
        self.actor = nn.Linear(config.hidden, 3)
        self.critic = nn.Linear(config.hidden, 1)
        self.log_std = nn.Parameter(torch.full((3,), float(config.initial_log_std)))

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)
        z = self.body(x)
        mean = torch.tanh(self.actor(z)) * self.action_scale
        value = self.critic(z).squeeze(-1)
        log_std = self.log_std.clamp(min=self.config.minimum_log_std).expand_as(mean)
        if squeeze:
            mean, log_std, value = mean[0], log_std[0], value[0]
        return mean, log_std, value


class FlatGraphArm(_MlpArm):
    """The ontology graph without the relational structure."""
    input_kind = "graph_flat"

    def __init__(self, config: MlpConfig = MlpConfig(), constants=DEFAULT_CONSTANTS):
        super().__init__(len(NODES) * FEATURE_DIM + len(EDGES), config, constants)

    @staticmethod
    def inputs(stack_vector, graph: GraphInstance):
        return np.concatenate([graph.features.reshape(-1), graph.edge_weight]).astype(np.float32)


class VectorArm(_MlpArm):
    """The raw minimal observation, K frames."""
    input_kind = "vector_stack"

    def __init__(self, config: MlpConfig = MlpConfig(), constants=DEFAULT_CONSTANTS):
        super().__init__(VECTOR_SIZE * VECTOR_STACK + 3, config, constants)

    @staticmethod
    def inputs(stack_vector, graph: GraphInstance):
        from .ontology import NODE_INDEX
        bias = graph.features[NODE_INDEX["TrackingBias"], :3]
        return np.concatenate([stack_vector, bias]).astype(np.float32)


class GraphArm(nn.Module):
    """Adapter so the R-GAT policy takes one packed input like the others."""
    input_kind = "graph"

    def __init__(self, config: GraphPolicyConfig = GraphPolicyConfig(),
                 constants=DEFAULT_CONSTANTS):
        super().__init__()
        self.policy = MinimalGraphPolicy(config, constants)
        self.schema_hash = self.policy.schema_hash
        self.config = config

    @staticmethod
    def inputs(stack_vector, graph: GraphInstance):
        return np.concatenate([graph.features.reshape(-1), graph.edge_weight]).astype(np.float32)

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)
        n = len(NODES) * FEATURE_DIM
        features = x[:, :n].reshape(-1, len(NODES), FEATURE_DIM)
        mean, log_std, value = self.policy(features, x[:, n:])
        if squeeze:
            mean, log_std, value = mean[0], log_std[0], value[0]
        return mean, log_std, value

    def relational_activity(self, x) -> float:
        x = np.asarray(x, dtype=np.float32)
        n = len(NODES) * FEATURE_DIM
        return self.policy.relational_activity(x[:n].reshape(len(NODES), FEATURE_DIM), x[n:])


def build_arm(name: str, seed: int = 0):
    if name == "ppo_ontology_rgat":
        return GraphArm(GraphPolicyConfig(seed=seed))
    if name == "ppo_semantic_flat":
        return FlatGraphArm(MlpConfig(seed=seed))
    if name == "ppo_vector_canonical":
        return VectorArm(MlpConfig(seed=seed))
    raise ValueError(f"unknown arm {name!r}; expected one of {ARMS}")


def save_arm(arm, name: str, path, **metadata) -> None:
    torch.save({"arm": name, "schema_hash": arm.schema_hash,
                "config": asdict(arm.config), "state_dict": arm.state_dict(),
                **metadata}, path)


def load_arm(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["schema_hash"] != schema_hash():
        raise ValueError(f"checkpoint schema {payload['schema_hash']} != {schema_hash()}")
    name = payload["arm"]
    config = payload["config"]
    arm = (GraphArm(GraphPolicyConfig(**config)) if name == "ppo_ontology_rgat"
           else (FlatGraphArm if name == "ppo_semantic_flat" else VectorArm)(MlpConfig(**config)))
    arm.load_state_dict(payload["state_dict"])
    return name, arm, payload
