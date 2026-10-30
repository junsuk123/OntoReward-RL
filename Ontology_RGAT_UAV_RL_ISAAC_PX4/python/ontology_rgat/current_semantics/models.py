"""Budget-matched actor-critics of spatial_current_semantics_v1 (guide 8.3-8.4).

The four arms differ only in what the actor reads -- raw21, phi44, or raw21
plus an 8-dimensional readout of the untyped or the typed graph -- and share
one raw21 critic design and a global, unclamped log_std.  The policy is a
latent Gaussian squashed once, ``action = a_max * tanh(z)``, with the
log-probability taken in latent space; nothing else rewrites the action, so
the action PPO scores is the action the environment receives.  Every
parameter is drawn from a private generator keyed by (seed, component):
construction never reads or advances the global torch RNG, the same seed gives
every arm a bit-identical critic, and the two graph arms start from the same
input projection and readout, differing at initialisation in the relational
layer alone.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import pickle
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import MODEL_VERSION, SCHEMA_ID
from .contracts import (ACTION_DIMENSION, ARM_ACTOR_INPUT, ARMS, ENCODER_RELATIONS,
                        FEATURE_DIMENSION, MODEL_ARCHITECTURE, NODE_INPUT_DIMENSION, NODE_ORDER,
                        OBSERVATION_DIMENSION, CurrentSemanticsConfig, contract_manifest)
from .features import NODE_FEATURES, NODE_ID_COLUMN, NODE_IDS, FeatureFrame, raw_from_flat
from .graph import GraphSpec, graph_for_arm


NEGATIVE_SLOPE = float(MODEL_ARCHITECTURE["attention_negative_slope"])
ATTENTION_INIT_STD = float(MODEL_ARCHITECTURE["attention_init_std"])
ACTOR_OUTPUT_INIT_SCALE = float(MODEL_ARCHITECTURE["actor_output_init_scale"])
if MODEL_ARCHITECTURE["relational_weight_init"] != "normal(0, 1/node_embedding)":
    raise AssertionError("models.py implements only the registered LeCun W_r init")
PARAMETER_GROUPS = ("encoder", "actor_head", "log_std", "critic")
CHECKPOINT_HASHES = ("algorithm_hash", "feature_hash", "graph_hash", "calibration_hash",
                     "task_hash")
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def component_generator(seed: int, component: str) -> torch.Generator:
    """The private stream of one component, a function of (seed, component) only,
    so neither the arm nor the construction order can shift it."""
    material = hashlib.sha256(f"{SCHEMA_ID}:{seed}:{component}".encode("utf-8")).digest()
    return torch.Generator().manual_seed(int.from_bytes(material[:8], "little"))


def _linear(inputs: int, outputs: int, generator: torch.Generator, *,
            bias: bool = True) -> nn.Linear:
    """nn.Linear without its global-RNG default init: U(+-1/sqrt(fan_in)) from ``generator``."""
    layer = nn.utils.skip_init(nn.Linear, inputs, outputs, bias=bias)
    bound = 1.0 / math.sqrt(inputs)
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.copy_((torch.rand(parameter.shape, generator=generator) * 2.0 - 1.0)
                            * bound)
    return layer


def _mlp(inputs: int, hidden: int, outputs: int, generator: torch.Generator) -> nn.Sequential:
    return nn.Sequential(_linear(inputs, hidden, generator), nn.Tanh(),
                         _linear(hidden, outputs, generator))


class RelationalAttentionLayer(nn.Module):
    """One relational attention layer over a fixed edge list.

    Edge j -> i of relation r sends ``m_ij = W_r h_j`` scored by
    ``leaky_relu(att_src . m_ij + att_dst . h_i + relation_bias_r)``; the
    softmax runs over the incoming edges of i only, and
    ``h_i' = tanh(sum_j alpha_ij m_ij + bias)``.  A node with no incoming
    edge would get ``tanh(bias)``; every registered graph gives each node a
    self loop.  The per-target maximum is subtracted as a constant, which
    leaves the softmax and its gradient exact.
    """

    def __init__(self, dim: int, relations: int, *, generator: torch.Generator):
        super().__init__()
        self.dim, self.relations = int(dim), int(relations)
        self.weight = nn.Parameter(torch.empty(self.relations, self.dim, self.dim))
        self.relation_bias = nn.Parameter(torch.empty(self.relations))
        self.att_src = nn.Parameter(torch.empty(self.dim))
        self.att_dst = nn.Parameter(torch.empty(self.dim))
        self.bias = nn.Parameter(torch.empty(self.dim))
        self.reset_parameters(generator)

    def reset_parameters(self, generator: torch.Generator) -> None:
        with torch.no_grad():
            self.weight.copy_(torch.randn(self.weight.shape, generator=generator)
                              / math.sqrt(self.dim))
            for parameter in (self.att_src, self.att_dst):
                parameter.copy_(torch.randn(parameter.shape, generator=generator)
                                * ATTENTION_INIT_STD)
            self.relation_bias.zero_()
            self.bias.zero_()

    def forward(self, h: torch.Tensor, source: torch.Tensor, target: torch.Tensor,
                relation: torch.Tensor) -> torch.Tensor:
        if h.dim() != 3 or h.shape[-1] != self.dim:
            raise ValueError(f"node states must be (B, N, {self.dim}), got {tuple(h.shape)}")
        batch, nodes, _ = h.shape
        messages = torch.einsum("eoi,bei->beo", self.weight[relation], h[:, source])
        score = F.leaky_relu(messages @ self.att_src + (h @ self.att_dst)[:, target]
                             + self.relation_bias[relation], NEGATIVE_SLOPE)
        with torch.no_grad():
            peak = score.new_full((batch, nodes), -math.inf).scatter_reduce(
                1, target.expand(batch, -1), score, reduce="amax", include_self=True)
        weight = torch.exp(score - peak[:, target])
        total = weight.new_zeros((batch, nodes)).index_add(1, target, weight)
        alpha = weight / total[:, target]
        aggregated = h.new_zeros((batch, nodes, self.dim)).index_add(
            1, target, alpha.unsqueeze(-1) * messages)
        return torch.tanh(aggregated + self.bias)


class GraphEncoder(nn.Module):
    """``tanh(Linear)`` node projection, ONE relational layer applied ``hops``
    times, and a bias-free linear readout of ``spec.readout_node``.

    Sharing the layer across hops is what makes the hop count a reach, not
    extra capacity: two hops let every primitive node reach LandingContext
    through P, O or C without adding a parameter.
    """

    def __init__(self, spec: GraphSpec, *, dim: int, hops: int, seed: int,
                 node_input: int = NODE_INPUT_DIMENSION):
        super().__init__()
        if int(hops) < 1:
            raise ValueError("the encoder needs at least one hop")
        self.spec, self.dim, self.hops, self.node_input = spec, int(dim), int(hops), int(node_input)
        self.readout_index = spec.nodes.index(spec.readout_node)
        self.input = _linear(self.node_input, self.dim, component_generator(seed, "encoder.input"))
        self.layer = RelationalAttentionLayer(self.dim, len(spec.relations),
                                              generator=component_generator(seed, "encoder.layer"))
        self.readout = _linear(self.dim, self.dim, component_generator(seed, "encoder.readout"),
                               bias=False)
        for name, index in zip(("source", "target", "relation"), spec.edge_index()):
            self.register_buffer(name, torch.as_tensor(index, dtype=torch.long), persistent=False)

    def node_states(self, nodes: torch.Tensor) -> torch.Tensor:
        shape = (len(self.spec.nodes), self.node_input)
        if nodes.dim() != 3 or tuple(nodes.shape[1:]) != shape:
            raise ValueError(f"node input must be (B, {shape[0]}, {shape[1]}), "
                             f"got {tuple(nodes.shape)}")
        h = torch.tanh(self.input(nodes))
        for _ in range(self.hops):
            h = self.layer(h, self.source, self.target, self.relation)
        return h

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        return self.readout(self.node_states(nodes)[:, self.readout_index])


_INPUT_TAILS = (("raw21", (OBSERVATION_DIMENSION,)), ("flat44", (FEATURE_DIMENSION,)),
                ("nodes", (len(NODE_ORDER), NODE_INPUT_DIMENSION)))


def _check_inputs(raw21: Any, flat44: Any, nodes: Any) -> None:
    for (name, tail), value in zip(_INPUT_TAILS, (raw21, flat44, nodes)):
        if not isinstance(value, torch.Tensor) or tuple(value.shape[1:]) != tail \
                or value.dim() != 1 + len(tail):
            shape = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value)
            raise ValueError(f"{name} must be a tensor of shape (B, "
                             f"{', '.join(map(str, tail))}), got {shape}")
    if not raw21.shape[0] == flat44.shape[0] == nodes.shape[0]:
        raise ValueError("raw21, flat44 and nodes must describe the same batch of decisions")


class CurrentActorCritic(nn.Module):
    """One arm: actor head over its registered input, the common raw21 critic,
    and a global log_std that is the policy's sigma exactly (no clamp).

    ``forward`` takes all three views of a decision so every arm is called the
    same way; each reads only its ``contracts.ARM_ACTOR_INPUT``.  The critic
    never reads phi44 or the graph.
    """

    def __init__(self, arm: str, config: CurrentSemanticsConfig, *, seed: int):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; registered: {ARMS}")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative int")
        design = config.model
        self.arm, self.config, self.seed = arm, config, seed
        self.actor_input = ARM_ACTOR_INPUT[arm]
        self.encoder: GraphEncoder | None = None
        if arm in ENCODER_RELATIONS:
            spec = graph_for_arm(arm)
            if spec.nodes != NODE_ORDER or spec.readout_node != design.readout:
                raise ValueError(f"{arm}: graph nodes/readout disagree with the feature layout")
            self.encoder = GraphEncoder(spec, dim=design.node_embedding, hops=design.hops,
                                        seed=seed)
        inputs = {"raw21": OBSERVATION_DIMENSION, "flat44": FEATURE_DIMENSION,
                  "raw21+graph": OBSERVATION_DIMENSION + design.node_embedding}
        self.actor_head = _mlp(inputs[self.actor_input], design.head(arm), ACTION_DIMENSION,
                               component_generator(seed, "actor_head"))
        with torch.no_grad():  # a near-zero initial mean (MODEL_ARCHITECTURE)
            self.actor_head[-1].weight.mul_(ACTOR_OUTPUT_INIT_SCALE)
            self.actor_head[-1].bias.zero_()
        self.critic = _mlp(OBSERVATION_DIMENSION, design.critic_hidden, 1,
                           component_generator(seed, "critic"))
        self.log_std = nn.Parameter(torch.full((ACTION_DIMENSION,),
                                               float(design.initial_log_std)))
        self.register_buffer("action_maximum",
                             torch.tensor(config.action.maximum_m_s2, dtype=torch.float32),
                             persistent=False)

    @property
    def is_graph(self) -> bool:
        return self.encoder is not None

    def graph_readout(self, nodes: torch.Tensor) -> torch.Tensor:
        """g8, the LandingContext readout the actor concatenates to raw21 (for audits)."""
        if self.encoder is None:
            raise ValueError(f"{self.arm} has no graph encoder")
        return self.encoder(nodes)

    def forward(self, raw21: torch.Tensor, flat44: torch.Tensor,
                nodes: torch.Tensor) -> torch.Tensor:
        """Latent mean (B, 3)."""
        _check_inputs(raw21, flat44, nodes)
        if self.actor_input == "raw21":
            features = raw21
        elif self.actor_input == "flat44":
            features = flat44
        else:
            features = torch.cat((raw21, self.graph_readout(nodes)), dim=-1)
        return self.actor_head(features)

    def value(self, raw21: torch.Tensor) -> torch.Tensor:
        """V(raw21), shape (B,)."""
        name, tail = _INPUT_TAILS[0]
        if not isinstance(raw21, torch.Tensor) or raw21.dim() != 2 \
                or tuple(raw21.shape[1:]) != tail:
            raise ValueError(f"{name} must be a tensor of shape (B, {tail[0]})")
        return self.critic(raw21).squeeze(-1)

    def action_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.action_maximum * torch.tanh(latent)

    def _latent_log_prob(self, latent: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        if latent.shape != mean.shape:
            raise ValueError(f"latent must have shape {tuple(mean.shape)}, "
                             f"got {tuple(latent.shape)}")
        z = (latent - mean) / self.log_std.exp()
        return (-0.5 * z * z - self.log_std - _LOG_SQRT_2PI).sum(-1)

    def log_prob(self, latent: torch.Tensor, raw21: torch.Tensor, flat44: torch.Tensor,
                 nodes: torch.Tensor) -> torch.Tensor:
        """log N(latent; mu, diag(exp(log_std)^2)) summed over the axes, shape (B,)."""
        return self._latent_log_prob(latent, self(raw21, flat44, nodes))

    def entropy(self) -> torch.Tensor:
        """Entropy of the latent Gaussian; state-independent because log_std is global."""
        return (0.5 + _LOG_SQRT_2PI + self.log_std).sum()

    def act(self, raw21: torch.Tensor, flat44: torch.Tensor, nodes: torch.Tensor, *,
            deterministic: bool,
            generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
        """latent, action = a_max tanh(latent), its latent log-probability and V(raw21)."""
        if not isinstance(deterministic, bool):
            raise TypeError("deterministic must be a bool")
        mean = self(raw21, flat44, nodes)
        latent = mean
        if not deterministic:
            noise = torch.randn(mean.shape, generator=generator, dtype=mean.dtype,
                                device=mean.device)
            latent = mean + self.log_std.exp() * noise
        return {"latent": latent, "action": self.action_from_latent(latent),
                "log_prob": self._latent_log_prob(latent, mean), "value": self.value(raw21)}

    def _parameter_groups(self) -> tuple[tuple[str, list[nn.Parameter]], ...]:
        return (("encoder", [] if self.encoder is None else list(self.encoder.parameters())),
                ("actor_head", list(self.actor_head.parameters())),
                ("log_std", [self.log_std]),
                ("critic", list(self.critic.parameters())))

    def optimizer_groups(self) -> list[dict[str, Any]]:
        """Named param groups covering every trainable parameter exactly once
        (an arm without an encoder has no encoder group)."""
        groups = [{"name": name, "params": [p for p in params if p.requires_grad]}
                  for name, params in self._parameter_groups()]
        covered = sorted(id(p) for group in groups for p in group["params"])
        if covered != sorted(id(p) for p in self.parameters() if p.requires_grad):
            raise AssertionError("optimizer groups do not cover the trainable parameters "
                                 "exactly once")
        return [group for group in groups if group["params"]]

    def parameter_report(self) -> dict[str, int]:
        """Measured trainable counts, comparable to ``contracts.design_parameter_counts``."""
        counts = {name: sum(p.numel() for p in params if p.requires_grad)
                  for name, params in self._parameter_groups()}
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if total != sum(counts.values()):
            raise AssertionError("a trainable parameter lies outside "
                                 f"{PARAMETER_GROUPS}")
        return {"encoder": counts["encoder"], "actor_head": counts["actor_head"],
                "critic": counts["critic"], "log_std": counts["log_std"], "total": total}


# ----------------------------------------------------------------- tensors

def _read_only(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


_NUMERIC = np.zeros((len(NODE_ORDER), NODE_INPUT_DIMENSION), dtype=bool)
for _row, _node in enumerate(NODE_ORDER):
    _NUMERIC[_row, :len(NODE_FEATURES[_node])] = True
_PADDING = ~_NUMERIC
_PADDING[:, NODE_ID_COLUMN] = False
_NUMERIC, _PADDING = _read_only(_NUMERIC), _read_only(_PADDING)
del _row, _node


def tensors_from_batch(raw: Any, flat: Any,
                       nodes: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """float32 copies of (raw (N, 21), flat (N, 44), nodes (N, 8, 10)).

    Refused unless the three are views of the same decisions: raw21 is what
    flat44 carries by name, nodes carry flat44 in the registered slots with
    the registered node ids, and padding is exactly 0.  The encoder reads all
    ten columns of every row through one shared projection and does not mask
    padding, so this is where a corrupt layout is stopped.
    """
    raw, flat, nodes = (np.asarray(value, dtype=np.float64) for value in (raw, flat, nodes))
    rows = raw.shape[0] if raw.ndim == 2 else -1
    if (raw.shape != (rows, OBSERVATION_DIMENSION) or flat.shape != (rows, FEATURE_DIMENSION)
            or nodes.shape != (rows, len(NODE_ORDER), NODE_INPUT_DIMENSION)):
        raise ValueError(f"expected raw (N, {OBSERVATION_DIMENSION}), flat (N, "
                         f"{FEATURE_DIMENSION}), nodes (N, {len(NODE_ORDER)}, "
                         f"{NODE_INPUT_DIMENSION}); got {raw.shape}, {flat.shape}, {nodes.shape}")
    if not (np.isfinite(raw).all() and np.isfinite(flat).all() and np.isfinite(nodes).all()):
        raise ValueError("model inputs must be finite")
    if not np.array_equal(raw_from_flat(flat), raw):
        raise ValueError("raw21 is not the raw content of flat44")
    if not np.array_equal(nodes[:, _NUMERIC], flat):
        raise ValueError("nodes do not carry flat44 in the registered slots")
    if np.any(nodes[:, _PADDING] != 0.0):
        raise ValueError("node padding must be exactly 0")
    if not np.array_equal(nodes[..., NODE_ID_COLUMN],
                          np.broadcast_to(NODE_IDS, (rows, len(NODE_ORDER)))):
        raise ValueError("node-id column differs from the registered ids")
    return tuple(torch.tensor(value, dtype=torch.float32) for value in (raw, flat, nodes))


def tensors_from_frames(frames: Iterable[FeatureFrame]
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``tensors_from_batch`` of a sequence of ``features.build_features`` frames."""
    frames = list(frames)
    if not frames:
        raise ValueError("no frames")
    if not all(isinstance(frame, FeatureFrame) for frame in frames):
        raise TypeError("frames must be features.FeatureFrame")
    return tensors_from_batch(np.stack([frame.raw for frame in frames]),
                              np.stack([frame.flat for frame in frames]),
                              np.stack([frame.nodes for frame in frames]))


# -------------------------------------------------------------- checkpoint

def checkpoint_metadata(model: CurrentActorCritic, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """What a checkpoint binds to; refused unless ``manifest`` is the complete
    contract of the model's own config (a stale or partial manifest never binds)."""
    if not isinstance(manifest, Mapping) or manifest.get("schema_id") != SCHEMA_ID:
        raise ValueError(f"manifest is not a {SCHEMA_ID} contract")
    incomplete = [key for key in CHECKPOINT_HASHES if not manifest.get(key)]
    if incomplete:
        raise ValueError(f"manifest has no {incomplete}; an incomplete contract cannot bind "
                         "a checkpoint")
    own = contract_manifest(model.config)
    stale = [key for key in CHECKPOINT_HASHES if manifest[key] != own[key]]
    if stale:
        raise ValueError(f"manifest {stale} differ from the contract of the model's config")
    return {"schema_id": SCHEMA_ID, "model_version": MODEL_VERSION, "arm": model.arm,
            **{key: manifest[key] for key in CHECKPOINT_HASHES},
            "parameter_report": model.parameter_report()}


def save_checkpoint(path: str | Path, model: CurrentActorCritic, *,
                    manifest: Mapping[str, Any], extra: Any = None) -> Path:
    """Write atomically, and only what a ``weights_only`` load reads back:
    ``extra`` holds tensors and plain python types, never numpy scalars."""
    path = Path(path)
    payload = {"schema_id": SCHEMA_ID, "metadata": checkpoint_metadata(model, manifest),
               "model_state": {key: value.detach().cpu().clone()
                               for key, value in model.state_dict().items()},
               "extra": extra}
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        try:
            torch.load(temporary, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError as exc:
            raise ValueError("checkpoint would not load with weights_only=True; keep `extra` "
                             f"to tensors and plain python types ({exc})") from exc
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_checkpoint(path: str | Path, model: CurrentActorCritic, *,
                    manifest: Mapping[str, Any],
                    allow_foreign_task: bool = False) -> dict[str, Any]:
    """Load into ``model`` after every binding matches; the model is untouched on refusal.

    Returns ``{"metadata", "extra"}``.  A payload without this schema_id --
    a direct_policy checkpoint, a bare state_dict -- is refused before any
    weight is read.  ``allow_foreign_task`` (diagnostic re-flights only) accepts a
    checkpoint TRAINED on another task (training reward) when every other binding --
    algorithm, features, graph, calibration -- still matches: the policy reads the
    same observation through the same model, and its verdicts are the same task's.
    """
    expected = checkpoint_metadata(model, manifest)
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as exc:
        raise ValueError(f"{path} is not a {SCHEMA_ID} checkpoint ({exc})") from exc
    schema = payload.get("schema_id") if isinstance(payload, dict) else None
    if schema != SCHEMA_ID:
        raise ValueError(f"{path} is not a {SCHEMA_ID} checkpoint (schema_id {schema!r})")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{path} carries no checkpoint metadata")
    differing = sorted(key for key in set(expected) | set(metadata)
                       if metadata.get(key) != expected.get(key)
                       and not (allow_foreign_task and key == "task_hash"))
    if differing:
        raise ValueError(f"checkpoint is incompatible with this model and contract on "
                         f"{differing}")
    state, own = payload.get("model_state"), model.state_dict()
    if not isinstance(state, dict) or set(state) != set(own) or any(
            not isinstance(state[key], torch.Tensor) or state[key].shape != own[key].shape
            for key in own):
        raise ValueError("checkpoint model_state does not match the model's parameters")
    model.load_state_dict(state, strict=True)
    return {"metadata": metadata, "extra": payload.get("extra")}
