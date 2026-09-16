"""Direct R-GAT scalar readout of near-future field-of-view unavailability.

The regression target is the fraction of the next ``H`` control steps in which
the geometric pad centre is outside the camera frustum.  It is a conditional
expectation, not a binary-loss probability.

The scalar is produced by the graph itself: the second relational layer has
``units=1`` and the ``FutureFOVUnavailability`` node of that layer is the
output.  There is no separate readout MLP, no separate linear head, and no
residual adding the 24-dimensional first-layer state to the 1-dimensional
output.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .fov_graph import (FOV_GOAL_NODE, FOV_GRAPH_INPUT_DIM, FOV_GRAPH_VERSION,
                        FOV_NODE_NAMES, FOV_RELATION_NAMES, empty_fov_graph)
from .layers import RelationalGraphAttention
from .topology import Topology


FOV_RISK_MODEL_FORMAT = "ontology_rgat.future_fov_unavailability_readout/2"

# Layer settings shared by both relational layers. They are fixed here rather
# than exposed, so a checkpoint cannot silently change the attention kernel.
_LAYER_KWARGS = dict(
    head_aggregation="mean", attention_mode="wirgat", attention_style="dot",
    attention_units=8, leaky_relu_slope=0.2, kernel_basis_size=0,
    attn_kernel_basis_size=0, feature_dropout=0.0, support_dropout=0.0,
    softmax_floor=1e-12, stable_softmax=True)


def state_dict_digest(state_dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        value = np.ascontiguousarray(tensor.detach().cpu().numpy())
        digest.update(name.encode("utf-8"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


class FOVRiskModel(nn.Module):
    """Two relational layers whose second layer emits the scalar directly."""

    def __init__(self, *, hidden_dim: int = 24, relation_dim: int = 6,
                 heads: int = 1, seed: int = 42):
        super().__init__()
        graph = empty_fov_graph()
        self.topology = Topology.from_graph(graph, len(FOV_RELATION_NAMES))
        self.hidden_dim = int(hidden_dim)
        self.layer1 = RelationalGraphAttention(
            FOV_GRAPH_INPUT_DIM, self.hidden_dim, self.topology,
            relation_dim=int(relation_dim), heads=int(heads), **_LAYER_KWARGS)
        self.layer2 = RelationalGraphAttention(
            self.layer1.out_dim, 1, self.topology,
            relation_dim=int(relation_dim), heads=int(heads), **_LAYER_KWARGS)
        if self.layer2.out_dim != 1:
            raise ValueError("the FOV readout layer must return one unit per node")
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.layer1.reset_parameters(generator, scheme="matlab", scale=0.12)
        self.layer2.reset_parameters(generator, scheme="matlab", scale=0.12)

    @property
    def goal_node(self) -> int:
        return int(self.topology.goal_node)

    def module_description(self) -> dict[str, Any]:
        return {
            "name": type(self).__name__,
            "kind": "direct R-GAT scalar readout",
            "output_node": FOV_GOAL_NODE,
            "output_activation": "sigmoid",
            "separate_output_mlp": False,
            "separate_linear_readout": False,
            "last_layer_residual": False,
            "layers": [{
                "name": f"R-GAT layer {index}",
                "input_dim": int(layer.in_dim),
                "units": int(layer.units),
                "output_dim": int(layer.out_dim),
                "attention_heads": int(layer.heads),
                "head_aggregation": str(layer.head_aggregation),
            } for index, layer in enumerate((self.layer1, self.layer2), start=1)],
        }

    def forward_logits(self, X: torch.Tensor) -> torch.Tensor:
        if X.dim() == 2:
            X = X.unsqueeze(0)
        hidden = torch.tanh(self.layer1(X))
        # No residual here: the first layer is 24-wide and the readout is a
        # single unit, so there is nothing to add without reintroducing a
        # projection outside the graph.
        return self.layer2(hidden)[:, self.goal_node, 0]

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(X))

    @torch.no_grad()
    def predict(self, graph) -> float:
        parameter = next(self.parameters())
        X = torch.as_tensor(
            graph.X.T, dtype=parameter.dtype, device=parameter.device).unsqueeze(0)
        was_training = self.training
        self.eval()
        try:
            return float(self(X).item())
        finally:
            self.train(was_training)

    @torch.no_grad()
    def predict_batch(self, X) -> np.ndarray:
        parameter = next(self.parameters())
        tensor = torch.as_tensor(
            np.asarray(X), dtype=parameter.dtype, device=parameter.device)
        was_training = self.training
        self.eval()
        try:
            return self(tensor).detach().cpu().numpy()
        finally:
            self.train(was_training)


def model_config(model: FOVRiskModel) -> dict[str, int]:
    return {
        "hidden_dim": int(model.hidden_dim),
        "relation_dim": int(model.layer1.relation_dim),
        "heads": int(model.layer1.heads),
    }


def save_fov_risk_model(model: FOVRiskModel, path: str | Path, *, metadata: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    payload = {
        "format": FOV_RISK_MODEL_FORMAT,
        "graph_version": FOV_GRAPH_VERSION,
        "node_names": list(FOV_NODE_NAMES),
        "relation_names": list(FOV_RELATION_NAMES),
        "model_config": model_config(model),
        "architecture": model.module_description(),
        "model_checksum": state_dict_digest(state),
        "metadata": dict(metadata),
        "state_dict": state,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


class FrozenFOVRiskPredictor:
    """Validated immutable scalar readout used during PPO and evaluation."""

    def __init__(self, path: str | Path, *, expected_config_hash: str | None = None,
                 device: str | torch.device = "cpu"):
        self.path = Path(path)
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        if payload.get("format") != FOV_RISK_MODEL_FORMAT:
            raise ValueError("unsupported FOV-risk model format")
        readout = dict(payload.get("architecture") or {})
        if (readout.get("separate_output_mlp") or readout.get("separate_linear_readout")
                or readout.get("last_layer_residual")):
            raise ValueError("FOV readout must come from the graph, not a separate head")
        if payload.get("graph_version") != FOV_GRAPH_VERSION:
            raise ValueError("FOV-risk graph version mismatch")
        if tuple(payload.get("node_names", ())) != FOV_NODE_NAMES:
            raise ValueError("FOV-risk ontology node schema mismatch")
        if tuple(payload.get("relation_names", ())) != FOV_RELATION_NAMES:
            raise ValueError("FOV-risk ontology relation schema mismatch")
        self.metadata = dict(payload.get("metadata") or {})
        if (expected_config_hash is not None
                and self.metadata.get("dataset_config_hash") != expected_config_hash):
            raise ValueError("FOV-risk model configuration mismatch")
        config = dict(payload.get("model_config") or {})
        self.model = FOVRiskModel(**config).to(device)
        self.model.load_state_dict(payload["state_dict"])
        checksum = state_dict_digest(self.model.state_dict())
        if checksum != payload.get("model_checksum"):
            raise ValueError("FOV-risk model checksum mismatch")
        self.sha256 = checksum
        self.design_id = f"fov-risk-{checksum[:12]}"
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self._frozen_checksum = checksum

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    def predict(self, graph) -> float:
        self.assert_frozen()
        return self.model.predict(graph)

    def assert_frozen(self) -> None:
        if self.model.training:
            raise RuntimeError("FOV-risk R-GAT must remain in evaluation mode")
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("FOV-risk R-GAT parameters must remain frozen")
        if state_dict_digest(self.model.state_dict()) != self._frozen_checksum:
            raise RuntimeError("FOV-risk R-GAT parameters changed during PPO")

    def manifest(self) -> dict[str, Any]:
        return {
            "design_id": self.design_id,
            "model_checksum": self.sha256,
            "metadata": json.loads(json.dumps(self.metadata)),
        }
