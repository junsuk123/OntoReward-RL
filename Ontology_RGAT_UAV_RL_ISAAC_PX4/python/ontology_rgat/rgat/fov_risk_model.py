"""R-GAT binary classifier for near-future field-of-view loss."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .fov_graph import (FOV_GRAPH_INPUT_DIM, FOV_GRAPH_VERSION,
                        FOV_NODE_NAMES, FOV_RELATION_NAMES, empty_fov_graph)
from .model import RGATEncoder
from .topology import Topology


FOV_RISK_MODEL_FORMAT = "ontology_rgat.future_fov_loss_classifier/1"


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
    """Two-layer relation-aware graph encoder with one sigmoid risk head."""

    def __init__(self, *, hidden_dim: int = 24, relation_dim: int = 6,
                 heads: int = 1, seed: int = 42):
        super().__init__()
        graph = empty_fov_graph()
        topology = Topology.from_graph(graph, len(FOV_RELATION_NAMES))
        self.encoder = RGATEncoder(
            topology, FOV_GRAPH_INPUT_DIM, int(hidden_dim),
            relation_dim=int(relation_dim), residual=True, heads=int(heads),
            head_aggregation="mean", attention_mode="wirgat",
            attention_style="dot", attention_units=8,
            leaky_relu_slope=0.2, kernel_basis_size=0,
            attn_kernel_basis_size=0, feature_dropout=0.0,
            support_dropout=0.0, softmax_floor=1e-12,
            stable_softmax=True)
        self.risk_head = nn.Linear(self.encoder.out_dim, 1)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.encoder.reset_encoder_parameters(generator)
        with torch.no_grad():
            bound = (6.0 / (self.encoder.out_dim + 1)) ** 0.5
            self.risk_head.weight.uniform_(-bound, bound, generator=generator)
            self.risk_head.bias.zero_()

    @property
    def topology(self):
        return self.encoder.topology

    def forward_logits(self, X: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(X)
        goal = encoded[:, self.topology.goal_node, :]
        return self.risk_head(goal).squeeze(-1)

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
        "hidden_dim": int(model.encoder.hidden_dim),
        "relation_dim": int(model.encoder.layer1.relation_dim),
        "heads": int(model.encoder.layer1.heads),
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
        "model_checksum": state_dict_digest(state),
        "metadata": dict(metadata),
        "state_dict": state,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


class FrozenFOVRiskPredictor:
    """Validated immutable classifier used during PPO and evaluation."""

    def __init__(self, path: str | Path, *, expected_config_hash: str | None = None,
                 device: str | torch.device = "cpu"):
        self.path = Path(path)
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        if payload.get("format") != FOV_RISK_MODEL_FORMAT:
            raise ValueError("unsupported FOV-risk model format")
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
