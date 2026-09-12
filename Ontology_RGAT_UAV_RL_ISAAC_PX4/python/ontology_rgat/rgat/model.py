"""The ontology potential ``Phi(G)`` built from two R-GAT layers."""
from __future__ import annotations

from pathlib import Path
import os
from typing import Any

import numpy as np
import torch
from torch import nn

from ..config import Config
from ..semantic import OntologyGraph
from .layers import RelationalGraphAttention
from .topology import Topology

__all__ = ["RGATEncoder", "PotentialHead", "RGATPotential", "build_potential",
           "save_potential", "load_potential"]


class RGATEncoder(nn.Module):
    """공통 2계층 관계 그래프 인코더.

    기존 potential 모델의 ``layer1``/``layer2`` 파라미터 경로를 그대로
    유지한다. 따라서 이 리팩터링 이전 체크포인트도 동일한 state dict로
    읽을 수 있고, 적응 보상 가중치 모델은 같은 구현을 재사용한다.
    """

    def __init__(self, topology: Topology, in_dim: int, hidden_dim: int, *,
                 relation_dim: int = 6, residual: bool = True, **layer_kwargs: Any):
        super().__init__()
        self.topology = topology
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.residual = bool(residual)
        self.layer1 = RelationalGraphAttention(
            in_dim, hidden_dim, topology, relation_dim=relation_dim, **layer_kwargs)
        self.layer2 = RelationalGraphAttention(
            self.layer1.out_dim, hidden_dim, topology, relation_dim=relation_dim,
            **layer_kwargs)
        if self.residual and self.layer2.out_dim != self.layer1.out_dim:
            raise ValueError(
                "The residual around the second layer needs both layers to "
                f"return the same width, got {self.layer1.out_dim} and "
                f"{self.layer2.out_dim}. Use head_aggregation='mean' or set "
                "cfg.rgat.residual=False.")

    @property
    def out_dim(self) -> int:
        return int(self.layer2.out_dim)

    def reset_encoder_parameters(self, generator: torch.Generator, *,
                                 scheme: str = "matlab", scale: float = 0.12) -> None:
        self.layer1.reset_parameters(generator, scheme=scheme, scale=scale)
        self.layer2.reset_parameters(generator, scheme=scheme, scale=scale)

    def forward(self, X: torch.Tensor, *, return_attention: bool = False):
        if X.dim() == 2:
            X = X.unsqueeze(0)
        h1 = torch.tanh(self.layer1(X))
        if return_attention:
            h2, attention = self.layer2(h1, return_attention=True)
        else:
            h2 = self.layer2(h1)
        encoded = torch.tanh(h2 + h1) if self.residual else torch.tanh(h2)
        return (encoded, attention) if return_attention else encoded


class PotentialHead(nn.Module):
    """목표 노드 임베딩을 bounded scalar potential로 읽는 무상태 head.

    학습 파라미터는 하위 호환성을 위해 계속 ``RGATPotential.w_out``과
    ``b_out``에 둔다. head의 계산 책임만 분리했으므로 기존 state-dict의
    키와 PBRS 수치가 바뀌지 않는다.
    """

    def forward(self, goal_embedding: torch.Tensor, weight: torch.Tensor,
                bias: torch.Tensor) -> torch.Tensor:
        return torch.tanh(goal_embedding @ weight.t() + bias).squeeze(-1)


class RGATPotential(RGATEncoder):
    """Bounded potential ``Phi(G)`` in ``[-1, 1]`` for one graph or a batch.

    Two relational attention layers with a ``tanh`` between them, a residual
    around the second, and a linear read-out from the goal node. The residual
    is why the second layer must return ``hidden_dim``, which rules out
    ``head_aggregation='concat'`` with more than one head unless the residual
    is switched off.
    """

    def __init__(self, topology: Topology, in_dim: int, hidden_dim: int, *,
                 relation_dim: int = 6, residual: bool = True, **layer_kwargs: Any):
        super().__init__(topology, in_dim, hidden_dim, relation_dim=relation_dim,
                         residual=residual, **layer_kwargs)
        self.w_out = nn.Parameter(torch.empty(1, self.layer2.out_dim))
        self.b_out = nn.Parameter(torch.zeros(1))
        self.potential_head = PotentialHead()

    def reset_parameters(self, seed: int, scheme: str = "matlab",
                         scale: float = 0.12) -> None:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.reset_encoder_parameters(generator, scheme=scheme, scale=scale)
        with torch.no_grad():
            if scheme == "matlab":
                self.w_out.normal_(0.0, 1.0, generator=generator).mul_(scale)
            else:
                bound = (6.0 / (self.layer2.out_dim + 1)) ** 0.5
                self.w_out.uniform_(-bound, bound, generator=generator)
            self.b_out.zero_()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """``X`` is ``[batch, nodes, in_dim]`` (or one graph); returns ``[batch]``."""
        encoded = super().forward(X)
        goal = encoded[:, self.topology.goal_node, :]
        return self.potential_head(goal, self.w_out, self.b_out)

    # -------------------------------------------------------------- inference
    @torch.no_grad()
    def predict(self, graph: OntologyGraph) -> float:
        """Potential of one graph, as a plain float.

        Called twice per control step by the PBRS reward inside a loop paced to
        PX4's 50 Hz, so it stays on whatever device the module is on, allocates
        one small tensor and never builds an autograd graph.
        """
        was_training = self.training
        self.eval()
        try:
            device = self.w_out.device
            X = torch.as_tensor(graph.X.T, dtype=self.w_out.dtype, device=device)
            return float(self(X.unsqueeze(0)).item())
        finally:
            self.train(was_training)

    @torch.no_grad()
    def predict_batch(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """Potentials for ``[batch, nodes, in_dim]`` features."""
        was_training = self.training
        self.eval()
        try:
            device = self.w_out.device
            t = torch.as_tensor(np.asarray(X), dtype=self.w_out.dtype, device=device)
            return self(t).detach().cpu().numpy()
        finally:
            self.train(was_training)

    @torch.no_grad()
    def explain(self, graph: OntologyGraph) -> dict[str, Any]:
        """Second-layer edge attention, for interpretability only.

        Attention is learned importance, not causal proof.
        """
        device = self.w_out.device
        X = torch.as_tensor(graph.X.T, dtype=self.w_out.dtype, device=device).unsqueeze(0)
        _, alpha = super().forward(X, return_attention=True)
        edge_alpha = alpha.mean(dim=(0, 1)).cpu().numpy()   # mean over batch and heads
        rel = np.asarray(graph.rel)
        n_rel = len(graph.relation_names)
        relation_mean = np.array([
            float(edge_alpha[rel == r].mean()) if np.any(rel == r) else 0.0
            for r in range(n_rel)])
        return {
            "edge_alpha": edge_alpha,
            "src": np.asarray(graph.src),
            "dst": np.asarray(graph.dst),
            "rel": rel,
            "relation_mean": relation_mean,
            "relation_names": list(graph.relation_names),
            "node_names": list(graph.node_names),
        }


def build_potential(cfg: Config, graph: OntologyGraph,
                    device: torch.device | str = "cpu") -> RGATPotential:
    """Construct and initialise the potential from the configuration."""
    r = cfg.rgat
    topology = Topology.from_graph(graph, cfg.ontology.n_relations)
    model = RGATPotential(
        topology, cfg.ontology.in_dim, r.hidden_dim,
        relation_dim=r.rel_dim, residual=r.residual,
        heads=r.heads, head_aggregation=r.head_aggregation,
        attention_mode=r.attention_mode, attention_style=r.attention_style,
        attention_units=r.attention_units,
        leaky_relu_slope=r.attn_leaky_relu_slope,
        kernel_basis_size=r.kernel_basis_size,
        attn_kernel_basis_size=r.attn_kernel_basis_size,
        feature_dropout=r.feature_dropout, support_dropout=r.support_dropout,
        softmax_floor=r.softmax_floor, stable_softmax=r.stable_softmax)
    # Same offset the retired rgat.initModel used, so a run seeded alike starts
    # from the same distribution of weights.
    model.reset_parameters(cfg.seed + 101)
    return model.to(device)


def _schema(cfg: Config) -> dict[str, Any]:
    return {
        "n_nodes": cfg.ontology.n_nodes,
        "n_relations": cfg.ontology.n_relations,
        "in_dim": cfg.ontology.in_dim,
        "hidden_dim": cfg.rgat.hidden_dim,
        "rel_dim": cfg.rgat.rel_dim,
        "node_names": list(cfg.ontology.node_names),
        "attention_mode": cfg.rgat.attention_mode,
        "attention_style": cfg.rgat.attention_style,
        "heads": cfg.rgat.heads,
        "head_aggregation": cfg.rgat.head_aggregation,
        "kernel_basis_size": cfg.rgat.kernel_basis_size,
        "residual": cfg.rgat.residual,
    }


def save_potential(model: RGATPotential, cfg: Config, path: str | Path,
                   history: dict[str, Any] | None = None) -> Path:
    """Write a CPU checkpoint plus the schema it was trained against.

    Always CPU: a checkpoint that only loads on a machine with a GPU is a
    checkpoint that cannot be shared with the reviewers.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "ontology_rgat.potential/1",
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "schema": _schema(cfg),
        "history": history or {},
        "optimizer_state": getattr(model, "_optimizer_state", None),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def load_potential(path: str | Path, cfg: Config, graph: OntologyGraph,
                   device: torch.device | str = "cpu") -> tuple[RGATPotential, dict[str, Any]]:
    """Load a checkpoint, refusing one trained against a different schema.

    An old fixed-pad model has 11 nodes and a narrower input, so loading it
    here would silently attend over the wrong nodes. The dimensions are checked
    before the weights are touched.
    """
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    saved = blob.get("schema", {})
    wanted = _schema(cfg)
    for key in ("n_nodes", "n_relations", "in_dim", "hidden_dim", "rel_dim",
                "attention_mode", "attention_style", "heads", "head_aggregation",
                "residual"):
        if key in saved and saved[key] != wanted[key]:
            raise ValueError(
                f"R-GAT checkpoint {Path(path).name} was trained with {key}="
                f"{saved[key]!r} but this configuration wants {wanted[key]!r}. "
                "Retrain rather than transferring the model.")
    model = build_potential(cfg, graph, device="cpu")
    model.load_state_dict(blob["state_dict"])
    model._optimizer_state = blob.get("optimizer_state")
    return model.to(device), blob.get("history", {})
