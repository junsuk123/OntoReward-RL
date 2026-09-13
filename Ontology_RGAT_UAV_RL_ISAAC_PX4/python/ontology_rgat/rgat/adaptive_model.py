"""상태 적응형 보상 가중치용 ontology graph, encoder, head와 동결 wrapper."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from ..perception.semantic_observation import (
    SEMANTIC_FEATURE_NAMES, SEMANTIC_NODE_NAMES, SEMANTIC_RELATION_NAMES,
    SemanticObservation)
from ..reward_modes.adaptive_weight import (
    BASELINE_REWARD_WEIGHTS, REWARD_COMPONENT_NAMES,
    RewardComponentNormalizer, constrained_adaptive_weights)
from ..semantic import OntologyGraph
from .model import RGATEncoder
from .topology import Topology


ADAPTIVE_REWARD_NODE_NAMES = (
    "LateralProgress", "VerticalProgress", "VerticalSpeedSafety",
    "UndershootRisk", "YawStability",
)
ADAPTIVE_NODE_NAMES = tuple(SEMANTIC_NODE_NAMES) + ADAPTIVE_REWARD_NODE_NAMES
ADAPTIVE_RELATION_NAMES = tuple(SEMANTIC_RELATION_NAMES)
ADAPTIVE_GRAPH_INPUT_DIM = 6 + len(ADAPTIVE_NODE_NAMES)
ADAPTIVE_GRAPH_VERSION = "ontology_rgat.adaptive_reward_graph/2-hybrid-potential"
ADAPTIVE_MODEL_FORMAT = "ontology_rgat.adaptive_reward_weights/2-hybrid-potential"


def adaptive_reward_graph(observation: SemanticObservation) -> OntologyGraph:
    """visual/onboard state만으로 보상항 개념 노드를 포함한 그래프를 만든다.

    노드 활성값은 보상 성분이나 terminal outcome이 아니다. 현재 영상과
    proprioception에서 얻은 의미 상태만 사용하며, 실제 ``rho``는 그래프와
    분리된 감독/목적 데이터로만 보관한다.
    """
    if not isinstance(observation, SemanticObservation):
        raise TypeError("adaptive reward graph requires a SemanticObservation")
    direct = observation.feature_vector.astype(np.float64)
    feature = dict(zip(SEMANTIC_FEATURE_NAMES, direct))
    base_values = np.r_[
        direct,
        np.mean([feature["keypoint_confidence"],
                 feature["visible_keypoint_fraction"],
                 feature["image_alignment"]]),
        feature["apparent_target_scale"],
        np.mean([feature["image_plane_motion_safety"],
                 feature["scale_rate_safety"]]),
        np.mean([feature["visibility_memory"], feature["reacquisition_trend"],
                 1.0 - feature["visual_loss_risk"]]),
        np.mean([feature["vertical_motion_safety"],
                 feature["attitude_stability"]]),
        0.0,
    ]
    reward_values = np.asarray((
        feature["image_alignment"],
        feature["apparent_target_scale"],
        feature["vertical_motion_safety"],
        1.0 - max(feature["image_alignment"], feature["vertical_motion_safety"]),
        np.mean([feature["image_plane_motion_safety"],
                 feature["attitude_stability"]]),
    ), dtype=np.float64)
    values = np.r_[base_values, reward_values]
    n_nodes = len(ADAPTIVE_NODE_NAMES)
    features = np.zeros((ADAPTIVE_GRAPH_INPUT_DIM, n_nodes), dtype=np.float32)
    features[0] = values
    features[1] = 1.0 - values
    direct_count = len(SEMANTIC_FEATURE_NAMES)
    base_count = len(SEMANTIC_NODE_NAMES)
    features[2, :direct_count] = 1.0
    features[3, direct_count:base_count - 1] = 1.0
    features[3, base_count:] = 1.0
    features[4, [10, 11, base_count + 3]] = 1.0
    features[5, SEMANTIC_NODE_NAMES.index("SafeLanding")] = 1.0
    features[6:] = np.eye(n_nodes, dtype=np.float32)

    names = {name: index for index, name in enumerate(ADAPTIVE_NODE_NAMES)}
    base_edges = [
        ("KeypointConfidence", "PerceptionQuality", "indicates"),
        ("VisibleKeypointFraction", "PerceptionQuality", "indicates"),
        ("ImageAlignment", "PerceptionQuality", "indicates"),
        ("ApparentScale", "ApproachState", "indicates"),
        ("ImagePlaneMotion", "ApproachStability", "indicates"),
        ("ScaleRate", "ApproachStability", "indicates"),
        ("VisibilityMemory", "RecoveryState", "indicates"),
        ("ReacquisitionTrend", "RecoveryState", "indicates"),
        ("VisualLossRisk", "RecoveryState", "constrains"),
        ("PerceptionQuality", "SafeLanding", "supports"),
        ("ApproachState", "ApproachStability", "supports"),
        ("ApproachStability", "SafeLanding", "supports"),
        ("RecoveryState", "SafeLanding", "supports"),
        ("VerticalMotionSafety", "DescentSafety", "supports"),
        ("AttitudeStability", "DescentSafety", "supports"),
        ("DescentSafety", "SafeLanding", "supports"),
        ("BatteryRisk", "SafeLanding", "constrains"),
    ]
    reward_edges = [
        ("ImageAlignment", "LateralProgress", "indicates"),
        ("PerceptionQuality", "LateralProgress", "supports"),
        ("ApparentScale", "VerticalProgress", "indicates"),
        ("VerticalMotionSafety", "VerticalSpeedSafety", "indicates"),
        ("DescentSafety", "VerticalSpeedSafety", "supports"),
        ("ApparentScale", "UndershootRisk", "constrains"),
        ("VerticalMotionSafety", "UndershootRisk", "constrains"),
        ("ImagePlaneMotion", "YawStability", "supports"),
        ("AttitudeStability", "YawStability", "supports"),
        ("LateralProgress", "SafeLanding", "supports"),
        ("VerticalProgress", "SafeLanding", "supports"),
        ("VerticalSpeedSafety", "SafeLanding", "supports"),
        ("UndershootRisk", "SafeLanding", "constrains"),
        ("YawStability", "SafeLanding", "supports"),
    ]
    relation = {name: index for index, name in enumerate(ADAPTIVE_RELATION_NAMES)}
    edge_specs = base_edges + reward_edges
    src = [names[source] for source, _, _ in edge_specs] + list(range(n_nodes))
    dst = [names[target] for _, target, _ in edge_specs] + list(range(n_nodes))
    rel = [relation[kind] for _, _, kind in edge_specs] + [relation["self"]] * n_nodes
    return OntologyGraph(
        X=features, src=np.asarray(src, dtype=np.int64),
        dst=np.asarray(dst, dtype=np.int64), rel=np.asarray(rel, dtype=np.int64),
        goal_node=names["SafeLanding"], node_names=ADAPTIVE_NODE_NAMES,
        relation_names=ADAPTIVE_RELATION_NAMES)


def empty_adaptive_reward_graph() -> OntologyGraph:
    return adaptive_reward_graph(SemanticObservation(
        keypoint_confidence=0.0, visible_keypoint_fraction=0.0,
        image_alignment=0.0, apparent_target_scale=0.0,
        image_plane_motion_safety=0.0, scale_rate_safety=0.0,
        visibility_memory=0.0, reacquisition_trend=0.0,
        vertical_motion_safety=0.0, attitude_stability=0.0,
        battery_risk=0.0, visual_loss_risk=1.0,
        centroid_xy=(0.0, 0.0), raw_scale=0.0))


class MLPGraphEncoder(nn.Module):
    """같은 입력/손실을 쓰는 관계 없는 구조 ablation."""

    def __init__(self, topology: Topology, in_dim: int, hidden_dim: int):
        super().__init__()
        self.topology = topology
        self.network = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Tanh(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.out_dim = int(hidden_dim)

    def forward(self, X, *, return_attention=False):
        encoded = self.network(X)
        if return_attention:
            return encoded, None
        return encoded


class AdaptiveRewardWeightHead(nn.Module):
    """SafeLanding/보상항 노드 쌍에서 항별 logit 5개를 읽는다."""

    def __init__(self, hidden_dim: int, reward_node_indices, goal_node: int):
        super().__init__()
        self.reward_node_indices = tuple(int(v) for v in reward_node_indices)
        self.goal_node = int(goal_node)
        self.pair_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        goal = embeddings[:, self.goal_node:self.goal_node + 1, :]
        goal = goal.expand(-1, len(self.reward_node_indices), -1)
        terms = embeddings[:, self.reward_node_indices, :]
        return self.pair_mlp(torch.cat((goal, terms), dim=-1)).squeeze(-1)


class AdaptiveSemanticPotentialHead(nn.Module):
    """Graph-level observability/safety value used by the PBRS supplement.

    Adaptive weighting alone cannot reward regaining a marker: none of the five
    Shin reward components contains visibility.  The head shares the R-GAT
    encoder, but reads the SafeLanding embedding and a graph mean to provide a
    bounded semantic potential without adding estimator or simulator truth to
    the deployed information boundary.
    """

    def __init__(self, hidden_dim: int, goal_node: int, scale: float = 2.0):
        super().__init__()
        self.goal_node = int(goal_node)
        self.scale = float(scale)
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("semantic potential scale must be positive and finite")
        self.network = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        goal = embeddings[:, self.goal_node]
        pooled = embeddings.mean(dim=1)
        return self.scale * torch.tanh(
            self.network(torch.cat((goal, pooled), dim=-1)).squeeze(-1))


class AdaptiveRewardWeightModel(nn.Module):
    """공통 graph encoder와 항별 pair head로 bounded weights를 출력한다."""

    def __init__(self, topology: Topology, in_dim: int, hidden_dim: int = 24, *,
                 architecture: str = "rgat", relation_dim: int = 6,
                 baseline_weights=BASELINE_REWARD_WEIGHTS,
                 total_weight: float = 5.5, kappa: float = 0.69314718056,
                 epsilon: float = 0.2, potential_scale: float = 2.0,
                 **layer_kwargs: Any):
        super().__init__()
        architecture = str(architecture).lower()
        if architecture not in {"rgat", "gat", "mlp"}:
            raise ValueError("adaptive architecture must be rgat, gat or mlp")
        self.architecture = architecture
        if architecture == "mlp":
            self.encoder = MLPGraphEncoder(topology, in_dim, hidden_dim)
        else:
            encoder_topology = topology
            if architecture == "gat":
                graph = empty_adaptive_reward_graph()
                graph = OntologyGraph(
                    X=graph.X, src=graph.src, dst=graph.dst,
                    rel=np.zeros_like(graph.rel), goal_node=graph.goal_node,
                    node_names=graph.node_names, relation_names=("edge",))
                encoder_topology = Topology.from_graph(graph, 1)
            self.encoder = RGATEncoder(
                encoder_topology, in_dim, hidden_dim, relation_dim=relation_dim,
                **layer_kwargs)
        names = {name: index for index, name in enumerate(ADAPTIVE_NODE_NAMES)}
        self.head = AdaptiveRewardWeightHead(
            self.encoder.out_dim, [names[name] for name in ADAPTIVE_REWARD_NODE_NAMES],
            names["SafeLanding"])
        self.potential_head = AdaptiveSemanticPotentialHead(
            self.encoder.out_dim, names["SafeLanding"], potential_scale)
        self.register_buffer("baseline_weights", torch.as_tensor(
            baseline_weights, dtype=torch.float32))
        self.total_weight = float(total_weight)
        self.kappa = float(kappa)
        self.epsilon = float(epsilon)

    def reset_parameters(self, seed: int) -> None:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        if isinstance(self.encoder, RGATEncoder):
            self.encoder.reset_encoder_parameters(generator)
        for module in list(self.head.modules()) + list(self.potential_head.modules()):
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    module.weight.normal_(0.0, 0.08, generator=generator)
                    module.bias.zero_()
        if isinstance(self.encoder, MLPGraphEncoder):
            for module in self.encoder.modules():
                if isinstance(module, nn.Linear):
                    with torch.no_grad():
                        module.weight.normal_(0.0, 0.08, generator=generator)
                        module.bias.zero_()

    def forward(self, X, *, return_logits=False, return_potential=False):
        if X.dim() == 2:
            X = X.unsqueeze(0)
        embeddings = self.encoder(X)
        logits = self.head(embeddings)
        weights = constrained_adaptive_weights(
            logits, baseline_weights=self.baseline_weights,
            total_weight=self.total_weight, kappa=self.kappa,
            epsilon=self.epsilon)
        potential = self.potential_head(embeddings)
        if return_logits and return_potential:
            return weights, logits, potential
        if return_logits:
            return weights, logits
        if return_potential:
            return weights, potential
        return weights

    def potential(self, X) -> torch.Tensor:
        if X.dim() == 2:
            X = X.unsqueeze(0)
        return self.potential_head(self.encoder(X))

    def attention(self, X):
        if not isinstance(self.encoder, RGATEncoder):
            return None
        _, attention = self.encoder(X, return_attention=True)
        return attention


def adaptive_model_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def adaptive_model_config(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = dict(settings or {})
    return {
        "hidden_dim": int(settings.get("hidden_dim", 24)),
        "architecture": str(settings.get("architecture", "rgat")),
        "relation_dim": int(settings.get("relation_dim", 6)),
        "heads": int(settings.get("heads", 1)),
        "head_aggregation": str(settings.get("head_aggregation", "mean")),
        "attention_mode": str(settings.get("attention_mode", "argat")),
        "attention_style": str(settings.get("attention_style", "sum")),
        "attention_units": int(settings.get("attention_units", 1)),
        "leaky_relu_slope": float(settings.get("attn_leaky_relu_slope", 0.2)),
        "kernel_basis_size": settings.get("kernel_basis_size"),
        "attn_kernel_basis_size": settings.get("attn_kernel_basis_size"),
        "feature_dropout": float(settings.get("feature_dropout", 0.0)),
        "support_dropout": float(settings.get("support_dropout", 0.0)),
        "softmax_floor": float(settings.get("softmax_floor", 1e-9)),
        "stable_softmax": bool(settings.get("stable_softmax", True)),
        "total_weight": float(settings.get("total_weight", 5.5)),
        "kappa": float(settings.get(
            "logit_scale_kappa", settings.get("kappa", 0.69314718056))),
        "epsilon": float(settings.get(
            "baseline_mixture_epsilon", settings.get("epsilon", 0.2))),
        "potential_scale": float(settings.get("potential_scale", 2.0)),
        "baseline_weights": list(settings.get(
            "baseline_weights", BASELINE_REWARD_WEIGHTS.tolist())),
    }


def build_adaptive_reward_model(settings=None, *, seed=42, device="cpu"):
    graph = empty_adaptive_reward_graph()
    top = Topology.from_graph(graph, len(ADAPTIVE_RELATION_NAMES))
    config = adaptive_model_config(settings)
    model = AdaptiveRewardWeightModel(
        top, ADAPTIVE_GRAPH_INPUT_DIM, **config)
    model.reset_parameters(int(seed) + 101)
    return model.to(device)


def freeze_adaptive_reward_model(model: nn.Module) -> str:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    digest = adaptive_model_digest(model)
    model._frozen_sha256 = digest
    return digest


class FrozenAdaptiveRewardWeights:
    """PPO optimizer와 완전히 분리된 deterministic CPU inference wrapper."""

    def __init__(self, path: str | Path, *, expected_config_hash: str | None = None):
        self.path = Path(path).resolve()
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata") or {}
        if payload.get("format") != ADAPTIVE_MODEL_FORMAT:
            raise ValueError("adaptive reward artifact format mismatch")
        if metadata.get("graph_schema_version") != ADAPTIVE_GRAPH_VERSION:
            raise ValueError("adaptive reward graph schema mismatch")
        if not bool(metadata.get("frozen", False)):
            raise ValueError("adaptive reward model must be frozen before PPO")
        if expected_config_hash is not None and (
                metadata.get("dataset_config_hash") != str(expected_config_hash)):
            raise ValueError("adaptive reward configuration mismatch")
        self.model = build_adaptive_reward_model(
            metadata.get("model_config"), seed=int(metadata.get("seed", 42)))
        self.model.load_state_dict(payload["state_dict"])
        digest = adaptive_model_digest(self.model)
        if digest != metadata.get("model_sha256"):
            raise ValueError("adaptive reward model digest mismatch")
        freeze_adaptive_reward_model(self.model)
        self._model_sha256 = digest
        normalization = metadata["normalization"]
        self.normalizer = RewardComponentNormalizer(
            scales=tuple(normalization["scales"]),
            exact_paper_raw=bool(normalization.get("exact_paper_raw", False)),
            source=str(normalization.get("source", "artifact")))
        self.design_id = str(metadata["design_id"])
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.metadata = metadata
        quality = dict(metadata.get("quality_gate") or {})
        if bool(quality.get("enabled", False)) and not bool(quality.get("passed", False)):
            raise ValueError("adaptive reward artifact failed its quality gate")
        self.last_relation_attention = None
        self.weight_source = "frozen_rgat_adaptive"

    def assert_frozen(self) -> None:
        if self.model.training or any(parameter.requires_grad
                                      for parameter in self.model.parameters()):
            raise RuntimeError("adaptive reward model left frozen eval mode")
        if adaptive_model_digest(self.model) != self._model_sha256:
            raise RuntimeError("adaptive reward model changed during PPO")

    @torch.no_grad()
    def __call__(self, graph: OntologyGraph, *, return_latency=False):
        if tuple(graph.node_names) != ADAPTIVE_NODE_NAMES:
            raise ValueError("adaptive reward node schema mismatch")
        if tuple(graph.relation_names) != ADAPTIVE_RELATION_NAMES:
            raise ValueError("adaptive reward relation schema mismatch")
        self.assert_frozen()
        started = time.perf_counter()
        X = torch.as_tensor(graph.X.T[None], dtype=torch.float32)
        weights = self.model(X).squeeze(0).cpu().numpy()
        layer = getattr(self.model.encoder, "layer2", None)
        alpha = None if layer is None else layer.last_attention
        if alpha is not None:
            edge = alpha.mean(dim=(0, 1)).cpu().numpy()
            rel = np.asarray(graph.rel)
            self.last_relation_attention = np.asarray([
                float(edge[rel == relation].mean()) if np.any(rel == relation) else 0.0
                for relation in range(len(ADAPTIVE_RELATION_NAMES))])
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return (weights, elapsed_ms) if return_latency else weights

    @torch.no_grad()
    def transition(self, graph: OntologyGraph, next_graph: OntologyGraph, *,
                   absorbing=False):
        """Evaluate weights and both PBRS potentials with one batched forward."""
        for value in (graph, next_graph):
            if tuple(value.node_names) != ADAPTIVE_NODE_NAMES:
                raise ValueError("adaptive reward node schema mismatch")
            if tuple(value.relation_names) != ADAPTIVE_RELATION_NAMES:
                raise ValueError("adaptive reward relation schema mismatch")
        self.assert_frozen()
        started = time.perf_counter()
        tensors = [graph.X.T] if absorbing else [graph.X.T, next_graph.X.T]
        X = torch.as_tensor(np.stack(tensors), dtype=torch.float32)
        weights, potential = self.model(X, return_potential=True)
        weights_np = weights[0].cpu().numpy()
        phi = float(potential[0].cpu())
        phi_next = 0.0 if absorbing else float(potential[1].cpu())
        layer = getattr(self.model.encoder, "layer2", None)
        alpha = None if layer is None else layer.last_attention
        if alpha is not None:
            edge = alpha[0].mean(dim=0).cpu().numpy()
            rel = np.asarray(graph.rel)
            self.last_relation_attention = np.asarray([
                float(edge[rel == relation].mean()) if np.any(rel == relation) else 0.0
                for relation in range(len(ADAPTIVE_RELATION_NAMES))])
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return weights_np, phi, phi_next, elapsed_ms

    @torch.no_grad()
    def potential(self, graph: OntologyGraph) -> float:
        self.assert_frozen()
        X = torch.as_tensor(graph.X.T[None], dtype=torch.float32)
        return float(self.model.potential(X)[0].cpu())

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    @torch.no_grad()
    def explain_attention(self, graph: OntologyGraph):
        X = torch.as_tensor(graph.X.T[None], dtype=torch.float32)
        attention = self.model.attention(X)
        return None if attention is None else attention.cpu().numpy()


def save_adaptive_reward_artifact(path: str | Path, model: nn.Module, *,
                                  metadata: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": ADAPTIVE_MODEL_FORMAT,
               "state_dict": {name: value.detach().cpu()
                              for name, value in model.state_dict().items()},
               "metadata": metadata}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    manifest_path = path.with_suffix(".manifest.json")
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_manifest.write_text(json.dumps(metadata, indent=2, allow_nan=False),
                                  encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    return path
