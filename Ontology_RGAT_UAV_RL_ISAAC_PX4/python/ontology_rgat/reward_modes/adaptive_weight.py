"""Shin Table-III 성분과 상태 적응형 보상 가중치 조합.

이 모듈의 성분 함수는 순수 함수다. 오프라인 R-GAT 설계 데이터와 PPO
온라인 보상이 같은 구현을 호출하므로 수식의 이중 구현을 방지한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .shin2026 import ShinRewardConfig, active_perception_reward
from .sparse import sparse_terminal_reward


REWARD_COMPONENT_NAMES = (
    "lateral_progress", "vertical_progress", "vertical_speed_safety",
    "undershoot_risk", "yaw_stability",
)
BASELINE_REWARD_WEIGHTS = np.asarray((1.0, 1.0, 0.5, 1.0, 2.0),
                                     dtype=np.float64)
TOTAL_REWARD_WEIGHT = 5.5
DEFAULT_COMPONENT_SCALES = np.asarray((1.0, 1.0, 1.5, 3.0, np.pi / 2.0),
                                      dtype=np.float64)


def optional_estimation_error_component(next_estimation_loss: float, *,
                                        tau: float = 0.01) -> float:
    """향후 SE 6번째 항 ``-clip(L_est_next-tau,0,1)``용 확장 interface."""
    value = float(next_estimation_loss)
    if not np.isfinite(value) or not np.isfinite(float(tau)):
        raise ValueError("estimation component inputs must be finite")
    return -float(np.clip(value - float(tau), 0.0, 1.0))


def shin_reward_components(current_relative_state, next_relative_state, action,
                           *, next_uav_vertical_velocity: float) -> np.ndarray:
    """전이 ``t-1 -> t``에서 논문 Table-III의 무가중 5개 성분을 계산한다."""
    previous = np.asarray(current_relative_state, dtype=np.float64).reshape(-1)
    following = np.asarray(next_relative_state, dtype=np.float64).reshape(-1)
    command = np.asarray(action, dtype=np.float64).reshape(-1)
    if previous.shape != (6,) or following.shape != (6,):
        raise ValueError("reward components require two six-dimensional relative states")
    if command.shape != (4,):
        raise ValueError("reward components require a four-dimensional command")
    if not (np.isfinite(previous).all() and np.isfinite(following).all()
            and np.isfinite(command).all()
            and np.isfinite(float(next_uav_vertical_velocity))):
        raise ValueError("reward components require finite transition values")
    dxy_previous = float(np.linalg.norm(previous[:2]))
    dxy_next = float(np.linalg.norm(following[:2]))
    dz_next = float(following[2])
    return np.asarray((
        np.clip(dxy_previous - dxy_next, -1.0, 1.0),
        np.clip(abs(float(previous[2])) - abs(dz_next), -1.0, 1.0)
        / max(dxy_next, 1.0),
        -max(float(next_uav_vertical_velocity) + 0.5, 0.0),
        -dz_next if dz_next > 0.0 else 0.0,
        -abs(float(command[3])),
    ), dtype=np.float64)


def constrained_adaptive_weights(logits, *,
                                 baseline_weights=BASELINE_REWARD_WEIGHTS,
                                 total_weight: float = TOTAL_REWARD_WEIGHT,
                                 kappa: float = 0.69314718056,
                                 epsilon: float = 0.2):
    """보존적 multiplicative tilt를 적용한다.

    NumPy 배열과 torch Tensor를 모두 지원하며 Tensor 입력의 gradient를
    유지한다. 모든 출력은 양수이고 합은 ``total_weight``다.
    """
    total = float(total_weight)
    if total <= 0.0 or float(kappa) < 0.0 or not 0.0 <= float(epsilon) <= 1.0:
        raise ValueError("invalid adaptive reward constraint parameters")
    try:
        import torch
        if isinstance(logits, torch.Tensor):
            base = torch.as_tensor(baseline_weights, dtype=logits.dtype,
                                   device=logits.device)
            if logits.shape[-1] != base.numel():
                raise ValueError("adaptive reward logits/components disagree")
            if torch.any(base <= 0.0) or abs(float(base.sum()) - total) > 1e-6:
                raise ValueError("baseline weights must be positive and sum to total_weight")
            p0 = base / total
            tilted = p0 * torch.exp(float(kappa) * torch.tanh(logits))
            tilted = tilted / tilted.sum(dim=-1, keepdim=True)
            probability = float(epsilon) * p0 + (1.0 - float(epsilon)) * tilted
            return total * probability
    except ImportError:  # pragma: no cover - torch is a project dependency
        pass
    values = np.asarray(logits, dtype=np.float64)
    base = np.asarray(baseline_weights, dtype=np.float64).reshape(-1)
    if values.shape[-1] != base.size:
        raise ValueError("adaptive reward logits/components disagree")
    if np.any(base <= 0.0) or abs(float(base.sum()) - total) > 1e-8:
        raise ValueError("baseline weights must be positive and sum to total_weight")
    p0 = base / total
    tilted = p0 * np.exp(float(kappa) * np.tanh(values))
    tilted /= tilted.sum(axis=-1, keepdims=True)
    return total * (float(epsilon) * p0 + (1.0 - float(epsilon)) * tilted)


@dataclass(frozen=True)
class RewardComponentNormalizer:
    """성분별 ``clip(rho/c,-1,1)`` 정규화 계약."""

    scales: tuple[float, ...] = tuple(float(v) for v in DEFAULT_COMPONENT_SCALES)
    exact_paper_raw: bool = False
    source: str = "configured_physical_bounds"

    def __post_init__(self) -> None:
        values = np.asarray(self.scales, dtype=np.float64)
        if values.shape != (5,) or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("component normalization scales must be five positive values")

    def transform(self, components) -> np.ndarray:
        values = np.asarray(components, dtype=np.float64)
        if values.shape[-1] != 5 or not np.isfinite(values).all():
            raise ValueError("reward components must end in five finite values")
        if self.exact_paper_raw:
            return values.copy()
        return np.clip(values / np.asarray(self.scales), -1.0, 1.0)

    def to_manifest(self) -> dict:
        return {"scales": list(self.scales), "exact_paper_raw": self.exact_paper_raw,
                "source": self.source,
                "formula": "clip(rho_k / c_k, -1, 1)"}

    @classmethod
    def fit_training_split(cls, raw_components, *, physical_scales=None,
                           quantile: float = 0.99):
        """훈련 split만으로 scale을 정한다(물리 bound가 있으면 우선)."""
        raw = np.asarray(raw_components, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != 5 or raw.shape[0] == 0:
            raise ValueError("normalizer fit requires non-empty [transition,5] training data")
        if physical_scales is not None:
            return cls(tuple(float(v) for v in physical_scales), source="physical_bounds")
        if not 0.5 <= float(quantile) <= 1.0:
            raise ValueError("normalization quantile must be in [0.5,1]")
        scales = np.quantile(np.abs(raw), float(quantile), axis=0)
        scales = np.maximum(scales, 1e-6)
        return cls(tuple(float(v) for v in scales), source="training_split_quantile")


@dataclass(frozen=True)
class AdaptiveRewardConfig:
    baseline_weights: tuple[float, ...] = (1.0, 1.0, 0.5, 1.0, 2.0)
    total_weight: float = TOTAL_REWARD_WEIGHT
    logit_scale_kappa: float = 0.69314718056
    baseline_mixture_epsilon: float = 0.2
    active_enabled: bool = False
    success_value: float = 10.0
    failure_value: float = -10.0


class AdaptiveWeightReward:
    """동결된 graph->weight 모델과 Table-III 성분을 결합하는 온라인 보상."""

    def __init__(self, weight_provider: Callable, *,
                 config: AdaptiveRewardConfig | None = None,
                 normalizer: RewardComponentNormalizer | None = None):
        self.weight_provider = weight_provider
        self.config = config or AdaptiveRewardConfig()
        self.normalizer = normalizer or RewardComponentNormalizer(exact_paper_raw=True)

    def __call__(self, graph, current_relative_state, next_relative_state, action,
                 *, next_uav_vertical_velocity: float,
                 next_estimation_loss: float | None = None,
                 physical_contact=False, crash=False, excessive_drift=False,
                 battery_depleted=False, terminal=False):
        cfg = self.config
        task = sparse_terminal_reward(
            physical_contact=physical_contact, crash=crash,
            excessive_drift=excessive_drift, battery_depleted=battery_depleted,
            terminal=terminal, success_value=cfg.success_value,
            failure_value=cfg.failure_value)
        raw = shin_reward_components(
            current_relative_state, next_relative_state, action,
            next_uav_vertical_velocity=next_uav_vertical_velocity)
        normalized = self.normalizer.transform(raw)
        weights, latency_ms = self.weight_provider(graph, return_latency=True)
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if weights.shape != (5,) or np.any(weights <= 0.0):
            raise ValueError("adaptive reward provider returned invalid weights")
        if abs(float(weights.sum()) - float(cfg.total_weight)) > 1e-5:
            raise ValueError("adaptive reward weights violate total-weight conservation")
        weight_source = str(getattr(
            self.weight_provider, "weight_source", "adaptive"))
        relation_attention = getattr(
            self.weight_provider, "last_relation_attention", None)
        # Terminal reward replaces every shaping term, as in the controlled
        # Shin reward. We still log the pre-action graph's diagnostic weights,
        # but every applied component contribution is zero.
        if task != 0.0:
            parts = {"task": float(task), "adaptive_shaping": 0.0,
                     "active_perception": 0.0,
                     "rgat_inference_latency_ms": float(latency_ms),
                     "terminal_reward": float(task),
                     "total_shaping_reward": 0.0,
                     "active_perception_reward": 0.0,
                     "final_reward": float(task),
                     "weight_source": weight_source}
            if relation_attention is not None:
                for index, value in enumerate(relation_attention, start=1):
                    parts[f"attention_relation_{index}"] = float(value)
            for index, name in enumerate(REWARD_COMPONENT_NAMES):
                parts[f"rho_{index + 1}_{name}"] = float(raw[index])
                parts[f"rho_normalized_{index + 1}"] = float(normalized[index])
                parts[f"weight_{index + 1}"] = float(weights[index])
                parts[f"weighted_{index + 1}"] = 0.0
                parts[f"raw_rho_{index + 1}"] = float(raw[index])
                parts[f"normalized_rho_{index + 1}"] = float(normalized[index])
                parts[f"weighted_rho_{index + 1}"] = 0.0
            return float(task), parts
        contributions = weights * normalized
        active = 0.0
        if cfg.active_enabled:
            if next_estimation_loss is None:
                raise ValueError("SE adaptive reward requires next estimation loss")
            active = active_perception_reward(next_estimation_loss, ShinRewardConfig())
        shaping = float(contributions.sum())
        parts = {"task": 0.0, "adaptive_shaping": shaping,
                 "active_perception": float(active),
                 "rgat_inference_latency_ms": float(latency_ms),
                 "terminal_reward": 0.0,
                 "total_shaping_reward": shaping,
                 "active_perception_reward": float(active),
                 "final_reward": shaping + float(active),
                 "weight_source": weight_source}
        if relation_attention is not None:
            for index, value in enumerate(relation_attention, start=1):
                parts[f"attention_relation_{index}"] = float(value)
        for index, name in enumerate(REWARD_COMPONENT_NAMES):
            parts[f"rho_{index + 1}_{name}"] = float(raw[index])
            parts[f"rho_normalized_{index + 1}"] = float(normalized[index])
            parts[f"weight_{index + 1}"] = float(weights[index])
            parts[f"weighted_{index + 1}"] = float(contributions[index])
            parts[f"raw_rho_{index + 1}"] = float(raw[index])
            parts[f"normalized_rho_{index + 1}"] = float(normalized[index])
            parts[f"weighted_rho_{index + 1}"] = float(contributions[index])
        return shaping + float(active), parts


class FixedBaselineRewardWeights:
    """동일한 성분/정규화 함수로 fixed arm을 계산하기 위한 provider."""

    def __init__(self, normalizer: RewardComponentNormalizer):
        self.normalizer = normalizer
        self.weight_source = "fixed_baseline"

    def __call__(self, _graph, *, return_latency=False):
        weights = BASELINE_REWARD_WEIGHTS.copy()
        return (weights, 0.0) if return_latency else weights
