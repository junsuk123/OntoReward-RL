"""상태 적응형 보상 가중치의 trajectory-level 오프라인 학습."""
from __future__ import annotations

import csv
import copy
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..reward_modes.adaptive_weight import BASELINE_REWARD_WEIGHTS
from .adaptive_dataset import adaptive_dataset_digest, validate_adaptive_dataset
from .adaptive_model import (
    ADAPTIVE_GRAPH_VERSION, ADAPTIVE_MODEL_FORMAT, ADAPTIVE_NODE_NAMES,
    adaptive_model_config, adaptive_model_digest, build_adaptive_reward_model,
    freeze_adaptive_reward_model, save_adaptive_reward_artifact)


DEFAULT_LOSS_CONFIG = {
    "outcome_bce": 1.0,
    "ranking": 0.5,
    "baseline_prior": 0.05,
    "temporal_smoothness": 0.02,
    "ontology_hinge": 0.10,
    "contextual_weight": 0.35,
    "semantic_potential": 0.75,
    "observability_monotonic": 0.25,
    "observability_margin": 0.02,
    "ontology_margin": 0.05,
    "yaw_safety_floor": 0.20,
    "condition_threshold": 0.60,
    "ranking_temperature": 1.0,
}


def _episode_indices(dataset, split_name: str):
    episode = np.asarray(dataset["episode_id"], dtype=np.int64)
    split = np.asarray(dataset["split"]).astype(str)
    return [(int(ep), np.flatnonzero((episode == ep) & (split == split_name)))
            for ep in np.unique(episode[split == split_name])]


def _trajectory_scores(weights, rho, dataset, split_name: str, gamma: float):
    scores, outcomes, ids = [], [], []
    episode = np.asarray(dataset["episode_id"], dtype=np.int64)
    times = np.asarray(dataset["time_index"], dtype=np.int64)
    success = np.asarray(dataset["success"], dtype=np.float32)
    split = np.asarray(dataset["split"]).astype(str)
    for ep in np.unique(episode[split == split_name]):
        index = np.flatnonzero((episode == ep) & (split == split_name))
        index = index[np.argsort(times[index])]
        idx = torch.as_tensor(index, dtype=torch.long, device=weights.device)
        discounts = torch.pow(
            torch.as_tensor(float(gamma), dtype=weights.dtype, device=weights.device),
            torch.arange(len(index), dtype=weights.dtype, device=weights.device))
        contribution = (weights.index_select(0, idx) * rho.index_select(0, idx)).sum(-1)
        # Compare a discounted mean, not an unnormalised sum.  Otherwise a
        # 300-step timeout can outrank a short safe touchdown merely because it
        # has more shaping terms.
        scores.append((discounts * contribution).sum() / discounts.sum().clamp_min(1e-6))
        outcomes.append(float(success[index[0]]))
        ids.append(int(ep))
    if not scores:
        return torch.empty(0, device=weights.device), torch.empty(0, device=weights.device), []
    return (torch.stack(scores),
            torch.as_tensor(outcomes, dtype=weights.dtype, device=weights.device), ids)


def select_ranking_pairs(dataset, split_name="train") -> tuple[list[tuple[int, int]], dict]:
    """명확한 good/bad episode 쌍만 선택한다.

    모든 성공은 모든 실패보다 우선한다. 성공-성공 쌍은 touchdown error,
    |vertical speed|, |roll|, |pitch|, landing time 모두에서 Pareto 지배할
    때만 사용한다.
    """
    entries = []
    for ep, index in _episode_indices(dataset, split_name):
        first = int(index[0])
        entries.append({
            "episode": ep, "success": int(dataset["success"][first]),
            "metrics": np.asarray((
                float(dataset["touchdown_error"][first]),
                abs(float(dataset["touchdown_vertical_speed"][first])),
                abs(float(dataset["touchdown_roll"][first])),
                abs(float(dataset["touchdown_pitch"][first])),
                float(dataset["duration"][first]),
            )),
        })
    pairs, success_failure, pareto, unclear = [], 0, 0, 0
    for i, left in enumerate(entries):
        for right in entries[i + 1:]:
            if left["success"] != right["success"]:
                good, bad = (left, right) if left["success"] else (right, left)
                pairs.append((good["episode"], bad["episode"]))
                success_failure += 1
            elif left["success"]:
                l_dom = np.all(left["metrics"] <= right["metrics"]) and np.any(
                    left["metrics"] < right["metrics"])
                r_dom = np.all(right["metrics"] <= left["metrics"]) and np.any(
                    right["metrics"] < left["metrics"])
                if l_dom or r_dom:
                    pairs.append((left["episode"], right["episode"])
                                 if l_dom else (right["episode"], left["episode"]))
                    pareto += 1
                else:
                    unclear += 1
            else:
                unclear += 1
    return pairs, {"success_over_failure": success_failure,
                   "successful_pareto": pareto, "excluded_unclear": unclear,
                   "total_selected": len(pairs)}


def temporal_smoothness_pairs(dataset, split_name="train") -> list[tuple[int, int]]:
    """같은 episode 안에서 시간상 인접한 transition index만 반환한다."""
    episode = np.asarray(dataset["episode_id"], dtype=np.int64)
    times = np.asarray(dataset["time_index"], dtype=np.int64)
    split = np.asarray(dataset["split"]).astype(str)
    adjacent = []
    for ep in np.unique(episode[split == str(split_name)]):
        index = np.flatnonzero((episode == ep) & (split == str(split_name)))
        index = index[np.argsort(times[index])]
        adjacent.extend((int(left), int(right))
                        for left, right in zip(index[:-1], index[1:]))
    return adjacent


def _semantic_quality(X: torch.Tensor) -> torch.Tensor:
    """Estimator-free observability/safety score used only as an inductive prior."""
    names = {name: index for index, name in enumerate(ADAPTIVE_NODE_NAMES)}
    value = X[:, :, 0]
    selected = [
        "KeypointConfidence", "VisibleKeypointFraction", "ImageAlignment",
        "ImagePlaneMotion", "ScaleRate", "VisibilityMemory",
        "ReacquisitionTrend", "VerticalMotionSafety", "AttitudeStability",
    ]
    quality = value[:, [names[name] for name in selected]].mean(dim=1)
    visual_risk = value[:, names["VisualLossRisk"]]
    battery_risk = value[:, names["BatteryRisk"]]
    return torch.clamp(quality - .20 * visual_risk - .10 * battery_risk, 0.0, 1.0)


def _potential_targets(dataset, gamma: float, device) -> torch.Tensor:
    """Discounted terminal utility for every state, without graph-label leakage."""
    episode = np.asarray(dataset["episode_id"], dtype=np.int64)
    times = np.asarray(dataset["time_index"], dtype=np.int64)
    success = np.asarray(dataset["success"], dtype=np.int64)
    failure = np.asarray(dataset["failure_type"]).astype(str)
    lateral = np.asarray(dataset["touchdown_error"], dtype=np.float64)
    target = np.empty(len(episode), dtype=np.float32)
    for ep in np.unique(episode):
        index = np.flatnonzero(episode == ep)
        order = index[np.argsort(times[index])]
        first = int(order[0])
        if success[first]:
            utility = 1.0
        elif failure[first] in {"collision", "unsafe_pad_contact", "battery_depleted"}:
            utility = -1.0
        elif failure[first] == "excessive_drift":
            utility = -0.85
        else:
            # Preserve a useful distinction between a near miss and a blind,
            # distant timeout instead of collapsing every failure to one label.
            utility = -float(np.clip(.25 + lateral[first] / 2.0, .25, .80))
        remaining = np.arange(len(order) - 1, -1, -1, dtype=np.float64)
        target[order] = utility * np.power(float(gamma), remaining)
    return torch.as_tensor(target, dtype=torch.float32, device=device)


def _contextual_weight_target(X: torch.Tensor, baseline: torch.Tensor,
                              total_weight: float) -> torch.Tensor:
    """Safe graph-only prior that breaks the constant-weight local optimum."""
    names = {name: index for index, name in enumerate(ADAPTIVE_NODE_NAMES)}
    value = X[:, :, 0]
    alignment = value[:, names["ImageAlignment"]]
    scale = value[:, names["ApparentScale"]]
    vertical = value[:, names["VerticalMotionSafety"]]
    attitude = value[:, names["AttitudeStability"]]
    motion = value[:, names["ImagePlaneMotion"]]
    visibility = .5 * (value[:, names["VisibleKeypointFraction"]]
                       + value[:, names["VisibilityMemory"]])
    risk = 1.0 - visibility
    factors = torch.stack((
        1.0 + 1.4 * (1.0 - alignment) + .5 * risk,
        .45 + 1.3 * alignment * visibility * (1.0 - .5 * scale),
        .70 + 1.6 * scale + 1.2 * (1.0 - vertical) + .6 * risk,
        .70 + 1.4 * scale * (1.0 - vertical),
        .80 + 1.1 * (1.0 - motion) + .8 * (1.0 - attitude),
    ), dim=-1)
    desired = baseline[None] * factors
    return float(total_weight) * desired / desired.sum(dim=-1, keepdim=True)


def _ontology_hinge(weights, X, config):
    names = {name: index for index, name in enumerate(ADAPTIVE_NODE_NAMES)}
    value = X[:, :, 0]
    threshold = float(config["condition_threshold"])
    margin = float(config["ontology_margin"])
    near_pad = value[:, names["ApparentScale"]] >= threshold
    high_descent = value[:, names["VerticalMotionSafety"]] <= 1.0 - threshold
    large_offset = value[:, names["ImageAlignment"]] <= 1.0 - threshold
    safe_altitude = value[:, names["ApparentScale"]] <= 1.0 - threshold
    # 현재 proprioception에 yaw-rate 채널이 없으므로 이미지 운동/자세 안정의
    # 보수적 proxy를 사용한다. 실제 yaw 명령(rho5)은 graph input이 아니다.
    high_yaw_proxy = ((value[:, names["ImagePlaneMotion"]] <= 1.0 - threshold)
                      & (value[:, names["AttitudeStability"]] <= threshold))

    losses = []
    mask = near_pad & high_descent
    if torch.any(mask):
        losses.append(F.relu(
            weights[mask, 0] + margin - weights[mask, 2]).square().mean())
    mask = large_offset & safe_altitude
    if torch.any(mask):
        losses.append(F.relu(
            weights[mask, 1] + margin - weights[mask, 0]).square().mean())
    if torch.any(high_yaw_proxy):
        floor = float(config["yaw_safety_floor"])
        losses.append(F.relu(
            floor - weights[high_yaw_proxy, 4]).square().mean())
    return (torch.stack(losses).mean() if losses else
            weights.sum() * 0.0), {
        "near_pad_high_descent": int((near_pad & high_descent).sum()),
        "large_offset_safe_altitude": int((large_offset & safe_altitude).sum()),
        "high_yaw_proxy": int(high_yaw_proxy.sum()),
    }


class OutcomeCalibrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_scale = nn.Parameter(torch.tensor(-2.25))
        self.bias = nn.Parameter(torch.zeros(()))

    @property
    def scale(self):
        return F.softplus(self.raw_scale) + 1e-6

    def forward(self, scores):
        return self.scale * scores + self.bias


def train_adaptive_reward_weights(dataset: Mapping[str, Any], *, settings=None,
                                  seed=42, device="cpu", verbose=True):
    """짧은 dataset에도 동작하는 CPU/GPU offline trainer."""
    validate_adaptive_dataset(dataset, require_both_classes=True)
    config = dict(settings or {})
    model_config = adaptive_model_config(config)
    loss_config = {**DEFAULT_LOSS_CONFIG, **dict(config.get("loss") or {})}
    rank_temperature = float(loss_config["ranking_temperature"])
    gamma = float(config.get("trajectory_gamma", 0.99))
    epochs = int(config.get("epochs", 20))
    learning_rate = float(config.get("learning_rate", 5e-4))
    if (not 0.0 < gamma <= 1.0 or epochs < 1 or learning_rate <= 0.0
            or rank_temperature <= 0.0):
        raise ValueError("invalid adaptive reward training hyperparameters")
    torch.manual_seed(int(seed))
    model = build_adaptive_reward_model(model_config, seed=seed, device=device)
    calibrator = OutcomeCalibrator().to(device)
    X = torch.as_tensor(np.asarray(dataset["X"], dtype=np.float32), device=device)
    rho = torch.as_tensor(np.asarray(dataset["rho_normalized"], dtype=np.float32),
                          device=device)
    train_mask = torch.as_tensor(
        np.asarray(dataset["split"]).astype(str) == "train", device=device)
    validation_mask = torch.as_tensor(
        np.asarray(dataset["split"]).astype(str) == "validation", device=device)
    for name, mask in (("training", train_mask), ("validation", validation_mask)):
        labels = set(np.asarray(dataset["success"], dtype=int)[
            mask.detach().cpu().numpy()].tolist())
        if labels != {0, 1}:
            raise ValueError(
                f"adaptive {name} split requires both success and failure episodes")
    pairs, pair_stats = select_ranking_pairs(dataset, "train")
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(calibrator.parameters()), lr=learning_rate)
    p0 = torch.as_tensor(BASELINE_REWARD_WEIGHTS / BASELINE_REWARD_WEIGHTS.sum(),
                         dtype=torch.float32, device=device)
    baseline = torch.as_tensor(BASELINE_REWARD_WEIGHTS, dtype=torch.float32,
                               device=device)
    potential_target = _potential_targets(dataset, gamma, device)
    contextual_target = _contextual_weight_target(
        X, baseline, float(model.total_weight)).detach()
    semantic_quality = _semantic_quality(X).detach()
    adjacent = temporal_smoothness_pairs(dataset, "train")
    adjacent_index = (None if not adjacent else torch.as_tensor(
        adjacent, dtype=torch.long, device=device))
    history = []
    rule_stats = {}
    best_state = None
    best_calibrator = None
    best_validation = float("inf")
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        weights, _, potentials = model(
            X, return_logits=True, return_potential=True)
        scores, outcomes, episode_ids = _trajectory_scores(
            weights, rho, dataset, "train", gamma)
        logits = calibrator(scores)
        outcome_loss = F.binary_cross_entropy_with_logits(logits, outcomes)
        score_by_episode = {ep: scores[index] for index, ep in enumerate(episode_ids)}
        ranking_terms = [-F.logsigmoid(
            (score_by_episode[good] - score_by_episode[bad]) / rank_temperature)
                         for good, bad in pairs
                         if good in score_by_episode and bad in score_by_episode]
        ranking_loss = (torch.stack(ranking_terms).mean() if ranking_terms else
                        weights.sum() * 0.0)
        probability = weights[train_mask] / float(model.total_weight)
        prior_loss = ((probability - p0) ** 2).mean()
        smooth_loss = (weights.sum() * 0.0 if adjacent_index is None else
                       ((weights[adjacent_index[:, 1]] / float(model.total_weight)
                         - weights[adjacent_index[:, 0]]
                         / float(model.total_weight)) ** 2).mean())
        ontology_loss, rule_stats = _ontology_hinge(
            weights[train_mask], X[train_mask], loss_config)
        contextual_loss = ((weights[train_mask] / float(model.total_weight)
                            - contextual_target[train_mask]
                            / float(model.total_weight)) ** 2).mean()
        potential_loss = F.smooth_l1_loss(
            potentials[train_mask], potential_target[train_mask])
        monotonic_terms = []
        if adjacent_index is not None:
            left, right = adjacent_index[:, 0], adjacent_index[:, 1]
            quality_delta = semantic_quality[right] - semantic_quality[left]
            changed = quality_delta.abs() >= .02
            if torch.any(changed):
                direction = torch.sign(quality_delta[changed])
                potential_delta = potentials[right[changed]] - potentials[left[changed]]
                monotonic_terms.append(F.relu(
                    float(loss_config["observability_margin"])
                    - direction * potential_delta).square().mean())
        observability_loss = (torch.stack(monotonic_terms).mean()
                              if monotonic_terms else weights.sum() * 0.0)
        total = (float(loss_config["outcome_bce"]) * outcome_loss
                 + float(loss_config["ranking"]) * ranking_loss
                 + float(loss_config["baseline_prior"]) * prior_loss
                 + float(loss_config["temporal_smoothness"]) * smooth_loss
                 + float(loss_config["ontology_hinge"]) * ontology_loss
                 + float(loss_config["contextual_weight"]) * contextual_loss
                 + float(loss_config["semantic_potential"]) * potential_loss
                 + float(loss_config["observability_monotonic"])
                 * observability_loss)
        if not torch.isfinite(total):
            raise FloatingPointError("adaptive reward offline loss is non-finite")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(calibrator.parameters()), 5.0,
            error_if_nonfinite=True)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            candidate_weights, candidate_potential = model(
                X, return_potential=True)
            validation_scores, validation_outcomes, _ = _trajectory_scores(
                candidate_weights, rho, dataset, "validation", gamma)
            validation_logits = calibrator(validation_scores)
            validation_bce = F.binary_cross_entropy_with_logits(
                validation_logits, validation_outcomes)
            validation_potential = F.smooth_l1_loss(
                candidate_potential[validation_mask],
                potential_target[validation_mask])
            validation_objective = float(
                validation_bce + float(loss_config["semantic_potential"])
                * validation_potential)
        if validation_objective < best_validation:
            best_validation = validation_objective
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_calibrator = copy.deepcopy(calibrator.state_dict())
        row = {"epoch": epoch, "total_loss": float(total.detach()),
               "outcome_bce": float(outcome_loss.detach()),
               "ranking_loss": float(ranking_loss.detach()),
               "baseline_prior": float(prior_loss.detach()),
               "temporal_smoothness": float(smooth_loss.detach()),
               "ontology_hinge": float(ontology_loss.detach()),
               "contextual_weight": float(contextual_loss.detach()),
               "semantic_potential": float(potential_loss.detach()),
               "observability_monotonic": float(observability_loss.detach()),
               "validation_bce": float(validation_bce),
               "validation_potential": float(validation_potential),
               "validation_objective": validation_objective,
               "calibration_scale": float(calibrator.scale.detach()),
               "calibration_bias": float(calibrator.bias.detach())}
        history.append(row)
        if verbose:
            print(f"adaptive R-GAT epoch {epoch:3d}/{epochs} "
                  f"loss={row['total_loss']:.4f} outcome={row['outcome_bce']:.4f} "
                  f"rank={row['ranking_loss']:.4f} val={validation_objective:.4f}")

    if best_state is None or best_calibrator is None:
        raise RuntimeError("adaptive R-GAT did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    calibrator.load_state_dict(best_calibrator)
    model.eval()
    with torch.no_grad():
        weights, potentials = model(X, return_potential=True)
        validation_scores, validation_outcomes, _ = _trajectory_scores(
            weights, rho, dataset, "validation", gamma)
        if validation_scores.numel():
            validation_logits = calibrator(validation_scores)
            validation_bce = float(F.binary_cross_entropy_with_logits(
                validation_logits, validation_outcomes))
            validation_accuracy = float(((validation_logits >= 0.0)
                                         == (validation_outcomes >= 0.5)).float().mean())
            validation_good = validation_scores[validation_outcomes >= .5]
            validation_bad = validation_scores[validation_outcomes < .5]
            if validation_good.numel() and validation_bad.numel():
                differences = validation_good[:, None] - validation_bad[None, :]
                validation_ranking_accuracy = float(
                    ((differences > 0.0).float()
                     + .5 * (differences == 0.0).float()).mean())
            else:
                validation_ranking_accuracy = None
        else:
            validation_bce = None
            validation_accuracy = None
            validation_ranking_accuracy = None
        weight_mean = weights.detach().cpu().numpy().mean(axis=0).tolist()
        weight_array = weights.detach().cpu().numpy()
        weight_cv = (weight_array.std(axis=0)
                     / np.maximum(np.abs(weight_array.mean(axis=0)), 1e-8))
        potential_array = potentials.detach().cpu().numpy()
        quality_array = semantic_quality.detach().cpu().numpy()
        potential_quality_correlation = float(np.corrcoef(
            potential_array, quality_array)[0, 1])
        if not np.isfinite(potential_quality_correlation):
            potential_quality_correlation = 0.0
        audit_pairs = temporal_smoothness_pairs(dataset, "validation")
        if not audit_pairs:
            audit_pairs = temporal_smoothness_pairs(dataset, "train")
        agreements = []
        for left, right in audit_pairs:
            quality_delta = quality_array[right] - quality_array[left]
            if abs(float(quality_delta)) < .02:
                continue
            potential_delta = potential_array[right] - potential_array[left]
            agreements.append(float(quality_delta * potential_delta >= 0.0))
        potential_monotonic_compliance = (
            float(np.mean(agreements)) if agreements else 1.0)
    model_sha = freeze_adaptive_reward_model(model)
    metrics = {"final": history[-1], "validation_outcome_bce": validation_bce,
               "validation_accuracy": validation_accuracy,
               "validation_ranking_accuracy": validation_ranking_accuracy,
               "best_epoch": int(best_epoch),
               "best_validation_objective": float(best_validation),
               "mean_weights": weight_mean,
               "weight_coefficient_of_variation": weight_cv.tolist(),
               "mean_weight_coefficient_of_variation": float(weight_cv.mean()),
               "potential_quality_correlation": potential_quality_correlation,
               "potential_observability_monotonic_compliance": (
                   potential_monotonic_compliance),
               "ranking_pairs": pair_stats,
               "ontology_rule_activations": rule_stats,
               "parameter_count": int(sum(p.numel() for p in model.parameters()))}
    calibration = {"raw_scale": float(calibrator.raw_scale.detach().cpu()),
                   "scale": float(calibrator.scale.detach().cpu()),
                   "bias": float(calibrator.bias.detach().cpu())}
    model._frozen_sha256 = model_sha
    return model.cpu(), history, metrics, calibration


def prepare_adaptive_reward_artifact(path: str | Path, dataset: Mapping[str, Any], *,
                                     dataset_manifest: Mapping[str, Any],
                                     config_hash: str, settings=None, seed=42):
    """학습 후 PPO용 immutable artifact와 재현 metadata를 원자적으로 저장."""
    settings = dict(settings or {})
    model, history, metrics, calibration = train_adaptive_reward_weights(
        dataset, settings=settings, seed=seed,
        device=str(settings.get("device", "cpu")), verbose=bool(
            settings.get("verbose", True)))
    model_config = adaptive_model_config(settings)
    gate_config = dict(settings.get("quality_gate") or {})
    gate_enabled = bool(gate_config.get("enabled", False))
    validation_episode_count = len(set(np.asarray(dataset["episode_id"])[
        np.asarray(dataset["split"]).astype(str) == "validation"].astype(int).tolist()))
    strata = dict(dataset_manifest.get("outcome_strata") or {})
    checks = {
        "validation_episode_count": (
            validation_episode_count >= int(gate_config.get(
                "minimum_validation_episodes", 2))),
        "validation_accuracy": (
            metrics["validation_accuracy"] is not None
            and float(metrics["validation_accuracy"])
            >= float(gate_config.get("minimum_validation_accuracy", .50))),
        "validation_ranking_accuracy": (
            metrics["validation_ranking_accuracy"] is not None
            and float(metrics["validation_ranking_accuracy"])
            >= float(gate_config.get(
                "minimum_validation_ranking_accuracy", .50))),
        "weight_state_variation": (
            float(metrics["mean_weight_coefficient_of_variation"])
            >= float(gate_config.get("minimum_mean_weight_cv", .005))),
        "potential_observability_direction": (
            float(metrics["potential_observability_monotonic_compliance"])
            >= float(gate_config.get(
                "minimum_potential_monotonic_compliance", .55))),
        "unsafe_failure_coverage": (
            int(strata.get("unsafe_pad_contact", 0))
            + int(strata.get("collision", 0))
            + int(strata.get("excessive_drift", 0))
            >= int(gate_config.get("minimum_unsafe_failure_episodes", 0))),
    }
    quality_gate = {
        "enabled": gate_enabled,
        "passed": bool(not gate_enabled or all(checks.values())),
        "checks": checks,
        "thresholds": {
            "minimum_validation_episodes": int(gate_config.get(
                "minimum_validation_episodes", 2)),
            "minimum_validation_accuracy": float(gate_config.get(
                "minimum_validation_accuracy", .50)),
            "minimum_validation_ranking_accuracy": float(gate_config.get(
                "minimum_validation_ranking_accuracy", .50)),
            "minimum_mean_weight_cv": float(gate_config.get(
                "minimum_mean_weight_cv", .005)),
            "minimum_potential_monotonic_compliance": float(gate_config.get(
                "minimum_potential_monotonic_compliance", .55)),
            "minimum_unsafe_failure_episodes": int(gate_config.get(
                "minimum_unsafe_failure_episodes", 0)),
        },
    }
    if not quality_gate["passed"]:
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise RuntimeError(
            "adaptive R-GAT quality gate rejected the artifact: " + failed)
    metadata = {
        "format": ADAPTIVE_MODEL_FORMAT, "frozen": True,
        "graph_schema_version": ADAPTIVE_GRAPH_VERSION,
        "dataset_config_hash": str(config_hash),
        "dataset_sha256": adaptive_dataset_digest(dataset),
        "dataset_manifest": dict(dataset_manifest),
        "normalization": dict(dataset["normalization"]),
        "model_config": model_config,
        "training_config": settings,
        "loss": (
            "lambda_out*BCE(sigmoid(a*J+b), outcome) + lambda_rank*ranking + "
            "lambda_prior*||p-p0||^2 + lambda_smooth*||p_t-p_t-1||^2 + "
            "lambda_onto*ontology_hinge + lambda_context*context_prior + "
            "lambda_phi*Huber(Phi,target) + lambda_obs*monotonic_visibility"),
        "trajectory_score": (
            "sum_t gamma^t w(G_t)^T normalized_rho_t / sum_t gamma^t"),
        "semantic_potential": (
            "shared frozen R-GAT encoder; Phi learns discounted terminal utility "
            "with graph-only observability monotonic regularization"),
        "quality_gate": quality_gate,
        "pair_selection": metrics["ranking_pairs"],
        "ontology_rules": [
            "NearPad & HighDescentRate => VerticalSpeedSafety >= LateralProgress + margin",
            "LargeLateralOffset & SafeAltitude => LateralProgress >= VerticalProgress + margin",
            "HighYawRateProxy => YawStability >= safety_floor",
        ],
        "metrics": metrics, "calibration": calibration,
        "seed": int(seed), "model_sha256": adaptive_model_digest(model),
        "train_episode_ids": sorted(set(np.asarray(dataset["episode_id"])[
            np.asarray(dataset["split"]).astype(str) == "train"].astype(int).tolist())),
        "validation_episode_ids": sorted(set(np.asarray(dataset["episode_id"])[
            np.asarray(dataset["split"]).astype(str) == "validation"].astype(int).tolist())),
    }
    canonical = json.dumps(metadata, sort_keys=True, separators=(",", ":"),
                           allow_nan=False)
    metadata["design_id"] = __import__("hashlib").sha256(
        canonical.encode()).hexdigest()[:16]
    save_adaptive_reward_artifact(path, model, metadata=metadata)
    history_path = Path(path).parent / "adaptive_training_history.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return Path(path), metadata
