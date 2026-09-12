"""상태 적응형 보상 가중치의 trajectory-level 오프라인 학습."""
from __future__ import annotations

import csv
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
        scores.append((discounts * contribution).sum())
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
    pairs, pair_stats = select_ranking_pairs(dataset, "train")
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(calibrator.parameters()), lr=learning_rate)
    p0 = torch.as_tensor(BASELINE_REWARD_WEIGHTS / BASELINE_REWARD_WEIGHTS.sum(),
                         dtype=torch.float32, device=device)
    adjacent = temporal_smoothness_pairs(dataset, "train")
    adjacent_index = (None if not adjacent else torch.as_tensor(
        adjacent, dtype=torch.long, device=device))
    history = []
    rule_stats = {}
    for epoch in range(1, epochs + 1):
        model.train()
        weights = model(X)
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
        total = (float(loss_config["outcome_bce"]) * outcome_loss
                 + float(loss_config["ranking"]) * ranking_loss
                 + float(loss_config["baseline_prior"]) * prior_loss
                 + float(loss_config["temporal_smoothness"]) * smooth_loss
                 + float(loss_config["ontology_hinge"]) * ontology_loss)
        if not torch.isfinite(total):
            raise FloatingPointError("adaptive reward offline loss is non-finite")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(calibrator.parameters()), 5.0,
            error_if_nonfinite=True)
        optimizer.step()
        row = {"epoch": epoch, "total_loss": float(total.detach()),
               "outcome_bce": float(outcome_loss.detach()),
               "ranking_loss": float(ranking_loss.detach()),
               "baseline_prior": float(prior_loss.detach()),
               "temporal_smoothness": float(smooth_loss.detach()),
               "ontology_hinge": float(ontology_loss.detach()),
               "calibration_scale": float(calibrator.scale.detach()),
               "calibration_bias": float(calibrator.bias.detach())}
        history.append(row)
        if verbose:
            print(f"adaptive R-GAT epoch {epoch:3d}/{epochs} "
                  f"loss={row['total_loss']:.4f} outcome={row['outcome_bce']:.4f} "
                  f"rank={row['ranking_loss']:.4f}")

    model.eval()
    with torch.no_grad():
        weights = model(X)
        validation_scores, validation_outcomes, _ = _trajectory_scores(
            weights, rho, dataset, "validation", gamma)
        if validation_scores.numel():
            validation_logits = calibrator(validation_scores)
            validation_bce = float(F.binary_cross_entropy_with_logits(
                validation_logits, validation_outcomes))
            validation_accuracy = float(((validation_logits >= 0.0)
                                         == (validation_outcomes >= 0.5)).float().mean())
        else:
            validation_bce = None
            validation_accuracy = None
        weight_mean = weights.detach().cpu().numpy().mean(axis=0).tolist()
    model_sha = freeze_adaptive_reward_model(model)
    metrics = {"final": history[-1], "validation_outcome_bce": validation_bce,
               "validation_accuracy": validation_accuracy,
               "mean_weights": weight_mean, "ranking_pairs": pair_stats,
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
            "lambda_onto*ontology_hinge"),
        "trajectory_score": "sum_t gamma^t w(G_t)^T normalized_rho_t",
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
