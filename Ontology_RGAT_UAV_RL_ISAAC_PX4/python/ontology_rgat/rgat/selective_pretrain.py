"""Episode-split artifact contract for offline selective R-GAT pretraining."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..ppo.selective_graph_encoder import SELECTIVE_ARTIFACT_FORMAT
from .selective_state import RELATION_PARTITION_HASH, SELECTIVE_SCHEMA_HASH


TARGET_NAMES = ("visibility_risk", "motion_trend", "capture_sign",
                "distance_sign", "measurement_reliability")


def episode_split(episode_ids, *, seed: int, validation_fraction: float = .2,
                  test_fraction: float = .2) -> dict[str, tuple[int, ...]]:
    unique = np.unique(np.asarray(episode_ids, dtype=np.int64))
    if unique.size < 3:
        raise ValueError("pretraining requires at least three independent episodes")
    if validation_fraction <= 0 or test_fraction <= 0 or (
            validation_fraction + test_fraction >= 1):
        raise ValueError("train/validation/test fractions are invalid")
    shuffled = unique.copy()
    np.random.default_rng(int(seed)).shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * test_fraction)))
    n_validation = max(1, int(round(len(shuffled) * validation_fraction)))
    n_validation = min(n_validation, len(shuffled) - n_test - 1)
    split = {
        "test": tuple(int(v) for v in shuffled[:n_test]),
        "validation": tuple(int(v) for v in shuffled[n_test:n_test + n_validation]),
        "train": tuple(int(v) for v in shuffled[n_test + n_validation:]),
    }
    if set(split["train"]) & set(split["validation"]) or set(split["train"]) & set(split["test"]):
        raise RuntimeError("episode split leakage")
    return split


def training_normalization(observations: np.ndarray, episode_ids,
                           split: dict[str, tuple[int, ...]]) -> tuple[np.ndarray, np.ndarray]:
    observations = np.asarray(observations, dtype=np.float64)
    ids = np.asarray(episode_ids, dtype=np.int64)
    mask = np.isin(ids, np.asarray(split["train"], dtype=np.int64))
    if observations.ndim != 2 or observations.shape[0] != ids.size or not mask.any():
        raise ValueError("normalization requires 2-D observations and train episodes")
    mean = observations[mask].mean(0)
    std = observations[mask].std(0)
    std = np.maximum(std, 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def save_selective_pretraining_artifact(
        path: str | Path, encoder, *, observation_registry_hash: str,
        dataset_id: str, split: dict[str, tuple[int, ...]], seed: int,
        metrics: dict[str, float], control_sufficiency_passed: bool) -> dict:
    """Save base weights only; relation gates deliberately remain fresh/identity."""
    if not control_sufficiency_passed:
        raise ValueError("refusing an artifact that failed control sufficiency")
    normalization_hash = _digest({
        "mean": encoder.normalization_mean.detach().cpu().tolist(),
        "std": encoder.normalization_std.detach().cpu().tolist()})
    metadata = {
        "format": SELECTIVE_ARTIFACT_FORMAT,
        "graph_schema_hash": SELECTIVE_SCHEMA_HASH,
        "relation_partition_hash": RELATION_PARTITION_HASH,
        "observation_registry_hash": str(observation_registry_hash),
        "normalization_hash": normalization_hash,
        "dataset_id": str(dataset_id),
        "split_id": _digest(split),
        "episode_split": {key: list(value) for key, value in split.items()},
        "seed": int(seed),
        "targets": list(TARGET_NAMES),
        "metrics": {key: float(value) for key, value in metrics.items()},
        "control_sufficiency_passed": True,
        "effective_rank": float(metrics.get("effective_rank", 0.0)),
        "version": 1,
    }
    base_names = ("layer1.", "layer2.", "readout.",
                  "normalization_mean", "normalization_std")
    base_state = {name: value.detach().cpu()
                  for name, value in encoder.state_dict().items()
                  if name.startswith(base_names)}
    payload = {"metadata": metadata, "base_state": base_state}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)
    return metadata


def pretrain_selective_base(encoder, graph_features, observations, targets,
                            episode_ids, *, seed: int = 42, epochs: int = 100,
                            learning_rate: float = 5e-4,
                            control_sufficiency_threshold: float = .55):
    """Fit the frozen base offline; validation/test never enter gradients."""
    torch.manual_seed(int(seed))
    X = torch.as_tensor(graph_features, dtype=torch.float32)
    O = torch.as_tensor(observations, dtype=torch.float32)
    Y = torch.as_tensor(targets, dtype=torch.float32)
    ids = np.asarray(episode_ids, dtype=np.int64)
    if X.shape[0] != O.shape[0] or X.shape[0] != Y.shape[0] or X.shape[0] != ids.size:
        raise ValueError("pretraining arrays must have identical sample counts")
    if Y.ndim != 2 or Y.shape[1] != len(TARGET_NAMES):
        raise ValueError(f"pretraining targets must have width {len(TARGET_NAMES)}")
    split = episode_split(ids, seed=seed)
    mean, std = training_normalization(O.numpy(), ids, split)
    encoder.normalization_mean.copy_(torch.as_tensor(mean))
    encoder.normalization_std.copy_(torch.as_tensor(std))
    for module in (encoder.layer1, encoder.layer2, encoder.readout):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    for parameter in encoder.gate.parameters():
        parameter.requires_grad_(False)
    head = nn.Linear(encoder.output_dim, len(TARGET_NAMES))
    parameters = [parameter for module in (
        encoder.layer1, encoder.layer2, encoder.readout, head)
        for parameter in module.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=float(learning_rate))
    masks = {name: torch.as_tensor(np.isin(ids, values))
             for name, values in split.items()}
    history = []
    best = None
    best_loss = float("inf")
    for epoch in range(1, int(epochs) + 1):
        encoder.train(True)
        prediction = head(encoder(X[masks["train"]], O[masks["train"]]))
        loss = torch.nn.functional.mse_loss(prediction, Y[masks["train"]])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            encoder.eval()
            validation = head(encoder(
                X[masks["validation"]], O[masks["validation"]]))
            validation_loss = torch.nn.functional.mse_loss(
                validation, Y[masks["validation"]])
        row = {"epoch": epoch, "train_loss": float(loss),
               "validation_loss": float(validation_loss)}
        history.append(row)
        if row["validation_loss"] < best_loss:
            best_loss = row["validation_loss"]
            best = {name: value.detach().clone()
                    for name, value in encoder.state_dict().items()}
    if best is None:
        raise RuntimeError("pretraining produced no model")
    encoder.load_state_dict(best)
    encoder._freeze_base()
    for parameter in encoder.gate.parameters():
        parameter.requires_grad_(True)
    with torch.no_grad():
        validation_latent = encoder(X[masks["validation"]], O[masks["validation"]])
        validation_prediction = head(validation_latent)
        test_prediction = head(encoder(X[masks["test"]], O[masks["test"]]))
        validation_sign = float((torch.sign(validation_prediction)
                                 == torch.sign(Y[masks["validation"]])).float().mean())
        test_loss = float(torch.nn.functional.mse_loss(
            test_prediction, Y[masks["test"]]))
        singular = torch.linalg.svdvals(validation_latent - validation_latent.mean(0))
        probability = singular / singular.sum().clamp_min(1e-12)
        effective_rank = float(torch.exp(
            -(probability * probability.clamp_min(1e-12).log()).sum()))
    metrics = {"best_validation_loss": best_loss, "test_loss": test_loss,
               "control_sufficiency": validation_sign,
               "effective_rank": effective_rank}
    return {"split": split, "normalization_mean": mean,
            "normalization_std": std, "history": history,
            "metrics": metrics,
            "control_sufficiency_passed": (
                validation_sign >= float(control_sufficiency_threshold))}
