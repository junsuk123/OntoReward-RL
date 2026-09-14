"""Offline BCE training for the frozen future-FOV-loss R-GAT."""
from __future__ import annotations

import copy
import csv
from pathlib import Path

import numpy as np
import torch

from .fov_risk_dataset import (dataset_digest, split_by_episode,
                               validate_fov_risk_dataset)
from .fov_risk_model import (FOVRiskModel, FrozenFOVRiskPredictor,
                             save_fov_risk_model)


def binary_classification_metrics(labels, probabilities, *, threshold=0.5) -> dict:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if y.shape != p.shape or y.size == 0:
        raise ValueError("classification metrics require equally sized non-empty arrays")
    predicted = p >= float(threshold)
    positive = y == 1
    tp = int(np.sum(predicted & positive))
    tn = int(np.sum(~predicted & ~positive))
    fp = int(np.sum(predicted & ~positive))
    fn = int(np.sum(~predicted & positive))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    positives = p[positive]
    negatives = p[~positive]
    if positives.size and negatives.size:
        # Mann-Whitney interpretation, including half credit for ties.
        comparisons = positives[:, None] - negatives[None, :]
        auroc = float(np.mean((comparisons > 0.0) + 0.5 * (comparisons == 0.0)))
    else:
        auroc = 0.5
    return {
        "auroc": auroc,
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "threshold": float(threshold),
    }


def train_fov_risk_model(dataset, *, seed=42, validation_fraction=0.2,
                         epochs=80, batch_size=64, learning_rate=5e-4,
                         hidden_dim=24, relation_dim=6, heads=1,
                         device="cpu"):
    """Train with one objective only: binary cross entropy."""
    X, y, meta = validate_fov_risk_dataset(dataset)
    training, validation = split_by_episode(
        dataset, validation_fraction=validation_fraction, seed=seed)
    torch.manual_seed(int(seed))
    model = FOVRiskModel(
        hidden_dim=hidden_dim, relation_dim=relation_dim,
        heads=heads, seed=seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    objective = torch.nn.BCEWithLogitsLoss()
    train_X = torch.as_tensor(X[training], dtype=torch.float32, device=device)
    train_y = torch.as_tensor(y[training], dtype=torch.float32, device=device)
    val_X = torch.as_tensor(X[validation], dtype=torch.float32, device=device)
    val_y = torch.as_tensor(y[validation], dtype=torch.float32, device=device)
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 17)
    history = []
    best_loss = float("inf")
    best_state = None
    count = max(1, int(batch_size))
    for epoch in range(1, max(1, int(epochs)) + 1):
        model.train()
        order = torch.randperm(len(train_X), generator=generator)
        total = 0.0
        seen = 0
        for start in range(0, len(order), count):
            indices = order[start:start + count].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model.forward_logits(train_X[indices]), train_y[indices])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.item()) * len(indices)
            seen += len(indices)
        model.eval()
        with torch.no_grad():
            validation_loss = float(objective(
                model.forward_logits(val_X), val_y).item())
        row = {
            "epoch": epoch,
            "train_bce": total / max(seen, 1),
            "validation_bce": validation_loss,
        }
        history.append(row)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("FOV-risk training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        validation_probability = model(val_X).cpu().numpy()
    metrics = binary_classification_metrics(y[validation], validation_probability)
    train_episodes = sorted(int(value) for value in np.unique(meta[training, 0]))
    validation_episodes = sorted(int(value) for value in np.unique(meta[validation, 0]))
    if set(train_episodes) & set(validation_episodes):
        raise AssertionError("FOV-risk episode leakage detected after training")
    return model, history, {
        **metrics,
        "best_validation_bce": float(best_loss),
        "train_episode_ids": train_episodes,
        "validation_episode_ids": validation_episodes,
    }


def prepare_fov_risk_artifact(path: str | Path, dataset, *,
                              dataset_manifest: dict, config_hash: str,
                              seed=42, settings=None):
    settings = dict(settings or {})
    model, history, metrics = train_fov_risk_model(
        dataset,
        seed=seed,
        validation_fraction=float(settings.get("validation_fraction", 0.2)),
        epochs=int(settings.get("epochs", 80)),
        batch_size=int(settings.get("batch_size", 64)),
        learning_rate=float(settings.get("learning_rate", 5e-4)),
        hidden_dim=int(settings.get("hidden_dim", 24)),
        relation_dim=int(settings.get("relation_dim", 6)),
        heads=int(settings.get("heads", 1)),
        device=str(settings.get("device", "cpu")),
    )
    metadata = {
        "frozen": True,
        "dataset_version": dataset_manifest["dataset_version"],
        "dataset_sha256": dataset_digest(dataset),
        "dataset_manifest": dict(dataset_manifest),
        "dataset_config_hash": str(config_hash),
        "seed": int(seed),
        "graph_version": dataset_manifest["graph_version"],
        "prediction_horizon_seconds": dataset_manifest[
            "prediction_horizon_seconds"],
        "prediction_horizon_steps": dataset_manifest[
            "prediction_horizon_steps"],
        "loss": "binary_cross_entropy_only",
        "checkpoint_selection": "minimum_validation_bce",
        "validation_metrics": metrics,
        "training_config": {
            key: settings.get(key, default) for key, default in (
                ("validation_fraction", 0.2), ("epochs", 80),
                ("batch_size", 64), ("learning_rate", 5e-4),
                ("hidden_dim", 24), ("relation_dim", 6), ("heads", 1))
        },
    }
    save_fov_risk_model(model, path, metadata=metadata)
    history_path = Path(path).parent / "fov_risk_training_history.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "epoch", "train_bce", "validation_bce"))
        writer.writeheader()
        writer.writerows(history)
    frozen = FrozenFOVRiskPredictor(
        path, expected_config_hash=config_hash,
        device=str(settings.get("device", "cpu")))
    metadata["model_checksum"] = frozen.sha256
    return frozen, metadata
