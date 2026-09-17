"""Offline regression training for the frozen future-FOV-unavailability R-GAT.

The objective has two explicitly implemented terms:

``huber``
    Huber regression of ``y_t``, the fraction of the next ``H`` steps with the
    pad centre outside the frustum, over supervised (unmasked) samples only.

``contract`` (ontology rule ``R-04``)
    A one-sided hinge that penalises a *decrease* of the predicted
    unavailability when the ``MeasurementAge`` node value is raised and nothing
    else changes.  The perturbation is node-local: it edits only that node's
    value and complement rows, so the penalty states exactly the rule it
    enforces and no more.
"""
from __future__ import annotations

import copy
import csv
from pathlib import Path

import numpy as np
import torch

from .fov_graph import FOV_FEATURE_NAMES, FOV_NODE_NAMES
from .fov_risk_dataset import (dataset_digest, split_by_episode,
                               split_by_episode_ids,
                               validate_fov_risk_dataset)
from .fov_risk_model import (FOVRiskModel, FrozenFOVRiskPredictor,
                             save_fov_risk_model)


CONTRACT_RULE_ID = "R-04"
_AGE_NODE = FOV_NODE_NAMES.index("MeasurementAge")


def regression_metrics(targets, predictions, *, reference_mean=None) -> dict:
    """Error metrics plus the constant-predictor baseline they must beat."""
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    p = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if y.shape != p.shape or y.size == 0:
        raise ValueError("regression metrics require equally sized non-empty arrays")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("regression metrics require finite arrays")
    residual = p - y
    constant = float(np.mean(y) if reference_mean is None else reference_mean)
    baseline_residual = constant - y
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "bias": float(np.mean(residual)),
        "r2": (float(1.0 - np.sum(residual ** 2) / denominator)
               if denominator > 0.0 else float("nan")),
        "constant_predictor_rmse": float(np.sqrt(np.mean(baseline_residual ** 2))),
        "constant_predictor_value": constant,
        "target_mean": float(np.mean(y)),
        "prediction_mean": float(np.mean(p)),
        "target_nonzero_fraction": float(np.mean(y > 0.0)),
    }


def _age_perturbed(X: torch.Tensor, delta: float) -> torch.Tensor:
    """Copy of ``X`` with the MeasurementAge node value raised by ``delta``."""
    perturbed = X.clone()
    value = perturbed[:, _AGE_NODE, 0]
    raised = torch.clamp(value + float(delta), 0.0, 1.0)
    perturbed[:, _AGE_NODE, 0] = raised
    perturbed[:, _AGE_NODE, 1] = 1.0 - raised
    return perturbed


def contract_loss(model: FOVRiskModel, X: torch.Tensor, *,
                  delta: float = 0.25) -> torch.Tensor:
    """Hinge on rule R-04: a staler measurement may not look safer."""
    current = torch.sigmoid(model.forward_logits(X))
    staler = torch.sigmoid(model.forward_logits(_age_perturbed(X, delta)))
    return torch.relu(current - staler).mean()


def train_fov_risk_model(dataset, *, seed=42, validation_fraction=0.2,
                         epochs=80, batch_size=64, learning_rate=5e-4,
                         hidden_dim=24, relation_dim=6, heads=1,
                         huber_delta=0.1, contract_weight=0.1,
                         contract_delta=0.25, device="cpu", progress=None,
                         validation_episodes=None):
    """Huber regression of the unavailability fraction plus rule R-04.

    ``progress`` is called with each epoch's history row and the running best
    validation loss. Offline training is minutes of otherwise silent work in
    the middle of a long flight run, so a caller that wants to show it can.
    """
    X, y, valid, meta = validate_fov_risk_dataset(dataset)
    # An accumulating datastore decides the split itself, from each episode's
    # seed, so that an episode held out by one run is never trained on by the
    # next. Only a one-off dataset falls back to the seeded draw.
    training, validation = (
        split_by_episode_ids(dataset, validation_episodes)
        if validation_episodes is not None else
        split_by_episode(dataset, validation_fraction=validation_fraction,
                         seed=seed))
    if not training.any() or not validation.any():
        raise ValueError("FOV-risk split produced an empty supervised side")
    torch.manual_seed(int(seed))
    model = FOVRiskModel(
        hidden_dim=hidden_dim, relation_dim=relation_dim,
        heads=heads, seed=seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    objective = torch.nn.HuberLoss(delta=float(huber_delta))
    weight = float(contract_weight)
    if weight < 0.0:
        raise ValueError("contract_weight must be non-negative")
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
        totals = np.zeros(3)
        seen = 0
        for start in range(0, len(order), count):
            indices = order[start:start + count].to(device)
            batch = train_X[indices]
            optimizer.zero_grad(set_to_none=True)
            fit = objective(torch.sigmoid(model.forward_logits(batch)),
                            train_y[indices])
            rule = (contract_loss(model, batch, delta=contract_delta)
                    if weight > 0.0 else torch.zeros((), device=fit.device))
            loss = fit + weight * rule
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            totals += np.asarray([float(loss.item()), float(fit.item()),
                                  float(rule.item())]) * len(indices)
            seen += len(indices)
        model.eval()
        with torch.no_grad():
            validation_fit = float(objective(
                torch.sigmoid(model.forward_logits(val_X)), val_y).item())
            validation_rule = float(contract_loss(
                model, val_X, delta=contract_delta).item())
        validation_loss = validation_fit + weight * validation_rule
        row = {
            "epoch": epoch,
            "train_loss": totals[0] / max(seen, 1),
            "train_huber": totals[1] / max(seen, 1),
            "train_contract": totals[2] / max(seen, 1),
            "validation_loss": validation_loss,
            "validation_huber": validation_fit,
            "validation_contract": validation_rule,
        }
        history.append(row)
        improved = validation_loss < best_loss
        if improved:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
        if progress is not None:
            progress({**row, "total_epochs": int(max(1, int(epochs))),
                      "best_validation_loss": float(best_loss),
                      "improved": bool(improved)})
    if best_state is None:
        raise RuntimeError("FOV-risk training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        validation_prediction = model(val_X).cpu().numpy()
        validation_rule = float(contract_loss(
            model, val_X, delta=contract_delta).item())
    metrics = regression_metrics(
        y[validation], validation_prediction,
        reference_mean=float(np.mean(y[training])))
    train_episodes = sorted(int(value) for value in np.unique(meta[training, 0]))
    validation_episodes = sorted(int(value) for value in np.unique(meta[validation, 0]))
    if set(train_episodes) & set(validation_episodes):
        raise AssertionError("FOV-risk episode leakage detected after training")
    return model, history, {
        **metrics,
        "best_validation_loss": float(best_loss),
        "validation_contract_violation": validation_rule,
        "contract_rule_id": CONTRACT_RULE_ID,
        "supervised_samples": int(np.count_nonzero(valid)),
        "masked_tail_samples": int(np.count_nonzero(~valid)),
        "train_episode_ids": train_episodes,
        "validation_episode_ids": validation_episodes,
    }


_TRAINING_DEFAULTS = (
    ("validation_fraction", 0.2), ("epochs", 80), ("batch_size", 64),
    ("learning_rate", 5e-4), ("hidden_dim", 24), ("relation_dim", 6),
    ("heads", 1), ("huber_delta", 0.1), ("contract_weight", 0.1),
    ("contract_delta", 0.25))


def prepare_fov_risk_artifact(path: str | Path, dataset, *,
                              dataset_manifest: dict, config_hash: str,
                              seed=42, settings=None, progress=None,
                              validation_episodes=None):
    settings = dict(settings or {})
    model, history, metrics = train_fov_risk_model(
        dataset,
        seed=seed,
        validation_episodes=validation_episodes,
        validation_fraction=float(settings.get("validation_fraction", 0.2)),
        epochs=int(settings.get("epochs", 80)),
        batch_size=int(settings.get("batch_size", 64)),
        learning_rate=float(settings.get("learning_rate", 5e-4)),
        hidden_dim=int(settings.get("hidden_dim", 24)),
        relation_dim=int(settings.get("relation_dim", 6)),
        heads=int(settings.get("heads", 1)),
        huber_delta=float(settings.get("huber_delta", 0.1)),
        contract_weight=float(settings.get("contract_weight", 0.1)),
        contract_delta=float(settings.get("contract_delta", 0.25)),
        device=str(settings.get("device", "cpu")),
        progress=progress,
    )
    metadata = {
        "frozen": True,
        "dataset_version": dataset_manifest["dataset_version"],
        "dataset_sha256": dataset_digest(dataset),
        "dataset_manifest": dict(dataset_manifest),
        "dataset_config_hash": str(config_hash),
        "seed": int(seed),
        "graph_version": dataset_manifest["graph_version"],
        "feature_names": list(FOV_FEATURE_NAMES),
        "prediction_horizon_seconds": dataset_manifest[
            "prediction_horizon_seconds"],
        "prediction_horizon_steps": dataset_manifest[
            "prediction_horizon_steps"],
        "target": dataset_manifest["target"],
        "output_interpretation": (
            "E[y_t | graph of current and past observations]; a time fraction, "
            "not a binary loss probability"),
        "loss": "huber_regression_plus_contract_rule_R-04",
        "checkpoint_selection": "minimum_validation_loss",
        # A fresh FOVRiskModel every time. The accumulation below therefore
        # gives this training more data, never more epochs over the same data.
        "trained_from_scratch": True,
        "split_source": ("frozen per-seed datastore split"
                         if validation_episodes is not None
                         else "seeded per-dataset draw"),
        "validation_metrics": metrics,
        "training_config": {
            key: settings.get(key, default) for key, default in _TRAINING_DEFAULTS
        },
    }
    save_fov_risk_model(model, path, metadata=metadata)
    history_path = Path(path).parent / "fov_risk_training_history.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "epoch", "train_loss", "train_huber", "train_contract",
            "validation_loss", "validation_huber", "validation_contract"))
        writer.writeheader()
        writer.writerows(history)
    frozen = FrozenFOVRiskPredictor(
        path, expected_config_hash=config_hash,
        device=str(settings.get("device", "cpu")))
    metadata["model_checksum"] = frozen.sha256
    return frozen, metadata
