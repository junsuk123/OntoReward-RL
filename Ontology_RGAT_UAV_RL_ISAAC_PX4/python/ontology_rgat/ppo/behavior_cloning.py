"""Shared visual-teacher warm start for deadline-limited live experiments."""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


DEMONSTRATION_FORMAT = "ontology-rgat-encoded-visual-demonstrations-v1"
DATA_FIELDS = ("embedding", "proprioception", "action", "truth", "episode_id")


def encoded_demonstration_episode(model, rows, *, episode_id: int,
                                  batch_size: int = 64) -> dict[str, torch.Tensor]:
    """Encode one real flight without retaining hundreds of camera frames."""
    if not rows:
        raise ValueError("a behavior-cloning episode cannot be empty")
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(rows), max(1, int(batch_size))):
            chunk = rows[start:start + max(1, int(batch_size))]
            images = np.stack([row["image"] for row in chunk])[:, None]
            output = model.encoder(torch.as_tensor(
                images, dtype=torch.float32, device=model.device) / 255.0)
            embeddings.append(output.embedding.detach().cpu())
    count = len(rows)
    return {
        "embedding": torch.cat(embeddings).float(),
        "proprioception": torch.as_tensor(np.stack([
            row["proprioception"] for row in rows]), dtype=torch.float32),
        "action": torch.as_tensor(np.stack([
            row["action"] for row in rows]), dtype=torch.float32),
        "truth": torch.as_tensor(np.stack([
            row["truth"] for row in rows]), dtype=torch.float32),
        "episode_id": torch.full((count,), int(episode_id), dtype=torch.int64),
    }


def merge_encoded_demonstrations(current, episode):
    if current is None:
        return {name: episode[name].detach().cpu() for name in DATA_FIELDS}
    for name in DATA_FIELDS:
        if name not in current or name not in episode:
            raise ValueError(f"encoded demonstration is missing {name}")
    return {name: torch.cat((current[name], episode[name].detach().cpu()))
            for name in DATA_FIELDS}


def validate_encoded_demonstrations(dataset) -> int:
    if not isinstance(dataset, dict):
        raise ValueError("encoded demonstrations must be a mapping")
    for name in DATA_FIELDS:
        if name not in dataset or not isinstance(dataset[name], torch.Tensor):
            raise ValueError(f"encoded demonstrations require tensor {name}")
    count = int(dataset["episode_id"].shape[0])
    expected = {
        "embedding": (count, None), "proprioception": (count, 7),
        "action": (count, 4), "truth": (count, 6), "episode_id": (count,),
    }
    for name, shape in expected.items():
        actual = tuple(dataset[name].shape)
        if len(actual) != len(shape) or any(
                wanted is not None and got != wanted
                for got, wanted in zip(actual, shape)):
            raise ValueError(
                f"encoded demonstration {name} has shape {actual}, expected {shape}")
        if name != "episode_id" and not torch.isfinite(dataset[name]).all():
            raise ValueError(f"encoded demonstration {name} contains non-finite values")
    if count < 1 or torch.unique(dataset["episode_id"]).numel() < 1:
        raise ValueError("encoded demonstrations contain no episodes")
    return count


def save_encoded_demonstrations(path, dataset, *, config_hash: str,
                                encoder_sha256: str, attempted_seeds,
                                environment_steps: int) -> dict:
    count = validate_encoded_demonstrations(dataset)
    episode_count = int(torch.unique(dataset["episode_id"]).numel())
    payload = {
        "format": DEMONSTRATION_FORMAT,
        "config_hash": str(config_hash),
        "encoder_sha256": str(encoder_sha256),
        "teacher": "onboard-keypoint semantic visual servo; no critic truth action input",
        "successful_episodes": episode_count,
        "attempted_seeds": [int(value) for value in attempted_seeds],
        "environment_steps": int(environment_steps),
        "transitions": count,
        "dataset": {name: dataset[name].detach().cpu() for name in DATA_FIELDS},
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)
    return payload


def load_encoded_demonstrations(path, *, config_hash: str,
                                encoder_sha256: str) -> dict:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format") != DEMONSTRATION_FORMAT:
        raise ValueError("unsupported visual demonstration format")
    if payload.get("config_hash") != str(config_hash):
        raise ValueError("visual demonstration config mismatch")
    if payload.get("encoder_sha256") != str(encoder_sha256):
        raise ValueError("visual demonstration encoder mismatch")
    count = validate_encoded_demonstrations(payload.get("dataset"))
    if count != int(payload.get("transitions", -1)):
        raise ValueError("visual demonstration transition count mismatch")
    return payload


def _cloning_pass(model, dataset, *, optimizer, sequence_length: int,
                  auxiliary_coefficient: float, train: bool) -> tuple[list[float], list[float]]:
    device = model.device
    action_losses, auxiliary_losses = [], []
    episode_ids = torch.unique(dataset["episode_id"], sorted=True).tolist()
    for episode_id in episode_ids:
        indices = torch.nonzero(
            dataset["episode_id"] == int(episode_id), as_tuple=False).flatten()
        hidden = model.initial_state(1)
        for offset in range(0, len(indices), int(sequence_length)):
            selected = indices[offset:offset + int(sequence_length)]
            embedding = dataset["embedding"][selected].to(device)[None]
            proprioception = dataset["proprioception"][selected].to(device)[None]
            temporal = model.temporal_backbone(
                embedding, proprioception, hidden,
                torch.zeros((1, len(selected)), dtype=torch.bool, device=device))
            features = torch.cat((
                temporal.latent[..., model.pipeline_spec.actor_latent_slice],
                proprioception), dim=-1)
            prediction = model.actor(features)
            target_action = torch.clamp(
                dataset["action"][selected].to(device)[None], -.999, .999)
            target = torch.atanh(target_action)
            action_loss = F.smooth_l1_loss(prediction, target)
            loss = action_loss
            auxiliary_loss = None
            if model.relative_state_head is not None:
                estimate = model.relative_state_head(temporal.latent)
                truth = dataset["truth"][selected].to(device)[None]
                auxiliary_loss = model.relative_state_head.loss(estimate, truth)
                loss = loss + float(auxiliary_coefficient) * auxiliary_loss
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [parameter for group in optimizer.param_groups
                     for parameter in group["params"]], 5.0)
                optimizer.step()
            action_losses.append(float(action_loss.detach()))
            if auxiliary_loss is not None:
                auxiliary_losses.append(float(auxiliary_loss.detach()))
            hidden = tuple(value.detach() for value in temporal.hidden)
    return action_losses, auxiliary_losses


def behavior_clone(model, dataset, *, epochs: int = 12,
                   learning_rate: float = 3e-4, sequence_length: int = 48,
                   auxiliary_coefficient: float = .20,
                   post_log_std: float = -1.8) -> dict[str, float]:
    """Warm-start one arm from a shared visual-only teacher dataset."""
    transitions = validate_encoded_demonstrations(dataset)
    epochs = int(epochs)
    sequence_length = int(sequence_length)
    learning_rate = float(learning_rate)
    if epochs < 1 or sequence_length < 1 or not math.isfinite(learning_rate) \
            or learning_rate <= 0.0:
        raise ValueError("behavior cloning requires positive epochs, sequence and learning rate")
    parameters = [*model.temporal_backbone.parameters(), *model.actor.parameters()]
    if model.relative_state_head is not None:
        parameters.extend(model.relative_state_head.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    with torch.no_grad():
        before, before_aux = _cloning_pass(
            model, dataset, optimizer=optimizer, sequence_length=sequence_length,
            auxiliary_coefficient=auxiliary_coefficient, train=False)
    training_losses = []
    for _ in range(epochs):
        losses, _ = _cloning_pass(
            model, dataset, optimizer=optimizer, sequence_length=sequence_length,
            auxiliary_coefficient=auxiliary_coefficient, train=True)
        training_losses.extend(losses)
    with torch.no_grad():
        after, after_aux = _cloning_pass(
            model, dataset, optimizer=optimizer, sequence_length=sequence_length,
            auxiliary_coefficient=auxiliary_coefficient, train=False)
        model.log_std.fill_(float(post_log_std))
    return {
        "transitions": transitions,
        "successful_demonstration_episodes": int(torch.unique(
            dataset["episode_id"]).numel()),
        "epochs": epochs,
        "action_loss_before": float(np.mean(before)),
        "action_loss_train": float(np.mean(training_losses)),
        "action_loss_after": float(np.mean(after)),
        "auxiliary_loss_before": (float(np.mean(before_aux)) if before_aux else 0.0),
        "auxiliary_loss_after": (float(np.mean(after_aux)) if after_aux else 0.0),
        "post_action_std": float(math.exp(float(post_log_std))),
    }
