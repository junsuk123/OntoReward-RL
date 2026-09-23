"""Shared behavior-teacher warm start for deadline-limited live experiments."""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..controllers import PLANAR_ACTION_DIM
from ..rgat.state_graph import STATE_GRAPH_INPUT_DIM, STATE_NODE_NAMES


# v3 retains the camera frames (PNG-encoded) next to the embeddings, so a
# demonstration set survives an encoder change by re-embedding instead of by
# re-flying the teacher. v2 files (embeddings only) are still readable.
# v4 adds the ontology situation graph of every transition. The proposed arm's
# actor takes it as an input, so a warm start that lacks it cannot be cloned
# into that arm at all -- and recomputing it after the fact would need the
# semantic recurrence the flight had and a recording does not.
DEMONSTRATION_FORMAT = "ontology-rgat-encoded-behavior-demonstrations-v4"
_LEGACY_DEMONSTRATION_FORMATS = (
    "ontology-rgat-encoded-behavior-demonstrations-v3",
    "ontology-rgat-encoded-behavior-demonstrations-v2")
DATA_FIELDS = ("embedding", "proprioception", "action", "truth", "episode_id")
# Optional, like the frames: present from v4, and required only by an arm whose
# observation carries the graph.
STATE_GRAPH_FIELD = "state_graph"
# One PNG per transition, aligned with the DATA_FIELDS rows. Optional: a set
# assembled before v3, or one whose frames were deliberately dropped, has none.
FRAME_FIELD = "frames_png"


def _encode_frames(images) -> list:
    import cv2

    blobs = []
    for image in np.asarray(images, dtype=np.uint8):
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise ValueError("could not PNG-encode a demonstration frame")
        blobs.append(encoded.tobytes())
    return blobs


def _decode_frames(blobs) -> np.ndarray:
    import cv2

    frames = []
    for blob in blobs:
        image = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("could not decode a stored demonstration frame")
        frames.append(image)
    return np.stack(frames)


def embed_frames(model, images, *, batch_size: int = 64) -> torch.Tensor:
    """Run the frozen encoder over uint8 frames ``(N, H, W)`` in batches."""
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(images), max(1, int(batch_size))):
            chunk = np.asarray(images[start:start + max(1, int(batch_size))])[:, None]
            output = model.encoder(torch.as_tensor(
                chunk, dtype=torch.float32, device=model.device) / 255.0)
            embeddings.append(output.embedding.detach().float().cpu())
    return torch.cat(embeddings)


def encoded_demonstration_episode(model, rows, *, episode_id: int,
                                  batch_size: int = 64,
                                  keep_frames: bool = True) -> dict:
    """Encode one real flight; keep its frames as PNG so it can be re-embedded.

    A 300-step episode of dark 512x320 frames is roughly 10-15 MB as PNG.
    The 2026-09-20 encoder redesign had to re-fly 26 teacher attempts because
    the demonstrations held only the old encoder's embeddings.
    """
    if not rows:
        raise ValueError("a behavior-cloning episode cannot be empty")
    images = np.stack([np.asarray(row["image"], dtype=np.uint8) for row in rows])
    count = len(rows)
    episode = {
        "embedding": embed_frames(model, images, batch_size=batch_size),
        "proprioception": torch.as_tensor(np.stack([
            row["proprioception"] for row in rows]), dtype=torch.float32),
        "action": torch.as_tensor(np.stack([
            row["action"] for row in rows]), dtype=torch.float32),
        "truth": torch.as_tensor(np.stack([
            row["truth"] for row in rows]), dtype=torch.float32),
        "episode_id": torch.full((count,), int(episode_id), dtype=torch.int64),
    }
    graphs = [row.get("state_graph_X") for row in rows]
    if all(graph is not None for graph in graphs):
        # Stored transposed, ``[transitions, nodes, features]``, which is the
        # layout the encoder consumes.
        episode[STATE_GRAPH_FIELD] = torch.as_tensor(
            np.stack([np.asarray(graph, dtype=np.float32).T for graph in graphs]),
            dtype=torch.float32)
    if keep_frames:
        episode[FRAME_FIELD] = _encode_frames(images)
    return episode


def merge_encoded_demonstrations(current, episode):
    if current is None:
        merged = {name: episode[name].detach().cpu() for name in DATA_FIELDS}
        if episode.get(STATE_GRAPH_FIELD) is not None:
            merged[STATE_GRAPH_FIELD] = episode[STATE_GRAPH_FIELD].detach().cpu()
        if episode.get(FRAME_FIELD) is not None:
            merged[FRAME_FIELD] = list(episode[FRAME_FIELD])
        return merged
    for name in DATA_FIELDS:
        if name not in current or name not in episode:
            raise ValueError(f"encoded demonstration is missing {name}")
    merged = {name: torch.cat((current[name], episode[name].detach().cpu()))
              for name in DATA_FIELDS}
    # Kept only while every episode carries one, for the same reason frames
    # are: a partially graphed set cannot warm-start the proposed arm.
    if (current.get(STATE_GRAPH_FIELD) is not None
            and episode.get(STATE_GRAPH_FIELD) is not None):
        merged[STATE_GRAPH_FIELD] = torch.cat(
            (current[STATE_GRAPH_FIELD],
             episode[STATE_GRAPH_FIELD].detach().cpu()))
    # Frames are kept only while every episode carries them: a partially
    # framed set could not be re-embedded as a whole.
    if current.get(FRAME_FIELD) is not None and episode.get(FRAME_FIELD) is not None:
        merged[FRAME_FIELD] = list(current[FRAME_FIELD]) + list(episode[FRAME_FIELD])
    return merged


def validate_encoded_demonstrations(dataset) -> int:
    if not isinstance(dataset, dict):
        raise ValueError("encoded demonstrations must be a mapping")
    for name in DATA_FIELDS:
        if name not in dataset or not isinstance(dataset[name], torch.Tensor):
            raise ValueError(f"encoded demonstrations require tensor {name}")
    count = int(dataset["episode_id"].shape[0])
    expected = {
        "embedding": (count, None), "proprioception": (count, 7),
        "action": (count, PLANAR_ACTION_DIM), "truth": (count, 6),
        "episode_id": (count,),
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
    frames = dataset.get(FRAME_FIELD)
    if frames is not None and len(frames) != count:
        raise ValueError(
            f"encoded demonstrations carry {len(frames)} frames for {count} transitions")
    graphs = dataset.get(STATE_GRAPH_FIELD)
    if graphs is not None:
        shape = tuple(graphs.shape)
        if (len(shape) != 3 or shape[0] != count
                or shape[1] != len(STATE_NODE_NAMES)
                or shape[2] != STATE_GRAPH_INPUT_DIM):
            raise ValueError(
                f"encoded demonstration situation graphs have shape {shape}, "
                f"expected ({count}, {len(STATE_NODE_NAMES)}, "
                f"{STATE_GRAPH_INPUT_DIM})")
        if not torch.isfinite(graphs).all():
            raise ValueError("encoded demonstration situation graphs are not finite")
    return count


def save_encoded_demonstrations(path, dataset, *, config_hash: str,
                                encoder_sha256: str, attempted_seeds,
                                environment_steps: int,
                                teacher: str = "unspecified",
                                demonstration_fingerprint: str | None = None) -> dict:
    count = validate_encoded_demonstrations(dataset)
    episode_count = int(torch.unique(dataset["episode_id"]).numel())
    payload = {
        "format": DEMONSTRATION_FORMAT,
        "config_hash": str(config_hash),
        # What the flights depended on (teacher, its gains and curriculum, the
        # deck, the horizon); a PPO budget edit changes config_hash but not
        # one demonstration. Reuse is keyed on this when it is present.
        "demonstration_fingerprint": (
            None if demonstration_fingerprint is None
            else str(demonstration_fingerprint)),
        "encoder_sha256": str(encoder_sha256),
        "teacher": str(teacher),
        "successful_episodes": episode_count,
        "attempted_seeds": [int(value) for value in attempted_seeds],
        "environment_steps": int(environment_steps),
        "transitions": count,
        "frames_retained": dataset.get(FRAME_FIELD) is not None,
        "situation_graphs_retained": dataset.get(STATE_GRAPH_FIELD) is not None,
        "dataset": {name: dataset[name].detach().cpu() for name in DATA_FIELDS},
    }
    if dataset.get(STATE_GRAPH_FIELD) is not None:
        payload["dataset"][STATE_GRAPH_FIELD] = (
            dataset[STATE_GRAPH_FIELD].detach().cpu())
    if dataset.get(FRAME_FIELD) is not None:
        payload["dataset"][FRAME_FIELD] = list(dataset[FRAME_FIELD])
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)
    return payload


def load_encoded_demonstrations(path, *, config_hash: str,
                                encoder_sha256: str, model=None,
                                batch_size: int = 64,
                                demonstration_fingerprint: str | None = None) -> dict:
    """Load a demonstration set bound to the encoder and to its provenance.

    Provenance is the ``demonstration_fingerprint`` when both sides have one
    (the teacher, its settings and the deck the flights were made on);
    otherwise the whole experiment ``config_hash`` as before.

    With ``model`` given and the stored frames retained, a set encoded by a
    *different* encoder is re-embedded with ``model.encoder`` instead of being
    rejected; the returned payload then carries the new ``encoder_sha256`` and
    ``re_embedded_from_encoder_sha256`` so the caller can save it back.
    Without frames (or without a model) an encoder mismatch is still an error,
    because embeddings of one encoder mean nothing to another.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format") not in (DEMONSTRATION_FORMAT, *_LEGACY_DEMONSTRATION_FORMATS):
        raise ValueError("unsupported behavior demonstration format")
    stored_fingerprint = payload.get("demonstration_fingerprint")
    if demonstration_fingerprint is not None and stored_fingerprint is not None:
        if str(stored_fingerprint) != str(demonstration_fingerprint):
            raise ValueError("behavior demonstration provenance mismatch")
    elif payload.get("config_hash") != str(config_hash):
        raise ValueError("behavior demonstration config mismatch")
    dataset = payload.get("dataset")
    if payload.get("encoder_sha256") != str(encoder_sha256):
        frames = dataset.get(FRAME_FIELD) if isinstance(dataset, dict) else None
        if model is None or not frames:
            raise ValueError("behavior demonstration encoder mismatch")
        dataset["embedding"] = embed_frames(
            model, _decode_frames(frames), batch_size=batch_size)
        payload["re_embedded_from_encoder_sha256"] = str(payload.get("encoder_sha256"))
        payload["encoder_sha256"] = str(encoder_sha256)
        payload["format"] = DEMONSTRATION_FORMAT
    count = validate_encoded_demonstrations(dataset)
    if count != int(payload.get("transitions", -1)):
        raise ValueError("behavior demonstration transition count mismatch")
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
            parts = [temporal.latent[..., model.pipeline_spec.actor_latent_slice],
                     proprioception]
            if model.graph_state_enabled:
                graphs = dataset.get(STATE_GRAPH_FIELD)
                if graphs is None:
                    raise ValueError(
                        "warm-starting the graph state arm needs demonstrations "
                        "recorded with their situation graphs (format v4); "
                        "re-fly the teacher or clone only the baseline")
                parts.append(model.policy_graph_encoder(
                    graphs[selected].to(device)[None]))
            prediction = model.actor(torch.cat(parts, dim=-1))
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
                   post_log_std: float | None = -1.8) -> dict[str, float]:
    """Warm-start one arm from a shared encoded observation/action dataset."""
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
    if model.graph_state_enabled:
        # The actor's encoder is part of the actor. Cloning the mapping without
        # it would ask a randomly initialised g_t to be useful from episode one.
        parameters.extend(model.policy_graph_encoder.parameters())
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
        # Initial BC deliberately chooses a low exploration variance. During
        # PPO anchoring, None preserves the variance learned by the policy and
        # only regularizes the recurrent actor/optional estimator.
        if post_log_std is not None:
            model.log_std.fill_(float(post_log_std))
        action_std = float(model.log_std.exp().mean())
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
        "post_action_std": action_std,
    }
