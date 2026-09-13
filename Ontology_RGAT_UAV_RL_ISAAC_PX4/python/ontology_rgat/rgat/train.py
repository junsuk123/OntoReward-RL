"""Supervised training of the ontology potential, kept on the GPU.

The target is the discounted future safe-landing outcome of the behaviour
rollout each graph came from, so the potential learns "how well is this episode
going to end", not "what is the label of this state".

Why this is worth putting on a GPU at all
-----------------------------------------
The primary graph is 18 nodes with a 24-wide hidden layer, so one graph is far too
small to fill a GPU: the layer is kernel-launch bound, not FLOP bound. The win
comes entirely from batching graphs, and then from not stalling the launch
queue between batches. This trainer therefore

* uploads the whole dataset once -- it is a few megabytes, so it never has to
  be streamed -- and indexes it on the device, so there is no host-to-device
  copy inside the epoch loop;
* keeps the shuffle, the loss accumulation and the metrics on the device, so
  nothing forces a synchronisation until the epoch ends;
* enables TF32 and optionally bfloat16 autocast, which are free on Ampere and
  later;
* optionally ``torch.compile``s the potential, which fuses the many small
  einsums the relational layer is made of.

``rgat/benchmark.py`` measures the crossover rather than assuming it; the
device decision in :func:`select_device` reads the result.
"""
from __future__ import annotations

import time
import copy
from contextlib import nullcontext
from typing import Any, Callable

import numpy as np
import torch

from ..config import Config
from ..semantic import OntologyGraph
from .model import RGATPotential, build_potential

__all__ = ["select_device", "train_potential", "TrainHistory"]


class TrainHistory(dict):
    """Per-epoch losses, plus whatever the trainer wants to report alongside."""


def select_device(cfg: Config, batch_size: int) -> tuple[torch.device, str]:
    """Pick the device for a batch of this size, and say why.

    ``'auto'`` declines the GPU below ``cfg.device.min_batch_for_gpu`` rather
    than quietly rebatching: raising the batch size changes how many Adam steps
    the potential sees, so it is a hyperparameter decision and not a free
    speedup.
    """
    mode = str(cfg.device.rgat).lower()
    if mode == "cpu":
        return torch.device("cpu"), "cfg.device.rgat is cpu"
    if not torch.cuda.is_available():
        return torch.device("cpu"), "no CUDA device is available"
    if mode == "cuda":
        return torch.device("cuda"), "cfg.device.rgat is cuda"
    if mode != "auto":
        raise ValueError("cfg.device.rgat must be 'auto', 'cuda' or 'cpu'")
    minimum = int(cfg.device.min_batch_for_gpu)
    if batch_size >= minimum:
        return torch.device("cuda"), f"batch {batch_size} >= cfg.device.min_batch_for_gpu {minimum}"
    return torch.device("cpu"), (
        f"batch {batch_size} < cfg.device.min_batch_for_gpu {minimum}; at this "
        "size the layer is kernel-launch bound and the CPU wins")


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision not in {"bfloat16", "float16"}:
        return nullcontext()
    return torch.autocast("cuda", dtype=getattr(torch, precision))


def train_potential(dataset: dict[str, Any], cfg: Config, *,
                    graph: OntologyGraph | None = None,
                    initial_model: RGATPotential | None = None,
                    initial_history: dict[str, Any] | None = None,
                    on_epoch: Callable[[int, TrainHistory, np.ndarray, np.ndarray,
                                        RGATPotential], None] | None = None,
                    verbose: bool = True) -> tuple[RGATPotential, TrainHistory]:
    """Fit ``Phi(G)`` to the discounted outcome labels in ``dataset``.

    ``dataset`` carries ``X`` as ``[n, nodes, in_dim]``, ``y`` as ``[n]`` and
    the ``graph`` template the samples share. The returned model is on the CPU
    in float32, because everything downstream -- the PBRS reward inside the
    real-time control loop, the attention read-out, the saved checkpoint -- is
    CPU numeric and a checkpoint tied to one device is not shareable.
    """
    graph = graph or dataset["graph"]
    X_all = np.asarray(dataset["X"], dtype=np.float32)
    y_all = np.asarray(dataset["y"], dtype=np.float32).reshape(-1)
    if X_all.shape[0] != y_all.shape[0]:
        raise ValueError("dataset X and y disagree on the number of samples")
    if X_all.shape[0] == 0:
        raise ValueError("the R-GAT dataset is empty; the rollout stage produced nothing")
    if not np.isfinite(X_all).all() or not np.isfinite(y_all).all():
        raise ValueError("the R-GAT dataset contains non-finite features or labels")
    monotonic_all = dataset.get("monotonic_X")
    if monotonic_all is not None:
        monotonic_all = np.asarray(monotonic_all, dtype=np.float32)
        if (monotonic_all.ndim != 4 or monotonic_all.shape[0] != X_all.shape[0]
                or monotonic_all.shape[2:] != X_all.shape[1:]
                or not np.isfinite(monotonic_all).all()):
            raise ValueError(
                "monotonic counterfactuals must have finite [sample,case,node,feature] shape")

    batch_size = int(cfg.rgat.batch_size)
    device, why = select_device(cfg, batch_size)
    if cfg.device.allow_tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if verbose:
        print(f"R-GAT training on {'the GPU' if device.type == 'cuda' else 'the CPU'}"
              f" ({device.type}): {why}.")

    model = (initial_model.to(device) if initial_model is not None
             else build_potential(cfg, graph, device=device))
    forward = model
    if cfg.device.compile and device.type == "cuda":
        try:
            forward = torch.compile(model)
        except Exception as exc:                       # pragma: no cover - env dependent
            print(f"torch.compile unavailable ({exc}); running eager.")

    # One upload, then everything is device-resident.
    X = torch.as_tensor(X_all, device=device)
    y = torch.as_tensor(y_all, device=device)
    monotonic = (None if monotonic_all is None else
                 torch.as_tensor(monotonic_all, device=device))
    monotonic_weight = float(getattr(cfg.rgat, "monotonic_weight", 0.0))
    monotonic_margin = float(getattr(cfg.rgat, "monotonic_margin", 0.0))
    if monotonic_weight < 0.0 or monotonic_margin < 0.0:
        raise ValueError("monotonic weight and margin must be non-negative")
    n = X.shape[0]
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed + 303)
    split_unit = "sample"
    if bool(dataset.get("split_by_episode", False)):
        meta = np.asarray(dataset.get("meta"), dtype=np.float64)
        if meta.shape != (n, 4):
            raise ValueError("episode-level R-GAT split requires [samples,4] metadata")
        episode_ids = torch.as_tensor(meta[:, 0], dtype=torch.float64)
        episodes = torch.unique(episode_ids, sorted=True)
        if episodes.numel() < 2:
            raise ValueError("episode-level R-GAT split requires at least two rollouts")
        outcomes = {int(ep): int(meta[np.flatnonzero(meta[:, 0] == ep)[0], 2])
                    for ep in episodes.tolist()}
        strata = {}
        for ep in episodes.tolist():
            strata.setdefault(outcomes[int(ep)], []).append(int(ep))
        for label, values in strata.items():
            order = torch.randperm(len(values), generator=generator).tolist()
            strata[label] = [values[index] for index in order]
        n_validation = max(1, int(round(cfg.rgat.val_fraction * episodes.numel())))
        represented = [label for label, values in strata.items() if len(values) >= 2]
        if len(represented) > 1:
            n_validation = max(n_validation, len(represented))
        n_validation = min(n_validation, int(episodes.numel()) - 1)
        validation_episode_ids = [strata[label][0] for label in sorted(represented)]
        remainder = [ep for label in sorted(strata) for ep in strata[label]
                     if ep not in validation_episode_ids]
        validation_episode_ids.extend(
            remainder[:max(0, n_validation - len(validation_episode_ids))])
        train_episode_ids = [int(ep) for ep in episodes.tolist()
                             if int(ep) not in validation_episode_ids]
        train_mask = torch.isin(
            episode_ids, torch.as_tensor(train_episode_ids, dtype=episode_ids.dtype))
        train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze(1).to(device)
        val_idx = torch.nonzero(~train_mask, as_tuple=False).squeeze(1).to(device)
        split_unit = "episode"
    else:
        order = torch.randperm(n, generator=generator)
        n_train = max(1, int(round((1.0 - cfg.rgat.val_fraction) * n)))
        train_idx = order[:n_train].to(device)
        val_idx = order[n_train:].to(device)
        if val_idx.numel() == 0:
            val_idx = train_idx
        train_episode_ids = []
        validation_episode_ids = []

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.rgat.lr),
                                 betas=(0.9, 0.999), eps=1e-8)
    optimizer_state = getattr(model, "_optimizer_state", None)
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)
        for values in optimizer.state.values():
            for key, value in values.items():
                if isinstance(value, torch.Tensor):
                    values[key] = value.to(device)
    precision = str(cfg.device.rgat_precision)
    previous = dict(initial_history or {})
    history = TrainHistory(
        train_loss=list(previous.get("train_loss", [])),
        val_loss=list(previous.get("val_loss", [])),
        monotonic_compliance=list(previous.get("monotonic_compliance", [])),
        epoch_seconds=list(previous.get("epoch_seconds", [])),
        device=device.type, device_reason=why,
        samples=int(n), train_samples=int(train_idx.numel()),
        val_samples=int(val_idx.numel()), split_unit=split_unit)
    history["train_episode_ids"] = train_episode_ids
    history["validation_episode_ids"] = validation_episode_ids
    completed_epochs = len(history["train_loss"])
    best_state = None
    best_optimizer_state = None
    best_validation_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, int(cfg.rgat.epochs) + 1):
        started = time.perf_counter()
        model.train()
        shuffled = train_idx[torch.randperm(train_idx.numel(), generator=generator,
                                            device="cpu").to(device)]
        # Accumulated on the device: reading a loss per minibatch would put a
        # synchronisation between every launch and undo the batching.
        running = torch.zeros((), device=device)
        batches = 0
        for start in range(0, shuffled.numel(), batch_size):
            idx = shuffled[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, precision):
                pred = forward(X.index_select(0, idx))
                target = y.index_select(0, idx)
                loss = ((pred - target) ** 2).mean() + cfg.rgat.output_l2 * (pred ** 2).mean()
                if monotonic is not None and monotonic_weight > 0.0:
                    cases = monotonic.index_select(0, idx)
                    degraded = forward(cases.reshape(-1, *cases.shape[2:])).reshape(
                        cases.shape[0], cases.shape[1])
                    changed = (cases - X.index_select(0, idx)[:, None]).abs().amax(
                        dim=(-1, -2)) > 1e-7
                    violation = torch.relu(
                        monotonic_margin - (pred[:, None] - degraded)) * changed
                    loss = loss + monotonic_weight * (
                        violation.square().sum() / changed.sum().clamp_min(1))
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite R-GAT loss at epoch {completed_epochs + epoch}; "
                    "enable stable_softmax or reduce the learning rate")
            loss.float().backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(cfg.ppo.grad_clip), error_if_nonfinite=True)
            optimizer.step()
            running += loss.detach().float()
            batches += 1

        model.eval()
        with torch.no_grad(), _autocast(device, precision):
            val_pred = forward(X.index_select(0, val_idx)).float()
            val_target = y.index_select(0, val_idx)
            val_loss = float(((val_pred - val_target) ** 2).mean())
            if monotonic is None:
                compliance = 1.0
            else:
                val_cases = monotonic.index_select(0, val_idx)
                val_degraded = forward(
                    val_cases.reshape(-1, *val_cases.shape[2:])).float().reshape(
                        val_cases.shape[0], val_cases.shape[1])
                val_changed = (
                    val_cases - X.index_select(0, val_idx)[:, None]).abs().amax(
                        dim=(-1, -2)) > 1e-7
                compliant = (val_pred[:, None] >= val_degraded) | ~val_changed
                compliance = float(compliant.float().mean())
        train_loss = float(running / max(batches, 1))
        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            raise FloatingPointError(
                f"non-finite R-GAT metric at epoch {completed_epochs + epoch}")
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["monotonic_compliance"].append(compliance)
        history["epoch_seconds"].append(elapsed)
        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            best_epoch = completed_epochs + epoch
            best_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
        model._optimizer_state = optimizer.state_dict()
        cumulative_epoch = completed_epochs + epoch
        if on_epoch is not None:
            # The live model goes with the losses: the attention read-out is
            # only interesting while it is still moving.
            on_epoch(cumulative_epoch, history, val_target.cpu().numpy(),
                     val_pred.cpu().numpy(), model)
        if verbose:
            print(f"R-GAT epoch {cumulative_epoch:3d} "
                  f"(+{epoch}/{cfg.rgat.epochs} this run) | "
                  f"train {train_loss:.4f} | val {val_loss:.4f} | "
                  f"mono {compliance:.1%} | {elapsed:.2f} s")

    if best_state is not None:
        model.load_state_dict(best_state)
        model._optimizer_state = best_optimizer_state
    history["best_validation_loss"] = float(best_validation_loss)
    history["best_validation_epoch"] = int(best_epoch)
    return model.float().cpu(), history
