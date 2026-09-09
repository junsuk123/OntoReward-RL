"""Behaviour rollouts labelled with their own future outcome."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..config import Config
from ..env import run_episode
from ..expert import PolicySpec

__all__ = ["generate_dataset", "save_dataset", "load_dataset"]


def generate_dataset(cfg: Config, *, monitor=None, episode_monitor=None) -> dict[str, Any]:
    """Fly perturbed expert episodes and label every sampled graph.

    The label is the episode's outcome discounted back to the sample, so the
    potential learns "how well is this going to end" rather than a per-state
    class. The perturbation is drawn log-uniformly: the expert's success
    boundary sits near sigma=0.05, so a uniform sweep to 0.65 would label
    almost the whole dataset negative and leave the potential nothing to fit.

    Unlike the retired MATLAB version this does not reset a second environment
    just to obtain a graph template. External environments own a UDP port and
    can arm SITL, so the template is captured from the first real rollout.
    """
    episodes = int(cfg.rgat.data_episodes)
    print(f"Generating {episodes} external behavior episodes...")
    rng = np.random.default_rng(cfg.seed + 404)
    lo, hi = (float(v) for v in cfg.rgat.noise_range)

    features: list[np.ndarray] = []
    labels: list[float] = []
    meta: list[tuple[int, int, float, float]] = []
    template = None

    for episode in range(1, episodes + 1):
        severity = float(np.exp(np.log(lo) + (np.log(hi) - np.log(lo)) * rng.random()))
        policy = PolicySpec("expert_noisy", noise_std=severity, deterministic=False,
                            rng=np.random.default_rng(cfg.seed + 5000 + episode))
        if episode_monitor is not None:
            episode_monitor.reset(f"dataset ep {episode}")
        log = run_episode(policy, "manual", None, 1000 + episode, cfg,
                          monitor=episode_monitor)
        outcome = 2.0 * log.metrics["success"] - 1.0
        total = len(log.graph_x)
        indices = range(0, total, max(1, int(cfg.rgat.sample_stride)))
        count = 0
        for k in indices:
            features.append(log.graph_x[k].T)          # [nodes, in_dim]
            labels.append(outcome * cfg.reward.pbrs.gamma ** (total - k - 1))
            meta.append((episode, k, log.metrics["success"], severity))
            count += 1
        if template is None:
            template = log.graph_template
        if monitor is not None:
            monitor.update(episode, log.metrics["success"], severity, count)
        print(f"  ep {episode:3d}/{episodes} | success={int(log.metrics['success'])} | "
              f"noise={severity:.2f} | samples={count}")

    if monitor is not None:
        monitor.finish()
    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    successes = int(sum(1 for m in meta if m[1] == 0 and m[2] > 0))
    print(f"Dataset: {y.size} samples, positive {100 * float(np.mean(y > 0)):.1f}% | "
          f"successful episodes {successes}/{episodes}")
    return {"X": X, "y": y, "meta": np.asarray(meta, dtype=np.float64),
            "graph": template}


def save_dataset(dataset: dict[str, Any], path: str | Path) -> Path:
    """Write the arrays as ``.npz`` plus the graph template beside them."""
    import pickle

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=dataset["X"], y=dataset["y"], meta=dataset["meta"])
    with path.with_suffix(".graph.pkl").open("wb") as handle:
        pickle.dump(dataset["graph"], handle)
    return path


def load_dataset(path: str | Path) -> dict[str, Any]:
    import pickle

    path = Path(path)
    blob = np.load(path)
    with path.with_suffix(".graph.pkl").open("rb") as handle:
        graph = pickle.load(handle)
    return {"X": blob["X"], "y": blob["y"], "meta": blob["meta"], "graph": graph}
