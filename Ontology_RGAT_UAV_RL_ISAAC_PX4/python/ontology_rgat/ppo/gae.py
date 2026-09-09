"""Generalized advantage estimation for one episode segment."""
from __future__ import annotations

import numpy as np

from ..config import Config

__all__ = ["compute_gae"]


def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray,
                last_value: float, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(advantages, returns)``.

    Advantages come back unnormalised: the trainer standardises them once over
    the whole multi-episode rollout, which is the correct scope. Doing it per
    episode would rescale every short episode against its own noise.
    """
    r = np.asarray(rewards, dtype=float)
    v = np.asarray(values, dtype=float)
    d = np.asarray(dones, dtype=float)
    T = r.size
    adv = np.zeros(T)
    gae = 0.0
    for t in range(T - 1, -1, -1):
        v_next = last_value if t == T - 1 else v[t + 1]
        delta = r[t] + cfg.ppo.gamma * (1.0 - d[t]) * v_next - v[t]
        gae = delta + cfg.ppo.gamma * cfg.ppo.lambda_gae * (1.0 - d[t]) * gae
        adv[t] = gae
    return adv, adv + v
