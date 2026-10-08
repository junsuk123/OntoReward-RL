"""Run an arm (or the teacher) on the local minimal contract.

One ``ArmController`` per episode owns the ontology and the observation
stack, so every arm is fed from the same ``LandingObservation`` stream in the
same order the ROS nodes would see it.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch

from .arms import ObservationStack
from .local_env import MinimalLandingEnv
from .ontology import MinimalOntology

UNSAFE = frozenset({"UNSAFE_CONTACT", "MISSED_PAD_CONTACT", "SAFETY_ENVELOPE_VIOLATION",
                    "UNAUTHORIZED_CONTACT"})


class ArmController:
    def __init__(self, arm=None, *, deterministic: bool = True, sigma: float | None = None,
                 rng: np.random.Generator | None = None):
        self.arm, self.deterministic, self.sigma = arm, deterministic, sigma
        self.rng = rng or np.random.default_rng(0)
        self.ontology = MinimalOntology()
        self.stack = ObservationStack()
        self.last_input = None
        self.last_graph = None

    def inputs(self, obs):
        stacked = self.stack.push(obs)
        graph = self.ontology.build(obs)
        self.last_graph = graph
        self.last_input = None if self.arm is None else self.arm.inputs(stacked, graph)
        return stacked, graph

    @torch.no_grad()
    def act(self, obs) -> np.ndarray:
        self.inputs(obs)
        mean, log_std, _ = self.arm(self.last_input)
        action = mean.numpy().astype(float)
        if not self.deterministic:
            std = float(self.sigma) if self.sigma is not None else log_std.exp().numpy()
            action = action + self.rng.normal(size=3) * std
        return action


def make_env(scenario: str | None = None, difficulty: float = 1.0):
    """Nominal env, or an evaluation-only stress scenario (stress.py)."""
    if scenario is None:
        return MinimalLandingEnv(difficulty=difficulty)
    from ..spatial.core import SpatialConfig
    from .stress import SCENARIOS, RandomizedStressBackend, StressBackend
    cfg = SpatialConfig()
    if scenario == "dr":
        return MinimalLandingEnv(cfg, backend=RandomizedStressBackend(cfg))
    return MinimalLandingEnv(cfg, backend=StressBackend(cfg, SCENARIOS[scenario]))


def evaluate(arm, seeds, *, deterministic=True, sigma=None, difficulty=1.0,
             scenario: str | None = None) -> dict:
    env = make_env(scenario, difficulty)
    statuses, times, activity = Counter(), [], []
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        ctl = ArmController(arm, deterministic=deterministic, sigma=sigma,
                            rng=np.random.default_rng(int(seed)))
        done, info = False, {}
        while not done:
            action = ctl.act(obs)
            if hasattr(arm, "relational_activity") and len(activity) < 2000:
                activity.append(arm.relational_activity(ctl.last_input))
            obs, _, done, info = env.step(action)
        statuses[info["status"]] += 1
        if info["status"] == "SUCCESS":
            times.append(info["elapsed_s"])
    n = len(seeds)
    return {
        "episodes": n, "statuses": dict(statuses),
        "landing": statuses["SUCCESS"] / n,
        "unsafe": sum(statuses[s] for s in UNSAFE) / n,
        "abort": statuses["SAFE_ABORT"] / n, "timeout": statuses["TASK_TIMEOUT"] / n,
        "mean_success_time_s": float(np.mean(times)) if times else None,
        "relational_activity": float(np.mean(activity)) if activity else None,
    }
