"""Paired platform-motion curriculum described by Shin et al. (2026)."""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import numpy as np


def fitted_update_interval(episodes: int, levels: int) -> int:
    """Fit all curriculum levels into a shortened per-method budget."""
    count = int(episodes)
    level_count = int(levels)
    if count < 1 or level_count < 1:
        raise ValueError("episodes and curriculum levels must be positive")
    if level_count == 1:
        return count
    return max(1, (count - 1) // (level_count - 1))


@dataclass
class PlatformMotionCurriculum:
    levels: int = 80
    episodes_per_update: int = 512
    initial_level: int = 1
    performance_gated: bool = False
    assessment_window: int = 20
    minimum_episodes_at_level: int = 20
    success_rate_threshold: float = 0.20
    max_position_rmse_m: float = 2.0
    max_geometric_fov_loss_fraction: float = 0.50

    def __post_init__(self):
        if (self.levels <= 0 or self.episodes_per_update <= 0
                or self.assessment_window <= 0
                or self.minimum_episodes_at_level <= 0):
            raise ValueError("curriculum levels and update interval must be positive")
        for value in (self.success_rate_threshold, self.max_geometric_fov_loss_fraction):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError("curriculum probability thresholds must be in [0, 1]")
        if float(self.max_position_rmse_m) <= 0:
            raise ValueError("curriculum RMSE threshold must be positive")
        self.level = min(max(int(self.initial_level), 1), int(self.levels))
        self._episodes_at_level = 0
        self._recent = deque(maxlen=int(self.assessment_window))

    @property
    def c(self) -> float:
        return 1.0 if self.levels == 1 else (self.level - 1) / (self.levels - 1)

    def update(self, completed_episodes: int) -> float:
        # Fig. 3 states that the level is updated every 512 episodes.  The
        # paper does not publish the performance gate, so linear advancement
        # is an explicit Isaac/PX4 adaptation and remains configurable.
        if not self.performance_gated:
            increments = max(0, int(completed_episodes)) // self.episodes_per_update
            self.level = min(self.levels, self.initial_level + increments)
        return self.c

    def observe(self, metric: dict) -> bool:
        """Advance one level only after demonstrated landing competence.

        The live PX4 adaptation cannot safely force all 80 levels into a short
        seminar budget.  It therefore holds the current difficulty until the
        rolling physical success/FOV criteria and (when present) estimator
        position RMSE criterion all pass.
        """
        if not self.performance_gated or self.level >= self.levels:
            return False
        sample = {
            "success": float(metric.get("paper_success", 0.0)),
            "fov": float(metric.get("geometric_fov_loss_fraction", 1.0)),
            "position_rmse": (
                None if metric.get(
                    "relative_position_rmse_m", metric.get("position_rmse"))
                in (None, "") else float(metric.get(
                    "relative_position_rmse_m", metric.get("position_rmse")))),
        }
        self._recent.append(sample)
        self._episodes_at_level += 1
        if (self._episodes_at_level < int(self.minimum_episodes_at_level)
                or len(self._recent) < int(self.assessment_window)):
            return False
        success = float(np.mean([item["success"] for item in self._recent]))
        fov = float(np.mean([item["fov"] for item in self._recent]))
        estimator_values = [item["position_rmse"] for item in self._recent
                            if item["position_rmse"] is not None]
        estimator_ok = (not estimator_values or
                        float(np.mean(estimator_values)) <= self.max_position_rmse_m)
        if (success < self.success_rate_threshold
                or fov > self.max_geometric_fov_loss_fraction or not estimator_ok):
            return False
        self.level += 1
        self._episodes_at_level = 0
        self._recent.clear()
        return True

    def scale(self, full_value: float) -> float:
        return self.c * float(full_value)

    def state_dict(self) -> dict:
        return {"levels": self.levels, "episodes_per_update": self.episodes_per_update,
                "initial_level": self.initial_level, "level": self.level, "c": self.c,
                "performance_gated": self.performance_gated,
                "assessment_window": self.assessment_window,
                "minimum_episodes_at_level": self.minimum_episodes_at_level,
                "success_rate_threshold": self.success_rate_threshold,
                "max_position_rmse_m": self.max_position_rmse_m,
                "max_geometric_fov_loss_fraction": self.max_geometric_fov_loss_fraction,
                "episodes_at_level": self._episodes_at_level,
                "recent": list(self._recent)}

    def load_state_dict(self, state: dict) -> None:
        for name in ("levels", "episodes_per_update", "initial_level"):
            if int(state[name]) != int(getattr(self, name)):
                raise ValueError(f"curriculum checkpoint mismatch for {name}")
        self.level = min(max(int(state["level"]), 1), self.levels)
        if self.performance_gated:
            for name in ("performance_gated", "assessment_window",
                         "minimum_episodes_at_level", "success_rate_threshold",
                         "max_position_rmse_m", "max_geometric_fov_loss_fraction"):
                if name in state and state[name] != getattr(self, name):
                    raise ValueError(f"curriculum checkpoint mismatch for {name}")
            self._episodes_at_level = int(state.get("episodes_at_level", 0))
            self._recent.extend(state.get("recent", ()))
