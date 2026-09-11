"""Paired platform-motion curriculum described by Shin et al. (2026)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlatformMotionCurriculum:
    levels: int = 80
    episodes_per_update: int = 512
    initial_level: int = 1

    def __post_init__(self):
        if self.levels <= 0 or self.episodes_per_update <= 0:
            raise ValueError("curriculum levels and update interval must be positive")
        self.level = min(max(int(self.initial_level), 1), int(self.levels))

    @property
    def c(self) -> float:
        return 1.0 if self.levels == 1 else (self.level - 1) / (self.levels - 1)

    def update(self, completed_episodes: int) -> float:
        # Fig. 3 states that the level is updated every 512 episodes.  The
        # paper does not publish the performance gate, so linear advancement
        # is an explicit Isaac/PX4 adaptation and remains configurable.
        increments = max(0, int(completed_episodes)) // self.episodes_per_update
        self.level = min(self.levels, self.initial_level + increments)
        return self.c

    def scale(self, full_value: float) -> float:
        return self.c * float(full_value)

    def state_dict(self) -> dict:
        return {"levels": self.levels, "episodes_per_update": self.episodes_per_update,
                "initial_level": self.initial_level, "level": self.level, "c": self.c}

    def load_state_dict(self, state: dict) -> None:
        for name in ("levels", "episodes_per_update", "initial_level"):
            if int(state[name]) != int(getattr(self, name)):
                raise ValueError(f"curriculum checkpoint mismatch for {name}")
        self.level = min(max(int(state["level"]), 1), self.levels)
