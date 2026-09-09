"""Controller interface shared by every driving algorithm."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class DriveCommand:
    """Body-frame velocity target for a differential-drive base."""

    linear: float = 0.0  # m/s, +x forward
    angular: float = 0.0  # rad/s, +z counter-clockwise


@dataclass(frozen=True)
class Observation:
    """What a controller gets to see on a given step."""

    t: float  # seconds since the run started
    robot_xy: Tuple[float, float]
    robot_yaw: float  # radians
    people_xy: Tuple[Tuple[float, float], ...] = ()


class DriveController(ABC):
    """Maps observations to drive commands.

    Subclasses set :attr:`name` so they can be registered in
    :mod:`simlab.algorithms.registry` and selected from config.
    """

    name: str = "base"

    @abstractmethod
    def step(self, obs: Observation) -> DriveCommand:
        """Return the command to apply for this simulation step."""

    def reset(self) -> None:
        """Clear any internal state. Called before the run starts."""
