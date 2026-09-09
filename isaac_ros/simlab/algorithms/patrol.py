"""Open-loop patrol controllers."""

from __future__ import annotations

from dataclasses import dataclass

from simlab.algorithms.base import DriveCommand, DriveController, Observation


@dataclass
class SquareLoopController(DriveController):
    """Alternate straight runs and in-place turns, tracing a rounded square.

    Open loop: it ignores the observation entirely and switches on elapsed
    time. Useful as a moving-obstacle baseline.
    """

    linear_speed: float = 0.5  # m/s
    angular_speed: float = 0.8  # rad/s
    straight_s: float = 4.0
    turn_s: float = 2.0

    name = "square_loop"

    def __post_init__(self) -> None:
        if self.straight_s < 0 or self.turn_s < 0:
            raise ValueError("straight_s and turn_s must be non-negative")
        if self.straight_s + self.turn_s <= 0:
            raise ValueError("straight_s + turn_s must be positive")

    @property
    def period(self) -> float:
        return self.straight_s + self.turn_s

    def step(self, obs: Observation) -> DriveCommand:
        if obs.t % self.period < self.straight_s:
            return DriveCommand(linear=self.linear_speed, angular=0.0)
        return DriveCommand(linear=0.0, angular=self.angular_speed)


@dataclass
class StandStillController(DriveController):
    """Parks the robot. Handy when only the people matter."""

    name = "stand_still"

    def step(self, obs: Observation) -> DriveCommand:
        return DriveCommand()
