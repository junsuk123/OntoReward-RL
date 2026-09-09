"""Periodic pose reporting, so a headless run can be checked for real motion."""

from __future__ import annotations

import math
from typing import Sequence

from simlab.algorithms.base import Observation
from simlab.config.schema import TelemetryConfig
from simlab.utils.logging import get_logger

log = get_logger("scene")


class PoseReporter:
    """Prints robot and character positions at a fixed wall-of-sim interval."""

    def __init__(self, cfg: TelemetryConfig, names: Sequence[str]) -> None:
        self.interval = cfg.report_every_s
        self.names = list(names)
        self._next_at = 0.0

    @property
    def enabled(self) -> bool:
        return self.interval > 0

    def maybe_report(self, obs: Observation) -> bool:
        """Report if ``obs.t`` has reached the next interval. Returns whether it printed."""
        if not self.enabled or obs.t < self._next_at:
            return False
        self.report(obs)
        # Skip ahead rather than accumulate, so a slow frame does not cause a burst.
        self._next_at = (int(obs.t / self.interval) + 1) * self.interval
        return True

    def report(self, obs: Observation) -> None:
        people = " ".join(
            f"{name}=({x:.2f},{y:.2f})"
            for name, (x, y) in zip(self.names, obs.people_xy)
        )
        nearest = self.nearest_person(obs)
        gap = f" nearest={nearest:.2f}m" if nearest is not None else ""
        log(
            f"t={obs.t:5.1f}s ugv=({obs.robot_xy[0]:.2f},{obs.robot_xy[1]:.2f}) "
            f"yaw={obs.robot_yaw:+.2f}{gap} {people}".rstrip()
        )

    @staticmethod
    def nearest_person(obs: Observation):
        """Distance to the closest person, or None when nobody is in the scene."""
        if not obs.people_xy:
            return None
        return min(
            math.hypot(obs.robot_xy[0] - x, obs.robot_xy[1] - y)
            for x, y in obs.people_xy
        )
