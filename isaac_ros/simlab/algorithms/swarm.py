"""Straight shuttle trajectories with randomized camera-view crossings."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from simlab.config.schema import DronesConfig, ScenariosConfig
from simlab.scenarios.motion import LanePlan, lane_plans

Vec3 = Tuple[float, float, float]


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(v: Vec3, scale: float) -> Vec3:
    return (v[0] * scale, v[1] * scale, v[2] * scale)


@dataclass
class DroneState:
    name: str
    team: str
    model: str
    position: Vec3
    velocity: Vec3 = (0.0, 0.0, 0.0)
    yaw: float = 0.0
    cruise_altitude: float = 3.2
    #: The episode-long assignment: lane depth, crossing group, formation slot.
    plan: LanePlan | None = None

    @property
    def start_side(self) -> float:
        return self.plan.start_side if self.plan else 1.0


class SwarmController:
    """Move the roster along the activity bar and overlap it in the front cameras.

    Each aircraft flies its own lane on the camera's optical axis, so aircraft
    that share a crossing point merge in the image without ever closing to
    within ``drones.safety_radius`` in 3-D.
    """

    def __init__(self, cfg: DronesConfig, scenarios: ScenariosConfig | None = None) -> None:
        self.cfg = cfg
        self.scenarios = scenarios
        self.states: List[DroneState] = []
        self.minimum_separation = math.inf
        self.seed = (
            cfg.trajectory_seed
            if cfg.trajectory_seed is not None
            else self._scenario_seed()
            or random.SystemRandom().randrange(0, 2**31)
        )
        self._crossings: dict[Tuple[int, int], Tuple[float, float]] = {}
        self._build_roster()

    def _scenario_seed(self) -> int | None:
        """An episode seed makes the whole run reproducible from its YAML alone."""
        if self.scenarios is not None and self.scenarios.enabled:
            return self.scenarios.seed
        return None

    def _build_roster(self) -> None:
        cfg = self.cfg
        teams = (("friendly", cfg.friendly), ("enemy", cfg.enemy))
        plans = lane_plans(cfg, self.scenarios, self.seed)
        index_in_plan = 0
        for team_name, team in teams:
            for index in range(team.count):
                plan = plans[index_in_plan]
                index_in_plan += 1
                prefix = "".join(ch if ch.isalnum() else "_" for ch in team.label).strip("_")
                prefix = prefix or ("Ally" if team_name == "friendly" else "Enemy")
                start_y = plan.start_side * plan.half_length + plan.formation_offset_y
                self.states.append(
                    DroneState(
                        name=f"{prefix}_{index + 1:02d}",
                        team=team_name,
                        model=team.model,
                        position=(plan.lane_x, start_y, 0.12),
                        cruise_altitude=team.cruise_altitude,
                        plan=plan,
                    )
                )

    def crossing_parameters(self, leg: int, group: int = 0) -> Tuple[float, float]:
        """This leg's random (y coordinate, normalized crossing time) for a group.

        Aircraft in the same crossing group receive the same answer, which is
        what puts them on the same pixel at the same moment.
        """
        key = (leg, group)
        if key not in self._crossings:
            generator = random.Random(self.seed + 104729 * leg + 7919 * group)
            y = generator.uniform(
                -self.cfg.crossing_offset_limit, self.cfg.crossing_offset_limit
            )
            fraction = 0.5 + generator.uniform(
                -self.cfg.crossing_time_jitter, self.cfg.crossing_time_jitter
            )
            self._crossings[key] = (y, fraction)
        return self._crossings[key]

    def _target_position(self, state: DroneState, t: float) -> Vec3:
        cfg = self.cfg
        plan = state.plan
        x = plan.lane_x
        start_y = plan.start_side * plan.half_length + plan.formation_offset_y
        if t < cfg.takeoff_duration_s:
            progress = max(0.0, min(1.0, t / cfg.takeoff_duration_s))
            smooth = progress * progress * (3.0 - 2.0 * progress)
            z = 0.12 + (state.cruise_altitude - 0.12) * smooth
            return (x, start_y, z)

        elapsed = t - cfg.takeoff_duration_s
        leg = int(elapsed // plan.leg_duration_s)
        progress = (elapsed % plan.leg_duration_s) / plan.leg_duration_s
        direction = -1.0 if leg % 2 else 1.0
        leg_start = plan.start_side * plan.half_length * direction + plan.formation_offset_y
        leg_end = -plan.start_side * plan.half_length * direction + plan.formation_offset_y
        # The crossing point itself carries no formation offset: that is what
        # makes a group converge on one image position and then fan back out.
        crossing_y, crossing_time = self.crossing_parameters(leg, plan.crossing_group)
        if progress <= crossing_time:
            local = progress / crossing_time
            y = leg_start + (crossing_y - leg_start) * local
        else:
            local = (progress - crossing_time) / (1.0 - crossing_time)
            y = crossing_y + (leg_end - crossing_y) * local
        return (x, y, state.cruise_altitude)

    def step(self, t: float, dt: float) -> Sequence[DroneState]:
        for state in self.states:
            previous = state.position
            state.position = self._target_position(state, t)
            state.velocity = _scale(_sub(state.position, previous), 1.0 / max(dt, 1e-6))
            horizontal_speed = math.hypot(state.velocity[0], state.velocity[1])
            if horizontal_speed > 0.08:
                state.yaw = math.atan2(state.velocity[1], state.velocity[0])
        for index, state in enumerate(self.states):
            for other in self.states[index + 1 :]:
                self.minimum_separation = min(
                    self.minimum_separation, math.dist(state.position, other.position)
                )
        return self.states
