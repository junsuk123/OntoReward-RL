"""People-aware navigation: go to waypoints while keeping clear of pedestrians."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from simlab.algorithms.base import DriveCommand, DriveController, Observation

Point = Tuple[float, float]


def wrap_angle(angle: float) -> float:
    """Fold an angle into [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _as_points(raw: Sequence) -> List[Point]:
    points: List[Point] = []
    for item in raw:
        values = [float(v) for v in item]
        if len(values) < 2:
            raise ValueError(f"waypoint needs at least x and y, got {item!r}")
        points.append((values[0], values[1]))
    return points


@dataclass
class SocialForceController(DriveController):
    """Potential-field controller: attracted to the goal, repelled by people.

    The attractive term is a unit vector toward the active waypoint. Each person
    inside ``influence_radius`` adds a repulsive term that grows as they get
    closer, using the classic ``1/d - 1/R`` shape so it decays smoothly to zero
    at the edge of the radius instead of switching on abruptly.

    Forward speed is scaled by ``cos`` of the heading error, so the robot turns
    in place when the goal is behind it, and is scaled down again when someone
    is within ``caution_radius``.
    """

    waypoints: List[Point] = field(
        default_factory=lambda: [(2.0, 0.0), (0.0, 2.0), (-2.0, 0.0), (0.0, -2.0)]
    )
    goal_tolerance: float = 0.6  # m, how close counts as "arrived"
    linear_speed: float = 0.6  # m/s cruise
    max_angular_speed: float = 1.2  # rad/s
    heading_gain: float = 1.6  # rad/s per rad of heading error
    influence_radius: float = 2.5  # m, people further away are ignored
    repulsion_gain: float = 1.6
    caution_radius: float = 1.5  # m, start slowing down inside this
    stop_radius: float = 0.9  # m, back away instead of pushing through
    reverse_speed: float = 0.5  # m/s used while backing off
    swirl_gain: float = 0.9  # tangential push, breaks head-on deadlocks
    min_speed_factor: float = 0.1  # never fully stop unless blocked head-on
    loop: bool = True

    name = "social_force"

    def __post_init__(self) -> None:
        self.waypoints = _as_points(self.waypoints)
        if not self.waypoints:
            raise ValueError("social_force needs at least one waypoint")
        for field_name in ("goal_tolerance", "influence_radius", "caution_radius", "stop_radius"):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.stop_radius >= self.caution_radius:
            raise ValueError("stop_radius must be smaller than caution_radius")
        self._index = 0

    # -- state -------------------------------------------------------------
    def reset(self) -> None:
        self._index = 0

    @property
    def goal(self) -> Point:
        return self.waypoints[self._index]

    def _advance_if_reached(self, robot_xy: Point) -> None:
        gx, gy = self.goal
        if math.hypot(gx - robot_xy[0], gy - robot_xy[1]) > self.goal_tolerance:
            return
        if self._index + 1 < len(self.waypoints):
            self._index += 1
        elif self.loop:
            self._index = 0

    # -- forces ------------------------------------------------------------
    def _attraction(self, robot_xy: Point) -> Tuple[float, float]:
        dx, dy = self.goal[0] - robot_xy[0], self.goal[1] - robot_xy[1]
        distance = math.hypot(dx, dy)
        if distance < 1e-6:
            return (0.0, 0.0)
        return (dx / distance, dy / distance)

    def _repulsion(
        self, robot_xy: Point, people: Sequence[Point], attraction: Tuple[float, float]
    ) -> Tuple[Tuple[float, float], float, Optional[Point]]:
        """Summed repulsion vector, plus the distance to the nearest person.

        Pure radial repulsion deadlocks head-on: a person directly between the
        robot and the goal pushes back exactly along the attraction, the two
        cancel in the lateral axis and the robot drives straight at them. Each
        person therefore also gets a tangential ("swirl") push, oriented toward
        whichever side leads to the goal, so the robot slides around instead.
        """
        fx = fy = 0.0
        nearest = math.inf
        nearest_point: Optional[Point] = None
        for px, py in people:
            dx, dy = robot_xy[0] - px, robot_xy[1] - py
            distance = math.hypot(dx, dy)
            if distance < nearest:
                nearest, nearest_point = distance, (px, py)
            if distance >= self.influence_radius:
                continue
            distance = max(distance, 1e-3)  # never divide by zero on a direct hit
            ux, uy = dx / distance, dy / distance
            strength = self.repulsion_gain * (1.0 / distance - 1.0 / self.influence_radius)
            fx += strength * ux
            fy += strength * uy

            # Rotate the away-vector by +90 deg, flipping it if the other side
            # points more towards the goal. An exact tie keeps +90, so the robot
            # always picks the same side rather than dithering.
            tx, ty = -uy, ux
            if tx * attraction[0] + ty * attraction[1] < 0.0:
                tx, ty = uy, -ux
            fx += self.swirl_gain * strength * tx
            fy += self.swirl_gain * strength * ty
        return (fx, fy), nearest, nearest_point

    # -- control -----------------------------------------------------------
    def step(self, obs: Observation) -> DriveCommand:
        self._advance_if_reached(obs.robot_xy)

        ax, ay = self._attraction(obs.robot_xy)
        (rx, ry), nearest, nearest_point = self._repulsion(
            obs.robot_xy, obs.people_xy, (ax, ay)
        )

        # Someone is inside the safety bubble. Repulsion alone only slows the
        # robot down, which is not enough when a pedestrian walks into it, so
        # give up on the goal, turn away and back off until they have passed.
        if nearest_point is not None and nearest < self.stop_radius:
            away = math.atan2(
                obs.robot_xy[1] - nearest_point[1], obs.robot_xy[0] - nearest_point[0]
            )
            error = wrap_angle(away - obs.robot_yaw)
            angular = max(
                -self.max_angular_speed,
                min(self.max_angular_speed, self.heading_gain * error),
            )
            backoff = self.reverse_speed * (1.0 - nearest / self.stop_radius)
            return DriveCommand(linear=-backoff, angular=angular)

        dx, dy = ax + rx, ay + ry
        if math.hypot(dx, dy) < 1e-6:
            # Attraction and repulsion cancelled out; turn away from the crowd.
            return DriveCommand(linear=0.0, angular=self.max_angular_speed * 0.5)

        heading_error = wrap_angle(math.atan2(dy, dx) - obs.robot_yaw)
        angular = max(
            -self.max_angular_speed,
            min(self.max_angular_speed, self.heading_gain * heading_error),
        )

        # Only drive forward to the extent we are already facing the right way.
        speed = self.linear_speed * max(0.0, math.cos(heading_error))
        if nearest < self.caution_radius:
            factor = max(self.min_speed_factor, nearest / self.caution_radius)
            speed *= factor
        return DriveCommand(linear=speed, angular=angular)
