"""Robot decision-making. Pure Python -- no Isaac Sim import, so unit-testable.

A controller consumes an :class:`Observation` (robot pose + people positions at
time ``t``) and returns a :class:`DriveCommand` (body-frame linear/angular
velocity). The simulation layer is responsible for turning that into wheel
velocities.
"""

from simlab.algorithms.avoidance import SocialForceController, wrap_angle
from simlab.algorithms.base import DriveCommand, DriveController, Observation
from simlab.algorithms.registry import CONTROLLERS, build_controller

__all__ = [
    "DriveCommand",
    "DriveController",
    "Observation",
    "SocialForceController",
    "wrap_angle",
    "CONTROLLERS",
    "build_controller",
]
