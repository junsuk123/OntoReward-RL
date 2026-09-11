"""PX4/Isaac adapter enforcing the benchmark information boundary."""
from __future__ import annotations

from typing import Callable

import numpy as np

from ..controllers import VelocityYawRateController
from ..mathx import quat_to_rotm
from .shin2026 import ActorObservation, CriticObservation


def actor_observation_from_state(state: dict, image: np.ndarray) -> ActorObservation:
    """Build actor input from UAV-only telemetry and raw onboard pixels."""
    quaternion = np.asarray(state["quaternion_wxyz"], dtype=float)
    world = state.get("world") if isinstance(state.get("world"), dict) else {}
    velocity_world = np.asarray(world.get("velocity", state.get("world_velocity", ())),
                                dtype=float).reshape(-1)
    if velocity_world.shape != (3,):
        raise ValueError("gateway must expose the UAV's own world velocity for Shin actor")
    velocity_body = quat_to_rotm(quaternion).T @ velocity_world
    return ActorObservation(image=image, body_velocity=velocity_body,
                            attitude_quaternion=quaternion)


def critic_observation_from_state(actor: ActorObservation, state: dict) -> CriticObservation:
    """Transform simulator UAV-minus-pad truth to platform-in-body truth."""
    truth = state.get("truth") if isinstance(state.get("truth"), dict) else {}
    if not truth.get("valid", False):
        raise ValueError("asymmetric critic requires valid training-only simulator truth")
    rotation = quat_to_rotm(actor.attitude_quaternion)
    platform_position_body = rotation.T @ -np.asarray(truth["position"], dtype=float)
    platform_velocity_body = rotation.T @ -np.asarray(truth["velocity"], dtype=float)
    return CriticObservation(
        actor=actor,
        true_relative_state=np.r_[platform_position_body, platform_velocity_body])


class ShinPX4Adapter:
    """Common action/observation adapter used unchanged by every reward mode."""

    def __init__(self, bridge, image_source: Callable[[], np.ndarray],
                 controller: VelocityYawRateController):
        self.bridge = bridge
        self.image_source = image_source
        self.controller = controller

    def reset(self, seed: int, scenario: str = "training_random_walk"):
        self.controller.reset()
        state = self.bridge.reset(seed, scenario=scenario)
        return actor_observation_from_state(state, self.image_source()), state

    def step(self, normalized_action):
        command = self.controller.command(normalized_action)
        state = self.bridge.step_velocity(command.as_array())
        return actor_observation_from_state(state, self.image_source()), state, command
