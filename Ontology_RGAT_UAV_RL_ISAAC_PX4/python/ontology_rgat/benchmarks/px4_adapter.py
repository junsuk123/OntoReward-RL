"""PX4/Isaac adapter enforcing the benchmark information boundary."""
from __future__ import annotations

from typing import Callable

import numpy as np

from ..bridge import SimulatorFrameTimeout
from ..controllers import PlanarCommand, PlanarLongitudinalController
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
    """Common action/observation adapter used unchanged by every reward mode.

    It owns the two constraints that make the envelope planar, and it owns them
    for every arm at once: the lateral velocity the controller emits is already
    zero, and the heading the gateway is told to hold is the one the entry was
    flown to -- the deck's own constant heading. Neither is something a policy
    or a reward mode can reach.
    """

    def __init__(self, bridge, image_source: Callable[[], np.ndarray],
                 controller: PlanarLongitudinalController):
        self.bridge = bridge
        self.image_source = image_source
        self.controller = controller
        # Set at every reset from the reset acknowledgement; ``None`` leaves
        # the gateway's free-yaw behaviour in place.
        self.yaw_hold_rad: float | None = None

    def _image(self) -> np.ndarray:
        """One frame, with a stalled renderer reported as infrastructure.

        This is the only place the camera enters the RL loop, and the only
        place that knows a missing frame means the simulator rather than the
        task. The ROS buffer raises a bare ``TimeoutError``; left as one it
        escapes the episode retry that exists for exactly this condition.
        """
        try:
            return self.image_source()
        except TimeoutError as exc:
            raise SimulatorFrameTimeout(
                f"the actor camera delivered no frame: {exc}") from exc

    def reset(self, seed: int, scenario: str = "training_random_walk",
              *, initial_condition_scale: float | None = None):
        self.controller.reset()
        state = self.bridge.reset(
            seed, scenario=scenario,
            initial_condition_scale=initial_condition_scale)
        self.yaw_hold_rad = self._entry_heading()
        return actor_observation_from_state(state, self._image()), state

    def _entry_heading(self) -> float | None:
        """The absolute heading the entry was flown to, in ENU.

        On a planar profile this is the deck's heading: the entry draw sets the
        yaw misalignment to zero and the deck's heading is constant for the
        episode, so holding it is the "always aligned with the deck" constraint
        expressed as a setpoint rather than as an assumption. A gateway or a
        recording that carries no entry yaw leaves the hold off.
        """
        detail = (getattr(self.bridge, "last_reset_ack", {}) or {}).get("detail")
        if not isinstance(detail, dict) or "entry_yaw_enu_rad" not in detail:
            return None
        heading = float(detail["entry_yaw_enu_rad"])
        return heading if np.isfinite(heading) else None

    def step(self, normalized_action):
        command = self.controller.command(normalized_action)
        tilt = float(getattr(command, "longitudinal_tilt_rad", 0.0))
        state = self.bridge.step_velocity(
            command.as_array(), tilt_rad=tilt, yaw_rad=self.yaw_hold_rad)
        return actor_observation_from_state(state, self._image()), state, command
