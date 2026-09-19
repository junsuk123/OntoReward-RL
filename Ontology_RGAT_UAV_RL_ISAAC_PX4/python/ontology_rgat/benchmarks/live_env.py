"""Live Isaac/Pegasus/PX4 environment for the controlled benchmark."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..bridge import BridgeError, PX4Bridge
from ..controllers import VelocityYawRateController
from ..initialization import curriculum_motion_scale
from ..mathx import quat_to_euler_zyx
from ..perception.pad_geometry import CameraModel, project_landing_pad
from .px4_adapter import (ShinPX4Adapter, critic_observation_from_state)
from .shin2026 import ActorObservation, CriticObservation


@dataclass(frozen=True)
class LiveStep:
    actor: ActorObservation
    critic: CriticObservation
    state: dict
    command: np.ndarray
    physical_contact: bool
    unsafe_pad_contact: bool
    crash: bool
    excessive_drift: bool
    battery_depleted: bool
    terminal: bool
    timeout: bool
    strict_success: bool
    # Geometric camera-frustum visibility of the landing-pad centre, from
    # simulator geometry alone. Evaluation/label use only -- it is never part
    # of ``actor``, never reaches the online R-GAT, and is deliberately
    # independent of how well the learned keypoint encoder is doing.
    geometric_pad_center_in_fov: bool
    landing_metrics: dict


class LiveShinEnvironment:
    """One common environment implementation for every reward arm."""

    def __init__(self, cfg, image_source, *, horizon_steps=300):
        self.cfg = cfg
        self.image_source = image_source
        # The rendered landing camera, so geometric FOV truth is evaluated
        # against the camera Isaac actually renders with.
        self.landing_camera = CameraModel.from_mapping(
            getattr(cfg.external, "landing_camera", None))
        self._connect()
        self.horizon_steps = int(horizon_steps)
        self.steps = 0
        self.last_step: LiveStep | None = None

    def _connect(self):
        from .. import stack as stack_module
        owned = stack_module.current()
        # Parallel workers share one Isaac/PX4 stack, so a peer may be halfway
        # through rebuilding it. Saying hello into that window reaches a
        # gateway that has been stopped and not started again, and the setup
        # budget is sized for a boot already under way rather than for a full
        # rebuild -- it expires on a simulator that was never going to answer.
        if owned is not None and hasattr(owned, "wait_for_restart"):
            owned.wait_for_restart()
        self.bridge = PX4Bridge(self.cfg)
        self._stack_generation = int(getattr(owned, "generation", 0))
        self.control = getattr(self.cfg, "benchmark_control", {})
        self.adapter = ShinPX4Adapter(
            self.bridge, self.image_source,
            VelocityYawRateController.from_mapping(
                self.control, dt=float(self.cfg.sim.dt)))

    def geometric_pad_center_in_fov(self, state) -> bool:
        """Simulator-geometry field-of-view truth for the landing-pad centre.

        The pad centre is transformed into the camera optical frame, required
        to have positive depth, projected with the configured intrinsics and
        tested against the normalized frame bounds.  Neither marker decoding
        nor learned keypoint confidence takes part: a pad that is blurred,
        occluded or simply not detected has *not* left the field of view, and
        reporting it as an FOV loss is exactly the contradiction this replaces.

        Attitude comes from simulator truth when the gateway publishes it, so
        the quantity is geometric end to end; older recordings without
        attitude truth fall back to the estimator's attitude, which is
        reported through ``extra`` rather than hidden.
        """
        truth = state.get("truth") if isinstance(state.get("truth"), dict) else {}
        if not truth.get("valid", False):
            raise ValueError(
                "geometric pad-centre FOV requires valid simulator truth")
        camera = getattr(self, "landing_camera", None)
        if camera is None:
            camera = CameraModel.from_mapping(
                getattr(self.cfg.external, "landing_camera", None)
                if hasattr(self.cfg, "external") else None)
            self.landing_camera = camera
        quaternion = (truth["quaternion_wxyz"]
                      if truth.get("attitude_valid", False)
                      else state["quaternion_wxyz"])
        projection = project_landing_pad(
            truth["position"], quaternion, camera=camera)
        return bool(projection.geometric_pad_center_in_fov)

    def _classify(self, actor, state, command, *, timeout=False) -> LiveStep:
        critic = critic_observation_from_state(actor, state)
        extra = state.get("extra") or {}
        contact = bool(extra.get("pad_contact", False))
        # The contact topic and vehicle odometry are asynchronous.  The first
        # state carrying pad_contact may therefore already contain the upward
        # rebound caused by the deck impulse.  Judging that post-impact sample
        # as the touchdown velocity rejected gentle, centred landings.  Keep
        # the contact-position sample, but judge touchdown kinematics from the
        # latest known airborne sample.  This also remains conservative for a
        # genuinely hard approach because its pre-impact descent is retained.
        touchdown_step = None
        previous_step = getattr(self, "last_step", None)
        if (contact and previous_step is not None
                and not previous_step.physical_contact):
            touchdown_step = previous_step
        touchdown_actor = touchdown_step.actor if touchdown_step else actor
        touchdown_critic = touchdown_step.critic if touchdown_step else critic
        touchdown_state = touchdown_step.state if touchdown_step else state
        touchdown_rpy = quat_to_euler_zyx(np.asarray(
            touchdown_state["quaternion_wxyz"], dtype=float))
        touchdown_tilt = float(np.linalg.norm(touchdown_rpy[:2]))
        raw_truth = state.get("truth") or {}
        truth_position = np.asarray(raw_truth.get("position", (math.inf,) * 3), dtype=float)
        drift = bool(np.linalg.norm(truth_position[:2]) > float(self.cfg.sim.world_xy_limit))
        truth_is_finite = bool(truth_position.shape == (3,)
                               and np.isfinite(truth_position).all())
        near_deck = bool(truth_is_finite
                         and truth_position[2] <= max(0.5, float(self.cfg.sim.ground_z)))
        off_pad_ground = bool(
            state.get("landed", False) and not contact and self.steps > 0
            and (near_deck or not truth_is_finite))
        battery = state.get("battery") if isinstance(state.get("battery"), dict) else {}
        battery_depleted = bool(battery.get("enabled", False)
                                and battery.get("depleted", False))
        rel = critic.true_relative_state
        touchdown_rel = touchdown_critic.true_relative_state
        touchdown_vertical_velocity = float(touchdown_actor.body_velocity[2])
        touchdown_relative_horizontal_speed = float(
            np.linalg.norm(touchdown_rel[3:5]))
        touchdown_angular_rate = float(np.linalg.norm(np.asarray(
            touchdown_state["angular_velocity"], dtype=float)))
        strict = bool(contact
                      and np.linalg.norm(rel[:2]) <= float(self.cfg.criteria.xy)
                      and abs(touchdown_vertical_velocity) <= float(
                          self.cfg.criteria.vz)
                      and touchdown_relative_horizontal_speed <= float(
                          self.cfg.criteria.rel_speed_xy)
                      and touchdown_tilt <= float(self.cfg.criteria.tilt)
                      and touchdown_angular_rate <= float(self.cfg.criteria.rate))
        unsafe_contact = bool(contact and not strict)
        crash = bool(off_pad_ground
                     or touchdown_tilt > float(self.cfg.sim.crash_tilt)
                     or unsafe_contact)
        terminal = bool(contact or crash or drift or battery_depleted or timeout)
        landing_metrics = {
            "lateral_error": float(np.linalg.norm(rel[:2])),
            "vertical_velocity": touchdown_vertical_velocity,
            "relative_horizontal_speed": touchdown_relative_horizontal_speed,
            "tilt": touchdown_tilt,
            "roll": float(touchdown_rpy[0]),
            "pitch": float(touchdown_rpy[1]),
            "angular_rate": touchdown_angular_rate,
            "kinematic_sample": (
                "pre_contact" if touchdown_step is not None else "current"),
        }
        return LiveStep(
            actor=actor, critic=critic, state=state,
            command=np.asarray(command, dtype=float), physical_contact=contact,
            unsafe_pad_contact=unsafe_contact,
            crash=crash, excessive_drift=drift,
            battery_depleted=battery_depleted, terminal=terminal,
            timeout=bool(timeout), strict_success=strict,
            geometric_pad_center_in_fov=self.geometric_pad_center_in_fov(state),
            landing_metrics=landing_metrics)

    def reset(self, seed: int, curriculum: float = 1.0,
              scenario: str = "training_random_walk") -> LiveStep:
        self.steps = 0
        # Never let the previous episode become the pre-contact sample for a
        # reset that starts with a stale latched contact bit.
        self.last_step = None
        # Initial pose and action difficulty still start at c=0, but a separate
        # floor keeps the UGV visibly and observably moving from episode one.
        # A replacement adapter after recovery receives the same two scales.
        curriculum = float(np.clip(curriculum, 0.0, 1.0))
        control = getattr(self, "control", {})
        minimum_motion = float(control.get(
            "curriculum_min_pad_motion_scale", 0.35))
        motion_scale = curriculum_motion_scale(curriculum, minimum_motion)
        self.pad_motion_scale = motion_scale
        from .. import stack as stack_module

        attempts = 1 + max(0, int(self.cfg.external.reset_recoveries))
        for attempt in range(1, attempts + 1):
            self.bridge.cfg.pad_scale = motion_scale
            self.bridge.cfg.require_pad_in_view = bool(
                control.get("require_initial_pad_visible",
                            control.get(
                                "require_initial_pad_visible_during_curriculum",
                                True)))
            self.adapter.controller.set_curriculum(curriculum)
            try:
                actor, state = self.adapter.reset(
                    int(seed), scenario=scenario,
                    initial_condition_scale=curriculum)
                break
            except BridgeError as exc:
                from ..bridge import ArmingRefused

                owned = stack_module.current()
                if (isinstance(exc, ArmingRefused)
                        and not stack_module.can_replace_simulator()):
                    # No reset clears a failed preflight, and this run cannot
                    # give itself a new simulator to clear it with.
                    raise
                if attempt == attempts or owned is None:
                    raise
                self.bridge.close()
                pair_index = int(getattr(self.cfg.external, "pair_index", 0))
                print(f"WARNING: [pair {pair_index}] benchmark reset failed "
                      f"({exc}). Restarting the "
                      f"simulator and retrying ({attempt} of {attempts - 1}).")
                if hasattr(owned, "restart_if_generation"):
                    owned.restart_if_generation(self._stack_generation)
                else:
                    owned.restart()
                self._connect()
        self.last_step = self._classify(actor, state, np.zeros(4))
        return self.last_step

    def step(self, normalized_action) -> LiveStep:
        actor, state, command = self.adapter.step(normalized_action)
        self.steps += 1
        self.last_step = self._classify(
            actor, state, command.as_array(),
            timeout=self.steps >= self.horizon_steps)
        return self.last_step

    def recover_infrastructure(self, attempts: int = 3) -> bool:
        """Cycle an owned SITL stack after a mid-episode transport failure.

        A partially observed episode must never enter PPO. The caller discards
        it and retries the same seed after this method establishes a fresh
        bridge. Hardware or manually adopted stacks are intentionally not
        restarted because the learner does not own them.

        The reconnection is retried for the same reason opening the environment
        is: several workers share one stack, and a hello can still land in the
        window where a peer's rebuild has torn the gateway down but not yet
        brought it back. ``wait_for_restart`` only says that no rebuild holds
        the lock right now, not that the simulator answers -- so a single
        timeout here used to end a checkpointed multi-hour run outright.
        """
        from .. import stack as stack_module

        owned = stack_module.current()
        if owned is None:
            raise BridgeError(
                "cannot recover the flight infrastructure because this run "
                "does not own the simulator stack")
        self.bridge.close()
        if hasattr(owned, "restart_if_generation"):
            restarted = bool(owned.restart_if_generation(self._stack_generation))
        else:
            owned.restart()
            restarted = True
        attempts = max(1, int(attempts))
        for attempt in range(1, attempts + 1):
            generation = int(getattr(owned, "generation", 0))
            try:
                self._connect()
                return restarted
            except BridgeError as exc:
                if attempt >= attempts:
                    raise
                print(f"WARNING: could not reach the gateway after recovering "
                      f"the simulator ({exc}). Cycling it and retrying "
                      f"({attempt} of {attempts - 1}).")
                # A peer may already have rebuilt the stack while this worker
                # was waiting on its hello; then there is nothing left to
                # cycle and the next attempt simply reconnects.
                if hasattr(owned, "restart_if_generation"):
                    owned.restart_if_generation(generation)
                else:
                    owned.restart()
                restarted = True
        raise AssertionError("unreachable infrastructure-recovery state")

    def finish_episode(self):
        if self.bridge.last_state:
            try:
                extra = self.bridge.last_state.get("extra") or {}
                if (self.bridge.last_state.get("landed", False)
                        or extra.get("pad_contact", False)):
                    self.bridge.stop_after_outcome()
                elif bool(self.cfg.external.get("start_airborne", False)):
                    # A tilt threshold can terminate the measured episode while
                    # the vehicle is still physically airborne. Sending that
                    # case to AUTO.LAND left PX4 descending through the next
                    # reset and eventually made every arm request fail. PX4's
                    # position controller can recover the attitude in flight;
                    # stage it at a bounded hover and only use the landed branch
                    # above after genuine ground/deck contact.
                    self.bridge.hold_for_next_airborne_reset()
                else:
                    self.bridge.land_and_wait()
            except Exception:
                # Reset recovery owns stack cycling if PX4 cannot complete the
                # between-episode landing.
                pass

    def close(self):
        self.bridge.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
