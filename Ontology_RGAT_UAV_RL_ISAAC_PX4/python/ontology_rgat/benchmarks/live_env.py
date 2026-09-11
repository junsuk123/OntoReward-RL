"""Live Isaac/Pegasus/PX4 environment for the controlled benchmark."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..bridge import BridgeError, PX4Bridge
from ..controllers import VelocityYawRateController
from ..mathx import quat_to_euler_zyx
from .px4_adapter import (ShinPX4Adapter, critic_observation_from_state)
from .shin2026 import ActorObservation, CriticObservation


@dataclass(frozen=True)
class LiveStep:
    actor: ActorObservation
    critic: CriticObservation
    state: dict
    command: np.ndarray
    physical_contact: bool
    crash: bool
    excessive_drift: bool
    terminal: bool
    timeout: bool
    strict_success: bool
    pad_in_fov: bool


class LiveShinEnvironment:
    """One common environment implementation for every reward arm."""

    def __init__(self, cfg, image_source, *, horizon_steps=300):
        self.cfg = cfg
        self.image_source = image_source
        self._connect()
        self.horizon_steps = int(horizon_steps)
        self.steps = 0
        self.last_step: LiveStep | None = None

    def _connect(self):
        self.bridge = PX4Bridge(self.cfg)
        control = getattr(self.cfg, "benchmark_control", {})
        self.adapter = ShinPX4Adapter(
            self.bridge, self.image_source,
            VelocityYawRateController.from_mapping(
                control, dt=float(self.cfg.sim.dt)))

    def _classify(self, actor, state, command, *, timeout=False) -> LiveStep:
        critic = critic_observation_from_state(actor, state)
        extra = state.get("extra") or {}
        contact = bool(state.get("landed", False) and extra.get("pad_contact", False))
        rpy = quat_to_euler_zyx(np.asarray(state["quaternion_wxyz"], dtype=float))
        tilt = float(np.linalg.norm(rpy[:2]))
        raw_truth = state.get("truth") or {}
        truth_position = np.asarray(raw_truth.get("position", (math.inf,) * 3), dtype=float)
        drift = bool(np.linalg.norm(truth_position[:2]) > float(self.cfg.sim.world_xy_limit))
        off_pad_ground = bool(state.get("landed", False) and not contact and self.steps > 0)
        crash = bool(off_pad_ground or tilt > float(self.cfg.sim.crash_tilt))
        terminal = bool(contact or crash or drift or timeout)
        rel = critic.true_relative_state
        angular_rate = float(np.linalg.norm(np.asarray(state["angular_velocity"], dtype=float)))
        strict = bool(contact
                      and np.linalg.norm(rel[:2]) <= float(self.cfg.criteria.xy)
                      and abs(actor.body_velocity[2]) <= float(self.cfg.criteria.vz)
                      and np.linalg.norm(rel[3:5]) <= float(self.cfg.criteria.rel_speed_xy)
                      and tilt <= float(self.cfg.criteria.tilt)
                      and angular_rate <= float(self.cfg.criteria.rate))
        return LiveStep(
            actor=actor, critic=critic, state=state,
            command=np.asarray(command, dtype=float), physical_contact=contact,
            crash=crash, excessive_drift=drift, terminal=terminal,
            timeout=bool(timeout), strict_success=strict,
            pad_in_fov=float(state.get("marker_quality", 0.0)) > 0.0)

    def reset(self, seed: int, curriculum: float = 1.0,
              scenario: str = "training_random_walk") -> LiveStep:
        self.steps = 0
        # The same scalar scales both platform difficulty and the UAV action
        # envelope for every reward arm. A replacement adapter after recovery
        # receives the same level before its reset.
        curriculum = float(np.clip(curriculum, 0.0, 1.0))
        from .. import stack as stack_module

        attempts = 1 + max(0, int(self.cfg.external.reset_recoveries))
        for attempt in range(1, attempts + 1):
            self.bridge.cfg.pad_scale = curriculum
            self.adapter.controller.set_curriculum(curriculum)
            try:
                actor, state = self.adapter.reset(int(seed), scenario=scenario)
                break
            except BridgeError as exc:
                owned = stack_module.current()
                if attempt == attempts or owned is None:
                    raise
                self.bridge.close()
                print(f"WARNING: benchmark reset failed ({exc}). Restarting the "
                      f"simulator and retrying ({attempt} of {attempts - 1}).")
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
