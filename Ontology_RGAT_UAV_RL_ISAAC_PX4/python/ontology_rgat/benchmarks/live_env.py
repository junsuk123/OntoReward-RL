"""Live Isaac/Pegasus/PX4 environment for the controlled benchmark."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..bridge import PX4Bridge
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
        self.bridge = PX4Bridge(cfg)
        self.adapter = ShinPX4Adapter(
            self.bridge, image_source,
            VelocityYawRateController(dt=float(cfg.sim.dt)))
        self.horizon_steps = int(horizon_steps)
        self.steps = 0
        self.last_step: LiveStep | None = None

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
        # The same scalar scales platform speed and perturbations for every arm.
        self.bridge.cfg.pad_scale = float(np.clip(curriculum, 0.0, 1.0))
        actor, state = self.adapter.reset(int(seed), scenario=scenario)
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
                elif self.last_step is not None and self.last_step.crash:
                    # A ground/tilt crash is not an airborne staging state.
                    # Complete PX4's landing/disarm before the next physical
                    # hover; otherwise the following reset tries to hold a
                    # grounded, failsafe-controlled vehicle in OFFBOARD.
                    self.bridge.land_and_wait()
                elif bool(self.cfg.external.get("start_airborne", False)):
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
