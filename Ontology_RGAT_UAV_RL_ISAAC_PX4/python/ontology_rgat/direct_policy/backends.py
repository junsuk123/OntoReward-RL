"""Replay and lightweight local backends for pre-Isaac regression gates."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from .contracts import ContractBundle
from .observation import PlanarEstimate, SpatialEstimate, planar_vector, spatial_vector
from .reward import (TerminalEvent, TruthState, ViewMeasurement, classify_terminal,
                     reward_v5)


@dataclass(frozen=True)
class StepResult:
    observation: np.ndarray
    reward: float
    terminated: bool
    status: str
    info: dict


class ReferenceReplayBackend:
    """Replay stored exogenous/observation fixtures without synthesizing seeds."""
    name = "replay"

    def __init__(self, path):
        path = Path(path)
        if path.suffix == ".npz":
            payload = np.load(path, allow_pickle=False)
            self.observations = np.asarray(payload["observations"], dtype=np.float64)
            self.rewards = np.asarray(payload.get("rewards", np.zeros(len(self.observations))))
            self.statuses = [str(item) for item in payload.get(
                "statuses", np.array(["RUNNING"]*len(self.observations)))]
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.observations = np.asarray(payload["observations"], dtype=np.float64)
            self.rewards = np.asarray(payload.get("rewards", [0]*len(self.observations)))
            self.statuses = payload.get("statuses", ["RUNNING"]*len(self.observations))
        self.index = 0

    def reset(self, seed=None):
        self.index = 0
        return self.observations[0].copy(), {"backend": self.name, "seed": seed}

    def step(self, action):
        self.index += 1
        if self.index >= len(self.observations):
            raise RuntimeError("replay exhausted")
        status = self.statuses[self.index]
        return StepResult(self.observations[self.index].copy(),
                          float(self.rewards[self.index]), status != "RUNNING", status,
                          {"backend": self.name, "requested_acceleration": list(action),
                           "applied_acceleration": list(action), "intervened": False})


class LocalReferenceBackend:
    """Small deterministic plant for import/smoke/training wiring tests.

    It is not reported as MATLAB parity or Isaac flight.  A separate sensor RNG
    creates the pad estimate consumed by the policy; truth remains confined to
    the simulated sensor, contact classifier and reward.
    """
    name = "local-reference"

    def __init__(self, contract: ContractBundle):
        self.contract = contract
        self.dimension = int(contract.metadata["dimension"])
        self.dt = contract.scenarios.policy_dt_s

    def reset(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.sensor_rng = np.random.default_rng(seed + 1_000_000)
        self.elapsed = 0.0
        horizontal = self.dimension - 1
        self.pad_position = np.zeros(self.dimension)
        self.pad_velocity = np.zeros(self.dimension)
        self.pad_velocity[0] = self.rng.uniform(0.5, 2.5)
        self.own_position = np.r_[np.zeros(horizontal), self.rng.uniform(0.45, 1.0)]
        self.own_velocity = np.r_[self.pad_velocity[:-1], 0.0]
        self.attitude = np.zeros(horizontal)
        self.angular_rate = np.zeros(horizontal)
        self.previous_truth = self._truth()
        self.last_estimated_pad = self.pad_position.copy()
        self.estimated_velocity = self.pad_velocity.copy()
        observation = self._observation(updated=True)
        return observation, {"backend": self.name, "seed": seed,
                             "policy_version": 0, "decision_id": 0}

    def _truth(self):
        relative = self.pad_position-self.own_position
        # The policy/reward convention stores height positive and own vz last.
        position = np.r_[relative[:-1], -relative[-1]]
        velocity = np.r_[self.pad_velocity[:-1]-self.own_velocity[:-1],
                         self.own_velocity[-1]]
        return TruthState(tuple(position), tuple(velocity), tuple(self.attitude),
                          tuple(self.angular_rate))

    def _observation(self, *, updated):
        noisy = self.pad_position + self.sensor_rng.normal(0, 0.02, self.dimension)
        if updated:
            innovation = noisy-self.last_estimated_pad
            self.estimated_velocity = 0.8*self.estimated_velocity + 0.2*innovation/self.dt
            self.last_estimated_pad = noisy
        if self.dimension == 2:
            state = PlanarEstimate(
                pad_position_xz=tuple(self.last_estimated_pad),
                pad_velocity_xz=tuple(self.estimated_velocity),
                own_position_xz=tuple(self.own_position),
                own_velocity_xz=tuple(self.own_velocity),
                pitch_rad=float(self.attitude[0]),
                pitch_rate_rad_s=float(self.angular_rate[0]),
                vision_updated=updated, vision_age_s=0 if updated else self.dt,
                navigation_valid=True, navigation_age_s=0,
                # Local pad_position is the deck already; represent the source
                # UGV reference point below it for the planar vector formula.
                pad_offset_z_m=self.contract.scenarios.pad_height_m)
            pad_reference = np.asarray(state.pad_position_xz).copy()
            pad_reference[1] -= state.pad_offset_z_m
            state = PlanarEstimate(**{**state.__dict__,
                                     "pad_position_xz": tuple(pad_reference)})
            return planar_vector(state)
        roll_pitch_yaw = (float(self.attitude[0]), float(self.attitude[1]), 0.0)
        body_rates = (float(self.angular_rate[0]), float(self.angular_rate[1]), 0.0)
        return spatial_vector(SpatialEstimate(
            pad_position_enu=tuple(self.last_estimated_pad),
            pad_velocity_enu=tuple(self.estimated_velocity),
            own_position_enu=tuple(self.own_position),
            own_velocity_enu=tuple(self.own_velocity),
            roll_pitch_yaw_rad=roll_pitch_yaw, body_rate_rad_s=body_rates,
            vision_updated=updated, vision_age_s=0 if updated else self.dt,
            navigation_valid=True, navigation_age_s=0))

    def step(self, action) -> StepResult:
        requested = np.asarray(action, dtype=float)
        if requested.shape != (self.dimension,):
            raise ValueError(f"expected {self.dimension} acceleration values")
        applied = requested.copy()
        intervened = False
        if self.contract.safety.profile == "shielded":
            clipped = np.clip(applied, -np.asarray(self.contract.action.maximum),
                              np.asarray(self.contract.action.maximum))
            intervened = not np.array_equal(clipped, applied)
            applied = clipped
        previous = self._truth()
        # Smooth CV-CA-CV pad event; all arms see the same seed-derived event.
        acceleration = 0.0
        if 1.0 <= self.elapsed < 2.0:
            acceleration = 0.8
        elif 2.0 <= self.elapsed < 3.0:
            acceleration = -0.8
        self.pad_velocity[0] += acceleration*self.dt
        self.pad_position += self.pad_velocity*self.dt
        self.own_velocity += applied*self.dt
        self.own_position += self.own_velocity*self.dt
        desired_attitude = -applied[:-1] / 9.81
        old_attitude = self.attitude.copy()
        self.attitude += (desired_attitude-self.attitude)*min(1.0, 10*self.dt)
        self.angular_rate = (self.attitude-old_attitude)/self.dt
        self.elapsed += self.dt
        current = self._truth()
        contact = self.own_position[-1] <= self.pad_position[-1] \
            + self.contract.termination.touchdown_height_m
        event = classify_terminal(current, elapsed_s=self.elapsed, contact=contact,
                                  terminal=self.contract.termination)
        normalized = applied/np.asarray(self.contract.action.maximum)
        reward, components = reward_v5(
            previous, current, ViewMeasurement(True, (0.0,), (math.radians(60),)),
            normalized, self.dt, event, self.contract.reward,
            self.contract.termination)
        observation = self._observation(updated=True)
        return StepResult(observation, reward, event.occurred,
                          event.reason if event.occurred else "RUNNING",
                          {"backend": self.name, "requested_acceleration": requested.tolist(),
                           "applied_acceleration": applied.tolist(),
                           "intervened": intervened, "reward_components": components,
                           "elapsed_s": self.elapsed})
