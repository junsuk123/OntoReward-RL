"""Replay, local reference, and owned live Isaac/PX4 backends."""
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
from .isaac_adapter import (AuthorityRecord, CausalIsaacObservation,
                            CommandOwnership, direct_acceleration_payload,
                            embed_planar_action, enu_to_ned)


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
            self.transition_s = np.asarray(payload.get(
                "transition_s", [0.1]*len(self.observations)), dtype=float)
        if path.suffix == ".npz":
            self.transition_s = np.asarray(payload.get(
                "transition_s", np.full(len(self.observations), 0.1)), dtype=float)
        if len(self.observations) < 2:
            raise ValueError("replay fixture needs an initial observation and one transition")
        if not (len(self.rewards) == len(self.statuses) == len(self.transition_s)
                == len(self.observations)):
            raise ValueError("replay fields must have the same length")
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
                           "applied_acceleration": list(action), "intervened": False,
                           "transition_s": float(self.transition_s[self.index])})

    def close(self):
        pass


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
                             "policy_version": getattr(self, "policy_version", 0),
                             "decision_id": 0}

    def set_policy_version(self, version: int):
        self.policy_version = int(version)

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
                           "elapsed_s": self.elapsed, "transition_s": self.dt})

    def close(self):
        pass


class DirectIsaacBackend:
    """Direct-policy adapter over the repository's owned Isaac/PX4 backend.

    The policy owns x-z (planar) or x-y-z (spatial) acceleration.  In planar
    mode only, a separate own-navigation y hold keeps the vehicle in the
    profile's fixed y plane.  No landing supervisor or truth-fed correction is
    inserted into the action path.
    """
    name = "actual-isaac-px4-direct"

    def __init__(self, contract: ContractBundle, *, deployment, pair=0,
                 flight=None):
        if contract.safety.profile != "direct":
            raise ValueError("live MATLAB-port adapter currently supports direct profile only")
        from ..controllers.spatial_controller import SpatialAccelerationController
        from ..spatial.core import SpatialConfig
        from ..spatial.environment import IsaacBackend

        self.contract = contract
        self.dimension = int(contract.metadata["dimension"])
        self.dt = contract.scenarios.policy_dt_s
        self.deployment = deployment
        cfg = SpatialConfig(schema="spatial-reference/12",
                            isaac_profile_sha256=deployment["sha256"])
        scenario = ("matlab_planar_cv_ca_cv" if self.dimension == 2
                    else "spatial_reference_cv_ca_cv")
        self.flight = flight or IsaacBackend(
            cfg, pair=pair, scenario=scenario, deployment=deployment)
        self.controller = SpatialAccelerationController(
            max_velocity=cfg.max_velocity,
            max_acceleration=cfg.max_acceleration,
            dt=self.dt, acceleration_only=True, max_tilt_deg=25.)
        self.observer = CausalIsaacObservation(
            self.dimension, pad_offset_z_m=contract.scenarios.pad_height_m)
        self.ownership = CommandOwnership(contract.safety.command_owner)
        self.closed = False
        self.episode_active = False
        self.policy_version = 0

    def set_policy_version(self, version: int):
        self.policy_version = int(version)

    @staticmethod
    def _yaw(measurement):
        from scipy.spatial.transform import Rotation
        q = np.asarray(measurement.quaternion)
        return float(Rotation.from_quat(q[[1, 2, 3, 0]]).as_euler("xyz")[2])

    def _truth(self, truth):
        r, v = np.asarray(truth.relative_position), np.asarray(truth.relative_velocity)
        if self.dimension == 2:
            return TruthState((-float(r[0]), float(r[2])),
                              (-float(v[0]), float(v[2])),
                              (float(truth.roll_pitch[1]),),
                              (float(truth.angular_rate[1]),))
        return TruthState((-float(r[0]), -float(r[1]), float(r[2])),
                          (-float(v[0]), -float(v[1]), float(v[2])),
                          tuple(float(x) for x in truth.roll_pitch),
                          tuple(float(x) for x in truth.angular_rate[:2]))

    def reset(self, seed: int):
        measurement, truth = self.flight.reset(int(seed))
        self.episode_active = True
        self.episode_id = f"matlab-port-{self.dimension}d-{int(seed)}"
        self.ownership.reset(self.episode_id)
        self.observer.reset(self.episode_id)
        self.observer.ingest(measurement)
        self.controller.reset(own_velocity_enu_m_s=measurement.own_velocity,
                              yaw_enu_rad=self._yaw(measurement))
        self.initial_y = float(measurement.own_position[1])
        self.previous_truth = self._truth(truth)
        self.previous_velocity = np.asarray(measurement.own_velocity).copy()
        self.start_s = self.decision_s = float(measurement.time_s)
        self.decision_id = 0
        observation, provenance = self.observer.vector()
        self.decision_provenance = provenance
        return observation, {"backend": self.name, "seed": int(seed),
                             "episode_id": self.episode_id,
                             "observation_provenance": provenance}

    def _planar_hold(self, measurement):
        error = float(measurement.own_position[1])-self.initial_y
        return float(np.clip(-2.0*error-1.5*measurement.own_velocity[1], -2.5, 2.5))

    def step(self, action) -> StepResult:
        measurement = self.observer.last_measurement
        command_provenance = self.decision_provenance
        policy_action = np.asarray(action, dtype=float)
        expected = (self.dimension,)
        if policy_action.shape != expected or not np.isfinite(policy_action).all():
            raise ValueError(f"expected finite {self.dimension}D acceleration")
        requested = (embed_planar_action(policy_action) if self.dimension == 2
                     else policy_action.copy())
        if self.dimension == 2:
            requested[1] = self._planar_hold(measurement)
        maximum = np.array([2.5, 2.5, 2.0])
        normalized = requested/maximum
        command = self.controller.command(
            normalized, own_velocity_enu_m_s=measurement.own_velocity)
        payload = direct_acceleration_payload(
            requested, heading_rad=command.yaw_enu_rad,
            decision_id=self.decision_id, episode_id=self.episode_id,
            policy_version=self.policy_version, stamp_s=self.decision_s)
        self.ownership.validate(payload, writer=self.contract.safety.command_owner)
        truth = self.flight.advance(command, self.observer.ingest)
        observation, provenance = self.observer.vector()
        new_measurement = self.observer.last_measurement
        transition_s = float(new_measurement.time_s-self.decision_s)
        elapsed = float(new_measurement.time_s-self.start_s)
        current_truth = self._truth(truth)
        tilt = float(np.linalg.norm(truth.roll_pitch))
        event = classify_terminal(
            current_truth, elapsed_s=elapsed, contact=bool(truth.contact),
            hard_envelope_violation=tilt > math.radians(25.0),
            terminal=self.contract.termination)
        optical = new_measurement.optical_position
        bearings = ()
        if optical is not None:
            relative = -np.asarray(optical)
            depth = max(abs(float(relative[2])), 1e-6)
            axes = (relative[:1] if self.dimension == 2 else relative[:2])
            bearings = tuple(float(math.atan2(x, depth)) for x in axes)
        reward, components = reward_v5(
            self.previous_truth, current_truth,
            ViewMeasurement(optical is not None, bearings,
                            tuple([math.radians(90)]*max(1, len(bearings)))),
            policy_action/np.asarray(self.contract.action.maximum), self.dt,
            event, self.contract.reward, self.contract.termination)
        measured = ((np.asarray(new_measurement.own_velocity)-self.previous_velocity)
                    / max(float(new_measurement.time_s-self.decision_s), 1e-6))
        record = AuthorityRecord(
            episode_id=self.episode_id, decision_id=self.decision_id,
            policy_version=self.policy_version,
            capture_stamp_s=command_provenance["capture_stamp_s"],
            receive_stamp_s=command_provenance["receive_stamp_s"],
            decision_stamp_s=self.decision_s,
            command_stamp_s=self.decision_s,
            requested_acceleration_enu_m_s2=tuple(float(x) for x in requested),
            limited_acceleration_enu_m_s2=tuple(float(x) for x in command.acceleration_enu_m_s2),
            gateway_acceleration_ned_m_s2=tuple(float(x) for x in enu_to_ned(
                command.acceleration_enu_m_s2)),
            measured_acceleration_enu_m_s2=tuple(float(x) for x in measured),
            control_profile="direct", emergency_intervention=False)
        self.previous_truth = current_truth
        self.previous_velocity = np.asarray(new_measurement.own_velocity).copy()
        self.decision_s = float(new_measurement.time_s)
        self.decision_provenance = provenance
        self.decision_id += 1
        status = event.reason if event.occurred else "RUNNING"
        if event.occurred:
            self.flight.finish()
            self.episode_active = False
        return StepResult(
            observation, reward, event.occurred, status,
            {"backend": self.name,
             "policy_requested_acceleration": policy_action.tolist(),
             "requested_acceleration": requested.tolist(),
             "applied_acceleration": command.acceleration_enu_m_s2.tolist(),
             "intervened": bool(command.constrained),
             "planar_y_hold_active": self.dimension == 2,
             "yaw_hold_rad": command.yaw_enu_rad,
            "reward_components": components,
            "observation_provenance": provenance,
             "authority": record.to_dict(), "elapsed_s": elapsed,
             "transition_s": transition_s})

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                if self.episode_active:
                    self.flight.finish()
                    self.episode_active = False
            finally:
                self.flight.close()


def backend_factory(contract: ContractBundle, *, replay_path=None, deployment=None,
                    pair=0, flight=None):
    """Return a zero-argument factory for the backend named by the contract.

    Missing replay data or Isaac deployment state is an error.  In particular,
    an Isaac contract can never fall back to the local reference plant.
    """
    name = contract.backend.name
    if name == "local":
        return lambda: LocalReferenceBackend(contract)
    if name == "replay":
        if replay_path is None:
            raise ValueError("replay backend requires --replay-fixture")
        return lambda: ReferenceReplayBackend(replay_path)
    if name == "isaac":
        if deployment is None:
            raise ValueError("Isaac backend requires a resolved deployment manifest")
        return lambda: DirectIsaacBackend(
            contract, deployment=deployment, pair=pair, flight=flight)
    raise ValueError(f"unknown direct-policy backend: {name}")
