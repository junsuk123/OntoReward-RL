"""Common reset/step environment for all two-axis policy representations."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..contracts.observation import CausalObservationPacket, ObservationRegistry
from .config import ExperimentConfig, load_config
from .contracts import load_v2_registry, make_causal_packet
from .dynamics import PlanarState, step_planar
from .estimation import CausalPadEstimator, TrackEstimate
from .ontology import ContextGraph, build_context_graph
from .reward import RewardBreakdown, compute_reward
from .safety import (ContactState, SafetyDecision, SafetySupervisor,
                     TerminalReason, classify_contact, hard_envelope_violation,
                     interpolate_contact)
from .scenario import PadScenario, sample_scenario, split_seed
from .sensing import (DropoutSchedule, PadMeasurement, observe_pad)


@dataclass(frozen=True)
class ObservationBundle:
    packet: CausalObservationPacket
    graph: ContextGraph


class TwoAxisLandingEnv:
    """In-process research environment with no external-flight side effects."""

    def __init__(self, config: ExperimentConfig | None = None, *,
                 perturbations: bool = True):
        self.config = config or load_config()
        self.registry: ObservationRegistry = load_v2_registry()
        self.perturbations = bool(perturbations)
        self.estimator = CausalPadEstimator(self.config.estimator)
        self.supervisor = SafetySupervisor(
            self.config.dynamics, self.config.estimator, self.config.safety)
        self.scenario: PadScenario | None = None
        self.state: PlanarState | None = None
        self.measurement: PadMeasurement | None = None
        self.track: TrackEstimate | None = None
        self.safety: SafetyDecision | None = None
        self.observation: ObservationBundle | None = None
        self.dropout_schedule = DropoutSchedule()
        self._sensor_rng = np.random.default_rng(0)
        self._previous_action = np.zeros(2, dtype=float)
        self._terminated = False
        self._terminal_reason: TerminalReason | None = None

    def reset(self, *, seed: int) -> tuple[ObservationBundle, dict[str, Any]]:
        streams = split_seed(seed)
        self.scenario = sample_scenario(seed, self.config.scenario,
                                        self.config.timing)
        self._sensor_rng = np.random.default_rng(streams.sensor_seed)
        self.dropout_schedule = DropoutSchedule.randomized(
            self._sensor_rng, self.scenario.duration_s,
            enabled=self.perturbations)
        pad_x, pad_v, _ = self.scenario.state_at(0.0)
        self.state = PlanarState(
            x_m=pad_x - 2.0, z_m=self.scenario.initial_height_m,
            vx_m_s=pad_v, vz_m_s=0.0, theta_rad=0.0,
            pitch_rate_rad_s=0.0,
            thrust_n=self.config.dynamics.mass_kg
            * self.config.dynamics.gravity_m_s2, time_s=0.0)
        self.estimator.reset()
        self.supervisor.reset()
        self._previous_action = np.zeros(2, dtype=float)
        self._terminated = False
        self._terminal_reason = None
        self.measurement = self._observe()
        self.track = self.estimator.update(
            self.measurement, own_x_m=self.state.x_m)
        self.safety = self.supervisor.apply(
            np.zeros(2), self.state, self.track)
        self.observation = self._make_observation()
        info = {
            "seed_streams": asdict(streams),
            "scenario": self.scenario.to_manifest(),
            "dropout_schedule": self.dropout_schedule.intervals_s,
            "observation_registry_hash": self.registry.sha256,
            "graph_schema_hash": self.observation.graph.schema_hash,
        }
        return self.observation, info

    def _require_reset(self) -> None:
        if self.state is None or self.scenario is None or self.track is None:
            raise RuntimeError("reset must be called before step")

    def _observe(self) -> PadMeasurement:
        assert self.state is not None and self.scenario is not None
        pad_x, _, _ = self.scenario.state_at(self.state.time_s)
        return observe_pad(
            timestamp_s=self.state.time_s, ex_m=pad_x - self.state.x_m,
            h_m=self.state.z_m, theta_rad=self.state.theta_rad,
            config=self.config.camera, rng=self._sensor_rng,
            dropout_schedule=self.dropout_schedule)

    def _make_observation(self) -> ObservationBundle:
        assert self.state is not None and self.track is not None
        assert self.measurement is not None and self.safety is not None
        packet = make_causal_packet(
            state=self.state, track=self.track, measurement=self.measurement,
            previous_action=self._previous_action,
            landing_inhibited=self.safety.landing_inhibited,
            abort_requested=self.safety.abort_requested,
            mission_deadline_s=self.scenario.duration_s,
            fov_rad=self.config.camera.fov_rad, registry=self.registry)
        return ObservationBundle(packet, build_context_graph(packet, self.registry))

    def step(self, normalized_action: np.ndarray
             ) -> tuple[ObservationBundle, float, bool, bool, dict[str, Any]]:
        self._require_reset()
        if self._terminated:
            raise RuntimeError("step called after task termination")
        assert self.state is not None and self.scenario is not None
        assert self.track is not None and self.measurement is not None
        action = np.asarray(normalized_action, dtype=float).reshape(2)
        if not np.isfinite(action).all():
            raise ValueError("policy action must be finite")
        action = np.clip(action, -1.0, 1.0)
        request = np.array([
            action[0] * self.config.dynamics.ax_max_m_s2,
            action[1] * self.config.dynamics.az_max_m_s2], dtype=float)
        applied_decision = self.supervisor.apply(request, self.state, self.track)
        start_time = self.state.time_s
        terminal: TerminalReason | None = None
        contact: ContactState | None = None
        physics_steps = 0
        target_time = min(start_time + self.config.timing.policy_dt_s,
                          self.scenario.duration_s)
        while self.state.time_s < target_time - 1e-12:
            dt = min(self.config.timing.physics_dt_s,
                     target_time - self.state.time_s)
            before = self.state
            pad_before = self.scenario.state_at(before.time_s)[:2]
            after = step_planar(before, applied_decision.applied_m_s2, dt,
                                self.config.dynamics)
            pad_after = self.scenario.state_at(after.time_s)[:2]
            contact = interpolate_contact(before, after, pad_before, pad_after)
            self.state = after
            physics_steps += 1
            if contact is not None:
                authorized = bool(
                    not applied_decision.landing_inhibited
                    and not applied_decision.abort_requested
                    and self.track.initialized
                    and self.track.time_since_detection_s
                    <= self.config.estimator.recent_track_grace_s)
                terminal = classify_contact(
                    contact, authorized=authorized, config=self.config.safety)
                break
            if hard_envelope_violation(after, self.config.safety):
                terminal = TerminalReason.SAFETY_ENVELOPE_VIOLATION
                break
            if applied_decision.abort_requested:
                feasible_hold = (
                    after.z_m >= self.config.safety.minimum_abort_hold_height_m
                    and abs(after.vz_m_s) <= 0.1 and abs(after.vx_m_s) <= 0.2
                    and abs(after.theta_rad)
                    <= self.config.safety.touchdown_pitch_rad)
                if feasible_hold:
                    terminal = TerminalReason.SAFE_ABORT
                    break
        # Contact/hard-safety takes precedence at the exact mission deadline.
        if terminal is None and self.state.time_s >= self.scenario.duration_s - 1e-12:
            terminal = TerminalReason.TASK_TIMEOUT

        elapsed = self.state.time_s - start_time
        self._previous_action = action.copy()
        self.measurement = self._observe()
        self.track = self.estimator.update(
            self.measurement, own_x_m=self.state.x_m)
        self.safety = self.supervisor.apply(
            np.zeros(2), self.state, self.track)
        self.observation = self._make_observation()
        pad_x, _, _ = self.scenario.state_at(self.state.time_s)
        reward: RewardBreakdown = compute_reward(
            ex_true_m=pad_x - self.state.x_m, h_true_m=max(0.0, self.state.z_m),
            measured_bearing_rad=self.measurement.bearing_rad,
            bearing_valid=self.measurement.bearing_valid,
            normalized_policy_action=action, fov_rad=self.config.camera.fov_rad,
            dt_s=elapsed, terminal_reason=terminal, config=self.config.reward)
        self._terminated = terminal is not None
        self._terminal_reason = terminal
        info: dict[str, Any] = {
            "status": terminal.value if terminal else "RUNNING",
            "requested_acceleration_m_s2": request.copy(),
            "applied_acceleration_m_s2": applied_decision.applied_m_s2.copy(),
            "normalized_policy_action": action.copy(),
            "safety_intervened": applied_decision.intervened,
            "safety_reasons": applied_decision.reasons,
            "landing_inhibited": applied_decision.landing_inhibited,
            "abort_requested": applied_decision.abort_requested,
            "stopping_margin_m": applied_decision.stopping_margin_m,
            "reward_components": asdict(reward),
            "dt_s": elapsed, "physics_steps": physics_steps,
            "measurement_detected": self.measurement.detected,
            # Evaluation metadata only; it is absent from packet/graph inputs.
            "injected_sensor_cause": self.measurement.injected_cause,
            "geometric_visible": self.measurement.geometric_visible,
            "contact": asdict(contact) if contact else None,
            "terminated": self._terminated, "truncated": False,
        }
        return self.observation, reward.total, self._terminated, False, info

    @property
    def terminal_reason(self) -> TerminalReason | None:
        return self._terminal_reason
