"""Common reset/step environment for all two-axis policy representations."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np

from ..contracts.observation import CausalObservationPacket, ObservationRegistry
from .config import ExperimentConfig, load_config
from .contracts import load_v2_registry, load_reference_registry, make_causal_packet
from .curriculum import StageConfigs, stage_configs
from .dynamics import PlanarState, step_planar
from .estimation import CausalPadEstimator, TrackEstimate
from .ontology import ContextGraph, build_context_graph
from .reward import RewardBreakdown, compute_reward, landing_readiness, goal_cost
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
        self.registry: ObservationRegistry = (load_reference_registry()
            if self.config.ontology.schema == "compact_context_graph_v3_grouped" else load_v2_registry())
        self.perturbations = bool(perturbations)
        self.estimator = CausalPadEstimator(self.config.estimator)
        # Difficulty only reshapes the training episode distribution and the
        # training-only tolerance ramp; 1.0 reproduces the nominal contract.
        self.stage: StageConfigs = stage_configs(self.config, 1.0)
        self.supervisor = SafetySupervisor(
            self.config.dynamics, self.config.estimator, self.stage.safety,
            self.config.camera)
        self.scenario: PadScenario | None = None
        self.state: PlanarState | None = None
        self.measurement: PadMeasurement | None = None
        self.track: TrackEstimate | None = None
        self.safety: SafetyDecision | None = None
        self.observation: ObservationBundle | None = None
        self.dropout_schedule = DropoutSchedule()
        self._sensor_rng = np.random.default_rng(0)
        self._previous_action = np.zeros(2, dtype=float)
        self._previous_readiness = 0.0
        self._previous_goal_cost = 0.0
        self._terminated = False
        self._terminal_reason: TerminalReason | None = None

    def reset(self, *, seed: int, difficulty: float = 1.0
              ) -> tuple[ObservationBundle, dict[str, Any]]:
        streams = split_seed(seed)
        self.stage = stage_configs(self.config, difficulty)
        self.supervisor = SafetySupervisor(
            self.config.dynamics, self.config.estimator, self.stage.safety,
            self.config.camera)
        self.scenario = sample_scenario(seed, self.stage.scenario,
                                        self.config.timing)
        self._sensor_rng = np.random.default_rng(streams.sensor_seed)
        event_sampler = (DropoutSchedule.reference_mixture
            if self.config.camera.perturbation_profile == "reference_mixture_v28"
            else DropoutSchedule.randomized)
        self.dropout_schedule = event_sampler(
            self._sensor_rng, self.scenario.duration_s,
            enabled=self.perturbations)
        pad_x, pad_v, _ = self.scenario.state_at(0.0)
        # The start offset is capped by what the camera can see from the start
        # height. A fixed 2 m offset sits comfortably inside the frame at the
        # nominal 4-8 m, but at a 1.2 m curriculum start it is 59 degrees
        # against a 35 degree half-FOV, so the "easy" episode began blind and
        # was strictly harder than the nominal task. The nominal case is
        # unchanged: 0.5 * h is at least 2 m for every height at or above 4 m.
        start_offset_m = min(2.0, 0.5 * self.scenario.initial_height_m)
        if self.config.ontology.schema == "compact_context_graph_v3_grouped":
            # Upstream reset starts centered over the pad with matched speed.
            start_offset_m = 0.0
        self.state = PlanarState(
            x_m=pad_x - start_offset_m, z_m=self.scenario.initial_height_m,
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
            self.measurement, own_x_m=self.state.x_m, own_vx_m_s=self.state.vx_m_s)
        self.safety = self.supervisor.apply(
            np.zeros(2), self.state, self.track)
        self.observation = self._make_observation()
        self._previous_readiness = self._readiness()
        self._previous_goal_cost = goal_cost(pad_x-self.state.x_m, self.state.z_m, self.stage.reward)
        info = {
            "seed_streams": asdict(streams),
            "scenario": self.scenario.to_manifest(),
            "dropout_schedule": self.dropout_schedule.intervals_s,
            "pitch_event": self.dropout_schedule.pitch_event,
            "observation_registry_hash": self.registry.sha256,
            "graph_schema_hash": self.observation.graph.schema_hash,
            "curriculum_difficulty": self.stage.difficulty,
        }
        return self.observation, info

    def _readiness(self) -> float:
        """Causal-free training signal; isolated to the reward, never an input."""
        assert self.state is not None and self.scenario is not None
        pad_x, pad_v, _ = self.scenario.state_at(self.state.time_s)
        return landing_readiness(
            ex_true_m=pad_x - self.state.x_m,
            relative_speed_m_s=pad_v - self.state.vx_m_s,
            h_true_m=self.state.z_m, pitch_rad=self.state.theta_rad,
            vertical_speed_m_s=self.state.vz_m_s, safety=self.stage.safety,
            pitch_rate_rad_s=self.state.pitch_rate_rad_s,
            reward_config=self.stage.reward)

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
            prediction_horizon_s=self.config.estimator.prediction_horizon_s,
            reference_normalization=self.config.ontology.schema == "compact_context_graph_v3_grouped",
            mission_limit_s=self.config.timing.mission_duration_limit_s,
            prolonged_loss_s=self.config.estimator.prolonged_loss_s,
            fov_rad=self.config.camera.fov_rad, registry=self.registry)
        if self.config.ontology.schema == "compact_context_graph_v3_grouped":
            from .ontology_v28 import build_context_graph as grouped_graph
            graph = grouped_graph(packet, self.registry, self.config)
        else:
            graph = build_context_graph(packet, self.registry)
        return ObservationBundle(packet, graph)

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
        applied_integral = np.zeros(2)
        any_intervened = False
        all_reasons = set()
        target_time = min(start_time + self.config.timing.policy_dt_s,
                          self.scenario.duration_s)
        while self.state.time_s < target_time - 1e-12:
            dt = min(self.config.timing.physics_dt_s,
                     target_time - self.state.time_s)
            before = self.state
            if self.config.timing.sensor_dt_s < self.config.timing.policy_dt_s:
                applied_decision = self.supervisor.apply(request, before, self.track)
            any_intervened |= applied_decision.intervened
            all_reasons.update(applied_decision.reasons)
            pad_before = self.scenario.state_at(before.time_s)[:2]
            after = step_planar(before, applied_decision.applied_m_s2, dt,
                                self.config.dynamics, pitch_offset_rad=(
                                    self.dropout_schedule.pitch_rate_at(before.time_s)*dt))
            pad_after = self.scenario.state_at(after.time_s)[:2]
            contact = interpolate_contact(
                before, after, pad_before, pad_after,
                contact_height_m=self.stage.safety.touchdown_height_m)
            self.state = after
            physics_steps += 1
            actual_dt = dt if contact is None else contact.time_s-before.time_s
            applied_integral += applied_decision.applied_m_s2*actual_dt
            if contact is not None:
                fraction = (contact.time_s-before.time_s)/dt
                self.state = PlanarState(**{
                    field.name: getattr(before,field.name) + fraction*(
                        getattr(after,field.name)-getattr(before,field.name))
                    for field in fields(PlanarState)})
                # One rule, evaluated by the shared supervisor: track trust,
                # the touchdown corridor and the abort latch already decide
                # whether a descent was authorized.
                authorized = bool(not applied_decision.landing_inhibited
                                  and not applied_decision.abort_requested)
                terminal = classify_contact(
                    contact, authorized=authorized, config=self.stage.safety)
                break
            sensor_tick = self.state.time_s / self.config.timing.sensor_dt_s
            if np.isclose(sensor_tick, round(sensor_tick), atol=1e-8, rtol=0):
                self.measurement = self._observe()
                self.track = self.estimator.update(self.measurement, own_x_m=self.state.x_m,
                                                    own_vx_m_s=self.state.vx_m_s)
            if (hard_envelope_violation(after, self.stage.safety, pad_x_m=pad_after[0])
                or (not self.config.dynamics.clip_actual_pitch
                    and abs(after.theta_rad) > self.config.dynamics.pitch_limit_rad+np.deg2rad(1))):
                terminal = TerminalReason.SAFETY_ENVELOPE_VIOLATION
                break
            # A latched abort is a bounded recovery manoeuvre, not a terminal
            # state: the supervisor climbs and holds while tracking the last
            # causal estimate, and the episode only ends once the recovery
            # window has expired without reacquiring the pad.
            if applied_decision.abort_expired:
                terminal = TerminalReason.SAFE_ABORT
                break
        # Contact/hard-safety takes precedence at the exact mission deadline.
        if terminal is None and self.state.time_s >= self.scenario.duration_s - 1e-12:
            terminal = TerminalReason.TASK_TIMEOUT

        elapsed = self.state.time_s - start_time
        self._previous_action = action.copy()
        if terminal is None and self.measurement.timestamp_s < self.state.time_s - 1e-10:
            self.measurement = self._observe()
            self.track = self.estimator.update(self.measurement, own_x_m=self.state.x_m,
                                                own_vx_m_s=self.state.vx_m_s)
        self.safety = self.supervisor.apply(
            np.zeros(2), self.state, self.track)
        self.observation = self._make_observation()
        pad_x, _, _ = self.scenario.state_at(self.state.time_s)
        readiness = self._readiness()
        reward: RewardBreakdown = compute_reward(
            ex_true_m=pad_x - self.state.x_m, h_true_m=max(0.0, self.state.z_m),
            measured_bearing_rad=self.measurement.bearing_rad,
            bearing_valid=self.measurement.bearing_valid,
            normalized_policy_action=action, fov_rad=self.config.camera.fov_rad,
            dt_s=elapsed, terminal_reason=terminal, config=self.stage.reward,
            readiness=readiness, previous_readiness=self._previous_readiness,
            previous_goal_cost=self._previous_goal_cost,
            discount_time_constant_s=self.config.timing.discount_time_constant_s)
        self._previous_readiness = readiness
        self._previous_goal_cost = reward.goal_cost
        self._terminated = terminal is not None
        self._terminal_reason = terminal
        info: dict[str, Any] = {
            "status": terminal.value if terminal else "RUNNING",
            "requested_acceleration_m_s2": request.copy(),
            "applied_acceleration_m_s2": applied_integral/max(elapsed,1e-12),
            "normalized_policy_action": action.copy(),
            "safety_intervened": any_intervened,
            "safety_reasons": tuple(sorted(all_reasons)),
            "landing_inhibited": applied_decision.landing_inhibited,
            "abort_requested": applied_decision.abort_requested,
            "abort_expired": applied_decision.abort_expired,
            "abort_recovered": applied_decision.abort_recovered,
            "terminal_descent": applied_decision.terminal_descent,
            "touchdown_gate_width_m": applied_decision.touchdown_gate_width_m,
            "stopping_margin_m": applied_decision.stopping_margin_m,
            "landing_readiness": readiness,
            "reward_components": asdict(reward),
            "dt_s": elapsed, "physics_steps": physics_steps,
            "measurement_detected": self.measurement.detected,
            # Evaluation metadata only; it is absent from packet/graph inputs.
            "injected_sensor_cause": self.measurement.injected_cause,
            "geometric_visible": self.measurement.geometric_visible,
            "contact": asdict(contact) if contact else None,
            "curriculum_difficulty": self.stage.difficulty,
            "terminated": self._terminated, "truncated": False,
        }
        return self.observation, reward.total, self._terminated, False, info

    @property
    def terminal_reason(self) -> TerminalReason | None:
        return self._terminal_reason
