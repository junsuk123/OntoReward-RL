"""Common causal task over local spatial dynamics or actual Isaac telemetry.

Truth has one consumer: ``Evaluator`` for reward/outcomes. Neither backend
passes it through the Measurement/Estimator/Observation policy boundary.
"""
from dataclasses import dataclass, replace
import math
import time
import numpy as np
from scipy.spatial.transform import Rotation

from .core import (
    SpatialConfig,
    Measurement,
    Estimator,
    safety_status,
    supervised_action,
    observation,
    camera_bearings,
)
from ..controllers.spatial_controller import SpatialAccelerationController
from ..landing.terminal import REFERENCE_TERMINAL_REWARDS, UNSAFE_REASONS


@dataclass(frozen=True)
class Truth:
    relative_position: np.ndarray
    relative_velocity: np.ndarray
    roll_pitch: np.ndarray
    angular_rate: np.ndarray
    contact: bool


class Evaluator:
    # One table, shared with the 2D route and with upstream MATLAB. A 3D
    # extension of a 2D task is only comparable to it if the objective is the
    # same, so these are NOT redefined here -- see landing/terminal.py for the
    # measurement that retired the separate spatial table.
    UNSAFE_STATUSES = frozenset(UNSAFE_REASONS)
    BONUSES = dict(REFERENCE_TERMINAL_REWARDS)
    NOMINAL_UNSAFE_PENALTY = max(
        REFERENCE_TERMINAL_REWARDS[name] for name in UNSAFE_REASONS)

    def __init__(self, cfg, *, unsafe_penalty=NOMINAL_UNSAFE_PENALTY):
        self.cfg = cfg
        self.unsafe_penalty = float(unsafe_penalty)
        if not self.unsafe_penalty < self.BONUSES["SAFE_ABORT"]:
            raise ValueError("unsafe penalty must remain worse than safe abort")
        self.previous_goal = None
        self.previous_readiness = None

    def terms(self, truth):
        cfg = self.cfg
        r, v = truth.relative_position, truth.relative_velocity
        xy = float(np.linalg.norm(r[:2]))
        sat = lambda value: value * value / (1 + value * value)
        goal = 0.65 * sat(xy / 3.0) + 0.35 * sat(max(r[2], 0) / 4.0)
        desired_xy = -r[:2] * min(0.6, cfg.touchdown_xy_speed / max(xy, 1e-9))
        desired_z = -min(0.8 * cfg.touchdown_z_speed, 0.5 * max(r[2], 0))
        if cfg.schema != "spatial-causal-rgat/3" and 0 < r[2] <= 0.45:
            # The offset nadir camera loses the pad before physical contact.
            # Preserve the existing sub-limit sink target in this near-field
            # interval instead of rewarding an asymptotic blind hover.
            # Derived from task limits, never from diagnostic actions.
            desired_z = -0.8 * cfg.touchdown_z_speed
        risk = (xy / cfg.pad_half_width) ** 2 + max(r[2], 0) ** 2
        risk += float(np.sum(((v[:2] - desired_xy) / cfg.touchdown_xy_speed) ** 2))
        risk += ((v[2] - desired_z) / cfg.touchdown_z_speed) ** 2
        risk += float(np.sum((truth.roll_pitch / cfg.touchdown_tilt) ** 2))
        risk += float(np.sum((truth.angular_rate[:2] / cfg.touchdown_rate) ** 2))
        return goal, math.exp(-0.5 * min(risk, 100))

    def reset(self, truth):
        self.previous_goal, self.previous_readiness = self.terms(truth)

    def evaluate(
        self, truth, *, elapsed, dt, action, safety, abort_elapsed, bearings, visible
    ):
        cfg = self.cfg
        r, v = truth.relative_position, truth.relative_velocity
        xy = float(np.linalg.norm(r[:2]))
        status = "RUNNING"
        # Contact/unsafe ordering always precedes benign timeouts/abort.
        if truth.contact:
            if xy > cfg.pad_half_width:
                status = "MISSED_PAD_CONTACT"
            elif (
                np.linalg.norm(v[:2]) > cfg.touchdown_xy_speed
                or abs(v[2]) > cfg.touchdown_z_speed
                or np.linalg.norm(truth.roll_pitch) > cfg.touchdown_tilt
                or np.linalg.norm(truth.angular_rate[:2]) > cfg.touchdown_rate
            ):
                status = "UNSAFE_CONTACT"
            elif safety.inhibited or safety.abort:
                status = "UNAUTHORIZED_CONTACT"
            else:
                status = "SUCCESS"
        elif xy > 60 or r[2] > 20 or r[2] < -0.2:
            status = "SAFETY_ENVELOPE_VIOLATION"
        elif cfg.reference_supervisor and math.acos(float(np.clip(
                math.cos(truth.roll_pitch[0])*math.cos(truth.roll_pitch[1]),-1.,1.))) > math.radians(21):
            status = "SAFETY_ENVELOPE_VIOLATION"
        elif abort_elapsed >= cfg.recovery_timeout:
            status = "SAFE_ABORT"
        elif elapsed >= cfg.horizon - 1e-8:
            status = "TASK_TIMEOUT"
        goal, readiness = self.terms(truth)
        view = (
            min(1.0, float(np.mean((bearings / (np.asarray(cfg.fov) / 2)) ** 2)))
            if visible
            else 1.0
        )
        control = float(np.mean(np.square(action)))
        running = -dt / 70.0 * (2.0 * goal + view + 0.25 * control) + 8.0 * (
            readiness - self.previous_readiness
        )
        phi_next = -2.0 * goal if status == "RUNNING" else 0.0
        potential = (
            math.exp(-dt / cfg.discount_tau) * phi_next + 2.0 * self.previous_goal
        )
        terminal = (
            self.unsafe_penalty
            if status in self.UNSAFE_STATUSES
            else self.BONUSES.get(status, 0.0)
        )
        self.previous_goal, self.previous_readiness = goal, readiness
        return (
            float(running + potential + terminal),
            status,
            {
                "running": running,
                "potential": potential,
                "terminal": terminal,
                "goal": goal,
                "readiness": readiness,
                "view": view,
                "control": control,
            },
        )


class LocalBackend:
    """Fast spatial sensor/plant fixture, never reported as Isaac flight."""

    name = "local-spatial"

    def __init__(self, cfg, *, difficulty=1.0):
        if not 0 <= difficulty <= 1:
            raise ValueError("difficulty must be in [0,1]")
        self.cfg = cfg
        self.difficulty = float(difficulty)

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.pad = np.zeros(3)
        self.position = np.r_[
            self.rng.uniform(-self.cfg.initial_xy, self.cfg.initial_xy, 2),
            self.rng.uniform(*self.cfg.initial_height),
        ]
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.angles = np.zeros(3)
        self.rates = np.zeros(3)
        self.euler_rates = np.zeros(3)
        self.thrust_acceleration = 9.80665
        self.domain_sample = None
        self.domain_initial_pending = False
        if self.cfg.matched_disturbances:
            from ..benchmarks.randomization import sample_domain_randomization
            self.domain_sample = sample_domain_randomization(seed)
            self.domain_initial_pending = True
        self.t1 = float(self.rng.uniform(0.5, 2.0))
        self.t2 = float(self.rng.uniform(1.0, 2.0))
        self.v0 = float(self.rng.uniform(0.3, 0.8))
        self.a2 = float(self.rng.uniform(0.1, 0.4))
        if self.difficulty < 1.0:
            d = self.difficulty
            c = self.cfg.curriculum

            def interpolate_sample(value, nominal, start):
                fraction = (value - nominal[0]) / (nominal[1] - nominal[0])
                easy = start[0] + fraction * (start[1] - start[0])
                return (1 - d) * easy + d * value

            self.position[:2] *= 0.15 + 0.85 * d
            self.position[2] = interpolate_sample(
                self.position[2], self.cfg.initial_height, c.start_height_range_m
            )
            if self.cfg.schema != "spatial-causal-rgat/3":
                # The mounted camera's depth, not body altitude, determines
                # the easy spawn footprint. No extra random draw is consumed.
                depth = max(self.position[2] - 0.16, 0.01)
                limit = 0.4 * depth * min(np.tan(np.asarray(self.cfg.fov) / 2))
                radius = np.linalg.norm(self.position[:2])
                self.position[:2] *= min(1.0, limit / max(radius, 1e-12))
            self.v0 = interpolate_sample(self.v0, (0.3, 0.8), c.start_v1_range_m_s)
            self.a2 = interpolate_sample(self.a2, (0.1, 0.4), c.start_a2_range_m_s2)
            self.t1 = interpolate_sample(self.t1, (0.5, 2.0), c.start_T1_range_s)
        heading = float(self.rng.uniform(-0.2, 0.2))
        self.direction = np.array([math.cos(heading), math.sin(heading), 0.0])
        self.pad_velocity = self.direction * self.v0
        self.sensor_id = 0
        self.camera_sequence = 0
        self.camera_pending = []
        self.camera_time = -math.inf
        self.optical_position = None
        self.optical_confidence = 0.0
        return self.measure(), self.truth()

    def measure(self):
        q = Rotation.from_euler("xyz", self.angles).as_quat()[[3, 0, 1, 2]]
        if self.t - self.camera_time >= self.cfg.camera_dt - 1e-9:
            r = self.position - self.pad
            bearing, depth = camera_bearings(r, q, self.cfg)
            noise = self.rng.normal(0, 0.02, 3)  # paired draw even on missed frames
            visible = bool(
                np.all(np.abs(bearing) < np.asarray(self.cfg.fov) / 2) and depth > 0
            )
            self.camera_time = self.t
            self.camera_sequence += 1
            frame = (self.camera_sequence,self.t,r+noise if visible else None,
                     .98 if visible else 0.,self.position.copy(),q.copy())
            if self.cfg.reference_context and self.t > 0:
                self.camera_pending.append(frame)
            else:
                # Reset primes an already available pre-policy optical track,
                # just as actual handover waits for the first valid image.
                self._receive_camera(frame)
        if self.cfg.reference_context:
            while self.camera_pending and self.camera_pending[0][1] <= self.t-self.cfg.camera_transport_delay+1e-9:
                self._receive_camera(self.camera_pending.pop(0))
        return Measurement(
            self.t,
            self.position.copy(),
            self.velocity.copy(),
            q,
            self.rates.copy(),
            self.optical_position,
            self.optical_confidence,
            self.sensor_id,
            self.optical_capture_time if self.cfg.reference_context else None,
            self.optical_own_position.copy() if self.cfg.reference_context else None,
            self.optical_quaternion.copy() if self.cfg.reference_context else None,
        )

    def _receive_camera(self, frame):
        (self.sensor_id,self.optical_capture_time,self.optical_position,
         self.optical_confidence,self.optical_own_position,self.optical_quaternion)=frame

    def pad_state(self, t):
        """Exact CV–CA–CV trajectory, continuous in both position and velocity."""
        middle = float(np.clip(t - self.t1, 0, self.t2))
        coast = max(0.0, t - self.t1 - self.t2)
        distance = self.v0 * t + 0.5 * self.a2 * middle ** 2 + self.a2 * self.t2 * coast
        return self.direction * distance, self.direction * (self.v0 + self.a2 * middle)

    def truth(self):
        r = self.position - self.pad
        return Truth(
            r.copy(),
            self.velocity - self.pad_velocity,
            self.angles[:2].copy(),
            self.rates.copy(),
            bool(r[2] <= self.cfg.contact_height),
        )

    def advance(self, command, callback):
        if self.domain_initial_pending:
            sample = self.domain_sample
            scale = self.difficulty
            self.velocity += scale*Rotation.from_euler("xyz", self.angles).apply(
                sample.initial_velocity_m_s)
            self.rates += scale*sample.initial_angular_rate_rad_s
            self.euler_rates += scale*sample.initial_angular_rate_rad_s
            self.domain_initial_pending = False
        end = self.t + self.cfg.dt
        while self.t < end - 1e-9:
            h = min(self.cfg.sensor_dt, end - self.t)
            previous_r = self.position - self.pad
            previous_v = self.velocity.copy()
            previous_position = self.position.copy()
            previous_angles = self.angles.copy()
            previous_rates = self.rates.copy()
            self.pad, self.pad_velocity = self.pad_state(self.t + h)
            if self.cfg.direct_acceleration:
                from .dynamics import advance_attitude_thrust
                disturbance = {}
                if self.domain_sample is not None:
                    sample = self.domain_sample
                    d = self.difficulty
                    gains = np.array([sample.attitude_gain_roll_pitch/1.725]*2 +
                                     [sample.attitude_gain_yaw/.325])
                    disturbance = dict(force_body=d*sample.external_force_n,
                        torque_body=d*sample.external_torque_nm,
                        attitude_gain_scale=1+d*(gains-1))
                (self.angles, self.euler_rates, self.thrust_acceleration,
                 self.acceleration, self.rates) = advance_attitude_thrust(
                    self.angles, self.euler_rates, self.thrust_acceleration, command, h,
                    **disturbance)
                self.position += self.velocity*h + 0.5*self.acceleration*h*h
                self.velocity += self.acceleration*h
            else:
                desired = command.acceleration_enu_m_s2 + 1.8 * (
                    command.velocity_enu_m_s - self.velocity
                )
                self.acceleration += (desired - self.acceleration) * min(1, h / 0.08)
                self.velocity += self.acceleration * h
                self.position += self.velocity * h
                target_angles = np.r_[command.derived_roll_pitch_rad, command.yaw_enu_rad]
                self.rates = (target_angles - self.angles) / 0.15
                self.angles += self.rates * h
            self.t += h
            # Interpolate first crossing to keep contact time and state coherent.
            r = self.position - self.pad
            if (
                r[2] <= self.cfg.contact_height
                and previous_r[2] > self.cfg.contact_height
            ):
                alpha = (previous_r[2] - self.cfg.contact_height) / (
                    previous_r[2] - r[2]
                )
                self.t -= h * (1 - alpha)
                self.pad, self.pad_velocity = self.pad_state(self.t)
                self.position = previous_position + alpha * (
                    self.position - previous_position
                )
                self.position[2] = self.pad[2] + self.cfg.contact_height
                self.velocity = previous_v + alpha * (self.velocity - previous_v)
                self.angles = previous_angles + alpha * (self.angles - previous_angles)
                self.rates = previous_rates + alpha * (self.rates - previous_rates)
                callback(self.measure())
                break
            callback(self.measure())
        return self.truth()

    def close(self):
        pass

    def finish(self):
        return True


class IsaacBackend:
    name = "isaac-px4-spatial"

    def __init__(self, cfg, *, pair=0, scenario=None):
        from ..config import default_config
        from ..bridge import PX4Bridge
        from .runtime_contract import deployment_profile, runtime_source_hash, wait_runtime_identity

        self.profile_sha256 = deployment_profile(cfg.schema)["sha256"]
        if cfg.isaac_profile_sha256 and cfg.isaac_profile_sha256 != self.profile_sha256:
            raise ValueError(
                "Isaac scientific profile differs from checkpoint deployment contract"
            )
        self.cfg = cfg
        self.pair = pair
        self.scenario = scenario or ('spatial_reference_cv_ca_cv' if cfg.reference_context
                                     else 'segmented_cruise_slow')
        bridge_cfg = default_config("quick")
        bridge_cfg.external.gateway_port = 14650 + 2 * pair
        bridge_cfg.external.local_port = 14651 + 2 * pair
        bridge_cfg.external.timeout = 8.0
        bridge_cfg.external.setup_timeout = 30.0
        bridge_cfg.external.entry_timeout = 180.0
        bridge_cfg.external.entry_sim_budget = 90.0
        bridge_cfg.external.entry_settle = 0.5
        bridge_cfg.external.entry_speed_tolerance = 0.6
        bridge_cfg.external.entry_require_offboard = True
        if cfg.direct_acceleration:
            # Legacy moving-entry tolerance (.9 m/.6 m/s) admitted a spatial
            # handover at 1.68 m for a seeded 2.04 m start, before the SAME
            # deliberate DR impulse. The pad is parked throughout entry here.
            # Require convergence before applying that impulse; never redraw
            # the seed or relax any flight/landing/preflight criterion.
            bridge_cfg.external.entry_tolerance = 0.2
            bridge_cfg.external.entry_speed_tolerance = 0.15
            bridge_cfg.external.entry_settle = 1.0
        self.entry_contract = {
            "position_tolerance_m": bridge_cfg.external.entry_tolerance,
            "speed_tolerance_m_s": bridge_cfg.external.entry_speed_tolerance,
            "settle_s": bridge_cfg.external.entry_settle,
            "seeded_disturbance_after_entry": True,
            "offboard_throughout_settle": True,
        }
        bridge_cfg.external.landing_camera = {
            "resolution": [640, 480],
            "horizontal_fov_deg": 90.0,
            "pitch_down_deg": 90.0,
            "mount_translation_flu_m": [0, 0, -0.16],
        }
        self.bridge = PX4Bridge(bridge_cfg)
        try:
            # Verify the configuration actually loaded by the running world,
            # not merely the YAML currently on disk, BEFORE reset/arming.
            # Runtime identity is a read-only preflight query, not an accepted
            # control sample. A degraded *previous* episode must not prevent
            # an explicitly scheduled cold restart in reset(). The reset and
            # every policy measurement still enforce estimator validity.
            self.identity_readiness = wait_runtime_identity(self.bridge,
                profile_sha256=self.profile_sha256, source_sha256=runtime_source_hash())
            from .lifecycle import record_flight_event
            record_flight_event(dict(phase='runtime_identity_ready',
                                     **self.identity_readiness))
        except BaseException:
            self.bridge.close()
            raise
        # Timing uses the explicit Isaac clock below, not PX4 DDS timestamps.
        self.bridge.control_period_us = 0
        self.last_state = None
        self.finished = True

    def reset(self, seed):
        from ..bridge import EntryResetError
        from .lifecycle import recover_refused_reset, prepare_isolated_episode
        if prepare_isolated_episode(seed=seed,release=self.bridge.close):
            self.__init__(self.cfg,pair=self.pair,scenario=self.scenario)
        while True:
            try:
                return self._reset_once(seed)
            except EntryResetError as exc:
                def release():
                    # The owned stack takes responsibility for stopping SITL.
                    # Do not issue cleanup commands through a closed socket.
                    self.finished = True
                    self.bridge.close()
                if not recover_refused_reset(exc, seed=seed, release=release):
                    raise
                self.__init__(self.cfg, pair=self.pair, scenario=self.scenario)

    def _reset_once(self, seed):
        self.finished = False
        self.seed = int(seed)
        self.start_s = math.inf
        self.last_state = self.bridge.reset(
            seed, scenario=self.scenario, initial_condition_scale=1.0
        )
        start = time.monotonic()
        while True:
            self.assert_policy_flight(self.last_state, pre_policy=True)
            m = Measurement.from_wire(self.last_state,capture_aligned=self.cfg.reference_context)
            if m.optical_position is not None:
                break
            if time.monotonic() - start > 15:
                from ..bridge import EntryResetError
                raise EntryResetError("no real optical pad measurement at handover; no policy transition collected")
            time.sleep(0.03)
            self.last_state = self.bridge.get_state()
        self.start_s = m.time_s
        self.t = m.time_s
        return m, self.truth()

    def assert_policy_flight(self, state, *, allow_contact=False, pre_policy=False):
        """Reject setup/failed-control states before they become RL samples.

        A fresh evaluator-only contact may force-disarm between physics and
        telemetry callbacks. That terminal snapshot remains evaluable; an old
        contact or an unrelated disarm must not authorize continued learning.
        """
        event = state.get("extra", {}).get("truth_contact_event")
        if (allow_contact and event is not None
                and event.get("seed") == self.seed
                and event.get("sample_time_s", -math.inf) >= self.start_s):
            return
        if state.get("armed") is not True or state.get("nav_state") != 14:
            if pre_policy:
                from ..bridge import EntryResetError
                from .lifecycle import record_flight_event
                record_flight_event(dict(phase='entry_refused_before_policy',
                    seed=getattr(self,'seed',None),transitions_collected=0,
                    armed=state.get('armed'),nav_state=state.get('nav_state')))
                error_type = EntryResetError
            else:
                error_type = RuntimeError
            raise error_type(
                "spatial policy requires armed PX4 in OFFBOARD; "
                f"armed={state.get('armed')}, nav_state={state.get('nav_state')}; "
                "setup/arming failure is not an RL episode")

    def truth(self):
        s = self.last_state
        event = s.get("extra", {}).get("truth_contact_event")
        if (
            event is not None
            and event.get("seed") == self.seed
            and event.get("sample_time_s", -math.inf) >= self.start_s
        ):
            return Truth(
                np.asarray(event["relative_position"]),
                np.asarray(event["relative_velocity"]),
                np.asarray(event["roll_pitch"]),
                np.asarray(event["angular_rate"]),
                True,
            )
        truth = s.get("truth", {})
        if not truth.get("valid") or not truth.get("attitude_valid"):
            raise RuntimeError(
                "isolated evaluator needs valid truth; never substitute measurement"
            )
        q = np.asarray(truth["quaternion_wxyz"])
        angles = Rotation.from_quat(q[[1, 2, 3, 0]]).as_euler("xyz")
        # Contact is evaluator-only. Policy and supervisor never receive it.
        contact = bool(s.get("extra", {}).get("pad_contact"))
        if contact:
            raise RuntimeError(
                "physical contact has no fresh pre-impact evaluator snapshot"
            )
        return Truth(
            np.asarray(truth["position"]),
            np.asarray(truth["velocity"]),
            angles[:2],
            np.asarray(s["angular_velocity"]),
            contact,
        )

    def advance(self, command, callback):
        self.assert_policy_flight(self.last_state)
        end = self.t + self.cfg.dt
        self.last_state = (self.bridge.step_spatial_acceleration(command)
                           if self.cfg.direct_acceleration
                           else self.bridge.step_spatial_velocity(command))
        started = time.monotonic()
        while True:
            self.assert_policy_flight(self.last_state, allow_contact=True)
            measurement = Measurement.from_wire(self.last_state,capture_aligned=self.cfg.reference_context)
            if measurement.time_s >= self.t:
                callback(measurement)
                self.t = measurement.time_s
            extra = self.last_state.get("extra", {})
            event = extra.get("truth_contact_event")
            first_contact = (
                event is not None
                and event.get("seed") == self.seed
                and event.get("sample_time_s", -math.inf) >= self.start_s
            )
            if first_contact or (self.t >= end - 1e-8 and not extra.get("pad_contact")):
                break
            if time.monotonic() - started > 8.0:
                raise RuntimeError("Isaac clock stalled during a spatial step")
            time.sleep(0.01)
            self.last_state = self.bridge.validate_state_with_estimator_grace(
                self.bridge.transact("state", {}, ("state",)), timeout=2.0)
            self.bridge.last_state = self.last_state
        return self.truth()

    def finish(self):
        from .lifecycle import record_confirmed_stop, record_flight_event
        self.finished = False
        started = time.monotonic()
        try:
            self.finished = self.bridge.stop_after_outcome(timeout=180.0)
            return self.finished
        finally:
            record_confirmed_stop(self.finished)
            record_flight_event(dict(phase="cleanup_finished", seed=getattr(self, "seed", None),
                stop_confirmed=self.finished, wall_seconds=time.monotonic()-started))

    def close(self):
        try:
            if not self.finished and not self.finish():
                raise RuntimeError("spatial flight stop was not confirmed")
        finally:
            self.bridge.close()


class SpatialLandingEnv:
    def __init__(self, config=None, backend=None, *, difficulty=1.0):
        self.cfg = config or SpatialConfig()
        if backend is not None and difficulty != 1.0:
            raise ValueError("Isaac validation cannot silently use a curriculum")
        self.backend = backend or LocalBackend(self.cfg, difficulty=difficulty)
        self.difficulty = float(difficulty)
        self.controller = SpatialAccelerationController(
            max_velocity=self.cfg.max_velocity,
            max_acceleration=self.cfg.max_acceleration,
            dt=self.cfg.dt,
            acceleration_only=self.cfg.direct_acceleration,
            max_tilt_deg=20. if self.cfg.reference_supervisor else 25.,
        )
        self.done = True
        self.episode_history = []

    def reset(self, *, seed, difficulty=None, angular_curriculum_scales=(1.0, 1.0),
              loss_curriculum_start=None):
        if loss_curriculum_start is not None:
            if (not isinstance(self.backend, LocalBackend)
                    or not np.isfinite(loss_curriculum_start)
                    or not self.cfg.loss_timeout <= loss_curriculum_start <= 30.):
                raise ValueError('loss-timeout curriculum is local training only, in [nominal,30]')
        scales = np.asarray(angular_curriculum_scales, dtype=float)
        if (
            scales.shape != (2,)
            or not np.isfinite(scales).all()
            or np.any(scales < 1)
            or np.any(scales > 4)
        ):
            raise ValueError("angular curriculum scales must be two values in [1,4]")
        if not isinstance(self.backend, LocalBackend) and np.any(scales != 1):
            raise ValueError("angular curriculum is restricted to local training")
        if difficulty is not None:
            if not isinstance(self.backend, LocalBackend):
                raise ValueError("curriculum is restricted to local training")
            if not 0 <= difficulty <= 1:
                raise ValueError("difficulty must be in [0,1]")
            self.difficulty = self.backend.difficulty = float(difficulty)
        d, c = self.difficulty, self.cfg.curriculum
        # At nominal difficulty preserve the exact object and all semantics.
        self.task_cfg = (
            self.cfg
            if d == 1
            else replace(
                self.cfg,
                touchdown_xy_speed=(1 - d) * c.start_touchdown_relative_speed_m_s
                + d * self.cfg.touchdown_xy_speed,
                touchdown_z_speed=(1 - d) * c.start_touchdown_vertical_speed_m_s
                + d * self.cfg.touchdown_z_speed,
                touchdown_tilt=((1 - d) * scales[0] + d) * self.cfg.touchdown_tilt,
                touchdown_rate=((1 - d) * scales[1] + d) * self.cfg.touchdown_rate,
                loss_timeout=(self.cfg.loss_timeout if loss_curriculum_start is None else
                              (1-d)*loss_curriculum_start+d*self.cfg.loss_timeout),
            )
        )
        self.episode_seed = int(seed)
        m, truth = self.backend.reset(seed)
        self.estimator = Estimator(self.cfg)
        self.estimator.update(m)
        self.start = m.time_s
        self.time = m.time_s
        self.abort_since = None
        self.evaluator = Evaluator(
            self.task_cfg,
            unsafe_penalty=(
                Evaluator.NOMINAL_UNSAFE_PENALTY if d == 1
                else (1 - d) * c.start_unsafe_contact_penalty
                + Evaluator.NOMINAL_UNSAFE_PENALTY * d
            ),
        )
        self.evaluator.reset(truth)
        yaw = Rotation.from_quat(m.quaternion[[1, 2, 3, 0]]).as_euler("xyz")[2]
        self.controller.reset(
            own_velocity_enu_m_s=m.own_velocity, yaw_enu_rad=float(yaw)
        )
        self.supervisor = None
        if self.cfg.reference_supervisor:
            from .safety import ReferenceSpatialSupervisor
            self.supervisor = ReferenceSpatialSupervisor()
        self.safety = (safety_status(self.estimator, self.task_cfg) if self.supervisor is None
                       else self.supervisor.status(self.estimator, self.task_cfg))
        self.done = False
        return observation(self.estimator, self.safety, 0.0, self.cfg), {
            "backend": self.backend.name
        }

    def step(self, action):
        if self.done:
            raise RuntimeError("reset required after terminal")
        est = self.estimator
        applied = (supervised_action(action, est, self.safety, self.task_cfg)
                   if self.supervisor is None else
                   self.supervisor.action(action, est, self.safety, self.task_cfg))
        command = self.controller.command(applied, own_velocity_enu_m_s=est.own.own_velocity)
        preceding = self.safety
        truth = self.backend.advance(command, est.update)
        dt = est.last_t - self.time
        if dt <= 0 or dt > 2:
            raise RuntimeError("invalid elapsed spatial control time")
        self.time = est.last_t
        elapsed = self.time - self.start
        self.safety = (safety_status(est, self.task_cfg) if self.supervisor is None
                       else self.supervisor.status(est, self.task_cfg))
        if self.safety.abort:
            if self.abort_since is None:
                self.abort_since = elapsed
        else:
            self.abort_since = None
        bearings, _ = camera_bearings(est.r, est.own.quaternion, self.cfg)
        view_available = est.detected
        if self.cfg.reference_supervisor:
            measurement = est.own
            view_available = measurement.optical_position is not None
            if view_available:
                bearings, _ = camera_bearings(measurement.optical_position,
                    measurement.optical_quaternion, self.cfg)
        reward, status, components = self.evaluator.evaluate(
            truth,
            elapsed=elapsed,
            dt=dt,
            action=action,
            safety=preceding,
            abort_elapsed=0 if self.abort_since is None else elapsed - self.abort_since,
            bearings=bearings,
            visible=view_available,
        )
        self.done = status != "RUNNING"
        if self.done and isinstance(self.backend, IsaacBackend):
            from .lifecycle import record_flight_event
            record_flight_event(dict(phase="terminal_before_cleanup", seed=self.episode_seed,
                status=status, elapsed_s=elapsed, stop_confirmed=False,
                truth_relative_position=truth.relative_position.tolist(),
                truth_relative_velocity=truth.relative_velocity.tolist(),
                requested_action=np.asarray(action).tolist(),
                applied_acceleration=command.acceleration_enu_m_s2.tolist()))
        if self.done and not self.backend.finish():
            raise RuntimeError(
                "spatial terminal was scored but PX4 stop is unconfirmed"
            )
        if self.done:
            self.episode_history.append(
                {
                    "seed": self.episode_seed,
                    "status": status,
                    "elapsed_s": elapsed,
                    "backend": self.backend.name,
                    "stop_confirmed": True,
                    "difficulty": self.difficulty,
                }
            )
        info = {
            "status": status,
            "dt_s": dt,
            "elapsed_s": elapsed,
            "requested_acceleration_m_s2": (
                np.asarray(action) * self.cfg.max_acceleration
            ).tolist(),
            "applied_acceleration_m_s2": command.acceleration_enu_m_s2.tolist(),
            "reward_components": components,
            "safety_intervened": bool(not np.allclose(action, applied)),
            "safety_reasons": preceding.reasons,
            "abort_requested": preceding.abort,
            "truth_relative_position": truth.relative_position.tolist(),
            "truth_relative_velocity": truth.relative_velocity.tolist(),
            "estimated_relative_position": est.r.tolist(),
            "estimated_relative_velocity": est.rv.tolist(),
            "estimated_pad_velocity": est.pad_v.tolist(),
            "optical_age_s": est.age,
            "optical_capture_time_s": est.own.optical_time_s,
            "optical_sample_id": est.own.sample_id,
            "optical_transport_age_s": (None if est.own.optical_time_s is None else
                                        est.own.time_s-est.own.optical_time_s),
            "optical_detected": est.detected,
            "truth_contact": truth.contact,
            "truth_roll_pitch": truth.roll_pitch.tolist(),
            "truth_angular_rate": truth.angular_rate.tolist(),
            "estimator_updates": est.updates,
            "backend": self.backend.name,
            "backend_profile_sha256": getattr(self.backend, "profile_sha256", None),
            "own_velocity_enu_m_s": est.own.own_velocity.tolist(),
            "own_quaternion_wxyz": est.own.quaternion.tolist(),
            "commanded_velocity_enu_m_s": command.velocity_enu_m_s.tolist(),
            "acceleration_only": command.acceleration_only,
            "optical_relative_position": (None if est.own.optical_position is None
                                           else est.own.optical_position.tolist()),
        }
        if isinstance(self.backend, IsaacBackend):
            # Audit-only, never part of actor/critic observations.
            info["flight_armed"] = self.backend.last_state["armed"]
            info["flight_nav_state"] = self.backend.last_state["nav_state"]
            info["backend_entry_contract"] = self.backend.entry_contract
        if self.cfg.reference_tracking:
            est.previous_action = np.clip(np.asarray(action, dtype=float), -1., 1.).copy()
        return (
            observation(est, self.safety, elapsed, self.cfg),
            reward,
            self.done,
            False,
            info,
        )

    def close(self):
        self.backend.close()
