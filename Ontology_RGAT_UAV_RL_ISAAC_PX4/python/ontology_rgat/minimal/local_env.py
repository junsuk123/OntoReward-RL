"""The minimal contract on the local spatial plant.

The PLANT, sensors, timing, disturbances and the verdict are exactly those of
the active spatial contract (``SpatialConfig()``: spatial-reference/12 --
seeded disturbances, 0.1 s actuation delay, 75 ms optical transport, optical
realism of the 17-tag board). What differs is everything after the sensor:

* the policy sees ``LandingObservation`` (13), built by the SAME
  ``ObservationAssembler`` the ROS node runs, fed from the local measurement
  stream at 100 Hz (own state) and 20 Hz (camera frames, missed ones too);
* the supervisor is ``MinimalSafetySupervisor`` (S1-S6), not the reference one;
* ``Evaluator`` is reused unchanged, with ``abort`` = ABORT_HOLD and
  ``inhibited`` = DESCENT_HOLD of the preceding decision.

Local own state is the plant's own position/velocity (no EKF noise), as in the
existing local route. Passing ``backend=IsaacBackend(cfg)`` flies the same
contract in Isaac/PX4: own state is then PX4 EKF2 (IMU + GNSS) through the
existing gateway, the optical solve is Isaac's capture-stamped ArUco PnP, and
entry, OFFBOARD handover and confirmed cleanup stay the gateway's (AGENTS.md
lifecycle rules); the supervised command reaches PX4 through that gateway. Simulator truth reaches the evaluator and the ``info``
dict only -- never the observation, the ontology or the supervisor.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from ..spatial.core import SpatialConfig, camera_bearings
from ..spatial.environment import Evaluator, LocalBackend
from ..controllers.spatial_controller import SpatialAccelerationController
from .constants import DEFAULT_CONSTANTS, LandingConstants
from .observation import ObservationAssembler, OwnState, PadDetectionSample, quat_wxyz_to_matrix
from .supervisor import MODE_ABORT_HOLD, MODE_DESCENT_HOLD, MinimalSafetySupervisor


@dataclass(frozen=True)
class _SafetyView:
    """What ``Evaluator`` reads of a supervisor."""
    abort: bool
    inhibited: bool


class MinimalLandingEnv:
    def __init__(self, config: SpatialConfig | None = None, *, difficulty: float = 1.0,
                 constants: LandingConstants = DEFAULT_CONSTANTS, backend=None):
        self.cfg = config or SpatialConfig()
        self.constants = constants
        if backend is not None and difficulty != 1.0:
            raise ValueError("Isaac flight cannot silently use a curriculum")
        self.backend = backend or LocalBackend(self.cfg, difficulty=difficulty)
        self.local = isinstance(self.backend, LocalBackend)
        self.difficulty = float(difficulty)
        self.controller = SpatialAccelerationController(
            max_velocity=self.cfg.max_velocity, max_acceleration=self.cfg.max_acceleration,
            dt=self.cfg.dt, acceleration_only=self.cfg.direct_acceleration,
            max_tilt_deg=20.0)
        self.done = True

    # ------------------------------------------------------------ measurement
    def _on_measurement(self, m) -> None:
        self.assembler.on_odometry(OwnState(
            float(m.time_s), np.asarray(m.own_position, float),
            np.asarray(m.own_velocity, float), np.asarray(m.quaternion, float), True))
        self.last_quaternion = np.asarray(m.quaternion, float)
        if m.sample_id == self.last_sample_id or m.sample_id < 0:
            return
        self.last_sample_id = m.sample_id
        detected = m.optical_position is not None and m.optical_time_s is not None
        position_body = np.zeros(3)
        if detected:
            # The local sensor solves body-minus-pad in ENU at capture; express
            # the pad in the body frame at capture so the assembler's rotation
            # (the one the ROS node runs) is exercised, not bypassed.
            rotation = quat_wxyz_to_matrix(m.optical_quaternion)
            position_body = rotation.T @ (-np.asarray(m.optical_position, float))
        # A frame without a solve still counts as a frame (a miss); its
        # capture time is the measurement's when the sensor gives none.
        capture = m.optical_time_s if m.optical_time_s is not None else m.time_s
        self.assembler.on_pad_detection(PadDetectionSample(
            float(capture), detected, position_body))

    # ------------------------------------------------------------------ API
    def reset(self, *, seed: int, difficulty: float | None = None):
        if difficulty is not None:
            if not self.local:
                raise ValueError("curriculum is restricted to local training")
            self.difficulty = self.backend.difficulty = float(difficulty)
        d, c = self.difficulty, self.cfg.curriculum
        self.task_cfg = self.cfg if d == 1 else replace(
            self.cfg,
            touchdown_xy_speed=(1 - d) * c.start_touchdown_relative_speed_m_s
            + d * self.cfg.touchdown_xy_speed,
            touchdown_z_speed=(1 - d) * c.start_touchdown_vertical_speed_m_s
            + d * self.cfg.touchdown_z_speed)
        self.seed = int(seed)
        self.assembler = ObservationAssembler(self.constants)
        self.supervisor = MinimalSafetySupervisor(self.constants)
        self.last_sample_id = None
        m, truth = self.backend.reset(seed)
        self._on_measurement(m)
        self.start = self.time = float(m.time_s)
        self.abort_since = None
        self.evaluator = Evaluator(
            self.task_cfg, unsafe_penalty=(
                Evaluator.NOMINAL_UNSAFE_PENALTY if d == 1 else
                (1 - d) * c.start_unsafe_contact_penalty + Evaluator.NOMINAL_UNSAFE_PENALTY * d))
        self.evaluator.reset(truth)
        yaw = math.atan2(2 * (m.quaternion[0] * m.quaternion[3] + m.quaternion[1] * m.quaternion[2]),
                         1 - 2 * (m.quaternion[2] ** 2 + m.quaternion[3] ** 2))
        self.controller.reset(own_velocity_enu_m_s=m.own_velocity, yaw_enu_rad=float(yaw))
        self.done = False
        self.observation = self.assembler.assemble(self.time)
        return self.observation, {"truth_relative_position": (-truth.relative_position).tolist()}

    def step(self, requested_acceleration_m_s2, *, external_decision=None):
        """Advance one decision.

        ``external_decision`` is a supervisor decision made OUTSIDE this
        process -- the ROS chain's ``safety_supervisor_node`` -- given as a
        dict with ``mode``, ``requested``, ``applied`` and ``intervened``. It
        is applied as is: supervising it a second time here would fly a
        different law than the one being tested. The in-process supervisor
        still observes, so its classification stays comparable.
        """
        if self.done:
            raise RuntimeError("reset required after terminal")
        obs = self.observation
        decision = self.supervisor.step(obs, requested_acceleration_m_s2)
        if external_decision is not None:
            from dataclasses import replace as _replace
            decision = _replace(
                decision, mode=int(external_decision["mode"]),
                requested=np.asarray(external_decision["requested"], float),
                applied=np.asarray(external_decision["applied"], float),
                intervened=tuple(bool(x) for x in external_decision["intervened"]))
        limits = np.asarray(self.cfg.max_acceleration, float)
        normalized = np.clip(decision.applied / limits, -1.0, 1.0)
        own_velocity = self.assembler.odometry.latest.velocity
        command = self.controller.command(normalized, own_velocity_enu_m_s=own_velocity)
        truth = self.backend.advance(command, self._on_measurement)
        now = self.assembler.odometry.latest.stamp
        dt = now - self.time
        self.time = now
        elapsed = now - self.start
        if decision.mode == MODE_ABORT_HOLD:
            self.abort_since = elapsed if self.abort_since is None else self.abort_since
        else:
            self.abort_since = None
        self.observation = self.assembler.assemble(now)
        visible = self.observation.pad_detected
        bearings, _ = camera_bearings(-self.observation.pad_relative_position,
                                      self.last_quaternion, self.cfg)
        reward, status, components = self.evaluator.evaluate(
            truth, elapsed=elapsed, dt=dt,
            action=np.asarray(requested_acceleration_m_s2, float) / limits,
            safety=_SafetyView(decision.mode == MODE_ABORT_HOLD,
                               decision.mode == MODE_DESCENT_HOLD),
            abort_elapsed=0.0 if self.abort_since is None else elapsed - self.abort_since,
            bearings=bearings, visible=visible)
        self.done = status != "RUNNING"
        if self.done and not self.backend.finish():
            raise RuntimeError("terminal was scored but the PX4 stop is unconfirmed")
        info = {
            "status": status, "elapsed_s": elapsed, "mode": decision.mode,
            "pad_class": decision.assessment.pad_class,
            "pad_reason": decision.assessment.reason,
            "terminal_score": decision.assessment.terminal_score,
            "intervened": decision.intervened,
            "requested": decision.requested.tolist(), "applied": decision.applied.tolist(),
            "reward_components": components,
            # truth: evaluator / audit only
            "truth_pad_minus_body": (-truth.relative_position).tolist(),
            "truth_relative_velocity": (-truth.relative_velocity).tolist(),
            "truth_contact": truth.contact,
            "backend": self.backend.name,
        }
        return self.observation, reward, self.done, info
