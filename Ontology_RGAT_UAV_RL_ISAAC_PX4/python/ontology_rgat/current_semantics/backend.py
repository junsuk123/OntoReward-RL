"""Isaac-linked backend boundary of spatial_current_semantics_v1 (guide 9.1-9.2).

Both registered flights -- the spatial-reference/12 ``LocalBackend``, the plant
calibrated to Isaac v12, and the actual ``IsaacBackend`` -- deliver the same
``spatial.core.Measurement`` stream, so one observer and one action adapter
serve both and a local result is never a different observation path from an
Isaac one.  Between the policy and the flight there is only the constant
adapter of ``contracts.ActionContract``: no supervisor, planar hold, prior or
integral, and every change it makes is recorded per step.  Truth reaches only
the registered task (``contracts.TaskContract``: the spatial-reference/12
``Evaluator``'s verdict and terminal table plus the common dense reward of
``reward.py``, scored on the actual transition time) and
``info["evaluator_only"]``; no policy code reads either.  The 21D vector is the
direct-policy encoder's, rescaled to the registered scales (``from_source``).  The local
plant carries three Isaac-measured calibrations (``contracts.LocalPlant``):
contact height, handover-after-first-frame and optical cadence.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from ..controllers.spatial_controller import SpatialAccelerationController
from ..direct_policy.isaac_adapter import enu_to_ned
from ..direct_policy.observation import SpatialEstimate, spatial_vector
from ..spatial.core import (REFERENCE_OPTICAL_MISS_PROBABILITY, Measurement, SpatialConfig,
                           camera_bearings)
from ..spatial.environment import Evaluator, LocalBackend
from ..spatial.estimation import ReferenceSpatialTrack
from .contracts import (ACTION_DIMENSION, TASK_EVALUATOR, TRACK_REACQUIRE_AFTER_REJECTIONS,
                        CurrentSemanticsConfig, OpticalCadence, TrainingRandomization,
                        VisionBlackout)
from .observation import from_source
from .reward import CommonRewardFunction


FLIGHT_SCHEMA = "spatial-reference/12"
ISAAC_SCENARIO = "spatial_reference_cv_ca_cv"

NAVIGATION_NOTE = (
    "drone_navigationValid is 1 and drone_navigationAge 0 by construction: Measurement carries "
    "no navigation validity (Measurement.from_wire raises on an invalid EKF) and own state is "
    "stamped at the decision sample. Open item; changing it needs a 21D schema revision for "
    "all four arms.")
PX4_SETPOINT_NOTE = ("enu_to_ned of the adapter-limited command; the gateway's actual "
                     "TrajectorySetpoint is not visible to this backend")
G3_STATUS = ("passed 2026-10-28: local half on 8 pilot seeds (coupling_local.json) and Isaac "
             "half on an owned stack, 6 episodes (coupling_isaac.json); calibration gaps stay "
             "registered in CALIBRATION_GAPS")
LATENT_TOLERANCE_M_S2 = 1e-5  # float32 runtime action vs float64 a_max*tanh(z)


# --------------------------------------------------------------- registries

@dataclass(frozen=True)
class CalibrationGap:
    """A known local-vs-Isaac difference: an open G3 item, never corrected silently.

    ``numbers`` carries the values reports compare against; local entries are
    read from the spatial-reference/12 config, Isaac entries are the audited
    measurements.
    """
    name: str
    local: str
    isaac: str
    evidence: str
    numbers: Any = MappingProxyType({})
    status: str = "open G3 item; the local plant is not adjusted to hide it"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "local": self.local, "isaac": self.isaac,
                "evidence": self.evidence, "status": self.status,
                "numbers": {key: list(value) if isinstance(value, tuple) else value
                            for key, value in self.numbers.items()}}


_AUDIT = "docs/current_semantics/SOURCE_AUDIT_KO.md"
_LOCAL = SpatialConfig(schema=FLIGHT_SCHEMA)
_CALIBRATED = "calibrated in the training plant (contracts.LocalPlant: {}); verify in Isaac G3"
CALIBRATION_GAPS = MappingProxyType({gap.name: gap for gap in (
    CalibrationGap(
        "optical_update_rate",
        local=("camera_dt captures (20 Hz) delivered after SpatialConfig.camera_transport_delay; "
               "a 0.1 s decision normally receives two frames"),
        isaac=("accepted captures 0.092/0.096 s apart (~10.6 Hz, 0.188 s across a miss); "
               "transport median 0.096-0.100 s at the decision (v5-v11 flight traces)"),
        evidence=f"{_AUDIT} section 2, optical update rate",
        numbers=MappingProxyType({
            "local_camera_dt_s": _LOCAL.camera_dt,
            "local_transport_s": _LOCAL.camera_transport_delay,
            "isaac_accepted_capture_spacing_s": (0.092, 0.096),
            "isaac_capture_spacing_across_miss_s": 0.188,
            "isaac_accepted_rate_hz": 10.6,
            "isaac_transport_median_at_decision_s": (0.096, 0.100),
            "isaac_miss_gap_fraction": 0.125}),
        status=("open G3 item; Isaac G3 (2026-10-28) measured ~1 frame per decision at the "
                "observer (mean 0.93, spacing 0.092-0.096 s) against local 2, while the 21D "
                "flag and age already match (updated in view 0.988 vs 0.977, age at update "
                "median 0.100 s both); thinning local frames was rejected because the "
                "reference track locked out on 11/16 local episodes, which Isaac does not show "
                "(1/161; cause not identified)")),
    CalibrationGap(
        "touchdown_body_height",
        local="contact declared at body height SpatialConfig.touchdown_height (no gear)",
        isaac=("legless first contact at body height 0.0636-0.0732 m over 78 contacts; camera "
               "0.087-0.096 m below the pad plane"),
        evidence=f"{_AUDIT} section 2, contact height",
        numbers=MappingProxyType({
            "local_contact_body_height_m": _LOCAL.touchdown_height,
            "isaac_first_contact_body_height_m": (0.0636, 0.0732),
            "isaac_contacts": 78,
            "isaac_camera_below_pad_m": (0.087, 0.096),
            "isaac_median_first_contact_body_height_m": 0.0672}),
        status=_CALIBRATED.format("contact_height_m = the Isaac median")),
    CalibrationGap(
        "entry_yaw",
        local="yaw 0 at reset (LocalBackend.angles start at zero)",
        isaac="deck heading + U(-5, 5) deg at handover",
        evidence=f"{_AUDIT} section 1, spatial/scenarios.py row",
        numbers=MappingProxyType({"local_entry_yaw_rad": 0.0,
                                  "isaac_entry_yaw_offset_deg": (-5.0, 5.0)})),
    CalibrationGap(
        "board_yaw",
        local="board axis-aligned with ENU; the pad translates along its heading without yawing",
        isaac="deck and board yaw with the pad heading, up to 0.2 rad",
        evidence=f"{_AUDIT} section 1, spatial/scenarios.py row",
        numbers=MappingProxyType({"local_board_yaw_rad": 0.0, "isaac_board_yaw_max_rad": 0.2})),
    CalibrationGap(
        "own_navigation",
        local="own position, velocity, attitude and rates are the plant state exactly",
        isaac=("PX4 EKF output; the capture-time pose comes from EKF history stamped at the "
               "gateway's receive sim-clock"),
        evidence=f"{_AUDIT} section 2, capture alignment chain"),
    CalibrationGap(
        "optical_position_noise",
        local=("pad solve = geometry + N(0, 0.02 m) isotropic per frame, independent of PnP "
               "geometry, tag count and depth; 5 % frame misses"),
        isaac="cv2 ArUco + PnP on rendered frames behind PnP-IMU, geometry and innovation gates",
        evidence=f"{_AUDIT} section 2; spatial/environment.py LocalBackend.measure",
        numbers=MappingProxyType({"local_noise_std_m": 0.02,
                                  "local_miss_probability": REFERENCE_OPTICAL_MISS_PROBABILITY})),
    CalibrationGap(
        "edge_visibility_rule",
        local=("a tag counts when |atan(X/Z)| + atan(padded half side / Z) < half-FOV on both "
               "axes (LocalBackend._optical_quality); angles do not add in the image plane, so "
               "near the frame edge this drops a tag earlier than its pinhole padded quad"),
        isaac="cv2 ArUco detection on rendered frames; its edge behaviour is not recorded",
        evidence="measured by coupling.check_fov_exit against the deployed board's pinhole rule"),
    CalibrationGap(
        "handover_detection",
        local=("LocalBackend.reset primes its first frame at the decision itself (age 0, no "
               "transport) and that frame takes the 5 % miss draw, so a reset can hand over "
               "with no pad estimate (G3 pilot seed 990007)"),
        isaac=("IsaacBackend._reset_once withholds handover until an optical pad measurement "
               "exists (EntryResetError after 15 s); that frame arrives with its transport delay"),
        evidence="spatial/environment.py LocalBackend.measure/reset, IsaacBackend._reset_once",
        status=_CALIBRATED.format("the local reset hovers until the first accepted frame, "
                                  "within handover_wait_max_s")),
)})


@dataclass(frozen=True)
class BackendSpec:
    name: str
    role: str
    status: str
    flight: str
    trains: bool
    requires: tuple[str, ...]
    reasons: tuple[str, ...]
    calibration_gaps: tuple[str, ...] = ()


BACKENDS = MappingProxyType({spec.name: spec for spec in (
    BackendSpec(
        name="spatial_local_v12", role="training plant", status="g3_passed",
        flight=("spatial.environment.LocalBackend(SpatialConfig(schema='spatial-reference/12', "
                "contact_height=local_plant.contact_height_m)) + LocalOpticalCadence"),
        trains=True,
        requires=("G3 passed (local coupling checks and the Isaac half): met 2026-10-28",),
        reasons=("calibrated to Isaac v12: the CV-CA-CV generator isaac_sim/pad_motion.py "
                 "imports, direct net acceleration, 0.10 s actuation delay, attitude servo "
                 "10 rad/s at zeta 0.7, seeded disturbances, 17-tag board optics, capture-aligned "
                 "frames with 0.075 s transport",
                 f"G3 {G3_STATUS}",
                 "Isaac-measured contact height, handover rule and optical cadence applied; "
                 "every other gap stays registered in CALIBRATION_GAPS"),
        calibration_gaps=tuple(CALIBRATION_GAPS)),
    BackendSpec(
        name="isaac_v12", role="evaluation and G3 backend", status="requires_allow_isaac",
        flight=(f"spatial.environment.IsaacBackend(SpatialConfig(schema='{FLIGHT_SCHEMA}'), "
                f"scenario='{ISAAC_SCENARIO}', deployment=deployment_profile('{FLIGHT_SCHEMA}'))"),
        trains=False,
        requires=("--allow-isaac", "a user-approved owned Isaac/PX4 stack"),
        reasons=("constructing it connects to a running stack",
                 "evaluation and G3 only; training runs on spatial_local_v12")),
    BackendSpec(
        name="direct_local_toy", role="none", status="refused",
        flight="direct_policy.backends.LocalReferenceBackend", trains=False, requires=(),
        reasons=("always visible: vision_updated 1 and vision age 0 at every step",
                 "x-only pad motion and disturbance",
                 "attitude axis-swapped and pitch-flipped vs Isaac (roll=-ax/g, pitch=-ay/g)",
                 "navigation validity and age constant")),
)})
FLIGHT_BACKENDS = ("spatial_local_v12", "isaac_v12")


def _refusal(spec: BackendSpec, why: str) -> RuntimeError:
    return RuntimeError(f"backend {spec.name!r} refused: {why}; " + "; ".join(spec.reasons))


def _spec(name: str) -> BackendSpec:
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; registered: {sorted(BACKENDS)}")
    return BACKENDS[name]


def require_training_backend(name: str, *, allow_pending_g3: bool = False) -> BackendSpec:
    """The only backend that may train is spatial_local_v12, and only after G3.

    ``allow_pending_g3`` admits it before G3 for smoke and coupling runs, whose
    outputs are never scientific.
    """
    spec = _spec(name)
    if spec.status == "refused":
        raise _refusal(spec, "smoke-only plant, never a training or evaluation backend")
    if not spec.trains:
        raise _refusal(spec, "not a training backend")
    if spec.status == "pending_g3" and not allow_pending_g3:
        raise _refusal(spec, f"G3 has not passed ({G3_STATUS})")
    return spec


# ------------------------------------------------------------ local plant

def local_plant_config(config: CurrentSemanticsConfig) -> SpatialConfig:
    """spatial-reference/12 with the Isaac-measured contact height; nothing else differs."""
    return SpatialConfig(schema=FLIGHT_SCHEMA,
                         contact_height=config.local_plant.contact_height_m)


class LocalOpticalCadence:
    """Thins the local plant's 20 Hz frames to Isaac's measured cadence.

    Keeps captures on every ``keep_every_nth_capture``-th camera tick and drops
    a further ``extra_drop_probability`` of those, so frames come ~0.1 s apart
    and 0.2 s across a miss (Isaac: 0.092/0.096 s, and 0.188 s on 12.5 % of
    intervals).  The drop is a pure function of (episode seed, capture tick),
    so paired runs of one seed thin identically and no plant stream moves.
    Only the optical fields of a withheld frame are hidden from the track; own
    state is untouched.  The first frame of an episode is never dropped:
    Isaac hands over only once a solve exists.
    """

    def __init__(self, cadence: OpticalCadence, camera_dt: float):
        self.keep_every = int(cadence.keep_every_nth_capture)
        self.drop = float(cadence.extra_drop_probability)
        self.camera_dt = float(camera_dt)
        self.reset(0)

    def reset(self, seed: int) -> None:
        self.seed = int(seed)
        self.decided: dict[int, bool] = {}

    def keeps(self, capture_s: float) -> bool:
        """One decision per capture tick, fixed for the episode (a frame is
        re-presented by every sample until the next one arrives)."""
        tick = int(round(capture_s / self.camera_dt))
        if tick not in self.decided:
            if tick % self.keep_every:
                keep = False
            elif not any(self.decided.values()):
                keep = True
            else:
                keep = bool(np.random.default_rng([self.seed, 0xCADE, tick]).uniform()
                            >= self.drop)
            self.decided[tick] = keep
        return self.decided[tick]

    def __call__(self, measurement: Measurement) -> Measurement:
        if measurement.optical_position is None or measurement.optical_time_s is None:
            return measurement
        if self.keeps(float(measurement.optical_time_s)):
            return measurement
        return replace(measurement, optical_position=None)


# ----------------------------------------------------------------- observer

class CurrentObserver:
    """Measurement stream -> the 21D current observation at each decision.

    The pad estimate is the capture-time-aligned ABG track of the spatial route
    (``ReferenceSpatialTrack(capture_aligned=True)``, level ground), fed every
    sample: its innovation is formed at the frame's capture time with the own
    pose paired to that capture, and repeated, out-of-order, future and >0.5 s
    old frames are not assimilated.  This replaces the direct-policy adapter,
    which updates its filter and vision age at the receive stamp.  Vision age
    is decision time minus the capture time of the last ACCEPTED frame; the
    updated flag says a frame was accepted since the previous decision.  Only
    ``Measurement`` is accepted, so simulator state cannot enter.  An optional
    ``cadence`` (local plant only) withholds frames from the track; the raw
    sample stays the decision sample.  The track re-acquires after
    ``TRACK_REACQUIRE_AFTER_REJECTIONS`` consecutive rejected fresh detections
    (contracts.TRACK_REACQUIRE_RULE); while it accepts frames nothing changes.
    """

    def __init__(self, cadence: LocalOpticalCadence | None = None) -> None:
        self.cadence = cadence
        self.reset(None)

    def reset(self, episode_id: Any, *, seed: int | None = None) -> None:
        self.episode_id = episode_id
        if self.cadence is not None:
            self.cadence.reset(0 if seed is None else seed)
        self.track = ReferenceSpatialTrack(capture_aligned=True)
        self.rejected: list[tuple[float, np.ndarray]] = []  # (capture, measured pad position)
        self.reacquisitions = 0
        self.last_measurement: Measurement | None = None
        self.frames_since_decision = 0
        self.frames_accepted = 0
        self.last_frame: tuple[int, float, float] | None = None  # sample, capture, receive
        # Latest capture the SENSOR delivered with a detection, before the cadence
        # filter: where detection physically stops, as opposed to what the track saw.
        self.last_sensor_capture: float | None = None

    def ingest(self, measurement: Measurement) -> bool:
        """Feed one sample; True when it delivered a frame the track accepted."""
        if not isinstance(measurement, Measurement):
            raise TypeError("CurrentObserver accepts only spatial.core.Measurement")
        if measurement.optical_position is not None and measurement.optical_time_s is not None:
            capture = float(measurement.optical_time_s)
            if self.last_sensor_capture is None or capture > self.last_sensor_capture:
                self.last_sensor_capture = capture
        seen = measurement if self.cadence is None else self.cadence(measurement)
        track = self.track
        detection = self._fresh_detection(seen) if track.initialized else None
        accepted = track.update(seen, level_ground=True)
        if accepted:
            self.rejected.clear()
        elif detection is not None and track.last_sample == seen.sample_id:
            self.rejected.append(detection)
            if len(self.rejected) >= TRACK_REACQUIRE_AFTER_REJECTIONS:
                accepted = self._reacquire(float(seen.time_s))
        if accepted:
            self.frames_since_decision += 1
            self.frames_accepted += 1
            self.last_frame = (int(measurement.sample_id), float(self.track.last_detection),
                               float(measurement.time_s))
        self.last_measurement = measurement
        return bool(accepted)

    def _fresh_detection(self, m: Measurement) -> tuple[float, np.ndarray] | None:
        """The detection the capture-aligned track would test, by its own freshness rule."""
        capture = m.optical_time_s
        if (m.optical_position is None or m.sample_id == self.track.last_sample
                or capture is None or m.optical_own_position is None
                or not capture > self.track.last_detection
                or not 0.0 <= float(m.time_s) - float(capture) <= 0.5):
            return None
        measured = (np.asarray(m.optical_own_position, dtype=np.float64)
                    - np.asarray(m.optical_position, dtype=np.float64))
        return float(capture), measured

    def _reacquire(self, now: float) -> bool:
        """Restart the track from the rejected detections (TRACK_REACQUIRE_RULE)."""
        times = np.array([t for t, _ in self.rejected])
        points = np.stack([p for _, p in self.rejected])
        if times[-1] - times[0] <= 1e-6:
            return False
        centre = times.mean()
        slope = ((times - centre)[:, None] * (points - points.mean(0))).sum(0) \
            / ((times - centre) ** 2).sum()
        slope[2] = 0.0
        fresh = ReferenceSpatialTrack(capture_aligned=True)
        track = self.track
        track.position = points.mean(0) + slope * (now - centre)
        track.velocity = slope
        track.acceleration = np.zeros(3)
        track.std = fresh.std.copy()
        track.last_detection = float(times[-1])
        self.rejected.clear()
        self.reacquisitions += 1
        return True

    def vector(self) -> tuple[np.ndarray, dict[str, Any]]:
        """(21D vector, provenance) at the latest sample; closes the decision window."""
        m = self.last_measurement
        if m is None:
            raise RuntimeError("a Measurement is required before the first decision")
        track = self.track
        initialized = bool(track.initialized)
        decision = float(m.time_s)
        age = decision - float(track.last_detection) if initialized else math.inf
        if age < 0.0:
            raise RuntimeError("accepted capture lies after the decision sample")
        angles = Rotation.from_quat(np.asarray(m.quaternion, dtype=np.float64)[[1, 2, 3, 0]]
                                    ).as_euler("xyz")
        zero = (0.0, 0.0, 0.0)
        estimate = SpatialEstimate(
            pad_position_enu=tuple(track.position.tolist()) if initialized else zero,
            pad_velocity_enu=tuple(track.velocity.tolist()) if initialized else zero,
            own_position_enu=tuple(np.asarray(m.own_position, dtype=np.float64).tolist()),
            own_velocity_enu=tuple(np.asarray(m.own_velocity, dtype=np.float64).tolist()),
            roll_pitch_yaw_rad=tuple(angles.tolist()),
            body_rate_rad_s=tuple(np.asarray(m.angular_rate, dtype=np.float64).tolist()),
            vision_updated=self.frames_since_decision > 0,
            vision_age_s=age,
            navigation_valid=True,
            navigation_age_s=0.0,
            estimate_initialized=initialized)
        vector = from_source(spatial_vector(estimate))
        sample, capture, receive = self.last_frame if self.last_frame else (None, None, None)
        provenance = {
            "episode_id": self.episode_id,
            "decision_stamp_s": decision,
            "sample_id": sample,
            "capture_stamp_s": capture,
            "receive_stamp_s": receive,
            "sensor_capture_stamp_s": self.last_sensor_capture,
            "frame_used": self.frames_since_decision > 0,
            "frames_since_decision": self.frames_since_decision,
            "frames_accepted": self.frames_accepted,
            "track_reacquisitions": self.reacquisitions,
            "estimate_initialized": initialized,
            "vision_age_s": age if initialized else None,
            "navigation": "valid 1, age 0 by construction (backend.NAVIGATION_NOTE)",
            # Evaluator-only: never part of the 21D vector.
            "track_std": track.std.tolist() if initialized else None,
        }
        self.frames_since_decision = 0
        return vector, provenance


# -------------------------------------------------------------- action trace

def policy_action(latent: Any, maximum: Any) -> np.ndarray:
    """``a_max * tanh(z)``, the only latent-to-action map of the path."""
    return np.asarray(maximum, dtype=np.float64) * np.tanh(np.asarray(latent, dtype=np.float64))


@dataclass(frozen=True)
class ActionTrace:
    """latent_z -> policy_accel -> adapter_limited_accel -> PX4 setpoint -> measured_accel.

    The last two are different quantities: the setpoint is what the adapter
    hands the flight, the measurement is the own-velocity change over the
    actual transition.
    """
    decision_id: int
    decision_stamp_s: float
    latent_z: tuple[float, ...] | None
    policy_accel: tuple[float, ...]
    adapter_limited_accel: tuple[float, ...]
    adapter_change: tuple[float, ...]
    adapter_constrained: bool
    px4_setpoint_ned: tuple[float, ...]
    px4_setpoint_note: str
    measured_accel: tuple[float, ...]
    measured_over_s: float

    def to_dict(self) -> dict[str, Any]:
        return {name: list(value) if isinstance(value, tuple) else value
                for name, value in self.__dict__.items()}


@dataclass(frozen=True, eq=False)
class CurrentStep:
    observation: np.ndarray
    info: dict[str, Any]
    contact: bool
    done: bool
    elapsed_s: float
    transition_s: float
    trace: ActionTrace
    reward: float
    status: str


@dataclass(frozen=True)
class _NoSupervisor:
    """The Evaluator's ``safety`` argument on a path without a supervisor."""
    abort: bool = False
    inhibited: bool = False
    reasons: tuple[str, ...] = ()


NO_SUPERVISOR = _NoSupervisor()


def _triple(values: Any) -> tuple[float, ...]:
    return tuple(float(v) for v in np.asarray(values, dtype=np.float64).reshape(-1))


def _evaluator_only(truth: Any) -> dict[str, Any]:
    """Truth for evaluation and reports; the spatial convention is own minus pad."""
    return {"relative_position_own_minus_pad_m": _triple(truth.relative_position),
            "relative_velocity_own_minus_pad_m_s": _triple(truth.relative_velocity),
            "roll_pitch_rad": _triple(truth.roll_pitch),
            "angular_rate_rad_s": _triple(truth.angular_rate),
            "contact": bool(truth.contact)}


# ------------------------------------------------------------------ backend

def config_task_label(config: CurrentSemanticsConfig) -> dict[str, str]:
    return {"version": config.task.version, "evaluator": config.task.evaluator,
            "supervisor": config.task.supervisor, "unsafe_penalty": config.task.unsafe_penalty}


def _flight_type(name: str) -> type:
    if name == "spatial_local_v12":
        return LocalBackend
    from ..spatial.environment import IsaacBackend
    return IsaacBackend


class CurrentSpatialBackend:
    """One spatial-reference/12 flight seen through the current observation.

    ``reset(seed)`` -> (obs21, info); ``step(action_enu)`` takes world ENU net
    acceleration in m/s^2 and returns a ``CurrentStep`` with the registered
    task's reward and status.  The episode ends when the evaluator returns a
    terminal (contact verdict, envelope or the 70 s horizon); its cleanup is
    recorded separately and never folded into the verdict.
    """

    def __init__(self, flight: Any, config: CurrentSemanticsConfig, *, name: str,
                 scientific: bool, randomization: TrainingRandomization | None = None,
                 dropout: VisionBlackout | None = None, split: str | None = None,
                 factors: Any = None):
        if dropout is not None and randomization is not None:
            raise ValueError("an evaluation dropout never rides on a training randomisation")
        if (split is not None or factors is not None) and name != "isaac_v12":
            raise ValueError("split/factors select the task-v5 draw of an Isaac flight; the "
                             "local plant carries its own (ScenarioLocalBackend)")
        if factors is not None and split is None:
            raise ValueError("plant factors need a task-v5 scenario split")
        if randomization is not None and name != "spatial_local_v12":
            raise ValueError("the training randomisation (handover own-state error, vision "
                             "blackout) is of the local plant only")
        if randomization is not None and not isinstance(randomization, TrainingRandomization):
            raise TypeError("randomization must be a contracts.TrainingRandomization")
        spec = _spec(name)
        if name not in FLIGHT_BACKENDS:
            raise _refusal(spec, "not a spatial-reference/12 flight backend")
        if not isinstance(flight, _flight_type(name)):
            raise TypeError(f"{name} needs a {_flight_type(name).__name__}, "
                            f"got {type(flight).__name__}")
        if not isinstance(scientific, bool):
            raise TypeError("scientific must be a bool")
        if scientific and spec.status != "g3_passed":
            raise _refusal(spec, f"scientific use needs status 'g3_passed', not {spec.status!r}")
        cfg = flight.cfg
        expected = (local_plant_config(config) if name == "spatial_local_v12"
                    else SpatialConfig(schema=FLIGHT_SCHEMA))
        if replace(cfg, isaac_profile_sha256="") != expected:
            raise ValueError(f"{name} must fly the registered {FLIGHT_SCHEMA} SpatialConfig "
                             f"({'calibrated local plant' if name == 'spatial_local_v12' else 'unmodified'})")
        if config.task.evaluator != TASK_EVALUATOR:
            raise ValueError(f"unregistered task evaluator {config.task.evaluator!r}")
        profile = getattr(flight, "profile_sha256", None)
        if profile is not None and profile != config.calibration.source.sha256:
            raise ValueError("flight deployment profile differs from the registered calibration")
        if name == "isaac_v12" and getattr(flight, "scenario", None) != ISAAC_SCENARIO:
            # A flight shared between task-v5 conditions carries the previous
            # episode's current-omni name; every reset of such a wrapper replaces it.
            from ..spatial.scenarios import parse_current_omni
            if split is None or parse_current_omni(getattr(flight, "scenario", "")) is None:
                raise ValueError(f"isaac_v12 must fly {ISAAC_SCENARIO} (or, with a split, "
                                 "the task-v5 draw)")
        if tuple(config.action.maximum_m_s2) != tuple(float(v) for v in cfg.max_acceleration):
            raise ValueError("action maxima differ from the flight's acceleration authority; "
                             "the adapter would silently rescale every action")
        self.randomization = randomization
        self.dropout = dropout
        if split is not None:
            from .scenario import SPLITS, PlantFactors
            if split not in SPLITS:
                raise ValueError(f"split must be one of {SPLITS}")
            factors = PlantFactors() if factors is None else factors
            if factors.optical_noise_scale != 1.0:
                raise ValueError("the optical-noise factor is applied by the local plant only")
        self.split, self.factors = split, factors
        self.isaac_draw: tuple[Any, dict[str, Any]] | None = None  # (EpisodeScenario, ack)
        self.error: tuple[np.ndarray, float, float] | None = None  # (e0, tau, t0)
        self.blackout: tuple[float, float] | None = None  # (start, duration) after handover
        self.start_s: float | None = None
        self.flight, self.config, self.spatial = flight, config, cfg
        self.name, self.spec, self.scientific = name, spec, scientific
        local = name == "spatial_local_v12"
        cadence = config.local_plant.optical_cadence
        thinning = cadence.keep_every_nth_capture > 1 or cadence.extra_drop_probability > 0
        self.observer = CurrentObserver(
            LocalOpticalCadence(cadence, cfg.camera_dt) if local and thinning else None)
        self.handover_wait_steps = (int(math.ceil(config.local_plant.handover_wait_max_s
                                                  / cfg.dt - 1e-9)) if local else 0)
        self.evaluator = Evaluator(cfg)
        self.reward = CommonRewardFunction(config.task.reward,
                                           config.local_plant.contact_height_m)
        self.controller = SpatialAccelerationController(
            max_velocity=cfg.max_velocity, max_acceleration=cfg.max_acceleration, dt=cfg.dt,
            acceleration_only=True, max_tilt_deg=config.action.adapter_max_tilt_deg)
        self.maximum = np.asarray(cfg.max_acceleration, dtype=np.float64)
        self.resets = 0
        self.active = False
        self.closed = False

    def _info(self, truth: Any, provenance: dict[str, Any]) -> dict[str, Any]:
        return {"backend": self.name, "flight": getattr(self.flight, "name", None),
                "episode_id": self.episode_id, "decision_id": self.decision_id,
                "scientific_use": self.scientific,
                "difficulty": getattr(self.flight, "difficulty", None),
                "profile_sha256": getattr(self.flight, "profile_sha256", None),
                "observation_provenance": provenance,
                "task": config_task_label(self.config),
                "evaluator_only": _evaluator_only(truth)}

    def reset(self, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
        if self.closed:
            raise RuntimeError("backend is closed")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError("seed must be an int")
        if self.active:
            self.flight.finish()
        drawn = None
        if self.split is not None:  # Isaac flies the learner's task-v5 draw (scenario.py)
            from .scenario import check_isaac_reset, isaac_scenario
            drawn, scenario_name = isaac_scenario(self.config.task.scenario, int(seed),
                                                  self.split, self.factors)
            self.flight.scenario = scenario_name
        measurement, truth = self.flight.reset(int(seed))
        if drawn is not None:
            ack = check_isaac_reset((self.flight.bridge.last_reset_ack or {}).get("detail"),
                                    int(seed), scenario_name)
            self.isaac_draw = (drawn, ack)
        self.resets += 1
        self.seed, self.episode_id = int(seed), f"{self.name}:{int(seed)}:{self.resets}"
        self.start_s = None  # no blackout before handover
        self.error = self._draw_error(float(measurement.time_s))
        self.blackout = self._draw_blackout()
        self.observer.reset(self.episode_id, seed=self.seed)
        self._ingest(measurement)
        measurement = self.observer.last_measurement
        self._reset_controller(measurement)
        # Isaac hands over only once an optical solve exists (IsaacBackend
        # withholds handover itself); the local plant hovers until its first
        # accepted frame instead of handing over blind.  Not policy time.
        waited = 0
        while not self.observer.track.initialized and waited < self.handover_wait_steps:
            hover = self.controller.command(np.zeros(ACTION_DIMENSION),
                                            own_velocity_enu_m_s=measurement.own_velocity)
            truth = self.flight.advance(hover, self._ingest)
            measurement = self.observer.last_measurement
            waited += 1
        if waited:
            self._reset_controller(measurement)
        self.handover_wait_decisions = waited
        if self.name == "spatial_local_v12" and not self.observer.track.initialized:
            self.flight.finish()
            raise RuntimeError(f"seed {self.seed}: no accepted optical frame within "
                               f"{self.config.local_plant.handover_wait_max_s} s of reset; "
                               "Isaac would refuse this handover")
        self.evaluator.reset(truth)
        self.start_s = self.decision_s = float(measurement.time_s)
        self.decision_id = 0
        self.cleanup_confirmed: bool | None = None
        self.active = True
        observation, provenance = self.observer.vector()
        self.reward.reset(truth, vision_age_s=provenance["vision_age_s"])
        info = self._info(truth, provenance)
        info["seed"] = self.seed
        info["handover_wait_decisions"] = waited
        drawn = getattr(self.flight, "episode_scenario", None)
        if drawn is not None:  # evaluator-only: the composition draw of this episode
            info["scenario"] = drawn.to_dict()
        elif self.split is not None:
            info["scenario"] = {**self.isaac_draw[0].to_dict(), **self.isaac_draw[1]}
        if self.dropout is not None:
            info["evaluation_dropout_s"] = None if self.blackout is None else list(self.blackout)
        if self.randomization is not None:  # evaluator-only record of the randomisation
            info["training_randomization"] = {
                "own_state_error_m_s": _triple(self.error[0]), "tau_s": self.error[1],
                "vision_blackout_s": None if self.blackout is None else list(self.blackout)}
        return observation, info

    def _draw_error(self, t0: float) -> tuple[np.ndarray, float, float] | None:
        """contracts.HandoverOwnStateError for this episode, from the episode seed."""
        if self.randomization is None:
            return None
        spec = self.randomization.handover_own_state_error
        rng = np.random.default_rng([self.seed, 0x0E5E])
        heading = rng.uniform(0.0, 2.0 * math.pi)
        speed = rng.uniform(0.0, spec.horizontal_max_m_s)
        e0 = np.array([speed * math.cos(heading), speed * math.sin(heading),
                       rng.uniform(-spec.vertical_max_m_s, spec.vertical_max_m_s)])
        return e0, float(rng.uniform(spec.tau_min_s, spec.tau_max_s)), t0

    def _draw_blackout(self) -> tuple[float, float] | None:
        """contracts.VisionBlackout for this episode (start after handover, duration),
        from the episode seed; None when this episode has none.  The training
        randomisation and an evaluation dropout use separate streams."""
        if self.randomization is not None:
            spec, stream = self.randomization.vision_blackout, 0xB1AC
        elif self.dropout is not None:
            spec, stream = self.dropout, 0xD0E7
        else:
            return None
        rng = np.random.default_rng([self.seed, stream])
        start = float(rng.uniform(spec.start_min_s, spec.start_max_s))
        duration = float(rng.uniform(spec.duration_min_s, spec.duration_max_s))
        return (start, duration) if rng.uniform() < spec.probability else None

    def _ingest(self, measurement: Measurement) -> bool:
        """The observer's input.  With a training randomisation the own velocity carries
        e0 exp(-t/tau) and the own position (and the pose paired with each image) its
        integral e0 tau (1 - exp(-t/tau)), t from reset, and inside the episode's
        blackout window no optical frame arrives.  Truth and the plant never change."""
        if self.blackout is not None and self.start_s is not None:
            start, duration = self.blackout
            if 0.0 <= float(measurement.time_s) - self.start_s - start < duration:
                measurement = replace(measurement, optical_position=None, confidence=0.0,
                                      optical_time_s=None, optical_own_position=None,
                                      optical_quaternion=None, marker_corners_px=None,
                                      marker_capture_time_s=None)
        if self.error is None:
            return self.observer.ingest(measurement)
        e0, tau, t0 = self.error
        drift = lambda t: e0 * tau * (1.0 - math.exp(-max(t - t0, 0.0) / tau))
        now = float(measurement.time_s)
        changes = {"own_velocity": np.asarray(measurement.own_velocity, dtype=np.float64)
                   + e0 * math.exp(-max(now - t0, 0.0) / tau),
                   "own_position": np.asarray(measurement.own_position, dtype=np.float64)
                   + drift(now)}
        if measurement.optical_own_position is not None and measurement.optical_time_s is not None:
            changes["optical_own_position"] = (
                np.asarray(measurement.optical_own_position, dtype=np.float64)
                + drift(float(measurement.optical_time_s)))
        return self.observer.ingest(replace(measurement, **changes))

    def _reset_controller(self, measurement: Measurement) -> None:
        yaw = Rotation.from_quat(np.asarray(measurement.quaternion)[[1, 2, 3, 0]]
                                 ).as_euler("xyz")[2]
        self.controller.reset(own_velocity_enu_m_s=measurement.own_velocity,
                              yaw_enu_rad=float(yaw))

    def step(self, action_enu: Any, *, latent: Any = None) -> CurrentStep:
        if not self.active:
            raise RuntimeError("reset required: no active episode")
        requested = np.asarray(action_enu, dtype=np.float64)
        if requested.shape != (ACTION_DIMENSION,) or not np.isfinite(requested).all():
            raise ValueError("action must be a finite world ENU acceleration (3,) in m/s^2")
        latent_z = None
        if latent is not None:
            latent_z = np.asarray(latent, dtype=np.float64)
            if latent_z.shape != (ACTION_DIMENSION,) or not np.isfinite(latent_z).all():
                raise ValueError("latent must be a finite (3,) vector")
            if not np.allclose(policy_action(latent_z, self.config.action.maximum), requested,
                               rtol=0.0, atol=LATENT_TOLERANCE_M_S2):
                raise ValueError("action is not a_max * tanh(latent): a hidden transform")
        measurement = self.observer.last_measurement
        command = self.controller.command(requested / self.maximum,
                                          own_velocity_enu_m_s=measurement.own_velocity)
        truth = self.flight.advance(command, self._ingest)
        new = self.observer.last_measurement
        transition = float(new.time_s) - self.decision_s
        if not transition > 0.0:
            raise RuntimeError("the flight clock did not advance over a decision")
        elapsed = float(new.time_s) - self.start_s
        limited = np.asarray(command.acceleration_enu_m_s2, dtype=np.float64)
        measured = (np.asarray(new.own_velocity, dtype=np.float64)
                    - np.asarray(measurement.own_velocity, dtype=np.float64)) / transition
        trace = ActionTrace(
            decision_id=self.decision_id, decision_stamp_s=self.decision_s,
            latent_z=None if latent_z is None else _triple(latent_z),
            policy_accel=_triple(requested), adapter_limited_accel=_triple(limited),
            adapter_change=_triple(limited - requested),
            adapter_constrained=bool(command.constrained),
            px4_setpoint_ned=_triple(enu_to_ned(limited)), px4_setpoint_note=PX4_SETPOINT_NOTE,
            measured_accel=_triple(measured), measured_over_s=transition)
        observation, provenance = self.observer.vector()
        self.decision_s = float(new.time_s)
        self.decision_id += 1
        contact = bool(truth.contact)
        # The registered task (guide 9.3, task v2): the spatial-reference/12
        # evaluator's verdict and terminal table, no supervisor; the trained
        # reward is the common reward (reward.py) on the policy's own normalized
        # action, and the evaluator's reward is kept only as reference_reward.
        visible = new.optical_position is not None
        bearings = (camera_bearings(new.optical_position, new.optical_quaternion,
                                    self.spatial)[0] if visible else np.zeros(2))
        reference, status, _ = self.evaluator.evaluate(
            truth, elapsed=elapsed, dt=transition, action=requested / self.maximum,
            safety=NO_SUPERVISOR, abort_elapsed=0.0, bearings=bearings, visible=visible)
        age = provenance["vision_age_s"]
        reward, components = self.reward.step(truth, status=status, dt=transition,
                                              action_normalized=requested / self.maximum,
                                              vision_age_s=age)
        done = status != "RUNNING"
        info = self._info(truth, provenance)
        info["action_trace"] = trace.to_dict()
        info["status"], info["reward"] = status, float(reward)
        info["reward_components"] = {**{key: float(value) for key, value in components.items()},
                                     "reference_reward": float(reference)}
        # How far the field asked for the search climb (diagnostic, reward /4).
        info["lost_view_weight"] = self.reward.lost_view(truth, age)
        info["termination"] = None
        if done:
            self.active = False
            self.cleanup_confirmed = bool(self.flight.finish())
            # A terminal and its cleanup are separate events; an unconfirmed
            # stop is recorded, never folded into the episode's outcome.
            reason = ("truth_contact" if contact else
                      "horizon" if status == "TASK_TIMEOUT" else "envelope")
            info["termination"] = {"reason": reason, "verdict": status,
                                   "cleanup_confirmed": self.cleanup_confirmed}
        return CurrentStep(observation=observation, info=info, contact=contact, done=done,
                           elapsed_s=elapsed, transition_s=transition, trace=trace,
                           reward=float(reward), status=status)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if self.active:
                self.active = False
                self.cleanup_confirmed = bool(self.flight.finish())
        finally:
            self.flight.close()


# ---------------------------------------------------------------- factories

def make_local_backend(config: CurrentSemanticsConfig, *, difficulty: float = 1.0,
                       randomization: TrainingRandomization | None = None,
                       scenario: str | None = None, factors: Any = None,
                       dropout: VisionBlackout | None = None) -> CurrentSpatialBackend:
    """The spatial-reference/12 local plant; ``difficulty`` < 1 and ``randomization``
    (contracts.TrainingRandomization) are training-only.  ``scenario`` selects the
    task-v5 composition split ("in_distribution" or "composition", scenario.py) that
    every learning and evaluation run uses; None keeps the /12 plant's own draw for
    the implementation checks (G3 probes, fixtures)."""
    if scenario is None:
        if factors is not None:
            raise ValueError("plant factors need a task-v5 scenario split")
        flight = LocalBackend(local_plant_config(config), difficulty=difficulty)
    else:
        from .scenario import PlantFactors, ScenarioLocalBackend
        flight = ScenarioLocalBackend(local_plant_config(config), difficulty=difficulty,
                                      scenario=config.task.scenario, split=scenario,
                                      factors=PlantFactors() if factors is None else factors)
    return CurrentSpatialBackend(flight, config, name="spatial_local_v12", scientific=False,
                                 randomization=randomization, dropout=dropout)


def make_isaac_backend(config: CurrentSemanticsConfig, *, pair: int = 0,
                       allow_isaac: bool = False, split: str | None = None,
                       factors: Any = None,
                       dropout: VisionBlackout | None = None) -> CurrentSpatialBackend:
    """Connect to a running, user-approved Isaac/PX4 stack.  Refused unless asked.
    ``split`` flies the task-v5 draw ("in_distribution" or "composition") instead of
    the /12 deck; ``factors`` (pad acceleration only) and ``dropout`` are evaluation
    conditions.  None keeps the /12 deck that G3 and isaac_eval/1-5 flew."""
    if allow_isaac is not True:
        raise _refusal(BACKENDS["isaac_v12"], "requires allow_isaac=True (--allow-isaac)")
    from ..spatial.environment import IsaacBackend
    from ..spatial.runtime_contract import deployment_profile
    deployment = deployment_profile(FLIGHT_SCHEMA)
    if deployment["sha256"] != config.calibration.source.sha256:
        raise ValueError("resolved deployment profile differs from the registered calibration")
    flight = IsaacBackend(SpatialConfig(schema=FLIGHT_SCHEMA,
                                        isaac_profile_sha256=deployment["sha256"]),
                          pair=pair, scenario=ISAAC_SCENARIO, deployment=deployment)
    try:
        return CurrentSpatialBackend(flight, config, name="isaac_v12", scientific=False,
                                     split=split, factors=factors, dropout=dropout)
    except BaseException:
        flight.close()
        raise
