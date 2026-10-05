"""Causal measurement/estimation/packet/graph/supervision boundaries.

Only ``Measurement`` can enter the estimator. It has no truth, contact,
reward, outcome, or platform ground-truth fields. Simulator truth is isolated
in the environment evaluator. Spatial relative position means UAV minus pad.
"""
from dataclasses import dataclass, field, asdict
import hashlib
import json
import math
import numpy as np

from ..two_axis.config import GraphConfig, CurriculumConfig
from ..two_axis.ontology_v28 import GRAPH_SCHEMA_HASH as TOPOLOGY_HASH


FIELDS = tuple(
    "rx ry rz rvx rvy rvz pvx pvy pvz pax pay paz vx vy vz qw qx qy qz wx wy wz bx by mx my detected confidence age initialized posstd velstd remaining inhibited abort".split()
)
SCALES = np.array(
    [
        3,
        3,
        4,
        3,
        3,
        2,
        4,
        4,
        2,
        2,
        2,
        2,
        4,
        4,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0.785,
        0.644,
        0.785,
        0.644,
        1,
        1,
        3,
        1,
        1,
        1,
        1,
        1,
        1.0,
    ]
)
DIRECT = np.array(
    [
        FIELDS.index(k)
        for k in "qw qx qy qz detected confidence initialized remaining inhibited abort".split()
    ]
)
LEGACY_REGISTRY_HASH = hashlib.sha256(
    json.dumps(
        {
            "schema": "spatial-causal-packet/1",
            "fields": FIELDS,
            "scales": SCALES.tolist(),
            "normalization": "smooth-signed",
            "relative_convention": "uav-minus-pad-ENU",
            "graph": "spatial-9x12/1",
            "topology": TOPOLOGY_HASH,
        },
        sort_keys=True,
    ).encode()
).hexdigest()
REGISTRY_HASH = hashlib.sha256(
    json.dumps(
        {
            "previous_registry": LEGACY_REGISTRY_HASH,
            "schema": "spatial-causal-packet/2",
            "bearings": "actual-attitude optical frame, body mount [0,0,-0.16]",
        },
        sort_keys=True,
    ).encode()
).hexdigest()
FIELDS_V6 = FIELDS + ("pred_bx", "pred_by", "pred_mx", "pred_my")
REGISTRY_HASH_V6 = hashlib.sha256(json.dumps({
    "schema": "spatial-causal-packet/3", "previous_registry": REGISTRY_HASH,
    "fields": FIELDS_V6, "prediction": "0.5s relative translation and own body-rate attitude",
    "graph": "spatial-9x12/2-predicted-view", "topology": TOPOLOGY_HASH,
}, sort_keys=True).encode()).hexdigest()
FIELDS_V7 = FIELDS_V6 + ("accstd", "previous_ax", "previous_ay", "previous_az")
REGISTRY_HASH_V7 = hashlib.sha256(json.dumps({
    "schema": "spatial-causal-packet/4", "previous_registry": REGISTRY_HASH_V6,
    "fields": FIELDS_V7, "tracker": "absolute-pad-abg-reference-v28-20Hz-camera",
    "graph": "spatial-9x12/3-acceleration-uncertainty", "topology": TOPOLOGY_HASH,
}, sort_keys=True).encode()).hexdigest()
FIELDS_V9 = FIELDS_V7 + ('meas_bx','meas_by','optical_valid','optical_age')
REGISTRY_HASH_V9 = hashlib.sha256(json.dumps({
    'schema': 'spatial-causal-packet/5', 'previous_registry': REGISTRY_HASH_V7,
    'fields': FIELDS_V9, 'graph': 'two-ENU-planes-each-reference-9x12',
    'topology': TOPOLOGY_HASH, 'descent': 'joint-3D-reference-risk-product',
    'optical': 'unique-capture-time-with-own-state-alignment',
    'local_camera_transport_delay_s': .075,
    'trajectory': 'shared-seeded-spatial-reference-CV-CA-CV',
    'attitude_context': 'ENU-thrust-axis-tilt-and-derivative',
}, sort_keys=True).encode()).hexdigest()
REGISTRY_HASH_V10 = hashlib.sha256(json.dumps({
    'previous_registry': REGISTRY_HASH_V9,
    'supervisor': 'reference-trust-latch-delay-stopping-corridor/1',
    'persistent_disturbance_abort': 'own-EKF-position-anchor-full-xyz-override',
    'reward_view': 'captured-optical-bearing-not-estimated-bearing',
    'attitude_limit': '20deg-command-21deg-hard-envelope',
    'stopping_brake': 'net-az-limit-minus-sampled-force-norm-bound',
}, sort_keys=True).encode()).hexdigest()


def camera_bearings(relative_enu, quaternion_wxyz, cfg):
    """Causal optical bearings/depth; no simulator or target-truth access.

    Caller supplies either an estimate (policy/reward view term) or physical
    pose (sensor generation only). These uses never share an input object.
    """
    r = np.asarray(relative_enu)
    if cfg.schema == "spatial-causal-rgat/3":
        depth = float(r[2] - 0.16)
        return np.arctan2(r[:2], max(depth, 0.01)), depth
    from scipy.spatial.transform import Rotation

    q = np.asarray(quaternion_wxyz)
    body_to_pad = Rotation.from_quat(q[[1, 2, 3, 0]]).inv().apply(-r)
    optical = (body_to_pad - np.array([0, 0, -0.16])) * np.array([1, -1, -1])
    return np.arctan2(optical[:2], optical[2]), float(optical[2])


@dataclass(frozen=True)
class SpatialConfig:
    schema: str = "spatial-causal-rgat/5"
    isaac_profile_sha256: str = ""
    dt: float = 0.1
    sensor_dt: float = 0.01
    camera_dt: float = 0.05
    horizon: float = 70.0
    # Five times the horizon, not one. At tau == horizon a terminal at the
    # deadline is discounted to 0.368 of its value while an early one pays
    # 0.892, so running out the clock is the cheapest failure available and no
    # terminal table can fix it: the ordering demands |unsafe| > |timeout|, and
    # the discount then makes the early failure cost more than the late one.
    # Lengthening the horizon makes WHAT happened dominate WHEN.
    discount_tau: float = 350.0
    max_velocity: tuple = (10.0, 10.0, 5.0)
    max_acceleration: tuple = (2.5, 2.5, 2.0)
    pad_half_width: float = 0.5
    contact_height: float = 0.12
    touchdown_xy_speed: float = 0.35
    touchdown_z_speed: float = 0.3
    touchdown_tilt: float = math.radians(5)
    touchdown_rate: float = math.radians(10)
    fov: tuple = (math.radians(90), 2 * math.atan(0.75))
    loss_timeout: float = 3.0
    recovery_timeout: float = 8.0
    initial_height: tuple = (2.0, 2.5)
    initial_xy: float = 0.4
    level_ground_pad: bool = True
    curriculum: CurriculumConfig = field(
        default_factory=lambda: CurriculumConfig(
            start_v1_range_m_s=(0.045, 0.12),
            start_a2_range_m_s2=(0.03, 0.12),
            start_height_range_m=(0.2, 0.5),
            start_touchdown_vertical_speed_m_s=0.6,
            # Must stay strictly below SAFE_ABORT (-40) at EVERY rung, or the
            # easy end makes crashing cheaper than aborting and the arms learn
            # to dive -- two_axis measured 92-98 % unsafe contact doing exactly
            # that. The shared default of -20 inverted the ordering once
            # SAFE_ABORT moved from -15 to -40. At -42 the discounted
            # break-even runs 25.2 % -> 29.4 % -> 33.2 % across the ramp:
            # monotone, reachable, and the safety preference is kept.
            start_unsafe_contact_penalty=-42.0,
            promotion_landing_rate=0.1,
            promotion_window_episodes=18,
            scheduled_floor=True,
        )
    )
    ontology: GraphConfig = field(
        default_factory=lambda: GraphConfig(
            schema="compact_context_graph_v3_grouped",
            hidden_dimension=8,
            policy_hidden_dimension=48,
            relation_dimension=4,
            freeze_static_backbone=True,
            preserve_raw_during_adaptation=True,
            adaptation_warmup_fraction=0.9,
            pretrain_episodes=4,
            pretrain_decisions=32,
            pretrain_epochs=4,
            selection_score_model="outcome_weighted_v28",
            selection_score_margin=5.0,
            relation_activation_enabled=True,
        )
    )

    def __post_init__(self):
        if (
            self.schema not in ("spatial-causal-rgat/3", "spatial-causal-rgat/4",
                                "spatial-causal-rgat/5", "spatial-causal-rgat/6", "spatial-causal-rgat/7", "spatial-causal-rgat/8", "spatial-causal-rgat/9", "spatial-causal-rgat/10")
            or not 0 < self.sensor_dt <= self.dt <= 0.2
        ):
            raise ValueError("invalid spatial schema/timing")
        if self.horizon <= 0 or self.discount_tau <= 0:
            raise ValueError("positive spatial horizons required")
        if not self.sensor_dt <= self.camera_dt <= 0.2:
            raise ValueError("camera sampling must match the spatial sensor profile")

    @property
    def signature(self):
        return {
            "schema": self.schema,
            "packet_registry": self.registry_hash,
            "action": ("spatial-enu-direct-net-acceleration-v2" if self.direct_acceleration
                       else "spatial-enu-net-acceleration-v1"),
            "config_sha256": hashlib.sha256(
                json.dumps(asdict(self), sort_keys=True).encode()
            ).hexdigest(),
        }

    @property
    def registry_hash(self):
        if self.reference_supervisor:
            return REGISTRY_HASH_V10
        if self.reference_context:
            return REGISTRY_HASH_V9
        if self.reference_tracking:
            return REGISTRY_HASH_V7
        if self.direct_acceleration:
            return REGISTRY_HASH_V6
        return (
            LEGACY_REGISTRY_HASH
            if self.schema == "spatial-causal-rgat/3"
            else REGISTRY_HASH
        )

    @property
    def direct_acceleration(self):
        return self.schema in ("spatial-causal-rgat/6", "spatial-causal-rgat/7", "spatial-causal-rgat/8", "spatial-causal-rgat/9", "spatial-causal-rgat/10")

    @property
    def reference_tracking(self):
        return self.schema in ("spatial-causal-rgat/7", "spatial-causal-rgat/8", "spatial-causal-rgat/9", "spatial-causal-rgat/10")

    @property
    def matched_disturbances(self):
        return self.schema in ("spatial-causal-rgat/8", "spatial-causal-rgat/9", "spatial-causal-rgat/10")

    @property
    def reference_context(self):
        return self.schema in ("spatial-causal-rgat/9", "spatial-causal-rgat/10")

    @property
    def reference_supervisor(self):
        return self.schema == "spatial-causal-rgat/10"

    @property
    def camera_transport_delay(self):
        # v9 capture diagnostic: detected-frame median age 96 ms at decisions.
        # 75 ms delivery delay + 20 Hz sampling approximates that cadence;
        # this is not a claim to reproduce PnP image/noise/dropout distributions.
        return .075 if self.reference_context else 0.

    @property
    def packet_fields(self):
        if self.reference_context:
            return FIELDS_V9
        if self.reference_tracking:
            return FIELDS_V7
        return FIELDS_V6 if self.direct_acceleration else FIELDS


@dataclass(frozen=True)
class Measurement:
    time_s: float
    own_position: np.ndarray
    own_velocity: np.ndarray
    quaternion: np.ndarray
    angular_rate: np.ndarray
    optical_position: np.ndarray | None
    confidence: float
    sample_id: int
    optical_time_s: float | None = None
    optical_own_position: np.ndarray | None = None
    optical_quaternion: np.ndarray | None = None

    def __post_init__(self):
        for value, size in (
            (self.own_position, 3),
            (self.own_velocity, 3),
            (self.quaternion, 4),
            (self.angular_rate, 3),
        ):
            if np.shape(value) != (size,) or not np.isfinite(value).all():
                raise ValueError("nonfinite spatial own measurement")
        if not math.isfinite(self.time_s) or not 0 <= self.confidence <= 1:
            raise ValueError("invalid measurement time/confidence")
        if self.optical_position is not None and (
            np.shape(self.optical_position) != (3,)
            or not np.isfinite(self.optical_position).all()
        ):
            raise ValueError("invalid optical solve")
        if self.optical_time_s is not None and (
            not math.isfinite(self.optical_time_s) or self.optical_time_s > self.time_s + .02
        ):
            raise ValueError('invalid/future optical capture time')
        if self.optical_own_position is not None and (
            np.shape(self.optical_own_position) != (3,) or not np.isfinite(self.optical_own_position).all()
        ):
            raise ValueError('invalid capture-aligned own position')
        if self.optical_quaternion is not None and (
            np.shape(self.optical_quaternion) != (4,) or not np.isfinite(self.optical_quaternion).all()
            or np.linalg.norm(self.optical_quaternion) < 1e-9
        ):
            raise ValueError('invalid capture-aligned own attitude')

    @classmethod
    def from_wire(cls, state, *, capture_aligned=False):
        # Deliberate allowlist: never pass a whole wire state to the estimator.
        extra = state.get("extra", {})
        clock, optical = extra.get("spatial_clock", {}), extra.get(
            "optical_measurement", {}
        )
        if not clock.get("valid") or not state.get("estimator_valid"):
            raise ValueError("valid Isaac clock and own EKF are required")
        valid = optical.get("valid") and optical.get("frame") == "pad_enu"
        if valid and capture_aligned and any(optical.get(name) is None for name in (
            'capture_time_s','own_position_at_capture_m','own_quaternion_at_capture_wxyz'
        )):
            raise ValueError('v9 requires capture-aligned optical telemetry')
        capture = optical.get('capture_time_s')
        if capture is not None and capture > float(clock['sim_time_s']):
            # Independent DDS topics may deliver the frame before its clock.
            # Defer it until that time, never treat it as a future observation.
            valid, capture = False, None
        return cls(
            float(clock["sim_time_s"]),
            np.asarray(state["world"]["position"]),
            np.asarray(state["world"]["velocity"]),
            np.asarray(state["quaternion_wxyz"]),
            np.asarray(state["angular_velocity"]),
            np.asarray(optical["position_m"]) if valid else None,
            float(optical.get("confidence", 0)) if valid else 0.0,
            int(optical.get("sample_id", -1)),
            capture,
            (np.asarray(optical['own_position_at_capture_m'])
             if optical.get('own_position_at_capture_m') is not None else None),
            (np.asarray(optical['own_quaternion_at_capture_wxyz'])
             if optical.get('own_quaternion_at_capture_wxyz') is not None else None),
        )


class Estimator:
    def __init__(self, config):
        self.cfg = config
        self.initialized = False
        self.r = np.zeros(3)
        self.pad_v = np.zeros(3)
        self.pad_a = np.zeros(3)
        self.rv = np.zeros(3)
        self.last_t = None
        self.last_detection = -math.inf
        self.last_sample = -1
        self.std = 10.0
        self.velocity_std = 5.0
        self.updates = 0
        self.own = None
        self.detected = False
        self.last_optical_r = None
        self.last_optical_rv = None
        self.previous_action = np.zeros(3)
        if config.reference_tracking:
            from .estimation import ReferenceSpatialTrack
            self.reference_track = ReferenceSpatialTrack(capture_aligned=config.reference_context)
            self.acceleration_std = 2.

    def update(self, measurement):
        if not isinstance(measurement, Measurement):
            raise TypeError("only allowlisted Measurement may enter estimator")
        if self.cfg.reference_tracking:
            return self._update_reference(measurement)
        m = measurement
        if self.last_t is not None and m.time_s < self.last_t - 1e-8:
            raise ValueError("simulation clock reversed; reset the episode")
        dt = 0.0 if self.last_t is None else m.time_s - self.last_t
        if dt > 2.0:
            raise ValueError("sensor gap exceeds 2 simulated seconds")
        # Propagate at 100 Hz using ONLY the preceding own measurement.
        # Never interpolate a future measurement backwards into history.
        remaining = dt
        previous_v = m.own_velocity if self.own is None else self.own.own_velocity
        while remaining > 1e-9:
            h = min(remaining, self.cfg.sensor_dt)
            self.r += (previous_v - self.pad_v) * h - 0.5 * self.pad_a * h * h
            self.pad_v += self.pad_a * h
            if self.cfg.schema in ("spatial-causal-rgat/5", "spatial-causal-rgat/6"):
                # Reference updatePadTrack.m's 1.5 s acceleration decay. An
                # old noisy acceleration cannot drive unbounded target-speed
                # growth during an optical outage. No target truth is used.
                self.pad_a *= math.exp(-h / 1.5)
            self.std = min(10.0, self.std + 0.15 * h)
            self.velocity_std = min(5.0, self.velocity_std + 0.3 * h)
            remaining -= h
            self.updates += 1
        fresh = m.optical_position is not None and m.sample_id != self.last_sample
        self.detected = bool(m.optical_position is not None)
        if fresh:
            if not self.initialized:
                self.r = m.optical_position.copy()
                self.pad_v = m.own_velocity.copy()
                self.pad_a[:] = 0.0
                self.initialized = True
            else:
                elapsed = max(m.time_s - self.last_detection, self.cfg.sensor_dt)
                innovation = m.optical_position - self.r
                # Reject impossible PnP branch jumps; do not reset uncertainty.
                if np.linalg.norm(innovation) > max(1.0, 4.0 * elapsed):
                    fresh = False
                else:
                    self.r += 0.65 * innovation
                    self.pad_v -= 0.12 * innovation / elapsed
                    self.pad_a -= 0.005 * innovation / (elapsed * elapsed)
                    self.pad_v = np.clip(self.pad_v, -10, 10)
                    self.pad_a = np.clip(self.pad_a, -2, 2)
            if fresh:
                self.last_detection = m.time_s
                self.std = 0.03
                self.velocity_std = 0.15
            self.last_sample = m.sample_id
        self.rv = m.own_velocity - self.pad_v
        if self.cfg.level_ground_pad:
            # Known UGV kinematics, not the simulator's current state: a
            # ground carrier on a level plane cannot acquire vertical speed
            # from noisy camera innovations and coast upward during a miss.
            self.pad_v[2] = 0.0
            self.pad_a[2] = 0.0
            self.rv[2] = m.own_velocity[2]
        if fresh:
            self.last_optical_r = self.r.copy()
            self.last_optical_rv = self.rv.copy()
        self.last_t = m.time_s
        self.own = m

    def _update_reference(self, m):
        track = self.reference_track
        accepted = track.update(m, level_ground=self.cfg.level_ground_pad)
        self.initialized = track.initialized
        self.r = m.own_position-track.position if track.initialized else np.zeros(3)
        self.pad_v = track.velocity.copy()
        self.pad_a = track.acceleration.copy()
        self.rv = m.own_velocity-self.pad_v
        self.std, self.velocity_std, self.acceleration_std = np.max(track.std, axis=1)
        self.last_t, self.last_detection, self.last_sample = track.last_t, track.last_detection, track.last_sample
        self.detected = bool(m.optical_position is not None)
        self.own = m
        self.updates += 1
        if accepted:
            self.last_optical_r = self.r.copy()
            self.last_optical_rv = self.rv.copy()

    @property
    def age(self):
        return (
            1e3 if not self.initialized else max(0.0, self.last_t - self.last_detection)
        )


@dataclass(frozen=True)
class Safety:
    inhibited: bool
    abort: bool
    eligible: float
    reasons: tuple


def safety_status(est, cfg):
    if cfg.reference_supervisor:
        raise ValueError('v10 requires the episode-owned ReferenceSpatialSupervisor')
    lateral = float(np.linalg.norm(est.r[:2]))
    speed = float(np.linalg.norm(est.rv[:2]))
    from ..mathx import quat_to_euler_zyx

    angles = quat_to_euler_zyx(est.own.quaternion)
    tilt = float(np.linalg.norm(angles[:2]))
    # Shrinking footprint admits a physically safe corridor all the way to
    # contact; the abort hold does not replace authorized terminal descent.
    footprint = max(
        cfg.pad_half_width * 0.7,
        max(est.r[2], 0) * min(np.tan(np.asarray(cfg.fov) / 2)) * 0.65,
    )
    reasons = []
    last_r, last_v = est.last_optical_r, est.last_optical_rv
    terminal_coast = bool(
        last_r is not None
        and last_v is not None
        and 0 < last_r[2] <= 0.45
        and np.linalg.norm(last_r[:2]) <= 0.15
        and np.linalg.norm(last_v[:2]) <= 0.3
        and -0.35 <= last_v[2] <= 0.05
        and 0 < est.r[2] <= 0.45
        and est.age <= 1.5
        and lateral + 2 * est.std <= cfg.pad_half_width
        and speed <= cfg.touchdown_xy_speed
        and tilt <= cfg.touchdown_tilt
    )
    if not est.initialized or (est.age > 0.5 and not terminal_coast):
        reasons.append("visual_track_unavailable")
    if lateral > footprint:
        reasons.append("outside_visible_corridor")
    if tilt > math.radians(20):
        reasons.append("tilt")
    if est.r[2] < 1.0 and (lateral > cfg.pad_half_width or speed > 0.6):
        reasons.append("terminal_alignment")
    abort = est.age > cfg.loss_timeout or lateral > 60 or est.r[2] > 20
    eligible = float(
        np.clip(1 - lateral / max(footprint, 1e-6), 0, 1)
        * np.clip(1 - speed / 3.0, 0, 1)
        * (not reasons)
    )
    return Safety(bool(reasons), bool(abort), eligible, tuple(reasons))


def supervised_action(action, est, safety, cfg):
    if cfg.reference_supervisor:
        raise ValueError('v10 requires the episode-owned ReferenceSpatialSupervisor')
    action = np.asarray(action, dtype=float)
    if action.shape != (3,) or not np.isfinite(action).all():
        raise ValueError("finite spatial action required")
    applied = np.clip(action, -1, 1).copy()
    if safety.inhibited or safety.abort:
        # Hold/recover vertically. No privileged steering or PN teacher.
        target = max(1.0, est.r[2]) if safety.abort else max(0.12, est.r[2])
        az = np.clip(
            1.5 * (target - est.r[2]) - 1.8 * est.rv[2],
            -cfg.max_acceleration[2],
            cfg.max_acceleration[2],
        )
        applied[2] = max(applied[2], az / cfg.max_acceleration[2])
    if safety.abort and (cfg.schema == "spatial-causal-rgat/5" or cfg.direct_acceleration):
        # Reference abort_braking_hold, extended symmetrically to ENU x/y.
        # Own measured velocity is trustworthy even when target prediction is
        # not. Never chase an extrapolated, unobserved pad during an abort.
        applied[:2] = np.clip(-1.5 * est.own.own_velocity[:2] /
                             np.asarray(cfg.max_acceleration[:2]), -1., 1.)
    # Common causal braking envelope; it does not make unsafe contacts safe.
    height = max(est.r[2] - cfg.contact_height, 0.0)
    safe_sink = math.sqrt(cfg.touchdown_z_speed ** 2 + 2.0 * 0.7 * height)
    if est.rv[2] < -safe_sink:
        applied[2] = max(
            applied[2],
            min(1.0, (-safe_sink - est.rv[2]) / (cfg.dt * cfg.max_acceleration[2])),
        )
    return applied


@dataclass(frozen=True)
class Packet:
    values: np.ndarray
    registry_sha256: str = REGISTRY_HASH


@dataclass(frozen=True)
class Graph:
    X: np.ndarray
    schema_hash: str = REGISTRY_HASH


@dataclass(frozen=True)
class Observation:
    packet: Packet
    graph: Graph


def observation(est, safety, elapsed, cfg):
    m = est.own
    (bx, by), _ = camera_bearings(est.r, m.quaternion, cfg)
    margins = np.asarray(cfg.fov) / 2 - np.abs([bx, by])
    raw = np.r_[
        est.r,
        est.rv,
        est.pad_v,
        est.pad_a,
        m.own_velocity,
        m.quaternion,
        m.angular_rate,
        bx,
        by,
        margins,
        float(est.detected),
        m.confidence,
        min(est.age, 1000),
        float(est.initialized),
        est.std,
        est.velocity_std,
        max(0, 1 - elapsed / cfg.horizon),
        float(safety.inhibited),
        float(safety.abort),
    ]
    values = (raw / (SCALES + np.abs(raw))).astype(np.float32)
    values[DIRECT] = raw[DIRECT]
    if cfg.direct_acceleration:
        from scipy.spatial.transform import Rotation
        horizon = 0.5
        predicted_r = est.r + horizon*est.rv - 0.5*horizon**2*est.pad_a
        rotation = Rotation.from_quat(m.quaternion[[1, 2, 3, 0]])
        future_q = (rotation * Rotation.from_rotvec(horizon*m.angular_rate)).as_quat()
        predicted, depth = camera_bearings(predicted_r, future_q[[3, 0, 1, 2]], cfg)
        future = (np.r_[predicted, np.asarray(cfg.fov)/2-np.abs(predicted)]
                  if depth > 0 else np.r_[np.zeros(2), -np.asarray(cfg.fov)/2])
        if not est.initialized:
            future[:] = 0.
        scales = np.tile(np.asarray(cfg.fov)/2, 2)
        values = np.r_[values, future/(scales+np.abs(future))].astype(np.float32)
    if cfg.reference_tracking:
        values = np.r_[values, est.acceleration_std/(1+est.acceleration_std),
                       np.clip(est.previous_action, -1., 1.)].astype(np.float32)
    if cfg.reference_context:
        measured = np.zeros(2)
        valid = m.optical_position is not None
        if valid:
            measured, _ = camera_bearings(m.optical_position,
                m.quaternion if m.optical_quaternion is None else m.optical_quaternion, cfg)
        optical_age = max(0., m.time_s-(m.optical_time_s if m.optical_time_s is not None else m.time_s))
        values = np.r_[values, measured/(np.asarray(cfg.fov)/2+np.abs(measured)),
                       float(valid), optical_age/(1+optical_age)].astype(np.float32)
        from .context import reference_context_graphs
        X = reference_context_graphs(values, cfg)
        if not np.isfinite(values).all() or not np.isfinite(X).all():
            raise ValueError('nonfinite causal packet')
        return Observation(Packet(values,cfg.registry_hash),Graph(X,cfg.registry_hash))
    p = dict(zip(cfg.packet_fields, values))
    X = np.zeros((9, 12), dtype=np.float32)

    def put(node, names):
        fields = names.split()
        X[node, : len(fields)] = [p[name] for name in fields]

    # Nine named contexts, same typed edges and four grouped readouts as v2.8.
    put(0, "detected bx by mx my confidence age initialized")
    put(1, "pvx pvy pvz pax pay paz velstd confidence initialized")
    put(2, "rz vx vy vz remaining")
    put(3, "qw qx qy qz wx wy wz")
    put(4, "rx ry rz rvx rvy rvz posstd velstd initialized")
    put(5, "rx ry rvx rvy pax pay confidence")
    put(6, "age bx by mx my initialized confidence")
    X[7, 0] = safety.eligible
    X[7, 1:9] = [p[k] for k in "rx ry rvx rvy rvz rz confidence inhibited".split()]
    put(8, "inhibited abort age posstd velstd initialized")
    if cfg.direct_acceleration:
        X[0, :9] = 0
        X[6, :9] = 0
        put(0, "detected bx by pred_bx pred_by confidence age initialized")
        put(6, "age pred_bx pred_by pred_mx pred_my initialized confidence")
    if cfg.reference_tracking:
        put(1, "pvx pvy pvz pax pay paz velstd accstd initialized")
    X[:, 9] = p["remaining"]
    X[:, 10] = 1.0
    X[:, 11] = np.arange(1, 10) / 9
    if not np.isfinite(values).all() or not np.isfinite(X).all():
        raise ValueError("nonfinite causal packet")
    return Observation(Packet(values, cfg.registry_hash), Graph(X, cfg.registry_hash))
