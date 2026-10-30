"""Registered contract of the current-observation 3D landing path.

One module holds what the new path may read and how it is scaled: the 21D
source channels (built from the direct-policy registry, never restated), the
nominal v12 camera, the provisional contact definitions, the action and the
budget-matched model design.  Hashes are split so a calibration or feature
change is never mistaken for a model change, and the identifiers are not
shared with any older route.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import yaml

from . import (ACTION_VERSION, FEATURE_VERSION, GRAPH_VERSION, MODEL_VERSION,
               OBSERVATION_VERSION, SCHEMA_ID)
from ..direct_policy.contracts import SPATIAL_FIELDS, digest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "current_semantics" / "spatial_view_contact_v1.yaml"

BASELINE_3D_SHA = "cab3dd2123c7eee3968a8c00753f2ba29d55fe71"
PROVENANCE_2D = MappingProxyType(dict(
    remote_main="d438a7d4096d5b186c731c894306754aa5ab628d",
    delivery_commit="5853781a21e82d6e3ada1f36aa16690ec3a17ad9",
    status=("user-reported; neither object nor the v1 evidence bundle exists on "
            "this machine (audit 2026-10-28)")))

NODE_ORDER = ("ObservationQuality", "RelativeGeometry", "MotionState", "AttitudeState",
              "ApproachSituation", "ViewGeometry", "ContactCondition", "LandingContext")

ARMS = ("raw_ppo", "semantic_mlp_ppo", "untyped_gnn_ppo", "ontology_rgat_ppo")
ARM_ACTOR_INPUT = MappingProxyType({"raw_ppo": "raw21", "semantic_mlp_ppo": "flat44",
                                    "untyped_gnn_ppo": "raw21+graph",
                                    "ontology_rgat_ppo": "raw21+graph"})
ENCODER_RELATIONS = MappingProxyType({"untyped_gnn_ppo": 2, "ontology_rgat_ppo": 6})
DESIGN_PARAMETER_TOTALS = MappingProxyType({"raw_ppo": 4022, "semantic_mlp_ppo": 4008,
                                            "untyped_gnn_ppo": 4020,
                                            "ontology_rgat_ppo": 4016})

OBSERVATION_DIMENSION = 21
FEATURE_DIMENSION = 44
NODE_INPUT_DIMENSION = 10
ACTION_DIMENSION = 3

STABLE_EXCLUDED_CHANNELS = ("tracking_bias_x", "tracking_bias_y", "handover_progress")
TRIG_TOLERANCE = 1e-6


# ---------------------------------------------------------------- observation

@dataclass(frozen=True)
class ObservationChannel:
    name: str
    meaning: str
    unit: str
    transform: str
    scale: float | None
    node: str
    source_scale: float | None = None


_Q, _G, _M, _A = NODE_ORDER[:4]
_CHANNEL_SEMANTICS = {
    "relative_x": ("m", "signed", _G),
    "relative_y": ("m", "signed", _G),
    "relative_height": ("m", "signed", _G),
    "relative_vx": ("m/s", "signed", _M),
    "relative_vy": ("m/s", "signed", _M),
    "ugv_vx": ("m/s", "signed", _M),
    "ugv_vy": ("m/s", "signed", _M),
    "drone_vz": ("m/s", "signed", _M),
    "drone_sinPhi": ("1", "trig", _A),
    "drone_cosPhi": ("1", "trig", _A),
    "drone_sinTheta": ("1", "trig", _A),
    "drone_cosTheta": ("1", "trig", _A),
    "drone_sinPsi": ("1", "trig", _A),
    "drone_cosPsi": ("1", "trig", _A),
    "drone_rollRate": ("rad/s", "signed", _A),
    "drone_pitchRate": ("rad/s", "signed", _A),
    "drone_yawRate": ("rad/s", "signed", _A),
    "ugv_visionUpdated": ("bool", "flag", _Q),
    "ugv_visionAge": ("s", "age", _Q),
    "drone_navigationValid": ("bool", "flag", _Q),
    "drone_navigationAge": ("s", "age", _Q),
}
TRANSFORM_BY_NAME = MappingProxyType({name: spec[1] for name, spec in _CHANNEL_SEMANTICS.items()})

# spatial-current-observation/2 (user request 2026-10-28: reshape the common
# observation so PPO can learn without BC). Same 21 channels, order and rules;
# only the signed position/velocity scales change, to the landing regime. The
# direct-policy source scales (3 m, 8 m, 10 m/s, 1.5 m/s) put the 0.30 m/s
# touchdown limit at 0.03 of a relative-velocity channel and a 0.2 m height at
# 0.025, so the raw actor and the COMMON raw21 critic needed ~10x larger weights
# before a landing-relevant difference moved their output; the derived phi44
# approach features already use 1 m and 1 m/s. Rates and ages keep their scales.
OBSERVATION_SCALE_OVERRIDES = MappingProxyType({
    "relative_x": 1.0, "relative_y": 1.0, "relative_height": 2.0,
    "relative_vx": 1.0, "relative_vy": 1.0, "ugv_vx": 1.0, "ugv_vy": 1.0, "drone_vz": 1.0})

OBSERVATION_CHANNELS = tuple(
    ObservationChannel(name=item.name, meaning=item.source,
                       unit=_CHANNEL_SEMANTICS[item.name][0],
                       transform=_CHANNEL_SEMANTICS[item.name][1],
                       scale=OBSERVATION_SCALE_OVERRIDES.get(item.name, item.scale),
                       node=_CHANNEL_SEMANTICS[item.name][2], source_scale=item.scale)
    for item in SPATIAL_FIELDS if item.name in _CHANNEL_SEMANTICS)
OBSERVATION_NAMES = tuple(item.name for item in OBSERVATION_CHANNELS)

ENCODING_RULES = MappingProxyType({
    "signed": "u = x/(|x|+s); inverse x = s*u/(1-|u|), requires |u| < 1",
    "trig": ("u = sin(angle) | cos(angle), raw; inverse angle = atan2(sin, cos), "
             f"requires |sin^2+cos^2-1| <= {TRIG_TOLERANCE:g}"),
    "flag": "u in {0, 1} exactly",
    "age": ("u = x/(x+s), x = inf -> u = 1 (sentinel); inverse x = s*u/(1-u), "
            "u = 1 -> inf, requires 0 <= u <= 1"),
})
# direct_policy.observation.spatial_vector zeroes out[:7] while the pad
# estimate is uninitialised; named here so nothing relies on the position.
ESTIMATE_CHANNELS = ("relative_x", "relative_y", "relative_height", "relative_vx",
                     "relative_vy", "ugv_vx", "ugv_vy")
ESTIMATE_VALIDITY_RULE = ("estimate_valid := ugv_visionAge < 1; when false every "
                          "estimate channel is exactly 0 and must not read as alignment")
TRIG_PAIRS = (("roll", "drone_sinPhi", "drone_cosPhi"),
              ("pitch", "drone_sinTheta", "drone_cosTheta"),
              ("yaw", "drone_sinPsi", "drone_cosPsi"))

_source_names = tuple(item.name for item in SPATIAL_FIELDS)
if (len(_source_names) != OBSERVATION_DIMENSION
        or set(TRANSFORM_BY_NAME) != set(_source_names)
        or OBSERVATION_NAMES != _source_names):
    raise AssertionError("every 21D source channel needs exactly one registered transform")
if OBSERVATION_NAMES[:len(ESTIMATE_CHANNELS)] != ESTIMATE_CHANNELS:
    raise AssertionError("estimate channels are no longer the zeroed source prefix")
for _channel in OBSERVATION_CHANNELS:
    if (_channel.scale is None) != (_channel.transform in ("trig", "flag")):
        raise AssertionError(f"{_channel.name}: scale presence contradicts its transform")
if any(TRANSFORM_BY_NAME[name] != "signed" for name in OBSERVATION_SCALE_OVERRIDES):
    raise AssertionError("only signed channels may be rescaled")


# spatial-current-observation/3 (user choice "a", 2026-10-29): the pad track
# RE-ACQUIRES. The reference ABG track initialises the pad velocity to the own
# velocity and corrects it by 0.02/gap per frame, so after the Isaac handover
# kick its prediction drifts past the innovation gate (max(4 sigma, 0.25 m)) and
# every later detection is rejected: 8 of 14 missed approaches in isaac-eval/2
# were such lockouts with the pad well inside the camera footprint
# (G0_G4_STATUS_KO.md 10.8). After this many CONSECUTIVE fresh detections the
# track rejected, it restarts from them: position = their least-squares line at
# the decision time, velocity = its slope (level ground), sigma = a fresh track's.
# Nothing changes while the track accepts frames.
TRACK_REACQUIRE_AFTER_REJECTIONS = 3
TRACK_REACQUIRE_RULE = (
    "after TRACK_REACQUIRE_AFTER_REJECTIONS consecutive fresh detections (the track's own "
    "freshness rule) rejected by the innovation gate, re-initialise the track from those "
    "detections: least-squares pad position and velocity over their capture times (vertical "
    "velocity 0, level ground), propagated to the sample time; acceleration 0; sigma reset "
    "to a fresh track's; counts as an accepted frame")


def observation_manifest() -> dict[str, Any]:
    return _plain({
        "version": OBSERVATION_VERSION,
        "dimension": OBSERVATION_DIMENSION,
        "source": "ontology_rgat.direct_policy.contracts.SPATIAL_FIELDS",
        "source_rescaling": ("observation.from_source: each signed channel of "
                             "direct_policy.observation.spatial_vector is decoded with its "
                             "source scale and re-encoded with the registered scale; every "
                             "other channel is copied"),
        "scale_overrides": OBSERVATION_SCALE_OVERRIDES,
        "pad_track": {"estimator": "spatial.estimation.ReferenceSpatialTrack(capture_aligned=True)",
                      "reacquire_after_rejections": TRACK_REACQUIRE_AFTER_REJECTIONS,
                      "reacquire_rule": TRACK_REACQUIRE_RULE},
        "channels": OBSERVATION_CHANNELS,
        "encoding_rules": ENCODING_RULES,
        "estimate_channels": ESTIMATE_CHANNELS,
        "estimate_validity": ESTIMATE_VALIDITY_RULE,
        "excluded_stable_channels": STABLE_EXCLUDED_CHANNELS,
    })


# ------------------------------------------------------------------- camera

DEPLOYMENT_PROFILE = "spatial-reference/12"
DEPLOYMENT_PROFILE_SHA256 = "af605e2beac683ede42805cf4e858a2ec32fa3501324ecf05381426174d72e18"
_PROFILE_FILES = MappingProxyType({"spatial-reference/12": "config/spatial-isaac-system-v12.yaml"})
# Kit logs of all 31 v12 Isaac runs (2026-10-10..13) record the as-flown camera:
# fx 320, optical axes in body diag(1,-1,-1), mount (0, 0, -0.1598) m.
CALIBRATION_STATUS = ("nominal formula; equals the as-flown renderer record (fx 320, optical "
                      "axes diag(1,-1,-1), mount -0.1598 m) in 31 v12 Kit logs "
                      "2026-10-10..13; the in-plane rotation is not asserted by code")
IMAGE_NORMALISATION = "xi = 2u/(W-1) - 1, eta = 2v/(H-1) - 1"
# Transpose of isaac_sim/marker_vision.py R_BODY_FROM_OPTICAL for the nadir
# mount: optical +X = body forward, +Y = body right, +Z = body down.
_NADIR_R_CB = ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0))


def _merge(base: dict, overlay: dict) -> dict:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_extending(path: Path, seen: frozenset = frozenset()) -> dict:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"cyclic deployment profile at {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = data.pop("extends", None)
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        data = _merge(_load_extending(parent_path, seen | {path}), data)
    if data.get("execution_status") == "control_boundary_only":
        raise ValueError("spatial control boundary has no compatible live training backend yet")
    return data


def resolve_deployment_profile(profile: str = DEPLOYMENT_PROFILE) -> dict[str, Any]:
    """The resolved scientific deployment profile, without its import chain.

    ``spatial.runtime_contract.deployment_profile`` is the authority, but
    importing it loads ``ontology_rgat.benchmarks``, which pulls in torch, cv2
    and the isaac_sim geometry modules.  This repeats its three steps
    (extends-merge, scientific view, canonical JSON hash); a test pins the
    result to the authority so the two cannot drift.  The path is recorded
    relative to the project root so the hash does not encode a checkout.
    """
    if profile not in _PROFILE_FILES:
        raise ValueError(f"unsupported deployment profile {profile!r}; "
                         f"registered: {sorted(_PROFILE_FILES)}")
    relative = _PROFILE_FILES[profile]
    resolved = deepcopy(_load_extending(PROJECT_ROOT / relative))
    resolved.get("isaac", {}).pop("viewport_follow", None)
    resolved.get("parallel", {}).pop("operator_view", None)
    payload = json.dumps(resolved, sort_keys=True, separators=(",", ":"))
    return {"profile": profile, "path": relative,
            "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "configuration": resolved}


@dataclass(frozen=True)
class ProfileSource:
    profile: str
    path: str
    sha256: str


@dataclass(frozen=True)
class CameraCalibration:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, ...]
    t_bc_flu: tuple[float, ...]
    r_cb: tuple[tuple[float, ...], ...]
    z_near_m: float
    status: str
    source: ProfileSource

    def __post_init__(self) -> None:
        object.__setattr__(self, "distortion", tuple(float(v) for v in self.distortion))
        object.__setattr__(self, "t_bc_flu", tuple(float(v) for v in self.t_bc_flu))
        object.__setattr__(self, "r_cb",
                           tuple(tuple(float(v) for v in row) for row in self.r_cb))
        if int(self.width) < 2 or int(self.height) < 2:
            raise ValueError("image must be at least 2x2 pixels")
        for name in ("fx", "fy", "z_near_m"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be finite and positive")
        if not (0 < self.cx < self.width and 0 < self.cy < self.height):
            raise ValueError("principal point must lie inside the image")
        if len(self.distortion) != 5 or len(self.t_bc_flu) != 3:
            raise ValueError("distortion needs 5 coefficients and t_bc_flu 3 components")
        rotation = np.asarray(self.r_cb, dtype=np.float64)
        if (rotation.shape != (3, 3)
                or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-12)):
            raise ValueError("r_cb must be a proper rotation")

    @property
    def hfov_rad(self) -> float:
        return math.atan(self.cx / self.fx) + math.atan((self.width - self.cx) / self.fx)

    @property
    def vfov_rad(self) -> float:
        return math.atan(self.cy / self.fy) + math.atan((self.height - self.cy) / self.fy)

    @property
    def r_cb_matrix(self) -> np.ndarray:
        return np.asarray(self.r_cb, dtype=np.float64)

    @property
    def t_bc_vector(self) -> np.ndarray:
        return np.asarray(self.t_bc_flu, dtype=np.float64)

    @property
    def has_distortion(self) -> bool:
        return any(value != 0.0 for value in self.distortion)

    def to_dict(self) -> dict[str, Any]:
        return _plain({
            "width": int(self.width), "height": int(self.height),
            "fx": float(self.fx), "fy": float(self.fy),
            "cx": float(self.cx), "cy": float(self.cy),
            "distortion": self.distortion, "t_bc_flu": self.t_bc_flu, "r_cb": self.r_cb,
            "z_near_m": float(self.z_near_m), "status": self.status,
            "source": self.source,
            "image_normalisation": IMAGE_NORMALISATION,
            "derived": {"hfov_deg": math.degrees(self.hfov_rad),
                        "vfov_deg": math.degrees(self.vfov_rad)},
        })

    @property
    def calibration_hash(self) -> str:
        return digest(self.to_dict())

    @classmethod
    def nominal_v12(cls, z_near_m: float) -> "CameraCalibration":
        """Nominal pinhole of the v12 nadir camera, rebuilt from the resolved profile.

        K follows ``isaac_sim/marker_vision.py`` ``intrinsics_from_fov``:
        square pixels, principal point (W/2, H/2), zero distortion.  The
        profile's ``vertical_fov_deg: 70`` is an inherited ZED value nothing
        reads, so VFOV is derived from K.  Isaac swaps in renderer intrinsics
        at runtime without recording them, hence the status.
        """
        profile = resolve_deployment_profile(DEPLOYMENT_PROFILE)
        camera = profile["configuration"]["vision"]["camera"]
        if camera.get("model") != "spatial_nadir_aruco_v1" or camera.get("pitch_down_deg") != 90.0:
            raise ValueError("nominal_v12 needs the nadir spatial_nadir_aruco_v1 camera; "
                             f"profile has model={camera.get('model')!r}, "
                             f"pitch_down_deg={camera.get('pitch_down_deg')!r}")
        width, height = (int(v) for v in camera["resolution"])
        hfov = math.radians(float(camera["horizontal_fov_deg"]))
        if not 0.0 < hfov < math.pi:
            raise ValueError("horizontal_fov_deg must be in (0, 180)")
        fx = (width / 2.0) / math.tan(hfov / 2.0)
        return cls(width=width, height=height, fx=fx, fy=fx, cx=width / 2.0, cy=height / 2.0,
                   distortion=(0.0,) * 5,
                   t_bc_flu=tuple(float(v) for v in camera["mount_translation_flu_m"]),
                   r_cb=_NADIR_R_CB, z_near_m=float(z_near_m), status=CALIBRATION_STATUS,
                   source=ProfileSource(profile=profile["profile"], path=profile["path"],
                                        sha256=profile["sha256"]))


nominal_v12 = CameraCalibration.nominal_v12


# ------------------------------------------------------------------ contact

def contact_tilt_rad(roll, pitch):
    """Thrust-axis angle from vertical, ``acos(cos(roll)cos(pitch))``.

    Provisional until G2.  It is the ``settled`` tilt of spatial/safety.py;
    ``direct_policy.reward.classify_terminal`` uses ``norm(roll, pitch)``
    instead.  Features and the future evaluator import this one so the two
    definitions cannot mix.
    """
    return np.arccos(np.clip(np.cos(roll) * np.cos(pitch), -1.0, 1.0))


def contact_rate_rad_s(p, q):
    """Body roll/pitch rate norm, the ``settled`` rate of spatial/safety.py (provisional)."""
    return np.hypot(p, q)


CONTACT_DEFINITIONS = MappingProxyType({
    "tilt": "acos(clip(cos(roll)*cos(pitch), -1, 1))",
    "rate": "hypot(p, q)",
    "pad": "circular, |(ex, ey)| <= pad_radius_m",
})


# ------------------------------------------------------------------- config

@dataclass(frozen=True)
class FeatureScales:
    approach_position_m: float
    approach_velocity_m_s: float
    approach_closure_m2_s: float
    image_coordinate: float
    image_rate_per_s: float


@dataclass(frozen=True)
class ContactLimits:
    pad_radius_m: float
    xy_speed_m_s: float
    z_speed_m_s: float
    tilt_deg: float
    rate_deg_s: float
    status: str

    @property
    def tilt_rad(self) -> float:
        return math.radians(self.tilt_deg)

    @property
    def rate_rad_s(self) -> float:
        return math.radians(self.rate_deg_s)


ADAPTER_MAX_TILT_LIMIT_DEG = 25.0  # SpatialAccelerationController refuses more


@dataclass(frozen=True)
class ActionContract:
    frame: str
    maximum_m_s2: tuple[float, float, float]
    adapter_max_tilt_deg: float
    version: str = ACTION_VERSION
    names: tuple[str, str, str] = ("a_x", "a_y", "a_z")
    distribution: str = "z ~ N(mu, diag(exp(log_std)^2)); action = maximum * tanh(z)"
    log_prob_space: str = "latent"
    log_std: str = "global, one per axis"
    px4_boundary: str = "ENU -> NED once, at the PX4 adapter"
    # The constant limits every arm's executed action passes through; a task
    # property, so it is hashed with the action, and every change it makes is
    # recorded per step (backend.ActionTrace), never hidden in the policy.
    adapter: str = ("controllers.spatial_controller.SpatialAccelerationController("
                    "acceleration_only=True, max_tilt_deg=adapter_max_tilt_deg) with the "
                    "spatial-reference/12 max_velocity/max_acceleration: per-axis clip, one-step "
                    "velocity envelope, joint tilt/thrust envelope; no supervisor, hold or prior")

    @property
    def maximum(self) -> np.ndarray:
        return np.asarray(self.maximum_m_s2, dtype=np.float64)


# ----------------------------------------------------------------------- G3

# SOURCE_AUDIT_KO.md section 3 lists the seeds with recorded local use.  Three
# entries are open-ended there ("12000+", "100000+", "1000001+"); each is
# closed at the block its generating code reaches, so a G3 pilot seed can be
# checked against every one.  (name, first, last, where the bound comes from)
AUDITED_SEED_RANGES = (
    ("2d_train", 291, 465, "audit section 3"),
    ("2d_environment", 9001, 9300, "audit section 3"),
    ("3d_train", 828, 832, "audit section 3"),
    ("3d_held_out", 4100, 4147, "audit section 3"),
    ("3d_teacher_stress", 5000, 5095, "audit section 3"),
    ("matlab_port_evaluation", 3001, 3200, "audit section 3"),
    ("isaac", 12000, 99999, "audit '12000+'; tools/evaluate_spatial_checkpoint.py refuses "
                            "test seeds >= 100000"),
    ("matlab_port_training", 100000, 199999, "audit '100000+'; direct_policy/runner.py "
                                             "seed*100000 + update*episodes + episode at "
                                             "its default --seed 1"),
    ("matlab_port_isaac_validation", 1000000, 1999999, "audit '1000001+'; direct_policy/"
                                                       "runner.py seed + 1000000"),
)
G3_PILOT_SEED_NOTE = ("implementation checks only, never development or confirmation; "
                      "load_config refuses a pilot seed inside any AUDITED_SEED_RANGES entry")


@dataclass(frozen=True)
class AxisCouplingProbe:
    """Paired runs from the reset hover: a lateral push of one magnitude along
    x, y or the diagonal, then a zero-command coast, minus a zero-command run of
    the same seed (same pad, disturbance and sensor draws)."""
    acceleration_m_s2: float
    push_steps: int
    coast_steps: int
    cross_axis_ratio_max: float
    superposition_residual_max: float


@dataclass(frozen=True)
class FovExitProbe:
    """Hold height, hold the estimated pad centred on the cross axis, and drive the
    estimated pad velocity relative to the vehicle to ``exit_speed_m_s`` along one
    image axis until detection stops, so the pad leaves through ONE edge."""
    exit_speed_m_s: float
    velocity_gain_per_s: float
    cross_position_gain_per_s2: float
    height_gain_per_s2: float
    acceleration_limit_m_s2: float
    max_steps: int
    steps_after_loss: int
    overshoot_tolerance_m: float


@dataclass(frozen=True)
class BlindZoneProbe:
    """Track the estimated pad laterally and descend along a height ramp until contact."""
    position_gain_per_s2: float
    velocity_gain_per_s: float
    sink_speed_m_s: float
    height_gain_per_s2: float
    vertical_velocity_gain_per_s: float
    acceleration_limit_m_s2: float
    max_steps: int


@dataclass(frozen=True)
class QualityProbe:
    """Float tolerance of the age identity age = decision - capture after the
    registered age encoding and decode (both float64)."""
    age_tolerance_s: float


@dataclass(frozen=True)
class G3LocalConfig:
    axis_coupling: AxisCouplingProbe
    fov_exit: FovExitProbe
    low_altitude_blind_zone: BlindZoneProbe
    quality_channels: QualityProbe


@dataclass(frozen=True)
class G3IsaacConfig:
    """The Isaac half: the local probe flights on a few pilot seeds through the
    owned Isaac/PX4 stack.  ``axis_sign_min_abs_correlation`` is the floor on
    corr(sin pitch, +a_x) and -corr(sin roll, +a_y); the audit measured 0.686 and
    0.966 on earlier Isaac flights (SOURCE_AUDIT_KO.md section 1)."""
    seeds: tuple[int, ...]
    axis_sign_min_abs_correlation: float


@dataclass(frozen=True)
class G3Config:
    pilot_seeds: tuple[int, ...]
    local: G3LocalConfig
    isaac: G3IsaacConfig
    pilot_seed_note: str = G3_PILOT_SEED_NOTE
    audited_seed_ranges: tuple[tuple[Any, ...], ...] = AUDITED_SEED_RANGES


# ---------------------------------------------------------------------- G4

# Pilot 1 (time-ramped curriculum, 300 updates) and pilot 2 (success-gated
# curriculum, 600 updates) are archived under results/.../g4_pilot_v1 and
# g4_pilot_v2 with their config snapshots; both verdicts were G4 failed. Pilot 3
# (task v2 + observation v2) PASSED and is archived in g4_pilot_v3 with the two
# Isaac evaluations of its checkpoints (isaac_eval_v1, isaac_eval_v2). Pilot 4
# (observation/3) and pilot 5 (reward /3 + handover own-state error) PASSED and are
# archived in g4_pilot_v4 / g4_pilot_v5 with isaac_eval_v3 / isaac_eval_v4. Pilot 6
# adds reward /4 (lost-view climb, no command-change cost) and the vision blackout,
# archived in g4_pilot_v6 with isaac_eval_v5. Pilot 7 re-checks learnability on task
# v5 (omnidirectional scenario, held-out composition) before development.
G4_PROTOCOL = "g4-pilot/7"

@dataclass(frozen=True)
class PPOSettings:
    """Guide 8.3's common PPO: clipped surrogate on the latent Gaussian, common raw
    critic, GAE on exp(-dt/tau) of the actual transition, post-step KL guard."""
    epochs: int
    minibatch: int
    clip: float
    gae_lambda: float
    policy_lr: float
    value_lr: float
    entropy_coefficient: float
    max_grad_norm: float
    target_kl: float


@dataclass(frozen=True)
class G4Criteria:
    """Registered before the pilot ran: on the held-out test seeds, per model seed."""
    deterministic_success_min: float
    sampled_success_min: float


@dataclass(frozen=True)
class SuccessGatedCurriculum:
    """Training-only difficulty that follows learning: promote by ``step`` once the
    sampled training success over the last ``window_updates`` updates at the current
    difficulty reaches ``promote_at_success``; never demote.  Pilot 1's time ramp
    reached 0.6 before the policy had stopped drifting (G0_G4_STATUS_KO.md section 9)."""
    start_difficulty: float
    step: float
    window_updates: int
    promote_at_success: float


@dataclass(frozen=True)
class HandoverOwnStateError:
    """Training-only randomisation (pilot 5, user 2026-10-29): right after handover
    the own velocity the observer receives carries e0 * exp(-t/tau) and the own
    position (and the pose paired with each image) its integral, as Isaac's EKF
    did after the handover kick (G0_G4_STATUS_KO.md 10.7: +-0.6 m/s between two
    decisions, relative velocity of the wrong sign for 0.2-0.4 s).  e0: uniform
    horizontal direction with magnitude U(0, horizontal_max), vertical
    U(-vertical_max, vertical_max); tau U(tau_min, tau_max).  Validation and test
    never carry it."""
    horizontal_max_m_s: float
    vertical_max_m_s: float
    tau_min_s: float
    tau_max_s: float


@dataclass(frozen=True)
class VisionBlackout:
    """Training-only randomisation (pilot 6, user 2026-10-29): with ``probability``
    per training episode the observer receives no optical frame for a window of
    U(duration_min, duration_max) s starting U(start_min, start_max) s after
    handover, so the policy meets a stale track at altitude.  The trained policies
    almost never lost the pad there on the local plant (0-5 of 48 episodes,
    <= 2 % of decisions, pilot6_design_checks), while every late Isaac failure on
    record began that way (G0_G4_STATUS_KO.md 12.4-12.5); a reward for that state
    teaches nothing unless training visits it.  Plant, truth and verdict are
    untouched; validation, test and Isaac never carry it."""
    probability: float
    start_min_s: float
    start_max_s: float
    duration_min_s: float
    duration_max_s: float


@dataclass(frozen=True)
class TrainingRandomization:
    """Everything training episodes carry that validation, test and Isaac do not."""
    handover_own_state_error: HandoverOwnStateError
    vision_blackout: VisionBlackout


@dataclass(frozen=True)
class G4PilotConfig:
    """Guide 10 G4: one common-PPO learnability pilot on the Raw arm, judged on
    both the deterministic and the sampled policy."""
    protocol: str
    arm: str
    model_seeds: tuple[int, ...]
    updates: int
    episodes_per_update: int
    curriculum: SuccessGatedCurriculum
    validate_every: int
    validation_seeds: tuple[int, ...]
    test_seeds: tuple[int, ...]
    train_seed_base: int
    workers: int
    ppo: PPOSettings
    criteria: G4Criteria
    training_randomization: TrainingRandomization

    def train_seed(self, model_seed: int, update: int, episode: int) -> int:
        """Environment seed of one training episode (never a validation/test seed)."""
        return (self.train_seed_base + model_seed * 1_000_000
                + update * self.episodes_per_update + episode)


# isaac-eval/1 (seeds 994001-994010) failed its criterion and those seeds then
# served the traced diagnosis (G0_G4_STATUS_KO.md 10.6-10.7); /2 re-flew the same
# pilot-3 checkpoints on 994011-994020, whose traces then shaped observation/3.
# Both are archived (results/.../isaac_eval_v1, _v2). /3 flew the pilot-4 and /4 the
# pilot-5 checkpoints (isaac_eval_v3, _v4); /5 flies the pilot-6 checkpoints on
# fresh seeds.
ISAAC_EVAL_PROTOCOL = "isaac-eval/5"


@dataclass(frozen=True)
class IsaacEvalConfig:
    """The selected G4 checkpoints flown on Isaac/PX4 (user decision 2026-10-28:
    every selected-checkpoint evaluation runs on Isaac).  Deterministic policy,
    the same seeds for every checkpoint, criterion registered before the flight."""
    protocol: str
    seeds: tuple[int, ...]
    deterministic_success_min: float


# -------------------------------------------------------------------- task

TASK_VERSION = "spatial-current-task/5"
TASK_EVALUATOR = "spatial.environment.Evaluator"
REWARD_VERSION = "spatial-current-reward/4"
# What the task scores, stated for the manifest. The VERDICT and the terminal
# table are the spatial-reference/12 Evaluator's, unchanged (its constants are
# read from landing/terminal.py and the flight's SpatialConfig, never restated).
# The dense part is this path's common reward (reward.CommonRewardFunction).
TASK_DEFINITION = MappingProxyType({
    "verdict": ("spatial-reference/12 Evaluator.evaluate unchanged: the status of every "
                "decision (SUCCESS / UNSAFE_CONTACT / MISSED_PAD_CONTACT / "
                "SAFETY_ENVELOPE_VIOLATION / TASK_TIMEOUT / RUNNING)"),
    "terminal": ("the Evaluator's own table on that status: landing.terminal "
                 "REFERENCE_TERMINAL_REWARDS (SUCCESS 25, TASK_TIMEOUT -12) and its nominal "
                 "unsafe penalty (-40) for every unsafe status"),
    "dense": (f"{REWARD_VERSION}: Phi(s') - Phi(s) of a velocity-field potential "
              "Phi = -(w_d * |distance to the touchdown point| + w_v * |v_rel - v_des(r)|) "
              "on truth (own minus pad), Phi kept at the terminal state, minus "
              "(time_cost + control_cost * mean(u^2)) * dt; replaces the Evaluator's "
              "goal/view/control/spin/readiness/potential terms (2026-10-28, user request: "
              "learnable without BC). /3 (2026-10-29) added tilt_cost_per_s * "
              "max(0, (tilt - onset)/(21 deg - onset))^2 * dt on the true thrust-axis tilt "
              "and a command-change cost; /4 removes the command-change cost (it charged "
              "the sampled action, so exploration noise paid it) and makes v_des depend on "
              "the policy's own vision age (lost_view)"),
    "desired_velocity": ("v_des_xy = -e * min(g_xy, v_xy_max/|e|); v_des_z = -align * "
                         "min(v_z_max, v_touch + g_z * h), align = clip(1 - |e|/(r_0 + "
                         "k_r * h), 0, 1); e own-minus-pad horizontal, h = height above the "
                         "contact height. lost_view (/4): v_des_z blends to min(v_climb, "
                         "g_climb * max(0, h_search - h)) with weight w = clip((age - "
                         "a_0)/(a_1 - a_0), 0, 1) * max(clip((h - h_t)/ramp, 0, 1), "
                         "clip((|e| - r_t)/ramp, 0, 1)); age = the observation's vision age "
                         "(inf before the first accepted frame), never truth"),
    "reference_reward": ("the Evaluator's own reward is recorded per step as "
                         "reward_components.reference_reward and never trained on"),
    "contact_verdict": ("impact speed (|v_xy| > touchdown_xy_speed or |v_z| > touchdown_z_speed "
                        "-> UNSAFE_CONTACT), outside pad_half_width -> MISSED_PAD_CONTACT, "
                        "thrust axis > 21 deg at contact -> SAFETY_ENVELOPE_VIOLATION"),
    "envelope": ("out of contact: horizontal 60 m, height 20 m, below pad -0.2 m, thrust "
                 "axis 21 deg -> SAFETY_ENVELOPE_VIOLATION"),
    "supervisor": ("none: abort and inhibit are always false, so SAFE_ABORT and "
                   "UNAUTHORIZED_CONTACT are unreachable"),
    "dt": "the actual transition time of each decision (reward and discount alike)",
    "action_cost": "normalized policy action a/a_max before the adapter",
})


@dataclass(frozen=True)
class LostView:
    """/4 (user 2026-10-29): what the field asks once the policy has not seen the pad
    for a while.  Past ``after_s`` of vision age (fully at ``full_after_s``) the
    vertical demand turns from the sink to a climb toward ``search_height_m`` above
    the contact height, where the camera footprint is wide again, except in the
    blind last stretch of a landing: below ``terminal_height_m`` and within
    ``terminal_radius_m`` of the pad centre the sink continues, both edges ramped
    over ``ramp_m``.  The age is the observation's own vision age (decision time
    minus the last accepted capture), so the demand is a function of what the policy
    sees.  Sized on measurements (pilot6_design_checks): every pilot-4/-5 local
    landing goes blind below 0.43 m within 0.56 m of the centre, and over the 60
    recorded Isaac episodes the weight reached 0.5 in none of the 36 landings and
    in all 10 late envelope violations (the lost-pad failures), 4.3-15.9 s before
    each ended."""
    after_s: float
    full_after_s: float
    climb_speed_m_s: float
    climb_gain_per_s: float
    search_height_m: float
    terminal_height_m: float
    terminal_radius_m: float
    ramp_m: float


@dataclass(frozen=True)
class VelocityField:
    """The approach the dense reward points along (reward.desired_velocity)."""
    lateral_gain_per_s: float
    lateral_speed_max_m_s: float
    descent_gain_per_s: float
    descent_speed_max_m_s: float
    touchdown_sink_m_s: float
    alignment_radius_m: float
    alignment_radius_per_m: float
    lost_view: LostView


@dataclass(frozen=True)
class CommonReward:
    """The dense reward every arm trains on (truth plus the policy's own vision age)."""
    distance_weight_per_m: float
    velocity_weight_per_m_s: float
    time_cost_per_s: float
    control_cost_per_s: float
    tilt_cost_per_s: float
    tilt_cost_onset_deg: float
    field: VelocityField
    version: str = REWARD_VERSION


@dataclass(frozen=True)
class TaskContract:
    """Guide 9.3: one reward and termination function for every arm, and (task v5) the
    one scenario distribution every arm trains and is tested on (scenario.py)."""
    evaluator: str
    supervisor: str
    unsafe_penalty: str
    reward: CommonReward
    scenario: Any  # scenario.CompositionScenario
    version: str = TASK_VERSION


@dataclass(frozen=True)
class OpticalCadence:
    """Thinning of the local plant's 20 Hz frames to Isaac's measured cadence."""
    keep_every_nth_capture: int
    extra_drop_probability: float


@dataclass(frozen=True)
class LocalPlant:
    """The training plant: spatial-reference/12 LocalBackend with three Isaac-measured
    calibrations (SOURCE_AUDIT_KO.md section 2; G0_G4_STATUS_KO.md section 5)."""
    contact_height_m: float
    handover_wait_max_s: float
    optical_cadence: OpticalCadence


@dataclass(frozen=True)
class ModelDesign:
    node_embedding: int
    hops: int
    readout: str
    initial_log_std: float
    heads: tuple[tuple[str, int], ...]
    critic_hidden: int

    def head(self, arm: str) -> int:
        widths = dict(self.heads)
        if arm not in widths:
            raise KeyError(f"unknown arm {arm!r}; registered: {ARMS}")
        return widths[arm]


@dataclass(frozen=True)
class ExperimentProposal:
    registered: bool
    note: str


@dataclass(frozen=True)
class CurrentSemanticsConfig:
    schema_id: str
    deployment_profile: str
    calibration: CameraCalibration
    feature_scales: FeatureScales
    contact: ContactLimits
    action: ActionContract
    model: ModelDesign
    experiment_proposal: ExperimentProposal
    g3: G3Config
    task: TaskContract
    local_plant: LocalPlant
    g4: G4PilotConfig
    isaac_eval: IsaacEvalConfig

    @property
    def z_near_m(self) -> float:
        return self.calibration.z_near_m


def _section(data: Any, keys: tuple[str, ...], where: str) -> dict:
    if not isinstance(data, dict):
        raise ValueError(f"{where} must be a mapping")
    unknown, missing = set(data) - set(keys), set(keys) - set(data)
    problems = ([f"unknown keys {sorted(unknown)}"] if unknown else []) \
        + ([f"missing keys {sorted(missing)}"] if missing else [])
    if problems:
        raise ValueError(f"{where}: " + "; ".join(problems))
    return data


def _positive(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a number")
    value = float(value)
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{where} must be finite and positive")
    return value


def _count(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{where} must be a positive integer")
    return value


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} must be a non-empty string")
    return value


_TOP_KEYS = ("schema_id", "deployment_profile", "camera", "feature_scales", "contact",
             "action", "model", "experiment_proposal", "g3_pilot_seeds", "g3_local", "g3_isaac",
             "task", "local_plant", "g4_pilot", "isaac_eval")
_SCALE_KEYS = tuple(FeatureScales.__dataclass_fields__)
_CONTACT_KEYS = tuple(ContactLimits.__dataclass_fields__)
_MODEL_KEYS = tuple(ModelDesign.__dataclass_fields__)
_ACTION_KEYS = ("frame", "maximum_m_s2", "adapter_max_tilt_deg")
_G3_PROBES = MappingProxyType({"axis_coupling": AxisCouplingProbe, "fov_exit": FovExitProbe,
                               "low_altitude_blind_zone": BlindZoneProbe,
                               "quality_channels": QualityProbe})
_G3_LOCAL_KEYS = tuple(G3LocalConfig.__dataclass_fields__)
if set(_G3_PROBES) != set(_G3_LOCAL_KEYS):
    raise AssertionError("every g3_local section needs its probe dataclass")


def _probe(cls: type, data: Any, where: str) -> Any:
    """A probe dataclass whose every field is a positive float or positive int."""
    fields = cls.__dataclass_fields__
    data = _section(data, tuple(fields), where)
    readers = {"float": _positive, "int": _count}
    return cls(**{name: readers[field.type](data[name], f"{where}.{name}")
                  for name, field in fields.items()})


def _g3_isaac(data: Any, pilot: tuple[int, ...]) -> G3IsaacConfig:
    data = _section(data, ("seeds", "axis_sign_min_abs_correlation"), "g3_isaac")
    seeds = data["seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("g3_isaac.seeds must be a non-empty list")
    seeds = tuple(_count(seed, "g3_isaac.seeds") for seed in seeds)
    if len(set(seeds)) != len(seeds) or not set(seeds) <= set(pilot):
        raise ValueError("g3_isaac.seeds must be distinct g3_pilot_seeds")
    floor = _positive(data["axis_sign_min_abs_correlation"],
                      "g3_isaac.axis_sign_min_abs_correlation")
    if floor >= 1.0:
        raise ValueError("g3_isaac.axis_sign_min_abs_correlation must be below 1")
    return G3IsaacConfig(seeds=seeds, axis_sign_min_abs_correlation=floor)


def _fraction(value: Any, where: str, *, low: float = 0.0, high: float = 1.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not low <= float(value) <= high:
        raise ValueError(f"{where} must lie in [{low}, {high}]")
    return float(value)


def _seed_list(value: Any, where: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{where} must be a non-empty list")
    seeds = tuple(_count(seed, where) for seed in value)
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{where} must be distinct")
    return seeds


def _randomization(data: Any) -> TrainingRandomization:
    data = _section(data, ("handover_own_state_error", "vision_blackout"),
                    "g4_pilot.training_randomization")
    spec = _section(data["handover_own_state_error"],
                    tuple(HandoverOwnStateError.__dataclass_fields__),
                    "g4_pilot.training_randomization.handover_own_state_error")
    where = "g4_pilot.training_randomization.handover_own_state_error."
    error = HandoverOwnStateError(**{key: _positive(spec[key], where + key) for key in spec})
    if not error.tau_min_s <= error.tau_max_s:
        raise ValueError(where + "tau_min_s must not exceed tau_max_s")
    spec = _section(data["vision_blackout"], tuple(VisionBlackout.__dataclass_fields__),
                    "g4_pilot.training_randomization.vision_blackout")
    where = "g4_pilot.training_randomization.vision_blackout."
    blackout = VisionBlackout(
        probability=_fraction(spec["probability"], where + "probability"),
        **{key: _positive(spec[key], where + key) for key in spec if key != "probability"})
    if not (blackout.start_min_s <= blackout.start_max_s
            and blackout.duration_min_s <= blackout.duration_max_s):
        raise ValueError(where + "each minimum must not exceed its maximum")
    return TrainingRandomization(handover_own_state_error=error, vision_blackout=blackout)


def _g4_pilot(data: Any, g3_seeds: tuple[int, ...]) -> G4PilotConfig:
    keys = ("protocol", "arm", "model_seeds", "updates", "episodes_per_update", "curriculum",
            "validate_every", "validation_seeds", "test_seeds", "train_seed_base", "workers",
            "ppo", "criteria", "training_randomization")
    data = _section(data, keys, "g4_pilot")
    if data["protocol"] != G4_PROTOCOL:
        raise ValueError(f"g4_pilot.protocol must be {G4_PROTOCOL!r} (earlier pilots are archived)")
    curriculum = _section(data["curriculum"], tuple(SuccessGatedCurriculum.__dataclass_fields__),
                          "g4_pilot.curriculum")
    if data["arm"] not in ARMS:
        raise ValueError(f"g4_pilot.arm must be one of {ARMS}")
    ppo = _section(data["ppo"], tuple(PPOSettings.__dataclass_fields__), "g4_pilot.ppo")
    criteria = _section(data["criteria"], tuple(G4Criteria.__dataclass_fields__),
                        "g4_pilot.criteria")
    model_seeds = tuple(int(s) for s in data["model_seeds"]) if isinstance(
        data["model_seeds"], list) and data["model_seeds"] else ()
    if not model_seeds or any(isinstance(s, bool) or s < 0 for s in data["model_seeds"]) \
            or len(set(model_seeds)) != len(model_seeds):
        raise ValueError("g4_pilot.model_seeds must be distinct non-negative integers")
    validation = _seed_list(data["validation_seeds"], "g4_pilot.validation_seeds")
    test = _seed_list(data["test_seeds"], "g4_pilot.test_seeds")
    pilot = G4PilotConfig(
        protocol=G4_PROTOCOL, arm=data["arm"], model_seeds=model_seeds,
        updates=_count(data["updates"], "g4_pilot.updates"),
        episodes_per_update=_count(data["episodes_per_update"], "g4_pilot.episodes_per_update"),
        curriculum=SuccessGatedCurriculum(
            start_difficulty=_fraction(curriculum["start_difficulty"],
                                       "g4_pilot.curriculum.start_difficulty"),
            step=_fraction(curriculum["step"], "g4_pilot.curriculum.step", low=1e-6),
            window_updates=_count(curriculum["window_updates"],
                                  "g4_pilot.curriculum.window_updates"),
            promote_at_success=_fraction(curriculum["promote_at_success"],
                                         "g4_pilot.curriculum.promote_at_success")),
        validate_every=_count(data["validate_every"], "g4_pilot.validate_every"),
        validation_seeds=validation, test_seeds=test,
        train_seed_base=_count(data["train_seed_base"], "g4_pilot.train_seed_base"),
        workers=_count(data["workers"], "g4_pilot.workers"),
        ppo=PPOSettings(
            epochs=_count(ppo["epochs"], "g4_pilot.ppo.epochs"),
            minibatch=_count(ppo["minibatch"], "g4_pilot.ppo.minibatch"),
            clip=_positive(ppo["clip"], "g4_pilot.ppo.clip"),
            gae_lambda=_fraction(ppo["gae_lambda"], "g4_pilot.ppo.gae_lambda"),
            policy_lr=_positive(ppo["policy_lr"], "g4_pilot.ppo.policy_lr"),
            value_lr=_positive(ppo["value_lr"], "g4_pilot.ppo.value_lr"),
            entropy_coefficient=_fraction(ppo["entropy_coefficient"],
                                          "g4_pilot.ppo.entropy_coefficient", high=1.0),
            max_grad_norm=_positive(ppo["max_grad_norm"], "g4_pilot.ppo.max_grad_norm"),
            target_kl=_positive(ppo["target_kl"], "g4_pilot.ppo.target_kl")),
        criteria=G4Criteria(
            deterministic_success_min=_fraction(criteria["deterministic_success_min"],
                                                "g4_pilot.criteria.deterministic_success_min"),
            sampled_success_min=_fraction(criteria["sampled_success_min"],
                                          "g4_pilot.criteria.sampled_success_min")),
        training_randomization=_randomization(data["training_randomization"]))
    # Every seed family is disjoint from every other and from all recorded use.
    last_train = pilot.train_seed(max(model_seeds), pilot.updates, 0)
    families = {"validation": set(validation), "test": set(test), "g3": set(g3_seeds)}
    if families["validation"] & families["test"] or (families["validation"]
                                                     | families["test"]) & families["g3"]:
        raise ValueError("g4 validation, g4 test and g3 pilot seeds must be disjoint")
    for seed in (*validation, *test, pilot.train_seed_base, last_train):
        used = [name for name, first, last, _ in AUDITED_SEED_RANGES if first <= seed <= last]
        if used:
            raise ValueError(f"g4 seed {seed} lies in audited range(s) {used}")
    if any(pilot.train_seed_base <= seed <= last_train for seed in (*validation, *test,
                                                                    *g3_seeds)):
        raise ValueError("g4 training seeds overlap validation/test/g3 seeds")
    return pilot


# Seeds used after the fact on 2026-10-28; not registered families, but used, so
# nothing registered may reuse them (G0_G4_STATUS_KO.md sections 9.2, 10, 10.6-10.7).
DIAGNOSTIC_SEED_RANGES = (
    ("pilot-2 and reward-v2 diagnostics", 996001, 996048),
    ("isaac-eval/1 flight and its traced diagnostic re-flight", 994001, 994010),
    ("observation/3 re-acquisition checks on the local plant", 997001, 997048),
    ("isaac-eval/2 flight, whose traces designed observation/3", 994011, 994020),
    ("isaac-eval/3 flight, whose failures motivated pilot 5", 994021, 994030),
    ("pilot-5 design checks on the local plant", 998001, 998048),
    ("isaac-eval/4 flight and the paired pilot-4 re-flight, whose traces motivated pilot 6",
     994031, 994040),
    ("pilot-6 design checks on the local plant", 995001, 995048),
)


def _isaac_eval(data: Any, g4: G4PilotConfig, g3_seeds: tuple[int, ...]) -> IsaacEvalConfig:
    data = _section(data, ("protocol", "seeds", "deterministic_success_min"), "isaac_eval")
    if data["protocol"] != ISAAC_EVAL_PROTOCOL:
        raise ValueError(f"isaac_eval.protocol must be {ISAAC_EVAL_PROTOCOL!r}")
    seeds = _seed_list(data["seeds"], "isaac_eval.seeds")
    last_train = g4.train_seed(max(g4.model_seeds), g4.updates, 0)
    taken = set(g4.validation_seeds) | set(g4.test_seeds) | set(g3_seeds)
    for seed in seeds:
        used = [name for name, first, last, _ in AUDITED_SEED_RANGES if first <= seed <= last]
        used += [name for name, first, last in DIAGNOSTIC_SEED_RANGES if first <= seed <= last]
        if used or seed in taken or g4.train_seed_base <= seed <= last_train:
            raise ValueError(f"isaac_eval seed {seed} is already used ({used or 'registered'})")
    return IsaacEvalConfig(protocol=ISAAC_EVAL_PROTOCOL, seeds=seeds,
                           deterministic_success_min=_fraction(
                               data["deterministic_success_min"],
                               "isaac_eval.deterministic_success_min"))


_FIELD_KEYS = tuple(VelocityField.__dataclass_fields__)
_LOST_VIEW_KEYS = tuple(LostView.__dataclass_fields__)
_REWARD_KEYS = ("version", "distance_weight_per_m", "velocity_weight_per_m_s",
                "time_cost_per_s", "control_cost_per_s",
                "tilt_cost_per_s", "tilt_cost_onset_deg", "field")
TILT_ENVELOPE_DEG = 21.0  # the Evaluator's hard thrust-axis envelope


def _onset(value: Any) -> float:
    onset = _positive(value, "task.reward.tilt_cost_onset_deg")
    if not onset < TILT_ENVELOPE_DEG:
        raise ValueError("task.reward.tilt_cost_onset_deg must lie below the 21 deg envelope")
    return onset


def _reward(data: Any, contact: ContactLimits) -> CommonReward:
    data = _section(data, _REWARD_KEYS, "task.reward")
    if data["version"] != REWARD_VERSION:
        raise ValueError(f"task.reward.version must be {REWARD_VERSION!r}")
    field = _section(data["field"], _FIELD_KEYS, "task.reward.field")
    lost = _section(field["lost_view"], _LOST_VIEW_KEYS, "task.reward.field.lost_view")
    lost = LostView(**{key: _positive(lost[key], f"task.reward.field.lost_view.{key}")
                       for key in _LOST_VIEW_KEYS})
    field = VelocityField(lost_view=lost,
                          **{key: _positive(field[key], f"task.reward.field.{key}")
                             for key in _FIELD_KEYS if key != "lost_view"})
    # The field must end in a contact the verdict accepts: sink inside the
    # touchdown limit, descent permitted only inside the pad.
    if not field.touchdown_sink_m_s < contact.z_speed_m_s:
        raise ValueError("task.reward.field.touchdown_sink_m_s must lie below the "
                         "touchdown vertical speed limit")
    if not field.alignment_radius_m < contact.pad_radius_m:
        raise ValueError("task.reward.field.alignment_radius_m must lie inside the pad")
    # The blind last stretch must cover every descent the field itself asks for at the
    # terminal height, and the search height must lie above it.
    if not lost.after_s < lost.full_after_s:
        raise ValueError("task.reward.field.lost_view.after_s must lie below full_after_s")
    if not lost.terminal_height_m < lost.search_height_m:
        raise ValueError("task.reward.field.lost_view.terminal_height_m must lie below "
                         "search_height_m")
    if not lost.terminal_radius_m >= (field.alignment_radius_m + field.alignment_radius_per_m
                                      * lost.terminal_height_m):
        raise ValueError("task.reward.field.lost_view.terminal_radius_m must cover the "
                         "alignment radius at the terminal height")
    return CommonReward(
        distance_weight_per_m=_positive(data["distance_weight_per_m"],
                                        "task.reward.distance_weight_per_m"),
        velocity_weight_per_m_s=_positive(data["velocity_weight_per_m_s"],
                                          "task.reward.velocity_weight_per_m_s"),
        time_cost_per_s=_positive(data["time_cost_per_s"], "task.reward.time_cost_per_s"),
        control_cost_per_s=_positive(data["control_cost_per_s"],
                                     "task.reward.control_cost_per_s"),
        tilt_cost_per_s=_positive(data["tilt_cost_per_s"], "task.reward.tilt_cost_per_s"),
        tilt_cost_onset_deg=_onset(data["tilt_cost_onset_deg"]),
        field=field)


def _scenario(data: Any, contact: ContactLimits) -> Any:
    from .scenario import SCENARIO_VERSION, CompositionScenario
    data = _section(data, ("version", "bins", "offset_radius_m", "held_out"), "task.scenario")
    if data["version"] != SCENARIO_VERSION:
        raise ValueError(f"task.scenario.version must be {SCENARIO_VERSION!r}")
    bins = data["bins"]
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 3:
        raise ValueError("task.scenario.bins must be an integer >= 3")
    radius = data["offset_radius_m"]
    if (not isinstance(radius, list) or len(radius) != 2
            or not 0 < _positive(radius[0], "task.scenario.offset_radius_m") < _positive(
                radius[1], "task.scenario.offset_radius_m")):
        raise ValueError("task.scenario.offset_radius_m must be [low, high] with 0 < low < high")
    held = data["held_out"]
    if not isinstance(held, list) or not all(
            isinstance(c, list) and len(c) == 2 and all(
                isinstance(v, int) and not isinstance(v, bool) and 0 <= v < bins for v in c)
            for c in held):
        raise ValueError("task.scenario.held_out must list [bearing bin, heading bin] pairs")
    held = tuple(tuple(c) for c in held)
    # A permutation: every bearing bin and every heading bin is held out exactly once,
    # so both marginals stay uniform over the trained combinations ...
    if sorted(a for a, _ in held) != list(range(bins)) or \
            sorted(b for _, b in held) != list(range(bins)):
        raise ValueError("task.scenario.held_out must hold out every bearing bin and every "
                         "heading bin exactly once")
    # ... and not a cyclic shift, so every relative-geometry class (B - A) keeps
    # trained combinations: the test is composition, not an unseen relative geometry.
    if len({(b - a) % bins for a, b in held}) == 1:
        raise ValueError("task.scenario.held_out must not be a cyclic shift (one relative "
                         "geometry class would never be trained)")
    return CompositionScenario(bins=bins, offset_radius_m=(float(radius[0]), float(radius[1])),
                               held_out=held)


def _task(data: Any, contact: ContactLimits) -> TaskContract:
    data = _section(data, ("evaluator", "supervisor", "unsafe_penalty", "reward", "scenario"),
                    "task")
    for key, expected in (("evaluator", TASK_EVALUATOR), ("supervisor", "none"),
                          ("unsafe_penalty", "nominal")):
        if data[key] != expected:
            raise ValueError(f"task.{key} must be {expected!r}: the only registered choice")
    return TaskContract(evaluator=TASK_EVALUATOR, supervisor="none", unsafe_penalty="nominal",
                        reward=_reward(data["reward"], contact),
                        scenario=_scenario(data["scenario"], contact))


def _local_plant(data: Any) -> LocalPlant:
    data = _section(data, ("contact_height_m", "handover_wait_max_s", "optical_cadence"),
                    "local_plant")
    cadence = _section(data["optical_cadence"],
                       ("keep_every_nth_capture", "extra_drop_probability"),
                       "local_plant.optical_cadence")
    drop = cadence["extra_drop_probability"]
    if isinstance(drop, bool) or not isinstance(drop, (int, float)) or not 0 <= drop < 1:
        raise ValueError("local_plant.optical_cadence.extra_drop_probability must be in [0, 1)")
    height = _positive(data["contact_height_m"], "local_plant.contact_height_m")
    if height >= 0.16:
        raise ValueError("local_plant.contact_height_m must stay below the 0.16 m camera mount")
    return LocalPlant(
        contact_height_m=height,
        handover_wait_max_s=_positive(data["handover_wait_max_s"],
                                      "local_plant.handover_wait_max_s"),
        optical_cadence=OpticalCadence(
            keep_every_nth_capture=_count(cadence["keep_every_nth_capture"],
                                          "local_plant.optical_cadence.keep_every_nth_capture"),
            extra_drop_probability=float(drop)))


def _pilot_seeds(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("g3_pilot_seeds must be a non-empty list")
    seeds = tuple(_count(seed, "g3_pilot_seeds") for seed in value)
    if len(set(seeds)) != len(seeds):
        raise ValueError("g3_pilot_seeds must be distinct")
    used = [(seed, name) for seed in seeds for name, first, last, _ in AUDITED_SEED_RANGES
            if first <= seed <= last]
    if used:
        raise ValueError(f"g3_pilot_seeds overlap audited seed ranges: {used}")
    return seeds


def load_config(path: str | Path | None = None) -> CurrentSemanticsConfig:
    """Load and validate the registered YAML; every key is required, none is extra."""
    path = DEFAULT_CONFIG_PATH if path is None else Path(path)
    data = _section(yaml.safe_load(Path(path).read_text(encoding="utf-8")), _TOP_KEYS, "config")
    if data["schema_id"] != SCHEMA_ID:
        raise ValueError(f"schema_id must be {SCHEMA_ID!r}, got {data['schema_id']!r}")
    if data["deployment_profile"] != DEPLOYMENT_PROFILE:
        raise ValueError(f"deployment_profile must be {DEPLOYMENT_PROFILE!r}")

    camera = _section(data["camera"], ("z_near_m",), "camera")
    scales = _section(data["feature_scales"], _SCALE_KEYS, "feature_scales")
    contact = _section(data["contact"], _CONTACT_KEYS, "contact")
    action = _section(data["action"], _ACTION_KEYS, "action")
    model = _section(data["model"], _MODEL_KEYS, "model")
    proposal = _section(data["experiment_proposal"], ("registered", "note"),
                        "experiment_proposal")

    if action["frame"] != "world_enu":
        raise ValueError("action.frame must be world_enu")
    maximum = action["maximum_m_s2"]
    if not isinstance(maximum, list) or len(maximum) != ACTION_DIMENSION:
        raise ValueError(f"action.maximum_m_s2 needs {ACTION_DIMENSION} values")
    tilt = _positive(action["adapter_max_tilt_deg"], "action.adapter_max_tilt_deg")
    if tilt > ADAPTER_MAX_TILT_LIMIT_DEG:
        raise ValueError(f"action.adapter_max_tilt_deg must not exceed "
                         f"{ADAPTER_MAX_TILT_LIMIT_DEG} (the adapter's own bound)")
    g3_local = _section(data["g3_local"], _G3_LOCAL_KEYS, "g3_local")
    heads = _section(model["heads"], ARMS, "model.heads")
    if model["readout"] != NODE_ORDER[-1]:
        raise ValueError(f"model.readout must be {NODE_ORDER[-1]}")
    log_std = model["initial_log_std"]
    if isinstance(log_std, bool) or not isinstance(log_std, (int, float)) \
            or not math.isfinite(float(log_std)):
        raise ValueError("model.initial_log_std must be a finite number")
    if proposal["registered"] is not False:
        raise ValueError("experiment_proposal.registered must stay false: registration "
                         "goes through the gated CLI, not a config edit")

    config = CurrentSemanticsConfig(
        schema_id=SCHEMA_ID,
        deployment_profile=DEPLOYMENT_PROFILE,
        calibration=nominal_v12(_positive(camera["z_near_m"], "camera.z_near_m")),
        feature_scales=FeatureScales(**{key: _positive(scales[key], f"feature_scales.{key}")
                                        for key in _SCALE_KEYS}),
        contact=(limits := ContactLimits(
            **{key: _positive(contact[key], f"contact.{key}")
               for key in _CONTACT_KEYS if key != "status"},
            status=_text(contact["status"], "contact.status"))),
        action=ActionContract(
            frame="world_enu",
            maximum_m_s2=tuple(_positive(v, "action.maximum_m_s2") for v in maximum),
            adapter_max_tilt_deg=tilt),
        model=ModelDesign(
            node_embedding=_count(model["node_embedding"], "model.node_embedding"),
            hops=_count(model["hops"], "model.hops"),
            readout=model["readout"],
            initial_log_std=float(log_std),
            heads=tuple((arm, _count(heads[arm], f"model.heads.{arm}")) for arm in ARMS),
            critic_hidden=_count(model["critic_hidden"], "model.critic_hidden")),
        experiment_proposal=ExperimentProposal(
            registered=False, note=_text(proposal["note"], "experiment_proposal.note")),
        g3=G3Config(
            pilot_seeds=(pilot := _pilot_seeds(data["g3_pilot_seeds"])),
            isaac=_g3_isaac(data["g3_isaac"], pilot),
            local=G3LocalConfig(**{
                key: _probe(_G3_PROBES[key], g3_local[key], f"g3_local.{key}")
                for key in _G3_LOCAL_KEYS})),
        task=_task(data["task"], limits),
        local_plant=_local_plant(data["local_plant"]),
        g4=(g4 := _g4_pilot(data["g4_pilot"], pilot)),
        isaac_eval=_isaac_eval(data["isaac_eval"], g4, pilot))
    totals = {arm: counts["total"]
              for arm, counts in design_parameter_counts(config.model).items()}
    if totals != DESIGN_PARAMETER_TOTALS:
        raise ValueError(f"model widths break the registered parameter budget: "
                         f"{totals} != {DESIGN_PARAMETER_TOTALS}")
    return config


# ------------------------------------------------------------- model design

def design_parameter_counts(model: ModelDesign) -> dict[str, dict[str, int]]:
    """Guide 8.4's design formula; models.py must measure the same trainable counts.

    Encoder: Linear(10, d) + one shared relational layer (R*d*d weights,
    R relation biases, two attention vectors and a bias) + bias-free readout.
    """
    d = model.node_embedding

    def mlp(inputs: int, hidden: int, outputs: int) -> int:
        return inputs * hidden + hidden + hidden * outputs + outputs

    def encoder(relations: int) -> int:
        return (NODE_INPUT_DIMENSION * d + d) + (relations * (d * d + 1) + 3 * d) + d * d

    critic = mlp(OBSERVATION_DIMENSION, model.critic_hidden, 1)
    inputs = {"raw21": OBSERVATION_DIMENSION, "flat44": FEATURE_DIMENSION,
              "raw21+graph": OBSERVATION_DIMENSION + d}
    counts = {}
    for arm in ARMS:
        enc = encoder(ENCODER_RELATIONS[arm]) if arm in ENCODER_RELATIONS else 0
        head = mlp(inputs[ARM_ACTOR_INPUT[arm]], model.head(arm), ACTION_DIMENSION)
        counts[arm] = {"encoder": enc, "actor_head": head, "critic": critic,
                       "log_std": ACTION_DIMENSION,
                       "total": enc + head + critic + ACTION_DIMENSION}
    return counts


# The counts above fix shapes, not equations: this is hashed into the model part
# so a changed layer, activation or initialisation cannot keep its
# algorithm_hash.  models.py reads the numeric entries.
MODEL_ARCHITECTURE = MappingProxyType({
    "encoder": ("h = tanh(Linear(node_input, d) x) per node; ONE relational layer applied "
                "`hops` times; g = Linear(d, d, bias=False) h[readout]"),
    "relational_layer": ("edge j -> i of relation r: m_ij = W_r h_j; e_ij = leaky_relu("
                         "att_src . m_ij + att_dst . h_i + relation_bias_r); alpha = softmax of e "
                         "over the incoming edges of i; h_i' = tanh(sum_j alpha_ij m_ij + bias)"),
    "attention_negative_slope": 0.2,
    "padding": "not masked; zero padding is checked where arrays enter the model",
    "actor_head": "Linear(input, head) tanh Linear(head, 3) -> latent mean",
    "critic": "Linear(raw21, critic_hidden) tanh Linear(critic_hidden, 1)",
    "log_std": "global, one per axis, unclamped",
    "initialisation": ("one private torch.Generator per (seed, component); Linear weight and "
                       "bias U(-1/sqrt(fan_in), 1/sqrt(fan_in)); W_r N(0, 1/d) (LeCun: the "
                       "shared layer applied twice neither shrinks nor grows node states); "
                       "att_src, att_dst N(0, attention_init_std^2); relation_bias and layer "
                       "bias 0; log_std = initial_log_std"),
    # Was N(0, 0.1^2) for W_r too: measured on 90 local-plant decisions, that
    # left g8 at RMS 0.004-0.006 against raw21 0.50 (two hops each shrink by
    # ~0.1*sqrt(8)); N(0, 1/d) gives 0.05-0.07.  Chosen by the rule, not tuned
    # to a target magnitude.
    "relational_weight_init": "normal(0, 1/node_embedding)",
    "attention_init_std": 0.1,
    # G4 pilot 1 (2026-10-28): the U(+-1/sqrt(fan_in)) output layer gave the initial
    # mean action a constant bias that integrates (net acceleration) into the 20 m /
    # 60 m envelope within ~15 s, so the curriculum's easy rungs were spent removing
    # drift.  The actor's OUTPUT layer is scaled by this factor and its bias zeroed,
    # so every arm starts from a mean near zero; hidden layers are unchanged.
    "actor_output_init_scale": 0.01,
})


# ----------------------------------------------------------------- manifest

PLUGGABLE_PARTS = ("features", "graph")


def _extension_part(name: str, config: CurrentSemanticsConfig) -> Any:
    """``current_semantics.<name>.manifest_part(config)`` once that module exists.

    Extension point for the later stages: ``features.py`` and ``graph.py``
    each define ``manifest_part(config) -> dict`` of plain JSON types.  Until
    they do, the part is None and every hash that depends on it is None, so
    an incomplete contract can never carry a complete-looking hash.
    """
    module_name = f"{__package__}.{name}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        return None
    hook = getattr(module, "manifest_part", None)
    return None if hook is None else _plain(hook(config))


def contract_manifest(config: CurrentSemanticsConfig) -> dict[str, Any]:
    observation = observation_manifest()
    calibration = config.calibration.to_dict()
    scales = _plain(config.feature_scales)
    contact = _plain({**asdict(config.contact), "tilt_rad": config.contact.tilt_rad,
                      "rate_rad_s": config.contact.rate_rad_s,
                      "definitions": CONTACT_DEFINITIONS})
    action = _plain(config.action)
    model = _plain({"version": MODEL_VERSION, "design": config.model,
                    "architecture": MODEL_ARCHITECTURE,
                    "arms": ARMS, "actor_input": ARM_ACTOR_INPUT,
                    "encoder_relations": ENCODER_RELATIONS,
                    "critic": "common raw21 critic for every arm",
                    "design_parameter_counts": design_parameter_counts(config.model)})
    # G3 implementation-check constants: hashed on their own and kept out of
    # algorithm_hash, so retuning a check never re-labels a model.
    g3 = _plain({"pilot_seeds": config.g3.pilot_seeds,
                 "pilot_seed_note": config.g3.pilot_seed_note,
                 "audited_seed_ranges": config.g3.audited_seed_ranges,
                 "local": config.g3.local, "isaac": config.g3.isaac})
    # The reward/termination function and the plant it is scored on: what a
    # checkpoint was trained against, hashed apart from the algorithm so a
    # task change is never mistaken for a model change (and vice versa).
    from ..landing.terminal import (REFERENCE_DISCOUNT_TAU_S, REFERENCE_MISSION_HORIZON_S,
                                    REFERENCE_TERMINAL_BONUS)
    task = _plain({"contract": config.task, "definition": TASK_DEFINITION,
                   "terminal_bonus": REFERENCE_TERMINAL_BONUS,
                   "discount_tau_s": REFERENCE_DISCOUNT_TAU_S,
                   "horizon_s": REFERENCE_MISSION_HORIZON_S,
                   "flight_schema": DEPLOYMENT_PROFILE,
                   "local_plant": config.local_plant})
    features = _extension_part("features", config)
    graph = _extension_part("graph", config)
    missing = [name for name, part in zip(PLUGGABLE_PARTS, (features, graph)) if part is None]

    hashes = {
        "observation_hash": digest(observation),
        "calibration_hash": digest(calibration),
        "action_hash": digest(action),
        "feature_hash": None if features is None else digest(
            {"version": FEATURE_VERSION, "features": features, "calibration": calibration,
             "feature_scales": scales, "contact": contact}),
        "graph_hash": None if graph is None else digest(
            {"version": GRAPH_VERSION, "graph": graph}),
        "g3_hash": digest(g3),
        "task_hash": digest(task),
        # The G4 pilot's protocol (PPO settings, budget, seeds, criteria): its own
        # hash, so a pilot result is only read under the protocol it ran.
        "g4_hash": digest(_plain(config.g4)),
        # The Isaac evaluation protocol of the selected checkpoints.
        "isaac_eval_hash": digest(_plain(config.isaac_eval)),
    }
    hashes["algorithm_hash"] = None if missing else digest(
        {"schema_id": SCHEMA_ID, "observation_hash": hashes["observation_hash"],
         "feature_hash": hashes["feature_hash"], "graph_hash": hashes["graph_hash"],
         "action_hash": hashes["action_hash"], "model": model})
    return _plain({
        "schema_id": SCHEMA_ID,
        "versions": {"observation": OBSERVATION_VERSION, "features": FEATURE_VERSION,
                     "graph": GRAPH_VERSION, "action": ACTION_VERSION,
                     "models": MODEL_VERSION},
        "observation": observation,
        "features": features,
        "graph": graph,
        "calibration": calibration,
        "feature_scales": scales,
        "contact": contact,
        "action": action,
        "model": model,
        "experiment_proposal": config.experiment_proposal,
        "g3": g3,
        "task": task,
        "g4_pilot": _plain(config.g4),
        "isaac_eval": _plain(config.isaac_eval),
        "provenance": {"baseline_3d_sha": BASELINE_3D_SHA, "provenance_2d": PROVENANCE_2D},
        "complete": not missing,
        "missing_parts": missing,
        **hashes,
    })


def _plain(value: Any) -> Any:
    """Dataclasses, tuples and numpy scalars to the JSON types ``digest`` accepts."""
    if hasattr(value, "__dataclass_fields__"):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value
