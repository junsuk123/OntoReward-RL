"""Versioned contracts for the MATLAB direct-policy port.

The port deliberately does not reuse the existing ``landing`` packet or graph:
those implement the reference-v28 nine-node study, while the pinned MATLAB
source uses a 12-value observation and a seven-node graph.  Hashes are split so
an algorithm change is not confused with a task or execution-backend change.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any

from ..landing.terminal import (
    REFERENCE_DISCOUNT_TAU_S,
    REFERENCE_MISSION_HORIZON_S,
    REFERENCE_TERMINAL_BONUS,
)


SOURCE_SHA = "608225805fa447c2f8e2756e33378c32fd975797"
TARGET_BASELINE_SHA = "3fc2a9fa659f0f933706b5da6918aafec6239a20"
PORT_VERSION = "matlab-direct-policy/1"


def _canonical(value: Any) -> bytes:
    def normalise(item: Any) -> Any:
        if hasattr(item, "__dataclass_fields__"):
            return normalise(asdict(item))
        if isinstance(item, dict):
            return {str(k): normalise(v) for k, v in sorted(item.items())}
        if isinstance(item, (list, tuple)):
            return [normalise(v) for v in item]
        if isinstance(item, float):
            if not (-float("inf") < item < float("inf")):
                raise ValueError("contract values must be finite")
            return float(item)
        return item

    return json.dumps(normalise(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True)
class ObservationField:
    name: str
    source: str
    scale: float | None = None


@dataclass(frozen=True)
class ObservationSchema:
    version: str
    dimension: int
    fields: tuple[ObservationField, ...]

    def __post_init__(self) -> None:
        if self.dimension != len(self.fields):
            raise ValueError("observation dimension does not match field count")
        names = [item.name for item in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("observation field names must be unique")


@dataclass(frozen=True)
class GraphSchema:
    version: str
    nodes: tuple[str, ...]
    relations: tuple[str, ...]
    edges: tuple[tuple[str, str, str], ...]
    feature_names: tuple[str, ...]
    readout_groups: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        node_set, relation_set = set(self.nodes), set(self.relations)
        if len(node_set) != len(self.nodes):
            raise ValueError("graph node names must be unique")
        for source, target, relation in self.edges:
            if source not in node_set or target not in node_set:
                raise ValueError(f"unknown graph endpoint: {source}->{target}")
            if relation not in relation_set:
                raise ValueError(f"unknown graph relation: {relation}")


@dataclass(frozen=True)
class ActionSpec:
    version: str
    names: tuple[str, ...]
    maximum: tuple[float, ...]
    distribution: str = "latent_normal_tanh_scale"

    def __post_init__(self) -> None:
        if len(self.names) != len(self.maximum) or min(self.maximum) <= 0:
            raise ValueError("invalid action limits")


@dataclass(frozen=True)
class RewardSpec:
    version: str = "matlab-reward-v5/1"
    reference_time_s: float = REFERENCE_MISSION_HORIZON_S
    discount_tau_s: float = REFERENCE_DISCOUNT_TAU_S
    goal_length_x_m: float = 3.0
    goal_length_h_m: float = 4.0
    goal_horizontal_share: float = 0.65
    goal_huber_delta: float = 1.0
    goal_weight: float = 2.0
    view_weight: float = 1.0
    control_weight: float = 0.25
    readiness_weight: float = 8.0
    readiness_height_m: float = 1.0
    potential_weight: float = 4.0
    velocity_potential_weight: float = 4.0
    velocity_length_m_s: float = 1.0
    approach_position_rate: float = 0.35
    target_relative_speed_m_s: float = 1.0
    vertical_potential_weight: float = 4.0
    vertical_speed_length_m_s: float = 0.5
    target_descent_speed_m_s: float = 0.4
    vertical_position_rate: float = 0.8
    terminal: tuple[tuple[str, float], ...] = REFERENCE_TERMINAL_BONUS


@dataclass(frozen=True)
class TerminationSpec:
    version: str = "matlab-direct-termination/1"
    mission_time_s: float = REFERENCE_MISSION_HORIZON_S
    touchdown_height_m: float = 0.04
    pad_half_length_m: float = 0.5
    pad_half_width_m: float = 0.5
    touchdown_xy_speed_m_s: float = 0.35
    touchdown_z_speed_m_s: float = 0.30
    touchdown_tilt_rad: float = 0.08726646259971647
    touchdown_rate_rad_s: float = 0.17453292519943295
    ceiling_m: float = 25.0
    floor_m: float = 0.0


@dataclass(frozen=True)
class TrainingSpec:
    version: str = "matlab-direct-ppo-v1/1"
    updates: int = 750
    episodes_per_update: int = 6
    epochs: int = 8
    minibatch_size: int = 256
    policy_lr: float = 2e-4
    value_lr: float = 5e-4
    encoder_lr: float = 1e-4
    entropy_weight: float = 0.004
    initial_log_std: float = -1.0
    minimum_log_std: float = -2.5
    clip_ratio: float = 0.2
    gae_lambda: float = 0.95
    hidden_size: int = 48
    graph_hidden_size: int = 16
    graph_output_size: int = 32
    evaluation_interval: int = 25
    pretraining: bool = False
    frozen_backbone: bool = False
    graph_warmup_fraction: float = 0.0


@dataclass(frozen=True)
class StableTrainingSpec(TrainingSpec):
    """Opt-in Isaac recipe that keeps sampled rollouts physically reachable.

    The source-parity profile above remains byte-for-byte representable.  This
    profile is a new algorithm contract: it removes the random network's
    vertical bias with a data-free analytic tracking prior, batches
    complete episodes, lowers exploration noise and bounds every accepted
    actor step by its post-update KL.  It uses no BC or DAgger data.
    """
    version: str = "matlab-direct-ppo-stable-v14/1"
    episodes_per_update: int = 12
    policy_lr: float = 2e-5
    encoder_lr: float = 2e-5
    entropy_weight: float = 0.0
    initial_log_std: float = -4.0
    minimum_log_std: float = -5.5
    pretraining: bool = False
    zero_last_layer_initialization: bool = True
    analytic_tracking_prior: bool = True
    target_kl: float = 0.02
    enforce_target_kl: bool = True
    maximum_log_std: float = -4.0
    tracking_kp: float = 0.5
    tracking_kd: float = 0.5
    tracking_ki: float = 0.1
    tracking_horizontal_cap_m_s2: float = 0.75
    tracking_descent_gate_m: float = 0.35
    tracking_max_vision_age_s: float = 2.0
    tracking_recovery_height_m: float = 1.2
    tracking_terminal_sink_m_s: float = 0.18
    tracking_handover_s: float = 2.0
    tracking_handover_climb_m_s2: float = 0.2
    tracking_stale_full_authority_s: float = 8.0
    tracking_stale_horizontal_cap_m_s2: float = 0.3
    tracking_stale_vertical_floor_m_s2: float = 0.4


@dataclass(frozen=True)
class ScenarioManifest:
    version: str = "matlab-planar-scenario/1"
    base_seed: int = 20261002
    train_seeds: tuple[int, int] = (1, 2000)
    validation_seeds: tuple[int, int] = (2001, 2200)
    test_seeds: tuple[int, int] = (3001, 3200)
    stress_seeds: tuple[int, int] = (9001, 9200)
    physics_dt_s: float = 0.01
    policy_dt_s: float = 0.10
    height_range_m: tuple[float, float] = (4.0, 8.0)
    pad_height_m: float = 0.6


@dataclass(frozen=True)
class BackendSpec:
    name: str
    version: str
    frame: str = "ENU/FLU; PX4 adapter NED/FRD"
    policy_rate_hz: float = 10.0


@dataclass(frozen=True)
class RuntimeSafetySpec:
    profile: str
    rewrites_policy_action: bool
    emergency_protection: bool = True
    command_owner: str = "matlab-port"


@dataclass(frozen=True)
class ContractBundle:
    observation: ObservationSchema
    graph: GraphSchema
    action: ActionSpec
    reward: RewardSpec
    termination: TerminationSpec
    training: TrainingSpec
    scenarios: ScenarioManifest
    backend: BackendSpec
    safety: RuntimeSafetySpec
    source_sha: str = SOURCE_SHA
    target_baseline_sha: str = TARGET_BASELINE_SHA
    port_version: str = PORT_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def algorithm_hash(self) -> str:
        return digest({"observation": self.observation, "graph": self.graph,
                       "action": self.action, "training": self.training})

    @property
    def task_contract_hash(self) -> str:
        return digest({"observation": self.observation, "action": self.action,
                       "reward": self.reward, "termination": self.termination,
                       "scenarios": self.scenarios})

    @property
    def execution_hash(self) -> str:
        return digest({"backend": self.backend, "safety": self.safety,
                       "source_sha": self.source_sha,
                       "target_baseline_sha": self.target_baseline_sha})

    def manifest(self) -> dict[str, Any]:
        value = asdict(self)
        value.update(algorithm_hash=self.algorithm_hash,
                     task_contract_hash=self.task_contract_hash,
                     execution_hash=self.execution_hash)
        return value


PLANAR_FIELDS = (
    ObservationField("relative_x", "estimated_pad_x-own_x", 3.0),
    ObservationField("relative_height", "own_z-(estimated_pad_z+pad_offset_z)", 8.0),
    ObservationField("relative_vx", "estimated_pad_vx-own_vx", 10.0),
    ObservationField("ugv_vx", "estimated_pad_vx", 10.0),
    ObservationField("drone_vz", "own_vz", 1.5),
    ObservationField("drone_sinTheta", "sin(own_pitch)"),
    ObservationField("drone_cosTheta", "cos(own_pitch)"),
    ObservationField("drone_pitchRate", "own_pitch_rate", 1.5707963267948966),
    ObservationField("ugv_visionUpdated", "marker_update_at_decision"),
    ObservationField("ugv_visionAge", "decision_stamp-last_capture_stamp", 3.0),
    ObservationField("drone_navigationValid", "causal_navigation_valid"),
    ObservationField("drone_navigationAge", "decision_stamp-navigation_stamp", 3.0),
)

SPATIAL_FIELDS = (
    ObservationField("relative_x", "estimated_pad_x-own_x", 3.0),
    ObservationField("relative_y", "estimated_pad_y-own_y", 3.0),
    ObservationField("relative_height", "own_z-estimated_pad_z", 8.0),
    ObservationField("relative_vx", "estimated_pad_vx-own_vx", 10.0),
    ObservationField("relative_vy", "estimated_pad_vy-own_vy", 10.0),
    ObservationField("ugv_vx", "estimated_pad_vx", 10.0),
    ObservationField("ugv_vy", "estimated_pad_vy", 10.0),
    ObservationField("drone_vz", "own_vz", 1.5),
    ObservationField("drone_sinPhi", "sin(own_roll)"),
    ObservationField("drone_cosPhi", "cos(own_roll)"),
    ObservationField("drone_sinTheta", "sin(own_pitch)"),
    ObservationField("drone_cosTheta", "cos(own_pitch)"),
    ObservationField("drone_sinPsi", "sin(own_yaw)"),
    ObservationField("drone_cosPsi", "cos(own_yaw)"),
    ObservationField("drone_rollRate", "own_body_rate_p", 1.5707963267948966),
    ObservationField("drone_pitchRate", "own_body_rate_q", 1.5707963267948966),
    ObservationField("drone_yawRate", "own_body_rate_r", 1.5707963267948966),
    ObservationField("ugv_visionUpdated", "marker_update_at_decision"),
    ObservationField("ugv_visionAge", "decision_stamp-last_capture_stamp", 3.0),
    ObservationField("drone_navigationValid", "causal_navigation_valid"),
    ObservationField("drone_navigationAge", "decision_stamp-navigation_stamp", 3.0),
)

NODES = ("RelativePosition", "RelativeVelocity", "PadVelocity", "VerticalMotion",
         "Attitude", "VisionQuality", "NavigationQuality")
RELATIONS = ("informs", "conditions", "couples", "self")
SEMANTIC_EDGES = (
    ("PadVelocity", "RelativeVelocity", "informs"),
    ("VisionQuality", "RelativePosition", "conditions"),
    ("VisionQuality", "RelativeVelocity", "conditions"),
    ("NavigationQuality", "VerticalMotion", "conditions"),
    ("NavigationQuality", "Attitude", "conditions"),
    ("RelativePosition", "VisionQuality", "couples"),
    ("Attitude", "VisionQuality", "couples"),
    ("RelativeVelocity", "RelativePosition", "informs"),
    ("VerticalMotion", "RelativePosition", "couples"),
)


def graph_schema(dimension: int = 2, *, tracking_bias: bool = False) -> GraphSchema:
    if dimension not in (2, 3):
        raise ValueError("dimension must be 2 or 3")
    features = (("primary", "signed", "secondary", "validity", "age", "typeId")
                if dimension == 2 else
                ("primary", "signed_x", "secondary_x", "signed_y", "secondary_y",
                 "validity", "age", "typeId"))
    if tracking_bias:
        features += (("tracking_bias_x",) if dimension == 2 else
                     ("tracking_bias_x", "tracking_bias_y"))
        features += ("handover_progress",)
    self_edges = tuple((node, node, "self") for node in NODES)
    return GraphSchema(
        version=(f"matlab-minimal-observation-rgat-v4/{dimension}d"
                 if tracking_bias else
                 f"matlab-minimal-observation-rgat-v3/{dimension}d"),
        nodes=NODES, relations=RELATIONS, edges=SEMANTIC_EDGES + self_edges,
        feature_names=features, readout_groups=tuple((i,) for i in range(len(NODES))))


def make_contract(dimension: int = 2, *, backend: str = "local",
                  safety_profile: str = "direct",
                  training_profile: str = "source") -> ContractBundle:
    if dimension not in (2, 3):
        raise ValueError("dimension must be 2 or 3")
    if safety_profile not in ("direct", "shielded"):
        raise ValueError("safety_profile must be direct or shielded")
    if training_profile not in ("source", "stable"):
        raise ValueError("training_profile must be source or stable")
    fields = PLANAR_FIELDS if dimension == 2 else SPATIAL_FIELDS
    if training_profile == "stable":
        fields += ((ObservationField("tracking_bias_x", "integral(relative_x)", 3.0),)
                   if dimension == 2 else
                   (ObservationField("tracking_bias_x", "integral(relative_x)", 3.0),
                    ObservationField("tracking_bias_y", "integral(relative_y)", 3.0)))
        fields += (ObservationField(
            "handover_progress", "clip(episode_elapsed/tracking_handover_s,0,1)"),)
    observation = ObservationSchema(
        version=(("matlab-planar-observation/2" if dimension == 2
                  else "spatial-direct-observation/2")
                 if training_profile == "stable" else
                 ("matlab-planar-observation/1" if dimension == 2
                  else "spatial-direct-observation/1")),
        dimension=len(fields), fields=fields)
    action = ActionSpec(
        version=f"matlab-direct-action/{dimension}d",
        names=(("a_x", "a_z") if dimension == 2 else ("a_x", "a_y", "a_z")),
        maximum=((2.5, 2.0) if dimension == 2 else (2.5, 2.5, 2.0)))
    return ContractBundle(
        observation=observation,
        graph=graph_schema(dimension, tracking_bias=training_profile == "stable"),
        action=action, reward=RewardSpec(), termination=TerminationSpec(),
        training=(TrainingSpec() if training_profile == "source"
                  else StableTrainingSpec()),
        scenarios=ScenarioManifest(),
        backend=BackendSpec(name=backend, version=f"{backend}-matlab-port/1"),
        safety=RuntimeSafetySpec(profile=safety_profile,
                                 rewrites_policy_action=safety_profile == "shielded"),
        metadata={"dimension": dimension, "running_normalization": False,
                  "training_profile": training_profile,
                  "base_observation_dimension": (12 if dimension == 2 else 21)})


def validate_checkpoint_metadata(payload: dict[str, Any], contract: ContractBundle) -> None:
    required = {"source_sha", "target_baseline_sha", "algorithm_hash", "method_hash",
                "task_contract_hash", "execution_hash", "training_step",
                "optimizer_state", "rng_state"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"checkpoint metadata missing: {sorted(missing)}")
    expected = contract.manifest()
    for key in ("source_sha", "target_baseline_sha", "algorithm_hash",
                "task_contract_hash", "execution_hash"):
        if payload[key] != expected[key]:
            raise ValueError(f"checkpoint {key} is incompatible")
