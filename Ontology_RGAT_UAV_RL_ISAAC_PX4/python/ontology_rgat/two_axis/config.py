"""Typed configuration for the two-axis planar landing experiment."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config/experiments/two_axis_context_rgat_comparison.yaml"
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "config/observation/causal_packet_v2.json"
REFERENCE_CONFIG_PATH = PROJECT_ROOT / "config/experiments/two_axis_reference_v28_active.yaml"


def _pair(value: Any, name: str) -> tuple[float, float]:
    values = tuple(float(v) for v in value)
    if len(values) != 2 or not values[0] < values[1]:
        raise ValueError(f"{name} must contain an increasing [low, high] pair")
    return values


@dataclass(frozen=True)
class TimingConfig:
    physics_dt_s: float = 0.01
    policy_dt_s: float = 0.1
    mission_duration_limit_s: float = 70.0
    discount_time_constant_s: float = 70.0
    sensor_dt_s: float = 0.1


@dataclass(frozen=True)
class DynamicsConfig:
    mass_kg: float = 1.5
    gravity_m_s2: float = 9.81
    pitch_limit_rad: float = math.radians(20.0)
    pitch_rate_limit_rad_s: float = math.radians(90.0)
    pitch_natural_frequency_rad_s: float = 10.0
    pitch_damping_ratio: float = 1.0
    thrust_time_constant_s: float = 0.05
    max_thrust_weight_ratio: float = 1.6
    ax_max_m_s2: float = 1.2
    az_max_m_s2: float = 2.0
    clip_actual_pitch: bool = True
    thrust_integrator: str = "exact"


@dataclass(frozen=True)
class ScenarioConfig:
    v1_range_m_s: tuple[float, float] = (0.5, 2.5)
    a2_range_m_s2: tuple[float, float] = (0.3, 1.5)
    T1_range_s: tuple[float, float] = (0.5, 4.0)
    T2_range_s: tuple[float, float] = (0.5, 4.0)
    T3_range_s: tuple[float, float] = (15.0, 55.0)
    initial_height_range_m: tuple[float, float] = (4.0, 8.0)
    max_final_pad_speed_m_s: float = 7.0
    final_speed_margin_m_s: float = 0.5
    parameterization: str = "duration"


@dataclass(frozen=True)
class CameraConfig:
    fov_rad: float = math.radians(70.0)
    maximum_range_m: float = 30.0
    bearing_noise_std_rad: float = math.radians(0.25)
    relative_position_noise_std_m: float = 0.03
    perturbation_profile: str = "paired_losses_v1"


@dataclass(frozen=True)
class EstimatorConfig:
    model: str = "causal_constant_acceleration_kalman_v1"
    process_acceleration_std_m_s2: float = 0.8
    measurement_position_std_m: float = 0.05
    recent_track_grace_s: float = 0.5
    prolonged_loss_s: float = 3.0
    prediction_horizon_s: float = 0.0


@dataclass(frozen=True)
class SafetyConfig:
    pad_half_width_m: float = 0.6
    touchdown_horizontal_error_m: float = 0.35
    touchdown_relative_speed_m_s: float = 0.35
    touchdown_vertical_speed_m_s: float = 0.35
    touchdown_pitch_rad: float = math.radians(5.0)
    touchdown_pitch_rate_rad_s: float = math.radians(10.0)
    minimum_abort_hold_height_m: float = 0.5
    maximum_backup_duration_s: float = 5.0
    touchdown_height_m: float = 0.04
    terminal_descent_height_m: float = 1.0
    terminal_descent_speed_margin: float = 1.5
    terminal_descent_commit_s: float = 1.5
    max_horizontal_range_m: float = 60.0
    max_height_m: float = 20.0
    max_descent_speed_m_s: float = 5.0


@dataclass(frozen=True)
class RewardConfig:
    reference_duration_s: float = 70.0
    horizontal_scale_m: float = 3.0
    height_scale_m: float = 4.0
    horizontal_priority: float = 0.65
    goal_weight: float = 1.0
    view_weight: float = 0.1
    control_weight: float = 0.02
    readiness_weight: float = 0.5
    readiness_model: str = "product_v25"
    readiness_height_m: float = 1.0
    potential_weight: float = 0.0
    terminal_bonus: tuple[tuple[str, float], ...] = (
        ("SUCCESS", 25.0), ("SAFE_ABORT", -15.0),
        ("TASK_TIMEOUT", -12.0), ("UNSAFE_CONTACT", -40.0),
        ("UNAUTHORIZED_CONTACT", -40.0),
        ("MISSED_PAD_CONTACT", -40.0),
        ("SAFETY_ENVELOPE_VIOLATION", -40.0),
    )

    def bonus(self, reason: str | None) -> float:
        if reason is None:
            return 0.0
        return dict(self.terminal_bonus)[reason]

    @property
    def unsafe_reasons(self) -> tuple[str, ...]:
        return ("UNSAFE_CONTACT", "UNAUTHORIZED_CONTACT",
                "MISSED_PAD_CONTACT", "SAFETY_ENVELOPE_VIOLATION")


@dataclass(frozen=True)
class CurriculumConfig:
    """Training-only difficulty schedule shared bit-for-bit by every arm."""

    enabled: bool = True
    start_v1_range_m_s: tuple[float, float] = (0.3, 0.8)
    start_a2_range_m_s2: tuple[float, float] = (0.1, 0.4)
    start_T1_range_s: tuple[float, float] = (0.10, 0.30)
    start_touchdown_relative_speed_m_s: float = 0.70
    start_touchdown_vertical_speed_m_s: float = 0.70
    start_unsafe_contact_penalty: float = -20.0
    promotion_landing_rate: float = 0.6
    promotion_window_episodes: int = 20
    difficulty_step: float = 0.1
    easy_replay_fraction: float = 1.0 / 6.0
    bridge_replay_fraction: float = 1.0 / 6.0
    start_height_range_m: tuple[float, float] | None = None
    floor_start_fraction: float = 0.3
    full_difficulty_fraction: float = 0.8
    scheduled_floor: bool = False


@dataclass(frozen=True)
class GraphConfig:
    schema: str = "compact_landing_context_graph_v1"
    hidden_dimension: int = 32
    policy_hidden_dimension: int = 32
    relation_dimension: int = 6
    adaptation_warmup_fraction: float = 0.0
    freeze_static_backbone: bool = False
    preserve_raw_during_adaptation: bool = False
    selection_score_margin: float = 0.0
    selection_score_model: str = "lexicographic_v1"
    pretrain_episodes: int = 0
    pretrain_decisions: int = 80
    pretrain_epochs: int = 8
    relation_activation_enabled: bool = False
    activation_iterations: int = 25
    activation_decisions: int = 2048


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: str = "two-axis-context-rgat-config/1"
    experiment: str = "two_axis_context_rgat_v1"
    algorithm_version: str = "two-axis-context-rgat-v1"
    pipelines: tuple[str, ...] = (
        "ppo_vector_canonical", "ppo_semantic_flat", "ppo_ontology_rgat")
    timing: TimingConfig = TimingConfig()
    dynamics: DynamicsConfig = DynamicsConfig()
    scenario: ScenarioConfig = ScenarioConfig()
    camera: CameraConfig = CameraConfig()
    estimator: EstimatorConfig = EstimatorConfig()
    safety: SafetyConfig = SafetyConfig()
    reward: RewardConfig = RewardConfig()
    curriculum: CurriculumConfig = CurriculumConfig()
    ontology: GraphConfig = GraphConfig()

    @property
    def canonical_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    @property
    def sha256(self) -> str:
        payload = json.dumps(self.canonical_dict, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _read_config(path: Path, parents: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in parents:
        raise ValueError(f"cyclic experiment inheritance: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment configuration must be a mapping")
    base = payload.pop("extends", None)
    if base is None:
        return payload
    def merge(left, right):
        result = dict(left)
        for key, value in right.items():
            result[key] = merge(result[key], value) if (
                isinstance(value, dict) and isinstance(result.get(key), dict)) else value
        return result
    return merge(_read_config(path.parent / base, parents + (path,)), payload)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> ExperimentConfig:
    """Load and validate the executable subset of the versioned YAML."""
    payload: Mapping[str, Any] = _read_config(Path(path))
    action = dict(payload.get("action") or {})
    if int(action.get("dimension", -1)) != 2:
        raise ValueError("the primary action contract has exactly two channels")
    if tuple(action.get("channels") or ()) != (
            "longitudinal_acceleration", "vertical_acceleration"):
        raise ValueError("action channels or ordering do not match the v1 contract")
    limits = tuple(float(v) for v in action.get("max_acceleration_m_s2", ()))
    if len(limits) != 2 or any(v <= 0 for v in limits):
        raise ValueError("two positive acceleration limits are required")
    legacy = dict(payload.get("legacy") or {})
    enabled_legacy = [name for name, value in legacy.items()
                      if value not in {False, "disabled"}]
    if enabled_legacy:
        raise ValueError(f"legacy mechanisms enabled in primary config: {enabled_legacy}")
    pipelines = tuple(str(v) for v in payload.get("pipelines") or ())
    expected = ("ppo_vector_canonical", "ppo_semantic_flat", "ppo_ontology_rgat")
    if pipelines != expected:
        raise ValueError(f"primary pipelines must be ordered {expected}")
    ppo = dict(payload.get("ppo") or {})
    if ppo.get("behavior_cloning") or ppo.get("learned_reward_weights") or ppo.get("transformed_action_density"):
        raise ValueError("the causal PPO contract forbids BC, learned reward weights and transformed density")
    if ppo.get("raw_action_probability") is not True:
        raise ValueError("PPO must score stored raw Gaussian commands")

    timing = dict(payload.get("timing") or {})
    dyn = dict(payload.get("dynamics") or {})
    scenario = dict(payload.get("scenario") or {})
    ranges = dict(scenario.get("ranges") or {})
    camera = dict(payload.get("camera") or {})
    estimator = dict(payload.get("estimator") or {})
    safety = dict(payload.get("safety") or {})
    reward = dict(payload.get("reward") or {})
    curriculum = dict(payload.get("curriculum") or {})
    weights = dict(reward.get("weights") or {})
    terminal = tuple((str(k), float(v)) for k, v in
                     dict(reward.get("terminal") or {}).items())
    cfg = ExperimentConfig(
        schema_version=str(payload["schema_version"]),
        experiment=str(payload["experiment"]),
        algorithm_version=str(payload["algorithm_version"]),
        pipelines=pipelines,
        timing=TimingConfig(**{k: float(timing.get(k, field.default))
                               for k, field in TimingConfig.__dataclass_fields__.items()}),
        dynamics=DynamicsConfig(
            mass_kg=float(dyn["mass_kg"]), gravity_m_s2=float(dyn["gravity_m_s2"]),
            pitch_limit_rad=math.radians(float(dyn["pitch_limit_deg"])),
            pitch_rate_limit_rad_s=math.radians(float(dyn["pitch_rate_limit_deg_s"])),
            pitch_natural_frequency_rad_s=float(dyn["pitch_natural_frequency_rad_s"]),
            pitch_damping_ratio=float(dyn["pitch_damping_ratio"]),
            thrust_time_constant_s=float(dyn["thrust_time_constant_s"]),
            max_thrust_weight_ratio=float(dyn["max_thrust_weight_ratio"]),
            ax_max_m_s2=limits[0], az_max_m_s2=limits[1],
            clip_actual_pitch=bool(dyn.get("clip_actual_pitch", True)),
            thrust_integrator=str(dyn.get("thrust_integrator", "exact"))),
        scenario=ScenarioConfig(
            v1_range_m_s=_pair(ranges["v1_m_s"], "v1_m_s"),
            a2_range_m_s2=_pair(ranges["a2_m_s2"], "a2_m_s2"),
            T1_range_s=_pair(ranges["T1_s"], "T1_s"),
            T2_range_s=_pair(ranges["T2_s"], "T2_s"),
            T3_range_s=_pair(ranges["T3_s"], "T3_s"),
            initial_height_range_m=_pair(
                ranges["initial_relative_height_m"], "initial_relative_height_m"),
            max_final_pad_speed_m_s=float(scenario["max_final_pad_speed_m_s"]),
            final_speed_margin_m_s=float(scenario["final_speed_margin_m_s"]),
            parameterization=str(scenario["parameterization"])),
        camera=CameraConfig(
            fov_rad=math.radians(float(camera["horizontal_fov_deg"])),
            maximum_range_m=float(camera["maximum_range_m"]),
            bearing_noise_std_rad=math.radians(float(camera["bearing_noise_std_deg"])),
            relative_position_noise_std_m=float(camera["relative_position_noise_std_m"]),
            perturbation_profile=str(camera.get("perturbation_profile", "paired_losses_v1"))),
        estimator=EstimatorConfig(**{
            key: (str(estimator.get(key, field.default)) if key == "model"
                  else float(estimator.get(key, field.default)))
            for key,field in EstimatorConfig.__dataclass_fields__.items()}),
        safety=SafetyConfig(
            pad_half_width_m=float(safety["pad_half_width_m"]),
            touchdown_horizontal_error_m=float(safety["touchdown_horizontal_error_m"]),
            touchdown_relative_speed_m_s=float(safety["touchdown_relative_speed_m_s"]),
            touchdown_vertical_speed_m_s=float(safety["touchdown_vertical_speed_m_s"]),
            touchdown_pitch_rad=math.radians(float(safety["touchdown_pitch_deg"])),
            touchdown_pitch_rate_rad_s=math.radians(
                float(safety["touchdown_pitch_rate_deg_s"])),
            minimum_abort_hold_height_m=float(safety["minimum_abort_hold_height_m"]),
            maximum_backup_duration_s=float(safety["maximum_backup_duration_s"]),
            touchdown_height_m=float(safety["touchdown_height_m"]),
            terminal_descent_height_m=float(safety["terminal_descent_height_m"]),
            terminal_descent_speed_margin=float(
                safety["terminal_descent_speed_margin"]),
            terminal_descent_commit_s=float(
                safety["terminal_descent_commit_s"])),
        reward=RewardConfig(
            reference_duration_s=float(reward["reference_duration_s"]),
            horizontal_scale_m=float(reward["horizontal_scale_m"]),
            height_scale_m=float(reward["height_scale_m"]),
            horizontal_priority=float(reward["horizontal_priority"]),
            goal_weight=float(weights["goal"]), view_weight=float(weights["view"]),
            control_weight=float(weights["control"]),
            readiness_weight=float(weights["readiness"]),
            readiness_model=str(reward.get("readiness_model", "product_v25")),
            readiness_height_m=float(reward.get("readiness_height_m", 1.0)),
            potential_weight=float(weights.get("potential", 0.0)),
            terminal_bonus=terminal),
        curriculum=CurriculumConfig(
            enabled=bool(curriculum["enabled"]),
            start_v1_range_m_s=_pair(curriculum["start_v1_m_s"], "start_v1_m_s"),
            start_a2_range_m_s2=_pair(curriculum["start_a2_m_s2"],
                                      "start_a2_m_s2"),
            start_T1_range_s=_pair(curriculum["start_T1_s"], "start_T1_s"),
            start_touchdown_relative_speed_m_s=float(
                curriculum["start_touchdown_relative_speed_m_s"]),
            start_touchdown_vertical_speed_m_s=float(
                curriculum["start_touchdown_vertical_speed_m_s"]),
            start_unsafe_contact_penalty=float(
                curriculum["start_unsafe_contact_penalty"]),
            promotion_landing_rate=float(curriculum["promotion_landing_rate"]),
            promotion_window_episodes=int(curriculum["promotion_window_episodes"]),
            difficulty_step=float(curriculum["difficulty_step"]),
            easy_replay_fraction=float(curriculum["easy_replay_fraction"]),
            bridge_replay_fraction=float(curriculum["bridge_replay_fraction"]),
            start_height_range_m=(_pair(curriculum["start_height_range_m"], "start_height_range_m")
                                  if "start_height_range_m" in curriculum else None),
            scheduled_floor=bool(curriculum.get("scheduled_floor", False)),
            floor_start_fraction=float(curriculum.get("floor_start_fraction", 0.3)),
            full_difficulty_fraction=float(curriculum.get("full_difficulty_fraction", 0.8))),
        ontology=GraphConfig(**{key: value for key, value in
            dict(payload.get("ontology") or {}).items() if key in GraphConfig.__dataclass_fields__}))
    _validate(cfg)
    if bool(ppo.get("potential_shaping")) != bool(cfg.reward.potential_weight):
        raise ValueError("potential declaration must match its executable reward weight")
    return cfg


def _validate(cfg: ExperimentConfig) -> None:
    from dataclasses import asdict
    def finite_numbers(value):
        if isinstance(value, dict):
            return all(finite_numbers(v) for v in value.values())
        if isinstance(value, (tuple, list)):
            return all(finite_numbers(v) for v in value)
        return not isinstance(value, (float, int)) or math.isfinite(value)
    if not finite_numbers(asdict(cfg)):
        raise ValueError("configuration numbers must be finite")
    if cfg.ontology.schema not in {"compact_landing_context_graph_v1", "compact_context_graph_v3_grouped"}:
        raise ValueError("unsupported graph schema")
    if cfg.reward.readiness_model not in {"product_v25", "velocity_target_v28"}:
        raise ValueError("unsupported readiness model")
    if cfg.estimator.model not in {"causal_constant_acceleration_kalman_v1", "alpha_beta_gamma_v28"}:
        raise ValueError("unsupported estimator model")
    if cfg.estimator.model == "alpha_beta_gamma_v28" and not math.isclose(cfg.timing.sensor_dt_s,.01):
        raise ValueError("v2.8 ABG gains require the 100 Hz sensor clock")
    if cfg.camera.perturbation_profile not in {"paired_losses_v1", "reference_mixture_v28"}:
        raise ValueError("unsupported perturbation profile")
    if cfg.dynamics.thrust_integrator not in {"euler", "exact"}:
        raise ValueError("unsupported thrust integrator")
    if cfg.timing.physics_dt_s <= 0:
        raise ValueError("physics interval must be positive")
    if not 0 < cfg.timing.sensor_dt_s <= cfg.timing.policy_dt_s:
        raise ValueError("sensor interval must be positive and no slower than policy")
    for interval in (cfg.timing.policy_dt_s, cfg.timing.sensor_dt_s):
        ratio = interval / cfg.timing.physics_dt_s
        if not math.isclose(ratio, round(ratio), abs_tol=1e-9):
            raise ValueError("policy and sensor intervals must align with physics ticks")
    if cfg.curriculum.scheduled_floor and not (
            0 <= cfg.curriculum.floor_start_fraction < cfg.curriculum.full_difficulty_fraction < 1):
        raise ValueError("curriculum floor must reach nominal before the budget ends")
    if not 0 <= cfg.ontology.adaptation_warmup_fraction < 1:
        raise ValueError("graph adaptation fraction must be in [0,1)")
    if cfg.ontology.relation_activation_enabled:
        if (cfg.ontology.schema != "compact_context_graph_v3_grouped"
                or not cfg.ontology.preserve_raw_during_adaptation):
            raise ValueError("relation activation requires grouped graph and protected raw path")
        if (type(cfg.ontology.activation_iterations) is not int
                or cfg.ontology.activation_iterations <= 0
                or type(cfg.ontology.activation_decisions) is not int
                or cfg.ontology.activation_decisions < 2):
            raise ValueError("relation activation requires positive integer PPO budgets")
    if cfg.timing.physics_dt_s <= 0 or cfg.timing.policy_dt_s <= 0:
        raise ValueError("simulation time steps must be positive")
    ratio = cfg.timing.policy_dt_s / cfg.timing.physics_dt_s
    if abs(ratio - round(ratio)) > 1e-9:
        raise ValueError("policy_dt_s must contain an integer number of physics steps")
    if cfg.scenario.parameterization not in {"duration", "distance"}:
        raise ValueError("scenario parameterization must be duration or distance")
    if cfg.reward.reference_duration_s != cfg.timing.mission_duration_limit_s:
        raise ValueError("reward reference duration must equal mission duration cap")
    if cfg.dynamics.max_thrust_weight_ratio <= 1.0:
        raise ValueError("the vehicle requires positive upward thrust margin")
    if set(dict(cfg.reward.terminal_bonus)) != {
            "SUCCESS", "SAFE_ABORT", "TASK_TIMEOUT", "UNSAFE_CONTACT",
            "UNAUTHORIZED_CONTACT", "MISSED_PAD_CONTACT",
            "SAFETY_ENVELOPE_VIOLATION"}:
        raise ValueError("terminal reward table does not cover the episode contract")
    if not 0.0 <= cfg.reward.horizontal_priority <= 1.0:
        raise ValueError("horizontal_priority must be a convex mixing weight")
    _validate_actuation_authority(cfg)
    _validate_terminal_ordering(cfg)
    _validate_curriculum_ordering(cfg)
    _validate_touchdown_reachability(cfg)


def _validate_curriculum_ordering(cfg: ExperimentConfig) -> None:
    """The outcome ranking must hold at every rung, not only at nominal.

    The curriculum softens the unsafe penalty so a first landing attempt is
    affordable. Softening it past TASK_TIMEOUT inverts the ranking on the easy
    rungs -- crashing becomes cheaper than flying on -- and a policy trained
    there learns to dive. Checking only the nominal table missed it.
    """
    if not cfg.curriculum.enabled:
        return
    table = dict(cfg.reward.terminal_bonus)
    start = cfg.curriculum.start_unsafe_contact_penalty
    if not start < table["SAFE_ABORT"]:
        raise ValueError(
            f"curriculum start_unsafe_contact_penalty ({start}) must stay below "
            f"SAFE_ABORT ({table['SAFE_ABORT']}); otherwise an unsafe "
            "contact outranks a safe abort on the easy rungs")
    if start > max(table[name] for name in cfg.reward.unsafe_reasons):
        return  # the ramp only ever tightens from here, which is the intent
    raise ValueError("the unsafe ramp must start softer than its nominal value")


def _validate_actuation_authority(cfg: ExperimentConfig) -> None:
    """A pad the vehicle cannot out-accelerate makes episodes untrackable."""
    pad_a_max = cfg.scenario.a2_range_m_s2[1]
    if cfg.dynamics.ax_max_m_s2 <= pad_a_max:
        raise ValueError(
            "longitudinal command limit "
            f"({cfg.dynamics.ax_max_m_s2} m/s^2) must exceed the largest pad "
            f"acceleration the sampler can draw ({pad_a_max} m/s^2); otherwise "
            "some scenarios are physically untrackable by any controller")
    pitch = math.atan2(cfg.dynamics.ax_max_m_s2, cfg.dynamics.gravity_m_s2)
    if pitch > cfg.dynamics.pitch_limit_rad:
        raise ValueError(
            "the longitudinal command limit is not reachable inside the pitch "
            f"limit ({math.degrees(pitch):.1f} deg required)")


def _validate_terminal_ordering(cfg: ExperimentConfig) -> None:
    """SUCCESS > TASK_TIMEOUT > SAFE_ABORT > unsafe outcomes.

    A controller that keeps the pad in view until the deadline must outrank one
    that induces its own visual loss, otherwise aborting is a reward shortcut.
    """
    table = dict(cfg.reward.terminal_bonus)
    unsafe = max(table[name] for name in cfg.reward.unsafe_reasons)
    ordering = (("SUCCESS", table["SUCCESS"]),
                ("TASK_TIMEOUT", table["TASK_TIMEOUT"]),
                ("SAFE_ABORT", table["SAFE_ABORT"]),
                ("unsafe_outcomes", unsafe))
    for (left, lvalue), (right, rvalue) in zip(ordering, ordering[1:]):
        if not lvalue > rvalue:
            raise ValueError(
                f"terminal reward ordering violated: {left} ({lvalue}) must "
                f"rank above {right} ({rvalue})")


def _validate_touchdown_reachability(cfg: ExperimentConfig) -> None:
    """An authorized touchdown must be geometrically possible.

    The abort-hold margin inhibits landing for every height below
    ``minimum_abort_hold_height_m``. Without a terminal-descent corridor below
    it, no contact can ever be authorized and SUCCESS is unreachable.
    """
    safety = cfg.safety
    if safety.terminal_descent_height_m <= safety.touchdown_height_m:
        raise ValueError("terminal descent corridor must sit above the gear plane")
    if safety.terminal_descent_height_m < safety.minimum_abort_hold_height_m:
        raise ValueError(
            "the terminal descent corridor must open at or above the abort-hold "
            "height, otherwise authorized contact is unreachable")
    if not 0.0 < safety.touchdown_height_m < safety.minimum_abort_hold_height_m:
        raise ValueError("gear contact plane must lie below the abort-hold height")
    if safety.terminal_descent_speed_margin < 1.0:
        raise ValueError("terminal descent speed margin may not tighten below nominal")
    if not 0.0 < safety.terminal_descent_commit_s < cfg.estimator.prolonged_loss_s:
        raise ValueError(
            "the terminal descent commit window must be shorter than the "
            "prolonged-loss abort threshold so a commit can never outlive it")
