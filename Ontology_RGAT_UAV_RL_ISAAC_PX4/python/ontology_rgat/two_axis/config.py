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


@dataclass(frozen=True)
class EstimatorConfig:
    process_acceleration_std_m_s2: float = 0.8
    measurement_position_std_m: float = 0.05
    recent_track_grace_s: float = 0.5
    prolonged_loss_s: float = 3.0


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
    max_horizontal_range_m: float = 60.0
    max_height_m: float = 20.0
    max_descent_speed_m_s: float = 5.0


@dataclass(frozen=True)
class RewardConfig:
    reference_duration_s: float = 70.0
    horizontal_scale_m: float = 3.0
    height_scale_m: float = 4.0
    goal_weight: float = 1.0
    view_weight: float = 0.1
    control_weight: float = 0.02
    terminal_bonus: tuple[tuple[str, float], ...] = (
        ("SUCCESS", 10.0), ("SAFE_ABORT", -3.0),
        ("TASK_TIMEOUT", -12.0), ("UNSAFE_CONTACT", -40.0),
        ("UNAUTHORIZED_CONTACT", -40.0),
        ("MISSED_PAD_CONTACT", -40.0),
        ("SAFETY_ENVELOPE_VIOLATION", -40.0),
    )

    def bonus(self, reason: str | None) -> float:
        if reason is None:
            return 0.0
        return dict(self.terminal_bonus)[reason]


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

    @property
    def canonical_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    @property
    def sha256(self) -> str:
        payload = json.dumps(self.canonical_dict, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> ExperimentConfig:
    """Load and validate the executable subset of the versioned YAML."""
    payload: Mapping[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
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

    timing = dict(payload.get("timing") or {})
    dyn = dict(payload.get("dynamics") or {})
    scenario = dict(payload.get("scenario") or {})
    ranges = dict(scenario.get("ranges") or {})
    camera = dict(payload.get("camera") or {})
    estimator = dict(payload.get("estimator") or {})
    safety = dict(payload.get("safety") or {})
    reward = dict(payload.get("reward") or {})
    weights = dict(reward.get("weights") or {})
    terminal = tuple((str(k), float(v)) for k, v in
                     dict(reward.get("terminal") or {}).items())
    cfg = ExperimentConfig(
        schema_version=str(payload["schema_version"]),
        experiment=str(payload["experiment"]),
        algorithm_version=str(payload["algorithm_version"]),
        pipelines=pipelines,
        timing=TimingConfig(**{k: float(timing[k]) for k in TimingConfig.__dataclass_fields__}),
        dynamics=DynamicsConfig(
            mass_kg=float(dyn["mass_kg"]), gravity_m_s2=float(dyn["gravity_m_s2"]),
            pitch_limit_rad=math.radians(float(dyn["pitch_limit_deg"])),
            pitch_rate_limit_rad_s=math.radians(float(dyn["pitch_rate_limit_deg_s"])),
            pitch_natural_frequency_rad_s=float(dyn["pitch_natural_frequency_rad_s"]),
            pitch_damping_ratio=float(dyn["pitch_damping_ratio"]),
            thrust_time_constant_s=float(dyn["thrust_time_constant_s"]),
            max_thrust_weight_ratio=float(dyn["max_thrust_weight_ratio"]),
            ax_max_m_s2=limits[0], az_max_m_s2=limits[1]),
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
            relative_position_noise_std_m=float(camera["relative_position_noise_std_m"])),
        estimator=EstimatorConfig(**{
            key: float(estimator[key]) for key in EstimatorConfig.__dataclass_fields__}),
        safety=SafetyConfig(
            pad_half_width_m=float(safety["pad_half_width_m"]),
            touchdown_horizontal_error_m=float(safety["touchdown_horizontal_error_m"]),
            touchdown_relative_speed_m_s=float(safety["touchdown_relative_speed_m_s"]),
            touchdown_vertical_speed_m_s=float(safety["touchdown_vertical_speed_m_s"]),
            touchdown_pitch_rad=math.radians(float(safety["touchdown_pitch_deg"])),
            touchdown_pitch_rate_rad_s=math.radians(
                float(safety["touchdown_pitch_rate_deg_s"])),
            minimum_abort_hold_height_m=float(safety["minimum_abort_hold_height_m"]),
            maximum_backup_duration_s=float(safety["maximum_backup_duration_s"])),
        reward=RewardConfig(
            reference_duration_s=float(reward["reference_duration_s"]),
            horizontal_scale_m=float(reward["horizontal_scale_m"]),
            height_scale_m=float(reward["height_scale_m"]),
            goal_weight=float(weights["goal"]), view_weight=float(weights["view"]),
            control_weight=float(weights["control"]), terminal_bonus=terminal))
    _validate(cfg)
    return cfg


def _validate(cfg: ExperimentConfig) -> None:
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
