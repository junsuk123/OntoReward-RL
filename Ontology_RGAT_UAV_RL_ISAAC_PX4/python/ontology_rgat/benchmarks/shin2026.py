"""Contracts for the Shin-2026-compatible controlled landing benchmark.

The important boundary in this module is not a convention: it is executable.
Only image, UAV body velocity and UAV attitude can be constructed as an actor
observation.  Simulator/platform truth lives in the critic or reward contracts,
whose types are intentionally not accepted by :class:`ActorObservation`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


FORBIDDEN_ACTOR_TOKENS = frozenset({
    "platform_position", "platform_velocity", "pad_position", "pad_velocity",
    "deck_position", "deck_velocity", "platform_gnss", "deck_gnss",
    "wheel_odometry", "v2v", "trajectory_parameters", "future_platform_motion",
    "simulator_truth", "ground_truth", "true_relative_state", "critic_observation",
    # Simulator pad geometry. It supervises keypoints, gates the reset and
    # labels the future-FOV-loss dataset; none of that may reach the actor.
    "pad_center", "pad_landmark", "keypoint_label", "geometric_fov",
    "in_fov", "marker_quality", "marker_pose",
})


def _normalise_key(value: object) -> str:
    return str(value).strip().lower().replace("-", "_").replace("/", "_")


def assert_actor_payload_safe(payload: Mapping[str, Any]) -> None:
    """Reject privileged fields recursively before actor inference.

    This checks names rather than values so a zero-filled truth channel cannot
    accidentally pass as harmless.  Callers should pass the unflattened payload
    at the actor boundary, before any concatenation obscures provenance.
    """
    def visit(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                name = _normalise_key(key)
                if (name in FORBIDDEN_ACTOR_TOKENS
                        or any(token in name for token in FORBIDDEN_ACTOR_TOKENS)):
                    raise ValueError(f"privileged actor field at {path}{key}: {name}")
                visit(value, f"{path}{key}.")
        elif isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                visit(value, f"{path}{index}.")
    visit(payload, "actor.")


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.float32).reshape(-1)
    if out.shape != (size,) or not np.isfinite(out).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return out


@dataclass(frozen=True)
class ActorObservation:
    """The complete deployed actor input; no platform communication is legal."""

    image: np.ndarray
    body_velocity: np.ndarray
    attitude_quaternion: np.ndarray

    def __post_init__(self) -> None:
        image = np.asarray(self.image)
        if image.ndim not in (2, 3):
            raise ValueError("image must be grayscale HxW or singleton-channel HxWx1")
        if image.ndim == 3 and image.shape[-1] != 1:
            raise ValueError("benchmark actor image must be grayscale")
        if image.size == 0 or not np.isfinite(image).all():
            raise ValueError("image must be non-empty and finite")
        object.__setattr__(self, "image", image)
        object.__setattr__(self, "body_velocity",
                           _finite_vector(self.body_velocity, 3, "body_velocity"))
        q = _finite_vector(self.attitude_quaternion, 4, "attitude_quaternion")
        norm = float(np.linalg.norm(q))
        if norm < 1e-8:
            raise ValueError("attitude_quaternion must have non-zero norm")
        object.__setattr__(self, "attitude_quaternion", q / norm)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ActorObservation":
        assert_actor_payload_safe(payload)
        allowed = {"image", "body_velocity", "attitude_quaternion"}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"unknown actor fields: {sorted(unknown)}")
        missing = allowed - set(payload)
        if missing:
            raise ValueError(f"missing actor fields: {sorted(missing)}")
        return cls(**{name: payload[name] for name in allowed})

    @property
    def proprioception(self) -> np.ndarray:
        return np.concatenate((self.body_velocity, self.attitude_quaternion))


@dataclass(frozen=True)
class CriticObservation:
    """Training-only asymmetric critic input."""

    actor: ActorObservation
    true_relative_state: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "true_relative_state",
                           _finite_vector(self.true_relative_state, 6,
                                          "true_relative_state"))

    @property
    def privileged_vector(self) -> np.ndarray:
        # Section III-D: o_priv = [u_t, s_rel_t], exactly 7 + 6 values.
        return np.concatenate((self.actor.proprioception, self.true_relative_state))


@dataclass(frozen=True)
class RewardSignals:
    """Training/reward-side signals that are never flattened into actor input."""

    estimated_relative_state: np.ndarray
    true_relative_state: np.ndarray | None = None
    physical_contact: bool = False
    crash: bool = False
    excessive_drift: bool = False
    terminal: bool = False
    geometric_pad_center_in_fov: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "estimated_relative_state",
                           _finite_vector(self.estimated_relative_state, 6,
                                          "estimated_relative_state"))
        if self.true_relative_state is not None:
            object.__setattr__(self, "true_relative_state",
                               _finite_vector(self.true_relative_state, 6,
                                              "true_relative_state"))


@dataclass(frozen=True)
class ShinBenchmarkConfig:
    dt_seconds: float = 0.1
    horizon_steps: int = 300
    image_width: int = 512
    image_height: int = 320
    horizontal_fov_deg: float = 90.0
    camera_pitch_deg: float = 60.0
    image_embedding: int = 512
    lstm_hidden: int = 512
    latent_dimension: int = 256
    relative_state_dimension: int = 6
    action_dimension: int = 4
    ppo_gamma: float = 0.99
    reward_mode: str = "shin2026"
    seed: int = 42
    curriculum_enabled: bool = True

    def validate(self) -> None:
        if self.dt_seconds <= 0 or self.horizon_steps <= 0:
            raise ValueError("positive control period and horizon are required")
        if (self.image_width, self.image_height) != (512, 320):
            raise ValueError("the Shin benchmark camera profile is 512x320")
        if self.relative_state_dimension != 6:
            raise ValueError("relative-state estimator output must be six-dimensional")
        if self.action_dimension != 4:
            raise ValueError("velocity/yaw-rate action must be four-dimensional")
        if self.reward_mode not in {
                "shin2026", "sparse", "manual_no_active", "ontoreward",
                "ontoreward_plus_active"}:
            raise ValueError(f"unknown benchmark reward mode: {self.reward_mode}")


def default_shin2026_config(**overrides: Any) -> ShinBenchmarkConfig:
    cfg = ShinBenchmarkConfig(**overrides)
    cfg.validate()
    return cfg
