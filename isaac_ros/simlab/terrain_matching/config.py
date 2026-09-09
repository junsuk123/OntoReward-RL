"""Strict, configuration-driven settings for terrain matching experiments."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

import yaml

from simlab.utils.paths import PROJECT_ROOT

DEFAULT_EXPERIMENT_CONFIG = PROJECT_ROOT / "config" / "experiment.yaml"


def _known(cls: type, data: Mapping[str, Any]) -> None:
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {sorted(unknown)}")


@dataclass(frozen=True)
class ImageConfig:
    size: int = 384
    grid_size: int = 4
    orb_features: int = 900
    canny_low: int = 60
    canny_high: int = 150


@dataclass(frozen=True)
class FlowConfig:
    max_corners: int = 300
    quality_level: float = 0.01
    min_distance: float = 7.0
    ransac_threshold_px: float = 2.0


@dataclass(frozen=True)
class MatchingConfig:
    ratio_test: float = 0.78
    min_matches: int = 10
    ransac_threshold_px: float = 4.0
    low_matchability_reject: float = 0.18
    correct_tolerance_m: float = 5.0


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    deterministic: bool = True
    terrain_classes: tuple[str, ...] = ("road", "grass", "urban_building", "water")
    conditions: tuple[str, ...] = ("normal", "bright", "dark", "blur", "yaw")
    repetitions: int = 1
    output_directory: str = "results"
    image: ImageConfig = field(default_factory=ImageConfig)
    optical_flow: FlowConfig = field(default_factory=FlowConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    rules_file: str = "config/ontology_rules.yaml"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentConfig":
        _known(cls, data)
        payload = dict(data)
        image = ImageConfig(**payload.pop("image", {}))
        flow = FlowConfig(**payload.pop("optical_flow", {}))
        matching = MatchingConfig(**payload.pop("matching", {}))
        payload["terrain_classes"] = tuple(payload.get("terrain_classes", cls.terrain_classes))
        payload["conditions"] = tuple(payload.get("conditions", cls.conditions))
        cfg = cls(image=image, optical_flow=flow, matching=matching, **payload)
        if cfg.seed < 0 or cfg.repetitions < 1 or cfg.image.size < 128:
            raise ValueError("seed >= 0, repetitions >= 1 and image.size >= 128 are required")
        return cfg


def load_experiment_config(path: str | Path = DEFAULT_EXPERIMENT_CONFIG) -> ExperimentConfig:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, Mapping):
        raise ValueError("experiment config must be a YAML mapping")
    return ExperimentConfig.from_dict(data)
