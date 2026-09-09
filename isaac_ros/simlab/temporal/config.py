"""Configuration model for accelerated temporal environment experiments."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

import yaml

from simlab.utils.paths import PROJECT_ROOT

DEFAULT_TEMPORAL_CONFIG = PROJECT_ROOT / "config" / "temporal_environment.yaml"


def _check(cls: type, data: Mapping[str, Any]) -> None:
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown: raise ValueError(f"{cls.__name__}: unknown keys {sorted(unknown)}")


@dataclass(frozen=True)
class EpochConfig:
    name: str
    year: float
    season: str
    weather: str
    component_overrides: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TimeConfig:
    mode: str = "epoch"
    days_per_real_second: float = 365.0
    maximum_year: float = 10.0


@dataclass(frozen=True)
class CameraConfig:
    resolution: tuple[int, int] = (512, 512)
    altitude_m: float = 45.0
    focal_length_mm: float = 24.0
    trajectory_mode: str = "spiral"
    trajectory_xy: tuple[tuple[float, float], ...] = ((-.45, 0), (0, 0), (.45, 0))
    spiral_center_xy_m: tuple[float, float] = (0.0, 0.0)
    spiral_start_radius_m: float = 12.0
    spiral_end_radius_m: float = 92.0
    spiral_turns: float = 2.5
    spiral_samples: int = 32
    clockwise: bool = False
    capture_fps: float = 8.0


@dataclass(frozen=True)
class MapConfig:
    extent_m: float = 300.0
    canvas_px: int = 1800


@dataclass(frozen=True)
class ChangeConfig:
    tree_initial_count: int = 80
    tree_new_count: int = 40
    tree_scale_range: tuple[float, float] = (.65, 1.55)
    building_complete_year: float = 5.0
    building_renovate_year: float = 10.0
    road_resurface_year: float = 5.0
    road_expand_year: float = 10.0


@dataclass(frozen=True)
class CaptureConfig:
    save_images: bool = True
    save_epoch_usd_layers: bool = True
    image_backend: str = "opencv"


@dataclass(frozen=True)
class TemporalConfig:
    random_seed: int = 73
    output_directory: str = "results/temporal"
    epochs: tuple[EpochConfig, ...] = field(default_factory=tuple)
    time: TimeConfig = field(default_factory=TimeConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    map: MapConfig = field(default_factory=MapConfig)
    changes: ChangeConfig = field(default_factory=ChangeConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    score_weights: dict[str, float] = field(default_factory=dict)
    events: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TemporalConfig":
        _check(cls, data); p = dict(data)
        epochs = tuple(EpochConfig(**item) for item in p.pop("epochs", []))
        time = TimeConfig(**p.pop("time", {})); changes = ChangeConfig(**p.pop("changes", {})); capture = CaptureConfig(**p.pop("capture", {}));map_cfg=MapConfig(**p.pop("map",{}))
        camera_data = dict(p.pop("camera", {}))
        camera_data["resolution"] = tuple(camera_data.get("resolution", CameraConfig.resolution))
        camera_data["trajectory_xy"] = tuple(tuple(x) for x in camera_data.get("trajectory_xy", CameraConfig.trajectory_xy))
        camera_data["spiral_center_xy_m"] = tuple(camera_data.get("spiral_center_xy_m",CameraConfig.spiral_center_xy_m))
        cfg = cls(epochs=epochs, time=time, camera=CameraConfig(**camera_data),map=map_cfg,changes=changes, capture=capture, events=tuple(p.pop("events", [])), **p)
        if not cfg.epochs or cfg.epochs[0].year != 0: raise ValueError("epochs must begin with the Year0 reference state")
        if cfg.time.maximum_year <= 0 or any(e.year < 0 or e.year > cfg.time.maximum_year for e in cfg.epochs): raise ValueError("epoch year is outside configured time range")
        if set(cfg.score_weights) != {"vegetation", "building", "road_appearance", "road_geometry", "season", "weather", "illumination"}: raise ValueError("score_weights must define all seven change components")
        if abs(sum(cfg.score_weights.values()) - 1.0) > 1e-6: raise ValueError("score_weights must sum to 1")
        if cfg.camera.trajectory_mode not in {"fixed","spiral"}:raise ValueError("camera.trajectory_mode must be fixed or spiral")
        if cfg.camera.spiral_samples<8 or not 0<cfg.camera.spiral_start_radius_m<cfg.camera.spiral_end_radius_m<cfg.map.extent_m/2:raise ValueError("spiral radii/samples must fit inside the map")
        if cfg.map.canvas_px<512 or cfg.map.extent_m<100:raise ValueError("map must be at least 100 m and 512 px")
        return cfg


def load_temporal_config(path: str | Path = DEFAULT_TEMPORAL_CONFIG) -> TemporalConfig:
    path = Path(path); path = path if path.is_absolute() else PROJECT_ROOT / path
    with path.open(encoding="utf-8") as stream: data = yaml.safe_load(stream) or {}
    if not isinstance(data, Mapping): raise ValueError("temporal config must be a mapping")
    return TemporalConfig.from_dict(data)
