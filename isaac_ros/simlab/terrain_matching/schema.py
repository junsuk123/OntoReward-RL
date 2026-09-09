"""Validated sensor-facing schema; ground truth is intentionally absent."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TerrainVisualObservation:
    frame_id: str
    timestamp_s: float
    texture_entropy: float
    texture_variance: float
    edge_density: float
    corner_density: float
    feature_count: int
    feature_distribution_uniformity: float
    repetitiveness_score: float
    brightness: float
    blur_score: float
    flow_mean: float = 0.0
    flow_std: float = 0.0
    flow_valid_ratio: float = 0.0
    flow_inlier_ratio: float = 0.0
    flow_direction_consistency: float = 0.0
    altitude_m: float = 10.0
    ground_speed_mps: float = 3.0
    roll_rad: float = 0.0
    pitch_rad: float = 0.0
    yaw_rate_rad_s: float = 0.0
    terrain_type_if_available: str | None = None
    dynamic_texture_suspected: bool = False
    low_texture_suspected: bool = False
    repetitive_pattern_suspected: bool = False
    normalized: dict[str, float] = field(default_factory=dict)
    visual_matchability: float | None = None
    matchability_class: str | None = None
    reasoning_trace: list[dict[str, Any]] = field(default_factory=list)

    def validate(self) -> None:
        if not self.frame_id or self.timestamp_s < 0 or self.feature_count < 0:
            raise ValueError("invalid frame identity, timestamp, or feature count")
        bounded = (
            "edge_density", "corner_density", "feature_distribution_uniformity",
            "repetitiveness_score", "flow_valid_ratio", "flow_inlier_ratio",
            "flow_direction_consistency",
        )
        for name in bounded:
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name}={value} is outside [0, 1]")
        if self.visual_matchability is not None and not 0 <= self.visual_matchability <= 1:
            raise ValueError("visual_matchability is outside [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)
