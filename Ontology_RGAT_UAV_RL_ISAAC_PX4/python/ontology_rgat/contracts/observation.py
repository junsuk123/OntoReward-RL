"""Canonical, causal actor-observation registry and packet.

The registry is the authority for both learned arms.  It records provenance,
normalisation and order before values are flattened, so a truth channel cannot
be hidden in a vector by renaming it downstream.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np


_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY = _ROOT / "config/observation/causal_packet_v1.json"


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("utf-8")


@dataclass(frozen=True)
class ObservationRegistry:
    schema: str
    clock: str
    fields: tuple[Mapping[str, object], ...]
    forbidden_sources: tuple[str, ...]
    sha256: str

    @property
    def dimension(self) -> int:
        return sum(int(field["size"]) for field in self.fields)

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(str(field["name"]) for field in self.fields)

    def assert_same(self, other: "ObservationRegistry") -> None:
        if self.sha256 != other.sha256:
            raise ValueError(
                "learned arms must use the identical observation registry hash: "
                f"{self.sha256} != {other.sha256}")


def load_observation_registry(path: str | Path = DEFAULT_REGISTRY
                              ) -> ObservationRegistry:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"schema", "clock", "fields", "forbidden_sources"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"observation registry is missing {sorted(missing)}")
    fields = tuple(payload["fields"])
    names = [str(field.get("name", "")) for field in fields]
    if not fields or len(names) != len(set(names)) or any(not name for name in names):
        raise ValueError("observation registry field names must be non-empty and unique")
    forbidden = tuple(str(value).lower() for value in payload["forbidden_sources"])
    for field in fields:
        if int(field.get("size", 0)) <= 0:
            raise ValueError(f"observation field {field.get('name')} has invalid size")
        provenance = f"{field.get('name')} {field.get('source')}".lower()
        hit = next((token for token in forbidden if token in provenance), None)
        if hit is not None:
            raise ValueError(
                f"observation field {field.get('name')} uses forbidden source {hit}")
        if not field.get("normalization"):
            raise ValueError(f"observation field {field.get('name')} lacks normalization")
    digest = hashlib.sha256(_canonical(payload)).hexdigest()
    return ObservationRegistry(str(payload["schema"]), str(payload["clock"]),
                               fields, forbidden, digest)


@dataclass(frozen=True)
class CausalObservationPacket:
    """One immutable packet in registry order, with no critic/reward handle."""

    values: np.ndarray
    registry_sha256: str
    timestamp_s: float

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float32).reshape(-1)
        if not np.isfinite(values).all():
            raise ValueError("causal observation packet must be finite")
        if not np.isfinite(float(self.timestamp_s)):
            raise ValueError("causal observation timestamp must be finite")
        object.__setattr__(self, "values", values)

    @classmethod
    def from_fields(cls, fields: Mapping[str, object], *, timestamp_s: float,
                    registry: ObservationRegistry | None = None
                    ) -> "CausalObservationPacket":
        registry = registry or load_observation_registry()
        unknown = set(fields) - set(registry.field_names)
        missing = set(registry.field_names) - set(fields)
        if unknown or missing:
            raise ValueError(
                f"observation packet fields mismatch; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}")
        chunks = []
        for field in registry.fields:
            name, size = str(field["name"]), int(field["size"])
            value = np.asarray(fields[name], dtype=np.float32).reshape(-1)
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f"observation field {name} must have {size} finite values")
            chunks.append(value)
        return cls(np.concatenate(chunks), registry.sha256, float(timestamp_s))

    def assert_registry(self, registry: ObservationRegistry) -> None:
        if self.registry_sha256 != registry.sha256:
            raise ValueError("causal observation packet registry hash mismatch")
        if self.values.shape != (registry.dimension,):
            raise ValueError("causal observation packet dimension mismatch")


def causal_packet_from_visual(*, keypoints, keypoint_visibility,
                              semantic_observation, proprioception,
                              timestamp_s: float, previous_packet=None,
                              previous_timestamp_s=None,
                              registry: ObservationRegistry | None = None
                              ) -> CausalObservationPacket:
    """Construct ``O_t`` only from encoder output, causal history and UAV state."""
    registry = registry or load_observation_registry()
    points = np.asarray(keypoints, dtype=np.float32).reshape(6, 2)
    visibility = np.asarray(keypoint_visibility, dtype=np.float32).reshape(6)
    proprio = np.asarray(proprioception, dtype=np.float32).reshape(7)
    centroid = np.asarray(semantic_observation.centroid_xy,
                          dtype=np.float32).reshape(2)
    scale = float(semantic_observation.raw_scale)
    age = float(semantic_observation.visual_loss_duration_s)
    if previous_packet is None:
        centroid_rate = np.zeros(2, dtype=np.float32)
        scale_rate = 0.0
        margin_rate = 0.0
    else:
        previous_packet.assert_registry(registry)
        if previous_timestamp_s is None:
            previous_timestamp_s = previous_packet.timestamp_s
        dt = float(timestamp_s) - float(previous_timestamp_s)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("causal packet history needs an increasing timestamp")
        offsets = {}
        cursor = 0
        for field in registry.fields:
            size = int(field["size"])
            offsets[str(field["name"])] = slice(cursor, cursor + size)
            cursor += size
        old_centroid = previous_packet.values[offsets["centroid_xy"]]
        old_scale = float(previous_packet.values[offsets["apparent_scale"]][0])
        old_margin = float(previous_packet.values[offsets["fov_margin"]][0])
        centroid_rate = (centroid - old_centroid) / dt
        scale_rate = (scale - old_scale) / dt
        margin = float(np.clip(1.0 - np.max(np.abs(centroid)), 0.0, 1.0))
        margin_rate = (margin - old_margin) / dt
    margin = float(np.clip(1.0 - np.max(np.abs(centroid)), 0.0, 1.0))
    valid = float(semantic_observation.visible_keypoint_fraction >= 0.5)
    fields = {
        "keypoint_xy": points.reshape(-1),
        "keypoint_visibility": visibility,
        "keypoint_confidence": [semantic_observation.keypoint_confidence],
        "centroid_xy": centroid,
        "apparent_scale": [scale],
        "centroid_rate_xy": centroid_rate,
        "scale_rate": [scale_rate],
        "visible_fraction": [semantic_observation.visible_keypoint_fraction],
        "measurement_valid": [valid],
        "measurement_age_s": [age],
        "stale": [float(age > 0.0)],
        "fov_margin": [margin],
        "fov_margin_rate": [margin_rate],
        "body_velocity": proprio[:3],
        "attitude_quaternion": proprio[3:],
    }
    return CausalObservationPacket.from_fields(
        fields, timestamp_s=timestamp_s, registry=registry)
