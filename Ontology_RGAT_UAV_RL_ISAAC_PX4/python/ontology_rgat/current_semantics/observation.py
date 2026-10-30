"""The current 21D source observation, decoded back into physical units.

Every feature of the new path is a pure function of this one vector, so the
decode is exact and strict.  The only in-vector estimate validity is the
vision-age sentinel: an uninitialised pad estimate arrives as a zeroed
estimate prefix with age 1.0, and those zeros must never read as alignment.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from ..direct_policy.observation import age_scale, signed_scale
from .contracts import (ESTIMATE_CHANNELS, OBSERVATION_CHANNELS, OBSERVATION_DIMENSION,
                        OBSERVATION_NAMES, TRIG_PAIRS, TRIG_TOLERANCE)


@dataclass(frozen=True)
class CurrentPhysical:
    """SI values of one decision; angles are scipy ``as_euler("xyz")`` of the
    ENU/FLU body->world attitude and p, q, r are FLU body rates."""
    ex: float
    ey: float
    h: float
    vrx: float
    vry: float
    vgx: float
    vgy: float
    vz: float
    roll: float
    pitch: float
    yaw: float
    p: float
    q: float
    r: float
    vision_updated: bool
    vision_age_s: float
    nav_valid: bool
    nav_age_s: float
    estimate_valid: bool

    @property
    def own_vx(self) -> float:
        return self.vgx - self.vrx

    @property
    def own_vy(self) -> float:
        return self.vgy - self.vry


_SIGNED = (("relative_x", "ex"), ("relative_y", "ey"), ("relative_height", "h"),
           ("relative_vx", "vrx"), ("relative_vy", "vry"), ("ugv_vx", "vgx"),
           ("ugv_vy", "vgy"), ("drone_vz", "vz"), ("drone_rollRate", "p"),
           ("drone_pitchRate", "q"), ("drone_yawRate", "r"))
_FLAGS = (("ugv_visionUpdated", "vision_updated"), ("drone_navigationValid", "nav_valid"))
_AGES = (("ugv_visionAge", "vision_age_s"), ("drone_navigationAge", "nav_age_s"))
_VISION_AGE = "ugv_visionAge"

_INDEX = {name: index for index, name in enumerate(OBSERVATION_NAMES)}
_SCALE = {item.name: item.scale for item in OBSERVATION_CHANNELS}
_SOURCE_SCALE = {item.name: item.source_scale for item in OBSERVATION_CHANNELS}
_TRANSFORM = {item.name: item.transform for item in OBSERVATION_CHANNELS}
_ESTIMATE_FIELDS = tuple(field for name, field in _SIGNED if name in ESTIMATE_CHANNELS)

_covered = ([(name, "signed") for name, _ in _SIGNED] + [(name, "flag") for name, _ in _FLAGS]
            + [(name, "age") for name, _ in _AGES]
            + [(name, "trig") for _, *pair in TRIG_PAIRS for name in pair])
if sorted(_covered) != sorted(_TRANSFORM.items()):
    raise AssertionError("decode must cover every registered channel exactly once")


def _as_vector(vector: Any) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64)
    if values.shape != (OBSERVATION_DIMENSION,):
        raise ValueError(f"current observation must have shape ({OBSERVATION_DIMENSION},), "
                         f"got {values.shape}; the stable 24D profile is excluded")
    if not np.all(np.isfinite(values)):
        raise ValueError("current observation must be finite")
    return values


def raw_by_name(vector: Any) -> dict[str, float]:
    return dict(zip(OBSERVATION_NAMES, _as_vector(vector).tolist()))


def decode(vector: Any) -> CurrentPhysical:
    """Invert each channel by its registered rule; malformed input raises ValueError."""
    raw = raw_by_name(vector)
    values: dict[str, Any] = {}
    for name, field in _SIGNED:
        u = raw[name]
        if not abs(u) < 1.0:
            raise ValueError(f"{name}: signed channel needs |u| < 1, got {u!r}")
        values[field] = _SCALE[name] * u / (1.0 - abs(u))
    for angle, sin_name, cos_name in TRIG_PAIRS:
        s, c = raw[sin_name], raw[cos_name]
        if abs(s * s + c * c - 1.0) > TRIG_TOLERANCE:
            raise ValueError(f"{sin_name}/{cos_name} are not a unit pair: {s!r}, {c!r}")
        values[angle] = math.atan2(s, c)
    for name, field in _FLAGS:
        if raw[name] not in (0.0, 1.0):
            raise ValueError(f"{name}: flag must be exactly 0 or 1, got {raw[name]!r}")
        values[field] = raw[name] == 1.0
    for name, field in _AGES:
        u = raw[name]
        if not 0.0 <= u <= 1.0:
            raise ValueError(f"{name}: age channel needs 0 <= u <= 1, got {u!r}")
        values[field] = math.inf if u == 1.0 else _SCALE[name] * u / (1.0 - u)
    valid = raw[_VISION_AGE] < 1.0
    if not valid:
        nonzero = [name for name in ESTIMATE_CHANNELS if raw[name] != 0.0]
        if nonzero:
            raise ValueError(f"vision age carries the uninitialised sentinel but "
                             f"{nonzero} are non-zero")
        values.update({field: 0.0 for field in _ESTIMATE_FIELDS})
    return CurrentPhysical(**values, estimate_valid=valid)


def from_source(vector: Any) -> np.ndarray:
    """``direct_policy.observation.spatial_vector`` -> the registered encoding.

    Each signed channel is decoded with its direct-policy source scale and
    re-encoded with the registered one (contracts.OBSERVATION_SCALE_OVERRIDES);
    trig, flag and age channels are copied, and a zeroed estimate stays zero.
    """
    values = _as_vector(vector).copy()
    for name, _ in _SIGNED:
        index, u = _INDEX[name], values[_INDEX[name]]
        if not abs(u) < 1.0:
            raise ValueError(f"{name}: signed channel needs |u| < 1, got {u!r}")
        if _SOURCE_SCALE[name] != _SCALE[name]:
            values[index] = signed_scale(_SOURCE_SCALE[name] * u / (1.0 - abs(u)), _SCALE[name])
    return values


def encode(physical: CurrentPhysical) -> np.ndarray:
    """Forward map for fixtures, using the direct-policy scaling functions."""
    values: dict[str, float] = {}
    for name, field in _SIGNED:
        x = float(getattr(physical, field))
        if field in _ESTIMATE_FIELDS and not physical.estimate_valid:
            x = 0.0
        if not math.isfinite(x):
            raise ValueError(f"{field} must be finite")
        values[name] = float(signed_scale(x, _SCALE[name]))
    for angle, sin_name, cos_name in TRIG_PAIRS:
        a = float(getattr(physical, angle))
        if not math.isfinite(a):
            raise ValueError(f"{angle} must be finite")
        values[sin_name], values[cos_name] = math.sin(a), math.cos(a)
    for name, field in _FLAGS:
        values[name] = float(bool(getattr(physical, field)))
    for name, field in _AGES:
        age = float(getattr(physical, field))
        if name == _VISION_AGE and not physical.estimate_valid:
            age = math.inf
        if math.isnan(age) or age < 0.0:
            raise ValueError(f"{field} must be a non-negative age or inf")
        values[name] = float(age_scale(age, _SCALE[name]))
    if physical.estimate_valid and values[_VISION_AGE] >= 1.0:
        raise ValueError("a valid estimate needs a vision age that encodes below the sentinel")
    return np.asarray([values[name] for name in OBSERVATION_NAMES], dtype=np.float64)
