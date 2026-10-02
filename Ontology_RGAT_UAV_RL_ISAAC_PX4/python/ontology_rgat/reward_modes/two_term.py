"""The shared two-term capture/distance reward.

Terminal outcomes replace the ongoing signals while remaining inside the same
two terms: safe landing is (+1,+1), failed contact is (-1,-1).  No third task
term exists, and no post-terminal transition is defined here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np


@dataclass(frozen=True)
class TwoTermRewardConfig:
    capture_weight: float = 0.20
    distance_weight: float = 0.71
    capture_boundary_value: float = -0.5
    capture_outside_scale: float = 1.0
    distance_rate_share: float = 0.30
    distance_rate_scale_m_s: float = 1.5
    distance_scale_m: float = 6.0
    distance_exponent: float = 0.5

    def __post_init__(self) -> None:
        if self.capture_weight <= 0 or self.distance_weight <= 0:
            raise ValueError("two-term reward weights must be positive")
        if not -1.0 < self.capture_boundary_value < 1.0:
            raise ValueError("capture boundary value must be in (-1,1)")
        if self.capture_outside_scale <= 0 or self.distance_rate_scale_m_s <= 0:
            raise ValueError("reward scales must be positive")
        if self.distance_scale_m <= 0 or self.distance_exponent <= 0:
            raise ValueError("distance scales must be positive")
        if not 0.0 <= self.distance_rate_share <= 1.0:
            raise ValueError("distance rate share must be in [0,1]")

    @property
    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True,
                             separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


def capture_signal(normalized_longitudinal_error: float, *, landed=False,
                   failed_contact=False,
                   config: TwoTermRewardConfig | None = None) -> float:
    cfg = config or TwoTermRewardConfig()
    if landed:
        return 1.0
    if failed_contact:
        return -1.0
    q = abs(float(normalized_longitudinal_error))
    if not math.isfinite(q):
        raise ValueError("normalized capture error must be finite")
    boundary = cfg.capture_boundary_value
    if q <= 1.0:
        value = 1.0 - (1.0 - boundary) * q
    else:
        outside = q - 1.0
        value = boundary - (1.0 + boundary) * outside / (
            outside + cfg.capture_outside_scale)
    return float(np.clip(value, -1.0, 1.0))


def distance_signal(distance_m: float, previous_distance_m: float, dt_s: float,
                    *, landed=False, failed_contact=False,
                    config: TwoTermRewardConfig | None = None) -> tuple[float, dict]:
    cfg = config or TwoTermRewardConfig()
    if landed:
        return 1.0, {"distance_rate": 0.0, "proximity": 1.0}
    if failed_contact:
        return -1.0, {"distance_rate": 0.0, "proximity": -1.0}
    distance = float(distance_m)
    previous = float(previous_distance_m)
    dt = float(dt_s)
    if not all(math.isfinite(value) for value in (distance, previous, dt)):
        raise ValueError("distance reward inputs must be finite")
    if distance < 0.0 or previous < 0.0 or dt <= 0.0:
        raise ValueError("distances must be non-negative and dt positive")
    rate = float(np.clip(
        (previous - distance) / (dt * cfg.distance_rate_scale_m_s), -1.0, 1.0))
    proximity = float(np.clip(
        1.0 - (distance / cfg.distance_scale_m) ** cfg.distance_exponent,
        -1.0, 1.0))
    share = cfg.distance_rate_share
    return (share * rate + (1.0 - share) * proximity,
            {"distance_rate": rate, "proximity": proximity})


class TwoTermReward:
    def __init__(self, config: TwoTermRewardConfig | None = None):
        self.config = config or TwoTermRewardConfig()

    def __call__(self, *, normalized_longitudinal_error: float,
                 distance_m: float, previous_distance_m: float, dt_s: float,
                 landed=False, failed_contact=False, terminal=False):
        if landed and failed_contact:
            raise ValueError("a transition cannot be both landed and failed contact")
        if (landed or failed_contact) and not terminal:
            raise ValueError("contact outcome must be terminal")
        capture = capture_signal(
            normalized_longitudinal_error, landed=landed,
            failed_contact=failed_contact, config=self.config)
        distance, detail = distance_signal(
            distance_m, previous_distance_m, dt_s, landed=landed,
            failed_contact=failed_contact, config=self.config)
        parts = {"capture": self.config.capture_weight * capture,
                 "distance": self.config.distance_weight * distance,
                 **detail}
        return float(parts["capture"] + parts["distance"]), parts
