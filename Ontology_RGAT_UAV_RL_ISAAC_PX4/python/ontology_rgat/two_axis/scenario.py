"""Seeded continuous CV-CA-CV moving-pad scenarios."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import ScenarioConfig, TimingConfig


@dataclass(frozen=True)
class SeedStreams:
    scenario_seed: int
    sensor_seed: int
    policy_seed: int


def split_seed(seed: int) -> SeedStreams:
    """Derive independent streams without consuming one stream from another."""
    children = np.random.SeedSequence(int(seed)).spawn(3)
    values = [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]
    return SeedStreams(*values)


@dataclass(frozen=True)
class PadScenario:
    seed: int
    x0_m: float
    v1_m_s: float
    a2_m_s2: float
    T1_s: float
    T2_s: float
    T3_s: float
    initial_height_m: float
    parameterization: str = "duration"
    feasibility_label: str = "nominal_range"
    rejection_count: int = 0

    @property
    def v3_m_s(self) -> float:
        return self.v1_m_s + self.a2_m_s2 * self.T2_s

    @property
    def x1_m(self) -> float:
        return self.x0_m + self.v1_m_s * self.T1_s

    @property
    def x2_m(self) -> float:
        return (self.x1_m + self.v1_m_s * self.T2_s
                + 0.5 * self.a2_m_s2 * self.T2_s ** 2)

    @property
    def duration_s(self) -> float:
        return self.T1_s + self.T2_s + self.T3_s

    @property
    def lengths_m(self) -> tuple[float, float, float]:
        return (self.v1_m_s * self.T1_s,
                self.v1_m_s * self.T2_s + 0.5 * self.a2_m_s2 * self.T2_s ** 2,
                self.v3_m_s * self.T3_s)

    def state_at(self, time_s: float) -> tuple[float, float, float]:
        """Return exact ``(x, vx, ax)`` without adding an artificial phase."""
        t = float(np.clip(time_s, 0.0, self.duration_s))
        if t < self.T1_s:
            return self.x0_m + self.v1_m_s * t, self.v1_m_s, 0.0
        if t < self.T1_s + self.T2_s:
            tau = t - self.T1_s
            return (self.x1_m + self.v1_m_s * tau
                    + 0.5 * self.a2_m_s2 * tau ** 2,
                    self.v1_m_s + self.a2_m_s2 * tau, self.a2_m_s2)
        tau = t - self.T1_s - self.T2_s
        return self.x2_m + self.v3_m_s * tau, self.v3_m_s, 0.0

    def to_manifest(self) -> dict:
        record = asdict(self)
        record.update({
            "v3_m_s": self.v3_m_s, "duration_s": self.duration_s,
            "L1_m": self.lengths_m[0], "L2_m": self.lengths_m[1],
            "L3_m": self.lengths_m[2], "time_unit": "s", "length_unit": "m",
        })
        return record


def scenario_from_distances(*, seed: int, v1_m_s: float, a2_m_s2: float,
                            L1_m: float, L2_m: float, L3_m: float,
                            initial_height_m: float, x0_m: float = 0.0
                            ) -> PadScenario:
    values = (v1_m_s, L1_m, L2_m, L3_m)
    if any(float(v) <= 0.0 for v in values) or a2_m_s2 < 0.0:
        raise ValueError("distance mode requires positive speed/lengths and nonnegative a2")
    v3 = math.sqrt(v1_m_s ** 2 + 2.0 * a2_m_s2 * L2_m)
    # Stable for a2=0 and avoids subtracting similar velocities.
    T2 = 2.0 * L2_m / (v1_m_s + v3)
    return PadScenario(seed=int(seed), x0_m=float(x0_m), v1_m_s=float(v1_m_s),
                       a2_m_s2=float(a2_m_s2), T1_s=L1_m / v1_m_s,
                       T2_s=T2, T3_s=L3_m / v3,
                       initial_height_m=float(initial_height_m),
                       parameterization="distance")


def sample_scenario(seed: int, config: ScenarioConfig, timing: TimingConfig,
                    *, max_attempts: int = 100) -> PadScenario:
    """Sample only independent variables; ``v3`` is always derived."""
    if config.parameterization != "duration":
        raise ValueError("random distance-mode ranges must be supplied explicitly")
    stream = split_seed(seed)
    rng = np.random.default_rng(stream.scenario_seed)
    rejected = 0
    for _ in range(max_attempts):
        p = PadScenario(
            seed=int(seed), x0_m=0.0,
            v1_m_s=float(rng.uniform(*config.v1_range_m_s)),
            a2_m_s2=float(rng.uniform(*config.a2_range_m_s2)),
            T1_s=float(rng.uniform(*config.T1_range_s)),
            T2_s=float(rng.uniform(*config.T2_range_s)),
            T3_s=float(rng.uniform(*config.T3_range_s)),
            initial_height_m=float(rng.uniform(*config.initial_height_range_m)),
            rejection_count=rejected)
        valid = (p.duration_s <= timing.mission_duration_limit_s
                 and p.v3_m_s + config.final_speed_margin_m_s
                 <= config.max_final_pad_speed_m_s)
        if valid:
            return p
        rejected += 1
    raise RuntimeError(f"could not sample a feasible scenario in {max_attempts} attempts")


def build_seed_manifest(seeds: Iterable[int], split: str,
                        config: ScenarioConfig, timing: TimingConfig) -> dict:
    if split not in {"train", "validation", "test", "stress"}:
        raise ValueError("unknown manifest split")
    sampled = [sample_scenario(int(seed), config, timing) for seed in seeds]
    if split == "stress":
        # Height is outside the training interval while pad kinematics retain
        # all mission-duration and final-speed feasibility checks.
        sampled = [replace(item,
                           initial_height_m=config.initial_height_range_m[1] + 1.0,
                           feasibility_label="out_of_range_height_stress")
                   for item in sampled]
    scenarios = [item.to_manifest() for item in sampled]
    payload = {"schema": "two_axis_scenario_manifest/1", "split": split,
               "scenarios": scenarios}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def write_seed_manifest(path: str | Path, manifest: dict) -> None:
    Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
