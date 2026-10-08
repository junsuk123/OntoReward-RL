"""Stress scenarios: EVALUATION-ONLY departures from the nominal contract.

Why they exist: on the nominal contract the graph arms land ~100 % after
behaviour cloning (results/minimal_pipeline_20261007), so the comparison has
no headroom. These scenarios widen it. They are never used for training,
validation or checkpoint selection -- difficulty 1.0 stays the contract, and a
stress number is always quoted with its scenario name.

Each scenario scales a quantity the nominal plant already samples, after the
nominal reset, so the seed's nominal draws are untouched and a stress episode
differs from its nominal twin only in the named factor:

``fast_pad``     pad speed and acceleration x2 (0.6-1.6 m/s, 0.2-0.8 m/s^2)
``strong_wind``  constant external force x2 (up to 1.5 N per axis, ~1 m/s^2)
``poor_vision``  extra frame misses: 15 % single misses plus 0.5 s bursts
                 starting on 4 % of frames (a separate RNG stream)
``combined``     all three

``RandomizedStressBackend`` is the TRAINING-side counterpart (domain
randomization, "dr"): each episode draws its own factors uniformly between
nominal and the stress level above, from a stream keyed by the seed. It is
for training only; validation and checkpoint selection stay nominal, and the
named scenarios above stay the test.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..spatial.environment import LocalBackend


@dataclass(frozen=True)
class StressScenario:
    name: str
    pad_speed_scale: float = 1.0
    force_scale: float = 1.0
    extra_miss_probability: float = 0.0
    burst_start_probability: float = 0.0
    burst_frames: int = 10


SCENARIOS = {
    "fast_pad": StressScenario("fast_pad", pad_speed_scale=2.0),
    "strong_wind": StressScenario("strong_wind", force_scale=2.0),
    "poor_vision": StressScenario("poor_vision", extra_miss_probability=0.15,
                                  burst_start_probability=0.04),
    "combined": StressScenario("combined", pad_speed_scale=2.0, force_scale=2.0,
                               extra_miss_probability=0.15, burst_start_probability=0.04),
}


class StressBackend(LocalBackend):
    name = "local-spatial-stress"

    def __init__(self, cfg, scenario: StressScenario):
        super().__init__(cfg, difficulty=1.0)
        self.scenario = scenario
        self.name = f"local-spatial-stress:{scenario.name}"

    def reset(self, seed):
        self.stress_rng = None          # the nominal reset's own frame is unstressed
        _, _ = super().reset(seed)
        s = self.scenario
        self.v0 *= s.pad_speed_scale
        self.a2 *= s.pad_speed_scale
        self.pad_velocity = self.direction * self.v0
        if self.domain_sample is not None and s.force_scale != 1.0:
            self.domain_sample = replace(
                self.domain_sample,
                external_force_n=np.asarray(self.domain_sample.external_force_n) * s.force_scale)
        self.stress_rng = np.random.default_rng([int(seed), 0x57E55])
        self.burst_left = 0
        return self.measure(), self.truth()

    def _optical_quality(self, r, q):
        visible, confidence = super()._optical_quality(r, q)
        if getattr(self, "stress_rng", None) is None:
            return visible, confidence
        s = self.scenario
        # Draw every frame so the stream does not depend on geometry.
        miss, burst = self.stress_rng.uniform(), self.stress_rng.uniform()
        if self.burst_left > 0:
            self.burst_left -= 1
            return False, 0.0
        if burst < s.burst_start_probability:
            self.burst_left = s.burst_frames - 1
            return False, 0.0
        if miss < s.extra_miss_probability:
            return False, 0.0
        return visible, confidence


class RandomizedStressBackend(StressBackend):
    """Per-episode factors in [nominal, stress], drawn from the episode seed."""
    name = "local-spatial-dr"

    def __init__(self, cfg):
        super().__init__(cfg, StressScenario("dr"))
        self.name = "local-spatial-dr"

    def reset(self, seed):
        rng = np.random.default_rng([int(seed), 0xD12])
        self.scenario = StressScenario(
            "dr", pad_speed_scale=float(rng.uniform(1.0, 2.0)),
            force_scale=float(rng.uniform(1.0, 2.0)),
            extra_miss_probability=float(rng.uniform(0.0, 0.15)),
            burst_start_probability=float(rng.uniform(0.0, 0.04)))
        return super().reset(seed)
