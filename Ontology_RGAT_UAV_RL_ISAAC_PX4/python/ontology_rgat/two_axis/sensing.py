"""Body-fixed camera geometry and reproducible detector perturbations."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config import CameraConfig


@dataclass(frozen=True)
class Projection:
    depth_m: float
    lateral_m: float
    bearing_rad: float
    geometric_visible: bool


def project_pad(ex_m: float, h_m: float, theta_rad: float,
                config: CameraConfig) -> Projection:
    """Project the pad using the explicit x-z/body-camera convention."""
    if h_m <= 1e-9:
        return Projection(0.0, 0.0, 0.0, False)
    depth = h_m * math.cos(theta_rad) - ex_m * math.sin(theta_rad)
    lateral = ex_m * math.cos(theta_rad) + h_m * math.sin(theta_rad)
    bearing = math.atan2(lateral, depth)
    distance = math.hypot(ex_m, h_m)
    visible = (depth > 0.0 and distance <= config.maximum_range_m
               and abs(bearing) <= 0.5 * config.fov_rad)
    return Projection(float(depth), float(lateral), float(bearing), bool(visible))


@dataclass(frozen=True)
class DropoutSchedule:
    intervals_s: tuple[tuple[float, float, str], ...] = ()
    pitch_event: tuple[float, float, float] | None = None

    def pitch_rate_at(self, time_s: float) -> float:
        if self.pitch_event is None:
            return 0.0
        start,end,rate = self.pitch_event
        return rate if start <= time_s < end else 0.0

    @classmethod
    def reference_mixture(cls, rng, duration, *, enabled=True):
        """Upstream sampleEvents: 50% clean/25% short/25% sustained losses.

        A separate 25% bounded pitch event is exogenous evaluator metadata,
        never a policy feature. Every arm samples before its first decision.
        """
        if not enabled:
            return cls()
        intervals, pitch = (), None
        draw = rng.random()
        if draw >= .5:
            start = float(rng.uniform(.5, max(.5, duration-5)))
            length = float(rng.uniform(.2,.5) if draw < .75 else rng.uniform(3.5,5.))
            cause = "short_detector_dropout" if draw < .75 else "sustained_detector_dropout"
            intervals = ((start,min(start+length,duration),cause),)
        if rng.random() < .25:
            start = float(rng.uniform(.5,max(.5,duration-1)))
            pitch = (start,min(start+float(rng.uniform(.15,.4)),duration),
                     float(rng.uniform(-1,1)*math.radians(2)))
        return cls(intervals,pitch)

    def cause_at(self, time_s: float) -> str | None:
        for start, end, cause in self.intervals_s:
            if start <= time_s < end:
                return cause
        return None

    @classmethod
    def randomized(cls, rng: np.random.Generator, mission_duration_s: float,
                   *, enabled: bool = True) -> "DropoutSchedule":
        if not enabled or mission_duration_s < 8.0:
            return cls()
        short_start = float(rng.uniform(2.0, min(8.0, mission_duration_s - 4.0)))
        sustained_start = float(rng.uniform(
            min(short_start + 2.0, mission_duration_s - 3.5),
            mission_duration_s - 3.0))
        return cls(((short_start, short_start + 0.3, "short_detector_dropout"),
                    (sustained_start, min(mission_duration_s,
                                          sustained_start + 3.2),
                     "sustained_detector_dropout")))


@dataclass(frozen=True)
class PadMeasurement:
    timestamp_s: float
    detected: bool
    bearing_rad: float
    bearing_valid: bool
    relative_x_m: float
    relative_x_valid: bool
    confidence: float
    geometric_visible: bool
    injected_cause: str | None = None


def observe_pad(*, timestamp_s: float, ex_m: float, h_m: float,
                theta_rad: float, config: CameraConfig,
                rng: np.random.Generator,
                dropout_schedule: DropoutSchedule = DropoutSchedule()
                ) -> PadMeasurement:
    projection = project_pad(ex_m, h_m, theta_rad, config)
    # Consume the same two draws at every sample, even when the pad is hidden.
    # Otherwise different policies' visibility histories shift their streams
    # and the supposedly paired sensor noise is no longer paired in time.
    bearing_noise, position_noise = rng.standard_normal(2)
    cause = dropout_schedule.cause_at(timestamp_s)
    detected = projection.geometric_visible and cause is None
    if not detected:
        return PadMeasurement(float(timestamp_s), False, 0.0, False, 0.0,
                              False, 0.0, projection.geometric_visible, cause)
    bearing = projection.bearing_rad + float(bearing_noise * config.bearing_noise_std_rad)
    relative_x = ex_m + float(position_noise * config.relative_position_noise_std_m)
    normalized = abs(projection.bearing_rad) / (0.5 * config.fov_rad)
    confidence = float(np.clip(1.0 - 0.35 * normalized, 0.05, 1.0))
    return PadMeasurement(float(timestamp_s), True, bearing, True, relative_x,
                          True, confidence, True, None)
