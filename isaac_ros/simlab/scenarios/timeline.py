"""The deterministic event schedule of one episode.

Nothing is broadcast at runtime. The simulator gates its camera publishers from
this table and the dataset collector, in a different process and a different
interpreter, rebuilds the identical table from the same episode YAML. Both sides
therefore agree on when an observation was supposed to exist -- which is what
lets the collector label a missing frame as "sensor blackout, track alive"
rather than "the drone is gone".
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from simlab.config.schema import GATEABLE_CAMERAS, SceneConfig, ScenariosConfig


_MASK64 = 0xFFFFFFFFFFFFFFFF


def stable_unit(*values: int) -> float:
    """A repeatable float in [0, 1) from integers -- same answer in any process.

    ``random.Random`` would do, but constructing one per camera per simulation
    step is wasteful and its stream is not guaranteed across interpreters. This
    is the murmur3 64-bit finalizer, so a one-bit change in the last argument
    still moves the whole output.
    """
    mixed = 0x9E3779B97F4A7C15
    for value in values:
        mixed = ((mixed ^ (int(value) & _MASK64)) * 0x100000001B3) & _MASK64
    mixed = ((mixed ^ (mixed >> 33)) * 0xFF51AFD7ED558CCD) & _MASK64
    mixed = ((mixed ^ (mixed >> 33)) * 0xC4CEB9FE1A85EC53) & _MASK64
    mixed ^= mixed >> 33
    return (mixed >> 11) / float(1 << 53)


@dataclass(frozen=True)
class SensorGap:
    """One scheduled loss of observation on one camera."""

    camera: str
    start_s: float
    end_s: float
    kind: str = "blackout"

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def contains(self, t: float) -> bool:
        return self.start_s <= t < self.end_s

    def as_dict(self) -> Dict[str, object]:
        return {
            "camera": self.camera,
            "start_s": round(self.start_s, 3),
            "end_s": round(self.end_s, 3),
            "duration_s": round(self.duration_s, 3),
            "kind": self.kind,
        }


@dataclass(frozen=True)
class ScenarioTimeline:
    """What happens, and when, for the episode named in the scene config."""

    scenario: str
    environment: str
    seed: int
    episode_id: str
    duration_s: float
    event_start_s: float
    gaps: Tuple[SensorGap, ...] = ()
    frame_drop_probability: float = 0.0
    cameras: Tuple[str, ...] = GATEABLE_CAMERAS

    # -- construction ------------------------------------------------------
    @classmethod
    def from_scene(cls, scene: SceneConfig) -> "ScenarioTimeline":
        return cls.plan(scene.scenarios, scene.world.environment)

    @classmethod
    def plan(cls, cfg: ScenariosConfig, environment: str) -> "ScenarioTimeline":
        scenario = cfg.active or "mutual_occlusion"
        seed = cfg.seed if cfg.seed is not None else random.SystemRandom().randrange(0, 2**31)
        duration = cfg.episode_duration_s
        gaps: Tuple[SensorGap, ...] = ()
        probability = 0.0
        if scenario == "sensor_dropout":
            dropout = cfg.sensor_dropout
            gaps = tuple(
                _schedule_gaps(dropout, seed, cfg.event_start_s, duration)
            )
            probability = dropout.frame_drop_probability
        return cls(
            scenario=scenario,
            environment=environment,
            seed=seed,
            episode_id=cfg.episode_id or f"{scenario}_{environment}_{seed:08d}",
            duration_s=duration,
            event_start_s=cfg.event_start_s,
            gaps=gaps,
            frame_drop_probability=probability,
            cameras=tuple(cfg.sensor_dropout.cameras) if scenario == "sensor_dropout" else (),
        )

    # -- queries -----------------------------------------------------------
    def blackout(self, camera: str, t: float) -> SensorGap | None:
        """The gap covering ``t`` on ``camera``, or None while it is observing."""
        for gap in self.gaps:
            if gap.camera == camera and gap.contains(t):
                return gap
        return None

    def frame_dropped(self, camera: str, step: int) -> bool:
        """Isolated single-frame loss -- deterministic in the step index."""
        if self.frame_drop_probability <= 0.0 or camera not in self.cameras:
            return False
        index = GATEABLE_CAMERAS.index(camera) if camera in GATEABLE_CAMERAS else 0
        return stable_unit(self.seed, index, step) < self.frame_drop_probability

    def observing(self, camera: str, t: float, step: int | None = None) -> bool:
        """False while the camera is dark for any reason."""
        if self.blackout(camera, t) is not None:
            return False
        return not (step is not None and self.frame_dropped(camera, step))

    def gap_elapsed(self, camera: str, t: float) -> float | None:
        """Seconds since the current blackout began, or None if observing."""
        gap = self.blackout(camera, t)
        return None if gap is None else max(0.0, t - gap.start_s)

    def next_gap(self, camera: str, t: float) -> SensorGap | None:
        upcoming = [gap for gap in self.gaps if gap.camera == camera and gap.start_s >= t]
        return min(upcoming, key=lambda gap: gap.start_s) if upcoming else None

    def describe(self) -> Dict[str, object]:
        return {
            "scenario": self.scenario,
            "environment": self.environment,
            "episode_id": self.episode_id,
            "seed": self.seed,
            "duration_s": self.duration_s,
            "event_start_s": self.event_start_s,
            "frame_drop_probability": self.frame_drop_probability,
            "gaps": [gap.as_dict() for gap in self.gaps],
        }


def _schedule_gaps(
    dropout, seed: int, event_start_s: float, duration_s: float
) -> List[SensorGap]:
    """Lay non-overlapping blackouts across the episode, one per time slot.

    Slotting rather than rejection sampling keeps the schedule reproducible and
    guarantees every blackout ends before the next begins, so an episode can
    never degenerate into one long outage that no tracker could survive.
    """
    if dropout.dropout_count <= 0:
        return []
    tail_margin = max(2.0, dropout.max_dropout_s)
    window_start = event_start_s
    window_end = duration_s - tail_margin
    if window_end <= window_start:
        return []
    slot = (window_end - window_start) / dropout.dropout_count
    if slot <= 0.1:
        return []
    generator = random.Random(seed * 1000003 + 17)
    cameras: Sequence[str] = list(dropout.cameras)
    gaps: List[SensorGap] = []
    for index in range(dropout.dropout_count):
        span = min(dropout.max_dropout_s, max(dropout.min_dropout_s, slot * 0.55))
        length = generator.uniform(dropout.min_dropout_s, span)
        latest = window_start + slot * (index + 1) - length
        earliest = window_start + slot * index
        start = generator.uniform(earliest, max(earliest, latest))
        targets = cameras if dropout.simultaneous else [cameras[index % len(cameras)]]
        for camera in targets:
            gaps.append(SensorGap(camera=camera, start_s=start, end_s=start + length))
    return sorted(gaps, key=lambda gap: (gap.start_s, gap.camera))
