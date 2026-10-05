"""Training-only difficulty schedule, shared bit-for-bit by every policy arm.

The schedule exists because a randomly initialised policy starting 4-8 m above
an accelerating pad never reaches a touchdown, so PPO sees no variation in the
terminal term and has nothing to learn a landing strategy from. It changes only
the *training* episode distribution and the training-only tolerance ramp.

Two invariants hold by construction and are asserted in the regression suite:

* ``difficulty == 1.0`` reproduces the nominal configuration exactly, so
  validation and test never run an easier task than the declared contract; and
* the schedule carries no arm identity, so the baseline, the flattened-graph
  control and the R-GAT arm receive the same episodes from the same seeds.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable

import numpy as np

from .config import (CurriculumConfig, ExperimentConfig, RewardConfig,
                     SafetyConfig, ScenarioConfig)


class EpisodeRole(str, Enum):
    """Why an episode was scheduled; replays never drive promotion."""

    ACTIVE = "active"
    EASY_REPLAY = "easy_replay"
    BRIDGE_REPLAY = "bridge_replay"


def _lerp(start: float, end: float, fraction: float) -> float:
    return float(start + (end - start) * float(fraction))


def _lerp_pair(start: tuple[float, float], end: tuple[float, float],
               fraction: float) -> tuple[float, float]:
    return (_lerp(start[0], end[0], fraction), _lerp(start[1], end[1], fraction))


@dataclass(frozen=True)
class StageConfigs:
    """The three configuration objects an episode at one difficulty needs."""

    difficulty: float
    scenario: ScenarioConfig
    safety: SafetyConfig
    reward: RewardConfig


def stage_configs(config: ExperimentConfig, difficulty: float) -> StageConfigs:
    """Interpolate the training distribution and the tolerance ramp."""
    d = float(np.clip(difficulty, 0.0, 1.0))
    curriculum = config.curriculum
    if not curriculum.enabled:
        d = 1.0
    if d == 1.0:
        # Avoid floating-point interpolation drift at the nominal contract.
        return StageConfigs(1.0, config.scenario, config.safety, config.reward)
    # Only the pad kinematics interpolate. The start altitude stays at the
    # nominal 4-8 m at every difficulty: the camera footprint is 0.7*h, so a
    # lower start shrinks the frame and makes the task harder, not easier --
    # measured, with the PN reference falling from 0.42 to 0.04.
    scenario = replace(
        config.scenario,
        v1_range_m_s=_lerp_pair(curriculum.start_v1_range_m_s,
                                config.scenario.v1_range_m_s, d),
        a2_range_m_s2=_lerp_pair(curriculum.start_a2_range_m_s2,
                                 config.scenario.a2_range_m_s2, d),
        # Even the easiest episode starts inside or just before the pad
        # acceleration event, so it cannot be solved by pure vertical descent
        # onto a constant-speed pad.
        T1_range_s=_lerp_pair(curriculum.start_T1_range_s,
                              config.scenario.T1_range_s, d),
        initial_height_range_m=(config.scenario.initial_height_range_m
            if curriculum.start_height_range_m is None else _lerp_pair(
                curriculum.start_height_range_m, config.scenario.initial_height_range_m, d)))
    safety = replace(
        config.safety,
        touchdown_relative_speed_m_s=_lerp(
            curriculum.start_touchdown_relative_speed_m_s,
            config.safety.touchdown_relative_speed_m_s, d),
        touchdown_vertical_speed_m_s=_lerp(
            curriculum.start_touchdown_vertical_speed_m_s,
            config.safety.touchdown_vertical_speed_m_s, d))
    nominal = dict(config.reward.terminal_bonus)
    # The unsafe penalty tightens with difficulty. Starting at the nominal -40
    # drowns the rare early success; starting too soft makes a fast impact the
    # cheapest way out of an easy episode.
    terminal = tuple(
        (name, _lerp(curriculum.start_unsafe_contact_penalty, value, d)
         if name in config.reward.unsafe_reasons else value)
        for name, value in config.reward.terminal_bonus)
    reward = replace(config.reward, terminal_bonus=terminal)
    assert set(dict(terminal)) == set(nominal)
    return StageConfigs(d, scenario, safety, reward)


def replay_difficulty(active: float, role: EpisodeRole) -> float:
    """Difficulty of a replay episode drawn against an active difficulty."""
    if role is EpisodeRole.ACTIVE:
        return float(active)
    if role is EpisodeRole.EASY_REPLAY:
        return 0.0
    return 0.5 * float(active)


class CurriculumScheduler:
    """Difficulty state machine driven only by active-difficulty outcomes."""

    def __init__(self, config: CurriculumConfig, *, difficulty: float = 0.0):
        self.config = config
        self.difficulty = 1.0 if not config.enabled else float(
            np.clip(difficulty, 0.0, 1.0))
        self._window: list[bool] = []
        self.episodes = 0
        self.promotions = 0

    def role_for(self, episode_index: int) -> EpisodeRole:
        """Deterministic interleave; identical for every arm and seed."""
        if not self.config.enabled or self.difficulty <= 0.0:
            return EpisodeRole.ACTIVE
        easy = max(1, round(1.0 / max(self.config.easy_replay_fraction, 1e-9)))
        bridge = max(1, round(1.0 / max(self.config.bridge_replay_fraction, 1e-9)))
        index = int(episode_index)
        if index % easy == easy - 1:
            return EpisodeRole.EASY_REPLAY
        if index % bridge == bridge - 2:
            return EpisodeRole.BRIDGE_REPLAY
        return EpisodeRole.ACTIVE

    def set_budget_progress(self, fraction: float) -> None:
        """Shared deterministic floor; failed promotions cannot stall forever."""
        if not self.config.enabled or not self.config.scheduled_floor:
            return
        floor = float(np.clip((fraction-self.config.floor_start_fraction) / (
            self.config.full_difficulty_fraction-self.config.floor_start_fraction), 0, 1))
        if floor > self.difficulty:
            self.difficulty = floor
            self._window.clear()

    def record(self, *, role: EpisodeRole, landed: bool) -> bool:
        """Register one finished episode; return True if difficulty advanced.

        Only episodes at the active difficulty count. An easy replay that lands
        says nothing about the task the curriculum is trying to promote past.
        """
        self.episodes += 1
        if role is not EpisodeRole.ACTIVE or not self.config.enabled:
            return False
        self._window.append(bool(landed))
        if len(self._window) < self.config.promotion_window_episodes:
            return False
        rate = float(np.mean(self._window))
        self._window.clear()
        if rate >= self.config.promotion_landing_rate and self.difficulty < 1.0:
            self.difficulty = float(min(
                1.0, self.difficulty + self.config.difficulty_step))
            self.promotions += 1
            return True
        return False

    @property
    def at_nominal(self) -> bool:
        return self.difficulty >= 1.0

    def state(self) -> dict[str, float | int]:
        return {"difficulty": self.difficulty, "episodes": self.episodes,
                "promotions": self.promotions,
                "window_filled": len(self._window)}


def checkpoint_is_eligible(difficulty: float) -> bool:
    """Only a policy trained at the full nominal task may be selected."""
    return float(difficulty) >= 1.0


def landing_rates(records: Iterable[tuple[EpisodeRole, float, str]]
                  ) -> dict[str, float]:
    """Split the training landing rate the MATLAB study found uninformative.

    ``records`` are ``(role, difficulty, terminal_reason)`` triples. Reporting
    only the overall rate hides whether nominal-difficulty landing was learned.
    """
    rows = list(records)
    if not rows:
        return {"overall": 0.0, "active_curriculum": 0.0, "nominal": 0.0,
                "safe_abort": 0.0, "task_timeout": 0.0, "unsafe": 0.0,
                "episodes": 0.0}
    unsafe = {"UNSAFE_CONTACT", "UNAUTHORIZED_CONTACT", "MISSED_PAD_CONTACT",
              "SAFETY_ENVELOPE_VIOLATION"}

    def rate(subset: list[tuple[EpisodeRole, float, str]], name: str) -> float:
        if not subset:
            return 0.0
        if name == "unsafe":
            return float(np.mean([row[2] in unsafe for row in subset]))
        return float(np.mean([row[2] == name for row in subset]))

    active = [row for row in rows if row[0] is EpisodeRole.ACTIVE]
    nominal = [row for row in rows if row[1] >= 1.0]
    return {
        "overall": rate(rows, "SUCCESS"),
        "active_curriculum": rate(active, "SUCCESS"),
        "nominal": rate(nominal, "SUCCESS"),
        "safe_abort": rate(rows, "SAFE_ABORT"),
        "task_timeout": rate(rows, "TASK_TIMEOUT"),
        "unsafe": rate(rows, "unsafe"),
        "episodes": float(len(rows)),
    }
