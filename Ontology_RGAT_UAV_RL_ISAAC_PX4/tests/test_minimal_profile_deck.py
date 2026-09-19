"""The analytic deck has to stay flyable across a long episode sequence.

Both failures below were found by a live run dying, not by reasoning:

* the deck snapped back to the origin at every reset while the drone was
  deliberately left flying, opening a 37 m gap the entry gate could not close;
* leaving the deck where it stands instead turns the sequence into a random
  walk that ends pinned on the arena clamp with zero velocity, which is a
  stationary target wearing a moving target's name.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "isaac_sim"))

from config_loader import load_config  # noqa: E402
from pad_motion import PadMotionConfig, PadTrajectory  # noqa: E402

SCENARIOS = ("straight_8mps", "circle", "linear_acceleration_wave",
             "zigzag", "u_turn", "vertical_heave_boat")
EPISODE_S = 30.0


def _profile():
    return load_config(str(ROOT / "config/shin2026-minimal-system.yaml"))


def _fly(trajectory, episodes, seed0=3000):
    """Return (reset jumps, sampled radii) over a sequence of episodes."""
    clock, jumps, radii, previous = 0.0, [], [], None
    for index in range(episodes):
        trajectory.reset(seed0 + index, clock,
                         scenario=SCENARIOS[index % len(SCENARIOS)])
        start = np.asarray(trajectory.pose(clock)[0][:2], dtype=float)
        if previous is not None:
            jumps.append(float(np.linalg.norm(start - previous)))
        for fraction in (0.25, 0.5, 0.75, 1.0):
            radii.append(float(np.linalg.norm(
                trajectory.pose(clock + EPISODE_S * fraction)[0][:2])))
        clock += EPISODE_S
        previous = np.asarray(trajectory.pose(clock)[0][:2], dtype=float)
    return jumps, radii


def test_the_deck_does_not_teleport_between_episodes():
    """The drone is kept flying across resets; the deck must not jump away."""
    trajectory = PadTrajectory(PadMotionConfig.from_mapping(_profile()))
    jumps, _ = _fly(trajectory, 40)
    assert max(jumps) == pytest.approx(0.0, abs=1e-6)


def test_the_deck_never_reaches_the_arena_clamp():
    """The clamp zeroes deck velocity, so reaching it ends the moving-deck task."""
    profile = _profile()
    config = PadMotionConfig.from_mapping(profile)
    _, radii = _fly(PadTrajectory(config), 300)
    assert max(radii) < config.arena_radius_m
    # And the drone, which has to follow it, stays inside its own bound.
    assert max(radii) < float(profile["landing"]["world_xy_limit_m"])


def test_a_seeded_restart_still_re_anchors_the_deck():
    """The teleporting behaviour remains available and is still what it was."""
    profile = _profile()
    profile["pad"] = dict(profile["pad"], route_start="seeded")
    trajectory = PadTrajectory(PadMotionConfig.from_mapping(profile))
    trajectory.reset(11, 0.0, scenario="straight_8mps")
    moved = np.asarray(trajectory.pose(EPISODE_S)[0][:2], dtype=float)
    assert np.linalg.norm(moved) > 10.0
    trajectory.reset(12, EPISODE_S, scenario="straight_8mps")
    restarted = np.asarray(trajectory.pose(EPISODE_S)[0][:2], dtype=float)
    assert np.linalg.norm(restarted) == pytest.approx(0.0, abs=1e-6)
