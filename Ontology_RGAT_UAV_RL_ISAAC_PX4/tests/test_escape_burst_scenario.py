"""The escape-burst deck makes the teacher lose, recover and re-follow the pad.

The behaviour-cloning demonstrations are the only flights the student is
cloned from. Flown against the training random walk alone, the privileged PD
teacher is aligned by step 15 and on the pad by step 80 and never loses the
deck, so the clone has seen no recovery. ``training_random_walk_escape_burst``
is the same walk with one straight dash laid over it: the pad accelerates to
the carrier's scenario peak, leaves the pitched camera's frame, and the
teacher's visual-loss rule climbs while it keeps tracking until the pad is
back in view.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "isaac_sim"))

from config_loader import load_config  # noqa: E402
from pad_motion import (BENCHMARK_PEAK_SPEED_M_S, BENCHMARK_SCENARIOS,  # noqa: E402
                        ESCAPE_BURST_DISTANCE_M, ESCAPE_BURST_FALLBACK_S,
                        ESCAPE_BURST_FOLLOW_ALTITUDE_M,
                        ESCAPE_BURST_FOLLOW_LATERAL_M, ESCAPE_BURST_MIN_FOLLOW_S,
                        ESCAPE_BURST_SCENARIO, ESCAPE_BURST_START_WINDOW_S,
                        PadMotionConfig, PadTrajectory, escape_burst_due)

from ontology_rgat.benchmarks.experiment import load_experiment  # noqa: E402
from run_three_pipeline import (_DEMONSTRATION_FLIGHT_KEYS,  # noqa: E402
                                behavior_cloning_settings)

CURRICULUM = 0.20          # the demonstration entry curriculum in the profile
PURSUER_M_S = 0.60         # the PD teacher's horizontal_speed_limit_m_s
DT = 0.1


def _config(trigger="timed"):
    """The minimal profile; ``timed`` gives the closed-form dash the tests
    can read off a reset alone, ``following`` the simulator-triggered one."""
    mapping = load_config(str(ROOT / "config/shin2026-minimal-system.yaml"))
    mapping = dict(mapping)
    mapping["pad"] = {**dict(mapping.get("pad") or {}), "escape_burst_trigger": trigger}
    return PadMotionConfig.from_mapping(mapping)


def _speeds(trajectory, seconds=60.0):
    times = np.arange(0.0, seconds, DT)
    return times, np.array([float(np.linalg.norm(trajectory.pose(t)[1][:2]))
                            for t in times])


def test_the_burst_is_a_named_scenario_everywhere_the_name_is_checked():
    assert ESCAPE_BURST_SCENARIO in BENCHMARK_SCENARIOS
    gateway = (ROOT / "ros2_ws/src/ontology_rgat_px4/ontology_rgat_px4"
               / "protocol.py").read_text()
    assert f'"{ESCAPE_BURST_SCENARIO}"' in gateway, (
        "the gateway validates scenario names on reset; it has to know this one")


@pytest.mark.parametrize("seed", [90000, 90001, 90004, 90007])
def test_the_deck_dashes_to_the_carrier_peak_and_eases_back(seed):
    cfg = _config()
    trajectory = PadTrajectory(cfg)
    info = trajectory.reset(seed, 0.0, speed_scale=CURRICULUM,
                            scenario=ESCAPE_BURST_SCENARIO)
    burst = info["escape_burst"]
    peak = BENCHMARK_PEAK_SPEED_M_S * cfg.benchmark_speed_scale
    assert ESCAPE_BURST_START_WINDOW_S[0] <= burst["start_s"] <= ESCAPE_BURST_START_WINDOW_S[1]
    assert burst["peak_m_s"] == pytest.approx(peak)
    assert burst["distance_m"] >= ESCAPE_BURST_DISTANCE_M
    times, speeds = _speeds(trajectory)
    walk = CURRICULUM * cfg.speed_max_m_s
    before = speeds[times < burst["start_s"] - DT]
    during = speeds[(times > burst["start_s"] + 1.5) & (times < burst["hold_end_s"] - DT)]
    after = speeds[times > burst["end_s"] + DT]
    assert before.max() <= walk + 1e-6, "the walk before the dash is the curriculum walk"
    assert during.min() == pytest.approx(peak, abs=1e-6)
    assert after.max() <= walk + 1e-6, "the deck eases back onto the walk"
    # Nothing jumps: the deck accelerates, it is not teleported.
    positions = np.array([trajectory.pose(t)[0][:2] for t in times])
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    assert steps.max() <= peak * DT * 1.01


def test_the_dash_is_straight_and_the_walk_resumes_from_its_heading():
    cfg = _config()
    trajectory = PadTrajectory(cfg)
    info = trajectory.reset(90002, 0.0, speed_scale=CURRICULUM,
                            scenario=ESCAPE_BURST_SCENARIO)
    burst = info["escape_burst"]
    times = np.arange(burst["start_s"] + 0.2, burst["end_s"] - 0.2, DT)
    headings = np.array([math.atan2(*trajectory.pose(t)[1][[1, 0]]) for t in times])
    unwrapped = np.unwrap(headings)
    assert np.ptp(unwrapped) < 1e-6, "the escape is a straight line"


def test_the_same_seed_gives_the_same_deck_until_the_dash():
    cfg = _config()
    plain, burst = PadTrajectory(cfg), PadTrajectory(cfg)
    plain.reset(90003, 0.0, speed_scale=CURRICULUM, scenario="training_random_walk")
    info = burst.reset(90003, 0.0, speed_scale=CURRICULUM, scenario=ESCAPE_BURST_SCENARIO)
    for t in np.arange(0.0, info["escape_burst"]["start_s"] - DT, DT):
        np.testing.assert_allclose(plain.pose(t)[0], burst.pose(t)[0], atol=1e-9)
    late = info["escape_burst"]["hold_end_s"]
    assert np.linalg.norm(plain.pose(late)[0] - burst.pose(late)[0]) > 1.0


@pytest.mark.parametrize("seed", [90000, 90001, 90002, 90003, 90005, 90006])
def test_the_dash_outruns_the_teacher_far_enough_to_leave_the_frame(seed):
    """Gap a 0.60 m/s pursuer that starts on top of the pad ends up with.

    The camera is pitched 60 deg with a 32 deg vertical half-angle and a 45 deg
    horizontal one: from 1.5 m up it sees about 2.8 m ahead, 1.5 m to the side
    and nothing behind. A relative displacement of 1.5 m therefore leaves the
    frame in every direction but straight ahead, and straight ahead needs
    more altitude than the teacher has left by the time the dash begins.
    """
    cfg = _config()
    trajectory = PadTrajectory(cfg)
    info = trajectory.reset(seed, 0.0, speed_scale=CURRICULUM,
                            scenario=ESCAPE_BURST_SCENARIO)
    times, speeds = _speeds(trajectory)
    gap = 0.0
    for t, speed in zip(times, speeds):
        if info["escape_burst"]["start_s"] <= t <= info["escape_burst"]["end_s"]:
            gap += max(0.0, speed - PURSUER_M_S) * DT
    assert gap >= 1.5
    # And the pursuer can catch up again afterwards: the walk stays below it.
    after = speeds[times > info["escape_burst"]["end_s"] + DT]
    assert after.max() < PURSUER_M_S


def test_the_experiment_demonstrates_against_the_burst_and_fingerprints_it():
    config = load_experiment(
        ROOT / "config/experiments/two_pipeline_comparison.yaml")
    settings = behavior_cloning_settings(config)
    assert settings["scenario"] == ESCAPE_BURST_SCENARIO
    assert "scenario" in _DEMONSTRATION_FLIGHT_KEYS, (
        "a different deck is a different demonstration; the stored set must not be reused")
    # A recovered flight is longer than an approach; the horizon has to hold it.
    assert int(settings["horizon_steps"]) >= 600


# ------------------------------------------------- simulator-triggered dash

def test_the_profile_lets_the_simulator_start_the_dash_while_following():
    cfg = PadMotionConfig.from_mapping(
        load_config(str(ROOT / "config/shin2026-minimal-system.yaml")))
    assert cfg.escape_burst_trigger == "following"
    with pytest.raises(ValueError, match="escape_burst_trigger"):
        _config(trigger="whenever")


def test_an_armed_deck_drives_the_plain_walk_until_it_is_triggered():
    cfg = _config(trigger="following")
    plain = PadTrajectory(_config(trigger="following"))
    plain.reset(90003, 0.0, speed_scale=CURRICULUM, scenario="training_random_walk")
    armed = PadTrajectory(cfg)
    info = armed.reset(90003, 0.0, speed_scale=CURRICULUM, scenario=ESCAPE_BURST_SCENARIO)
    assert info["escape_burst"] is None and armed.escape_burst_armed
    for t in np.arange(0.0, 60.0, 0.5):
        np.testing.assert_allclose(plain.pose(t)[0], armed.pose(t)[0], atol=1e-9)


def test_the_triggered_dash_starts_where_the_deck_is_and_heads_where_told():
    cfg = _config(trigger="following")
    trajectory = PadTrajectory(cfg)
    trajectory.reset(90000, 0.0, speed_scale=0.35, scenario=ESCAPE_BURST_SCENARIO)
    now = 18.27
    before, v_before = (np.array(x) for x in trajectory.pose(now))
    burst = trajectory.trigger_escape_burst(now, heading=1.0)
    assert burst is not None and burst["start_s"] == pytest.approx(18.3)
    assert burst["heading_rad"] == pytest.approx(1.0)
    after, v_after = (np.array(x) for x in trajectory.pose(now))
    np.testing.assert_allclose(after, before, atol=1e-9)
    np.testing.assert_allclose(v_after, v_before, atol=1e-9)
    peak = BENCHMARK_PEAK_SPEED_M_S * cfg.benchmark_speed_scale
    times, speeds = _speeds(trajectory)
    during = (times > burst["start_s"] + 1.5) & (times < burst["hold_end_s"] - DT)
    assert speeds[during].min() == pytest.approx(peak, abs=1e-6)
    headings = np.array([math.atan2(*trajectory.pose(t)[1][[1, 0]])
                         for t in times[during]])
    assert np.ptp(np.unwrap(headings)) < 1e-6 and headings[0] == pytest.approx(1.0)
    assert burst["distance_m"] >= ESCAPE_BURST_DISTANCE_M
    assert speeds[times > burst["end_s"] + DT].max() < PURSUER_M_S
    positions = np.array([trajectory.pose(t)[0][:2] for t in times])
    assert np.linalg.norm(np.diff(positions, axis=0), axis=1).max() <= peak * DT * 1.01
    # One dash per episode; a plain walk or a timed dash has nothing to trigger.
    assert trajectory.trigger_escape_burst(40.0, heading=0.0) is None
    plain = PadTrajectory(cfg)
    plain.reset(1, 0.0, scenario="training_random_walk")
    assert plain.trigger_escape_burst(5.0) is None
    timed = PadTrajectory(_config(trigger="timed"))
    timed.reset(1, 0.0, scenario=ESCAPE_BURST_SCENARIO)
    assert timed.trigger_escape_burst(5.0) is None


def test_the_dash_is_due_when_following_close_and_low_or_at_the_fallback():
    low, high = ESCAPE_BURST_FOLLOW_ALTITUDE_M
    following = dict(lateral_m=ESCAPE_BURST_FOLLOW_LATERAL_M - 0.1,
                     altitude_m=0.5 * (low + high))
    assert escape_burst_due(**following, elapsed_s=ESCAPE_BURST_MIN_FOLLOW_S + 0.1)
    assert not escape_burst_due(**following, elapsed_s=ESCAPE_BURST_MIN_FOLLOW_S - 0.5)
    assert not escape_burst_due(lateral_m=ESCAPE_BURST_FOLLOW_LATERAL_M + 0.5,
                                altitude_m=following["altitude_m"], elapsed_s=10.0)
    assert not escape_burst_due(lateral_m=0.2, altitude_m=high + 1.0, elapsed_s=10.0)
    assert not escape_burst_due(lateral_m=0.2, altitude_m=low - 0.2, elapsed_s=10.0), (
        "a loss inside the flare is the end of a landing, not a recovery")
    assert escape_burst_due(lateral_m=4.0, altitude_m=4.0,
                            elapsed_s=ESCAPE_BURST_FALLBACK_S)
    assert not escape_burst_due(lateral_m=float("nan"), altitude_m=1.0, elapsed_s=10.0)
