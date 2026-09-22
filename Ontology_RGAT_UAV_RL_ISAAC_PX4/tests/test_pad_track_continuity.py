"""The deck's reported track must be the track it actually drives.

Two defects found on 2026-09-22 from a dashboard trajectory panel that showed
the escape-burst deck as a folded band instead of the straight line that
scenario drives:

* ``PadTrajectory.pose`` interpolated between an already-anchored position and
  an un-anchored one, so the reported deck slid the whole ``route_start:
  continue`` anchor backwards within every motion-update interval and snapped
  back at the next. Silent on the first episode, where the anchor is zero.
* every replica of one arm reported through that arm's primary-pair monitor,
  so a single pair panel received two vehicles 150 m apart.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "isaac_sim"))

from config_loader import load_config
from pad_motion import PadMotionConfig, PadTrajectory
from ontology_rgat.ppo.recurrent_train import flown_episode_batches


def _trajectory():
    cfg = PadMotionConfig.from_mapping(
        load_config(ROOT / "config/shin2026-minimal-system.yaml"))
    assert cfg.mode == "random_walk", "the profile must use the scenario tracks"
    assert cfg.route_start == "continue", "the anchor is what this guards"
    return cfg, PadTrajectory(cfg)


def test_the_deck_never_jumps_inside_a_motion_update_interval():
    """The bug: a 0.1 s interval swept the 15 m anchor and snapped back."""
    cfg, traj = _trajectory()
    traj.reset(seed=70000, sim_time=0.0, scenario="straight_escape_burst")
    # A second episode is what makes the anchor non-zero; the first is not a
    # regression test at all, which is why this went unnoticed.
    traj.reset(seed=70001, sim_time=30.0, scenario="straight_escape_burst")
    anchor = float(np.linalg.norm(traj.random_walk_origin[:2]))
    assert anchor > 1.0, "the carried anchor must be large enough to expose it"

    dt = cfg.motion_update_dt_s
    times = 30.0 + np.arange(0, 80) * (dt / 4.0)
    points = np.array([traj.pose(t)[0][:2] for t in times])
    moves = np.linalg.norm(np.diff(points, axis=0), axis=1)
    expected = traj.speed * dt / 4.0
    assert moves.max() == pytest.approx(expected, rel=1e-6)
    assert moves.min() == pytest.approx(expected, rel=1e-6)


def test_the_escape_burst_deck_drives_a_straight_line_every_episode():
    """What the scenario is for: one deck, one heading, no folding."""
    _cfg, traj = _trajectory()
    time = 0.0
    for episode in range(8):
        traj.reset(seed=70000 + episode, sim_time=time,
                   scenario="straight_escape_burst")
        times = time + np.arange(0, 300) * 0.1
        points = np.array([traj.pose(t)[0][:2] for t in times])
        deltas = np.diff(points, axis=0)
        lengths = np.linalg.norm(deltas, axis=1)
        path, displacement = lengths.sum(), float(
            np.linalg.norm(points[-1] - points[0]))
        assert displacement == pytest.approx(path, rel=1e-9), (
            f"episode {episode} folds: {path:.2f} m driven for "
            f"{displacement:.2f} m of ground")
        moving = lengths > 1e-9
        headings = np.degrees(np.arctan2(deltas[moving, 1], deltas[moving, 0]))
        assert headings.max() - headings.min() < 1e-6
        time = times[-1] + 0.1


def test_the_anchored_deck_stays_continuous_across_the_interval_boundary():
    """Sampling either side of an index boundary must not step twice."""
    cfg, traj = _trajectory()
    traj.reset(seed=70000, sim_time=0.0, scenario="straight_escape_burst")
    traj.reset(seed=70002, sim_time=30.0, scenario="straight_escape_burst")
    dt = cfg.motion_update_dt_s
    before, _ = traj.pose(30.0 + 3 * dt - 1e-6)
    after, _ = traj.pose(30.0 + 3 * dt + 1e-6)
    assert float(np.linalg.norm(after[:2] - before[:2])) < 1e-4


# ------------------------------------------------- replica telemetry routing

class _StubModel:
    """Only ``state_dict`` is touched: the batch snapshots the rollout policy."""

    def state_dict(self):
        return {}

def test_every_replica_reports_to_the_pair_it_is_actually_flying():
    """One panel per physical pair, not one per arm."""
    seen = []

    def collect(env, _model, _method, seed, *, monitor, **_kwargs):
        seen.append((env, monitor, seed))
        return [{"reward": 0.0}], {"paper_success": 0}

    import ontology_rgat.ppo.recurrent_train as rt
    original = rt.collect_episode_resilient
    rt.collect_episode_resilient = collect
    try:
        envs = ["pair-0-env", "pair-1-env"]
        monitors = ["pair-0-monitor", "pair-1-monitor"]
        batches = list(flown_episode_batches(
            envs, model=_StubModel(), method="shin_se_fixed",
            seed_list=[11, 12], completed=0, scenarios=("deck",),
            warmup_episodes=0,
            curriculum=SimpleNamespace(update=lambda _n: 1.0),
            collect_episode_kwargs={"monitor": "primary-monitor"},
            env_monitors=monitors))
    finally:
        rt.collect_episode_resilient = original

    assert len(batches) == 2
    assert dict((env, monitor) for env, monitor, _seed in seen) == {
        "pair-0-env": "pair-0-monitor", "pair-1-env": "pair-1-monitor"}


def test_without_replica_monitors_the_single_monitor_is_still_used():
    """The sequential path is unchanged when no per-pair monitors are given."""
    seen = []

    def collect(env, _model, _method, seed, *, monitor, **_kwargs):
        seen.append((env, monitor))
        return [{"reward": 0.0}], {"paper_success": 0}

    import ontology_rgat.ppo.recurrent_train as rt
    original = rt.collect_episode_resilient
    rt.collect_episode_resilient = collect
    try:
        list(flown_episode_batches(
            ["only-env"], model=_StubModel(), method="shin_se_fixed",
            seed_list=[11], completed=0, scenarios=("deck",),
            warmup_episodes=0,
            curriculum=SimpleNamespace(update=lambda _n: 1.0),
            collect_episode_kwargs={"monitor": "primary-monitor"}))
    finally:
        rt.collect_episode_resilient = original

    assert seen == [("only-env", "primary-monitor")]


def test_train_live_rejects_a_monitor_list_that_does_not_match_the_pairs():
    from ontology_rgat.ppo.recurrent_train import train_live

    with pytest.raises(ValueError, match="one monitor per environment"):
        train_live(lambda: None, object(), "shin_se_fixed", [1], Path("/tmp"),
                   config_hash="x", env_factories=[lambda: None, lambda: None],
                   env_monitors=[object()])


# ------------------------------------------------------------ closed track

def test_the_closed_track_is_straights_joined_by_constant_speed_arcs():
    from pad_motion import stadium_track

    straight, radius = 20.0, 6.0
    turn = math.pi * radius
    arc = np.array([0.0, straight / 2, straight, straight + turn / 2,
                    straight + turn, 1.5 * straight + turn,
                    2 * straight + turn, 2 * straight + 1.5 * turn])
    heading, curvature = stadium_track(arc, straight, radius)
    # straights carry no curvature at all; the arcs carry exactly 1/R
    assert curvature.tolist() == [0.0, 0.0, 1 / radius, 1 / radius, 0.0, 0.0,
                                  1 / radius, 1 / radius]
    assert heading[0] == pytest.approx(0.0)
    assert heading[4] == pytest.approx(math.pi)
    # one lap returns to the starting heading
    perimeter = 2.0 * (straight + turn)
    lap, _ = stadium_track(np.array([perimeter, 2 * perimeter]), straight, radius)
    assert lap.tolist() == pytest.approx([0.0, 0.0])


def test_the_closed_track_deck_stays_bounded_for_hundreds_of_episodes():
    """The point of closing the loop: no arena clamp, no inward steering."""
    from pad_motion import STRAIGHT_ESCAPE_BURST_TRACK_SCENARIO as TRACK

    cfg, traj = _trajectory()
    time, worst = 0.0, 0.0
    for episode in range(120):
        traj.reset(seed=70000 + episode, sim_time=time, scenario=TRACK)
        # An adversarial dash: always the same direction, which is what made
        # an off-path dash walk the deck to the arena edge in 30 episodes.
        traj.trigger_escape_burst(time + 8.0,
                                  heading=float(traj.heading0 + math.pi))
        times = time + np.arange(0, 300) * 0.1
        points = np.array([traj.pose(t)[0][:2] for t in times])
        worst = max(worst, float(np.linalg.norm(points, axis=1).max()))
        time = times[-1] + 0.1
    # Well inside the half-arena, where the inward steering would start.
    assert worst < 0.5 * cfg.arena_radius_m
    assert worst < 40.0


def test_the_track_dash_is_a_straight_speed_doubling_on_a_straight():
    """The FOV-loss event itself: same speed profile, same straight line."""
    from pad_motion import (STRAIGHT_ESCAPE_BURST_TRACK_SCENARIO as TRACK,
                            BENCHMARK_PEAK_SPEED_M_S,
                            STRAIGHT_ESCAPE_CRUISE_SPEED_M_S)

    cfg, traj = _trajectory()
    traj.reset(seed=70000, sim_time=0.0, scenario=TRACK)  # phase 0 is a straight
    burst = traj.trigger_escape_burst(8.0)
    assert burst is not None

    times = np.arange(0, 250) * 0.1
    points = np.array([traj.pose(t)[0][:2] for t in times])
    speeds = np.array([float(np.linalg.norm(traj.pose(t)[1][:2])) for t in times])
    cruise = STRAIGHT_ESCAPE_CRUISE_SPEED_M_S * cfg.benchmark_speed_scale
    peak = BENCHMARK_PEAK_SPEED_M_S * cfg.benchmark_speed_scale
    assert speeds.min() == pytest.approx(cruise)
    assert speeds.max() == pytest.approx(peak)

    deltas = np.diff(points, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    headings = np.degrees(np.arctan2(deltas[lengths > 1e-9, 1],
                                     deltas[lengths > 1e-9, 0]))
    assert headings.max() - headings.min() < 1e-6, "the dash must not turn"
    assert float(np.linalg.norm(points[-1] - points[0])) == pytest.approx(
        lengths.sum(), rel=1e-9)


def test_the_track_carries_its_phase_rather_than_redrawing_its_heading():
    from pad_motion import (STRAIGHT_ESCAPE_BURST_TRACK_SCENARIO as TRACK,
                            STRAIGHT_ESCAPE_CRUISE_SPEED_M_S)

    cfg, traj = _trajectory()
    traj.reset(seed=70000, sim_time=0.0, scenario=TRACK)
    orientation = traj.heading0
    assert traj.track_phase_m == 0.0

    traj.reset(seed=99999, sim_time=30.0, scenario=TRACK)
    # A different seed must not rotate the loop under the vehicle parked on it.
    assert traj.heading0 == pytest.approx(orientation)
    # 30 s of cruising is exactly how far along the loop it has driven.
    cruise = STRAIGHT_ESCAPE_CRUISE_SPEED_M_S * cfg.benchmark_speed_scale
    assert traj.track_phase_m == pytest.approx(cruise * 30.0, rel=1e-6)

    # A lap later the deck is back where it started, to integration error.
    perimeter = 2.0 * (cfg.benchmark_track_straight_m
                       + math.pi * cfg.benchmark_track_radius_m)
    start = traj.pose(30.0)[0][:2].copy()
    lap_seconds = perimeter / cruise
    assert float(np.linalg.norm(traj.pose(30.0 + lap_seconds)[0][:2] - start)) < 0.1


def test_the_gateway_accepts_the_track_scenario():
    """The vocabulary the reset protocol validates against."""
    protocol = (ROOT / "ros2_ws/src/ontology_rgat_px4/ontology_rgat_px4"
                / "protocol.py").read_text(encoding="utf-8")
    assert '"straight_escape_burst_track"' in protocol


def test_every_shipped_profile_still_parses_with_the_track_geometry():
    """The turn-radius guard must not reject a deck that never drives a track."""
    from pad_motion import PadMotionConfig

    for profile in sorted((ROOT / "config").glob("*.yaml")):
        cfg = PadMotionConfig.from_mapping(load_config(profile))
        half_diagonal = 0.5 * math.hypot(*cfg.deck_size_m[:2])
        assert cfg.benchmark_track_radius_m > half_diagonal, profile.name


def test_a_turn_that_pivots_inside_the_deck_is_a_configuration_error():
    from pad_motion import PadMotionConfig

    system = load_config(ROOT / "config/shin2026-minimal-system.yaml")
    half_diagonal = 0.5 * math.hypot(*(float(v) for v in
                                       system["pad"]["deck_size_m"][:2]))
    tight = {**system, "pad": {**system["pad"],
                               "benchmark_track_radius_m": half_diagonal * 0.9}}
    with pytest.raises(ValueError, match="half-diagonal"):
        PadMotionConfig.from_mapping(tight)
