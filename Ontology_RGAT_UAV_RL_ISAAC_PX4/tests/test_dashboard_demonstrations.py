"""Demonstration flights and vehicle trajectories are live on the dashboard.

Until 2026-09-21 the behaviour-cloning warm start showed on the dashboard only
as a pair reading "training-only teacher demonstration · ep N": how many
landings were still needed, whether the teacher was climbing out of a lost
pad, and where the two vehicles actually went were all in files nobody reads
during a run. These tests pin the monitor side of the panels that changed
that: world-frame trajectories on every step point, the teacher's per-step
diagnostics with recovery counters, and the flight-by-flight progress.
"""
from __future__ import annotations

import threading

import numpy as np

from ontology_rgat.viz.dashboard import PAGE
from ontology_rgat.viz.live import BenchmarkMonitor, LiveStore
from run_three_pipeline import _LockedMonitor

METHODS = ["shin_se_fixed", "shin_se_onto_rgat_recovery"]


def _monitor(pair_count=2, rviz=None):
    store = LiveStore()
    monitor = BenchmarkMonitor(store, rviz=rviz)
    monitor.configure(
        methods=METHODS, mode="quick", config_hash="0" * 16,
        training_total=8, evaluation_total=4,
        pair_layout=[{
            "index": index, "method": METHODS[index % len(METHODS)],
            "route_phase_fraction": 0.0, "px4_namespace": "/fmu",
            "topic_root": f"/landing_pair_{index}",
            "camera_topic": f"/landing_pair_{index}/uav/perception/landing_camera/annotated",
            "rviz_namespace": f"/landing_rl/pair_{index}",
            "gateway_port": 14650 + 2 * index, "learner_port": 14651 + 2 * index,
        } for index in range(pair_count)])
    return monitor, store


def _state(uav, pad):
    return {"position": [0.1, -0.2, -2.0],
            "world": {"position": list(uav), "velocity": [0.3, 0.0, 0.0]},
            "pad": {"position": list(pad), "velocity": [0.12, 0.0, 0.0]},
            "battery": {"enabled": False}}


def _step(monitor, index, uav, pad, method="no_se_fixed", pair_index=0):
    monitor.step(index=index, dt=0.1, method=method, reward=0.0, reward_parts={},
                 estimate=None, truth=np.zeros(6), geometric_in_fov=True,
                 estimation_loss=None, state=_state(uav, pad),
                 pair_index=pair_index)


def test_every_step_point_carries_both_world_positions():
    monitor, store = _monitor()
    monitor.reset_episode(method="no_se_fixed", seed=90000, curriculum=0.2,
                          phase="training-only teacher demonstration",
                          scenario="training_random_walk_escape_burst",
                          pair_index=0)
    _step(monitor, 1, (1.0, 2.0, 5.0), (0.5, 2.5, 0.42))
    _step(monitor, 2, (1.2, 2.1, 4.8), (0.6, 2.5, 0.42))
    snapshot = store.snapshot()
    rows = snapshot["series"]["benchmark_step_pair_0"]
    assert [(r["uav_x"], r["uav_y"], r["uav_z"]) for r in rows] == [
        (1.0, 2.0, 5.0), (1.2, 2.1, 4.8)]
    assert [(r["pad_x"], r["pad_y"], r["pad_z"]) for r in rows] == [
        (0.5, 2.5, 0.42), (0.6, 2.5, 0.42)]
    pair = snapshot["scalars"]["parallel_pair_status"][0]
    assert pair["uav_xyz"] == [1.2, 2.1, 4.8] and pair["pad_xyz"] == [0.6, 2.5, 0.42]
    # The demonstration phase is counted as its own kind of episode.
    assert pair["episode_kind"] == "시연" and pair["episode"] == 1


def test_a_step_without_world_positions_still_publishes():
    monitor, store = _monitor()
    monitor.step(index=1, dt=0.1, method="shin_se_fixed", reward=0.0,
                 reward_parts={}, estimate=None, truth=np.zeros(6),
                 geometric_in_fov=True, estimation_loss=None,
                 state={"position": [0.0, 0.0, -1.0]}, pair_index=0)
    row = store.snapshot()["series"]["benchmark_step_pair_0"][-1]
    assert "uav_x" not in row and "pad_x" not in row
    assert store.snapshot()["scalars"]["parallel_pair_status"][0]["uav_xyz"] is None


def test_teacher_steps_publish_the_lose_climb_reacquire_sequence():
    monitor, store = _monitor()

    def teacher(step, **info):
        monitor.teacher_step(method="no_se_fixed", step=step, seed=90000,
                             pair_index=0, **info)
        return store.snapshot()["scalars"]["parallel_pair_status"][0]["teacher"]

    following = teacher(5, altitude_m=3.0, lateral_error_m=1.2, target_vz_m_s=0.0,
                        visual_lost=False, geometric_in_fov=True)
    assert following["mode"] == "following" and following["losses"] == 0
    lost = teacher(40, altitude_m=1.1, lateral_error_m=2.4, target_vz_m_s=0.22,
                   visual_lost=True, geometric_in_fov=False)
    assert lost["mode"] == "lost_climbing" and lost["losses"] == 1
    assert lost["climb_steps"] == 1 and lost["geometric_in_fov"] is False
    higher = teacher(41, altitude_m=1.3, lateral_error_m=2.3, target_vz_m_s=0.22,
                     visual_lost=True, geometric_in_fov=False)
    assert higher["losses"] == 1, "one continuous loss is one loss"
    assert higher["max_altitude_after_loss_m"] == 1.3
    back = teacher(60, altitude_m=2.0, lateral_error_m=1.0, target_vz_m_s=0.0,
                   visual_lost=False, geometric_in_fov=True)
    assert back["mode"] == "reacquired" and back["max_altitude_after_loss_m"] == 2.0
    descending = teacher(90, altitude_m=0.9, lateral_error_m=0.1, target_vz_m_s=-0.25,
                         visual_lost=False, geometric_in_fov=True)
    assert descending["mode"] == "descending"
    # A new flight (step counter restarts) resets the recovery counters.
    fresh = teacher(1, altitude_m=4.5, lateral_error_m=2.0, target_vz_m_s=0.0,
                    visual_lost=False, geometric_in_fov=True)
    assert fresh["losses"] == 0 and fresh["max_altitude_after_loss_m"] is None


def test_teacher_step_tolerates_missing_and_non_numeric_diagnostics():
    monitor, store = _monitor()
    monitor.teacher_step(method="no_se_fixed", step=3, seed=None, pair_index=0,
                         altitude_m=float("nan"), lateral_error_m=None,
                         target_vxy_m_s=[0.1, 0.2])
    live = store.snapshot()["scalars"]["parallel_pair_status"][0]["teacher"]
    assert live["altitude_m"] is None and live["lateral_error_m"] is None
    assert live["visual_lost"] is None and live["mode"] == "aligned"


def test_demonstration_progress_keeps_the_recent_flights():
    monitor, store = _monitor()
    attempts = [{"seed": 90000 + k, "steps": 80 + k, "status": "success",
                 "accepted_for_cloning": 1.0 if k % 2 == 0 else 0.0,
                 "touchdown_lateral_error": 0.1 * k,
                 "geometric_fov_loss_fraction": 0.05 * k,
                 "scenario": "training_random_walk_escape_burst"}
                for k in range(15)]
    attempts.append({"seed": 90020, "status": "infrastructure_failure",
                     "accepted_for_cloning": 0.0, "steps": 0})
    monitor.demonstration_progress(
        method="no_se_fixed", required=4, accepted=3, flights=15, max_flights=40,
        skips=1, teacher="privileged_relative_state_velocity_pd_v4",
        scenario="training_random_walk_escape_burst", fingerprint="1c7479639cbe",
        attempts=attempts, pair_index=0)
    demo = store.snapshot()["scalars"]["teacher_demonstrations"]
    assert demo["accepted"] == 3 and demo["required"] == 4 and not demo["complete"]
    assert demo["flights"] == 15 and demo["skips"] == 1 and demo["pair_index"] == 0
    assert len(demo["recent"]) == 12
    assert demo["recent"][-1]["status"] == "infrastructure_failure"
    assert demo["recent"][-2]["seed"] == 90014 and demo["recent"][-2]["accepted"] is True
    assert demo["recent"][-3]["accepted"] is False
    monitor.demonstration_progress(
        method="no_se_fixed", required=4, accepted=4, flights=16, max_flights=40,
        attempts=attempts, pair_index=0)
    assert store.snapshot()["scalars"]["teacher_demonstrations"]["complete"] is True


def test_the_locked_pair_monitor_routes_teacher_telemetry_to_its_pair():
    monitor, store = _monitor(pair_count=4)
    locked = _LockedMonitor(monitor, threading.Lock(), method="no_se_fixed",
                            pair_index=2, pair_count=4)
    locked.teacher_step(method="no_se_fixed", step=7, seed=90003,
                        altitude_m=2.0, lateral_error_m=0.3, target_vz_m_s=-0.5,
                        visual_lost=False, geometric_in_fov=True)
    locked.demonstration_progress(method="no_se_fixed", required=4, accepted=1,
                                  flights=2, max_flights=40)
    pairs = store.snapshot()["scalars"]["parallel_pair_status"]
    assert pairs[2]["teacher"]["step"] == 7 and "teacher" not in pairs[0]
    assert store.snapshot()["scalars"]["teacher_demonstrations"]["pair_index"] == 2


def test_every_collecting_pair_shows_its_own_demonstration_flight():
    """Four pairs fly the warm start, so four live flights are published.

    The progress itself (accepted/required, the attempt ledger) belongs to the
    stage rather than to a pair and stays a single reading; which pairs are
    flying it, and what each of them is doing right now, are per pair.
    """
    monitor, store = _monitor(pair_count=4)
    locked = [_LockedMonitor(monitor, threading.Lock(),
                             method=METHODS[index % len(METHODS)],
                             pair_index=index, pair_count=4)
              for index in range(4)]
    for index, worker in enumerate(locked):
        worker.teacher_step(method="no_se_fixed", step=10 + index,
                            seed=90000 + index, altitude_m=3.0 - index,
                            lateral_error_m=0.4, target_vz_m_s=-0.3,
                            visual_lost=False, geometric_in_fov=True)
    attempts = [{"seed": 90000 + index, "steps": 120, "status": "success",
                 "accepted_for_cloning": float(index == 0),
                 "physical_pair_index": index,
                 "touchdown_lateral_error": 0.08,
                 "geometric_fov_loss_fraction": 0.12}
                for index in range(4)]
    locked[0].demonstration_progress(
        method="no_se_fixed", required=4, accepted=1, flights=4, max_flights=40,
        teacher="image_based_visual_servo_v1",
        scenario="training_random_walk_escape_burst", fingerprint="6fdd99e3926f",
        attempts=attempts, collection_pairs=[0, 1, 2, 3])

    pairs = store.snapshot()["scalars"]["parallel_pair_status"]
    assert [pairs[index]["teacher"]["seed"] for index in range(4)] == [
        90000, 90001, 90002, 90003]
    assert [pairs[index]["teacher"]["step"] for index in range(4)] == [
        10, 11, 12, 13]
    demo = store.snapshot()["scalars"]["teacher_demonstrations"]
    assert demo["collection_pairs"] == [0, 1, 2, 3]
    assert demo["teacher"] == "image_based_visual_servo_v1"
    assert [row["pair_index"] for row in demo["recent"]] == [0.0, 1.0, 2.0, 3.0]


def test_a_single_pair_stage_still_names_the_pair_it_flew():
    monitor, store = _monitor(pair_count=1)
    monitor.demonstration_progress(
        method="no_se_fixed", required=4, accepted=0, flights=0, max_flights=40,
        attempts=(), pair_index=0)
    demo = store.snapshot()["scalars"]["teacher_demonstrations"]
    assert demo["collection_pairs"] == [0]


def test_the_rviz_pair_view_is_told_which_stage_it_is_watching():
    """RViz renders one identical panel per pair; the stage tells them apart."""
    calls = []

    class FakeRviz:
        def clear_trails(self, **kwargs):
            calls.append(("clear", kwargs))

        def publish_benchmark_step(self, **kwargs):
            calls.append(("step", kwargs))

    monitor, _store = _monitor(pair_count=4, rviz=FakeRviz())
    monitor.reset_episode(method="no_se_fixed", seed=90003, curriculum=0.2,
                          phase="training-only teacher demonstration",
                          scenario="training_random_walk_escape_burst",
                          pair_index=3)
    _step(monitor, 1, (1.0, 2.0, 5.0), (0.5, 2.5, 0.42), pair_index=3)

    assert ("clear", {"method": "no_se_fixed", "pair_index": 3}) in calls
    published = [kwargs for kind, kwargs in calls if kind == "step"][-1]
    assert published["pair_index"] == 3
    assert published["phase"] == "training-only teacher demonstration"


def test_the_page_renders_the_demonstration_and_trajectory_panels():
    for marker in ('id="teacher-demos"', 'id="traj-grid"', "function teacherPanel",
                   "function teacherTile", "collection_pairs", "class=\"tgrid\"",
                   "function drawTrajectories", "function drawTrajectory",
                   "benchmark_step_pair_", "uav_x", "pad_x", "teacher_demonstrations",
                   "lost_climbing", "kind:'teacher'", "kind:'trajectories'"):
        assert marker in PAGE, marker
