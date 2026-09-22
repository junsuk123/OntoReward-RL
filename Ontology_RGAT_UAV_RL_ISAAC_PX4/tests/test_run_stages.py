"""Data collection and training are two stages, and they stay in that order.

Until 2026-09-22 one run did both at once: the fixed-reward arms' PPO was
submitted first and the FOV-risk collector was handed whichever physical pair
the training assignment had left over. That made the collection single-pair on
a four-pair stage, and it meant the proposed arm's frozen readout was fitted
while its comparison arm was already hundreds of episodes into training.

``--stage`` separates them. The collection stage flies every dataset on every
pair and starts no PPO; the training stage fits the readouts, trains every arm
against the finished reward design, and refuses to fly for collection at all.
"""
from __future__ import annotations

from pathlib import Path
import re
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import run_three_pipeline as pipeline_runner
from run_three_pipeline import (COLLECTION_BASE_SCENARIO, CollectionUnavailable,
                                STAGES, STAGE_ALL, STAGE_COLLECT, STAGE_TRAIN,
                                _require_flight)
from ontology_rgat.benchmarks.experiment import load_experiment


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (ROOT / "python/run_three_pipeline.py").read_text(encoding="utf-8")


def _source_position(pattern: str) -> int:
    match = re.search(pattern, RUNNER)
    assert match is not None, f"the runner no longer contains {pattern!r}"
    return match.start()


# --------------------------------------------------------------- stage order

def test_the_default_stage_collects_and_then_trains():
    assert STAGES == (STAGE_ALL, STAGE_COLLECT, STAGE_TRAIN)
    assert 'choices=STAGES, default=STAGE_ALL' in RUNNER


def test_no_ppo_worker_is_submitted_before_every_dataset_is_collected():
    """The pool that flies PPO must be created after the last collector."""
    collectors = [
        _source_position(r"_collect_fov_risk_data\(\n"),
        _source_position(r"_collect_semantic_data\(\n"),
        _source_position(r"_collect_adaptive_data\(\n"),
    ]
    executor = _source_position(r'thread_name_prefix="landing-pair"')
    assert executor > max(collectors)


def test_every_frozen_readout_is_fitted_before_the_first_ppo_episode():
    executor = _source_position(r'thread_name_prefix="landing-pair"')
    for artifact in (r"prepare_fov_risk_artifact\(", r"prepare_semantic_rgat_artifact\(",
                     r"prepare_adaptive_reward_artifact\("):
        assert _source_position(artifact) < executor


def test_the_collect_stage_returns_before_any_readout_is_fitted():
    stop = _source_position(r"if not training_allowed:")
    assert stop < _source_position(r"prepare_fov_risk_artifact\(")


# ------------------------------------------------- the training stage refuses

def test_the_flight_guard_names_the_command_that_fills_the_gap():
    _require_flight(True, "anything")          # collecting: no objection
    with pytest.raises(CollectionUnavailable) as caught:
        _require_flight(False, "the FOV-risk rollouts", shortfall="3/40 episodes")
    message = str(caught.value)
    assert "the FOV-risk rollouts" in message and "3/40 episodes" in message
    assert "--stage collect" in message and "--stage all" in message


def _monitor():
    return SimpleNamespace(stage=lambda *_a, **_k: None,
                           fov_dataset=lambda *_a, **_k: None)


def test_the_training_stage_will_not_fly_the_fov_risk_rollouts(tmp_path):
    with pytest.raises(CollectionUnavailable, match="--stage collect"):
        pipeline_runner._collect_fov_risk_data(
            cfg=SimpleNamespace(sim=SimpleNamespace(dt=.1, max_steps=8)),
            camera=object(), model=object(), config={
                "fov_risk": {"prediction_horizon_seconds": 1.0},
                "fov_risk_design": {"episodes_quick": 4, "max_episodes_quick": 4},
                "seeds": {"fov_risk_dataset_start": 70000},
            }, config_hash="test", checkpoint_path=None,
            results_dir=tmp_path, mode="quick", monitor=_monitor(),
            flight_allowed=False)


def test_the_training_stage_will_not_fly_the_semantic_rollouts(tmp_path):
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"frozen source policy")
    with pytest.raises(CollectionUnavailable, match="--stage collect"):
        pipeline_runner._collect_semantic_data(
            cfg=SimpleNamespace(sim=SimpleNamespace(max_steps=8)),
            camera=object(), model=object(), config={
                "rgat_design": {"episodes_quick": 4, "max_episodes_quick": 4},
                "ppo": {"gamma": .99},
                "seeds": {"rgat_dataset_start": 70000},
            }, config_hash="test", checkpoint_path=checkpoint,
            results_dir=tmp_path, mode="quick", monitor=_monitor(),
            flight_allowed=False)


def test_the_training_stage_will_not_fly_the_adaptive_rollouts(tmp_path):
    with pytest.raises(CollectionUnavailable, match="--stage collect"):
        pipeline_runner._collect_adaptive_data(
            cfg=SimpleNamespace(sim=SimpleNamespace(max_steps=8)),
            camera=object(), model=object(), source_pipeline="no_se_fixed",
            config={
                "seed": 7,
                "adaptive_reward_design": {
                    "episodes_quick": 4, "max_episodes_quick": 4},
            }, config_hash="test", results_dir=tmp_path, mode="quick",
            monitor=_monitor(), flight_allowed=False)


def test_the_keypoint_survey_is_a_collection_stage_flight():
    """A training stage must not re-survey; the encoder labels every dataset."""
    guard = _source_position(
        r"the keypoint encoder's live Isaac calibration")
    assert guard < _source_position(r"calibrate_keypoint_encoder_in_flight\(")
    assert "needs_empirical_calibration(" in RUNNER


# --------------------------------------------- collection uses the whole stage

def test_fov_risk_rollouts_use_every_live_pair_concurrently(
        monkeypatch, tmp_path):
    """Three free pairs collect three episodes at once, not one after another.

    The barrier is the assertion: an episode that never sees the other two
    arrive times out, which is what a sequential collector would do.
    """
    barrier = threading.Barrier(3)
    active_pairs = set()

    class Environment:
        def __init__(self, cfg, _camera, horizon_steps):
            self.cfg = cfg
            assert horizon_steps == 8

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    def collect(environment, _model, _method, seed, **_kwargs):
        active_pairs.add(environment.cfg.pair)
        barrier.wait(timeout=5.0)
        row = {"fov_graph_X": np.zeros((2, 2)),
               "fov_graph_geometric_in_fov": 1,
               "next_fov_graph_X": np.zeros((2, 2)),
               "geometric_in_fov": 1,
               "seed": seed}
        return [row], {"geometric_fov_loss_episode_rate": 1}

    monkeypatch.setattr(pipeline_runner, "LiveShinEnvironment", Environment)
    monkeypatch.setattr(pipeline_runner, "collect_episode_resilient", collect)
    monkeypatch.setattr(
        pipeline_runner, "build_fov_risk_dataset",
        lambda episodes, **_kwargs: {"episodes": len(episodes)})
    monkeypatch.setattr(
        pipeline_runner, "_fov_dataset_progress",
        lambda dataset: {"supervised_samples": int(dataset["episodes"]),
                         "masked_samples": 0, "target_mean": .5,
                         "covered": True})
    monkeypatch.setattr(pipeline_runner, "_fov_target_coverage",
                        lambda _dataset: True)
    monkeypatch.setattr(
        pipeline_runner, "save_fov_risk_dataset",
        lambda dataset, *_args, **_kwargs: {"episodes": dataset["episodes"]})

    contexts = [{
        "cfg": SimpleNamespace(pair=index,
                               sim=SimpleNamespace(dt=.1, max_steps=8)),
        "camera": object(), "monitor": _monitor(),
    } for index in range(3)]
    _dataset, manifest, _path, flown_steps, reuse = (
        pipeline_runner._collect_fov_risk_data(
            cfg=contexts[0]["cfg"], camera=contexts[0]["camera"],
            model=SimpleNamespace(modules=lambda: ()), config={
                "fov_risk": {"prediction_horizon_seconds": 1.0},
                "fov_risk_design": {"episodes_quick": 3, "max_episodes_quick": 3},
                "seeds": {"fov_risk_dataset_start": 70000},
            }, config_hash="test", checkpoint_path=None,
            results_dir=tmp_path, mode="quick", monitor=contexts[0]["monitor"],
            parallel_contexts=contexts))

    assert active_pairs == {0, 1, 2}
    assert manifest["episodes"] == 3
    assert manifest["distinct_seeds"] == 3
    assert flown_steps == 3
    assert reuse == {}


def test_fov_risk_episode_identity_does_not_depend_on_which_pair_landed_first(
        monkeypatch, tmp_path):
    """Seeds, decks and behaviour variants are allocated before the batch flies.

    Otherwise a four-pair collection and a one-pair collection of the same
    seeds would disagree about which deck each episode flew, and the datastore
    would hold two different episodes under one initial condition.
    """
    seen = []
    order = threading.Lock()

    class Environment:
        def __init__(self, cfg, _camera, horizon_steps):
            self.cfg = cfg

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    def collect(environment, _model, _method, seed, *, scenario, **_kwargs):
        # Finish in reverse pair order so completion order is never seed order.
        threading.Event().wait(.02 * (2 - environment.cfg.pair))
        with order:
            seen.append((seed, scenario))
        row = {"fov_graph_X": np.zeros((2, 2)),
               "fov_graph_geometric_in_fov": 1,
               "next_fov_graph_X": np.zeros((2, 2)),
               "geometric_in_fov": 1}
        return [row], {"geometric_fov_loss_episode_rate": 0}

    monkeypatch.setattr(pipeline_runner, "LiveShinEnvironment", Environment)
    monkeypatch.setattr(pipeline_runner, "collect_episode_resilient", collect)
    monkeypatch.setattr(
        pipeline_runner, "build_fov_risk_dataset",
        lambda episodes, **_kwargs: {"episodes": len(episodes),
                                     "seeds": [e["seed"] for e in episodes]})
    monkeypatch.setattr(
        pipeline_runner, "_fov_dataset_progress",
        lambda dataset: {"supervised_samples": 1, "masked_samples": 0,
                         "target_mean": .5, "covered": True})
    monkeypatch.setattr(pipeline_runner, "_fov_target_coverage",
                        lambda _dataset: True)
    monkeypatch.setattr(
        pipeline_runner, "save_fov_risk_dataset",
        lambda dataset, *_args, **_kwargs: {"episodes": dataset["episodes"],
                                            "seeds": dataset["seeds"]})

    contexts = [{
        "cfg": SimpleNamespace(pair=index,
                               sim=SimpleNamespace(dt=.1, max_steps=8)),
        "camera": object(), "monitor": _monitor(),
    } for index in range(3)]
    _dataset, manifest, _path, _steps, _reuse = (
        pipeline_runner._collect_fov_risk_data(
            cfg=contexts[0]["cfg"], camera=contexts[0]["camera"],
            model=SimpleNamespace(modules=lambda: ()), config={
                "fov_risk": {"prediction_horizon_seconds": 1.0},
                "fov_risk_design": {"episodes_quick": 3, "max_episodes_quick": 3},
                "seeds": {"fov_risk_dataset_start": 70000},
            }, config_hash="test", checkpoint_path=None,
            results_dir=tmp_path, mode="quick", monitor=contexts[0]["monitor"],
            scenarios=("deck_a", "deck_b"), parallel_contexts=contexts))

    # Episode 1 flew deck_a on seed 70000, episode 2 deck_b on 70001, ...
    assert dict(seen) == {70000: "deck_a", 70001: "deck_b", 70002: "deck_a"}
    # ... and they are stored in seed order however the batch completed.
    assert manifest["seeds"] == [70000, 70001, 70002]


# ------------------------------------------------- the base collection deck

def test_the_base_collection_deck_is_the_closed_escape_burst_track():
    """The deck that produces the event under test in every episode.

    The pad cruises a closed oval, the vehicle settles into following it, and
    then it doubles speed on a straight and leaves the camera frame. A readout
    fitted on a deck that rarely leaves the frame has almost no positive target
    to learn from, which is why this is the default rather than the random
    walk; the loop is closed so the deck cannot walk out of the arena.
    """
    import sys
    sys.path.insert(0, str(ROOT / "isaac_sim"))
    from pad_motion import (BENCHMARK_SCENARIOS,
                            STRAIGHT_ESCAPE_BURST_TRACK_SCENARIO)

    assert COLLECTION_BASE_SCENARIO == "straight_escape_burst_track"
    assert COLLECTION_BASE_SCENARIO == STRAIGHT_ESCAPE_BURST_TRACK_SCENARIO
    assert COLLECTION_BASE_SCENARIO in BENCHMARK_SCENARIOS


def test_the_default_experiment_collects_and_trains_on_that_deck_only():
    config = load_experiment(
        ROOT / "config/experiments/three_arm_burst_comparison.yaml")
    assert list(config["training"]["scenarios"]) == [COLLECTION_BASE_SCENARIO]
    assert list(config["behavior_cloning"]["scenarios"]) == [
        COLLECTION_BASE_SCENARIO]
    flown = {name for name, count in config["evaluation"].items() if int(count)}
    assert flown == {COLLECTION_BASE_SCENARIO}


def test_the_fov_collector_defaults_to_the_base_deck():
    import inspect

    fov = inspect.signature(pipeline_runner._collect_fov_risk_data)
    assert fov.parameters["scenarios"].default == (COLLECTION_BASE_SCENARIO,)


def test_the_adaptive_collection_defaults_to_the_base_deck(monkeypatch, tmp_path):
    """Its fallback used to be a four-deck cycle led by the random walk."""
    flown = []

    class Environment:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    def collect(_environment, _model, _method, seed, *, scenario, **_kwargs):
        flown.append(scenario)
        return [object()], {"paper_success": len(flown) <= 2}

    monkeypatch.setattr(pipeline_runner, "LiveShinEnvironment", Environment)
    monkeypatch.setattr(pipeline_runner, "collect_episode_resilient", collect)
    monkeypatch.setattr(
        pipeline_runner, "adaptive_episode_records",
        lambda _rows, metric, *, episode_id, **_kwargs: [{
            "episode_id": episode_id, "success": int(metric["paper_success"]),
            "failure_type": "none" if metric["paper_success"] else "collision",
            "touchdown_error": .1, "touchdown_vertical_speed": .1}])
    monkeypatch.setattr(
        pipeline_runner, "build_adaptive_dataset",
        lambda records, **_kwargs: {
            key: np.asarray([row[key] for row in records])
            for key in ("episode_id", "success", "failure_type",
                        "touchdown_error", "touchdown_vertical_speed")}
        | {"split": np.asarray(["validation"] * len(records))})
    monkeypatch.setattr(
        pipeline_runner, "save_adaptive_dataset",
        lambda dataset, *_args, **kwargs: {
            "episodes": len(np.unique(dataset["episode_id"])),
            "source_behavior_policy": kwargs["source_behavior_policy"]})

    _dataset, manifest, _path, _steps = pipeline_runner._collect_adaptive_data(
        cfg=SimpleNamespace(sim=SimpleNamespace(max_steps=8)), camera=object(),
        model=object(), source_pipeline="no_se_fixed", config={
            "seed": 7,
            "adaptive_reward_design": {
                "episodes_quick": 4, "max_episodes_quick": 4,
                "minimum_successful_episodes": 2,
                "minimum_failed_episodes": 2,
                "minimum_risky_failures": 1},
        }, config_hash="test", results_dir=tmp_path, mode="quick",
        monitor=_monitor())

    assert set(flown) == {COLLECTION_BASE_SCENARIO}
    assert manifest["source_behavior_policy"]["scenario_cycle"] == [
        COLLECTION_BASE_SCENARIO]


def test_no_deck_that_stays_in_frame_is_hardcoded_into_a_collector():
    """The semantic collection had ``training_random_walk`` written into it."""
    assert 'scenario="training_random_walk"' not in RUNNER


def test_the_semantic_collection_flies_the_runs_deck_rotation(
        monkeypatch, tmp_path):
    """It used to be hardcoded to a deck the trained policy never flies."""
    flown = []

    class Environment:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    def collect(_environment, _model, _method, seed, *, scenario, **_kwargs):
        flown.append((seed, scenario))
        return [{"semantic_graph_X": np.zeros((2, 2))}], {
            "paper_success": len(flown) % 2, "status": "landed", "steps": 1,
            "geometric_fov_loss_events": 1,
            "geometric_fov_reacquisition_events": 1,
            "geometric_fov_reacquisition_rate": 1.0,
            "mean_geometric_fov_reacquisition_time_s": .5,
            "climb_during_geometric_fov_loss_fraction": .5,
            "descent_during_low_keypoint_visibility_fraction": .0,
            "successful_recovery_landing": 1,
        }

    monkeypatch.setattr(pipeline_runner, "LiveShinEnvironment", Environment)
    monkeypatch.setattr(pipeline_runner, "collect_episode_resilient", collect)
    monkeypatch.setattr(
        pipeline_runner, "semantic_episode_dataset",
        lambda samples, **kwargs: {"y": [0.0], "episode": kwargs["episode_id"]})
    monkeypatch.setattr(
        pipeline_runner, "merge_semantic_datasets",
        lambda dataset, episode: [*(dataset or []), episode])
    monkeypatch.setattr(
        pipeline_runner, "save_semantic_dataset",
        lambda dataset, *_args, **kwargs: {
            "episodes": len(dataset),
            "successful_episodes": sum(
                1 for _ in dataset) - 1,
            **{key: kwargs[key] for key in ("source_behavior_policy",)}})

    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"frozen source policy")
    _dataset, manifest, _path, _steps = pipeline_runner._collect_semantic_data(
        cfg=SimpleNamespace(sim=SimpleNamespace(max_steps=8)), camera=object(),
        model=object(), config={
            "rgat_design": {"episodes_quick": 2, "max_episodes_quick": 2,
                            "minimum_successful_recovery_episodes_quick": 1},
            "ppo": {"gamma": .99},
            "seeds": {"rgat_dataset_start": 70000},
        }, config_hash="test", checkpoint_path=checkpoint,
        results_dir=tmp_path, mode="quick", monitor=_monitor())

    # No deck given by the caller or the profile: the base scenario, rotated.
    assert [scenario for _seed, scenario in flown] == [
        COLLECTION_BASE_SCENARIO] * len(flown)
    assert manifest["source_behavior_policy"]["scenario_cycle"] == [
        COLLECTION_BASE_SCENARIO]


def test_the_fov_accumulation_is_keyed_on_the_deck_it_was_flown_on():
    """A six-deck accumulation must not silently extend a burst-only one."""
    from config_loader import load_config
    from ontology_rgat.rgat import fov_risk_data_fingerprint

    system = load_config(ROOT / "config/shin2026-minimal-system.yaml")
    common = {"prediction_steps": 10, "control_hz": 10.0,
              "keypoint_implementation": "encoder-v4"}
    burst = fov_risk_data_fingerprint(
        system, scenarios=(COLLECTION_BASE_SCENARIO,), **common)
    assert burst != fov_risk_data_fingerprint(
        system, scenarios=("training_random_walk",), **common)
    assert burst != fov_risk_data_fingerprint(
        system, scenarios=(COLLECTION_BASE_SCENARIO, "circle"), **common)
