from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest
import torch

import run_three_pipeline as pipeline_runner

from ontology_rgat.bridge import BridgeError, EntryResetError, GatewayTimeout, PX4Failsafe
from ontology_rgat.benchmarks.experiment import (configuration_hash,
                                                 controlled_training_seeds,
                                                 episodes_per_method,
                                                 load_experiment,
                                                 paired_seed_plan)
from ontology_rgat.evaluation.three_pipeline import (
    PHYSICAL_METRICS, learning_efficiency, paired_confidence_intervals,
    physical_summary)
from ontology_rgat.perception import (SEMANTIC_FEATURE_NAMES,
                                      SEMANTIC_GRAPH_INPUT_DIM,
                                      SEMANTIC_NODE_NAMES, SemanticObservation,
                                      semantic_graph,
                                      semantic_observation_from_payload)
from ontology_rgat.pipelines import (PIPELINES, PipelineSpec, get_pipeline,
                                     validate_pipeline_configuration)
from ontology_rgat.ppo.recurrent import (PipelineActorCritic,
                                         ShinRecurrentActorCritic,
                                         recurrent_ppo_loss)
from ontology_rgat.ppo import recurrent_train
from ontology_rgat.ppo.recurrent_train import collect_episode_resilient
from ontology_rgat.ppo.recurrent_train import (aggregate_deployment_validation,
                                               deployment_validation_key,
                                               training_health_issue,
                                               deployment_checkpoint_score,
                                               update_episode,
                                               visual_recovery_metrics)
from ontology_rgat.reward_modes import OntologyRewardContext, OntoRewardPBRS, TerminalFlags
from ontology_rgat.reward_modes import ShinRewardConfig, active_perception_reward
from ontology_rgat.rgat import (FrozenSemanticRGATPotential,
                                merge_semantic_datasets,
                                prepare_semantic_rgat_artifact,
                                save_semantic_dataset,
                                semantic_episode_dataset,
                                semantic_monotonic_counterfactuals,
                                validate_semantic_dataset)
from ontology_rgat.rgat.semantic_dataset import semantic_rgat_config
from ontology_rgat.rgat.train import train_potential
from run_three_pipeline import (_behavior_transform,
                                _adaptive_artifact_quality_issues,
                                _adaptive_reward_settings,
                                _balanced_training_pair_assignment,
                                _checkpoint_validation_plan,
                                _crossover_evaluation_tasks,
                                _hold_after_complete,
                                _privileged_velocity_teacher_action,
                                _reward_design_collection_contract)


ROOT = Path(__file__).resolve().parents[1]


def test_adaptive_reward_rollouts_use_every_live_pair_concurrently(
        monkeypatch, tmp_path):
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
        if seed < 80003:
            barrier.wait(timeout=2.0)
        success = seed in {80000, 80001}
        return [object()], {"paper_success": success}

    def episode_records(_rows, metric, *, episode_id, **_kwargs):
        success = int(metric["paper_success"])
        return [{
            "episode_id": episode_id,
            "success": success,
            "failure_type": "none" if success else "collision",
            "touchdown_error": .1,
            "touchdown_vertical_speed": .1,
        }]

    def build(records, **_kwargs):
        return {
            "episode_id": np.asarray([row["episode_id"] for row in records]),
            "success": np.asarray([row["success"] for row in records]),
            "failure_type": np.asarray(
                [row["failure_type"] for row in records]),
            "touchdown_error": np.asarray(
                [row["touchdown_error"] for row in records]),
            "touchdown_vertical_speed": np.asarray(
                [row["touchdown_vertical_speed"] for row in records]),
            "split": np.asarray(["validation"] * len(records)),
        }

    monkeypatch.setattr(pipeline_runner, "LiveShinEnvironment", Environment)
    monkeypatch.setattr(pipeline_runner, "collect_episode_resilient", collect)
    monkeypatch.setattr(
        pipeline_runner, "adaptive_episode_records", episode_records)
    monkeypatch.setattr(pipeline_runner, "build_adaptive_dataset", build)
    monkeypatch.setattr(
        pipeline_runner, "save_adaptive_dataset",
        lambda dataset, *_args, **_kwargs: {
            "episodes": len(np.unique(dataset["episode_id"]))})

    class Monitor:
        def stage(self, *_args):
            pass

    contexts = [{
        "cfg": SimpleNamespace(pair=index, sim=SimpleNamespace(max_steps=8)),
        "camera": object(), "monitor": Monitor(),
    } for index in range(3)]
    dataset, manifest, _path, steps = pipeline_runner._collect_adaptive_data(
        cfg=contexts[0]["cfg"], camera=contexts[0]["camera"], model=object(),
        source_pipeline="no_se_fixed", config={
            "seed": 7,
            "adaptive_reward_design": {
                "episodes_quick": 4, "max_episodes_quick": 4,
                "minimum_successful_episodes": 2,
                "minimum_failed_episodes": 2,
                "minimum_risky_failures": 1,
            },
        }, config_hash="test", results_dir=tmp_path, mode="quick",
        monitor=contexts[0]["monitor"], parallel_contexts=contexts)

    assert active_pairs == {0, 1, 2}
    assert manifest["episodes"] == 4
    assert steps == 4
    assert dataset["episode_id"].tolist() == [1, 2, 3, 4]


def _demonstration_pairs(count):
    """Live configs for ``count`` spawned pairs, as the runner builds them."""
    from ontology_rgat.config import default_config

    base = default_config()
    base.sim.max_steps = 40
    return [pipeline_runner._pair_live_config(base, index, count)
            for index in range(count)]


def _demonstration_config(**overrides):
    cloning = {
        "enabled": True, "source_pipeline": "no_se_fixed",
        "successful_episodes": 2, "max_attempts": 6, "curriculum": 1.0,
    }
    cloning.update(overrides)
    return {"seeds": {"behavior_cloning_start": 90000},
            "behavior_cloning": cloning}


def _stub_demonstration_stage(monkeypatch, collect, *, results_dir,
                              encoded=None):
    encoder = Path(results_dir) / "models/shared/keypoint_encoder.pt"
    encoder.parent.mkdir(parents=True, exist_ok=True)
    encoder.write_bytes(b"frozen keypoint encoder")

    class Environment:
        def __init__(self, cfg, _camera, horizon_steps):
            self.cfg = cfg
            assert horizon_steps == 40

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    def encode(_model, rows, *, episode_id, **_kwargs):
        if encoded is not None:
            encoded.append((int(episode_id), int(rows[0]["seed"])))
        return {"episode_id": int(episode_id)}

    monkeypatch.setattr(pipeline_runner, "LiveShinEnvironment", Environment)
    monkeypatch.setattr(pipeline_runner, "collect_episode_resilient", collect)
    monkeypatch.setattr(
        pipeline_runner, "_build_model",
        lambda *_args, **_kwargs: SimpleNamespace(modules=lambda: ()))
    monkeypatch.setattr(
        pipeline_runner, "encoded_demonstration_episode", encode)
    monkeypatch.setattr(
        pipeline_runner, "merge_encoded_demonstrations",
        lambda dataset, episode: [*(dataset or []), episode])
    monkeypatch.setattr(
        pipeline_runner, "save_encoded_demonstrations",
        lambda _path, dataset, **_kwargs: {
            "successful_episodes": len(dataset), "transitions": len(dataset),
            "dataset": dataset})


class _StageMonitor:
    def stage(self, *_args, **_kwargs):
        pass


def test_teacher_demonstrations_fly_every_spawned_pair_concurrently(
        monkeypatch, tmp_path):
    """The warm start runs before the first PPO worker, so every pair is free.

    Flying it on one pair left three of four spawned UAV/UGV pairs idle for
    the whole stage -- up to 40 attempts of real Isaac/PX4 flight time.
    """
    barrier = threading.Barrier(3)
    active_pairs = set()
    encoded = []

    def collect(environment, _model, _method, seed, **_kwargs):
        active_pairs.add(int(environment.cfg.external.pair_index))
        barrier.wait(timeout=10.0)
        return ([{"seed": int(seed)}], {
            "seed": int(seed), "steps": 7, "strict_success": 0.0,
            "paper_success": float(int(seed) in {90000, 90002})})

    _stub_demonstration_stage(monkeypatch, collect, results_dir=tmp_path,
                              encoded=encoded)
    pairs = _demonstration_pairs(3)
    contexts = [{"cfg": cfg, "camera": object(), "monitor": _StageMonitor()}
                for cfg in pairs]

    payload = pipeline_runner._prepare_fast_demonstrations(
        cfg=pairs[0], camera=contexts[0]["camera"],
        config=_demonstration_config(), config_hash="test",
        keypoint_pretraining=None, results_dir=tmp_path, device="cpu",
        model_seed=3, monitor=contexts[0]["monitor"],
        parallel_contexts=contexts)

    assert active_pairs == {0, 1, 2}
    assert payload["successful_episodes"] == 2
    # Episode ids follow the seeds, not the thread that happened to finish
    # first, so the stored set is the one those seeds make sequentially.
    assert encoded == [(1, 90000), (2, 90002)]
    attempts = pipeline_runner._read_csv(
        tmp_path / "training" / next(
            path.name for path in (tmp_path / "training").iterdir()
            if path.name.startswith("teacher_attempts_")))
    assert sorted(int(float(row["seed"])) for row in attempts) == [
        90000, 90001, 90002]
    assert sorted(int(float(row["physical_pair_index"]))
                  for row in attempts) == [0, 1, 2]


def test_parallel_demonstration_flights_never_exceed_the_attempt_budget(
        monkeypatch, tmp_path):
    """A batch is sized by what the attempt budget still pays for."""
    flown = []

    def collect(environment, _model, _method, seed, **_kwargs):
        flown.append(int(seed))
        return ([{"seed": int(seed)}], {
            "seed": int(seed), "steps": 5, "strict_success": 0.0,
            "paper_success": 0.0})

    _stub_demonstration_stage(monkeypatch, collect, results_dir=tmp_path)
    pairs = _demonstration_pairs(3)
    contexts = [{"cfg": cfg, "camera": object(), "monitor": _StageMonitor()}
                for cfg in pairs]

    with pytest.raises(RuntimeError, match="did not produce enough"):
        pipeline_runner._prepare_fast_demonstrations(
            cfg=pairs[0], camera=contexts[0]["camera"],
            config=_demonstration_config(successful_episodes=4,
                                         max_attempts=4),
            config_hash="test", keypoint_pretraining=None,
            results_dir=tmp_path, device="cpu", model_seed=3,
            monitor=contexts[0]["monitor"], parallel_contexts=contexts)

    assert flown == [90000, 90001, 90002, 90003]


def test_a_failed_pair_is_recovered_only_once_the_batch_has_landed(
        monkeypatch, tmp_path):
    """Rebuilding the shared stack mid-batch would take the siblings' flights.

    The stack is one process for every pair, so a worker that recovers while a
    sibling is still in step() ends that sibling's episode too. The recovery
    is therefore deferred to the main thread, after the batch has joined.
    """
    flying = set()
    recoveries = []
    order = []

    class Environment:
        def __init__(self, cfg, _camera, horizon_steps):
            self.cfg = cfg
            self.index = int(cfg.external.pair_index)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def recover_infrastructure(self):
            recoveries.append(self.index)
            assert not flying, "the stack was rebuilt under a flying pair"

    def collect(environment, _model, _method, seed, **_kwargs):
        flying.add(environment.index)
        try:
            barrier.wait(timeout=10.0)
            if environment.index == 1:
                raise BridgeError("PX4 preflight refused")
            order.append(int(seed))
            return ([{"seed": int(seed)}], {
                "seed": int(seed), "steps": 6, "strict_success": 0.0,
                "paper_success": float(int(seed) == 90000)})
        finally:
            flying.discard(environment.index)

    barrier = threading.Barrier(3)
    _stub_demonstration_stage(monkeypatch, collect, results_dir=tmp_path)
    monkeypatch.setattr(pipeline_runner, "LiveShinEnvironment", Environment)
    pairs = _demonstration_pairs(3)
    contexts = [{"cfg": cfg, "camera": object(), "monitor": _StageMonitor()}
                for cfg in pairs]

    payload = pipeline_runner._prepare_fast_demonstrations(
        cfg=pairs[0], camera=contexts[0]["camera"],
        config=_demonstration_config(successful_episodes=1),
        config_hash="test", keypoint_pretraining=None, results_dir=tmp_path,
        device="cpu", model_seed=3, monitor=contexts[0]["monitor"],
        parallel_contexts=contexts)

    assert payload["successful_episodes"] == 1
    assert recoveries == [1]


def test_each_demonstration_seed_flies_its_own_training_deck(
        monkeypatch, tmp_path):
    """The rotation is by seed, so batching and resuming cannot change it.

    The student is cloned from these flights and then trained on the same
    decks. Flown on one deck while training used six, the clone had never
    chased a deck that drives away and the baseline arm's health gate stopped
    the run at episode 48 (2026-09-21).
    """
    flown = {}

    def collect(environment, _model, _method, seed, *, scenario, **_kwargs):
        flown[int(seed)] = str(scenario)
        return ([{"seed": int(seed)}], {
            "seed": int(seed), "steps": 9, "strict_success": 0.0,
            "paper_success": float(int(seed) in {90002, 90007})})

    _stub_demonstration_stage(monkeypatch, collect, results_dir=tmp_path)
    decks = ["straight_8mps", "circle", "zigzag"]
    pairs = _demonstration_pairs(2)
    contexts = [{"cfg": cfg, "camera": object(), "monitor": _StageMonitor()}
                for cfg in pairs]

    payload = pipeline_runner._prepare_fast_demonstrations(
        cfg=pairs[0], camera=contexts[0]["camera"],
        config=_demonstration_config(successful_episodes=2, max_attempts=8,
                                     scenarios=decks),
        config_hash="test", keypoint_pretraining=None, results_dir=tmp_path,
        device="cpu", model_seed=3, monitor=contexts[0]["monitor"],
        parallel_contexts=contexts)

    assert payload["successful_episodes"] == 2
    # seed0 is 90000: the deck is (seed - seed0) % len(decks), whatever pair
    # flew it and whichever batch it landed in.
    for seed, scenario in flown.items():
        assert scenario == decks[(seed - 90000) % len(decks)], seed
    assert len(flown) >= 6, "the rotation should have reached every deck"


def test_the_demonstrations_fly_the_decks_the_student_is_trained_on():
    """A warm start on a different deck teaches the wrong thing.

    Checked against the config a bare ``./run.sh`` runs as well as the six-deck
    comparison, because the headline experiment changed on 2026-09-22 and a
    test naming one file stops guarding the default when it moves.
    """
    from conftest import default_experiment_config

    for path in {default_experiment_config(),
                 ROOT / "config/experiments/two_pipeline_comparison.yaml"}:
        config = load_experiment(path)
        cloning = pipeline_runner.behavior_cloning_settings(config)
        training = [str(name) for name in (config.get("training") or {}).get(
            "scenarios", ())]
        assert training, f"{path.name} should declare its training decks"
        demonstrations = [str(name) for name in cloning.get("scenarios", ())]
        assert demonstrations == training, path.name


def test_completed_system_hold_restarts_a_dead_owned_stack():
    events = []

    class Stack:
        checks = 0

        def is_ready(self):
            self.checks += 1
            return self.checks > 1

        def restart(self):
            events.append("restart")

    class Monitor:
        def stage(self, stage, detail):
            events.append((stage, detail))

    sleeps = []

    def stop_after_two_polls(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 2:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _hold_after_complete(Stack(), Monitor(), sleep=stop_after_two_polls)

    assert events[0] == (
        "complete", "results saved · simulator monitoring active")
    assert events.count("restart") == 1
    assert sleeps == [2.0, 2.0]


def _model(name):
    torch.manual_seed(5)
    return PipelineActorCritic(
        image_embedding=16, lstm_hidden=12, latent_dim=16,
        actor_hidden=8, critic_hidden=8, pipeline=name)


def _batch(model):
    images = torch.rand(1, 2, 1, 32, 32)
    proprio = torch.tensor([[[0., 0., 0., 1., 0., 0., 0.],
                             [0., 0., 0., 1., 0., 0., 0.]]])
    truth = torch.zeros(1, 2, 6)
    with torch.no_grad():
        out = model(images, proprio, true_relative_state=truth)
        pre = out.action_mean.clone()
        action = torch.tanh(pre)
        old = model.log_prob(pre, action, out.action_mean, out.action_std)
    return {
        "images": images, "proprioception": proprio,
        "true_relative_state": truth, "pre_squash_action": pre,
        "action": action, "old_log_prob": old,
        "advantage": torch.ones(1, 2), "return": torch.zeros(1, 2),
        "episode_start": torch.tensor([[True, False]]),
        "truth_valid": torch.ones(1, 2, dtype=torch.bool),
    }


def _observation(value=0.5):
    return SemanticObservation(
        keypoint_confidence=value, visible_keypoint_fraction=value,
        image_alignment=value,
        apparent_target_scale=value, image_plane_motion_safety=value,
        scale_rate_safety=value, visibility_memory=value,
        reacquisition_trend=value, vertical_motion_safety=value,
        attitude_stability=value, battery_risk=1.0 - value,
        visual_loss_risk=1.0 - value,
        centroid_xy=(0.0, 0.0), raw_scale=value * .2)


def _semantic_dataset():
    dataset = None
    for episode in range(1, 5):
        graph = semantic_graph(_observation(.2 * episode))
        current = semantic_episode_dataset(
            [{"graph_X": graph.X, "step_id": 0},
             {"graph_X": graph.X, "step_id": 1}],
            success=episode % 2 == 0, episode_id=episode,
            seed=100 + episode, sample_stride=1)
        dataset = merge_semantic_datasets(dataset, current)
    return dataset


def test_visual_teacher_is_directional_without_random_policy_leakage():
    semantic = replace(
        _observation(.8), centroid_xy=(.4, -.2),
        image_alignment=.8, keypoint_confidence=.8,
        visible_keypoint_fraction=1.0, visual_loss_risk=0.0,
        apparent_target_scale=.2)
    teacher = _behavior_transform(
        0, policy_blend=0.0, servo_gain=1.2,
        noise_std=0.0, yaw_blend=0.0)
    action = teacher(
        0, np.ones(4), semantic, np.random.default_rng(1))
    np.testing.assert_allclose(action, [.36, .18, -.55, 0.0], atol=1e-8)


def test_deadline_teacher_tracks_deck_and_descends_only_after_alignment():
    semantic = replace(
        _observation(.8), visible_keypoint_fraction=1.0,
        visual_loss_risk=0.0)
    # Platform moves forward at 0.2 m/s relative to a hovering UAV and is
    # 0.8 m ahead. The label includes feed-forward and position correction,
    # while altitude is held until lateral alignment.
    chase = _privileged_velocity_teacher_action(
        [.8, 0., -2., .2, 0., 0.], [0., 0., 0.], semantic,
        [2., 2., 1.], noise_std=0.)
    assert chase[0] == pytest.approx(.30)
    assert chase[2] == 0.0

    aligned = _privileged_velocity_teacher_action(
        [.1, 0., -2., 0., 0., 0.], [.2, 0., 0.], semantic,
        [2., 2., 1.], noise_std=0.)
    np.testing.assert_allclose(aligned, [.1175, 0., -.35, 0.], atol=1e-8)


def test_deadline_teacher_only_climbs_on_high_altitude_visual_loss():
    lost = replace(
        _observation(.2), visible_keypoint_fraction=0.0,
        visual_loss_risk=1.0)
    high = _privileged_velocity_teacher_action(
        [0., 0., -2., 0., 0., 0.], [0., 0., 0.], lost,
        [2., 2., 1.], noise_std=0.)
    assert high[2] > 0.0
    flare = _privileged_velocity_teacher_action(
        [0., 0., -.3, 0., 0., 0.], [0., 0., 0.], lost,
        [2., 2., 1.], noise_std=0.)
    assert flare[2] < 0.0


def test_learning_efficiency_reports_shared_behavior_cloning_cost():
    rows = [{
        "pipeline": "onto_rgat_adaptive_weight_no_se",
        "optimization_phase": "ppo", "episode": episode,
        "ppo_episode": episode,
        "ppo_environment_steps": episode * 10,
        "paper_success": float(episode == 2),
    } for episode in (1, 2)]
    [summary] = learning_efficiency(
        rows, reward_design_episodes=4, reward_design_steps=40,
        behavior_cloning_episodes=3, behavior_cloning_steps=30)
    assert summary["behavior_cloning_episodes"] == 3
    assert summary["total_environment_episodes"] == 9
    assert summary["total_environment_steps"] == 90


def test_learning_efficiency_uses_per_pipeline_reward_design_cost():
    rows = [{"pipeline": name, "optimization_phase": "ppo", "episode": 1,
             "ppo_episode": 1, "ppo_environment_steps": 10,
             "paper_success": 0.0}
            for name in ("no_se_fixed", "onto_rgat_adaptive_weight_no_se")]
    summary = {row["pipeline"]: row for row in learning_efficiency(
        rows, reward_design_costs={
            "onto_rgat_adaptive_weight_no_se": {"episodes": 12, "steps": 120}})}
    assert summary["no_se_fixed"]["reward_design_environment_steps"] == 0
    assert summary["onto_rgat_adaptive_weight_no_se"][
        "reward_design_environment_steps"] == 120


def test_safe_deployment_checkpoint_outranks_unsafe_contact_and_timeout():
    safe = {"paper_success": 1, "pad_contact": 1,
            "touchdown_lateral_error": .2, "geometric_fov_loss_fraction": .1,
            "touchdown_relative_horizontal_velocity": .1}
    unsafe = {"paper_success": 0, "pad_contact": 1, "unsafe_pad_contact": 1,
              "crash_failure": 1, "touchdown_lateral_error": .1,
              "geometric_fov_loss_fraction": 0.0,
              "touchdown_relative_horizontal_velocity": .1}
    timeout = {"paper_success": 0, "pad_contact": 0,
               "touchdown_lateral_error": .5, "geometric_fov_loss_fraction": .2,
               "touchdown_relative_horizontal_velocity": .1}
    assert deployment_checkpoint_score(safe) > deployment_checkpoint_score(timeout)
    assert deployment_checkpoint_score(timeout) > deployment_checkpoint_score(unsafe)


def test_held_out_checkpoint_selection_prioritizes_repeatable_safe_landings():
    lucky = aggregate_deployment_validation([
        {"paper_success": 1, "pad_contact": 1, "geometric_fov_loss_fraction": .1,
         "touchdown_lateral_error": .1},
        {"paper_success": 0, "pad_contact": 0, "geometric_fov_loss_fraction": .8,
         "touchdown_lateral_error": 2.0},
        {"paper_success": 0, "pad_contact": 0, "geometric_fov_loss_fraction": .8,
         "touchdown_lateral_error": 2.0},
    ])
    repeatable = aggregate_deployment_validation([
        {"paper_success": 1, "pad_contact": 1, "geometric_fov_loss_fraction": .2,
         "touchdown_lateral_error": .2},
        {"paper_success": 1, "pad_contact": 1, "geometric_fov_loss_fraction": .2,
         "touchdown_lateral_error": .2},
        {"paper_success": 0, "pad_contact": 0, "geometric_fov_loss_fraction": .7,
         "touchdown_lateral_error": 1.0},
    ])
    assert repeatable["success_rate"] == pytest.approx(2 / 3)
    assert deployment_validation_key(repeatable) > deployment_validation_key(lucky)


def test_held_out_checkpoint_selection_rejects_unsafe_contact_on_tie():
    safe_miss = aggregate_deployment_validation([
        {"paper_success": 0, "pad_contact": 0, "unsafe_pad_contact": 0,
         "geometric_fov_loss_fraction": .4, "touchdown_lateral_error": .5},
    ])
    collision = aggregate_deployment_validation([
        {"paper_success": 0, "pad_contact": 1, "unsafe_pad_contact": 1,
         "geometric_fov_loss_fraction": .1, "touchdown_lateral_error": .4},
    ])
    assert deployment_validation_key(safe_miss) > deployment_validation_key(collision)


def test_checkpoint_validation_seeds_are_disjoint_and_crossover_balances_pairs():
    pipelines = ["shin_se_fixed", "no_se_fixed",
                 "onto_rgat_adaptive_weight_no_se"]
    plan = paired_seed_plan(pipelines, {"training_random_walk": 3}, seed0=5000)
    tasks = _crossover_evaluation_tasks(plan, pipelines, 3)
    assert [len(rows) for rows in tasks] == [3, 3, 3]
    for name in pipelines:
        assert {row["physical_pair_index"] for rows in tasks for row in rows
                if row["method"] == name} == {0, 1, 2}
    validation = _checkpoint_validation_plan(
        pipelines[0], ["training_random_walk", "circle", "zigzag"], seed0=4000)
    assert {row["seed"] for row in validation}.isdisjoint(
        {row["seed"] for row in plan})
    assignments = [_balanced_training_pair_assignment(
        pipelines, 3, replicate)[0] for replicate in range(3)]
    for name in pipelines:
        assert {assignment[name] for assignment in assignments} == {0, 1, 2}


def test_robust_adaptive_profile_rejects_the_collapsed_seminar_artifact():
    config = load_experiment(
        ROOT / "config/experiments/seminar_fast_comparison.yaml")
    settings = _adaptive_reward_settings(config, robust=True)
    assert settings["runtime_profile"] == "robust_live_v2"
    assert settings["semantic_potential_shaping_lambda"] >= 1.5
    assert settings["loss"]["contextual_weight"] >= 1.5
    assert settings["quality_gate"]["minimum_mean_weight_cv"] == pytest.approx(.003)
    metadata = {
        "dataset_manifest": {
            "episodes": 12, "validation_episodes": 2,
            "outcome_strata": {"unsafe_pad_contact": 0}},
        "validation_episode_ids": [10, 11],
        "metrics": {
            "validation_accuracy": .5,
            "validation_ranking_accuracy": .5,
            "mean_weight_coefficient_of_variation": .001,
            "potential_observability_monotonic_compliance": .96,
        },
    }
    issues = _adaptive_artifact_quality_issues(
        metadata, settings, minimum_episodes=24)
    assert any("dataset episodes" in issue for issue in issues)
    assert any("validation ranking accuracy" in issue for issue in issues)
    assert any("weight CV" in issue for issue in issues)
    assert any("unsafe failure" in issue for issue in issues)


def test_excessive_post_update_kl_rolls_back_the_ppo_epoch():
    model = _model("no_se")
    optimizer = torch.optim.Adam(model.parameters(), lr=.05)
    images = torch.randint(0, 255, (4, 32, 32), dtype=torch.uint8)
    proprio = np.tile(np.asarray([0., 0., 0., 1., 0., 0., 0.]), (4, 1))
    truth = np.zeros((4, 6), dtype=np.float32)
    hidden = model.initial_state(1)
    with torch.no_grad():
        output = model(images[:, None].float()[None] / 255.0,
                       torch.as_tensor(proprio[None], dtype=torch.float32),
                       true_relative_state=torch.as_tensor(truth[None]))
        pre = output.action_mean[0].numpy()
        action = torch.tanh(output.action_mean[0]).numpy()
        log_prob = model.log_prob(
            output.action_mean, torch.tanh(output.action_mean),
            output.action_mean, output.action_std)[0].numpy()
        values = output.value[0].numpy()
    rows = [{
        "image": images[index].numpy(), "proprioception": proprio[index],
        "truth": truth[index], "pre_squash": pre[index],
        "action": action[index], "log_prob": float(log_prob[index]),
        "value": float(values[index]), "reward": float(index == 3),
        "done": float(index == 3),
        "hidden_h": hidden[0].numpy(), "hidden_c": hidden[1].numpy(),
    } for index in range(4)]
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    metrics = update_episode(
        model, optimizer, rows, epochs=1, sequence_length=4,
        target_kl=1e-12, rollback_on_excessive_kl=True,
        entropy_coef=0.0, value_coef=0.0)
    assert metrics["ppo_kl_rollback_count"] == 1.0
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0.0, atol=0.0)
    recovered = update_episode(
        model, optimizer, rows, epochs=1, sequence_length=4,
        target_kl=100.0, maximum_learning_rate=.05,
        learning_rate_recovery_factor=2.0,
        learning_rate_recovery_kl_fraction=.5,
        entropy_coef=0.0, value_coef=0.0)
    assert recovered["learning_rate_recovered"] == 1.0
    assert recovered["effective_learning_rate"] == pytest.approx(.05)


def test_primary_specs_encode_the_intended_information_boundaries():
    baseline = PIPELINES["shin_se_fixed"]
    proposed = PIPELINES["shin_se_onto_rgat_recovery"]
    for spec in (baseline, proposed):
        assert spec.state_estimation_enabled
        assert spec.auxiliary_estimation_loss_enabled
        assert spec.active_perception_enabled
        assert spec.reward_mode == "shin_table_active"
    assert not baseline.ontology_enabled
    assert proposed.ontology_enabled
    assert proposed.fov_risk_reward_enabled


def test_estimator_outputs_physical_units_beyond_tanh_and_normalizes_loss():
    model = _model("shin_se")
    linear = model.temporal_backbone.latent_head[0]
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.zero_()
        linear.bias[:6] = torch.tensor([2.0, -2.0, 1.5, 1.0, -1.0, 2.0])
        output = model(torch.zeros(1, 1, 1, 32, 32),
                       torch.zeros(1, 1, 7))
    # Physical estimator channels are no longer trapped in [-1, 1].
    assert output.relative_state[0, 0, 0].item() == pytest.approx(6.0)
    assert output.relative_state[0, 0, 2].item() == pytest.approx(12.0)
    # Policy-only latent features remain bounded.
    assert torch.max(torch.abs(output.latent[..., 6:])).item() <= 1.0
    target = torch.tensor([[[3.0, 3.0, 8.0, 3.0, 3.0, 2.0]]])
    zero = torch.zeros_like(target)
    assert model.relative_state_head.loss(zero, target).item() == pytest.approx(1.0)


def test_normalized_estimator_loss_restores_active_reward_gradient():
    model = _model("shin_se")
    physical_error = torch.tensor([[[1.5, 1.5, 4.0, 0.5, 0.5, 0.5]]])
    normalized = model.relative_state_head.loss(
        torch.zeros_like(physical_error), physical_error).item()
    raw = physical_error.square().mean().item()
    cfg = ShinRewardConfig()
    assert active_perception_reward(raw, cfg) == pytest.approx(-0.1)
    assert normalized == pytest.approx(0.14467593)
    assert -0.02 < active_perception_reward(normalized, cfg) < -0.01


def test_three_pipeline_keeps_all_physical_evaluation_scenarios():
    config = load_experiment(
        ROOT / "config/experiments/three_pipeline_comparison.yaml")
    assert set(config["evaluation"]) == {
        "training_random_walk", "straight_8mps", "linear_acceleration_wave",
        "circle", "zigzag", "u_turn", "vertical_heave_boat",
    }


def test_training_health_gate_reports_independent_learning_failures():
    history = [{
        "episode": episode, "paper_success": 0,
        "geometric_fov_loss_fraction": .2, "battery_depleted": 0,
        "position_rmse": 4.0,
        "active_reward_saturation_fraction": 1.0,
    } for episode in range(1, 41)]
    issue = training_health_issue(history, {
        "health_window_episodes": 20, "health_grace_episodes": 40,
    })
    assert "no landing" in issue
    assert "position RMSE stalled" in issue
    assert "active reward saturation stalled" in issue


def test_training_health_gate_never_terminates_on_the_compared_fov_metrics():
    """The Baseline's weakness is the result, not a reason to stop the run.

    Geometric FOV loss, reacquisition rate and low-keypoint-visibility descent
    are exactly what the proposed FOV-risk reward is meant to improve. A gate
    that aborted the Baseline for scoring badly on them would delete the very
    effect the experiment measures.
    """
    history = [{
        "episode": episode, "paper_success": 1,
        "geometric_fov_loss_fraction": .95, "battery_depleted": 0,
        "position_rmse": 0.2, "active_reward_saturation_fraction": 0.0,
        "geometric_fov_loss_events": 4,
        "geometric_fov_reacquisition_events": 0,
        "geometric_fov_reacquisition_rate": 0.0,
        "descent_during_low_keypoint_visibility_fraction": .9,
        "low_keypoint_visibility_fraction": .9,
        # The encoder is still working whenever the pad is actually in frame.
        "blind_perception_while_geometric_fov_fraction": .1,
        "geometric_fov_scored_step_fraction": .3,
    } for episode in range(1, 41)]

    assert training_health_issue(history, {
        "health_window_episodes": 20, "health_grace_episodes": 40,
    }) is None


def test_training_health_gate_still_stops_a_dead_perception_pipeline():
    """Blind while the pad is geometrically in frame is infrastructure, not policy."""
    history = [{
        "episode": episode, "paper_success": 1,
        "geometric_fov_loss_fraction": .1, "battery_depleted": 0,
        "position_rmse": 0.2, "active_reward_saturation_fraction": 0.0,
        "blind_perception_while_geometric_fov_fraction": 1.0,
        "geometric_fov_scored_step_fraction": .9,
    } for episode in range(1, 41)]

    issue = training_health_issue(history, {
        "health_window_episodes": 20, "health_grace_episodes": 40,
    })
    assert issue is not None
    assert "perception-infrastructure failure" in issue


def test_training_health_gate_catches_battery_failure_before_learning_grace():
    history = [{
        "episode": episode, "paper_success": 0,
        "geometric_fov_loss_fraction": .1, "battery_depleted": 1,
    } for episode in range(1, 21)]
    issue = training_health_issue(history, {
        "health_window_episodes": 20, "health_grace_episodes": 40,
        "health_max_battery_depletion_fraction": .6,
    })
    assert issue == "battery depletion is 100.0% (limit 60.0%)"


def test_training_health_gate_ignores_depletion_the_seeded_reserve_forces():
    """9--55 hover seconds against a 30 s flight depletes half the episodes."""
    history = []
    for episode in range(1, 21):
        short_reserve = episode % 3 != 0          # 13 of 20 episodes
        history.append({
            "episode": episode, "paper_success": 0, "geometric_fov_loss_fraction": .1,
            "battery_depleted": 1 if short_reserve else 0,
            "battery_energy_initial_j": 3000.0 if short_reserve else 8000.0,
            "battery_energy_used_j": 3000.0 if short_reserve else 5500.0,
            "status": "battery_depleted" if short_reserve else "timeout",
        })
    ppo = {"health_window_episodes": 20, "health_grace_episodes": 40,
           "health_max_battery_depletion_fraction": .6}
    assert training_health_issue(history, ppo) is None

    # A policy that burns the pack despite an ample reserve is still caught.
    wasteful = [dict(row, battery_depleted=1, battery_energy_used_j=8000.0,
                     status="battery_depleted")
                if row["status"] == "timeout" else dict(row)
                for row in history]
    wasteful[2].update(status="timeout", battery_depleted=0,
                       battery_energy_used_j=5500.0)
    issue = training_health_issue(wasteful, ppo)
    assert issue is not None
    assert issue.startswith("battery depletion beyond the seeded reserve is")


def test_short_live_budget_extends_only_the_no_landing_grace():
    history = [{
        "episode": episode, "paper_success": 0,
        "geometric_fov_loss_fraction": .2, "battery_depleted": 0,
        "position_rmse": 1.0,
        "active_reward_saturation_fraction": .1,
        "geometric_fov_loss_events": 0,
    } for episode in range(1, 41)]
    ppo = {"health_window_episodes": 20, "health_grace_episodes": 40}
    assert training_health_issue(
        history, ppo, planned_policy_episodes=160) is None

    history.extend({
        "episode": episode, "paper_success": 0,
        "geometric_fov_loss_fraction": .2, "battery_depleted": 0,
        "position_rmse": 1.0,
        "active_reward_saturation_fraction": .1,
        "geometric_fov_loss_events": 0,
    } for episode in range(41, 121))
    issue = training_health_issue(
        history, ppo, planned_policy_episodes=160)
    assert issue == "no landing in the last 20 policy episodes"


def test_no_landing_grace_is_capped_for_large_publication_run():
    history = [{
        "episode": episode, "paper_success": 0,
        "geometric_fov_loss_fraction": .2, "battery_depleted": 0,
        "position_rmse": 1.0,
        "active_reward_saturation_fraction": .1,
        "geometric_fov_loss_events": 0,
    } for episode in range(1, 121)]
    issue = training_health_issue(
        history, {}, planned_policy_episodes=40960)
    assert issue == "no landing in the last 20 policy episodes"


def test_transport_failure_restarts_and_retries_the_same_seed(monkeypatch):
    attempts = []
    recoveries = []

    def collect(_env, _model, _method, seed, **_kwargs):
        attempts.append(seed)
        if len(attempts) == 1:
            raise GatewayTimeout("gateway lost one state reply")
        return ["complete"], {"seed": seed}

    monkeypatch.setattr(recurrent_train, "collect_episode", collect)
    env = type("Env", (), {
        "cfg": type("Cfg", (), {"external": {"reset_recoveries": 2}})(),
        "recover_infrastructure": lambda self: recoveries.append("restart"),
    })()

    rows, metric = collect_episode_resilient(env, object(), "shin_se", 123)

    assert rows == ["complete"] and metric["seed"] == 123
    assert attempts == [123, 123]
    assert recoveries == ["restart"]


def test_offboard_failsafe_restarts_and_retries_the_same_seed(monkeypatch):
    attempts = []
    recoveries = []

    def collect(_env, _model, _method, seed, **_kwargs):
        attempts.append(seed)
        if len(attempts) == 1:
            raise PX4Failsafe(
                ["offboard_control_signal_lost"], recoverable=True)
        return ["complete"], {"seed": seed}

    monkeypatch.setattr(recurrent_train, "collect_episode", collect)
    env = type("Env", (), {
        "cfg": type("Cfg", (), {"external": {"episode_recoveries": 2}})(),
        "recover_infrastructure": lambda self: recoveries.append("restart"),
    })()

    rows, metric = collect_episode_resilient(env, object(), "shin_se", 321)

    assert rows == ["complete"] and metric["seed"] == 321
    assert attempts == [321, 321]
    assert recoveries == ["restart"]


def test_exhausted_entry_reset_restarts_and_retries_same_policy_seed(monkeypatch):
    attempts = []
    recoveries = []

    def collect(_env, _model, _method, seed, **_kwargs):
        attempts.append(seed)
        if len(attempts) == 1:
            raise EntryResetError("entry hover failed after local retries")
        return ["complete"], {"seed": seed}

    monkeypatch.setattr(recurrent_train, "collect_episode", collect)
    env = type("Env", (), {
        "cfg": type("Cfg", (), {"external": {"episode_recoveries": 2}})(),
        "recover_infrastructure": lambda self: recoveries.append("restart"),
    })()

    rows, metric = collect_episode_resilient(env, object(), "shin_se", 20015)

    assert rows == ["complete"] and metric["seed"] == 20015
    assert attempts == [20015, 20015]
    assert recoveries == ["restart"]


def test_peer_restart_estimator_warmup_retries_same_policy_seed(monkeypatch):
    attempts = []
    recoveries = []

    def collect(_env, _model, _method, seed, **_kwargs):
        attempts.append(seed)
        if len(attempts) == 1:
            raise BridgeError("PX4 estimator state is not valid yet.")
        return ["complete"], {"seed": seed}

    monkeypatch.setattr(recurrent_train, "collect_episode", collect)
    env = type("Env", (), {
        "cfg": type("Cfg", (), {"external": {"episode_recoveries": 2}})(),
        "recover_infrastructure": lambda self: recoveries.append("restart"),
    })()

    rows, metric = collect_episode_resilient(env, object(), "shin_se", 20016)

    assert rows == ["complete"] and metric["seed"] == 20016
    assert attempts == [20016, 20016]
    assert recoveries == ["restart"]


def test_a_self_clearing_link_failsafe_retries_without_rebuilding_the_stack(
        monkeypatch, capsys):
    """Minutes of Isaac boot -- and the other pair's episode -- for a blip.

    The gateway classifies an OFFBOARD heartbeat loss as a recoverable
    transport fault and it clears in well under a second. The interrupted
    trajectory is still discarded; only the rebuild is skipped.
    """
    attempts = []
    recoveries = []
    cleared = []

    def collect(_env, _model, _method, seed, **_kwargs):
        attempts.append(seed)
        if len(attempts) == 1:
            raise PX4Failsafe(["offboard_control_signal_lost"], recoverable=True)
        return ["complete"], {"seed": seed}

    monkeypatch.setattr(recurrent_train, "collect_episode", collect)
    env = type("Env", (), {
        "cfg": type("Cfg", (), {"external": {"episode_recoveries": 2}})(),
        "bridge": type("Bridge", (), {
            "wait_for_failsafe_clear": lambda self, timeout=None: (
                cleared.append("waited") or True)})(),
        "recover_infrastructure": lambda self: recoveries.append("restart"),
    })()

    rows, metric = collect_episode_resilient(env, object(), "shin_se", 20017)

    assert rows == ["complete"] and metric["seed"] == 20017
    assert attempts == [20017, 20017]
    assert cleared == ["waited"] and recoveries == []
    assert "without rebuilding the simulator" in capsys.readouterr().out


def test_a_link_failsafe_that_does_not_clear_still_rebuilds_the_stack(monkeypatch):
    attempts = []
    recoveries = []

    def collect(_env, _model, _method, seed, **_kwargs):
        attempts.append(seed)
        if len(attempts) == 1:
            raise PX4Failsafe(["offboard_control_signal_lost"], recoverable=True)
        return ["complete"], {"seed": seed}

    monkeypatch.setattr(recurrent_train, "collect_episode", collect)
    env = type("Env", (), {
        "cfg": type("Cfg", (), {"external": {"episode_recoveries": 2}})(),
        "bridge": type("Bridge", (), {
            "wait_for_failsafe_clear": lambda self, timeout=None: False})(),
        "recover_infrastructure": lambda self: recoveries.append("restart"),
    })()

    rows, _metric = collect_episode_resilient(env, object(), "shin_se", 20018)

    assert rows == ["complete"] and recoveries == ["restart"]


def test_hard_px4_failsafe_is_not_retried(monkeypatch):
    recoveries = []

    def collect(*_args, **_kwargs):
        raise PX4Failsafe(["battery_warning_2"], recoverable=False)

    monkeypatch.setattr(recurrent_train, "collect_episode", collect)
    env = type("Env", (), {
        "cfg": type("Cfg", (), {"external": {"episode_recoveries": 2}})(),
        "recover_infrastructure": lambda self: recoveries.append("restart"),
    })()

    with pytest.raises(PX4Failsafe, match="battery_warning_2"):
        collect_episode_resilient(env, object(), "shin_se", 321)
    assert recoveries == []


def test_illegal_pipeline_combinations_are_rejected():
    with pytest.raises(ValueError, match="active perception"):
        PipelineSpec("bad", False, False, True, "shin_table_active",
                     False, None, False)
    with pytest.raises(ValueError, match="cannot enable state estimation"):
        PipelineSpec("bad", True, True, False, "semantic_pbrs",
                     True, "semantic_observation", True)
    with pytest.raises(ValueError, match="shin_table_active"):
        PipelineSpec("bad", False, False, False, "shin_table_active",
                     False, None, False)


def test_exact_total_budget_subtracts_shin_warmup_before_equal_ppo_split():
    assert episodes_per_method(800, 3, overhead_episodes=8) == 264
    with pytest.raises(ValueError, match="not divisible"):
        episodes_per_method(800, 3)


def test_shin_warmup_seeds_do_not_shift_the_common_ppo_seed_range():
    shin = controlled_training_seeds(
        20000, 264, warmup_episodes=8, warmup_seed0=19000)
    no_se = controlled_training_seeds(20000, 264)
    assert shin[:8] == list(range(19000, 19008))
    assert shin[8:] == no_se


@pytest.mark.parametrize("name", ["no_se", "onto_no_se"])
def test_estimator_free_pipeline_ppo_backward_has_no_estimation_objective(name):
    model = _model(name)
    assert model.relative_state_head is None
    output = model(_batch(model)["images"], _batch(model)["proprioception"],
                   true_relative_state=_batch(model)["true_relative_state"])
    assert output.relative_state is None
    loss, metrics = recurrent_ppo_loss(model, _batch(model), auxiliary_coef=999.0)
    loss.backward()
    assert "auxiliary_estimation_loss" not in metrics
    assert float(metrics["state_estimation_enabled"]) == 0.0
    assert any(parameter.grad is not None for parameter in model.actor.parameters())


def test_shin_pipeline_has_six_state_auxiliary_supervision():
    model = _model("shin_se")
    output = model(_batch(model)["images"], _batch(model)["proprioception"],
                   true_relative_state=_batch(model)["true_relative_state"])
    assert model.relative_state_head is not None
    assert output.relative_state.shape == (1, 2, 6)
    _, metrics = recurrent_ppo_loss(model, _batch(model))
    assert "auxiliary_estimation_loss" in metrics


def test_all_actors_have_equal_deployed_boundary_capacity_and_actions():
    models = [_model(name) for name in PIPELINES]
    assert {model.actor[0].in_features for model in models} == {17}
    assert {model.action_dim for model in models} == {4}
    assert len({sum(parameter.numel() for parameter in model.parameters())
                for model in models}) == 1
    assert {model.pipeline_spec.actor_latent_slice.start for model in models} == {6}


@pytest.mark.parametrize("name", list(PIPELINES))
def test_privileged_critic_truth_cannot_change_deployed_actor_output(name):
    model = _model(name).eval()
    batch = _batch(model)
    with torch.no_grad():
        zero = model(batch["images"], batch["proprioception"],
                     true_relative_state=torch.zeros_like(
                         batch["true_relative_state"]))
        one = model(batch["images"], batch["proprioception"],
                    true_relative_state=torch.ones_like(
                        batch["true_relative_state"]))
    assert torch.equal(zero.latent, one.latent)
    assert torch.equal(zero.action_mean, one.action_mean)
    assert not torch.equal(zero.value, one.value)


def test_semantic_graph_uses_only_visual_and_onboard_payload():
    keypoints = np.array([[-.2, -.2], [.2, -.2], [.3, 0.],
                          [.2, .2], [-.2, .2], [-.3, 0.]])
    observation = semantic_observation_from_payload({
        "keypoints": keypoints, "heatmaps": np.zeros((6, 4, 5)),
        "proprioception": [0, 0, 0, 1, 0, 0, 0], "battery_reserve": .8})
    graph = semantic_graph(observation)
    assert graph.X.shape == (SEMANTIC_GRAPH_INPUT_DIM, len(SEMANTIC_NODE_NAMES))
    assert graph.node_names[-1] == "SafeLanding"
    assert set(graph.relation_names) == {"indicates", "supports", "constrains", "self"}


def test_uninformative_heatmaps_cannot_manufacture_geometry_and_memory_decays():
    points = np.array([[-.2, -.2], [.2, -.2], [.3, 0.],
                       [.2, .2], [-.2, .2], [-.3, 0.]])
    peaked = np.zeros((6, 4, 5))
    peaked[:, 1, 2] = 20.0
    payload = {"keypoints": points, "heatmaps": peaked,
               "proprioception": [0, 0, 0, 1, 0, 0, 0]}
    seen = semantic_observation_from_payload(payload)
    assert seen.visible_keypoint_fraction == pytest.approx(1.0)
    blind = semantic_observation_from_payload(
        {**payload, "heatmaps": np.zeros((6, 4, 5))}, previous=seen, dt=.1)
    assert blind.visible_keypoint_fraction == 0.0
    assert blind.image_alignment == 0.0
    assert blind.apparent_target_scale == 0.0
    assert blind.image_plane_motion_safety == 0.0
    assert blind.visual_loss_risk > 0.0
    assert blind.visibility_memory < seen.visibility_memory
    assert blind.centroid_xy == pytest.approx(seen.centroid_xy)
    explicit_absence = semantic_observation_from_payload(
        {**payload, "keypoint_visibility": np.zeros(6)}, previous=seen, dt=.1)
    assert explicit_absence.visible_keypoint_fraction == 0.0
    assert explicit_absence.image_alignment == 0.0


def test_geometric_fov_and_keypoint_metrics_stay_separate_families():
    """Geometric frustum truth and neural perception quality are not the same.

    Row 2 is geometrically in view while the encoder reports nothing: that is
    perception degradation, and it must be counted as such rather than as an
    FOV loss.
    """
    names = {name: index for index, name in enumerate(SEMANTIC_FEATURE_NAMES)}
    low = np.ones(len(SEMANTIC_FEATURE_NAMES))
    low[names["keypoint_confidence"]] = 0.0
    low[names["visible_keypoint_fraction"]] = 0.0
    high = np.ones(len(SEMANTIC_FEATURE_NAMES))
    rows = [
        {"geometric_in_fov": False, "command": np.array([0, 0, .3, 0]),
         "semantic_features": high, "reward_parts": {"phi": .5, "phi_next": .2}},
        {"geometric_in_fov": False, "command": np.array([0, 0, .3, 0]),
         "semantic_features": low, "reward_parts": {"phi": .2, "phi_next": .3}},
        {"geometric_in_fov": True, "command": np.array([0, 0, .1, 0]),
         "semantic_features": low, "reward_parts": {"phi": .3, "phi_next": .7}},
        # Geometrically in view, but the encoder sees nothing and the policy
        # descends anyway: perception degradation, not an FOV loss.
        {"geometric_in_fov": True, "command": np.array([0, 0, -.2, 0]),
         "semantic_features": low, "reward_parts": {"phi": .7, "phi_next": .8}},
    ]
    metric = visual_recovery_metrics(
        rows, initial_in_fov=True, success=True, dt=.1)
    assert metric["geometric_fov_loss_events"] == 1
    assert metric["geometric_fov_reacquisition_events"] == 1
    assert metric["geometric_fov_reacquisition_rate"] == 1
    assert metric["mean_geometric_fov_reacquisition_time_s"] == pytest.approx(.2)
    assert metric["climb_during_geometric_fov_loss_fraction"] == 1
    assert metric["descent_during_geometric_fov_loss_fraction"] == 0
    assert metric["successful_recovery_landing"] == 1
    # Three of the four steps have unusable keypoints; one of those descends.
    assert metric["low_keypoint_visibility_fraction"] == pytest.approx(.75)
    assert metric["descent_during_low_keypoint_visibility_fraction"] == pytest.approx(1 / 3)
    assert metric["keypoint_confidence_mean"] == pytest.approx(.25)
    assert metric["visible_keypoint_fraction_mean"] == pytest.approx(.25)
    # The blind-perception witness only counts steps that were in frame.
    assert metric["blind_perception_while_geometric_fov_fraction"] == pytest.approx(.5)
    assert metric["geometric_fov_scored_step_fraction"] == pytest.approx(.5)


@pytest.mark.parametrize("field", [
    "estimated_relative_state", "true_relative_state", "simulator_truth",
    "relative_position", "platform_velocity"])
def test_semantic_graph_boundary_rejects_privileged_fields(field):
    payload = {"keypoints": np.zeros((6, 2)), "heatmaps": np.zeros((6, 4, 5)),
               "proprioception": [0, 0, 0, 1, 0, 0, 0], field: np.zeros(6)}
    with pytest.raises(ValueError, match="forbidden semantic field"):
        semantic_observation_from_payload(payload)


def test_ontology_reward_context_has_no_relative_state_field():
    names = {field.name for field in fields(OntologyRewardContext)}
    assert names == {"graph", "next_graph", "terminal"}
    assert not any("relative" in name or "estimate" in name for name in names)


def test_semantic_dataset_rejects_privileged_sample_fields():
    graph = semantic_graph(_observation())
    with pytest.raises(ValueError, match="forbidden semantic rollout field"):
        semantic_episode_dataset(
            [{"graph_X": graph.X, "step_id": 0,
              "true_relative_state": np.zeros(6)}],
            success=True, episode_id=1, seed=2)


def test_semantic_dataset_requires_both_terminal_classes():
    graph = semantic_graph(_observation())
    dataset = semantic_episode_dataset(
        [{"graph_X": graph.X, "step_id": 0}],
        success=True, episode_id=1, seed=2)
    with pytest.raises(ValueError, match="successful and failed"):
        validate_semantic_dataset(dataset, require_both_classes=True)


def test_semantic_monotonic_counterfactuals_only_degrade_allowed_observations():
    graph = semantic_graph(_observation(.7))
    source = graph.X.T[None]
    counterfactuals = semantic_monotonic_counterfactuals(source)
    assert counterfactuals.shape == (1, 3, *source.shape[1:])
    names = {name: index for index, name in enumerate(SEMANTIC_NODE_NAMES)}
    assert counterfactuals[0, 0, names["PerceptionQuality"], 0] == 0.0
    assert counterfactuals[0, 1, names["VisualLossRisk"], 0] == 1.0
    assert counterfactuals[0, 2, names["BatteryRisk"], 0] == 1.0


def test_rgat_split_is_by_whole_episode():
    dataset = _semantic_dataset()
    cfg = semantic_rgat_config(
        "quick", 7, {"epochs": 1, "batch_size": 2, "device": "cpu"})
    _, history = train_potential(dataset, cfg, verbose=False)
    assert history["split_unit"] == "episode"
    assert set(history["train_episode_ids"]).isdisjoint(
        history["validation_episode_ids"])
    outcomes = {int(ep): int(dataset["meta"][dataset["meta"][:, 0] == ep, 2][0])
                for ep in np.unique(dataset["meta"][:, 0])}
    assert {outcomes[ep] for ep in history["train_episode_ids"]} == {0, 1}
    assert {outcomes[ep] for ep in history["validation_episode_ids"]} == {0, 1}


def test_direct_rgat_artifact_is_frozen_and_is_the_potential(tmp_path):
    dataset = _semantic_dataset()
    path = tmp_path / "semantic_rollouts.npz"
    save_semantic_dataset(
        dataset, path, config_hash="cfg",
        source_behavior_policy={"name": "estimator_free", "source_pipeline": "no_se"},
        completed_seeds=[101, 102, 103, 104])
    model_path = tmp_path / "rgat_model.pt"
    prepare_semantic_rgat_artifact(
        model_path, dataset, dataset_path=path, config_hash="cfg",
        mode="quick", seed=9,
        settings={"epochs": 1, "batch_size": 2, "device": "cpu"})
    potential = FrozenSemanticRGATPotential(
        model_path, expected_config_hash="cfg")
    graph = semantic_graph(_observation(.7))
    assert potential(graph) == pytest.approx(potential.model.predict(graph))
    assert all(not parameter.requires_grad for parameter in potential.model.parameters())
    assert potential.metadata["potential"] == "Phi(G)=frozen_R_GAT(G)"


def test_pbrs_contract_matches_ppo_gamma_and_zeroes_terminal_potential():
    class Constant:
        def __call__(self, graph):
            return .4
    graph = semantic_graph(_observation())
    with pytest.raises(ValueError, match="gamma"):
        OntoRewardPBRS(Constant(), gamma=.9, ppo_gamma=.99)
    reward, parts = OntoRewardPBRS(
        Constant(), gamma=.99, ppo_gamma=.99)(graph, graph, terminal=True)
    assert parts["phi_next"] == 0.0
    assert parts["shape"] == pytest.approx(-.4)


def test_primary_paired_plan_assigns_identical_seeds():
    plan = paired_seed_plan(list(PIPELINES), {"circle": 3}, seed0=50)
    by_pipeline = {name: [row["seed"] for row in plan if row["method"] == name]
                   for name in PIPELINES}
    assert len({tuple(value) for value in by_pipeline.values()}) == 1


def test_cross_pipeline_summary_excludes_episode_return():
    records = [{"pipeline": name, "scenario": "circle", "seed": index,
                **{metric: 1.0 for metric in PHYSICAL_METRICS},
                "episode_return": 1e9 * index}
               for index, name in enumerate(PIPELINES)]
    summary = physical_summary(records)
    assert summary
    assert not any("return" in key for row in summary for key in row)


def test_multiple_training_replicates_use_hierarchical_paired_bootstrap():
    records = []
    for replicate in (0, 1):
        for seed in (10, 11):
            for pipeline in PIPELINES:
                row = {
                    "training_replicate": replicate, "pipeline": pipeline,
                    "scenario": "circle", "seed": seed,
                }
                row.update({metric: float(pipeline == "onto_no_se")
                            for metric in PHYSICAL_METRICS})
                records.append(row)
    summary = physical_summary(records)
    assert {row["training_replicates"] for row in summary} == {2}
    intervals = paired_confidence_intervals(records, draws=100, seed=7)
    assert intervals
    assert {row["training_replicates"] for row in intervals} == {2}
    assert {row["bootstrap_unit"] for row in intervals} == {
        "training_replicate_then_episode"}


def test_pipeline_hash_changes_with_information_boundary_configuration():
    config = load_experiment(
        ROOT / "config/experiments/two_pipeline_comparison.yaml")
    altered = {**config, "pipeline_contract": {
        **config["pipeline_contract"], "shin_se_onto_rgat_recovery": {
            **config["pipeline_contract"]["shin_se_onto_rgat_recovery"],
            "reward": "illegal_changed_reward"}}}
    assert configuration_hash(config) != configuration_hash(altered)
    assert tuple(config["pipelines"]) == tuple(PIPELINES)


def test_pipeline_yaml_contract_and_legacy_shin_constructor_smoke():
    config = load_experiment(
        ROOT / "config/experiments/three_pipeline_comparison.yaml")
    validate_pipeline_configuration(config)
    legacy = ShinRecurrentActorCritic(
        image_embedding=16, lstm_hidden=12, latent_dim=16,
        actor_hidden=8, critic_hidden=8)
    assert get_pipeline("shin2026").name == "shin_se"
    assert legacy.pipeline_spec.name == "shin_se"


def test_reward_design_manifest_contract_uses_loaded_configuration():
    contract = _reward_design_collection_contract(
        {"minimum_successful_recovery_episodes_full": 2},
        "full", 40, 120)
    assert contract["minimum_episodes"] == 40
    assert contract["maximum_episodes"] == 120
    assert contract["minimum_successful_recovery_episodes"] == 2
    with pytest.raises(ValueError, match="positive recovery minimum"):
        _reward_design_collection_contract(
            {"minimum_successful_recovery_episodes_full": 0},
            "full", 40, 120)


def test_deadline_teacher_descends_over_a_partially_framed_deck():
    """Directly above the deck the 60-degree camera frames only 2 of 6
    landmarks between 0.4 and 1.0 m. That is a correct approach, not a
    visual loss; the teacher must keep descending. Measured 2026-09-20: the
    former ``visible fraction < 0.5`` trigger made 40 of 40 flights bounce at
    ~1.1 m once the encoder stopped hallucinating out-of-frame landmarks."""
    partial = replace(
        _observation(.6), visible_keypoint_fraction=2.0 / 6.0,
        visual_loss_risk=0.0)
    over_deck = _privileged_velocity_teacher_action(
        [0., 0., -1., 0., 0., 0.], [0., 0., 0.], partial,
        [2., 2., 1.], noise_std=0.)
    assert over_deck[2] < 0.0
    # A single blind step (0.1 s) is not yet a loss either ...
    flicker = replace(partial, visible_keypoint_fraction=1.0 / 6.0,
                      visual_loss_risk=0.05)
    steady = _privileged_velocity_teacher_action(
        [0., 0., -2., 0., 0., 0.], [0., 0., 0.], flicker,
        [2., 2., 1.], noise_std=0.)
    assert steady[2] < 0.0
    # ... but sustained blindness while still high is.
    blind = replace(partial, visible_keypoint_fraction=0.0, visual_loss_risk=0.3)
    climb = _privileged_velocity_teacher_action(
        [0., 0., -2., 0., 0., 0.], [0., 0., 0.], blind,
        [2., 2., 1.], noise_std=0.)
    assert climb[2] > 0.0
    # The partial-visibility trigger can be restored explicitly.
    legacy = _privileged_velocity_teacher_action(
        [0., 0., -2., 0., 0., 0.], [0., 0., 0.], partial,
        [2., 2., 1.], noise_std=0., visual_loss_climb_fraction=0.5)
    assert legacy[2] > 0.0


def test_deadline_teacher_can_take_its_visual_loss_cue_from_geometry():
    """With the geometric source the encoder's visibility does not steer the
    teacher at all: a pad that is inside the frustum is never a loss, a pad
    outside it is, whatever the semantic observation says."""
    blind = replace(_observation(.6), visible_keypoint_fraction=0.0,
                    visual_loss_risk=1.0)
    in_frame = _privileged_velocity_teacher_action(
        [0., 0., -2., 0., 0., 0.], [0., 0., 0.], blind, [2., 2., 1.],
        noise_std=0., visual_loss_climb_source="geometric", geometric_in_fov=True)
    assert in_frame[2] < 0.0
    seeing = replace(_observation(.6), visible_keypoint_fraction=1.0,
                     visual_loss_risk=0.0)
    out_of_frame = _privileged_velocity_teacher_action(
        [0., 0., -2., 0., 0., 0.], [0., 0., 0.], seeing, [2., 2., 1.],
        noise_std=0., visual_loss_climb_source="geometric", geometric_in_fov=False)
    assert out_of_frame[2] > 0.0
    with pytest.raises(ValueError, match="geometric_in_fov"):
        _privileged_velocity_teacher_action(
            [0., 0., -2., 0., 0., 0.], [0., 0., 0.], seeing, [2., 2., 1.],
            noise_std=0., visual_loss_climb_source="geometric")
    info = {}
    _privileged_velocity_teacher_action(
        [0.3, 0., -1.5, 0., 0., 0.], [0., 0., 0.], seeing, [2., 2., 1.],
        noise_std=0., visual_loss_climb_source="geometric", geometric_in_fov=True,
        info=info)
    assert info["altitude_m"] == pytest.approx(1.5)
    assert info["lateral_error_m"] == pytest.approx(0.3)
    assert info["visual_lost"] is False and info["geometric_in_fov"] is True


def test_deadline_teacher_vertical_schedule_is_configurable():
    seeing = replace(_observation(.6), visible_keypoint_fraction=1.0,
                     visual_loss_risk=0.0)
    common = dict(noise_std=0., visual_loss_climb_source="geometric",
                  geometric_in_fov=True, descent_high_m_s=.5, descent_mid_m_s=.25,
                  descent_flare_m_s=.1, approach_descent_m_s=.2)
    high = _privileged_velocity_teacher_action(
        [0.1, 0., -3., 0., 0., 0.], [0., 0., 0.], seeing, [2., 2., 1.], **common)
    assert high[2] == pytest.approx(-.5)
    mid = _privileged_velocity_teacher_action(
        [0.1, 0., -.8, 0., 0., 0.], [0., 0., 0.], seeing, [2., 2., 1.], **common)
    assert mid[2] == pytest.approx(-.25)
    flare = _privileged_velocity_teacher_action(
        [0.1, 0., -.3, 0., 0., 0.], [0., 0., 0.], seeing, [2., 2., 1.], **common)
    assert flare[2] == pytest.approx(-.1)
    # Roughly aligned and high: keep coming down while closing ...
    closing = _privileged_velocity_teacher_action(
        [1.0, 0., -4., 0., 0., 0.], [0., 0., 0.], seeing, [2., 2., 1.], **common)
    assert closing[2] == pytest.approx(-.2)
    # ... but not when low, far, or fast relative to the deck.
    low = _privileged_velocity_teacher_action(
        [1.0, 0., -2., 0., 0., 0.], [0., 0., 0.], seeing, [2., 2., 1.], **common)
    assert low[2] == 0.0
    far = _privileged_velocity_teacher_action(
        [2.0, 0., -4., 0., 0., 0.], [0., 0., 0.], seeing, [2., 2., 1.], **common)
    assert far[2] == 0.0
    fast = _privileged_velocity_teacher_action(
        [1.0, 0., -4., .8, 0., 0.], [0., 0., 0.], seeing, [2., 2., 1.], **common)
    assert fast[2] == 0.0
    # Defaults are the v4 ladder.
    legacy = _privileged_velocity_teacher_action(
        [1.0, 0., -4., 0., 0., 0.], [0., 0., 0.], seeing, [2., 2., 1.], noise_std=0.)
    assert legacy[2] == 0.0


# --------------------------------------------------------------------------
# Three-arm burst comparison: two learned arms plus a non-learned control
# condition. Everything here guards a path that assumes every arm trains --
# the class of bug that silently drops the control condition rather than
# failing, which is the worst way for it to go wrong.
# --------------------------------------------------------------------------


def test_baseline_arms_are_parsed_and_their_controller_is_validated():
    from run_three_pipeline import (_baseline_arms, PRIVILEGED_VELOCITY_TEACHER,
                                    VISUAL_SERVO_TEACHER)
    assert _baseline_arms({}) == []
    arms = _baseline_arms({"baseline_arms": [
        {"method": VISUAL_SERVO_TEACHER, "label": "Visual servo (non-learned)",
         "controller": VISUAL_SERVO_TEACHER, "learned": False}]})
    assert arms == [{"method": VISUAL_SERVO_TEACHER,
                     "label": "Visual servo (non-learned)",
                     "controller": VISUAL_SERVO_TEACHER, "learned": False}]
    # The PD fallback the profile documents is accepted too.
    assert _baseline_arms({"baseline_arms": [
        {"method": "pd", "controller": PRIVILEGED_VELOCITY_TEACHER}]})[0][
            "controller"] == PRIVILEGED_VELOCITY_TEACHER
    # A typo has to stop the run, not produce an arm that never flies.
    with pytest.raises(ValueError, match="unknown controller"):
        _baseline_arms({"baseline_arms": [{"method": "x", "controller": "nope"}]})
    with pytest.raises(ValueError, match="distinct"):
        _baseline_arms({"baseline_arms": [
            {"method": "a", "controller": VISUAL_SERVO_TEACHER},
            {"method": "a", "controller": VISUAL_SERVO_TEACHER}]})


def test_the_control_condition_flies_exactly_the_seeds_the_policies_fly():
    """The comparison is paired: same deck, same seeds, every arm."""
    from run_three_pipeline import _plan_with_baseline_arms
    plan = [{"method": "shin_se_fixed", "scenario": "straight_escape_burst", "seed": s}
            for s in (5000, 5001)]
    plan += [{"method": "shin_se_onto_rgat_recovery",
              "scenario": "straight_escape_burst", "seed": s} for s in (5000, 5001)]
    widened = _plan_with_baseline_arms(plan, [
        {"method": "servo", "label": "Visual servo", "controller": "x",
         "learned": False}])
    assert len(widened) == len(plan) + 2
    def keys(method):
        return sorted((row["scenario"], row["seed"])
                      for row in widened if row["method"] == method)
    assert keys("servo") == keys("shin_se_fixed") == keys(
        "shin_se_onto_rgat_recovery")
    assert _plan_with_baseline_arms(plan, []) == plan


def test_the_arm_manifest_separates_learned_arms_from_control_conditions():
    from run_three_pipeline import _arm_manifest
    manifest = _arm_manifest(
        ["shin_se_fixed", "shin_se_onto_rgat_recovery"],
        [{"method": "servo", "label": "Visual servo (non-learned)",
          "controller": "x", "learned": False}],
        {"shin_se_fixed": "Baseline · Shin SE fixed"})
    assert [arm["method"] for arm in manifest] == [
        "shin_se_fixed", "shin_se_onto_rgat_recovery", "servo"]
    assert [arm["learned"] for arm in manifest] == [True, True, False]
    # A label the profile does not give falls back to the id rather than to a
    # hard-coded display name in the dashboard.
    assert manifest[0]["label"] == "Baseline · Shin SE fixed"
    assert manifest[1]["label"] == "shin_se_onto_rgat_recovery"


def test_a_control_condition_row_survives_checkpoint_validation():
    """The runner-side trap: rows for an arm with no checkpoint.

    The evaluation reconciliation archives any row whose method is missing from
    ``selected_checkpoints`` as ``deployment_checkpoint_changed``. A control
    condition has no checkpoint by construction, so without the exemption every
    one of its flights would be silently moved to per_episode.superseded.csv
    and the arm would vanish from the comparison with no error anywhere.
    """
    baseline_ids = {"servo"}
    selected_checkpoints = {"shin_se_fixed": {
        "sha256": "abc", "episode": 7, "kind": "best",
        "summary": {"mean_physical_score": 1.0}}}
    rows = [
        {"pipeline": "servo", "scenario": "straight_escape_burst", "seed": 5000},
        {"pipeline": "shin_se_fixed", "scenario": "straight_escape_burst",
         "seed": 5000, "selected_checkpoint_sha256": "abc",
         "physical_pair_index": 0},
        {"pipeline": "shin_se_fixed", "scenario": "straight_escape_burst",
         "seed": 5001, "selected_checkpoint_sha256": "stale",
         "physical_pair_index": 0},
    ]
    kept, superseded = [], []
    for row in rows:
        name = row.get("pipeline", row.get("method"))
        if name in baseline_ids:
            kept.append(row)
            continue
        selected = selected_checkpoints.get(name)
        digest = str(row.get("selected_checkpoint_sha256", ""))
        if selected is None or (digest and digest != selected["sha256"]):
            superseded.append(row)
            continue
        kept.append(row)
    assert [row["pipeline"] for row in kept] == ["servo", "shin_se_fixed"]
    assert [row["seed"] for row in superseded] == [5001]


def test_the_three_arm_profile_flies_one_deck_with_one_control_condition():
    from ontology_rgat.benchmarks.experiment import load_experiment
    from run_three_pipeline import _arm_manifest, _baseline_arms
    config = load_experiment(
        ROOT / "config/experiments/three_arm_burst_comparison.yaml")
    assert config["pipelines"] == [
        "shin_se_fixed", "shin_se_onto_rgat_recovery"], (
        "only the learned arms may drive training and the pair division")
    assert config["training"]["scenarios"] == ["straight_escape_burst"]
    assert config["behavior_cloning"]["scenarios"] == ["straight_escape_burst"]
    # ``evaluation`` is a mapping the loader deep-merges, so the decks this
    # profile retires have to be zeroed explicitly or they come back.
    assert {deck for deck, count in config["evaluation"].items() if count} == {
        "straight_escape_burst"}
    manifest = _arm_manifest(config["pipelines"], _baseline_arms(config),
                             dict(config.get("arm_labels") or {}))
    assert len(manifest) == 3
    assert sum(not arm["learned"] for arm in manifest) == 1


def test_a_control_condition_has_no_pipeline_spec_so_it_borrows_one():
    """Why the servo arm flies under the reference arm's method.

    The reward dispatch in recurrent_train._reward keys on the pipeline spec
    looked up from the method name. A control condition is not a pipeline, so
    that lookup finds nothing and the dispatch falls through to its last branch,
    which demands a frozen controlled R-GAT potential and raises. That is a
    crash in the middle of evaluation, after the training budget has been spent
    -- which is exactly what happened on 2026-09-22 before this was fixed.

    So the arm id is a LABEL, and the flight is collected under the reference
    pipeline's method: same encoder, same reward machinery, same metric fields,
    with only the action overridden. The row is relabelled afterwards.
    """
    from ontology_rgat.pipelines.spec import get_pipeline
    from run_three_pipeline import VISUAL_SERVO_TEACHER
    with pytest.raises(Exception):
        get_pipeline(VISUAL_SERVO_TEACHER)
    # ...whereas the reference arm the control condition borrows resolves.
    assert get_pipeline("shin_se_fixed").name == "shin_se_fixed"


def test_the_runner_collects_control_conditions_under_the_reference_method():
    """Pin the call shape, since the failure mode is a mid-evaluation crash."""
    source = (ROOT / "python/run_three_pipeline.py").read_text(encoding="utf-8")
    assert "collect_method = reference_pipeline" in source, (
        "a control condition must not be passed to collect_episode as its own "
        "method; it has no pipeline spec and the reward dispatch will raise")
    assert "reference_pipeline = args.pipelines[0]" in source
    assert "environment, model, collect_method, int(item[\"seed\"])" in source, (
        "the collect call has to use the borrowed method, not the arm id")


def test_quick_mode_shrinks_the_evaluation_budget_without_reviving_retired_decks():
    """A zeroed deck must stay zero in every mode.

    ``evaluation`` is scenario -> episode count, and the quick-mode override
    used to rebuild it from the keys, which handed every retired deck 2 flights
    back. That made the quick pass exercise a scenario set the profile does not
    declare -- in the one run whose purpose is to validate the configuration.
    """
    configured = {"straight_escape_burst": 1200, "straight_8mps": 0,
                  "circle": 0, "training_random_walk": 0}
    quick = {name: (0 if not int(count)
                    else 4 if name == "training_random_walk" else 2)
             for name, count in configured.items()}
    assert quick == {"straight_escape_burst": 2, "straight_8mps": 0,
                     "circle": 0, "training_random_walk": 0}
    assert {d for d, c in quick.items() if c} == {"straight_escape_burst"}
    # and the walk keeps its larger share when it is actually enabled
    assert {n: (0 if not int(c) else 4 if n == "training_random_walk" else 2)
            for n, c in {"training_random_walk": 10000, "circle": 200}.items()
            } == {"training_random_walk": 4, "circle": 2}


def test_the_quick_override_in_the_runner_preserves_zeros():
    source = (ROOT / "python/run_three_pipeline.py").read_text(encoding="utf-8")
    assert "for name, count in evaluation_cfg.items()" in source, (
        "the quick-mode evaluation override must read the configured counts, "
        "not just the scenario names")


def test_the_manifest_writes_unavailable_statistics_as_null_not_nan(tmp_path):
    """A mean over zero samples must not fail the whole manifest write.

    Every adaptive-R-GAT column summarises to NaN in an experiment whose arms
    do not use adaptive reward weights, which is the normal case for the
    two-arm and three-arm profiles. Before this, that NaN reached json.dumps
    with allow_nan=False and killed the manifest write at the very end of a
    completed run -- evaluation rows, reports and checkpoints all committed,
    execution_status still reading "results pending".
    """
    import json as _json
    from run_three_pipeline import _json_safe, _write_json
    payload = {
        "execution_status": "complete real Isaac/Pegasus/PX4 run",
        "reports": {
            "adaptive_rgat_parameter_count_mean": float("nan"),
            "adaptive_rgat_inference_latency_ms_mean_ci95_low": float("-inf"),
            "paper_success_mean": 0.0,
            "per_arm": [{"arm": "servo", "score": float("nan")},
                        {"arm": "shin_se_fixed", "score": 0.25}],
        },
    }
    safe = _json_safe(payload)
    assert safe["reports"]["adaptive_rgat_parameter_count_mean"] is None
    assert safe["reports"]["adaptive_rgat_inference_latency_ms_mean_ci95_low"] is None
    assert safe["reports"]["per_arm"][0]["score"] is None
    # real values, including a legitimate zero, survive untouched
    assert safe["reports"]["paper_success_mean"] == 0.0
    assert safe["reports"]["per_arm"][1]["score"] == 0.25
    assert safe["execution_status"] == "complete real Isaac/Pegasus/PX4 run"

    path = tmp_path / "manifest.json"
    _write_json(path, payload)
    written = _json.loads(path.read_text(encoding="utf-8"))
    assert written["reports"]["adaptive_rgat_parameter_count_mean"] is None
    assert written["reports"]["paper_success_mean"] == 0.0


def test_simulation_pacing_is_off_unless_a_profile_asks_for_it():
    """The world loop must free-run exactly as before by default."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "isaac_sim"))
    from config_loader import load_config
    for profile in ("config/shin2026-minimal-system.yaml",
                    "config/shin2026-system.yaml"):
        isaac = load_config(str(ROOT / profile)).get("isaac") or {}
        assert not float(isaac.get("max_sim_speed_ratio") or 0.0), (
            f"{profile} must not enable pacing implicitly")
    source = (ROOT / "isaac_sim/landing_world.py").read_text(encoding="utf-8")
    assert 'CONFIG["isaac"].get("max_sim_speed_ratio") or 0.0' in source
    assert "if speed_ratio > 0.0 and not startup_rendering:" in source, (
        "pacing must not throttle the startup burst, and must be inert at 0")


def test_the_pacing_sleep_holds_the_configured_sim_to_wall_ratio():
    """One control step should span one control period at the right ratio.

    The learner takes ~0.9 s of wall clock per control step and the control
    period is 0.104 s of simulated time, so the ratio that makes those agree is
    0.104/0.9. Below it the loop sleeps; at or above it the loop does not.
    """
    def behind(sim_elapsed, wall_elapsed, ratio):
        return sim_elapsed / ratio - wall_elapsed

    ratio = 0.104 / 0.9
    # the world has run a full control period in a tenth of the wall time it
    # should have taken: sleep for the remainder
    assert behind(0.104, 0.09, ratio) == pytest.approx(0.81, abs=1e-6)
    # exactly on pace: no sleep
    assert behind(0.104, 0.9, ratio) == pytest.approx(0.0, abs=1e-9)
    # already slower than the cap: no sleep, the cap is an upper bound only
    assert behind(0.104, 1.2, ratio) < 0.0
    # and the free-running regime we measured -- 0.7 s of sim in 0.9 s of wall
    # -- is held back by most of a step
    assert behind(0.7, 0.9, ratio) == pytest.approx(5.157, abs=1e-3)


def test_the_manifest_records_every_arm_not_just_the_learned_ones():
    """Downstream readers must not infer the arm set from ``pipelines``.

    ``pipelines`` is the learned list: it drives training and keys the
    checkpoints. A reader that treats it as the arm set drops every control
    condition, which on 2026-09-22 produced a non-learned arm missing from all
    figures and -- worse -- a finished run reported as preliminary, because the
    evaluation completeness check wanted a checkpoint for an arm that has none.
    """
    source = (ROOT / "python/run_three_pipeline.py").read_text(encoding="utf-8")
    assert '"arms": arm_manifest,' in source, (
        "manifest.json has to carry the full arm list with its learned flags")
    # and the manifest key must be built before it is written
    assert source.index("arm_manifest = _arm_manifest(") < source.index(
        '"arms": arm_manifest,')


def test_an_arm_manifest_distinguishes_controls_for_a_downstream_reader():
    from run_three_pipeline import _arm_manifest
    manifest = _arm_manifest(
        ["shin_se_fixed", "shin_se_onto_rgat_recovery"],
        [{"method": "image_based_visual_servo_v1",
          "label": "Visual servo (non-learned)", "controller": "x",
          "learned": False}],
        {})
    learned = [a["method"] for a in manifest if a["learned"]]
    controls = [a["method"] for a in manifest if not a["learned"]]
    assert learned == ["shin_se_fixed", "shin_se_onto_rgat_recovery"]
    assert controls == ["image_based_visual_servo_v1"]
    # a completeness check keyed on checkpoints must use `learned`, not the
    # whole arm list, or a control condition makes a finished run look unfinished
    assert all(a["learned"] is False for a in manifest if a["method"] in controls)


def test_renaming_an_arm_does_not_invalidate_a_machine_day_of_training():
    """A legend is not part of the identity of an experiment.

    ``config_hash`` gates checkpoint compatibility: a changed hash archives
    every checkpoint and restarts training. ``arm_labels`` only names arms on a
    chart, so including it meant fixing a typo in a label threw away the run.
    Measured on 2026-09-22 -- adding arm_labels alone changed the hash.
    """
    from ontology_rgat.benchmarks.experiment import configuration_hash
    from run_three_pipeline import _DISPLAY_ONLY_CONFIG_KEYS
    assert "arm_labels" in _DISPLAY_ONLY_CONFIG_KEYS

    def hashed(config):
        return configuration_hash({
            "experiment": {k: v for k, v in config.items()
                           if k not in _DISPLAY_ONLY_CONFIG_KEYS},
            "system": {}, "training_replicate": 0, "model_seed": 0,
            "training_seed_start": 0, "outcome_contract": "x"})

    base = {"pipelines": ["a", "b"], "training": {"episodes_full": 1000}}
    renamed = dict(base, arm_labels={"a": "Baseline", "b": "Proposed"})
    retyped = dict(base, arm_labels={"a": "Baseline (fixed typo)", "b": "Proposed"})
    assert hashed(base) == hashed(renamed) == hashed(retyped)

    # ...but anything that changes what the run DOES still moves the hash
    assert hashed(base) != hashed(dict(base, training={"episodes_full": 2000}))
    assert hashed(base) != hashed(dict(base, pipelines=["a", "c"]))


def test_bare_run_sh_runs_the_three_arm_burst_comparison():
    """The zero-argument contract names the current headline experiment.

    ``./run.sh`` with no arguments is what "run the experiment" means to anyone
    reading the repository, so it has to select the comparison the paper
    actually makes. It pointed at the six-deck two-arm design until
    2026-09-22; that design is still declared and still runnable by --config.
    """
    source = (ROOT / "python/run_three_pipeline.py").read_text(encoding="utf-8")
    assert 'ROOT / "config/experiments/three_arm_burst_comparison.yaml"' in source
    assert '"three_arm_burst"' in source, (
        "--experiment must accept the experiment its default config declares")
    launcher = (ROOT.parent / "run.sh").read_text(encoding="utf-8")
    assert "three_arm_burst_comparison.yaml" in launcher, (
        "run.sh documents the zero-argument contract and must name the same file")
    # the replaced design stays reachable rather than being deleted
    assert (ROOT / "config/experiments/two_pipeline_comparison.yaml").is_file()
    assert "two_pipeline_comparison.yaml" in launcher
