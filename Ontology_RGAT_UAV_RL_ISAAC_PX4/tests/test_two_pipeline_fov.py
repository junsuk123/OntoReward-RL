from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ontology_rgat.benchmarks.experiment import load_experiment
from ontology_rgat.pipelines import (PIPELINES, assert_primary_baseline_equivalence,
                                     primary_pipeline_ids,
                                     validate_pipeline_configuration)
from ontology_rgat.ppo.recurrent import PipelineActorCritic
from ontology_rgat.ppo.recurrent_train import _reward
from ontology_rgat.reward_modes import ShinRewardConfig, ontology_fov_reward
from ontology_rgat.rgat import (
    FOV_FEATURE_NAMES, FOV_GRAPH_INPUT_DIM, FOV_NODE_NAMES, FOV_RELATION_NAMES,
    FOVSemanticObservation, FrozenFOVRiskPredictor, build_fov_graph,
    build_fov_risk_dataset, fov_margin,
    fov_observation_from_visual_semantics, future_fov_loss_labels,
    prepare_fov_risk_artifact, save_fov_risk_dataset, split_by_episode)
from run_three_pipeline import _balanced_training_pair_assignment


ROOT = Path(__file__).resolve().parents[1]


def _graph(value=0.5):
    return build_fov_graph(FOVSemanticObservation(*([value] * 8)))


def _transition():
    previous = SimpleNamespace(
        critic=SimpleNamespace(true_relative_state=np.asarray(
            [0.5, -0.2, -1.0, 0.1, 0.0, -0.1])),
        actor=SimpleNamespace(body_velocity=np.asarray([0.0, 0.0, -0.1])),
        state={})
    following = SimpleNamespace(
        critic=SimpleNamespace(true_relative_state=np.asarray(
            [0.45, -0.18, -0.9, 0.08, 0.0, -0.08])),
        actor=SimpleNamespace(body_velocity=np.asarray([0.0, 0.0, -0.08])),
        state={}, command=np.asarray([0.1, -0.1, -0.1, 0.02]),
        physical_contact=False, strict_success=False, crash=False,
        excessive_drift=False, battery_depleted=False, terminal=False)
    return previous, following


class _Risk:
    metadata = {"prediction_horizon_steps": 10}

    def __init__(self, probability):
        self.probability = probability

    def predict(self, _graph):
        return self.probability


def test_exactly_two_primary_specs_and_unchanged_baseline_contract():
    assert primary_pipeline_ids() == (
        "shin_se_fixed", "shin_se_onto_rgat_fov")
    assert tuple(PIPELINES) == primary_pipeline_ids()
    assert_primary_baseline_equivalence()
    baseline, proposed = (PIPELINES[name] for name in primary_pipeline_ids())
    assert baseline.state_estimation_enabled == proposed.state_estimation_enabled
    assert baseline.auxiliary_estimation_loss_enabled == proposed.auxiliary_estimation_loss_enabled
    assert baseline.active_perception_enabled == proposed.active_perception_enabled
    assert baseline.reward_mode == proposed.reward_mode == "shin_table_active"
    assert not baseline.ontology_enabled
    assert proposed.ontology_enabled and proposed.fov_risk_reward_enabled
    assert not proposed.use_adaptive_reward_weights
    assert not proposed.use_direct_rgat_potential
    assert [
        ShinRewardConfig().lateral_progress_weight,
        ShinRewardConfig().vertical_progress_weight,
        ShinRewardConfig().vertical_speed_weight,
        ShinRewardConfig().undershoot_weight,
        ShinRewardConfig().yaw_rate_weight,
    ] == [1.0, 1.0, 0.5, 1.0, 2.0]


def test_actor_critic_and_estimator_are_identical_between_primary_agents():
    torch.manual_seed(9)
    baseline = PipelineActorCritic(
        image_embedding=16, lstm_hidden=12, latent_dim=16,
        actor_hidden=8, critic_hidden=8, pipeline="shin_se_fixed")
    torch.manual_seed(9)
    proposed = PipelineActorCritic(
        image_embedding=16, lstm_hidden=12, latent_dim=16,
        actor_hidden=8, critic_hidden=8, pipeline="shin_se_onto_rgat_fov")
    assert baseline.relative_state_head is not None
    assert proposed.relative_state_head is not None
    assert baseline.state_dict().keys() == proposed.state_dict().keys()
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, proposed.state_dict()[name], rtol=0, atol=0)


def test_proposed_reward_is_baseline_plus_only_nonpositive_fov_term():
    previous, following = _transition()
    estimate = np.zeros(6)
    next_estimate = np.ones(6) * 0.05
    baseline, base_parts, _ = _reward(
        "shin_se_fixed", previous, following, estimate, next_estimate, None)
    proposed_disabled, disabled_parts, _ = _reward(
        "shin_se_onto_rgat_fov", previous, following, estimate, next_estimate,
        _Risk(0.7), current_fov_graph=_graph(), fov_risk_lambda=0.0)
    assert proposed_disabled == baseline
    for key, value in base_parts.items():
        assert disabled_parts[key] == value
    proposed, parts, _ = _reward(
        "shin_se_onto_rgat_fov", previous, following, estimate, next_estimate,
        _Risk(0.7), current_fov_graph=_graph(), fov_risk_lambda=0.1)
    assert proposed == pytest.approx(baseline - 0.07)
    assert parts["ontology_fov_reward"] == pytest.approx(-0.07)
    assert parts["predicted_fov_loss_probability"] == pytest.approx(0.7)
    assert ontology_fov_reward(0.7, 0.1) <= 0.0


def test_strict_fov_graph_schema_is_deterministic_and_visual_only():
    observation = FOVSemanticObservation(*np.linspace(0.1, 0.8, 8))
    first = build_fov_graph(observation)
    second = build_fov_graph(observation)
    assert len(FOV_FEATURE_NAMES) == 8
    assert tuple(first.node_names) == FOV_NODE_NAMES
    assert tuple(first.relation_names) == FOV_RELATION_NAMES
    assert first.goal_node == FOV_NODE_NAMES.index("FOVRetention")
    np.testing.assert_array_equal(first.X, second.X)
    np.testing.assert_array_equal(first.src, second.src)
    np.testing.assert_array_equal(first.dst, second.dst)
    np.testing.assert_array_equal(first.rel, second.rel)
    non_self_edges = tuple(
        (first.node_names[source], first.node_names[target],
         first.relation_names[relation])
        for source, target, relation in zip(first.src, first.dst, first.rel)
        if first.relation_names[relation] != "self")
    assert non_self_edges == (
        ("KeypointConfidence", "PerceptionQuality", "indicates"),
        ("VisibleKeypointFraction", "PerceptionQuality", "indicates"),
        ("FOVMargin", "BoundarySafety", "indicates"),
        ("ImagePlaneMotion", "TargetMotion", "indicates"),
        ("ScaleRate", "TargetMotion", "indicates"),
        ("VisibilityMemory", "VisualObservability", "supports"),
        ("ReacquisitionTrend", "VisualObservability", "supports"),
        ("PerceptionQuality", "FOVRetention", "supports"),
        ("BoundarySafety", "FOVRetention", "supports"),
        ("VisualObservability", "FOVRetention", "supports"),
        ("TargetMotion", "FOVRetention", "constrains"),
    )
    assert sum(first.relation_names[value] == "self" for value in first.rel) == 13
    assert tuple(inspect.signature(build_fov_graph).parameters) == ("observation",)
    with pytest.raises(TypeError):
        build_fov_graph({"simulator_truth": np.zeros(6)})
    assert fov_margin((0.0, 0.0)) == 1.0
    assert fov_margin((1.0, -0.2)) == 0.0

    class VisualBoundary:
        keypoint_confidence = 0.8
        visible_keypoint_fraction = 0.7
        centroid_xy = (0.2, -0.3)
        apparent_target_scale = 0.6
        image_plane_motion_safety = 0.6
        scale_rate_safety = 0.7
        visibility_memory = 0.5
        reacquisition_trend = 0.4

        def __getattr__(self, name):
            raise AssertionError(f"forbidden field accessed: {name}")

    projected = fov_observation_from_visual_semantics(VisualBoundary())
    assert projected.feature_vector.shape == (8,)


def _dataset():
    patterns = ([1, 1, 0, 0], [1, 1, 1, 1],
                [1, 0, 1, 1], [1, 1, 1, 0])
    episodes = []
    for episode_id, pattern in enumerate(patterns, start=1):
        episodes.append({
            "episode_id": episode_id, "seed": 100 + episode_id,
            "samples": [{"graph_X": _graph(0.15 * episode_id).X,
                         "in_fov": visible} for visible in pattern],
        })
    return build_fov_risk_dataset(episodes, prediction_steps=2)


def test_future_labels_and_episode_split_have_no_temporal_leakage():
    np.testing.assert_array_equal(
        future_fov_loss_labels([1, 1, 0, 1], 2), [1, 1, 0, 0])
    dataset = _dataset()
    training, validation = split_by_episode(
        dataset, validation_fraction=0.25, seed=7)
    train_ids = set(dataset["meta"][training, 0])
    validation_ids = set(dataset["meta"][validation, 0])
    assert train_ids and validation_ids
    assert train_ids.isdisjoint(validation_ids)


def test_fov_rgat_probability_artifact_and_freeze_contract(tmp_path):
    dataset = _dataset()
    dataset_path = tmp_path / "fov_rollouts.npz"
    manifest = save_fov_risk_dataset(
        dataset, dataset_path, config_hash="cfg", seed=42,
        horizon_seconds=1.0, control_hz=2.0)
    artifact = tmp_path / "fov_model.pt"
    frozen, metadata = prepare_fov_risk_artifact(
        artifact, dataset, dataset_manifest=manifest,
        config_hash="cfg", seed=5,
        settings={"epochs": 2, "batch_size": 4, "hidden_dim": 8,
                  "device": "cpu", "validation_fraction": 0.25})
    probability = frozen.predict(_graph())
    assert 0.0 <= probability <= 1.0
    assert metadata["loss"] == "binary_cross_entropy_only"
    assert not (set(metadata["validation_metrics"]["train_episode_ids"])
                & set(metadata["validation_metrics"]["validation_episode_ids"]))
    frozen.assert_frozen()
    assert all(not parameter.requires_grad for parameter in frozen.model.parameters())
    reloaded = FrozenFOVRiskPredictor(
        artifact, expected_config_hash="cfg", device="cpu")
    reloaded.assert_frozen()


def test_primary_yaml_declares_two_pairs_and_valid_contract():
    config = load_experiment(
        ROOT / "config/experiments/two_pipeline_comparison.yaml")
    validate_pipeline_configuration(config)
    assert tuple(config["pipelines"]) == primary_pipeline_ids()
    assert config["fov_risk"]["lambda_fov"] == pytest.approx(0.1)
    assert config["fov_risk"]["prediction_horizon_seconds"] == pytest.approx(1.0)


def test_two_agents_have_independent_and_two_pair_assignments():
    methods = list(primary_pipeline_ids())
    assignment, pair_methods = _balanced_training_pair_assignment(methods, 2, 0)
    assert assignment == {"shin_se_fixed": 0, "shin_se_onto_rgat_fov": 1}
    assert pair_methods == methods
    for method in methods:
        standalone, selected = _balanced_training_pair_assignment([method], 1, 0)
        assert standalone == {method: 0}
        assert selected == [method]
