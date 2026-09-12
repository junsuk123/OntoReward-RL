from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
import torch

from ontology_rgat.bridge import GatewayTimeout
from ontology_rgat.benchmarks.experiment import (configuration_hash,
                                                 controlled_training_seeds,
                                                 episodes_per_method,
                                                 load_experiment,
                                                 paired_seed_plan)
from ontology_rgat.evaluation.three_pipeline import (
    PHYSICAL_METRICS, paired_confidence_intervals, physical_summary)
from ontology_rgat.perception import (SEMANTIC_GRAPH_INPUT_DIM,
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
from ontology_rgat.reward_modes import OntologyRewardContext, OntoRewardPBRS, TerminalFlags
from ontology_rgat.rgat import (FrozenSemanticRGATPotential,
                                merge_semantic_datasets,
                                prepare_semantic_rgat_artifact,
                                save_semantic_dataset,
                                semantic_episode_dataset,
                                validate_semantic_dataset)
from ontology_rgat.rgat.semantic_dataset import semantic_rgat_config
from ontology_rgat.rgat.train import train_potential


ROOT = Path(__file__).resolve().parents[1]


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
        keypoint_confidence=value, image_alignment=value,
        apparent_target_scale=value, image_plane_motion_safety=value,
        scale_rate_safety=value, vertical_motion_safety=value,
        attitude_stability=value, battery_risk=1.0 - value,
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


def test_primary_specs_encode_the_intended_information_boundaries():
    shin = PIPELINES["shin_se"]
    assert shin.state_estimation_enabled
    assert shin.auxiliary_estimation_loss_enabled
    assert shin.active_perception_enabled
    for name in ("no_se", "onto_no_se"):
        spec = PIPELINES[name]
        assert not spec.state_estimation_enabled
        assert not spec.auxiliary_estimation_loss_enabled
        assert not spec.active_perception_enabled


def test_three_pipeline_keeps_all_physical_evaluation_scenarios():
    config = load_experiment(
        ROOT / "config/experiments/three_pipeline_comparison.yaml")
    assert set(config["evaluation"]) == {
        "training_random_walk", "straight_8mps", "linear_acceleration_wave",
        "circle", "zigzag", "u_turn", "vertical_heave_boat",
    }


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


def test_rgat_split_is_by_whole_episode():
    dataset = _semantic_dataset()
    cfg = semantic_rgat_config(
        "quick", 7, {"epochs": 1, "batch_size": 2, "device": "cpu"})
    _, history = train_potential(dataset, cfg, verbose=False)
    assert history["split_unit"] == "episode"
    assert set(history["train_episode_ids"]).isdisjoint(
        history["validation_episode_ids"])


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
        ROOT / "config/experiments/three_pipeline_comparison.yaml")
    altered = {**config, "pipeline_contract": {
        **config["pipeline_contract"], "onto_no_se": {
            **config["pipeline_contract"]["onto_no_se"],
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
