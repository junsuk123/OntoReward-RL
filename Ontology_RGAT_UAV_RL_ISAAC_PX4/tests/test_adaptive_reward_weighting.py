from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from ontology_rgat.perception import SemanticObservation, semantic_graph
from ontology_rgat.reward_modes import (
    AdaptiveRewardConfig, AdaptiveWeightReward, OntoRewardPBRS,
    RewardComponentNormalizer, constrained_adaptive_weights,
    shin_reward_components)
from ontology_rgat.rgat import (
    ADAPTIVE_GRAPH_INPUT_DIM, ADAPTIVE_NODE_NAMES,
    FrozenAdaptiveRewardWeights, adaptive_model_digest, adaptive_reward_graph,
    build_adaptive_dataset, build_adaptive_reward_model,
    load_adaptive_dataset, prepare_adaptive_reward_artifact,
    save_adaptive_dataset, select_ranking_pairs, temporal_smoothness_pairs)
from ontology_rgat.benchmarks.experiment import load_experiment
from ontology_rgat.pipelines import get_pipeline, validate_pipeline_configuration
from ontology_rgat.ppo.recurrent_train import _reward
from ontology_rgat.evaluation.adaptive_reward import write_adaptive_reward_figures


ROOT = Path(__file__).resolve().parents[1]


def _observation(value: float) -> SemanticObservation:
    value = float(np.clip(value, 0.02, 0.98))
    return SemanticObservation(
        keypoint_confidence=value, visible_keypoint_fraction=value,
        image_alignment=value, apparent_target_scale=value,
        image_plane_motion_safety=value, scale_rate_safety=value,
        visibility_memory=value, reacquisition_trend=value,
        vertical_motion_safety=value, attitude_stability=value,
        battery_risk=1.0 - value, visual_loss_risk=1.0 - value,
        centroid_xy=(0.0, 0.0), raw_scale=value * 0.2)


def _records():
    records = []
    for episode in range(6):
        success = episode % 2 == 0
        scenario = ("wind" if episode % 3 else "fov") + str(episode)
        for step in range(3):
            graph = adaptive_reward_graph(_observation(.12 + .12 * episode + .02 * step))
            sign = 1.0 if success else -1.0
            records.append({
                "graph_X": graph.X, "rho_raw": np.asarray(
                    [sign * .1, sign * .08, -.1 - .02 * step,
                     0.0 if success else -.2, -.02 * (step + 1)]),
                "episode_id": episode, "time_index": step,
                "success": success,
                "failure_type": "none" if success else "collision",
                "touchdown_error": .1 + .03 * episode,
                "touchdown_vertical_speed": -.2 - .02 * episode,
                "touchdown_roll": .01 * episode,
                "touchdown_pitch": .015 * episode,
                "duration": 3.0, "terminal_reason": "success" if success else "collision",
                "scenario": scenario, "seed": 100 + episode,
                "phase": "offline_design", "disturbance_level": float(episode) / 5.0,
            })
    return records


def _dataset():
    return build_adaptive_dataset(
        _records(), seed=7, validation_fraction=.25,
        physical_scales=[1, 1, 1, 1, 1])


def test_shin_components_match_hand_calculation():
    previous = [3, 4, -3, 0, 0, 0]
    following = [0, 4, -2.5, 0, 0, 0]
    got = shin_reward_components(
        previous, following, [0, 0, 0, -.2], next_uav_vertical_velocity=-.3)
    assert got == pytest.approx([1.0, .125, -.2, 0.0, -.2])
    assert np.dot([1, 1, .5, 1, 2], got) == pytest.approx(.625)


def test_weight_transform_is_positive_conservative_and_zero_is_baseline():
    zero = constrained_adaptive_weights(np.zeros(5))
    assert zero == pytest.approx([1, 1, .5, 1, 2])
    weight = constrained_adaptive_weights(np.asarray([100, -100, 3, -2, .4]))
    assert np.all(weight > 0)
    assert weight.sum() == pytest.approx(5.5)


def test_weight_transform_keeps_torch_gradient():
    logits = torch.randn(4, 5, requires_grad=True)
    weight = constrained_adaptive_weights(logits)
    weight[:, 0].sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_adaptive_graph_adds_exact_reward_concepts_without_changing_pbrs_graph():
    adaptive = adaptive_reward_graph(_observation(.5))
    legacy = semantic_graph(_observation(.5))
    assert adaptive.X.shape == (ADAPTIVE_GRAPH_INPUT_DIM, len(ADAPTIVE_NODE_NAMES))
    for name in ("LateralProgress", "VerticalProgress", "VerticalSpeedSafety",
                 "UndershootRisk", "YawStability", "SafeLanding"):
        assert name in adaptive.node_names
    assert len(legacy.node_names) == 18


def test_dataset_split_never_splits_episode_or_scenario_group():
    dataset = _dataset()
    for episode in np.unique(dataset["episode_id"]):
        index = dataset["episode_id"] == episode
        assert np.unique(dataset["split"][index]).size == 1
        assert np.unique(dataset["scenario"][index]).size == 1
    validation = dataset["split"] == "validation"
    training = dataset["split"] == "train"
    assert set(dataset["success"][validation].astype(int)) == {0, 1}
    assert set(dataset["success"][training].astype(int)) == {0, 1}


def test_outcome_fields_are_not_part_of_graph_tensor():
    dataset = _dataset()
    assert dataset["X"].shape[-2:] == (len(ADAPTIVE_NODE_NAMES), ADAPTIVE_GRAPH_INPUT_DIM)
    assert not any(name.lower() in {node.lower() for node in ADAPTIVE_NODE_NAMES}
                   for name in ("success", "terminal_reason", "touchdown_error"))


def test_normalizer_uses_only_provided_training_rows_and_matches_both_arms():
    train = np.asarray([[.1, .2, -.3, -.4, -.5], [.2, .4, -.6, -.8, -1.]])
    normalizer = RewardComponentNormalizer.fit_training_split(train, quantile=1.0)
    sample = np.asarray([.1, .2, -.3, -.4, -.5])
    assert normalizer.transform(sample) == pytest.approx(
        normalizer.transform(sample.copy()))
    assert normalizer.scales == pytest.approx([.2, .4, .6, .8, 1.])


def test_terminal_reward_replaces_all_adaptive_shaping():
    class Provider:
        normalizer = RewardComponentNormalizer(exact_paper_raw=True)
        def __call__(self, _graph, return_latency=False):
            value = np.asarray([1, 1, .5, 1, 2.])
            return (value, 0.1) if return_latency else value

    reward = AdaptiveWeightReward(Provider(), config=AdaptiveRewardConfig())
    value, parts = reward(
        adaptive_reward_graph(_observation(.5)), np.zeros(6), np.zeros(6),
        np.zeros(4), next_uav_vertical_velocity=-.5,
        physical_contact=True, terminal=True)
    assert value == 10.0
    assert parts["adaptive_shaping"] == 0.0
    assert [parts[f"weight_{k}"] for k in range(1, 6)] == pytest.approx(
        [1, 1, .5, 1, 2])
    assert all(parts[f"weighted_rho_{k}"] == 0.0 for k in range(1, 6))


def test_no_se_adaptive_reward_has_no_active_perception_term():
    model = build_adaptive_reward_model(seed=3)
    model.eval()
    graph = adaptive_reward_graph(_observation(.5))

    class Provider:
        normalizer = RewardComponentNormalizer(exact_paper_raw=True)
        def __call__(self, graph, return_latency=False):
            with torch.no_grad():
                value = model(torch.as_tensor(graph.X.T[None])).numpy()[0]
            return (value, 0.0) if return_latency else value

    value, parts = AdaptiveWeightReward(Provider())(
        graph, np.ones(6), np.zeros(6), np.zeros(4),
        next_uav_vertical_velocity=-.5)
    assert np.isfinite(value)
    assert parts["active_perception"] == 0.0


def test_proposed_dispatch_uses_no_estimator_and_adds_rgat_semantic_pbrs():
    graph = adaptive_reward_graph(_observation(.5))

    class Provider:
        normalizer = RewardComponentNormalizer(exact_paper_raw=True)
        last_relation_attention = None
        metadata = {"training_config": {
            "semantic_potential_shaping_lambda": .75}}
        def __call__(self, _graph, return_latency=False):
            value = np.asarray([1, 1, .5, 1, 2.])
            return (value, .01) if return_latency else value
        def transition(self, _graph, _next_graph, absorbing=False):
            return np.asarray([1, 1, .5, 1, 2.]), .2, (.0 if absorbing else .6), .02

    previous = SimpleNamespace(
        critic=SimpleNamespace(true_relative_state=np.ones(6)),
        actor=SimpleNamespace(body_velocity=np.asarray([0, 0, -.5])))
    following = SimpleNamespace(
        critic=SimpleNamespace(true_relative_state=np.zeros(6)),
        actor=SimpleNamespace(body_velocity=np.asarray([0, 0, -.4])),
        command=np.zeros(4), physical_contact=False, crash=False,
        excessive_drift=False, battery_depleted=False, terminal=False)
    value, parts, estimation_loss = _reward(
        "onto_rgat_adaptive_weight_no_se", previous, following,
        None, None, Provider(), current_adaptive_graph=graph,
        next_adaptive_graph=adaptive_reward_graph(_observation(.7)))
    assert np.isfinite(value)
    assert estimation_loss is None
    assert parts["active_perception"] == 0.0
    assert parts["phi"] == pytest.approx(.2)
    assert parts["phi_next"] == pytest.approx(.6)
    assert parts["semantic_potential_shaping"] > 0.0


def test_ranking_pairs_use_success_and_only_clear_pareto_pairs():
    dataset = _dataset()
    pairs, stats = select_ranking_pairs(dataset)
    assert stats["success_over_failure"] > 0
    assert stats["total_selected"] == len(pairs)
    outcomes = {int(ep): int(dataset["success"][np.flatnonzero(
        dataset["episode_id"] == ep)[0]]) for ep in np.unique(dataset["episode_id"])}
    assert all(outcomes[good] >= outcomes[bad] for good, bad in pairs)


def test_success_pair_with_landing_time_tradeoff_is_excluded():
    dataset = {
        "episode_id": np.asarray([0, 1]),
        "split": np.asarray(["train", "train"]),
        "success": np.asarray([1, 1]),
        "touchdown_error": np.asarray([.1, .2]),
        "touchdown_vertical_speed": np.asarray([-.1, -.2]),
        "touchdown_roll": np.asarray([.01, .02]),
        "touchdown_pitch": np.asarray([.01, .02]),
        # Episode 0 lands more safely but episode 1 lands faster, so neither
        # clearly Pareto-dominates the other.
        "duration": np.asarray([10.0, 5.0]),
    }
    pairs, stats = select_ranking_pairs(dataset)
    assert pairs == []
    assert stats["excluded_unclear"] == 1


def test_temporal_smoothness_does_not_cross_episode_boundary():
    dataset = _dataset()
    pairs = temporal_smoothness_pairs(dataset)
    assert pairs
    assert all(dataset["episode_id"][left] == dataset["episode_id"][right]
               for left, right in pairs)
    assert all(dataset["time_index"][right] > dataset["time_index"][left]
               for left, right in pairs)


def test_adaptive_model_outputs_nonconstant_weights_and_attention_separately():
    model = build_adaptive_reward_model(seed=9)
    X = torch.stack([torch.as_tensor(adaptive_reward_graph(_observation(v)).X.T)
                     for v in (.2, .8)])
    weight = model(X)
    assert weight.shape == (2, 5)
    assert not torch.allclose(weight[0], weight[1])
    attention = model.attention(X)
    assert attention is not None and attention.ndim == 3
    assert attention.shape[-1] != weight.shape[-1]
    graph = adaptive_reward_graph(_observation(.55))
    trace = model.explain(graph)
    assert trace["edge_alpha_heads"].shape[0] == len(graph.src)
    assert trace["node_embeddings"].shape[0] == len(graph.node_names)
    heads = trace["model"]["output_heads"]
    assert [head["name"] for head in heads] == [
        "AdaptiveRewardWeightHead", "AdaptiveSemanticPotentialHead"]
    assert len(heads[0]["outputs"]) == 5
    assert len(heads[1]["outputs"]) == 1


def test_offline_cpu_smoke_loss_and_gradients_are_finite():
    from ontology_rgat.rgat.adaptive_train import train_adaptive_reward_weights
    model, history, metrics, _ = train_adaptive_reward_weights(
        _dataset(), settings={"epochs": 4, "learning_rate": .002,
                              "verbose": False}, seed=5, device="cpu",
        verbose=False)
    assert len(history) == 4
    assert np.isfinite([row["total_loss"] for row in history]).all()
    assert min(row["total_loss"] for row in history[1:]) < history[0]["total_loss"]
    assert metrics["parameter_count"] > 0
    assert 0.0 <= metrics["validation_ranking_accuracy"] <= 1.0
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_checkpoint_roundtrip_is_deterministic_and_frozen(tmp_path):
    dataset = _dataset()
    dataset_path = tmp_path / "data.npz"
    manifest = save_adaptive_dataset(
        dataset, dataset_path, config_hash="abc",
        source_behavior_policy={"name": "test_mixture"})
    assert manifest["outcome_strata"]["risky_failure"] == 3
    assert manifest["outcome_strata"]["collision"] == 3
    assert manifest["validation_episodes"] >= 2
    assert manifest["validation_outcome_classes"] == [0, 1]
    loaded, _ = load_adaptive_dataset(dataset_path, config_hash="abc")
    assert loaded["X"] == pytest.approx(dataset["X"])
    model_path = tmp_path / "weight.pt"
    prepare_adaptive_reward_artifact(
        model_path, dataset, dataset_manifest=manifest, config_hash="abc",
        settings={"epochs": 2, "verbose": False}, seed=11)
    frozen = FrozenAdaptiveRewardWeights(model_path, expected_config_hash="abc")
    graph = adaptive_reward_graph(_observation(.55))
    first = frozen(graph)
    second = frozen(graph)
    assert first == pytest.approx(second)
    trace = frozen.explain(graph)
    assert trace["model"]["frozen"] is True
    assert trace["model"]["design_id"] == frozen.design_id
    assert [row["value"] for row in
            trace["model"]["output_heads"][0]["outputs"]] == pytest.approx(first)
    assert all(not parameter.requires_grad for parameter in frozen.model.parameters())
    assert not frozen.model.training


def test_quality_gate_refuses_a_nearly_constant_or_unvalidated_artifact(tmp_path):
    dataset = _dataset()
    manifest = save_adaptive_dataset(
        dataset, tmp_path / "data.npz", config_hash="abc",
        source_behavior_policy={"name": "test_mixture"})
    with pytest.raises(RuntimeError, match="quality gate rejected"):
        prepare_adaptive_reward_artifact(
            tmp_path / "rejected.pt", dataset, dataset_manifest=manifest,
            config_hash="abc", settings={
                "epochs": 2, "verbose": False,
                "quality_gate": {"enabled": True,
                                 "minimum_mean_weight_cv": 1.0}})
    assert not (tmp_path / "rejected.pt").exists()


def test_frozen_rgat_is_absent_from_ppo_optimizer_and_hash_unchanged():
    reward_model = build_adaptive_reward_model(seed=13)
    from ontology_rgat.rgat import freeze_adaptive_reward_model
    before = freeze_adaptive_reward_model(reward_model)
    actor = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(actor.parameters(), lr=.01)
    optimizer.zero_grad()
    actor(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert not optimizer_ids & {id(p) for p in reward_model.parameters()}
    assert adaptive_model_digest(reward_model) == before


def test_existing_pbrs_numerics_are_unchanged():
    reward = OntoRewardPBRS(lambda graph: graph, gamma=.9, ppo_gamma=.9,
                            shaping_lambda=2.0, frozen=True)
    value, parts = reward(.25, .5)
    assert value == pytest.approx(2.0 * (.9 * .5 - .25))
    assert parts["phi"] == .25 and parts["phi_next"] == .5


def test_incomplete_artifact_is_rejected(tmp_path):
    path = tmp_path / "broken.pt"
    path.write_bytes(b"not a torch checkpoint")
    with pytest.raises(Exception):
        FrozenAdaptiveRewardWeights(path)


def test_five_explicit_modes_and_configuration_are_executable():
    config = load_experiment(
        ROOT / "config/experiments/adaptive_reward_weight_comparison.yaml")
    validate_pipeline_configuration(config)
    assert tuple(config["pipelines"]) == (
        "shin_se_fixed", "shin_se_rgat_weight", "no_se_fixed",
        "onto_rgat_adaptive_weight_no_se",
        "onto_rgat_potential_pbrs_no_se")
    proposed = get_pipeline("onto_rgat_adaptive_weight_no_se")
    assert proposed.use_adaptive_reward_weights
    assert not proposed.active_perception_enabled
    assert proposed.reward_mode == "adaptive_weight"


def test_adaptive_plot_generator_keeps_attention_separate(tmp_path):
    trace_dir = tmp_path / "models/onto_rgat_adaptive_weight_no_se"
    trace_dir.mkdir(parents=True)
    records = []
    for episode, success in ((1, 1), (2, 0)):
        for step in range(2):
            records.append({
                "episode": episode, "time_index": step,
                "method": "onto_rgat_adaptive_weight_no_se",
                "phase": "ppo", "landing_phase": "approach" if step == 0 else "descent",
                "success": success, "disturbance_level": float(episode),
                "weight_source": "frozen_rgat_adaptive",
                "adaptive_shaping": .1,
                **{f"weight_{k}": [1, 1, .5, 1, 2][k - 1]
                   for k in range(1, 6)},
                **{f"weighted_{k}": .01 * k for k in range(1, 6)},
                **{f"attention_relation_{k}": .1 * k for k in range(1, 5)},
            })
    path = trace_dir / "onto_rgat_adaptive_weight_no_se_reward_steps.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in records) + "\n",
                    encoding="utf-8")
    figures = write_adaptive_reward_figures(tmp_path)
    names = {Path(path).name for path in figures}
    assert "weights_by_episode.png" in names
    assert "relation_attention_separate_from_weights.png" in names
    assert len(names) == 7
