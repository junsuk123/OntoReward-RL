from __future__ import annotations

from copy import deepcopy
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "isaac_sim"))

from config_loader import load_config as load_system  # noqa: E402
from keypoint_geometry import KEYPOINT_LAYOUT_ID  # noqa: E402

from ontology_rgat.benchmarks.experiment import (load_experiment,  # noqa: E402
                                                 paired_seed_plan)
from ontology_rgat.benchmarks.live_env import LiveShinEnvironment  # noqa: E402
from ontology_rgat.benchmarks.shin2026 import ActorObservation  # noqa: E402
from ontology_rgat.controllers import VelocityYawRateController  # noqa: E402
from ontology_rgat.perception import (assert_semantic_payload_safe,  # noqa: E402
                                      prepare_keypoint_encoder)
from ontology_rgat.pipelines import (  # noqa: E402
    GEOMETRIC_FOV_CRITERION, PIPELINES, assert_no_aruco_in_primary_system,
    assert_primary_baseline_equivalence, primary_pipeline_ids,
    validate_pipeline_configuration)
from ontology_rgat.ppo.recurrent import PipelineActorCritic  # noqa: E402
from ontology_rgat.ppo.recurrent_train import _reward  # noqa: E402
from ontology_rgat.reward_modes import (ShinRewardConfig,  # noqa: E402
                                        ontology_fov_reward)
from ontology_rgat.rgat import (  # noqa: E402
    FOV_FEATURE_NAMES, FOV_GRAPH_EDGES, FOV_GRAPH_INPUT_DIM,
    FOV_GRAPH_VERSION, FOV_NODE_NAMES,
    FOV_RELATION_NAMES, FOVRiskModel, FOVSemanticObservation,
    FrozenFOVRiskPredictor, build_fov_graph, build_fov_risk_dataset,
    contract_loss, fov_margin, fov_observation_from_visual_semantics,
    future_fov_unavailability_targets, horizon_steps,
    prepare_fov_risk_artifact, save_fov_risk_dataset, split_by_episode,
    unreachable_input_nodes, validate_fov_risk_dataset)
from run_three_pipeline import _balanced_training_pair_assignment  # noqa: E402


def _graph(value=0.5):
    return build_fov_graph(
        FOVSemanticObservation(*([value] * len(FOV_FEATURE_NAMES))))


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
        "shin_se_fixed", "shin_se_onto_rgat_recovery")
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
        actor_hidden=8, critic_hidden=8, pipeline="shin_se_onto_rgat_recovery")
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
        "shin_se_onto_rgat_recovery", previous, following, estimate, next_estimate,
        _Risk(0.7), current_fov_graph=_graph(), fov_risk_lambda=0.0)
    assert proposed_disabled == baseline
    for key, value in base_parts.items():
        assert disabled_parts[key] == value
    proposed, parts, _ = _reward(
        "shin_se_onto_rgat_recovery", previous, following, estimate, next_estimate,
        _Risk(0.7), current_fov_graph=_graph(), fov_risk_lambda=0.1)
    assert proposed == pytest.approx(baseline - 0.07)
    assert parts["ontology_fov_reward"] == pytest.approx(-0.07)
    assert parts["predicted_fov_unavailability"] == pytest.approx(0.7)
    assert ontology_fov_reward(0.7, 0.1) <= 0.0


def test_strict_fov_graph_schema_is_deterministic_and_visual_only():
    observation = FOVSemanticObservation(*np.linspace(0.1, 0.8, 10))
    first = build_fov_graph(observation)
    second = build_fov_graph(observation)
    assert len(FOV_FEATURE_NAMES) == 10
    assert tuple(first.node_names) == FOV_NODE_NAMES
    assert tuple(first.relation_names) == FOV_RELATION_NAMES
    assert first.goal_node == FOV_NODE_NAMES.index("FutureFOVUnavailability")
    np.testing.assert_array_equal(first.X, second.X)
    np.testing.assert_array_equal(first.src, second.src)
    np.testing.assert_array_equal(first.dst, second.dst)
    np.testing.assert_array_equal(first.rel, second.rel)
    non_self_edges = tuple(
        (first.node_names[source], first.node_names[target],
         first.relation_names[relation])
        for source, target, relation in zip(first.src, first.dst, first.rel)
        if first.relation_names[relation] != "self")
    assert non_self_edges == FOV_GRAPH_EDGES
    assert sum(first.relation_names[value] == "self" for value in first.rel) == 15
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
        visual_loss_duration_s = 0.0
        visual_loss_risk = 0.0

        def __getattr__(self, name):
            raise AssertionError(f"forbidden field accessed: {name}")

    projected = fov_observation_from_visual_semantics(VisualBoundary())
    assert projected.feature_vector.shape == (10,)


def _dataset():
    patterns = ([1, 1, 0, 0, 1, 1], [1, 1, 1, 1, 1, 1],
                [1, 0, 1, 1, 0, 0], [1, 1, 1, 0, 1, 1])
    episodes = []
    for episode_id, pattern in enumerate(patterns, start=1):
        episodes.append({
            "episode_id": episode_id, "seed": 100 + episode_id,
            "samples": [{"graph_X": _graph(0.15 * episode_id).X,
                         "geometric_in_fov": visible} for visible in pattern],
        })
    return build_fov_risk_dataset(episodes, prediction_steps=2)


def test_future_targets_and_episode_split_have_no_temporal_leakage():
    targets, valid = future_fov_unavailability_targets([1, 1, 0, 1], 2)
    # t=0 sees {t1,t2} = visible, lost -> 0.5; t=1 sees {t2,t3} = lost,
    # visible -> 0.5; t=2 and t=3 have no full window.
    np.testing.assert_allclose(targets[:2], [0.5, 0.5])
    np.testing.assert_array_equal(valid, [True, True, False, False])
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
    unavailability = frozen.predict(_graph())
    assert 0.0 <= unavailability <= 1.0
    assert metadata["loss"] == "huber_regression_plus_contract_rule_R-04"
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
    assert assignment == {"shin_se_fixed": 0, "shin_se_onto_rgat_recovery": 1}
    assert pair_methods == methods
    for method in methods:
        standalone, selected = _balanced_training_pair_assignment([method], 1, 0)
        assert standalone == {method: 0}
        assert selected == [method]


# --------------------------------------------------------------- ArUco ban

PRIMARY_RUNTIME_SOURCES = (
    "isaac_sim/keypoint_geometry.py",
    "isaac_sim/landing_pad_visual.py",
    "python/ontology_rgat/benchmarks/live_env.py",
    "python/ontology_rgat/benchmarks/px4_adapter.py",
    "python/ontology_rgat/benchmarks/shin2026.py",
    "python/ontology_rgat/estimation/relative_state_aux.py",
    "python/ontology_rgat/perception/keypoint_encoder.py",
    "python/ontology_rgat/perception/keypoint_pretrain.py",
    "python/ontology_rgat/perception/pad_geometry.py",
    "python/ontology_rgat/perception/semantic_observation.py",
    "python/ontology_rgat/ppo/recurrent.py",
    "python/ontology_rgat/ppo/recurrent_train.py",
    "python/ontology_rgat/ppo/temporal_backbone.py",
    "python/ontology_rgat/rgat/fov_graph.py",
    "python/ontology_rgat/rgat/fov_risk_dataset.py",
)


def _executable_names(path: Path):
    """Every attribute/name/import a module actually executes.

    Parsed rather than grepped so a docstring that merely *describes* the ban
    cannot be mistaken for a violation of it, and a real call cannot hide
    inside a line that also holds a comment.
    """
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
            if isinstance(node.value, ast.Name):
                names.add(f"{node.value.id}.{node.attr}")
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(alias.name for alias in node.names)
    return names


def test_no_primary_runtime_module_can_construct_an_aruco_detector():
    """Static ban: the deployed perception path has no detector in it at all."""
    for relative in PRIMARY_RUNTIME_SOURCES:
        names = _executable_names(ROOT / relative)
        assert "cv2.aruco" not in names, relative
        assert "aruco" not in names, relative
        assert "ArucoDetector" not in names, relative
        assert "getPredefinedDictionary" not in names, relative
        assert "marker_vision" not in names, relative


def test_keypoint_supervision_runs_with_the_aruco_detector_disabled(monkeypatch):
    """Executable ban: touching cv2.aruco anywhere here fails the run."""
    import cv2

    from ontology_rgat.perception.keypoint_pretrain import (
        empirical_keypoint_dataset, synthetic_keypoint_dataset)

    class _Forbidden:
        def __getattr__(self, name):
            raise AssertionError(f"primary pipeline touched cv2.aruco.{name}")

    monkeypatch.setattr(cv2, "aruco", _Forbidden())
    system = load_system(ROOT / "config/shin2026-system.yaml")

    rendered = synthetic_keypoint_dataset(system, samples=6, seed=5)
    labelled = empirical_keypoint_dataset(
        list(zip(rendered["images"], rendered["pad_relative_pose"])), system)

    assert len(rendered["images"]) == 6
    assert len(labelled["images"]) == 6


def test_the_primary_configuration_declares_no_aruco_dependency():
    system = load_system(ROOT / "config/shin2026-system.yaml")

    assert_no_aruco_in_primary_system(system)
    assert system["vision"]["mode"] == "keypoint_fiducial"
    assert not system["vision"].get("dictionary")
    assert not system["vision"].get("board")
    assert system["parallel"]["marker_dictionary"] is None

    # A profile that still paints a tag board is refused before the stack boots.
    with_board = deepcopy(system)
    with_board["vision"]["dictionary"] = "DICT_4X4_100"
    with pytest.raises(ValueError, match="forbidden in the primary experiment"):
        assert_no_aruco_in_primary_system(with_board)
    detector_pose = deepcopy(system)
    detector_pose["vision"]["pose_source_for_policy"] = True
    with pytest.raises(ValueError, match="never drive the primary policy"):
        assert_no_aruco_in_primary_system(detector_pose)


# ------------------------------------------------- identical actor contract

def _actor_payload():
    return {"image": np.zeros((320, 512), dtype=np.uint8),
            "body_velocity": np.array([0.1, 0.0, -0.2]),
            "attitude_quaternion": np.array([1.0, 0.0, 0.0, 0.0])}


def test_both_agents_read_exactly_the_same_actor_observation_schema():
    baseline = ActorObservation.from_payload(_actor_payload())
    proposed = ActorObservation.from_payload(_actor_payload())

    assert set(vars(baseline)) == set(vars(proposed)) == {
        "image", "body_velocity", "attitude_quaternion"}
    assert baseline.proprioception.shape == proposed.proprioception.shape == (7,)
    np.testing.assert_array_equal(baseline.image, proposed.image)
    # One image contract for both: 512x320 grayscale, no extra channel.
    assert baseline.image.shape == (320, 512)


def test_simulator_geometric_fov_truth_cannot_enter_the_actor_input():
    for field in ("geometric_pad_center_in_fov", "pad_center_normalized",
                  "pad_landmarks", "keypoint_labels", "marker_quality"):
        with pytest.raises(ValueError, match="privileged actor field"):
            ActorObservation.from_payload({**_actor_payload(), field: 1.0})


def test_simulator_geometric_fov_truth_cannot_enter_the_online_rgat_graph():
    # The graph's only constructor argument is the eight-feature visual
    # observation, and none of those eight is a geometric or simulator field.
    assert tuple(inspect.signature(build_fov_graph).parameters) == ("observation",)
    for name in FOV_FEATURE_NAMES:
        assert "geometric" not in name and "pad_center" not in name
        assert "truth" not in name and "marker" not in name
    with pytest.raises(TypeError):
        FOVSemanticObservation(*([0.5] * 8), geometric_pad_center_in_fov=True)
    for privileged in ("geometric_pad_center_in_fov", "pad_center",
                       "marker_quality", "true_relative_state"):
        with pytest.raises(ValueError, match="forbidden semantic field"):
            assert_semantic_payload_safe({
                "keypoints": np.zeros((6, 2)), "heatmaps": np.zeros((6, 4, 5)),
                "proprioception": [0, 0, 0, 1, 0, 0, 0], privileged: 1.0})


def test_both_agents_share_action_space_and_controller_limits():
    config = load_experiment(ROOT / "config/experiments/two_pipeline_comparison.yaml")
    control = config["control"]

    # There is one control section, so both arms necessarily build the same
    # controller. Build it twice anyway and compare the resulting envelope.
    controllers = [VelocityYawRateController.from_mapping(control, dt=0.1)
                   for _ in primary_pipeline_ids()]
    first, second = controllers
    np.testing.assert_array_equal(first.max_velocity, second.max_velocity)
    np.testing.assert_array_equal(first.max_acceleration, second.max_acceleration)
    assert first.max_yaw_rate == second.max_yaw_rate
    assert first.max_yaw_acceleration == second.max_yaw_acceleration
    assert first.curriculum_min_action_scale == second.curriculum_min_action_scale
    assert control["action"] == "velocity_yaw_rate"
    assert first.command(np.zeros(4)).as_array().shape == (4,)
    # The environment builds that controller from the shared benchmark control
    # block; the pipeline name is not one of its inputs.
    source = inspect.getsource(LiveShinEnvironment._connect)
    assert "benchmark_control" in source
    assert "pipeline" not in source


def test_both_agents_start_from_the_same_keypoint_encoder_checkpoint(tmp_path):
    system = load_system(ROOT / "config/shin2026-system.yaml")
    experiment = {"estimator": {
        "image_embedding": 16,
        "keypoint_pretraining": {
            "enabled": True, "samples_quick": 8, "epochs_quick": 1,
            "batch_size": 4, "seed": 3},
    }}
    artifact = prepare_keypoint_encoder(
        tmp_path / "encoder.pt", config_hash="cfg", experiment=experiment,
        system=system, mode="quick", device="cpu")

    models = {}
    for name in primary_pipeline_ids():
        torch.manual_seed(11)
        model = PipelineActorCritic(
            image_embedding=16, lstm_hidden=12, latent_dim=16, actor_hidden=8,
            critic_hidden=8, pipeline=name, freeze_keypoint=True)
        model.encoder.load_state_dict(artifact["encoder"])
        models[name] = model
    baseline, proposed = (models[name] for name in primary_pipeline_ids())

    baseline_encoder = baseline.encoder.state_dict()
    proposed_encoder = proposed.encoder.state_dict()
    assert baseline_encoder.keys() == proposed_encoder.keys() == artifact["encoder"].keys()
    for key, value in artifact["encoder"].items():
        torch.testing.assert_close(baseline_encoder[key], value, rtol=0, atol=0)
        torch.testing.assert_close(proposed_encoder[key], baseline_encoder[key],
                                   rtol=0, atol=0)
    # Frozen before PPO, and frozen identically for both arms.
    assert artifact["frozen_for_ppo"] is True
    assert artifact["landmark_layout"] == KEYPOINT_LAYOUT_ID
    for model in models.values():
        assert not any(parameter.requires_grad
                       for parameter in model.encoder.parameters())


# ------------------------------------------------- geometric labels, frozen

def test_future_fov_targets_come_only_from_geometric_pad_centre_visibility():
    graph = _graph().X
    episodes = [{
        "episode_id": 1, "seed": 3,
        "samples": [{"graph_X": graph, "geometric_in_fov": visible}
                    for visible in (True, True, False, True)],
    }, {
        "episode_id": 2, "seed": 4,
        "samples": [{"graph_X": graph, "geometric_in_fov": True}] * 4,
    }]
    dataset = build_fov_risk_dataset(episodes, prediction_steps=2)
    # Episode 1 is visible at t0,t1, lost at t2, visible at t3.  The target is
    # the out-of-FOV time fraction of the next two steps, and the last two
    # samples of each episode have no full window.
    np.testing.assert_allclose(dataset["y"][:2], [0.5, 0.5])
    assert not dataset["valid"][2:4].any()
    np.testing.assert_allclose(dataset["y"][4:6], [0.0, 0.0])
    assert not dataset["valid"][6:].any()

    # A sample keyed on anything else -- detector quality above all -- is
    # refused rather than silently relabelled.
    for wrong in ("in_fov", "marker_quality", "keypoint_confidence"):
        with pytest.raises(ValueError, match="geometric_in_fov"):
            build_fov_risk_dataset([{
                "episode_id": 1, "seed": 3,
                "samples": [{"graph_X": graph, wrong: True}],
            }], prediction_steps=2)

    # 10 Hz control with a 1.0 s horizon is exactly ten control steps.
    assert horizon_steps(1.0, 10.0) == 10


def test_the_experiment_refuses_a_non_geometric_visibility_criterion():
    config = load_experiment(ROOT / "config/experiments/two_pipeline_comparison.yaml")
    assert config["fov_risk"]["visibility_criterion"] == GEOMETRIC_FOV_CRITERION

    detector_labels = deepcopy(config)
    detector_labels["fov_risk"]["visibility_criterion"] = "marker_quality"
    with pytest.raises(ValueError, match="geometric pad-centre"):
        validate_pipeline_configuration(detector_labels)


def test_the_frozen_fov_rgat_is_absent_from_the_ppo_optimizer(tmp_path):
    dataset = _dataset()
    manifest = save_fov_risk_dataset(
        dataset, tmp_path / "rollouts.npz", config_hash="cfg", seed=42,
        horizon_seconds=1.0, control_hz=2.0)
    frozen, _ = prepare_fov_risk_artifact(
        tmp_path / "model.pt", dataset, dataset_manifest=manifest,
        config_hash="cfg", seed=5,
        settings={"epochs": 1, "batch_size": 4, "hidden_dim": 8,
                  "device": "cpu", "validation_fraction": 0.25})
    model = PipelineActorCritic(
        image_embedding=16, lstm_hidden=12, latent_dim=16, actor_hidden=8,
        critic_hidden=8, pipeline="shin_se_onto_rgat_recovery")

    # Exactly the optimizer recurrent_train builds.
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    optimized = {id(parameter) for group in optimizer.param_groups
                 for parameter in group["params"]}

    frozen.assert_frozen()
    assert frozen.model.parameters()
    for parameter in frozen.model.parameters():
        assert not parameter.requires_grad
        assert id(parameter) not in optimized


def test_both_agents_receive_the_same_paired_evaluation_conditions():
    config = load_experiment(ROOT / "config/experiments/two_pipeline_comparison.yaml")
    assert config["paired_seeds"] is True
    methods = list(primary_pipeline_ids())
    plan = paired_seed_plan(methods, config["evaluation"], seed0=5000)

    by_method = {method: [] for method in methods}
    for row in plan:
        by_method[row["method"]].append((row["scenario"], row["seed"]))
    baseline, proposed = (by_method[method] for method in methods)
    assert baseline == proposed
    assert len(set(baseline)) == len(baseline)


# ------------------------------------- ontology contract acceptance (P3)

CONTRACT_PATH = ROOT / "config/ontology/fov_recovery.yaml"
CONTRACT_RULE_FIELDS = ("rule_id", "condition", "rationale",
                        "input_provenance", "exceptions", "loss_or_validation")


def _contract():
    import yaml
    return yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_the_ontology_contract_declares_every_required_rule_field():
    contract = _contract()
    assert contract["output_node"] == "FutureFOVUnavailability"
    assert contract["graph_version"] == FOV_GRAPH_VERSION
    assert [feature["name"] for feature in contract["features"]] == list(
        FOV_FEATURE_NAMES)
    assert [feature["node"] for feature in contract["features"]] == list(
        FOV_NODE_NAMES[:len(FOV_FEATURE_NAMES)])
    rule_ids = [rule["rule_id"] for rule in contract["rules"]]
    assert len(rule_ids) == len(set(rule_ids))
    for rule in contract["rules"]:
        assert tuple(sorted(rule)) == tuple(sorted(CONTRACT_RULE_FIELDS))
        for field in CONTRACT_RULE_FIELDS:
            assert str(rule[field]).strip()
    # Edge names are typed relations, not enforced constraints, and the
    # contract has to say so rather than let the name imply a guarantee.
    assert "not enforced monotonicity constraints" in contract[
        "relation_names_are"]
    # A rule is enforced only where it names a loss term or a test; the
    # contract must not present a declared prior as an enforced one.
    enforced = [rule for rule in contract["rules"]
                if rule["loss_or_validation"].startswith("enforced")]
    assert {rule["rule_id"] for rule in enforced} == {"R-01", "R-02", "R-03",
                                                      "R-04", "R-05"}
    declared = [rule for rule in contract["rules"] if rule not in enforced]
    assert all(rule["loss_or_validation"].startswith("declared only")
               for rule in declared)
    assert set(contract["not_claimed"]) >= {
        "potential-based shaping or optimal-policy invariance",
        "operational safety guarantee",
        "attention weights as causal evidence"}


def test_every_declared_fov_input_reaches_the_output_node():
    """Contract rule R-01."""
    assert unreachable_input_nodes() == ()
    graph = _graph()
    goal = graph.node_names.index("FutureFOVUnavailability")
    # Reachability is a property of the built graph too, not only the table.
    successors: dict[int, list[int]] = {index: [] for index in
                                        range(len(graph.node_names))}
    for source, target, relation in zip(graph.src, graph.dst, graph.rel):
        if graph.relation_names[relation] != "self":
            successors[int(source)].append(int(target))
    for index in range(len(FOV_FEATURE_NAMES)):
        seen, stack = set(), list(successors[index])
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(successors[node])
        assert goal in seen, f"{graph.node_names[index]} cannot reach the output"


def test_fov_readout_comes_from_the_graph_without_a_separate_head():
    """Contract rule R-02."""
    model = FOVRiskModel(hidden_dim=24, seed=3)
    assert not [module for module in model.modules()
                if isinstance(module, torch.nn.Linear)]
    assert model.layer1.units == 24 and model.layer2.units == 1
    assert model.layer2.out_dim == 1
    description = model.module_description()
    assert description["output_node"] == "FutureFOVUnavailability"
    assert description["output_activation"] == "sigmoid"
    assert not description["separate_output_mlp"]
    assert not description["separate_linear_readout"]
    assert not description["last_layer_residual"]

    X = torch.as_tensor(_graph(0.4).X.T).unsqueeze(0)
    assert model.forward_logits(X).shape == (1,)
    value = model(X)
    assert value.shape == (1,) and 0.0 <= float(value) <= 1.0
    value.backward()
    # The gradient has to flow through both relational layers; a detached
    # graph with a trained head would leave layer1 without one.
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert float(model.layer1.kernel.weight.grad.abs().sum()) > 0.0


def test_a_missing_measurement_is_not_a_comfortable_margin():
    """Contract rule R-03."""
    centred = dict(zip(FOV_FEATURE_NAMES, [0.5] * len(FOV_FEATURE_NAMES)))
    centred.update(fov_margin=1.0, measurement_age=0.0)
    observed = build_fov_graph(FOVSemanticObservation(
        **{**centred, "measurement_validity": 1.0}))
    stale = build_fov_graph(FOVSemanticObservation(
        **{**centred, "measurement_validity": 0.0}))
    boundary = FOV_NODE_NAMES.index("BoundarySafety")
    assert float(observed.X[0, boundary]) == pytest.approx(1.0)
    assert float(stale.X[0, boundary]) == pytest.approx(0.0)
    assert not np.array_equal(observed.X, stale.X)

    model = FOVRiskModel(hidden_dim=8, seed=11)
    assert model.predict(observed) != model.predict(stale)

    # The projection marks a stale frame even when the last centroid was
    # perfectly centred, so the raw margin cannot stand in for validity.
    class Blind:
        keypoint_confidence = 0.0
        visible_keypoint_fraction = 0.0
        centroid_xy = (0.0, 0.0)
        apparent_target_scale = 0.0
        image_plane_motion_safety = 0.0
        scale_rate_safety = 0.0
        visibility_memory = 0.2
        reacquisition_trend = 0.0
        visual_loss_duration_s = 0.4
        visual_loss_risk = 0.4

    projected = fov_observation_from_visual_semantics(Blind())
    assert projected.fov_margin == pytest.approx(1.0)
    assert projected.measurement_validity == 0.0
    assert projected.measurement_age == pytest.approx(0.4)


def test_the_contract_loss_penalises_a_staler_measurement_looking_safer():
    """Contract rule R-04."""
    model = FOVRiskModel(hidden_dim=8, seed=5)
    X = torch.as_tensor(np.stack([_graph(0.3).X.T, _graph(0.6).X.T]))
    penalty = contract_loss(model, X, delta=0.3)
    assert penalty.shape == () and float(penalty) >= 0.0
    penalty.backward()
    assert model.layer2.kernel.weight.grad is not None

    # A model that already answers monotonically pays nothing.
    class Monotone(FOVRiskModel):
        def forward_logits(self, X):
            if X.dim() == 2:
                X = X.unsqueeze(0)
            age = FOV_NODE_NAMES.index("MeasurementAge")
            return X[:, age, 0] * 10.0 - 5.0

    assert float(contract_loss(Monotone(hidden_dim=4), X, delta=0.3)) == 0.0

    class Inverted(FOVRiskModel):
        def forward_logits(self, X):
            if X.dim() == 2:
                X = X.unsqueeze(0)
            age = FOV_NODE_NAMES.index("MeasurementAge")
            return 5.0 - X[:, age, 0] * 10.0

    assert float(contract_loss(Inverted(hidden_dim=4), X, delta=0.3)) > 0.0


def test_unobserved_future_windows_are_masked_not_zero_filled():
    """Contract rule R-05."""
    targets, valid = future_fov_unavailability_targets([1] * 5, 3)
    assert list(valid) == [True, True, False, False, False]
    assert np.isnan(targets[2:]).all()

    dataset = _dataset()
    X, y, mask, meta = validate_fov_risk_dataset(dataset)
    assert np.isnan(y[~mask]).all()
    assert np.isfinite(y[mask]).all()
    # Every episode contributes exactly prediction_steps masked tail samples,
    # and episodes are never concatenated across the boundary.
    steps = int(dataset["prediction_steps"])
    for episode in np.unique(meta[:, 0]):
        rows = meta[:, 0] == episode
        assert int(np.count_nonzero(~mask[rows])) == steps
        assert not mask[rows][-steps:].any()

    # A zero-filled tail is refused rather than silently accepted.
    broken = {**dataset, "y": np.nan_to_num(dataset["y"], nan=0.0)}
    with pytest.raises(ValueError, match="masked"):
        validate_fov_risk_dataset(broken)

    # The supervised split never hands a masked sample to the optimiser.
    training, validation = split_by_episode(
        dataset, validation_fraction=0.25, seed=7)
    assert not (training & ~mask).any()
    assert not (validation & ~mask).any()


def test_the_retired_binary_readout_keeps_its_own_id_and_cannot_be_run():
    from ontology_rgat.pipelines.spec import (ALL_PIPELINES, LEGACY_PIPELINES,
                                              PIPELINES)
    assert set(PIPELINES) == {"shin_se_fixed", "shin_se_onto_rgat_recovery"}
    assert PIPELINES["shin_se_onto_rgat_recovery"].fov_reward_readout == (
        "direct_graph_scalar")
    retired = LEGACY_PIPELINES["shin_se_onto_rgat_fov"]
    assert retired.fov_reward_readout == "binary_classifier_linear_head"
    assert retired.name not in PIPELINES
    assert ALL_PIPELINES[retired.name] is retired
    config = load_experiment(ROOT / "config/experiments/two_pipeline_comparison.yaml")
    revived = deepcopy(config)
    revived["pipelines"] = ["shin_se_fixed", "shin_se_onto_rgat_fov"]
    with pytest.raises(ValueError, match="retired FOV reward readout"):
        validate_pipeline_configuration(revived)
