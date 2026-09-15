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
    FOV_FEATURE_NAMES, FOV_GRAPH_INPUT_DIM, FOV_NODE_NAMES, FOV_RELATION_NAMES,
    FOVSemanticObservation, FrozenFOVRiskPredictor, build_fov_graph,
    build_fov_risk_dataset, fov_margin,
    fov_observation_from_visual_semantics, future_fov_loss_labels,
    horizon_steps, prepare_fov_risk_artifact, save_fov_risk_dataset,
    split_by_episode)
from run_three_pipeline import _balanced_training_pair_assignment  # noqa: E402


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
                         "geometric_in_fov": visible} for visible in pattern],
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

def test_future_fov_labels_come_only_from_geometric_pad_centre_visibility():
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
    np.testing.assert_array_equal(dataset["y"][:4], [1.0, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(dataset["y"][4:], np.zeros(4))

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
        critic_hidden=8, pipeline="shin_se_onto_rgat_fov")

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
