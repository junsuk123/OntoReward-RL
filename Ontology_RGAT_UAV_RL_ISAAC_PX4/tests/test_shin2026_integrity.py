import math
from pathlib import Path
import sys
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "isaac_sim"))

from config_loader import load_config  # noqa: E402
from pad_motion import PadMotionConfig, PadTrajectory  # noqa: E402
from ontology_rgat_px4.protocol import ProtocolError, validate_velocity_action

from ontology_rgat.benchmarks.experiment import (episodes_per_method,
                                                 load_experiment,
                                                 paired_seed_plan)
from ontology_rgat.benchmarks.live_env import LiveShinEnvironment
from ontology_rgat.benchmarks.px4_adapter import (actor_observation_from_state,
                                                  critic_observation_from_state)
from ontology_rgat.benchmarks.randomization import (px4_gain_parameters,
                                                     sample_domain_randomization,
                                                     sample_initial_condition)
from ontology_rgat.benchmarks.shin2026 import (ActorObservation,
                                               assert_actor_payload_safe,
                                               default_shin2026_config)
from ontology_rgat.bridge import BridgeError
from ontology_rgat.controllers import VelocityYawRateController
from ontology_rgat.curriculum import (PlatformMotionCurriculum,
                                      fitted_update_interval)
from ontology_rgat.initialization import (camera_centered_hover_offset,
                                          curriculum_motion_scale,
                                          yaw_aligned_hover_offset)
from ontology_rgat.estimation import LSTMRelativeStateEstimator
from ontology_rgat.perception import (PRETRAIN_FORMAT, ShinKeypointEncoder,
                                      empirical_keypoint_dataset,
                                      prepare_keypoint_encoder,
                                      synthetic_keypoint_dataset)
from ontology_rgat.ppo.recurrent import ShinRecurrentActorCritic
from ontology_rgat.ppo.recurrent_train import (_reward, train_live,
                                               update_episode)
from ontology_rgat.reward_modes import (OntoRewardPBRS, ShinReward,
                                        ShinRewardConfig, FrozenControlledPotential,
                                        active_perception_reward,
                                        episode_rollout_dataset,
                                        load_rollout_dataset,
                                        merge_rollout_datasets,
                                        prepare_controlled_rgat_artifact,
                                        save_rollout_dataset)


def _actor_payload():
    return {"image": np.zeros((320, 512), dtype=np.uint8),
            "body_velocity": np.zeros(3),
            "attitude_quaternion": np.array([1, 0, 0, 0])}


@pytest.mark.parametrize("field", ["platform_position", "pad_velocity",
                                    "deck_gnss", "wheel_odometry", "v2v_velocity",
                                    "simulator_truth", "ground_truth"])
def test_actor_observation_contains_no_platform_truth(field):
    payload = _actor_payload()
    payload[field] = np.zeros(3)
    with pytest.raises(ValueError, match="privileged"):
        assert_actor_payload_safe(payload)


def test_actor_schema_accepts_only_three_onboard_inputs():
    obs = ActorObservation.from_payload(_actor_payload())
    assert obs.proprioception.shape == (7,)
    with pytest.raises(ValueError, match="unknown actor fields"):
        ActorObservation.from_payload({**_actor_payload(), "marker_quality": 1.0})


def test_privileged_critic_data_not_used_by_actor():
    state = {
        "quaternion_wxyz": [1, 0, 0, 0], "angular_velocity": [0, 0, 0],
        "world": {"velocity": [1, 2, 3]},
        "truth": {"valid": True, "position": [4, 5, 6], "velocity": [1, 1, 1]},
        "pad": {"velocity": [99, 99, 99]},
    }
    actor = actor_observation_from_state(state, _actor_payload()["image"])
    critic = critic_observation_from_state(actor, state)
    np.testing.assert_allclose(actor.body_velocity, [1, 2, 3])
    assert critic.privileged_vector.shape == (13,)
    assert not hasattr(actor, "true_relative_state")


def test_relative_state_estimator_output_is_six_dimensional():
    model = LSTMRelativeStateEstimator(image_embedding=8, proprioception=7,
                                       hidden_size=16, latent_size=12)
    result = model(torch.zeros(2, 3, 8), torch.zeros(2, 3, 7))
    assert result.relative_state.shape == (2, 3, 6)


def test_lstm_state_resets_at_episode_boundary():
    torch.manual_seed(3)
    model = LSTMRelativeStateEstimator(image_embedding=8, proprioception=7,
                                       hidden_size=16, latent_size=12)
    image = torch.randn(1, 2, 8)
    proprio = torch.randn(1, 2, 7)
    sequence = model(image, proprio, episode_start=torch.tensor([[True, True]]))
    isolated = model(image[:, 1:], proprio[:, 1:], episode_start=torch.tensor([True]))
    torch.testing.assert_close(sequence.relative_state[:, 1],
                               isolated.relative_state[:, 0])


def test_velocity_action_dimension_is_four_and_rate_limited():
    controller = VelocityYawRateController(dt=0.1)
    command = controller.command([1, 1, 1, 1]).as_array()
    assert command.shape == (4,)
    np.testing.assert_allclose(command[:3], [0.15, 0.15, 0.1])
    assert command[3] == pytest.approx(math.radians(9.0))
    with pytest.raises(ValueError):
        controller.command([1, 2, 3])


def test_hover_curriculum_scales_the_shared_uav_action_envelope():
    controller = VelocityYawRateController(dt=0.1)
    controller.set_curriculum(0.0)
    for _ in range(30):
        hover_command = controller.command([1, 1, 1, 1]).as_array()
    np.testing.assert_allclose(hover_command[:3], [0.7, 0.7, 0.35])
    assert hover_command[3] == pytest.approx(math.radians(21.0))

    controller.reset()
    controller.set_curriculum(1.0)
    for _ in range(30):
        final_command = controller.command([1, 1, 1, 1]).as_array()
    np.testing.assert_allclose(final_command[:3], [2.0, 2.0, 1.0])
    assert final_command[3] == pytest.approx(math.radians(60.0))


def test_all_five_ablation_arms_inherit_one_stable_control_configuration():
    config = load_experiment(ROOT / "config/experiments/shin2026_ablation.yaml")
    assert config["reward_modes"] == [
        "shin2026", "sparse", "manual_no_active", "ontoreward",
        "ontoreward_plus_active"]
    controller = VelocityYawRateController.from_mapping(config["control"])
    np.testing.assert_allclose(controller.max_velocity, [2.0, 2.0, 1.0])
    np.testing.assert_allclose(controller.max_acceleration, [1.5, 1.5, 1.0])
    assert controller.curriculum_min_action_scale == pytest.approx(0.50)
    assert math.exp(float(config["ppo"]["init_log_std"])) == pytest.approx(
        0.3011942119)
    assert float(config["ppo"]["actor_output_gain"]) == pytest.approx(0.03)
    assert int(config["ppo"]["perception_warmup_episodes_full"]) == 8


def test_one_command_run_archives_an_incompatible_policy(tmp_path):
    class TinyPolicy(torch.nn.Linear):
        @property
        def device(self):
            return self.weight.device

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    checkpoint = model_dir / "shin2026.pt"
    torch.save({
        "format": "shin2026-recurrent-v1", "method": "shin2026",
        "config_hash": "old-control-config",
    }, checkpoint)
    (model_dir / "shin2026_training.csv").write_text(
        "episode,episode_return\n1,0\n", encoding="utf-8")

    history = train_live(
        lambda: None, TinyPolicy(1, 1), "shin2026", [], model_dir,
        config_hash="new-control-config", restart_incompatible=True)

    assert history == []
    assert not checkpoint.exists()
    assert len(list(model_dir.glob("shin2026.incompatible-old-control*.pt"))) == 1
    assert len(list(model_dir.glob(
        "shin2026.incompatible-old-control*_training.csv"))) == 1


def test_deadline_budget_rescales_a_compatible_curriculum_checkpoint(
        tmp_path, capsys):
    class TinyPolicy(torch.nn.Linear):
        @property
        def device(self):
            return self.weight.device

    model = TinyPolicy(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    torch.save({
        "format": "shin2026-recurrent-v1",
        "method": "shin2026",
        "episode": 48,
        "config_hash": "same-flight-contract",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "curriculum": {
            "levels": 80, "episodes_per_update": 512,
            "initial_level": 1, "level": 1, "c": 0.0,
        },
        "reward_design_sha256": None,
    }, model_dir / "shin2026.pt")

    history = train_live(
        lambda: pytest.fail("completed checkpoint must not open an environment"),
        TinyPolicy(1, 1), "shin2026", range(48), model_dir,
        config_hash="same-flight-contract",
        ppo={"allow_curriculum_interval_migration": True},
        curriculum_config={"levels": 80, "episodes_per_update": 5})

    assert history == []
    output = capsys.readouterr().out
    assert "interval from 512 to 5" in output
    assert "level 10 (c=0.114)" in output


def test_velocity_gateway_protocol_rejects_bad_commands():
    assert validate_velocity_action({"command": [1.0, -2.0, 0.5, 0.1]}) == (
        1.0, -2.0, 0.5, 0.1)
    with pytest.raises(ProtocolError):
        validate_velocity_action({"command": [11.0, 0.0, 0.0, 0.0]})


def test_reward_mode_does_not_change_actor_observation():
    base = _actor_payload()
    observations = [ActorObservation.from_payload(base) for _ in range(5)]
    for obs in observations[1:]:
        np.testing.assert_array_equal(obs.proprioception, observations[0].proprioception)


def test_reward_mode_does_not_change_trajectory_seed():
    plan = paired_seed_plan(["shin2026", "ontoreward"], {"circle": 3}, seed0=7)
    assert [row["seed"] for row in plan if row["method"] == "shin2026"] == [7, 8, 9]
    assert [row["seed"] for row in plan if row["method"] == "ontoreward"] == [7, 8, 9]
    a = sample_initial_condition(7)
    b = sample_initial_condition(7)
    np.testing.assert_array_equal(a["relative_position_m"], b["relative_position_m"])


def test_beginner_curriculum_starts_at_stationary_airborne_hover():
    easy = sample_initial_condition(7, curriculum=0.0)
    np.testing.assert_allclose(
        easy["relative_position_m"],
        camera_centered_hover_offset(4.5), atol=1e-8)
    assert easy["platform_yaw_misalignment_rad"] == 0.0
    assert easy["platform_speed_m_s"] == 0.0
    full = sample_initial_condition(7, curriculum=1.0)
    assert -3.0 <= full["relative_position_m"][0] <= 3.0
    assert -3.0 <= full["relative_position_m"][1] <= 3.0
    assert 2.0 <= full["relative_position_m"][2] <= 8.0


def test_beginner_hover_centres_pad_on_sixty_degree_camera_axis():
    mount = np.array([0.0, 0.0, -0.16])
    offset = camera_centered_hover_offset(4.5, 60.0, mount)
    assert offset[0] == pytest.approx(-4.34 / math.tan(math.radians(60.0)))
    np.testing.assert_allclose(offset[1:], [0.0, 4.5])
    ray = np.array([math.cos(math.radians(60.0)), 0.0,
                    -math.sin(math.radians(60.0))])
    camera = offset + mount
    pad_intersection = camera + ray * camera[2] / -ray[2]
    np.testing.assert_allclose(pad_intersection, np.zeros(3), atol=1e-9)
    config = load_config(ROOT / "config" / "shin2026-system.yaml")
    assert config["isaac"]["center_hover_on_landing_camera"] is True


def test_camera_centered_hover_rotates_with_entry_yaw():
    hover = camera_centered_hover_offset(4.5)
    north_facing = yaw_aligned_hover_offset(hover, math.pi / 2.0)
    np.testing.assert_allclose(
        north_facing, [0.0, hover[0], hover[2]], atol=1e-9)


def test_keypoint_heatmaps_drive_the_descriptor_embedding():
    encoder = ShinKeypointEncoder(embedding_dim=32, keypoints=6)
    output = encoder(torch.rand(2, 1, 320, 512))
    assert output.heatmaps.shape == (2, 6, 20, 32)
    assert output.keypoints.shape == (2, 6, 2)
    assert output.embedding.shape == (2, 32)
    output.embedding.square().mean().backward()
    assert encoder.heatmap.weight.grad is not None
    assert float(encoder.heatmap.weight.grad.abs().sum()) > 0.0


def test_synthetic_pretraining_labels_six_deployed_board_keypoints():
    config = load_config(ROOT / "config" / "shin2026-system.yaml")
    dataset = synthetic_keypoint_dataset(config, samples=4, seed=9)
    assert dataset["images"].shape == (4, 320, 512)
    assert dataset["heatmaps"].shape == (4, 6, 20, 32)
    assert dataset["coordinates"].shape == (4, 6, 2)
    assert float(dataset["visible"].mean()) > 0.3
    visible_heatmaps = dataset["heatmaps"][dataset["visible"].astype(bool)]
    np.testing.assert_allclose(visible_heatmaps.sum(axis=(1, 2)), 1.0, atol=1e-5)


def test_keypoint_pretraining_artifact_is_reused_and_frozen(tmp_path):
    system = load_config(ROOT / "config" / "shin2026-system.yaml")
    experiment = {"estimator": {
        "image_embedding": 32,
        "keypoint_pretraining": {
            "enabled": True, "samples_quick": 4, "epochs_quick": 1,
            "batch_size": 2, "learning_rate": 1e-3, "seed": 17,
        },
    }}
    path = tmp_path / "keypoints.pt"
    trained = prepare_keypoint_encoder(
        path, config_hash="cfg", experiment=experiment, system=system,
        mode="quick", device="cpu")
    reused = prepare_keypoint_encoder(
        path, config_hash="cfg", experiment=experiment, system=system,
        mode="quick", device="cpu")
    assert trained["format"] == PRETRAIN_FORMAT
    assert reused["metrics"] == trained["metrics"]
    model = ShinRecurrentActorCritic(image_embedding=32, freeze_keypoint=True)
    model.encoder.load_state_dict(reused["encoder"])
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())


def test_beginner_curriculum_keeps_the_ugv_moving_at_a_safe_fraction():
    assert curriculum_motion_scale(0.0, 0.35) == pytest.approx(0.35)
    assert curriculum_motion_scale(0.5, 0.35) == pytest.approx(0.675)
    assert curriculum_motion_scale(1.0, 0.35) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        curriculum_motion_scale(0.0, 1.1)


def test_empirical_keypoint_labelling_recovers_deployed_board_homography():
    system = load_config(ROOT / "config/shin2026-system.yaml")
    rendered = synthetic_keypoint_dataset(system, samples=16, seed=91)
    labelled = empirical_keypoint_dataset(rendered["images"], system)
    assert labelled["images"].shape[1:] == (320, 512)
    assert len(labelled["images"]) >= 12
    assert labelled["coordinates"].shape[1:] == (6, 2)
    assert labelled["heatmaps"].shape[1:] == (6, 20, 32)
    assert float(labelled["visible"].mean()) > 0.5


def test_deadline_budget_is_exact_and_preserves_all_curriculum_levels():
    assert episodes_per_method(800, 2) == 400
    assert episodes_per_method(800, 5) == 160
    assert episodes_per_method(800, 1) == 800
    with pytest.raises(ValueError, match="not divisible"):
        episodes_per_method(801, 2)
    assert fitted_update_interval(400, 80) == 5
    assert fitted_update_interval(800, 80) == 10
    curriculum = PlatformMotionCurriculum(levels=80, episodes_per_update=5)
    assert curriculum.update(0) == pytest.approx(0.0)
    assert curriculum.update(399) == pytest.approx(1.0)


def test_performance_curriculum_holds_failures_and_advances_only_after_competence():
    curriculum = PlatformMotionCurriculum(
        levels=80, performance_gated=True, assessment_window=4,
        minimum_episodes_at_level=4, success_rate_threshold=.5,
        max_position_rmse_m=2.0, max_fov_loss_fraction=.5)
    failing = {"paper_success": 0, "position_rmse": 4.0,
               "fov_loss_fraction": .9}
    for _ in range(40):
        assert not curriculum.observe(failing)
    assert curriculum.level == 1 and curriculum.c == 0.0
    passing = {"paper_success": 1, "position_rmse": 1.0,
               "fov_loss_fraction": .1}
    for _ in range(2):
        assert not curriculum.observe(passing)
    assert curriculum.observe(passing)
    assert curriculum.level == 2


def test_airborne_terminal_is_staged_in_hover_instead_of_auto_land():
    """A tilt failure can be terminal while the airframe is still flying."""
    calls = []
    bridge = SimpleNamespace(
        last_state={"landed": False, "extra": {"pad_contact": False}},
        hold_for_next_airborne_reset=lambda: calls.append("hold"),
        stop_after_outcome=lambda: calls.append("stop"),
        land_and_wait=lambda: calls.append("land"),
    )
    env = LiveShinEnvironment.__new__(LiveShinEnvironment)
    env.bridge = bridge
    env.cfg = SimpleNamespace(external={"start_airborne": True})
    env.last_step = SimpleNamespace(crash=True)

    env.finish_episode()

    assert calls == ["hold"]


def test_ground_contact_is_stopped_before_the_next_airborne_episode():
    calls = []
    bridge = SimpleNamespace(
        last_state={"landed": True, "extra": {"pad_contact": False}},
        hold_for_next_airborne_reset=lambda: calls.append("hold"),
        stop_after_outcome=lambda: calls.append("stop"),
        land_and_wait=lambda: calls.append("land"),
    )
    env = LiveShinEnvironment.__new__(LiveShinEnvironment)
    env.bridge = bridge
    env.cfg = SimpleNamespace(external={"start_airborne": True})
    env.last_step = SimpleNamespace(crash=True)

    env.finish_episode()

    assert calls == ["stop"]


def test_live_benchmark_restarts_owned_stack_after_reset_failure(monkeypatch):
    from ontology_rgat import stack as stack_module

    calls = []

    def failed_reset(*_args, **_kwargs):
        raise BridgeError("PX4 left OFFBOARD")

    scales = []
    reset_kwargs = []
    recovered_adapter = SimpleNamespace(
        controller=SimpleNamespace(set_curriculum=lambda value: scales.append(value)),
        reset=lambda *_args, **kwargs: reset_kwargs.append(kwargs)
        or ("actor", {"state": "ready"}))
    env = LiveShinEnvironment.__new__(LiveShinEnvironment)
    env.cfg = SimpleNamespace(external=SimpleNamespace(reset_recoveries=1))
    env.control = {"curriculum_min_pad_motion_scale": 0.35}
    env.bridge = SimpleNamespace(
        cfg=SimpleNamespace(pad_scale=0.0),
        close=lambda: calls.append("close"))
    env.adapter = SimpleNamespace(
        controller=SimpleNamespace(set_curriculum=lambda value: scales.append(value)),
        reset=failed_reset)
    env._connect = lambda: setattr(env, "adapter", recovered_adapter)
    env._classify = lambda actor, state, command: (actor, state, command.tolist())
    env.last_step = None
    owned = SimpleNamespace(restart=lambda: calls.append("restart"))
    monkeypatch.setattr(stack_module, "current", lambda: owned)

    result = env.reset(7, curriculum=0.0, scenario="circle")

    assert calls == ["close", "restart"]
    assert scales == [0.0, 0.0]
    assert env.bridge.cfg.pad_scale == pytest.approx(0.35)
    assert reset_kwargs == [{"scenario": "circle", "initial_condition_scale": 0.0}]
    assert result[:2] == ("actor", {"state": "ready"})


def test_pbrs_gamma_equals_ppo_gamma_and_terminal_potential_is_zero():
    with pytest.raises(ValueError, match="gamma"):
        OntoRewardPBRS(lambda value: value, gamma=0.9, ppo_gamma=0.99)
    reward = OntoRewardPBRS(lambda value: float(value), gamma=0.99,
                            ppo_gamma=0.99, shaping_lambda=2.0)
    _, parts = reward(0.25, 100.0, terminal=True)
    assert parts["phi_next"] == 0.0
    assert parts["shape"] == pytest.approx(-0.5)


def test_frozen_rgat_reward_design_does_not_update_during_ppo():
    with pytest.raises(ValueError, match="frozen"):
        OntoRewardPBRS(lambda _: 0.0, gamma=0.99, ppo_gamma=0.99, frozen=False)


def test_controlled_potential_rejects_legacy_or_unfrozen_artifact(tmp_path):
    artifact = tmp_path / "reward.json"
    artifact.write_text('{"profile":"urban","frozen":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="controlled_landing"):
        FrozenControlledPotential(artifact)


def test_controlled_potential_accepts_only_frozen_rgat_profile(tmp_path):
    artifact = tmp_path / "reward.json"
    artifact.write_text(json.dumps({
        "format": "ontology_rgat.controlled_reward/3",
        "profile": "controlled_landing", "frozen": True,
        "provenance": "rgat_distillation", "design_id": "test",
        "dataset_provenance": "isaac_px4_shin2026_rollouts",
        "dataset_config_hash": "cfg",
        "weights": {"lateral_error": .2, "altitude_error": .2,
                    "relative_horizontal_speed": .2,
                    "relative_vertical_speed": .2, "battery_risk": .2},
    }), encoding="utf-8")
    potential = FrozenControlledPotential(artifact, expected_config_hash="cfg")
    assert potential({"estimated_relative_state": np.zeros(6),
                      "battery_reserve": 1.0}) == 0.0
    assert potential({"estimated_relative_state": np.zeros(6),
                      "battery_reserve": 0.0}) == pytest.approx(-.2)


def test_controlled_potential_rejects_synthetic_bootstrap_artifact(tmp_path):
    artifact = tmp_path / "reward.json"
    artifact.write_text(json.dumps({
        "format": "ontology_rgat.controlled_reward/3",
        "profile": "controlled_landing", "frozen": True,
        "provenance": "rgat_distillation",
        "dataset_provenance": "synthetic_table_i_semantic_bootstrap",
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="actual Isaac/PX4"):
        FrozenControlledPotential(artifact)


def test_rgat_rollout_graph_uses_estimate_and_physical_outcome_not_truth():
    rows = [
        {"estimate": np.array([2.125, 0, -4, 4, 0, -1.5]),
         "truth": np.full(6, 999.0), "battery_reserve": .25},
        {"estimate": np.zeros(6), "truth": np.full(6, -999.0),
         "battery_reserve": 1.0},
    ]
    success = episode_rollout_dataset(
        rows, {"paper_success": 1.0}, episode=3, seed=12,
        gamma=.5, sample_stride=1)
    # First feature channel is four visual costs plus real battery risk.
    np.testing.assert_allclose(success["X"][0, :5, 0], [.5, .5, .5, .5, .75])
    np.testing.assert_allclose(success["y"], [.5, 1.0])
    failure = episode_rollout_dataset(
        rows, {"paper_success": 0.0}, episode=4, seed=13,
        gamma=.5, sample_stride=1)
    np.testing.assert_allclose(failure["y"], [-.5, -1.0])


def test_empirical_rgat_dataset_round_trip_checks_provenance(tmp_path):
    rows = [{"estimate": np.zeros(6), "battery_reserve": .8},
            {"estimate": np.ones(6), "battery_reserve": .6}]
    dataset = episode_rollout_dataset(
        rows, {"paper_success": 1.0}, episode=1, seed=70)
    path = tmp_path / "rollouts.npz"
    manifest = save_rollout_dataset(
        dataset, path, config_hash="cfg", source_checkpoint_sha256="policy",
        completed_seeds=[70])
    loaded, restored = load_rollout_dataset(
        path, config_hash="cfg", source_checkpoint_sha256="policy")
    np.testing.assert_array_equal(loaded["X"], dataset["X"])
    assert restored["dataset_sha256"] == manifest["dataset_sha256"]
    with pytest.raises(ValueError, match="configuration mismatch"):
        load_rollout_dataset(
            path, config_hash="other", source_checkpoint_sha256="policy")


def test_controlled_rgat_preparation_writes_loadable_frozen_artifact(tmp_path):
    rows = [{"estimate": np.asarray([x, 0, -2, .1, 0, -.1]),
             "battery_reserve": float(.2 + .6 * x / 2.0)}
            for x in np.linspace(.1, 2.0, 16)]
    success = episode_rollout_dataset(
        rows, {"paper_success": 1.0}, episode=1, seed=10, sample_stride=2)
    failure = episode_rollout_dataset(
        rows[::-1], {"paper_success": 0.0}, episode=2, seed=11, sample_stride=2)
    dataset = merge_rollout_datasets(success, failure)
    dataset_path = tmp_path / "rollouts.npz"
    save_rollout_dataset(
        dataset, dataset_path, config_hash="cfg",
        source_checkpoint_sha256="checkpoint", completed_seeds=[10, 11])
    path, metadata = prepare_controlled_rgat_artifact(
        tmp_path / "controlled.json", dataset, config_hash="cfg",
        source_checkpoint_sha256="checkpoint", dataset_path=dataset_path,
        mode="quick", epochs=1)
    potential = FrozenControlledPotential(path, expected_config_hash="cfg")
    assert metadata["provenance"] == "rgat_distillation"
    assert metadata["dataset_provenance"] == "isaac_px4_shin2026_rollouts"
    assert metadata["validation_split"] == "held-out rollout episodes"
    assert potential.design_id == metadata["design_id"]


def test_shin_active_reward_uses_training_only_estimation_target():
    cfg = ShinRewardConfig()
    assert active_perception_reward(0.0, cfg) == 0.0
    assert active_perception_reward(0.51, cfg) == pytest.approx(-0.05)
    assert active_perception_reward(2.0, cfg) == pytest.approx(-0.1)
    reward = ShinReward(cfg)
    with pytest.raises(ValueError, match="training-only"):
        reward(np.zeros(6), np.zeros(6), np.zeros(4), drone_vertical_velocity=0.0)


def test_shin_dense_reward_uses_truth_while_active_term_uses_estimate():
    previous = SimpleNamespace(
        critic=SimpleNamespace(true_relative_state=np.array([2, 0, -3, 0, 0, 0])),
        actor=SimpleNamespace(body_velocity=np.array([0, 0, -0.5])))
    following = SimpleNamespace(
        critic=SimpleNamespace(true_relative_state=np.array([0.5, 0, -2, 0, 0, 0])),
        command=np.zeros(4), physical_contact=False, crash=False,
        excessive_drift=False, battery_depleted=False, terminal=False)
    # Deliberately contradictory estimates would report motion away from the
    # pad if they accidentally entered the Table-III progress terms.
    estimate = np.array([100, 0, -100, 0, 0, 0], dtype=float)
    next_estimate = np.array([200, 0, -200, 0, 0, 0], dtype=float)
    total, parts, estimation_loss = _reward(
        "shin2026", previous, following, estimate, next_estimate, None)
    assert parts["lateral_progress"] == 1.0
    assert parts["vertical_progress"] == 1.0
    assert parts["active_perception"] == -0.1
    assert total == pytest.approx(1.9)
    assert estimation_loss > 1.0


def test_table_iii_reward_equations():
    reward = ShinReward(ShinRewardConfig(active_enabled=False))
    total, parts = reward([2, 0, -3, 0, 0, 0], [0.5, 0, -2, 0, 0, 0],
                          [0, 0, 0, 0.2], drone_vertical_velocity=-0.5)
    assert parts["lateral_progress"] == 1.0
    assert parts["vertical_progress"] == 1.0
    assert parts["vertical_speed_penalty"] == 0.0
    assert parts["undershoot_penalty"] == 0.0
    assert parts["yaw_rate_penalty"] == pytest.approx(-0.4)
    assert total == pytest.approx(1.6)


def test_table_iii_terminal_reward_replaces_dense_terms():
    reward = ShinReward(ShinRewardConfig(active_enabled=True))
    total, parts = reward(
        [2, 0, -3, 0, 0, 0], [0.5, 0, -2, 0, 0, 0],
        [0, 0, 0, 0.2], drone_vertical_velocity=1.0,
        next_estimation_loss=1.0, physical_contact=True, terminal=True)
    assert total == 10.0
    assert parts["task"] == 10.0
    assert all(value == 0.0 for name, value in parts.items() if name != "task")


def test_table_ii_domain_randomization_ranges():
    value = sample_domain_randomization(12)
    assert 2.7 <= value.velocity_gain_xy <= 3.3
    assert np.max(np.abs(value.external_force_n)) <= 0.75
    assert np.max(np.abs(value.external_torque_nm)) <= 4e-3
    assert 1 <= value.ground_texture_id <= 50
    assert json.loads(json.dumps(value.to_dict()))["ground_texture_id"] \
        == value.ground_texture_id
    gains = px4_gain_parameters(value)
    assert 1.62 <= gains["MPC_XY_VEL_P_ACC"] <= 1.98
    assert 3.46 <= gains["MPC_Z_VEL_P_ACC"] <= 4.54
    assert gains["MC_ROLL_P"] == pytest.approx(gains["MC_PITCH_P"])


def test_shin_profile_uses_table_ii_instead_of_uncontrolled_urban_wind():
    config = load_config(ROOT / "config/shin2026-system.yaml")
    assert config["domain_randomization"]["enabled"] is True
    assert config["wind"]["enabled"] is False


def test_table_i_platform_random_walk_is_seeded_and_bounded():
    cfg = PadMotionConfig.from_mapping({"pad": {
        "motion": "random_walk", "speed_range_m_s": [0.0, 8.0],
        "speed_step_perturbation_m_s": [-0.5, 0.5],
        "yaw_rate_step_perturbation_deg_s": [-3.0, 3.0],
        "motion_update_dt_s": 0.1, "arena_radius_m": 300.0,
    }})
    first, second = PadTrajectory(cfg), PadTrajectory(cfg)
    first.reset(91, 0.0)
    second.reset(91, 0.0)
    np.testing.assert_allclose(first.random_walk_velocity,
                               second.random_walk_velocity)
    speed = np.linalg.norm(first.random_walk_velocity[:, :2], axis=1)
    assert np.all(speed >= 0.0) and np.all(speed <= 8.0)
    assert np.max(np.abs(np.diff(speed))) <= 0.5 + 1e-10


@pytest.mark.parametrize("scenario", ["straight_8mps", "linear_acceleration_wave",
                                       "circle", "zigzag", "u_turn",
                                       "vertical_heave_boat"])
def test_named_benchmark_motion_scenarios_are_deterministic(scenario):
    cfg = PadMotionConfig.from_mapping({"pad": {
        "motion": "random_walk", "speed_range_m_s": [0.0, 8.0],
        "arena_radius_m": 300.0,
    }})
    first, second = PadTrajectory(cfg), PadTrajectory(cfg)
    first.reset(12, 0.0, scenario=scenario)
    second.reset(12, 0.0, scenario=scenario)
    np.testing.assert_allclose(first.pose(2.0)[0], second.pose(2.0)[0])
    if scenario == "straight_8mps":
        assert np.linalg.norm(first.pose(2.0)[1]) == pytest.approx(8.0)
    if scenario == "vertical_heave_boat":
        assert first.pose(2.0)[0][2] != pytest.approx(first.pose(0.0)[0][2])


def test_recurrent_ppo_update_supports_truncated_sequences():
    torch.manual_seed(7)
    model = ShinRecurrentActorCritic(image_embedding=8, lstm_hidden=16,
                                     latent_dim=12, actor_hidden=8,
                                     critic_hidden=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    rows = []
    for index in range(3):
        rows.append({
            "image": np.zeros((32, 32), dtype=np.uint8),
            "proprioception": np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
            "truth": np.zeros(6, dtype=np.float32),
            "pre_squash": np.zeros(4, dtype=np.float32),
            "action": np.zeros(4, dtype=np.float32),
            "log_prob": -3.676, "value": 0.0, "reward": float(index == 2),
            "done": float(index == 2),
            "hidden_h": np.zeros((1, 1, 16), dtype=np.float32),
            "hidden_c": np.zeros((1, 1, 16), dtype=np.float32),
        })
    metrics = update_episode(model, optimizer, rows, epochs=1, sequence_length=2)
    assert all(np.isfinite(value) for value in metrics.values())


def test_benchmark_dimensions_and_modes():
    for mode in ("shin2026", "sparse", "manual_no_active", "ontoreward",
                 "ontoreward_plus_active"):
        cfg = default_shin2026_config(reward_mode=mode)
        assert cfg.action_dimension == 4 and cfg.relative_state_dimension == 6
