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
from ontology_rgat.bridge import BridgeError, PX4Bridge
from ontology_rgat.config import default_config
from ontology_rgat.mathx import euler_to_quat
from ontology_rgat.controllers import VelocityYawRateController
from ontology_rgat.curriculum import (PlatformMotionCurriculum,
                                      fitted_update_interval)
from ontology_rgat.initialization import (R_BODY_FROM_OPTICAL_NADIR,
                                          body_from_optical,
                                          camera_centered_hover_offset,
                                          constrain_camera_visible_entry,
                                          curriculum_motion_scale,
                                          pad_in_camera_view,
                                          pad_view_margin,
                                          yaw_aligned_hover_offset)
from ontology_rgat.estimation import LSTMRelativeStateEstimator
from ontology_rgat.perception import (EMPIRICAL_CALIBRATION_FORMAT,
                                      PRETRAIN_FORMAT, ShinKeypointEncoder,
                                      calibrate_keypoint_encoder,
                                      calibration_viewpoints,
                                      empirical_keypoint_dataset,
                                      needs_empirical_calibration,
                                      prepare_keypoint_encoder,
                                      synthetic_keypoint_dataset)
from ontology_rgat.ppo.recurrent import ShinRecurrentActorCritic
from ontology_rgat.ppo.recurrent_train import (_reward, _terminal_flags, train_live,
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
        ActorObservation.from_payload({**_actor_payload(), "altimeter": 1.0})
    # Detector quality and simulator pad geometry are refused by name, before
    # the unknown-field check, so neither can arrive under a plausible alias.
    for privileged in ("marker_quality", "geometric_pad_center_in_fov",
                       "pad_center_normalized", "keypoint_labels"):
        with pytest.raises(ValueError, match="privileged actor field"):
            ActorObservation.from_payload({**_actor_payload(), privileged: 1.0})


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


def test_new_training_contract_archives_an_old_completed_policy(tmp_path):
    class TinyPolicy(torch.nn.Linear):
        @property
        def device(self):
            return self.weight.device

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    checkpoint = model_dir / "shin2026.pt"
    torch.save({
        "format": "three-pipeline-recurrent-v3-scaled-estimator",
        "method": "shin2026", "episode": 0,
        "config_hash": "same-flight-contract",
        "training_contract_id": None,
    }, checkpoint)

    history = train_live(
        lambda: pytest.fail("zero-episode replacement must not open an environment"),
        TinyPolicy(1, 1), "shin2026", [], model_dir,
        config_hash="same-flight-contract", restart_incompatible=True,
        training_contract_id="robust_ppo_anchor_lr_recovery_v1")

    assert history == []
    assert not checkpoint.exists()
    assert len(list(model_dir.glob(
        "shin2026.incompatible-same-flight*.pt"))) == 1


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


def test_full_curriculum_entry_is_conditioned_on_camera_visibility():
    raw = np.array([3.0, 3.0, 8.0])
    yaw = math.radians(60.0)
    constrained = constrain_camera_visible_entry(raw, yaw)
    centre = yaw_aligned_hover_offset(camera_centered_hover_offset(8.0), yaw)
    assert constrained[2] == raw[2]
    assert np.linalg.norm(constrained[:2] - centre[:2]) < np.linalg.norm(
        raw[:2] - centre[:2])
    np.testing.assert_allclose(
        constrain_camera_visible_entry(centre, yaw), centre, atol=1e-12)


def test_camera_visible_entry_erodes_footprint_by_marker_board_radius():
    raw = np.array([3.0, -2.0, 2.0])
    yaw = math.radians(35.0)
    point_only = constrain_camera_visible_entry(
        raw, yaw, target_radius_m=0.0)
    full_board = constrain_camera_visible_entry(
        raw, yaw, target_radius_m=0.75)
    centre = yaw_aligned_hover_offset(
        camera_centered_hover_offset(2.0), yaw)

    assert np.linalg.norm(full_board[:2] - centre[:2]) < np.linalg.norm(
        point_only[:2] - centre[:2])


def test_entry_view_geometry_uses_the_rendered_camera_frame():
    import marker_vision

    np.testing.assert_array_equal(
        R_BODY_FROM_OPTICAL_NADIR, marker_vision.R_BODY_FROM_OPTICAL)
    view_axis = body_from_optical(60.0)[:, 2]
    np.testing.assert_allclose(
        view_axis, [math.cos(math.radians(60.0)), 0.0,
                    -math.sin(math.radians(60.0))], atol=1e-12)


def test_camera_centred_hover_puts_the_pad_on_the_optical_axis_at_any_altitude():
    identity = [1.0, 0.0, 0.0, 0.0]
    for altitude in (2.0, 4.5, 7.55, 8.0):
        assert pad_view_margin(
            camera_centered_hover_offset(altitude), identity) < 1e-9
    yaw = math.radians(120.0)
    yawed = yaw_aligned_hover_offset(camera_centered_hover_offset(4.5), yaw)
    assert pad_view_margin(yawed, euler_to_quat([0.0, 0.0, yaw])) < 1e-9


def test_entry_view_margin_grows_towards_the_frame_edge_and_beyond():
    identity = [1.0, 0.0, 0.0, 0.0]
    centre = camera_centered_hover_offset(4.5)
    # The 64-deg narrow axis spans left/right: at the 4.5 m hover the slant
    # range is about 5.0 m, so the frame edge is about 3.1 m to the side.
    inside = pad_view_margin(centre + np.array([0.0, 1.0, 0.0]), identity)
    edge = pad_view_margin(centre + np.array([0.0, 3.5, 0.0]), identity)
    assert 0.0 < inside < 1.0 < edge
    assert pad_view_margin([3.0, 0.0, 4.5], identity) > 1.0
    # Ahead of and just above the deck, the forward/down camera has the
    # pad behind its image plane.
    assert math.isinf(pad_view_margin([5.0, 0.0, 1.0], identity))
    assert pad_view_margin([0.0, 0.0, 4.5], identity, pitch_down_deg=90.0) == 0.0
    assert pad_in_camera_view(centre, identity)
    assert not pad_in_camera_view(
        centre + np.array([0.0, 3.0, 0.0]), identity, margin_fraction=0.85)


def test_live_config_copies_the_rendered_camera_into_the_geometric_gate(tmp_path):
    from run_shin2026_pipeline import _live_config

    cfg = _live_config("quick", tmp_path, ROOT / "config" / "shin2026-system.yaml")
    camera = load_config(ROOT / "config" / "shin2026-system.yaml")["vision"]["camera"]
    assert list(cfg.external.landing_camera["resolution"]) == list(camera["resolution"])
    assert cfg.external.landing_camera["pitch_down_deg"] == camera["pitch_down_deg"]
    assert cfg.external.landing_camera["horizontal_fov_deg"] == camera["horizontal_fov_deg"]
    assert 0.0 < cfg.external.entry_view_margin <= 1.0
    # The detector-memory gate is gone: there is one geometric definition.
    assert not hasattr(cfg.external, "entry_marker_memory")
    assert not hasattr(cfg.external, "entry_view_geometry")


def test_keypoint_heatmaps_drive_the_descriptor_embedding():
    encoder = ShinKeypointEncoder(embedding_dim=32, keypoints=6)
    output = encoder(torch.rand(2, 1, 320, 512))
    assert output.heatmaps.shape == (2, 6, *ShinKeypointEncoder.heatmap_shape(320, 512))
    assert output.keypoints.shape == (2, 6, 2)
    assert output.visibility.shape == (2, 6)
    assert torch.all((output.visibility >= 0.0) & (output.visibility <= 1.0))
    assert output.embedding.shape == (2, 32)
    output.embedding.square().mean().backward()
    assert encoder.heatmap.weight.grad is not None
    assert float(encoder.heatmap.weight.grad.abs().sum()) > 0.0


def test_synthetic_pretraining_labels_six_deployed_board_keypoints():
    config = load_config(ROOT / "config" / "shin2026-system.yaml")
    dataset = synthetic_keypoint_dataset(config, samples=4, seed=9)
    assert dataset["images"].shape == (4, 320, 512)
    assert dataset["heatmaps"].shape == (4, 6, *ShinKeypointEncoder.heatmap_shape(320, 512))
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


def test_keypoint_pretraining_can_bootstrap_a_compatible_validated_encoder(tmp_path):
    system = load_config(ROOT / "config" / "shin2026-system.yaml")
    base_experiment = {"estimator": {
        "image_embedding": 32,
        "keypoint_pretraining": {
            "enabled": True, "samples_quick": 4, "epochs_quick": 1,
            "batch_size": 2, "learning_rate": 1e-3, "seed": 23,
        },
    }}
    source = tmp_path / "source.pt"
    original = prepare_keypoint_encoder(
        source, config_hash="old", experiment=base_experiment,
        system=system, mode="quick", device="cpu")
    copied_experiment = {"estimator": {
        "image_embedding": 32,
        "keypoint_pretraining": {
            **base_experiment["estimator"]["keypoint_pretraining"],
            "bootstrap_artifact": "../source.pt",
        },
    }}
    target = tmp_path / "copied" / "target.pt"
    copied = prepare_keypoint_encoder(
        target, config_hash="new", experiment=copied_experiment,
        system=system, mode="quick", device="cpu")
    assert copied["config_hash"] == "new"
    assert copied["bootstrap_source"] == str(source.resolve())
    for name in original["encoder"]:
        torch.testing.assert_close(copied["encoder"][name], original["encoder"][name])


def test_beginner_curriculum_keeps_the_ugv_moving_at_a_safe_fraction():
    assert curriculum_motion_scale(0.0, 0.35) == pytest.approx(0.35)
    assert curriculum_motion_scale(0.5, 0.35) == pytest.approx(0.675)
    assert curriculum_motion_scale(1.0, 0.35) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        curriculum_motion_scale(0.0, 1.1)


def test_empirical_keypoint_labelling_projects_known_pad_landmarks():
    """Empirical labels come from simulator geometry, not from a detector.

    Feeding the synthetic renders back with the poses they were rendered from
    must reproduce the synthetic supervision exactly. Nothing is decoded from
    the image, so an unreadable frame still receives correct labels.
    """
    system = load_config(ROOT / "config/shin2026-system.yaml")
    rendered = synthetic_keypoint_dataset(system, samples=16, seed=91)
    visible_any = rendered["visible"].any(axis=1)
    samples = list(zip(rendered["images"], rendered["pad_relative_pose"]))
    labelled = empirical_keypoint_dataset(samples, system)

    assert labelled["images"].shape[1:] == (320, 512)
    assert labelled["coordinates"].shape[1:] == (6, 2)
    assert labelled["heatmaps"].shape[1:] == (6, *ShinKeypointEncoder.heatmap_shape(320, 512))
    # Every frame whose pose projects a landmark into the image is labelled,
    # and the coordinates are identical to the synthetic supervision because
    # both come from the same projection.
    assert len(labelled["images"]) == len(rendered["images"])
    np.testing.assert_allclose(
        labelled["coordinates"], rendered["coordinates"], atol=1e-5)
    # Visibility agrees wherever the synthetic generator did not deliberately
    # replace the frame with a target-absent negative.
    geometric = np.flatnonzero(visible_any)
    np.testing.assert_array_equal(
        labelled["visible"][geometric], rendered["visible"][geometric])

    # A pose payload that is missing or malformed drops the frame instead of
    # guessing, and a frame with no landmark in view is not labelled at all.
    assert len(empirical_keypoint_dataset(
        [(rendered["images"][0], None)], system)["images"]) == 0
    assert len(empirical_keypoint_dataset(
        [(rendered["images"][0], np.zeros(7))], system)["images"]) == 0


def _surveyed_isaac_camera(system, *, jitter_m=0.02, seed=3):
    """A labelled camera that renders whichever viewpoint the survey flew to.

    Stands in for Isaac: the frames are rendered from the commanded pose, so a
    survey that never moves produces one pose and a survey that visits the
    configured viewpoints produces that many.
    """
    from ontology_rgat.perception.keypoint_pretrain import (
        _pad_texture, _quat_wxyz_from_yaw, _render_landing_target,
        _texture_pyramid, camera_model, landing_pad_settings)

    model = camera_model(system)
    settings = landing_pad_settings(system)
    pyramid = _texture_pyramid(_pad_texture(settings))
    rng = np.random.default_rng(seed)
    flown = {"viewpoint": None}

    def survey(viewpoint):
        flown["viewpoint"] = viewpoint
        return True

    def labelled():
        viewpoint = flown["viewpoint"]
        position = (np.asarray(viewpoint["position_pad_m"], dtype=float)
                    + rng.normal(0.0, jitter_m, 3))
        quaternion = _quat_wxyz_from_yaw(float(viewpoint["yaw_rad"]))
        image = np.clip(rng.normal(96.0, 6.0, (model.height, model.width)),
                        0.0, 255.0).astype(np.uint8)
        image = _render_landing_target(
            image, pyramid, settings["deck_size_m"], position, quaternion, model)
        return image, np.concatenate((position, quaternion))

    return survey, labelled, flown


def _calibration_experiment(**overrides):
    keypoint = {
        "enabled": True, "samples_quick": 24, "epochs_quick": 2,
        "batch_size": 8, "learning_rate": 1e-3, "seed": 17,
        "empirical_samples_quick": 24, "empirical_steps": 4,
        "empirical_learning_rate": 1e-3, "empirical_replay_samples": 16,
        "empirical_eval_interval": 2, "render_workers": 1,
        # The quality of a 32-dimensional encoder trained for two epochs is
        # not what these tests are about; the split and the gates are.
        "minimum_holdout_visibility_recall": 0.0,
        "minimum_holdout_spread_ratio": 0.0,
    }
    keypoint.update(overrides)
    return {"estimator": {"image_embedding": 32, "keypoint_pretraining": keypoint}}


def _prepared_artifact(path, system, experiment):
    return prepare_keypoint_encoder(
        path, config_hash="cfg", experiment=experiment, system=system,
        mode="quick", device="cpu")


def test_every_calibration_viewpoint_puts_the_pad_in_the_camera_frame():
    system = load_config(ROOT / "config/shin2026-system.yaml")
    viewpoints = calibration_viewpoints(system, {})
    assert len(viewpoints) >= 12
    # Several distinct altitudes, and a lateral ring wide enough that some
    # views are only partial -- which is where a real approach spends its time.
    assert len({point["altitude_m"] for point in viewpoints}) >= 3
    assert min(point["landmarks_in_frame"] for point in viewpoints) >= 4
    assert min(point["landmarks_in_frame"] for point in viewpoints) < 6
    for point in viewpoints:
        x, y, z = point["position_pad_m"]
        # Inside the gateway's pad-frame goto envelope.
        assert math.hypot(x, y) <= 9.5 and 0.0 < z <= 24.0


def test_keypoint_calibration_surveys_poses_and_holds_whole_viewpoints_out(tmp_path):
    """The held-out split is over poses, so it cannot be passed by memorising.

    The superseded procedure polled 48 frames of one hover and split them at
    random: its 100 % held-out score certified an encoder that reported no
    landmark on 99.7 % of the in-frame steps that followed.
    """
    system = load_config(ROOT / "config/shin2026-system.yaml")
    experiment = _calibration_experiment()
    path = tmp_path / "keypoint_encoder.pt"
    artifact = _prepared_artifact(path, system, experiment)
    assert needs_empirical_calibration(artifact)
    survey, labelled, _ = _surveyed_isaac_camera(system)

    calibrated = calibrate_keypoint_encoder(
        path, artifact, labelled, survey=survey, system=system,
        experiment=experiment, mode="quick", device="cpu")

    empirical = calibrated["empirical_calibration"]
    assert empirical["format"] == EMPIRICAL_CALIBRATION_FORMAT
    assert empirical["viewpoints"] >= 6
    assert len(empirical["held_out_viewpoints"]) >= 2
    assert empirical["pose_span"]["altitude_span_m"] > 1.5
    assert empirical["pose_span"]["lateral_span_m"] > 1.0
    assert "visibility_recall" in empirical["after"]

    # No frame of a held-out viewpoint may have been trained on.
    dataset = np.load(calibrated["empirical_dataset"])
    held_out = set(empirical["held_out_viewpoints"])
    validation = int(sum(int(point) in held_out for point in dataset["viewpoint"]))
    assert validation == empirical["validation_samples"]
    assert (empirical["training_samples"] + validation
            == len(dataset["images"]) == empirical["samples"])
    # A recalibrated artifact is not calibrated again on the next run.
    assert not needs_empirical_calibration(calibrated)


def test_an_accumulated_calibration_reuses_its_viewpoints_and_flies_nothing(
        tmp_path):
    """Surveying costs simulator minutes; a second run should not repay them.

    And it must not re-fly a viewpoint it already holds either: that would add
    a second, near-identical copy of a pose the fit is already anchored on.
    """
    from ontology_rgat.datastore import (KIND_KEYPOINT_CALIBRATION,
                                         CollectedDataStore)

    system = load_config(ROOT / "config/shin2026-system.yaml")
    experiment = _calibration_experiment()
    survey, labelled, _ = _surveyed_isaac_camera(system)
    store = CollectedDataStore(tmp_path / "collected.sqlite3", run_id="first")

    first_path = tmp_path / "first.pt"
    first = calibrate_keypoint_encoder(
        first_path, _prepared_artifact(first_path, system, experiment),
        labelled, survey=survey, system=system, experiment=experiment,
        mode="quick", device="cpu", datastore=store)
    empirical = first["empirical_calibration"]
    assert empirical["reused_viewpoints"] == 0
    assert empirical["split_source"] == "frozen per-viewpoint datastore split"
    stored = store.episodes(KIND_KEYPOINT_CALIBRATION,
                            empirical["datastore_fingerprint"])
    assert len(stored) == empirical["viewpoints"] >= 6

    flown = []

    def refuse_to_fly(viewpoint):
        flown.append(int(viewpoint["index"]))
        return True

    second_path = tmp_path / "second.pt"
    second = calibrate_keypoint_encoder(
        second_path, _prepared_artifact(second_path, system, experiment),
        labelled, survey=refuse_to_fly, system=system, experiment=experiment,
        mode="quick", device="cpu", datastore=store)

    assert flown == []
    reused = second["empirical_calibration"]
    assert reused["reused_viewpoints"] == empirical["viewpoints"]
    assert reused["samples"] == empirical["samples"]
    # The held-out poses are the store's, so the second run cannot train on a
    # viewpoint the first one validated against.
    assert reused["held_out_viewpoints"] == empirical["held_out_viewpoints"]
    store.close()


def test_keypoint_calibration_refuses_a_frame_set_that_never_moved(tmp_path):
    system = load_config(ROOT / "config/shin2026-system.yaml")
    experiment = _calibration_experiment()
    path = tmp_path / "keypoint_encoder.pt"
    artifact = _prepared_artifact(path, system, experiment)
    survey, labelled, flown = _surveyed_isaac_camera(system, jitter_m=0.0)
    single = calibration_viewpoints(system, {})[0]

    def frozen_survey(_viewpoint):
        return survey(single)

    with pytest.raises(RuntimeError, match="degenerate frame set"):
        calibrate_keypoint_encoder(
            path, artifact, labelled, survey=frozen_survey, system=system,
            experiment=experiment, mode="quick", device="cpu")
    assert flown["viewpoint"] is single


def test_keypoint_calibration_rejects_an_encoder_blind_on_held_out_poses(tmp_path):
    system = load_config(ROOT / "config/shin2026-system.yaml")
    experiment = _calibration_experiment(
        minimum_holdout_visibility_recall=1.01)
    path = tmp_path / "keypoint_encoder.pt"
    artifact = _prepared_artifact(path, system, experiment)
    survey, labelled, _ = _surveyed_isaac_camera(system)

    with pytest.raises(RuntimeError, match="held-out viewpoints"):
        calibrate_keypoint_encoder(
            path, artifact, labelled, survey=survey, system=system,
            experiment=experiment, mode="quick", device="cpu")


class _StubSurveyBridge:
    """A PX4 bridge that flies to whatever pad-frame goto it is given."""

    entry_state = staticmethod(PX4Bridge.entry_state)

    def __init__(self, *, arrive_after=3, reachable=True):
        self.gotos = []
        self.arrive_after = int(arrive_after)
        self.reachable = bool(reachable)
        self.target = np.zeros(3)
        self.samples = 0
        self.last_state = {}

    def transact(self, kind, fields, expected, timeout=None):
        assert kind == "goto" and expected == ("ack",)
        self.gotos.append(dict(fields))
        self.target = np.asarray(fields["position"], dtype=float)
        self.samples = 0
        return {"type": "ack"}

    def get_state(self):
        self.samples += 1
        arrived = self.reachable and self.samples >= self.arrive_after
        position = self.target if arrived else self.target + 5.0
        self.last_state = {"truth": {
            "valid": True, "position": position, "velocity": np.zeros(3)}}
        return self.last_state


class _StubSurveyEnvironment:
    def __init__(self, cfg, image_source, *, horizon_steps, bridge):
        self.cfg = cfg
        self.image_source = image_source
        self.horizon_steps = horizon_steps
        self.bridge = bridge
        self.resets = []
        self.finished = False
        self.closed = False

    def reset(self, seed, curriculum=1.0):
        self.resets.append((int(seed), float(curriculum)))

    def finish_episode(self):
        self.finished = True

    def close(self):
        self.closed = True


def _stub_survey_flight(monkeypatch, *, settings, bridge):
    import run_shin2026_pipeline as runner

    built = {}

    def factory(cfg, image_source, *, horizon_steps):
        built["environment"] = _StubSurveyEnvironment(
            cfg, image_source, horizon_steps=horizon_steps, bridge=bridge)
        return built["environment"]

    monkeypatch.setattr(runner, "LiveShinEnvironment", factory)
    flight = runner.KeypointCalibrationFlight(
        default_config(), object(), seed=31337, settings=settings)
    return flight, built


def test_survey_flight_commands_pad_frame_viewpoints_and_lands_after(monkeypatch):
    system = load_config(ROOT / "config/shin2026-system.yaml")
    viewpoint = calibration_viewpoints(system, {})[4]
    bridge = _StubSurveyBridge()
    flight, built = _stub_survey_flight(
        monkeypatch,
        settings={"survey_settle_s": 0.0, "survey_travel_timeout_s": 5.0},
        bridge=bridge)

    with flight as survey:
        assert survey(viewpoint) is True
        goto = bridge.gotos[-1]
        # The offset is commanded in the pad frame, because the deck moves and
        # the gateway re-aims a pad-frame target at it every control tick.
        assert goto["frame"] == "pad"
        assert goto["position"] == pytest.approx(
            list(viewpoint["position_pad_m"]))
        assert goto["yaw"] == pytest.approx(viewpoint["yaw_rad"])
        # An expired goto makes the gateway command a landing, so the hold has
        # to outlast the travel and the frames captured after arrival.
        assert goto["hold_s"] > 5.0

    environment = built["environment"]
    # Setup, not a measured episode: the gentlest seeded initial condition.
    assert environment.resets == [(31337, 0.0)]
    assert environment.finished and environment.closed


def test_survey_flight_skips_a_viewpoint_it_cannot_reach(monkeypatch, capsys):
    system = load_config(ROOT / "config/shin2026-system.yaml")
    viewpoint = calibration_viewpoints(system, {})[0]
    flight, _ = _stub_survey_flight(
        monkeypatch,
        settings={"survey_settle_s": 0.0, "survey_travel_timeout_s": 0.3},
        bridge=_StubSurveyBridge(reachable=False))

    with flight as survey:
        assert survey(viewpoint) is False
    assert "could not reach viewpoint" in capsys.readouterr().out


def test_a_single_pose_calibrated_artifact_is_recalibrated():
    """An artifact certified by the superseded procedure is not reused."""
    assert needs_empirical_calibration(
        {"empirical_calibration": {"validated": True}})
    assert needs_empirical_calibration(
        {"empirical_calibration": {"validated": True, "format": "v0"}})
    assert not needs_empirical_calibration(
        {"empirical_calibration": {"validated": True,
                                   "format": EMPIRICAL_CALIBRATION_FORMAT}})


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
        max_position_rmse_m=2.0, max_geometric_fov_loss_fraction=.5)
    failing = {"paper_success": 0, "position_rmse": 4.0,
               "geometric_fov_loss_fraction": .9}
    for _ in range(40):
        assert not curriculum.observe(failing)
    assert curriculum.level == 1 and curriculum.c == 0.0
    passing = {"paper_success": 1, "position_rmse": 1.0,
               "geometric_fov_loss_fraction": .1}
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


def _classify_pad_contact(*, lateral=0.10, roll_deg=0.0,
                          vertical_speed=-0.10, relative_speed=0.10):
    env = LiveShinEnvironment.__new__(LiveShinEnvironment)
    env.steps = 10
    env.cfg = SimpleNamespace(
        sim=SimpleNamespace(
            world_xy_limit=30.0, ground_z=0.08,
            crash_tilt=math.radians(75.0)),
        criteria=SimpleNamespace(
            xy=0.35, vz=0.55, tilt=math.radians(10.0),
            rate=math.radians(45.0), rel_speed_xy=0.45))
    half = math.radians(roll_deg) / 2.0
    quaternion = np.array([math.cos(half), math.sin(half), 0.0, 0.0])
    actor = ActorObservation(
        image=np.zeros((32, 32), dtype=np.uint8),
        body_velocity=np.array([0.0, 0.0, vertical_speed]),
        attitude_quaternion=quaternion)
    state = {
        "truth": {
            "valid": True, "position": [-lateral, 0.0, 0.2],
            "velocity": [-relative_speed, 0.0, 0.0]},
        "quaternion_wxyz": quaternion,
        "angular_velocity": np.zeros(3),
        "extra": {"pad_contact": True},
        "battery": {"enabled": False},
        "landed": False,
    }
    return env._classify(actor, state, np.zeros(4))


def test_pad_contact_is_success_only_when_position_and_attitude_are_safe():
    safe = _classify_pad_contact()
    assert safe.physical_contact
    assert safe.strict_success
    assert not safe.unsafe_pad_contact
    assert not safe.crash
    assert _terminal_flags(safe).physical_contact

    for unsafe in (
            _classify_pad_contact(lateral=.50),
            _classify_pad_contact(roll_deg=15.0),
            _classify_pad_contact(vertical_speed=-.70),
            _classify_pad_contact(relative_speed=.60)):
        assert unsafe.physical_contact
        assert not unsafe.strict_success
        assert unsafe.unsafe_pad_contact
        assert unsafe.crash
        flags = _terminal_flags(unsafe)
        assert not flags.physical_contact
        assert flags.crash


def test_contact_gate_uses_pre_contact_kinematics_not_rebound_velocity():
    """A gentle approach must not fail on the deck's upward contact impulse."""
    env = LiveShinEnvironment.__new__(LiveShinEnvironment)
    env.steps = 9
    env.cfg = SimpleNamespace(
        sim=SimpleNamespace(
            world_xy_limit=30.0, ground_z=0.08,
            crash_tilt=math.radians(75.0)),
        criteria=SimpleNamespace(
            xy=0.35, vz=0.55, tilt=math.radians(10.0),
            rate=math.radians(45.0), rel_speed_xy=0.45))
    quaternion = np.array([1.0, 0.0, 0.0, 0.0])

    def sample(*, contact, vertical_speed):
        actor = ActorObservation(
            image=np.zeros((32, 32), dtype=np.uint8),
            body_velocity=np.array([0.0, 0.0, vertical_speed]),
            attitude_quaternion=quaternion)
        state = {
            "truth": {
                "valid": True, "position": [-0.05, 0.0, 0.2],
                "velocity": [-0.10, 0.0, 0.0]},
            "quaternion_wxyz": quaternion,
            "angular_velocity": np.zeros(3),
            "extra": {"pad_contact": contact},
            "battery": {"enabled": False},
            "landed": False,
        }
        return actor, state

    actor, state = sample(contact=False, vertical_speed=-0.18)
    env.last_step = env._classify(actor, state, np.zeros(4))
    actor, state = sample(contact=True, vertical_speed=1.05)
    contact = env._classify(actor, state, np.zeros(4))

    assert contact.strict_success
    assert contact.landing_metrics["vertical_velocity"] == pytest.approx(-0.18)
    assert contact.landing_metrics["kinematic_sample"] == "pre_contact"


def test_contact_gate_retains_hard_pre_contact_descent():
    """A rebound must not hide an unsafe impact velocity."""
    env = LiveShinEnvironment.__new__(LiveShinEnvironment)
    env.steps = 9
    env.cfg = SimpleNamespace(
        sim=SimpleNamespace(
            world_xy_limit=30.0, ground_z=0.08,
            crash_tilt=math.radians(75.0)),
        criteria=SimpleNamespace(
            xy=0.35, vz=0.55, tilt=math.radians(10.0),
            rate=math.radians(45.0), rel_speed_xy=0.45))
    quaternion = np.array([1.0, 0.0, 0.0, 0.0])

    def classify(*, contact, vertical_speed):
        actor = ActorObservation(
            image=np.zeros((32, 32), dtype=np.uint8),
            body_velocity=np.array([0.0, 0.0, vertical_speed]),
            attitude_quaternion=quaternion)
        state = {
            "truth": {"valid": True, "position": [-0.05, 0.0, 0.2],
                      "velocity": [-0.10, 0.0, 0.0]},
            "quaternion_wxyz": quaternion,
            "angular_velocity": np.zeros(3),
            "extra": {"pad_contact": contact},
            "battery": {"enabled": False}, "landed": False,
        }
        return env._classify(actor, state, np.zeros(4))

    env.last_step = classify(contact=False, vertical_speed=-0.72)
    contact = classify(contact=True, vertical_speed=1.05)

    assert not contact.strict_success
    assert contact.unsafe_pad_contact
    assert contact.landing_metrics["vertical_velocity"] == pytest.approx(-0.72)


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
    assert env.bridge.cfg.require_pad_in_view is True
    assert reset_kwargs == [{"scenario": "circle", "initial_condition_scale": 0.0}]
    assert result[:2] == ("actor", {"state": "ready"})


def test_full_curriculum_still_requires_initial_pad_visibility():
    flags = []
    env = LiveShinEnvironment.__new__(LiveShinEnvironment)
    env.cfg = SimpleNamespace(external=SimpleNamespace(reset_recoveries=0))
    env.control = {"curriculum_min_pad_motion_scale": .35,
                   "require_initial_pad_visible": True}
    env.bridge = SimpleNamespace(cfg=SimpleNamespace(pad_scale=0.0))
    env.adapter = SimpleNamespace(
        controller=SimpleNamespace(set_curriculum=lambda _value: None),
        reset=lambda *_args, **_kwargs: flags.append(
            env.bridge.cfg.require_pad_in_view) or ("actor", {"ready": True}))
    env._classify = lambda actor, state, command: (actor, state, command.tolist())
    result = env.reset(9, curriculum=1.0)
    assert flags == [True]
    assert result[0] == "actor"


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


def test_keypoint_calibration_names_which_filter_discarded_the_frames():
    """"0 labelled frames" cannot tell a dead camera from a dead pose stream.

    Five separate conditions drop a sample silently, and each points at a
    different part of the stack. The message has to say which one fired.
    """
    from ontology_rgat.perception.keypoint_pretrain import (
        _calibration_diagnosis, empirical_keypoint_dataset)

    system = load_config(ROOT / "config/shin2026-system.yaml")
    image = np.zeros((320, 512), dtype=np.uint8)
    pose = np.array([0.0, 0.0, 4.0, 1.0, 0.0, 0.0, 0.0])

    # Frames arrive, the truth pose never does: today's failure.
    rejections = {}
    dataset = empirical_keypoint_dataset(
        [(image, None)] * 5, system, rejections=rejections)
    assert len(dataset["images"]) == 0
    assert rejections["no_pose"] == 5
    assert rejections["no_image"] == 0
    text = _calibration_diagnosis(5, rejections)
    assert "no_pose=5" in text
    assert "truth-pose topic is publishing" in text

    # No frames at all points at the camera instead.
    rejections = {}
    empirical_keypoint_dataset([(None, pose)] * 4, system,
                               rejections=rejections)
    assert rejections["no_image"] == 4
    assert "camera topic published nothing" in _calibration_diagnosis(
        4, rejections)

    # A pose that simply is not looking at the deck is its own case.
    rejections = {}
    empirical_keypoint_dataset(
        [(image, np.array([60.0, 60.0, 4.0, 1.0, 0.0, 0.0, 0.0]))] * 3,
        system, rejections=rejections)
    assert rejections["no_landmark_in_frame"] == 3
    assert "outside the frame" in _calibration_diagnosis(3, rejections)

    # A zero-norm quaternion is malformed, not merely out of view.
    rejections = {}
    empirical_keypoint_dataset(
        [(image, np.zeros(7))] * 2, system, rejections=rejections)
    assert rejections["degenerate_attitude"] == 2

    assert "never polled" in _calibration_diagnosis(0, {})


def test_valid_frames_still_label_and_report_no_rejections():
    from ontology_rgat.perception.keypoint_pretrain import (
        empirical_keypoint_dataset)

    system = load_config(ROOT / "config/shin2026-system.yaml")
    image = np.zeros((320, 512), dtype=np.uint8)
    pose = np.array([0.0, 0.0, 4.0, 1.0, 0.0, 0.0, 0.0])
    rejections = {}
    dataset = empirical_keypoint_dataset([(image, pose)] * 3, system,
                                         rejections=rejections)
    assert len(dataset["images"]) == 3
    assert sum(rejections.values()) == 0


def _trajectory_row_schema():
    """Keys the rollout row dict in ``collect_episode`` actually carries."""
    import ast

    source = (ROOT / "python/ontology_rgat/ppo/recurrent_train.py").read_text()
    tree = ast.parse(source)
    collect = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef)
                   and node.name == "collect_episode")
    written = set()
    for node in ast.walk(collect):
        # row = {...}: the literal schema of one trajectory step.
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
                and any(isinstance(target, ast.Name) and target.id == "row"
                        for target in node.targets)):
            written.update(key.value for key in node.value.keys
                           if isinstance(key, ast.Constant))
        # row["..."] = ...: fields attached after the literal.
        for target in getattr(node, "targets", []):
            if (isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "row"
                    and isinstance(target.slice, ast.Constant)):
                written.add(target.slice.value)
    assert "reward" in written and "geometric_in_fov" in written
    return tree, written


def test_every_trajectory_row_field_read_is_a_field_that_is_written():
    # ``collect_episode`` only runs against live Isaac/PX4, so a renamed row
    # key cannot be caught by an offline rollout test.  Renaming the readers
    # without the writer once cost a full run: the FOV-risk data stage died on
    # KeyError('fov_graph_geometric_in_fov') after the episodes were flown.
    import ast

    tree, written = _trajectory_row_schema()
    read = {node.slice.value for node in ast.walk(tree)
            if isinstance(node, ast.Subscript)
            and isinstance(node.ctx, ast.Load)
            and isinstance(node.value, ast.Name) and node.value.id == "row"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)}
    assert read <= written, f"rollout rows never carry {sorted(read - written)}"


def test_run_pipeline_reads_only_trajectory_fields_that_exist():
    _, written = _trajectory_row_schema()
    consumed = {"semantic_graph_X", "fov_graph_X", "next_fov_graph_X",
                "fov_graph_geometric_in_fov", "geometric_in_fov"}
    source = (ROOT / "python/run_three_pipeline.py").read_text()
    for key in sorted(consumed):
        assert f'row["{key}"]' in source or f'rows[-1]["{key}"]' in source, (
            f"{key} is no longer consumed by the run pipeline; drop it here")
    assert consumed <= written, (
        f"rollout rows never carry {sorted(consumed - written)}")


def _shipped_configs():
    return sorted((ROOT / "config").rglob("*.yaml"))


def test_every_shipped_config_is_parseable_yaml():
    # A config typo only surfaced once Isaac Sim, PX4 and the dashboard were
    # already up, throwing away the whole boot.  Parse every shipped config
    # here instead.
    import yaml

    configs = _shipped_configs()
    assert configs, "no experiment/system configs were found"
    for path in configs:
        try:
            yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            pytest.fail(f"{path.relative_to(ROOT)} is not valid YAML: {exc}")


def test_no_shipped_config_declares_a_key_twice():
    # PyYAML keeps the last duplicate silently, so a stray paste can change a
    # contract without any error at all.  A repeated key is always a mistake.
    import yaml

    class _StrictLoader(yaml.SafeLoader):
        pass

    def _no_duplicates(loader, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    None, None, f"duplicate key {key!r}", key_node.start_mark)
            seen.add(key)
        return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)

    _StrictLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates)

    for path in _shipped_configs():
        try:
            yaml.load(path.read_text(encoding="utf-8"), Loader=_StrictLoader)
        except yaml.YAMLError as exc:
            pytest.fail(f"{path.relative_to(ROOT)}: {exc}")


def test_the_fov_risk_arm_reads_its_reward_from_the_graph_itself():
    contract = load_experiment(
        ROOT / "config/experiments/two_pipeline_comparison.yaml")
    proposed = contract["pipeline_contract"]["shin_se_onto_rgat_recovery"]
    assert proposed["fov_reward_readout"] == "direct_graph_scalar"
    assert proposed["fov_risk_reward"] is True


def test_the_launcher_takes_over_a_previous_run_instead_of_refusing():
    launcher = (ROOT.parent / "run.sh").read_text(encoding="utf-8")
    # A new run must end the old one: two learners on one Isaac/PX4 resource
    # steal each other's UDP replies and resets.
    assert "stop_previous_flight_pipeline" in launcher
    assert "--no-takeover" in launcher, "the strict refusal must stay available"
    # Killing the new run's own process group would be suicide, and killing a
    # shell that merely mentions the flight programs would take out a terminal.
    assert '"$group" != "$own_group"' in launcher
    assert '[[ "$second" == "-c" ]] && return 1' in launcher
    # Signals escalate; SIGKILL alone leaves Isaac's context and PX4 unclean.
    assert "for signal in INT TERM KILL" in launcher
