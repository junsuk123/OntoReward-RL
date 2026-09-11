import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "isaac_sim"))

from pad_motion import PadMotionConfig, PadTrajectory  # noqa: E402
from ontology_rgat_px4.protocol import ProtocolError, validate_velocity_action

from ontology_rgat.benchmarks.experiment import paired_seed_plan
from ontology_rgat.benchmarks.px4_adapter import (actor_observation_from_state,
                                                  critic_observation_from_state)
from ontology_rgat.benchmarks.randomization import (sample_domain_randomization,
                                                     sample_initial_condition)
from ontology_rgat.benchmarks.shin2026 import (ActorObservation,
                                               assert_actor_payload_safe,
                                               default_shin2026_config)
from ontology_rgat.controllers import VelocityYawRateController
from ontology_rgat.estimation import LSTMRelativeStateEstimator
from ontology_rgat.reward_modes import (OntoRewardPBRS, ShinReward,
                                        ShinRewardConfig, FrozenControlledPotential,
                                        active_perception_reward)


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
    np.testing.assert_allclose(command[:3], [0.6, 0.6, 0.4])
    with pytest.raises(ValueError):
        controller.command([1, 2, 3])


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


def test_shin_active_reward_uses_training_only_estimation_target():
    cfg = ShinRewardConfig()
    assert active_perception_reward(0.0, cfg) == 0.0
    assert active_perception_reward(0.51, cfg) == pytest.approx(-0.05)
    assert active_perception_reward(2.0, cfg) == pytest.approx(-0.1)
    reward = ShinReward(cfg)
    with pytest.raises(ValueError, match="training-only"):
        reward(np.zeros(6), np.zeros(6), np.zeros(4), drone_vertical_velocity=0.0)


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


def test_benchmark_dimensions_and_modes():
    for mode in ("shin2026", "sparse", "manual_no_active", "ontoreward",
                 "ontoreward_plus_active"):
        cfg = default_shin2026_config(reward_mode=mode)
        assert cfg.action_dimension == 4 and cfg.relative_state_dimension == 6
