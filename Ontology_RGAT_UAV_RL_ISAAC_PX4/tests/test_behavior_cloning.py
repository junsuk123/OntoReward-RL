from __future__ import annotations

import torch

from ontology_rgat.ppo.behavior_cloning import (
    behavior_clone, load_encoded_demonstrations,
    save_encoded_demonstrations)
from ontology_rgat.ppo.recurrent import PipelineActorCritic


def _dataset():
    torch.manual_seed(71)
    count = 24
    return {
        "embedding": torch.randn(count, 8),
        "proprioception": torch.randn(count, 7) * .1,
        "action": torch.tensor([[.45, -.20, -.35, .05]]).repeat(count, 1),
        "truth": torch.zeros(count, 6),
        "episode_id": torch.tensor([1] * 12 + [2] * 12),
    }


def _model(pipeline):
    torch.manual_seed(19)
    return PipelineActorCritic(
        image_embedding=8, lstm_hidden=16, latent_dim=16,
        actor_hidden=16, critic_hidden=8, pipeline=pipeline)


def test_behavior_cloning_makes_an_estimator_free_actor_active():
    metrics = behavior_clone(
        _model("no_se_fixed"), _dataset(), epochs=16,
        learning_rate=2e-3, sequence_length=12, post_log_std=-1.8)
    assert metrics["action_loss_after"] < .35 * metrics["action_loss_before"]
    assert metrics["successful_demonstration_episodes"] == 2
    assert metrics["post_action_std"] < .2


def test_behavior_cloning_also_pretrains_the_shin_auxiliary_head():
    metrics = behavior_clone(
        _model("shin_se_fixed"), _dataset(), epochs=12,
        learning_rate=2e-3, sequence_length=12,
        auxiliary_coefficient=.2)
    assert metrics["action_loss_after"] < metrics["action_loss_before"]
    assert metrics["auxiliary_loss_after"] < metrics["auxiliary_loss_before"]


def test_encoded_demonstrations_are_hash_bound_and_restartable(tmp_path):
    path = tmp_path / "demonstrations.pt"
    saved = save_encoded_demonstrations(
        path, _dataset(), config_hash="cfg-a", encoder_sha256="encoder-a",
        attempted_seeds=[9, 10, 11], environment_steps=48)
    loaded = load_encoded_demonstrations(
        path, config_hash="cfg-a", encoder_sha256="encoder-a")
    assert saved["successful_episodes"] == 2
    assert loaded["transitions"] == 24
    assert loaded["attempted_seeds"] == [9, 10, 11]
    try:
        load_encoded_demonstrations(
            path, config_hash="cfg-b", encoder_sha256="encoder-a")
    except ValueError as exc:
        assert "config mismatch" in str(exc)
    else:
        raise AssertionError("mismatched demonstration config was accepted")
