from __future__ import annotations

import pytest
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
        "action": torch.tensor([[.45, -.35, .05]]).repeat(count, 1),
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


def test_behavior_cloning_anchor_preserves_ppo_exploration_variance():
    model = _model("no_se_fixed")
    with torch.no_grad():
        model.log_std.fill_(-2.35)
    metrics = behavior_clone(
        model, _dataset(), epochs=1, learning_rate=1e-4,
        sequence_length=12, post_log_std=None)
    assert torch.allclose(model.log_std, torch.full_like(model.log_std, -2.35))
    assert metrics["post_action_std"] == pytest.approx(
        torch.exp(torch.tensor(-2.35)).item())


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


def test_demonstrations_with_retained_frames_survive_an_encoder_change(tmp_path):
    """The 2026-09-20 encoder redesign had to re-fly every teacher attempt
    because the stored set held only the old embeddings. Frames are now kept
    (PNG) and a mismatched set is re-embedded rather than rejected."""
    import numpy as np

    from ontology_rgat.ppo.behavior_cloning import (
        FRAME_FIELD, encoded_demonstration_episode, merge_encoded_demonstrations)

    old_model, new_model = _model("no_se_fixed"), _model("no_se_fixed")
    torch.manual_seed(5)
    with torch.no_grad():
        for parameter in new_model.encoder.parameters():
            parameter.add_(0.05 * torch.randn_like(parameter))
    rng = np.random.default_rng(3)
    rows = [{"image": rng.integers(0, 120, (320, 512), dtype=np.uint8),
             "proprioception": np.zeros(7), "action": np.zeros(3), "truth": np.zeros(6)}
            for _ in range(5)]
    first = encoded_demonstration_episode(old_model, rows[:3], episode_id=1)
    second = encoded_demonstration_episode(old_model, rows[3:], episode_id=2)
    dataset = merge_encoded_demonstrations(merge_encoded_demonstrations(None, first), second)
    assert len(dataset[FRAME_FIELD]) == 5
    path = tmp_path / "demos.pt"
    saved = save_encoded_demonstrations(
        path, dataset, config_hash="cfg", encoder_sha256="old",
        attempted_seeds=[1, 2], environment_steps=5)
    assert saved["frames_retained"] is True

    # Same encoder: nothing changes.
    same = load_encoded_demonstrations(path, config_hash="cfg", encoder_sha256="old")
    torch.testing.assert_close(same["dataset"]["embedding"], dataset["embedding"])

    # New encoder, frames retained: re-embedded with the new model.
    loaded = load_encoded_demonstrations(
        path, config_hash="cfg", encoder_sha256="new", model=new_model)
    assert loaded["encoder_sha256"] == "new"
    assert loaded["re_embedded_from_encoder_sha256"] == "old"
    expected = new_model.encoder(torch.as_tensor(
        np.stack([row["image"] for row in rows])[:, None], dtype=torch.float32) / 255.0)
    torch.testing.assert_close(loaded["dataset"]["embedding"], expected.embedding,
                               atol=1e-5, rtol=0)
    assert not torch.allclose(loaded["dataset"]["embedding"], dataset["embedding"])

    # New encoder but no model to re-embed with: still refused.
    with pytest.raises(ValueError, match="encoder mismatch"):
        load_encoded_demonstrations(path, config_hash="cfg", encoder_sha256="new")

    # A set without frames cannot be re-embedded either.
    bare = {name: dataset[name] for name in dataset if name != FRAME_FIELD}
    bare_path = tmp_path / "bare.pt"
    save_encoded_demonstrations(bare_path, bare, config_hash="cfg", encoder_sha256="old",
                                attempted_seeds=[1], environment_steps=5)
    with pytest.raises(ValueError, match="encoder mismatch"):
        load_encoded_demonstrations(bare_path, config_hash="cfg", encoder_sha256="new",
                                    model=new_model)
