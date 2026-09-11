from pathlib import Path

import numpy as np

from ontology_rgat.config import default_config
from ontology_rgat.evaluation.compare import wilson_interval
from ontology_rgat.ppo.networks import load_agent, save_agent
from ontology_rgat.ppo.train import train_ppo
from ontology_rgat.rgat.dataset import load_dataset, merge_datasets, save_dataset
from ontology_rgat.rgat.model import load_potential, save_potential
from ontology_rgat.rgat.train import train_potential
from ontology_rgat.semantic import SemanticState, build_ontology_graph


def _dataset(graph, episode, label):
    return {
        "X": np.repeat(graph.X.T[None], 2, axis=0).astype(np.float32),
        "y": np.asarray([label, label], dtype=np.float32),
        "meta": np.asarray([[episode, 0, label > 0, 0.1],
                            [episode, 1, label > 0, 0.1]], dtype=np.float64),
        "graph": graph,
    }


def test_cumulative_dataset_is_appended_and_committed_atomically(tmp_path):
    cfg = default_config()
    graph = build_ontology_graph(SemanticState(), cfg)
    merged = merge_datasets(_dataset(graph, 1, -1.0), _dataset(graph, 2, 1.0))
    path = tmp_path / "dataset.npz"

    save_dataset(merged, path)
    loaded = load_dataset(path)

    assert loaded["X"].shape[0] == 4
    assert loaded["meta"][:, 0].tolist() == [1.0, 1.0, 2.0, 2.0]
    assert np.count_nonzero(loaded["y"] > 0.0) == 2
    assert not list(tmp_path.glob("*.tmp"))


def test_rgat_checkpoint_continues_weights_optimizer_and_history(tmp_path):
    cfg = default_config().derive(**{
        "rgat.epochs": 1, "rgat.batch_size": 2, "device.rgat": "cpu"})
    graph = build_ontology_graph(SemanticState(), cfg)
    dataset = merge_datasets(_dataset(graph, 1, -1.0), _dataset(graph, 2, 1.0))
    first, history1 = train_potential(dataset, cfg, verbose=False)
    path = tmp_path / "rgat.pt"
    save_potential(first, cfg, path, dict(history1))
    loaded, old_history = load_potential(path, cfg, graph)

    second, history2 = train_potential(
        dataset, cfg, initial_model=loaded, initial_history=old_history,
        verbose=False)

    assert len(history1["train_loss"]) == 1
    assert len(history2["train_loss"]) == 2
    assert getattr(second, "_optimizer_state")


def test_ppo_checkpoint_adds_episodes_instead_of_restarting(monkeypatch, tmp_path):
    cfg = default_config().derive(**{
        "ppo.train_episodes": 1, "ppo.rollout_steps": 1,
        "ppo.epochs": 1, "ppo.minibatch": 1, "device.ppo": "cpu"})

    def rollout(agent, reward_mode, potential, seed, cfg):
        return {
            "O": np.zeros((1, cfg.rl.obs_dim)),
            "U": np.zeros((1, cfg.rl.act_dim)),
            "A": np.zeros((1, cfg.rl.act_dim)),
            "R": np.ones(1), "V": np.zeros(1), "LP": np.zeros(1),
            "D": np.ones(1), "last_value": 0.0, "status": "success",
        }

    monkeypatch.setattr("ontology_rgat.ppo.train._collect_episode", rollout)
    first, history1 = train_ppo("manual", None, cfg, verbose=False)
    path = tmp_path / "ppo.pt"
    save_agent(first, cfg, path, dict(history1))
    loaded, old_history = load_agent(path, cfg)

    second, history2 = train_ppo(
        "manual", None, cfg, initial_agent=loaded,
        initial_history=old_history, verbose=False)

    assert history1["episode"] == [1]
    assert history2["episode"] == [1, 2]
    assert getattr(second, "_optimizer_state")


def test_success_interval_is_not_a_misleading_exact_zero():
    low, high = wilson_interval(0, 12)
    assert low == 0.0
    assert 0.20 < high < 0.30
