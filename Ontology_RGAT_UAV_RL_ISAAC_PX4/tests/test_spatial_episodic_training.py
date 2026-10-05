"""Opt-in upstream-style full episodes; legacy bounded rollouts stay unchanged."""
from dataclasses import replace
import copy

import numpy as np
import pytest
import torch

from ontology_rgat.spatial.core import SpatialConfig
from ontology_rgat.spatial.environment import SpatialLandingEnv
from ontology_rgat.spatial.training import SpatialAgent, train_arm
from ontology_rgat.two_axis.learning import collect_rollout
from ontology_rgat.two_axis.training import PPOHyperparameters, PPOTrainer


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_full_episode_batch_preserves_every_terminal_and_does_not_open_unused_episode():
    cfg = replace(SpatialConfig(), horizon=.3)
    env = SpatialLandingEnv(cfg)
    agent = SpatialAgent('ppo_vector_canonical', cfg, 29)
    batch = collect_rollout(agent, env, seed=100, episodes=3, max_episode_decisions=10)
    assert len(batch) == 9
    assert sum(t.terminated for t in batch) == 3
    assert not any(t.truncated for t in batch)
    assert env.done and env.episode_seed == 102
    assert len(env.episode_history) == 3


def test_full_episode_collector_is_identical_to_same_length_bounded_rollout():
    cfg = replace(SpatialConfig(), horizon=.3)
    a = SpatialAgent('ppo_semantic_flat', cfg, 29)
    b = copy.deepcopy(a)
    left = collect_rollout(a, SpatialLandingEnv(cfg), seed=10, episodes=2,
                           max_episode_decisions=10)
    right = collect_rollout(b, SpatialLandingEnv(cfg), seed=10, decisions=6)
    for x,y in zip(left,right):
        np.testing.assert_array_equal(x.raw_command,y.raw_command)
        assert x.reward == y.reward and x.terminated == y.terminated


def test_full_episode_collector_fails_closed_on_missing_terminal():
    cfg = replace(SpatialConfig(), horizon=70.)
    agent = SpatialAgent('ppo_vector_canonical', cfg, 29)
    with pytest.raises(RuntimeError, match='without a task terminal'):
        collect_rollout(agent, SpatialLandingEnv(cfg), seed=10, episodes=1,
                        max_episode_decisions=2)
    with pytest.raises(ValueError, match='exactly one'):
        collect_rollout(agent, SpatialLandingEnv(cfg), seed=10, episodes=1, decisions=2)


def test_episodic_training_reports_actual_steps_not_unused_decision_budget(tmp_path):
    cfg = replace(SpatialConfig(), horizon=.3)
    hyper = PPOHyperparameters(iterations=2, decisions_per_iteration=123,
                               epochs=1, minibatch_size=4,
                               advantage_normalization='rollout')
    result = train_arm('ppo_vector_canonical', cfg, seed=29, hyper=hyper,
                       output=tmp_path, episodes_per_iteration=2,
                       curriculum_enabled=False, activation_iterations=0)
    assert result['environment_steps'] == 12
    assert result['completed_nominal_episodes'] == 4
    assert result['rollout_budget_unit'] == 'complete_episodes'
    assert result['history'][-1]['environment_steps'] == 12
    assert result['hyperparameters']['advantage_normalization'] == 'rollout'


def test_rollout_whitening_is_an_explicit_distinct_optimizer_contract():
    cfg = replace(SpatialConfig(), horizon=1.2)
    a = SpatialAgent('ppo_vector_canonical', cfg, 29)
    b = copy.deepcopy(a)
    data = collect_rollout(a, SpatialLandingEnv(cfg), seed=10, decisions=12)
    hyper = PPOHyperparameters(epochs=1,minibatch_size=4,target_kl=100.)
    first = PPOTrainer(a, hyper).update(data,discount_time_constant_s=70.)
    second = PPOTrainer(b, replace(hyper,advantage_normalization='rollout')).update(
        data,discount_time_constant_s=70.)
    assert first['minibatches'] == second['minibatches'] == 3
    assert any(not torch.equal(x,y) for x,y in zip(a.actor.parameters(),b.actor.parameters()))
    with pytest.raises(ValueError,match='normalization'):
        replace(hyper,advantage_normalization='unknown')


def test_reference_loss_curriculum_cannot_change_nominal_contract():
    cfg = replace(SpatialConfig(),schema='spatial-causal-rgat/8')
    env=SpatialLandingEnv(cfg)
    for difficulty, expected in [(0.,12.),(.5,7.5),(1.,3.)]:
        env.reset(seed=10,difficulty=difficulty,loss_curriculum_start=12.)
        assert env.task_cfg.loss_timeout == expected
    assert env.task_cfg is cfg
    a=SpatialLandingEnv(cfg)
    b=SpatialLandingEnv(cfg)
    left,_=a.reset(seed=22,difficulty=1.,loss_curriculum_start=12.)
    right,_=b.reset(seed=22,difficulty=1.)
    np.testing.assert_array_equal(left.packet.values,right.packet.values)
    for _ in range(8):
        left,right=a.step(np.zeros(3)),b.step(np.zeros(3))
        np.testing.assert_array_equal(left[0].packet.values,right[0].packet.values)
        assert left[1:4]==right[1:4]
    with pytest.raises(ValueError,match='curriculum'):
        env.reset(seed=10,difficulty=0.,loss_curriculum_start=2.)
def test_validation_failure_keeps_an_ineligible_diagnostic_update(tmp_path, monkeypatch):
    from dataclasses import replace
    import pytest
    from ontology_rgat.spatial.core import SpatialConfig
    from ontology_rgat.spatial import training
    from ontology_rgat.two_axis.training import PPOHyperparameters
    cfg=replace(SpatialConfig(),horizon=.2)
    def failed_validation(*args,**kwargs):
        raise RuntimeError("simulated cleanup failure")
    monkeypatch.setattr(training,"evaluate",failed_validation)
    with pytest.raises(RuntimeError,match="cleanup"):
        training.train_arm("ppo_vector_canonical",cfg,seed=42,
            hyper=PPOHyperparameters(iterations=1,decisions_per_iteration=2,
                                     epochs=1,minibatch_size=2,evaluation_every=1),
            output=tmp_path,curriculum_enabled=False,activation_iterations=0)
    _,meta=training.load_agent(tmp_path/"checkpoint_last_update.pt",cfg)
    assert meta["eligible"] is False and meta["phase"]=="unvalidated_update_snapshot"
    assert meta["iteration"]==1 and not meta["optimizer_state_saved"]
    assert not (tmp_path/"checkpoint_best.pt").exists()
    assert not (tmp_path/"summary.json").exists()
