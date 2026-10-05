"""Guarded activation is not a forced nonzero-weight or performance claim."""
import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from ontology_rgat.two_axis.config import load_config, REFERENCE_CONFIG_PATH
from ontology_rgat.two_axis.models import TwoAxisPPOAgent
from ontology_rgat.two_axis.relational import (
    RelationGuardOptions, assert_raw_preserved, ensure_relational_path,
    guard_relational_candidate, relational_path_audit, scale_relational_readout)
from ontology_rgat.two_axis.training import PPOHyperparameters, VALIDATION_SEEDS, TEST_SEEDS


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def pair():
    config = load_config(REFERENCE_CONFIG_PATH)
    anchor = TwoAxisPPOAgent("ppo_ontology_rgat", seed=4, graph_config=config.ontology)
    candidate = copy.deepcopy(anchor)
    with torch.no_grad():
        for head in (candidate.actor, candidate.critic):
            head.encoder.readout.weight.fill_(.01)
            head.encoder.readout.bias.fill_(.02)
    return anchor, candidate


def metrics(residual=0., **overrides):
    return {"landing_rate": .72, "unsafe_rate": .09, "safe_abort_rate": .14,
            "task_timeout_rate": .05, "mean_return": 17.93, "selection_score": 498.43,
            "mean_abs_relation_residual": [residual, 0.], **overrides}


def test_guard_chooses_largest_safe_scale_and_preserves_both_inputs():
    anchor, candidate = pair()
    anchor_state, candidate_state = copy.deepcopy(anchor.state_dict()), copy.deepcopy(candidate.state_dict())
    def evaluate(agent):
        scale = float(agent.actor.encoder.readout.weight[0, 0]) / .01
        return metrics(residual=.01*scale)
    selected, report = guard_relational_candidate(
        anchor, candidate, evaluator=evaluate,
        options=RelationGuardOptions(scale_grid=(1., .5, .1), max_residual_norm=.0051))
    assert report["accepted"] and report["selected_scale"] == .5
    assert report["selected_audit"]["parameter_path_active"]
    assert_raw_preserved(anchor, selected)
    assert all(torch.equal(v, anchor.state_dict()[k]) for k,v in anchor_state.items())
    assert all(torch.equal(v, candidate.state_dict()[k]) for k,v in candidate_state.items())


@pytest.mark.parametrize("changes", [
    {"landing_rate": .71}, {"unsafe_rate": .10}, {"safe_abort_rate": .15},
    {"task_timeout_rate": .06}, {"mean_return": 17.}, {"selection_score": 497.},
    {"mean_abs_relation_residual": [0.,0.]}, {"mean_return": float("nan")},
])
def test_guard_rejects_degradation_or_inactive_or_invalid_output(changes):
    anchor, candidate = pair()
    selected, report = guard_relational_candidate(
        anchor, candidate,
        evaluator=lambda a: metrics() if a is anchor else metrics(.001, **changes),
        options=RelationGuardOptions(scale_grid=(1., .1)))
    assert selected is anchor and not report["accepted"]


def test_nonzero_weights_with_zero_actual_residual_are_not_accepted():
    anchor, candidate = pair()
    assert relational_path_audit(candidate)["parameter_path_active"]
    selected, report = guard_relational_candidate(
        anchor, candidate, evaluator=lambda _: metrics())
    assert selected is anchor and not report["accepted"]


def test_raw_or_log_std_change_is_rejected_before_validation():
    anchor, candidate = pair()
    with torch.no_grad():
        candidate.actor.log_std.add_(.01)
    with pytest.raises(ValueError, match="log_std"):
        guard_relational_candidate(anchor, candidate, evaluator=lambda _: pytest.fail())


def test_readout_scale_applies_to_bias_without_changing_heads_or_raw():
    anchor, candidate = pair()
    scaled = scale_relational_readout(candidate, .001)
    assert_raw_preserved(anchor, scaled)
    for role in ("actor", "critic"):
        a, b = getattr(candidate, role), getattr(scaled, role)
        assert torch.equal(b.encoder.readout.bias, a.encoder.readout.bias*.001)
        assert torch.equal(b.residual.weight, a.residual.weight)


def test_repair_refuses_test_split_before_training():
    anchor, _ = pair()
    with pytest.raises(ValueError, match="validation split"):
        ensure_relational_path(anchor, load_config(REFERENCE_CONFIG_PATH),
                               PPOHyperparameters(), seed=4, validation_seeds=TEST_SEEDS[:1])


def test_bounded_real_ppo_repair_changes_only_relation_parameters(monkeypatch):
    # Real causal environment and PPO gradients; mock ONLY the expensive final
    # outcome evaluator. Thus this tests wiring, not landing performance.
    import ontology_rgat.two_axis.training as training
    anchor, _ = pair()
    config = load_config(REFERENCE_CONFIG_PATH)
    config = replace(config, ontology=replace(config.ontology,
                     activation_iterations=2, activation_decisions=16))
    captured = []
    def evaluate(agent, config, *, seeds):
        captured.append(tuple(seeds))
        audit = relational_path_audit(agent)
        return metrics(.001 if audit["parameter_path_active"] else 0.)
    monkeypatch.setattr(training, "evaluate_policy", evaluate)
    selected, report = ensure_relational_path(
        anchor, config, PPOHyperparameters(epochs=2, minibatch_size=16), seed=4,
        validation_seeds=VALIDATION_SEEDS[:1])
    assert report["environment_steps"] == 32 and report["changed"]
    assert captured and set(captured) == {tuple(VALIDATION_SEEDS[:1])}
    assert_raw_preserved(anchor, selected)
    assert report["after_audit"]["parameter_path_active"]


@pytest.mark.parametrize("kwargs", [
    {"scale_grid": (.1,1.)}, {"min_residual_norm": .1, "max_residual_norm": .01},
    {"rate_tolerance": float("nan")},
])
def test_guard_options_are_validated(kwargs):
    with pytest.raises(ValueError):
        RelationGuardOptions(**kwargs)


def test_train_save_and_reload_with_real_guard_on_bounded_timeout_task(tmp_path):
    from ontology_rgat.two_axis.training import train_arm, load_agent
    config = load_config(REFERENCE_CONFIG_PATH)
    # A synthetic 0.2 s mission exercises selection, repair and persistence
    # cheaply. It is deliberately NOT a landing benchmark.
    config = replace(config,
        timing=replace(config.timing, mission_duration_limit_s=.2),
        scenario=replace(config.scenario, T1_range_s=(.04,.06),
                         T2_range_s=(.04,.06), T3_range_s=(.04,.06)),
        reward=replace(config.reward, reference_duration_s=.2),
        curriculum=replace(config.curriculum, enabled=False),
        ontology=replace(config.ontology, pretrain_episodes=0,
                         activation_iterations=1, activation_decisions=4))
    hyper = PPOHyperparameters(iterations=2, decisions_per_iteration=4,
        minibatch_size=4, epochs=1, evaluation_every=2, evaluation_episodes=1)
    summary = train_arm("ppo_ontology_rgat", seed=8, hyper=hyper,
                        output_dir=tmp_path, config=config)
    repair = summary["relation_activation"]
    assert repair["attempted"] and repair["environment_steps"] == 4
    assert summary["total_environment_steps_including_pretraining"] == 12
    selected = summary["selected_checkpoint"]
    name = selected.get("checkpoint", "checkpoint_best.pt")
    reloaded, payload = load_agent(tmp_path/name, config)
    anchor, _ = load_agent(tmp_path/"checkpoint_best.pt", config)
    assert_raw_preserved(anchor, reloaded)
    assert (tmp_path/"checkpoint_final.pt").exists()
    if repair["changed"]:
        assert name == "checkpoint_relational.pt"
        assert payload["relation_activation"]["guard"]["accepted"]
    else:
        assert name == "checkpoint_best.pt"
