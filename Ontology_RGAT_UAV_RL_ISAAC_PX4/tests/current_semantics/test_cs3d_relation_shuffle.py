"""relation-shuffle/1 (relation_control.py, guide 8.4, status doc 19).

A shuffle is the typed R-GAT with its fourteen semantic edges re-labelled: same
topology, parameters and initialisation, different weight sharing.  It must never
alias the typed arm, and the registered v1 contract must not move.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from ontology_rgat.current_semantics import relation_control as rc
from ontology_rgat.current_semantics.contracts import (DEFAULT_CONFIG_PATH, NODE_ORDER,
                                                       TASK_CONFIG_PATHS, contract_manifest,
                                                       load_config)
from ontology_rgat.current_semantics.development import CELLS
from ontology_rgat.current_semantics.graph import (SEMANTIC_EDGES, TYPED_GRAPH,
                                                   reachable_within)
from ontology_rgat.current_semantics.models import (CurrentActorCritic, build_model,
                                                    load_checkpoint, save_checkpoint)
from ontology_rgat.current_semantics.pilot import _model

from test_cs3d_models import batch

CONFIG = load_config(DEFAULT_CONFIG_PATH)
PLAN, DEVELOPMENT = rc.load_plan(rc.DEFAULT_PLAN_PATH, CONFIG)
GRAPHS = rc.graphs_of(PLAN, DEVELOPMENT)


def test_the_v1_contract_does_not_move():
    manifest = contract_manifest(CONFIG)
    assert manifest["graph_hash"].startswith("66e4881a316372ad")
    assert manifest["algorithm_hash"].startswith("70ebb7e0833ef441")
    assert manifest["task_hash"].startswith("4c903c992b94f586")


def test_ten_declared_shuffles_keep_the_topology_and_break_every_typed_relation():
    assert len(GRAPHS) == 10
    assert [g["shuffle_seed"] for g in GRAPHS.values()] == [
        7200, 7201, 7210, 7211, 7220, 7221, 7230, 7231, 7240, 7241]
    partitions = set()
    for graph in GRAPHS.values():
        spec, _ = rc.shuffled_graph(graph["shuffle_seed"])
        assert [(s, t) for s, t, _ in spec.edges] == [(s, t) for s, t, _ in TYPED_GRAPH.edges]
        assert spec.relation_counts() == TYPED_GRAPH.relation_counts()
        assert reachable_within(spec, NODE_ORDER[-1], 2) == set(NODE_ORDER)
        groups = rc._groups(spec.edges[:len(SEMANTIC_EDGES)])
        assert not groups & rc.TYPED_GROUPS
        partitions.add(groups)
        assert rc.shuffled_graph(graph["shuffle_seed"])[0].graph_hash == spec.graph_hash
        assert 0.0 < graph["rand_index_vs_typed"] < 1.0
    assert len(partitions) == 10
    assert rc.rand_index(TYPED_GRAPH) == 1.0


def test_a_shuffle_is_the_typed_arm_with_relabelled_edges():
    arm = GRAPHS["10/0"]["arm"]
    typed = CurrentActorCritic(rc.TYPED_ARM, CONFIG, seed=10)
    shuffled = build_model(arm, CONFIG, seed=10)
    assert shuffled.arm == arm and shuffled.parameter_report() == typed.parameter_report()
    assert all(torch.equal(a, b) for a, b in zip(typed.state_dict().values(),
                                                  shuffled.state_dict().values()))
    assert torch.equal(typed.encoder.source, shuffled.encoder.source)
    assert not torch.equal(typed.encoder.relation, shuffled.encoder.relation)
    _, _, nodes = batch(7)
    assert not torch.allclose(typed.graph_readout(nodes), shuffled.graph_readout(nodes))
    registered = build_model("raw_ppo", CONFIG, seed=3)
    reference = CurrentActorCritic("raw_ppo", CONFIG, seed=3)
    assert type(registered) is CurrentActorCritic and all(
        torch.equal(a, b) for a, b in zip(registered.state_dict().values(),
                                          reference.state_dict().values()))


def test_the_worker_path_rebuilds_the_shuffle_from_the_arm_name():
    arm = GRAPHS["12/1"]["arm"]
    trained = build_model(arm, CONFIG, seed=12)
    rebuilt = _model(CONFIG, arm, trained.state_dict())
    assert rebuilt.arm == arm
    assert torch.equal(rebuilt.encoder.relation, trained.encoder.relation)


def test_checkpoints_never_alias_typed_and_shuffled(tmp_path):
    manifest = contract_manifest(CONFIG)
    first, second = GRAPHS["10/0"]["arm"], GRAPHS["10/1"]["arm"]
    path = save_checkpoint(tmp_path / "s.pt", build_model(first, CONFIG, seed=10),
                           manifest=manifest)
    load_checkpoint(path, build_model(first, CONFIG, seed=0), manifest=manifest)
    for other in (rc.TYPED_ARM, second):
        with pytest.raises(ValueError, match="arm"):
            load_checkpoint(path, build_model(other, CONFIG, seed=0), manifest=manifest)
    typed = save_checkpoint(tmp_path / "t.pt", CurrentActorCritic(rc.TYPED_ARM, CONFIG, seed=10),
                            manifest=manifest)
    with pytest.raises(ValueError, match="arm"):
        load_checkpoint(typed, build_model(first, CONFIG, seed=0), manifest=manifest)


@pytest.mark.parametrize("arm, message", [
    ("ontology_rgat_shuffled_ppo:7200:0000000000000000", "no longer draws"),
    ("ontology_rgat_shuffled_ppo:x:da82671ae5d74c7a", "unknown arm"),
    ("semantic_gat_ppo", "unknown arm"),
])
def test_a_tampered_or_unknown_arm_is_refused(arm, message):
    with pytest.raises(ValueError, match=message):
        build_model(arm, CONFIG, seed=0)


def _variant(tmp_path, **edit):
    data = yaml.safe_load(Path(rc.DEFAULT_PLAN_PATH).read_text(encoding="utf-8"))
    data.update(edit)
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.mark.parametrize("edit, message", [
    ({"protocol": "relation-shuffle/2"}, "protocol must be"),
    ({"base_protocol": "development/2"}, "protocol must be"),
    ({"extra": 1}, "exactly the keys"),
    ({"shuffles_per_training_seed": 0}, ">= 1"),
])
def test_the_plan_loader_refuses_a_malformed_plan(tmp_path, edit, message):
    with pytest.raises(ValueError, match=message):
        rc.load_plan(_variant(tmp_path, **edit), CONFIG)


def test_the_control_belongs_to_task_v5_only():
    with pytest.raises(ValueError, match="extends development/1"):
        rc.load_plan(rc.DEFAULT_PLAN_PATH, load_config(TASK_CONFIG_PATHS["spatial-current-task/6"]))


def test_registration_needs_a_complete_development_1(tmp_path):
    status, payload = rc.register(DEFAULT_CONFIG_PATH, rc.DEFAULT_PLAN_PATH, tmp_path)
    assert status == 3 and "not registered" in payload["reason"]
    status, payload = rc.run(DEFAULT_CONFIG_PATH, rc.DEFAULT_PLAN_PATH, tmp_path)
    assert status == 3 and payload["status"] == "REFUSED"


def _summary(success: float, unsafe: float = 0.1) -> dict:
    return {"cells": {c: {"success": success, "unsafe": unsafe} for c in CELLS},
            "validation": {"deterministic_success": 0.9}}


def test_the_verdict_follows_the_registered_interval():
    seeds = DEVELOPMENT.training_seeds
    untyped = {s: _summary(0.8) for s in seeds}
    clear = rc.judge(DEVELOPMENT, {s: _summary(0.9 + 0.01 * i) for i, s in enumerate(seeds)},
                     untyped, {s: [_summary(0.80), _summary(0.82)] for s in seeds})
    assert clear["semantic_placement"] == "better"
    assert clear["primary"]["mean"] == pytest.approx(0.11)
    assert clear["metrics"]["all_cells"]["within_seed_shuffle_spread"] == pytest.approx(0.02)
    noisy = rc.judge(DEVELOPMENT, {s: _summary(v) for s, v in zip(seeds, (.9, .7, .9, .7, .85))},
                     untyped, {s: [_summary(0.8), _summary(0.8)] for s in seeds})
    assert noisy["semantic_placement"] == "not established"
    worse = rc.judge(DEVELOPMENT, {s: _summary(0.6 + 0.01 * i) for i, s in enumerate(seeds)},
                     untyped, {s: [_summary(0.8), _summary(0.8)] for s in seeds})
    assert worse["semantic_placement"] == "worse"
    assert np.isclose(worse["metrics"]["all_cells"]["typed_minus_untyped"]["mean"], -0.18)
