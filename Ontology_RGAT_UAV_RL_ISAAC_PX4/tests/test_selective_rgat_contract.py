from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from ontology_rgat.contracts import CausalObservationPacket, load_observation_registry
from ontology_rgat.contracts.signature import CheckpointSignature
from ontology_rgat.evaluation.selection import robust_checkpoint_score
from ontology_rgat.ppo.selective_graph_encoder import SelectiveGraphStateEncoder
from ontology_rgat.reward_modes.two_term import TwoTermReward, capture_signal
from ontology_rgat.rgat.selective_pretrain import episode_split, training_normalization
from ontology_rgat.rgat.selective_state import (
    ADAPTIVE_RELATIONS, INVARIANT_RELATIONS, SELECTIVE_GOAL_NODE,
    SELECTIVE_NODE_NAMES, SELECTIVE_RELATION_NAMES, build_selective_graph,
    unreachable_or_too_distant_nodes)


def _packet():
    registry = load_observation_registry()
    fields = {}
    for field in registry.fields:
        fields[field["name"]] = np.zeros(int(field["size"]), dtype=np.float32)
    fields.update({
        "keypoint_visibility": np.ones(6), "keypoint_confidence": [0.9],
        "visible_fraction": [1.0], "measurement_valid": [1.0],
        "fov_margin": [0.8], "attitude_quaternion": [1.0, 0.0, 0.0, 0.0],
    })
    return registry, CausalObservationPacket.from_fields(
        fields, timestamp_s=1.25, registry=registry)


def test_registry_and_graph_are_causal_and_connected():
    registry, packet = _packet()
    assert registry.dimension == packet.values.size
    assert unreachable_or_too_distant_nodes() == ()
    assert tuple(inspect.signature(build_selective_graph).parameters) == (
        "packet", "registry")
    graph = build_selective_graph(packet, registry=registry)
    assert graph.X[0, graph.goal_node] == 0.0
    assert graph.node_names[graph.goal_node] == SELECTIVE_GOAL_NODE
    assert set(SELECTIVE_RELATION_NAMES) == set(ADAPTIVE_RELATIONS + INVARIANT_RELATIONS)
    assert len(SELECTIVE_NODE_NAMES) == graph.X.shape[1]


def test_relation_gates_start_at_identity_and_are_bounded():
    registry, packet = _packet()
    graph = build_selective_graph(packet, registry=registry)
    X = torch.as_tensor(graph.X.T[None], dtype=torch.float32)
    observation = torch.as_tensor(packet.values[None], dtype=torch.float32)
    encoder = SelectiveGraphStateEncoder(
        observation_dim=registry.dimension, hidden_dim=8, graph_dim=7, seed=3)
    embedding, gates = encoder(X, observation, return_gates=True)
    torch.testing.assert_close(gates, torch.ones_like(gates))
    assert embedding.shape == (1, 7)
    with torch.no_grad():
        encoder.gate.linear.weight.fill_(10.0)
    gates = encoder.relation_gates(observation)
    assert torch.all(gates[..., :4] >= 0.65)
    assert torch.all(gates[..., :4] <= 1.35)
    torch.testing.assert_close(gates[..., 4:], torch.ones_like(gates[..., 4:]))


def test_only_gate_parameters_are_trainable_and_receive_gradient():
    registry, packet = _packet()
    graph = build_selective_graph(packet, registry=registry)
    X = torch.as_tensor(graph.X.T[None], dtype=torch.float32)
    observation = torch.as_tensor(packet.values[None], dtype=torch.float32)
    encoder = SelectiveGraphStateEncoder(
        observation_dim=registry.dimension, hidden_dim=8, graph_dim=6, seed=9)
    trainable = {name for name, value in encoder.named_parameters()
                 if value.requires_grad}
    assert trainable == {"gate.linear.weight", "gate.linear.bias"}
    loss = encoder(X, observation).square().sum()
    loss.backward()
    assert encoder.gate.linear.weight.grad is not None
    frozen = [value for name, value in encoder.named_parameters()
              if not name.startswith("gate.")]
    assert all(value.grad is None for value in frozen)
    optimizer = torch.optim.Adam(encoder.gate_parameters(), lr=1e-3)
    encoder.assert_optimizer_safe(optimizer)
    with pytest.raises(ValueError, match="frozen"):
        encoder.assert_optimizer_safe(torch.optim.Adam(encoder.parameters(), lr=1e-3))


def test_gate_gradient_matches_central_difference():
    registry, packet = _packet()
    graph = build_selective_graph(packet, registry=registry)
    X = torch.as_tensor(graph.X.T[None], dtype=torch.float32)
    observation = torch.as_tensor(packet.values[None], dtype=torch.float32)
    encoder = SelectiveGraphStateEncoder(
        observation_dim=registry.dimension, hidden_dim=6, graph_dim=4, seed=12)
    weight = encoder.gate.linear.weight
    encoder(X, observation).sum().backward()
    analytic = float(weight.grad[0, 18])
    epsilon = 1e-3
    with torch.no_grad():
        original = float(weight[0, 18])
        weight[0, 18] = original + epsilon
        plus = float(encoder(X, observation).sum())
        weight[0, 18] = original - epsilon
        minus = float(encoder(X, observation).sum())
        weight[0, 18] = original
    numerical = (plus - minus) / (2.0 * epsilon)
    assert analytic == pytest.approx(numerical, rel=2e-2, abs=2e-4)


def test_inference_does_not_create_gradients_or_update_gates():
    registry, packet = _packet()
    graph = build_selective_graph(packet, registry=registry)
    X = torch.as_tensor(graph.X.T[None], dtype=torch.float32)
    observation = torch.as_tensor(packet.values[None], dtype=torch.float32)
    encoder = SelectiveGraphStateEncoder(
        observation_dim=registry.dimension, hidden_dim=6, graph_dim=4, seed=2)
    before = {name: value.detach().clone() for name, value in encoder.named_parameters()}
    with torch.no_grad():
        encoder(X, observation)
    assert all(value.grad is None for value in encoder.parameters())
    for name, value in encoder.named_parameters():
        torch.testing.assert_close(value, before[name])


def test_two_term_reward_rejects_positive_failed_contact_proximity():
    reward = TwoTermReward()
    value, parts = reward(
        normalized_longitudinal_error=0.0, distance_m=0.0,
        previous_distance_m=0.1, dt_s=0.1, failed_contact=True, terminal=True)
    assert set(parts) == {"capture", "distance", "distance_rate", "proximity"}
    assert parts["capture"] < 0 and parts["distance"] < 0 and value < 0
    assert capture_signal(1.0) == pytest.approx(-0.5)
    assert capture_signal(2.0) < -0.5


def test_pretraining_split_and_normalization_never_read_validation_or_test():
    ids = np.repeat(np.arange(10), 2)
    split = episode_split(ids, seed=4)
    assert not set(split["train"]) & set(split["validation"])
    assert not set(split["train"]) & set(split["test"])
    values = np.arange(ids.size * 2, dtype=float).reshape(ids.size, 2)
    mean, std = training_normalization(values, ids, split)
    mask = np.isin(ids, split["train"])
    np.testing.assert_allclose(mean, values[mask].mean(0))
    np.testing.assert_allclose(std, values[mask].std(0))


def test_checkpoint_signature_rejects_any_contract_mismatch():
    signature = CheckpointSignature(*[str(index) for index in range(11)])
    stored = dict(signature.__dict__)
    signature.assert_compatible(stored)
    stored["reward_config_hash"] = "changed"
    with pytest.raises(ValueError, match="reward_config_hash"):
        signature.assert_compatible(stored)


def test_checkpoint_selection_is_reward_independent_and_safety_dominant():
    safe = [{"scenario": "a", "strict_success": 1, "nominal": True,
             "capture_rate": .8, "unsafe_contact": 0, "return": -100}]
    unsafe = [{"scenario": "a", "strict_success": 0, "nominal": True,
               "capture_rate": 1, "unsafe_contact": 1, "return": 1e6}]
    assert robust_checkpoint_score(safe)[0] > robust_checkpoint_score(unsafe)[0]
