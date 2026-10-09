import numpy as np
import torch
import json

from ontology_rgat.direct_policy.contracts import make_contract
from ontology_rgat.direct_policy.graph import planar_graph
from ontology_rgat.direct_policy.models import DirectActorCritic
from ontology_rgat.direct_policy.ppo import DirectPPO, RolloutBatch
from ontology_rgat.direct_policy.runner import _model, _rollout


def make_model(method="onto_rgat_ppo"):
    contract = make_contract(2)
    graph = None if method == "ppo" else planar_graph(np.zeros(12))
    return DirectActorCritic(contract, method, graph, seed=4)


def test_actor_and_critic_encoders_are_distinct_and_gradient_isolated():
    model = make_model()
    assert model.actor_encoder is not model.critic_encoder
    x = torch.randn(5, 7, 6)
    output = model(x)
    output.latent_mean.sum().backward()
    assert any(p.grad is not None for p in model.actor_encoder.parameters())
    assert all(p.grad is None for p in model.critic_encoder.parameters())
    model.zero_grad(set_to_none=True)
    model(x).value.sum().backward()
    assert all(p.grad is None for p in model.actor_encoder.parameters())
    assert any(p.grad is not None for p in model.critic_encoder.parameters())


def test_task_hash_is_shared_but_method_hash_is_distinct():
    ppo, rgat = make_model("ppo"), make_model("onto_rgat_ppo")
    assert ppo.contract.task_contract_hash == rgat.contract.task_contract_hash
    assert ppo.method_hash != rgat.method_hash


def test_graph_policy_reads_graph_only_and_relations_receive_gradient():
    model = make_model()
    x = torch.randn(8, 7, 6)
    loss = model(x).latent_mean.square().mean()
    loss.backward()
    assert model.actor[0].in_features == model.contract.training.graph_output_size
    gradient = model.actor_encoder.layer.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert float(gradient.norm()) > 0


def test_vectorized_relation_layer_matches_incoming_edge_reference():
    model = make_model()
    layer = model.actor_encoder.layer
    source = model.actor_encoder.source
    target = model.actor_encoder.target
    relation = model.actor_encoder.relation
    features = torch.randn(9, 7, 6)
    actual = layer(features, source, target, relation)

    projected = torch.einsum("rhi,bni->brnh", layer.weight, features)
    source_state = projected[:, relation, source]
    target_state = projected[:, relation, target]
    relation_state = layer.embedding[relation].unsqueeze(0).expand(len(features), -1, -1)
    joined = torch.cat((source_state, target_state, relation_state), dim=-1)
    raw = (joined * layer.attention[relation].unsqueeze(0)).sum(-1)
    score = torch.where(raw >= 0, raw, 0.2 * raw)
    alpha = torch.zeros_like(score)
    for node in range(features.shape[1]):
        mask = target == node
        alpha[:, mask] = torch.softmax(score[:, mask], dim=1)
    expected = torch.zeros_like(actual)
    for edge in range(source.numel()):
        expected[:, target[edge]] += alpha[:, edge, None] * source_state[:, edge]

    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)


def test_latent_distribution_and_tanh_action_are_consistent():
    model = make_model("ppo")
    x = torch.zeros(12)
    action, latent, logp, _ = model.act(x, deterministic=True)
    output = model(x)
    torch.testing.assert_close(latent, output.latent_mean)
    torch.testing.assert_close(action, torch.tanh(latent)*model.action_scale)
    torch.testing.assert_close(logp, model.latent_log_prob(
        latent, output.latent_mean, output.log_std))


def test_one_ppo_update_is_finite_and_versioned():
    model = make_model("ppo")
    trainer = DirectPPO(model)
    count = 12
    x = torch.randn(count, 12)
    with torch.no_grad():
        out = model(x)
        latent = out.latent_mean + 0.1*torch.randn_like(out.latent_mean)
        old = model.latent_log_prob(latent, out.latent_mean, out.log_std)
    batch = RolloutBatch(x, latent, old, torch.randn(count), out.value,
                         torch.full((count,), np.exp(-0.1/70)),
                         torch.tensor([False]*11+[True]),
                         torch.tensor([False]*11+[True]),
                         torch.zeros(count), policy_version=0)
    stats = trainer.update(batch, generator=torch.Generator().manual_seed(3))
    assert stats["policy_version"] == 1
    assert np.isfinite([stats["actor_loss"], stats["critic_loss"], stats["entropy"]]).all()


def test_replay_rollout_uses_recorded_transition_durations(tmp_path):
    fixture = tmp_path / "replay.json"
    fixture.write_text(json.dumps({
        "observations": [np.zeros(12).tolist() for _ in range(3)],
        "rewards": [0.0, 1.0, 2.0],
        "statuses": ["RUNNING", "RUNNING", "TASK_TIMEOUT"],
        "transition_s": [0.1, 0.2, 0.4],
    }))
    contract = make_contract(2, backend="replay")
    model = _model(contract, "ppo", 7)
    from ontology_rgat.direct_policy.backends import backend_factory
    batch, episode = _rollout(
        contract, model, 3, deterministic=True, max_steps=4,
        policy_version=5, backend_factory=backend_factory(
            contract, replay_path=fixture))
    torch.testing.assert_close(
        batch.discount,
        torch.tensor([np.exp(-0.2/70), np.exp(-0.4/70)], dtype=torch.float32))
    assert batch.policy_version == 5
    assert episode["backend"] == "replay"
    assert np.isclose(episode["simulated_seconds"], 0.6)
