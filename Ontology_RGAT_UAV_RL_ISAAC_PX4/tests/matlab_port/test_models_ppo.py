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


def test_stable_profile_initializes_tracking_prior_without_training_data():
    source = make_contract(3)
    stable = make_contract(3, training_profile="stable")
    assert source.algorithm_hash != stable.algorithm_hash
    assert stable.training.episodes_per_update == 12
    assert stable.training.enforce_target_kl
    model = _model(stable, "ppo", 4)
    high = torch.zeros(stable.observation.dimension)
    high[2] = 2/(2+8)
    high[0] = 1/(1+3)
    high[3] = .5/(.5+10)
    high[-1] = 1.0
    descending = high.clone()
    descending[2] = .05/(.05+8)
    descending[7] = -.2/(.2+1.5)
    high_action = model.action_from_latent(model(high).latent_mean)
    biased = high.clone()
    biased[0] = .2/(.2+3)
    biased[3] = .1/(.1+10)
    unbiased = biased.clone()
    biased[-3] = 1/(1+3)
    biased_action = model.action_from_latent(model(biased).latent_mean)
    unbiased_action = model.action_from_latent(model(unbiased).latent_mean)
    brake_action = model.action_from_latent(model(descending).latent_mean)
    assert high_action[0] > 0
    assert biased_action[0] > unbiased_action[0]
    torch.testing.assert_close(high_action[1], torch.tensor(0.0))
    torch.testing.assert_close(high_action[2], torch.tensor(0.0))
    assert brake_action[2] > 0
    aligned = high.clone()
    aligned[0] = aligned[3] = 0.0
    assert model.action_from_latent(model(aligned).latent_mean)[2] < 0
    stale = aligned.clone()
    stale[2] = .1/(.1+8)
    stale[18] = 3/(3+3)
    assert model.action_from_latent(model(stale).latent_mean)[2] > 0
    handover = aligned.clone()
    handover[-1] = 0.0
    assert model.action_from_latent(model(handover).latent_mean)[2] >= .19


def test_stable_ppo_rejects_an_excessive_post_step_kl():
    contract = make_contract(2, training_profile="stable")
    model = _model(contract, "ppo", 4)
    trainer = DirectPPO(model)
    for group in trainer.actor_optimizer.param_groups:
        group["lr"] = 1.0
    count = 32
    x = torch.randn(count, contract.observation.dimension)
    with torch.no_grad():
        out = model(x)
        latent = out.latent_mean + out.log_std.exp()*torch.randn_like(out.latent_mean)
        old = model.latent_log_prob(latent, out.latent_mean, out.log_std)
    batch = RolloutBatch(x, latent, old, torch.linspace(-3, 3, count), out.value,
                         torch.full((count,), np.exp(-0.1/70)),
                         torch.tensor([False]*31+[True]),
                         torch.tensor([False]*31+[True]), torch.zeros(count))
    stats = trainer.update(batch, generator=torch.Generator().manual_seed(3))
    assert stats["rejected_steps"] > 0
    assert stats["post_step_kl"] > 1.5*contract.training.target_kl


def test_stable_action_transform_enforces_handover_and_stale_track_limits():
    contract = make_contract(3, training_profile="stable")
    model = _model(contract, "ppo", 4)
    x = torch.zeros(contract.observation.dimension)
    x[18] = 10/(10+3)
    x[7] = -.1/(.1+1.5)
    x[-1] = 1.0
    action = model.action_from_latent(torch.tensor([10.0, 10.0, -10.0]), x)
    assert torch.linalg.vector_norm(action[:2]) <= \
        contract.training.tracking_stale_horizontal_cap_m_s2 + 1e-6
    assert action[2] >= contract.training.tracking_stale_vertical_floor_m_s2
    x[7] = .1/(.1+1.5)
    action = model.action_from_latent(torch.tensor([10.0, 10.0, -10.0]), x)
    assert action[2] < contract.training.tracking_stale_vertical_floor_m_s2
    x[18] = 3/(3+3)
    action = model.action_from_latent(torch.tensor([10.0, 10.0, -10.0]), x)
    assert torch.linalg.vector_norm(action[:2]) > \
        contract.training.tracking_stale_horizontal_cap_m_s2
    assert torch.linalg.vector_norm(action[:2]) <= \
        contract.training.tracking_horizontal_cap_m_s2 + 1e-6
    x[-1] = 0.0
    action = model.action_from_latent(torch.tensor([10.0, 10.0, -10.0]), x)
    assert action[2] >= contract.training.tracking_handover_climb_m_s2


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
