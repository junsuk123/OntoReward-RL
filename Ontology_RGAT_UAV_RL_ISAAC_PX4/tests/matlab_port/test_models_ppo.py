import numpy as np
import torch

from ontology_rgat.direct_policy.contracts import make_contract
from ontology_rgat.direct_policy.graph import planar_graph
from ontology_rgat.direct_policy.models import DirectActorCritic
from ontology_rgat.direct_policy.ppo import DirectPPO, RolloutBatch


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
