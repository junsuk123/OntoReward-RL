"""Budget-matched actor-critics of spatial_current_semantics_v1 (guide 8.3-8.4)."""
import copy
import math

import numpy as np
import pytest
import torch
from torch.nn import functional as F
import yaml

from ontology_rgat.current_semantics import contracts, models
from ontology_rgat.current_semantics.contracts import (
    ARM_ACTOR_INPUT, ARMS, DEFAULT_CONFIG_PATH, DESIGN_PARAMETER_TOTALS, ENCODER_RELATIONS,
    NODE_ORDER, contract_manifest, design_parameter_counts, load_config)
from ontology_rgat.current_semantics.features import (NODE_FEATURES, NODE_ID_COLUMN,
                                                      build_feature_batch, build_features)
from ontology_rgat.current_semantics.graph import (SELF, TYPED_GRAPH, GraphSpec, graph_for_arm,
                                                   reachable_within)
from ontology_rgat.current_semantics.models import (
    CurrentActorCritic, GraphEncoder, RelationalAttentionLayer, component_generator,
    load_checkpoint, save_checkpoint, tensors_from_batch, tensors_from_frames)
from ontology_rgat.current_semantics.observation import CurrentPhysical, encode


CONFIG = load_config()
MANIFEST = contract_manifest(CONFIG)
GRAPH_ARMS = tuple(ENCODER_RELATIONS)
Q, G, M, A, P, O, C, L = NODE_ORDER
PRIMITIVES, DERIVED = (Q, G, M, A), (P, O, C)
MEASURED = {"raw_ppo": (0, 2753, 4022), "semantic_mlp_ppo": (0, 2739, 4008),
            "untyped_gnn_ppo": (306, 2445, 4020), "ontology_rgat_ppo": (566, 2181, 4016)}
MAXIMUM = torch.tensor(CONFIG.action.maximum_m_s2, dtype=torch.float32)


def physical(**overrides):
    values = dict(ex=0.0, ey=0.0, h=2.0, vrx=0.0, vry=0.0, vgx=0.0, vgy=0.0, vz=0.0,
                  roll=0.0, pitch=0.0, yaw=0.0, p=0.0, q=0.0, r=0.0,
                  vision_updated=True, vision_age_s=0.05, nav_valid=True, nav_age_s=0.0,
                  estimate_valid=True)
    values.update(overrides)
    return CurrentPhysical(**values)


def random_physical(rng, **overrides):
    values = dict(ex=rng.uniform(-6, 6), ey=rng.uniform(-6, 6), h=rng.uniform(-1, 12),
                  vrx=rng.uniform(-4, 4), vry=rng.uniform(-4, 4),
                  vgx=rng.uniform(-4, 4), vgy=rng.uniform(-4, 4), vz=rng.uniform(-2, 2),
                  roll=rng.uniform(-0.6, 0.6), pitch=rng.uniform(-0.6, 0.6),
                  yaw=rng.uniform(-3.1, 3.1), p=rng.uniform(-3, 3), q=rng.uniform(-3, 3),
                  r=rng.uniform(-3, 3), vision_updated=bool(rng.integers(2)),
                  vision_age_s=rng.uniform(0, 6), nav_valid=bool(rng.integers(2)),
                  nav_age_s=rng.uniform(0, 6), estimate_valid=bool(rng.random() > 0.2))
    values.update(overrides)
    return physical(**values)


def arrays(seed, count=24):
    """Real decisions: encoded random states, valid and invalid estimates mixed."""
    rng = np.random.default_rng(seed)
    raw = np.stack([encode(random_physical(rng)) for _ in range(count)])
    flat, nodes = build_feature_batch(raw, CONFIG)
    return raw, flat, nodes


def batch(seed, count=24):
    return tensors_from_batch(*arrays(seed, count))


def model(arm, seed=0, config=CONFIG):
    return CurrentActorCritic(arm, config, seed=seed)


def state(module):
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def assert_same_state(first, second):
    assert first.keys() == second.keys()
    for key in first:
        assert torch.equal(first[key], second[key]), key


# ------------------------------------------------------------ parameters

def test_measured_trainable_counts_are_the_registered_design_exactly():
    design = design_parameter_counts(CONFIG.model)
    for arm in ARMS:
        net = model(arm)
        report = net.parameter_report()
        total = MEASURED[arm][2]
        assert report == design[arm], arm
        assert (report["encoder"], report["actor_head"], report["total"]) == MEASURED[arm]
        assert report["critic"] + report["log_std"] == 1269 and report["log_std"] == 3
        assert report["total"] == DESIGN_PARAMETER_TOTALS[arm]
        assert all(p.requires_grad for p in net.parameters())
        assert sum(p.numel() for p in net.parameters()) == total
    assert max(DESIGN_PARAMETER_TOTALS.values()) - min(DESIGN_PARAMETER_TOTALS.values()) == 14


def test_relational_layer_and_encoder_have_exactly_the_registered_shapes():
    for spec in (TYPED_GRAPH, graph_for_arm("untyped_gnn_ppo")):
        relations = len(spec.relations)
        layer = RelationalAttentionLayer(8, relations, generator=component_generator(0, "x"))
        assert {name: tuple(p.shape) for name, p in layer.named_parameters()} == {
            "weight": (relations, 8, 8), "relation_bias": (relations,), "att_src": (8,),
            "att_dst": (8,), "bias": (8,)}
        assert sum(p.numel() for p in layer.parameters()) == 65 * relations + 24
        encoder = GraphEncoder(spec, dim=8, hops=2, seed=0)
        shapes = {name: tuple(p.shape) for name, p in encoder.named_parameters()}
        assert shapes["input.weight"] == (8, 10) and shapes["input.bias"] == (8,)
        assert shapes["readout.weight"] == (8, 8) and "readout.bias" not in shapes
        assert sum(p.numel() for p in encoder.parameters()) == 176 + 65 * relations
        assert set(encoder.state_dict()) == set(shapes), "edge buffers must not be checkpointed"
    assert {arm: model(arm).parameter_report()["encoder"] for arm in GRAPH_ARMS} == {
        "untyped_gnn_ppo": 306, "ontology_rgat_ppo": 566}


def test_optimizer_groups_cover_every_trainable_parameter_exactly_once():
    for arm in ARMS:
        net = model(arm)
        groups = net.optimizer_groups()
        names = [group["name"] for group in groups]
        expected = ["encoder", "actor_head", "log_std", "critic"]
        assert names == (expected if net.is_graph else expected[1:]), arm
        ids = [id(p) for group in groups for p in group["params"]]
        assert len(ids) == len(set(ids))
        assert set(ids) == {id(p) for p in net.parameters() if p.requires_grad}
        report = net.parameter_report()
        for group in groups:
            assert sum(p.numel() for p in group["params"]) == report[group["name"]]
        optimizer = torch.optim.Adam(groups, lr=1e-3)
        assert [g["name"] for g in optimizer.param_groups] == names


# ------------------------------------------------------------- the layer

def reference_layer(layer, h, spec):
    """Per-node, per-edge loop of the registered equations, in float64."""
    weight, att_src, att_dst, relation_bias, bias = (
        p.detach().double() for p in (layer.weight, layer.att_src, layer.att_dst,
                                      layer.relation_bias, layer.bias))
    out = torch.empty_like(h)
    for i, node in enumerate(spec.nodes):
        incoming = [(spec.nodes.index(s), spec.relations.index(r)) for s, t, r in spec.edges
                    if t == node]
        if not incoming:
            out[i] = torch.tanh(bias)
            continue
        messages = [weight[r] @ h[j] for j, r in incoming]
        scores = torch.stack([F.leaky_relu(att_src @ m + att_dst @ h[i] + relation_bias[r], 0.2)
                              for m, (_, r) in zip(messages, incoming)])
        alpha = torch.softmax(scores, dim=0)
        out[i] = torch.tanh(sum(a * m for a, m in zip(alpha, messages)) + bias)
    return out


def test_vectorised_layer_is_the_per_edge_softmax_over_incoming_edges():
    isolated = GraphSpec(version="test", nodes=("a", "b", "c"), relations=("r", SELF),
                         edges=(("a", "b", "r"), ("a", "a", SELF), ("b", "b", SELF)),
                         readout_node="b")
    for spec in (TYPED_GRAPH, graph_for_arm("untyped_gnn_ppo"), isolated):
        generator = torch.Generator().manual_seed(11)
        layer = RelationalAttentionLayer(8, len(spec.relations), generator=generator)
        with torch.no_grad():
            for p in layer.parameters():
                p.copy_(torch.randn(p.shape, generator=generator))
        layer = copy.deepcopy(layer).double()
        h = torch.randn((5, len(spec.nodes), 8), generator=generator, dtype=torch.float64)
        source, target, relation = (torch.as_tensor(i) for i in spec.edge_index())
        out = layer(h, source, target, relation)
        for sample in range(h.shape[0]):
            torch.testing.assert_close(out[sample], reference_layer(layer, h[sample], spec),
                                       rtol=0, atol=1e-12)
        out.sum().backward()
        assert all(torch.isfinite(p.grad).all() for p in layer.parameters())


def test_hops_are_reach_primitives_affect_the_readout_only_through_the_second_hop():
    raw, flat, nodes = batch(1)
    for arm in GRAPH_ARMS:
        spec = graph_for_arm(arm)
        assert reachable_within(spec, L, 1) == set(DERIVED) | {L}
        two = model(arm).encoder
        one = GraphEncoder(spec, dim=8, hops=1, seed=0)
        assert_same_state(state(two), state(one))
        with torch.no_grad():
            base_two, base_one = two(nodes), one(nodes)
            for node in PRIMITIVES + DERIVED:
                row = NODE_ORDER.index(node)
                for slot in range(len(NODE_FEATURES[node])):
                    moved = nodes.clone()
                    moved[:, row, slot] += 0.25
                    change_two = (two(moved) - base_two).abs().amax(dim=-1)
                    change_one = (one(moved) - base_one).abs().amax(dim=-1)
                    assert torch.all(change_two > 0), (arm, node, slot)
                    if node in PRIMITIVES:
                        assert torch.all(change_one == 0), (arm, node, slot)
                    else:
                        assert torch.all(change_one > 0), (arm, node, slot)


def test_padding_and_landing_context_slots_are_read_unmasked_and_refused_at_the_boundary():
    """The shared projection reads all ten columns of every row: a non-zero
    padding value WOULD move the readout.  Padding is a contract of the input,
    enforced by ``tensors_from_batch``, not masked inside the model."""
    raw_np, flat_np, nodes_np = arrays(2)
    raw, flat, nodes = tensors_from_batch(raw_np, flat_np, nodes_np)
    padding = [(row, slot) for row, node in enumerate(NODE_ORDER)
               for slot in range(len(NODE_FEATURES[node]), NODE_ID_COLUMN)]
    assert (NODE_ORDER.index(L), 0) in padding and len(padding) == 8 * 9 - 44
    for arm in GRAPH_ARMS:
        net = model(arm)
        with torch.no_grad():
            base = net.graph_readout(nodes)
            for row, slot in padding + [(NODE_ORDER.index(Q), NODE_ID_COLUMN)]:
                moved = nodes.clone()
                moved[:, row, slot] += 0.25
                assert torch.all((net.graph_readout(moved) - base).abs().amax(-1) > 0)
    for row, slot in padding[::5] + [(NODE_ORDER.index(L), NODE_ID_COLUMN)]:
        corrupt = nodes_np.copy()
        corrupt[:, row, slot] += 0.25
        with pytest.raises(ValueError):
            tensors_from_batch(raw_np, flat_np, corrupt)


# ---------------------------------------------------------------- arms

def test_each_arm_reads_only_its_registered_actor_input():
    raw, flat, nodes = batch(3)
    other_raw, other_flat, other_nodes = batch(4)
    reads = {"raw21": {"raw"}, "flat44": {"flat"}, "raw21+graph": {"raw", "nodes"}}
    for arm in ARMS:
        net = model(arm)
        with torch.no_grad():
            base = net(raw, flat, nodes)
            assert base.shape == (raw.shape[0], 3)
            for name in ("raw", "flat", "nodes"):
                inputs = {"raw": raw, "flat": flat, "nodes": nodes}
                inputs[name] = {"raw": other_raw, "flat": other_flat, "nodes": other_nodes}[name]
                moved = net(inputs["raw"], inputs["flat"], inputs["nodes"])
                assert torch.equal(moved, base) is (name not in reads[ARM_ACTOR_INPUT[arm]]), \
                    (arm, name)
    with pytest.raises(ValueError):
        model("raw_ppo").graph_readout(nodes)
    with pytest.raises(ValueError):
        model("ontology_rgat_ppo")(raw[:3], flat, nodes)
    with pytest.raises(ValueError):
        model("raw_ppo")(raw, flat, nodes[:, :7])


def test_critic_reads_raw21_only_and_is_bit_identical_across_arms_for_a_seed():
    raw, flat, nodes = batch(5)
    _, other_flat, other_nodes = batch(6)
    critics = {arm: state(model(arm, seed=9).critic) for arm in ARMS}
    for arm in ARMS:
        assert_same_state(critics[arm], critics["raw_ppo"])
        net = model(arm, seed=9)
        with torch.no_grad():
            value = net.act(raw, flat, nodes, deterministic=True)["value"]
            moved = net.act(raw, other_flat, other_nodes, deterministic=True)["value"]
            assert value.shape == (raw.shape[0],)
            assert torch.equal(value, moved) and torch.equal(value, net.value(raw))
            assert torch.equal(value, model("raw_ppo", seed=9).value(raw))
            assert not torch.equal(value, net.value(batch(7)[0]))
    with pytest.raises(AssertionError):
        assert_same_state(critics["raw_ppo"], state(model("raw_ppo", seed=10).critic))


def test_same_seed_same_weights_and_construction_never_touches_the_global_rng():
    raw, flat, nodes = batch(8)
    torch.manual_seed(1234)
    before = torch.get_rng_state()
    for arm in ARMS:
        first, second, other = model(arm, seed=5), model(arm, seed=5), model(arm, seed=6)
        assert torch.equal(torch.get_rng_state(), before), arm
        assert all(torch.isfinite(p).all() for p in first.parameters())
        assert_same_state(state(first), state(second))
        with torch.no_grad():
            assert torch.equal(first(raw, flat, nodes), second(raw, flat, nodes))
        for (name, p), q in zip(first.named_parameters(), other.parameters()):
            constant = name in ("log_std", "encoder.layer.relation_bias", "encoder.layer.bias",
                                "actor_head.2.bias")  # zeroed output bias (actor_output_init_scale)
            assert torch.equal(p, q) is constant, (arm, name)
        assert torch.equal(first.log_std.detach(),
                           torch.full((3,), CONFIG.model.initial_log_std))
    typed, untyped = model("ontology_rgat_ppo", seed=5), model("untyped_gnn_ppo", seed=5)
    for part in ("input", "readout"):
        assert_same_state(state(getattr(typed.encoder, part)),
                          state(getattr(untyped.encoder, part)))
    with pytest.raises(ValueError):
        CurrentActorCritic("raw_ppo", CONFIG, seed=-1)
    with pytest.raises(ValueError):
        CurrentActorCritic("ppo", CONFIG, seed=0)


# -------------------------------------------------------------- policy

def test_action_is_a_max_tanh_latent_exactly_and_never_exceeds_a_max():
    raw, flat, nodes = batch(9, 64)
    for arm in ARMS:
        net = model(arm)
        for log_std in (CONFIG.model.initial_log_std, 3.0):
            with torch.no_grad():
                net.log_std.fill_(log_std)
                out = net.act(raw, flat, nodes, deterministic=False,
                              generator=torch.Generator().manual_seed(1))
            assert torch.equal(out["action"], MAXIMUM * torch.tanh(out["latent"]))
            assert torch.all(out["action"].abs() <= MAXIMUM)
        assert torch.any(out["action"].abs() == MAXIMUM), "sigma e^3 must saturate tanh"


def test_log_prob_and_entropy_are_the_latent_gaussian_of_the_mean():
    raw, flat, nodes = batch(10)
    for arm in ARMS:
        net = model(arm)
        with torch.no_grad():
            net.log_std.copy_(torch.tensor([-1.3, -0.4, 0.2]))
            mean = net(raw, flat, nodes)
            sigma = net.log_std.exp()
            out = net.act(raw, flat, nodes, deterministic=False,
                          generator=torch.Generator().manual_seed(2))
            noise = torch.randn(mean.shape, generator=torch.Generator().manual_seed(2))
            assert torch.equal(out["latent"], mean + sigma * noise)
            normal = torch.distributions.Normal(mean, sigma)
            torch.testing.assert_close(out["log_prob"], normal.log_prob(out["latent"]).sum(-1),
                                       rtol=0, atol=1e-5)
            torch.testing.assert_close(net.log_prob(out["latent"], raw, flat, nodes),
                                       out["log_prob"], rtol=0, atol=0)
            torch.testing.assert_close(net.entropy(), normal.entropy().sum(-1)[0],
                                       rtol=0, atol=1e-6)
            fixed = net.act(raw, flat, nodes, deterministic=True)
            assert torch.equal(fixed["latent"], mean)
            assert torch.equal(fixed["action"], MAXIMUM * torch.tanh(mean))
            torch.testing.assert_close(fixed["log_prob"], normal.log_prob(mean).sum(-1),
                                       rtol=0, atol=1e-5)
        with pytest.raises(TypeError):
            net.act(raw, flat, nodes, deterministic=1)


def test_every_parameter_and_every_relation_slice_receives_gradient():
    raw, flat, nodes = batch(11)
    for arm in ARMS:
        net = model(arm)
        with torch.no_grad():
            latent = net.act(raw, flat, nodes, deterministic=False,
                             generator=torch.Generator().manual_seed(3))["latent"]
        loss = -net.log_prob(latent, raw, flat, nodes).mean() + net.value(raw).pow(2).mean() \
            - 0.01 * net.entropy()
        loss.backward()
        for name, p in net.named_parameters():
            assert p.grad is not None and torch.any(p.grad != 0), (arm, name)

    for arm in GRAPH_ARMS:
        net = model(arm)
        net(raw, flat, nodes).pow(2).sum().backward()
        encoder = net.encoder
        assert torch.all(encoder.input.weight.grad.abs().sum(0) > 0), "every input column"
        assert torch.any(encoder.readout.weight.grad != 0)
        for index, relation in enumerate(encoder.spec.relations):
            assert torch.any(encoder.layer.weight.grad[index] != 0), (arm, relation)
        for name in ("att_src", "att_dst", "relation_bias", "bias"):
            assert torch.any(getattr(encoder.layer, name).grad != 0), (arm, name)
        assert all(p.grad is None for p in net.critic.parameters())
        assert net.log_std.grad is None


def test_batched_forward_equals_per_sample_forward():
    raw, flat, nodes = batch(12, 16)
    for arm in ARMS:
        net = model(arm)
        with torch.no_grad():
            mean, value = net(raw, flat, nodes), net.value(raw)
            readout = net.graph_readout(nodes) if net.is_graph else None
            for i in range(raw.shape[0]):
                one = slice(i, i + 1)
                torch.testing.assert_close(net(raw[one], flat[one], nodes[one]), mean[one],
                                           rtol=0, atol=1e-6)
                torch.testing.assert_close(net.value(raw[one]), value[one], rtol=0, atol=1e-6)
                if readout is not None:
                    torch.testing.assert_close(net.graph_readout(nodes[one]), readout[one],
                                               rtol=0, atol=1e-6)


# ------------------------------------------------------------- tensors

def test_tensors_from_frames_copy_the_frames_into_float32():
    rng = np.random.default_rng(13)
    frames = [build_features(encode(random_physical(rng)), CONFIG) for _ in range(10)]
    from_frames = tensors_from_frames(frames)
    raw, flat, nodes = (np.stack([getattr(f, name) for f in frames])
                        for name in ("raw", "flat", "nodes"))
    for tensor, array in zip(from_frames, tensors_from_batch(raw, flat, nodes)):
        assert tensor.dtype == torch.float32 and torch.equal(tensor, array)
    snapshot = frames[0].flat.copy()
    from_frames[1][0, 0] = 123.0
    np.testing.assert_array_equal(frames[0].flat, snapshot)
    with pytest.raises(ValueError):
        tensors_from_frames([])
    with pytest.raises(TypeError):
        tensors_from_frames([raw[0]])


def test_tensors_from_batch_refuses_views_of_different_decisions():
    raw, flat, nodes = arrays(14, 6)
    other_raw, other_flat, other_nodes = arrays(15, 6)
    derived_slot = (NODE_ORDER.index(P), 0)
    corrupt_derived = nodes.copy()
    corrupt_derived[:, derived_slot[0], derived_slot[1]] += 0.1
    nan = flat.copy()
    nan[0, 0] = np.nan
    cases = {
        "raw_vs_flat": (other_raw, flat, nodes),
        "flat_vs_nodes": (raw, flat, other_nodes),
        "derived_slot": (raw, flat, corrupt_derived),
        "nan": (raw, nan, nodes),
        "rows": (raw[:5], flat, nodes),
        "24d": (np.c_[raw, np.zeros((6, 3))], flat, nodes),
        "unbatched": (raw[0], flat[0], nodes[0]),
    }
    for name, case in cases.items():
        with pytest.raises(ValueError):
            tensors_from_batch(*case)
    assert [t.shape[0] for t in tensors_from_batch(other_raw, other_flat, other_nodes)] == [6] * 3


# ---------------------------------------------------------- checkpoints

def _variant_config(tmp_path, **camera):
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    data["camera"].update(camera)
    path = tmp_path / "variant.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return load_config(path)


def _tamper(path, edit):
    payload = torch.load(path, weights_only=True)
    edit(payload)
    torch.save(payload, path)


def test_checkpoint_round_trip_restores_outputs_exactly(tmp_path):
    raw, flat, nodes = batch(16)
    for arm in ARMS:
        source, target = model(arm, seed=1), model(arm, seed=2)
        path = tmp_path / f"{arm}.pt"
        assert save_checkpoint(path, source, manifest=MANIFEST,
                               extra={"iteration": 7, "lr": [1e-3, None]}) == path
        loaded = load_checkpoint(path, target, manifest=MANIFEST)
        assert_same_state(state(source), state(target))
        with torch.no_grad():
            for a, b in zip(source.act(raw, flat, nodes, deterministic=True).values(),
                            target.act(raw, flat, nodes, deterministic=True).values()):
                assert torch.equal(a, b)
        metadata = loaded["metadata"]
        assert loaded["extra"] == {"iteration": 7, "lr": [1e-3, None]}
        assert metadata == {"schema_id": contracts.SCHEMA_ID,
                            "model_version": contracts.MODEL_VERSION, "arm": arm,
                            **{key: MANIFEST[key] for key in models.CHECKPOINT_HASHES},
                            "parameter_report": source.parameter_report()}
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(f"{arm}.pt" for arm in ARMS)


def test_load_refuses_any_hash_or_report_mismatch_and_leaves_the_model_untouched(tmp_path):
    path = tmp_path / "typed.pt"
    save_checkpoint(path, model("ontology_rgat_ppo", seed=1), manifest=MANIFEST)
    edits = {
        "algorithm_hash": lambda p: p["metadata"].update(algorithm_hash="0" * 64),
        "feature_hash": lambda p: p["metadata"].update(feature_hash="0" * 64),
        "graph_hash": lambda p: p["metadata"].update(graph_hash="0" * 64),
        "calibration_hash": lambda p: p["metadata"].update(calibration_hash="0" * 64),
        "model_version": lambda p: p["metadata"].update(model_version="spatial-current-models/0"),
        "parameter_report": lambda p: p["metadata"]["parameter_report"].update(total=4015),
    }
    for field, edit in edits.items():
        tampered = tmp_path / f"{field}.pt"
        tampered.write_bytes(path.read_bytes())
        _tamper(tampered, edit)
        target = model("ontology_rgat_ppo", seed=2)
        before = state(target)
        with pytest.raises(ValueError, match=field):
            load_checkpoint(tampered, target, manifest=MANIFEST)
        assert_same_state(before, state(target))

    variant = _variant_config(tmp_path, z_near_m=0.06)
    variant_manifest = contract_manifest(variant)
    assert variant_manifest["calibration_hash"] != MANIFEST["calibration_hash"]
    target = CurrentActorCritic("ontology_rgat_ppo", variant, seed=2)
    with pytest.raises(ValueError, match="calibration_hash"):
        load_checkpoint(path, target, manifest=variant_manifest)
    with pytest.raises(ValueError, match="differ from the contract"):
        load_checkpoint(path, target, manifest=MANIFEST)


def test_load_refuses_a_different_arm(tmp_path):
    for saved, loading in (("ontology_rgat_ppo", "untyped_gnn_ppo"),
                           ("raw_ppo", "semantic_mlp_ppo"),
                           ("untyped_gnn_ppo", "raw_ppo")):
        path = tmp_path / f"{saved}.pt"
        save_checkpoint(path, model(saved), manifest=MANIFEST)
        target = model(loading)
        before = state(target)
        with pytest.raises(ValueError, match="arm"):
            load_checkpoint(path, target, manifest=MANIFEST)
        assert_same_state(before, state(target))


def test_load_refuses_payloads_without_this_schema_id(tmp_path):
    from ontology_rgat.direct_policy.contracts import make_contract
    from ontology_rgat.direct_policy.models import DirectActorCritic
    from ontology_rgat.direct_policy.models import save_checkpoint as direct_save

    with torch.random.fork_rng():
        direct = DirectActorCritic(make_contract(3, backend="local"), "ppo", seed=0)
    direct_path = tmp_path / "direct.pt"
    direct_save(direct_path, direct, training_step=0, optimizer_state={}, rng_state=None)
    assert "schema_id" not in torch.load(direct_path, weights_only=True)

    bare = tmp_path / "bare.pt"
    torch.save(model("raw_ppo").state_dict(), bare)
    pickled = tmp_path / "pickled.pt"
    torch.save({"schema_id": contracts.SCHEMA_ID, "metadata": np.float64(1.0)}, pickled)
    for path in (direct_path, bare, pickled):
        target = model("raw_ppo")
        before = state(target)
        with pytest.raises(ValueError, match=contracts.SCHEMA_ID):
            load_checkpoint(path, target, manifest=MANIFEST)
        assert_same_state(before, state(target))


def test_save_refuses_a_stale_or_incomplete_manifest_and_an_unloadable_extra(tmp_path):
    net = model("untyped_gnn_ppo")
    incomplete = dict(MANIFEST, algorithm_hash=None)
    stale = contract_manifest(_variant_config(tmp_path, z_near_m=0.07))
    for manifest, match in ((incomplete, "incomplete"), (stale, "differ from the contract"),
                            ({}, "not a")):
        with pytest.raises(ValueError, match=match):
            save_checkpoint(tmp_path / "x.pt", net, manifest=manifest)
    with pytest.raises(ValueError, match="weights_only"):
        save_checkpoint(tmp_path / "x.pt", net, manifest=MANIFEST,
                        extra={"step": np.int64(3)})
    assert not any(p.suffix in (".pt", ".tmp") for p in tmp_path.iterdir())


def test_manifest_hashes_the_layer_architecture(monkeypatch):
    assert MANIFEST["model"]["architecture"] == dict(contracts.MODEL_ARCHITECTURE)
    assert models.NEGATIVE_SLOPE == contracts.MODEL_ARCHITECTURE["attention_negative_slope"]
    assert models.ATTENTION_INIT_STD == contracts.MODEL_ARCHITECTURE["attention_init_std"]
    layer = CurrentActorCritic("ontology_rgat_ppo", CONFIG, seed=0).encoder.layer
    assert abs(float(layer.weight.std()) * math.sqrt(layer.dim) - 1.0) < 0.15
    changed = dict(contracts.MODEL_ARCHITECTURE, attention_negative_slope=0.1)
    monkeypatch.setattr(contracts, "MODEL_ARCHITECTURE", changed)
    other = contract_manifest(CONFIG)
    assert other["algorithm_hash"] != MANIFEST["algorithm_hash"]
    for key in ("feature_hash", "graph_hash", "calibration_hash", "observation_hash"):
        assert other[key] == MANIFEST[key], key


def test_actor_output_layer_starts_near_zero_and_still_learns():
    for arm in contracts.ARMS:
        net = CurrentActorCritic(arm, CONFIG, seed=0)
        out = net.actor_head[-1]
        assert torch.count_nonzero(out.bias) == 0
        bound = contracts.MODEL_ARCHITECTURE["actor_output_init_scale"] / math.sqrt(
            out.in_features)
        assert float(out.weight.abs().max()) <= bound and float(out.weight.abs().max()) > 0
