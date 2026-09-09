"""The batched R-GAT layer against a plain per-edge reference loop.

The whole justification for the dense batched formulation is that it changes
speed and nothing else, so it is checked against the arithmetic it replaces --
the same check the retired ``matlab/tests/test_rgat_equivalence.m`` performed
against the MATLAB edge loop, transcribed here.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ontology_rgat.config import default_config
from ontology_rgat.rgat.layers import RelationalGraphAttention
from ontology_rgat.rgat.model import build_potential
from ontology_rgat.rgat.topology import Topology
from ontology_rgat.semantic import SemanticState, build_ontology_graph


@pytest.fixture(scope="module")
def setup():
    cfg = default_config("quick")
    sem = SemanticState(
        position_error=0.7, vertical_speed=-0.4, tilt=0.1, angular_rate=0.2,
        wind_risk=0.3, marker_quality=0.8, visual_stability=0.8, alignment=0.4,
        attitude_stability=0.6, touchdown_safety=0.2, pad_motion=0.35,
        battery_reserve=0.6)
    graph = build_ontology_graph(sem, cfg)
    model = build_potential(cfg, graph).double()
    return cfg, graph, model


def reference_layer(H, layer, graph):
    """``H`` is ``[in_dim, nodes]``; returns ``[units, nodes]``.

    One destination node at a time, one edge at a time, exactly as the
    formulation is written down.
    """
    W = layer.kernel().detach().numpy()[:, 0]          # [R, in_dim, units]
    A = layer.attn_kernel().detach().numpy()[:, 0, :, 0]  # [R, 2U+dr]
    E = layer.relation_embedding.detach().numpy()      # [R, dr]
    units = W.shape[2]
    n = H.shape[1]
    Z = np.zeros((units, n))
    for j in range(n):
        incoming = np.flatnonzero(graph.dst == j)
        scores = np.zeros(incoming.size)
        messages = np.zeros((units, incoming.size))
        for k, e in enumerate(incoming):
            i, r = int(graph.src[e]), int(graph.rel[e])
            hs = W[r].T @ H[:, i]
            hd = W[r].T @ H[:, j]
            raw = float(A[r] @ np.concatenate([hs, hd, E[r]]))
            scores[k] = raw if raw > 0 else 0.2 * raw     # leaky ReLU, slope 0.2
            messages[:, k] = hs
        weights = np.exp(scores)
        weights = weights / (weights.sum() + 1e-9)
        Z[:, j] = messages @ weights
    return Z


def reference_forward(model, X, graph):
    """``X`` is ``[batch, nodes, in_dim]``; returns ``[batch]``."""
    w_out = model.w_out.detach().numpy().reshape(-1)
    b_out = float(model.b_out.detach().numpy().reshape(-1)[0])
    out = np.zeros(X.shape[0])
    for b in range(X.shape[0]):
        Hin = X[b].T
        h1 = np.tanh(reference_layer(Hin, model.layer1, graph))
        h2 = np.tanh(reference_layer(h1, model.layer2, graph) + h1)
        out[b] = np.tanh(w_out @ h2[:, graph.goal_node] + b_out)
    return out


def test_batched_matches_edge_loop(setup):
    cfg, graph, model = setup
    rng = np.random.default_rng(7)
    X = np.repeat(graph.X.T[None, :, :], 8, axis=0)
    X = X + 0.01 * rng.standard_normal(X.shape)

    expected = reference_forward(model, X, graph)
    actual = model(torch.as_tensor(X, dtype=torch.float64)).detach().numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_single_graph_matches_its_batch_entry(setup):
    cfg, graph, model = setup
    rng = np.random.default_rng(11)
    X = np.repeat(graph.X.T[None, :, :], 8, axis=0)
    X = X + 0.01 * rng.standard_normal(X.shape)
    batched = model(torch.as_tensor(X, dtype=torch.float64)).detach().numpy()
    single = model(torch.as_tensor(X[3], dtype=torch.float64)).detach().numpy()
    np.testing.assert_allclose(single, batched[3:4], atol=1e-12)


def test_wirgat_normalises_within_each_relation(setup):
    """WIRGAT attention sums to one per (destination, relation), ARGAT per destination."""
    cfg, graph, _ = setup
    top = Topology.from_graph(graph, cfg.ontology.n_relations)
    X = torch.as_tensor(graph.X.T[None], dtype=torch.float64)
    for mode, segments, n_segments in (("argat", top.dst, top.n_nodes),
                                       ("wirgat", top.seg_wirgat,
                                        top.n_nodes * top.n_relations)):
        layer = RelationalGraphAttention(
            cfg.ontology.in_dim, cfg.rgat.hidden_dim, top,
            attention_mode=mode, relation_dim=cfg.rgat.rel_dim).double()
        layer.reset_parameters(torch.Generator().manual_seed(3))
        _, alpha = layer(X, return_attention=True)
        totals = torch.zeros(n_segments, dtype=torch.float64)
        totals.index_add_(0, segments, alpha.detach().reshape(-1))
        occupied = totals > 0
        # Not exactly one: the softmax carries the MATLAB layer's 1e-9
        # denominator floor rather than subtracting a per-segment maximum.
        np.testing.assert_allclose(totals[occupied].numpy(),
                                   np.ones(int(occupied.sum())), atol=1e-7)


def test_multi_head_mean_equals_single_head_when_heads_are_tied(setup):
    """Averaging identical heads must return the single-head result."""
    cfg, graph, _ = setup
    top = Topology.from_graph(graph, cfg.ontology.n_relations)
    X = torch.as_tensor(graph.X.T[None], dtype=torch.float64)

    one = RelationalGraphAttention(cfg.ontology.in_dim, cfg.rgat.hidden_dim, top,
                                   heads=1, relation_dim=cfg.rgat.rel_dim).double()
    one.reset_parameters(torch.Generator().manual_seed(5))
    three = RelationalGraphAttention(cfg.ontology.in_dim, cfg.rgat.hidden_dim, top,
                                     heads=3, head_aggregation="mean",
                                     relation_dim=cfg.rgat.rel_dim).double()
    with torch.no_grad():
        three.kernel.weight.copy_(one.kernel.weight.expand(-1, 3, -1, -1))
        three.attn_kernel.weight.copy_(one.attn_kernel.weight.expand(-1, 3, -1, -1))
        three.relation_embedding.copy_(one.relation_embedding)
    np.testing.assert_allclose(three(X).detach().numpy(), one(X).detach().numpy(),
                               atol=1e-12)


def test_basis_decomposition_reproduces_its_dense_kernel(setup):
    """``W_r = sum_i c_{r,i} V_i`` must be usable as an ordinary kernel."""
    cfg, graph, _ = setup
    top = Topology.from_graph(graph, cfg.ontology.n_relations)
    X = torch.as_tensor(graph.X.T[None], dtype=torch.float64)
    basis = RelationalGraphAttention(cfg.ontology.in_dim, cfg.rgat.hidden_dim, top,
                                     kernel_basis_size=2,
                                     relation_dim=cfg.rgat.rel_dim).double()
    basis.reset_parameters(torch.Generator().manual_seed(9))
    dense = RelationalGraphAttention(cfg.ontology.in_dim, cfg.rgat.hidden_dim, top,
                                     relation_dim=cfg.rgat.rel_dim).double()
    with torch.no_grad():
        dense.kernel.weight.copy_(basis.kernel())
        dense.attn_kernel.weight.copy_(basis.attn_kernel())
        dense.relation_embedding.copy_(basis.relation_embedding)
    np.testing.assert_allclose(basis(X).detach().numpy(), dense(X).detach().numpy(),
                               atol=1e-12)


def test_potential_is_bounded_and_differentiable(setup):
    cfg, graph, model = setup
    X = torch.as_tensor(np.repeat(graph.X.T[None], 4, axis=0), dtype=torch.float64)
    phi = model(X)
    assert phi.shape == (4,)
    assert torch.all(phi.abs() <= 1.0)
    phi.sum().backward()
    assert model.w_out.grad is not None
    assert torch.isfinite(model.w_out.grad).all()


def test_eleven_node_checkpoint_is_refused(tmp_path, setup):
    """A model from the fixed-pad schema must not load against 13 nodes."""
    from ontology_rgat.rgat.model import load_potential, save_potential

    cfg, graph, model = setup
    path = tmp_path / "old.pt"
    save_potential(model, cfg, path)
    blob = torch.load(path, weights_only=False)
    blob["schema"]["n_nodes"] = 11
    blob["schema"]["in_dim"] = 15
    torch.save(blob, path)
    with pytest.raises(ValueError, match="n_nodes"):
        load_potential(path, cfg, graph)


# Every option this layer exposes is a path that has to actually run. The
# multiplicative style once dropped the relation embedding out of the graph
# entirely, which trains silently -- the loss goes down, one tensor never moves
# -- so the gradient of *every* parameter is checked, not just the output.
@pytest.mark.parametrize("attention_mode", ["argat", "wirgat"])
@pytest.mark.parametrize("attention_style", ["sum", "dot"])
@pytest.mark.parametrize("heads,head_aggregation",
                         [(1, "mean"), (3, "mean"), (3, "sum"),
                          (3, "concat"), (3, "projection")])
@pytest.mark.parametrize("kernel_basis_size", [None, 2])
def test_every_configuration_trains(setup, attention_mode, attention_style,
                                    heads, head_aggregation, kernel_basis_size):
    from ontology_rgat.rgat.model import build_potential

    cfg, graph, _ = setup
    cfg = cfg.derive(**{
        "rgat.attention_mode": attention_mode,
        "rgat.attention_style": attention_style,
        "rgat.attention_units": 1 if attention_style == "sum" else 4,
        "rgat.heads": heads,
        "rgat.head_aggregation": head_aggregation,
        "rgat.kernel_basis_size": kernel_basis_size,
        # The residual around the second layer needs both layers the same width.
        "rgat.residual": head_aggregation != "concat" or heads == 1,
    })
    model = build_potential(cfg, graph)
    X = torch.as_tensor(np.repeat(graph.X.T[None], 6, axis=0), dtype=torch.float32)

    phi = model(X)
    assert phi.shape == (6,)
    assert torch.all(phi.abs() <= 1.0)
    phi.sum().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(parameter.grad).all(), f"{name} has a non-finite gradient"
        assert parameter.grad.abs().sum() > 0, f"{name} has an all-zero gradient"

    explained = model.explain(graph)
    assert explained["relation_mean"].shape == (cfg.ontology.n_relations,)
