"""The dashboard's 3D ontology view: layout, payload and throttling.

The picture is an interpretability claim, so the parts that could quietly lie
are the ones worth pinning down: that the depth axis really is distance to the
goal, that the node values shown are the semantic channels and not the feature
matrix's padding, and that an edge's width comes from the model's attention
rather than from nothing at all.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from ontology_rgat.config import default_config
from ontology_rgat.semantic import GOAL_NODE, SemanticState, build_ontology_graph
from ontology_rgat.viz.dashboard import PAGE
from ontology_rgat.viz.graph3d import (GraphPublisher, graph_payload, layer_of,
                                       layout_3d)
from ontology_rgat.viz.live import LiveStore


@pytest.fixture(scope="module")
def graph():
    cfg = default_config("quick")
    sem = SemanticState(
        position_error=0.7, vertical_speed=-0.4, tilt=0.1, angular_rate=0.2,
        wind_risk=0.3, marker_quality=0.8, visual_stability=0.8, alignment=0.4,
        attitude_stability=0.6, touchdown_safety=0.2, pad_motion=0.35,
        battery_reserve=0.6, gnss_integrity=0.45)
    return cfg, sem, build_ontology_graph(sem, cfg)


def test_layers_follow_the_edges(graph):
    """Depth is longest path to a node, so every edge goes strictly forward."""
    _, _, g = graph
    layer = layer_of(g.src, g.dst, len(g.node_names))
    for s, d in zip(g.src, g.dst):
        if s != d:
            assert layer[d] > layer[s]
    # The raw semantic channels are sources; SafeLanding is the far end.
    assert layer[GOAL_NODE] == layer.max()
    assert layer[g.node_names.index("PositionError")] == 0


def test_layout_is_deterministic_and_bounded(graph):
    _, _, g = graph
    n = len(g.node_names)
    pos = layout_3d(g.src, g.dst, n, g.goal_node)
    assert pos.shape == (n, 3)
    assert np.allclose(pos, layout_3d(g.src, g.dst, n, g.goal_node))
    assert np.abs(pos).max() <= 2.5
    # The goal is on the axis at the far end: nothing is drawn past it.
    assert np.allclose(pos[GOAL_NODE, 1:], 0.0)
    assert pos[GOAL_NODE, 0] == pytest.approx(pos[:, 0].max())
    # No two nodes share a position, or the picture would hide one of them.
    assert len({tuple(np.round(p, 6)) for p in pos}) == n


def test_payload_carries_the_semantic_values_not_the_padding(graph):
    _, sem, g = graph
    payload = graph_payload(g, sem.node_values, source="test")
    assert [n["name"] for n in payload["nodes"]] == list(g.node_names)
    assert [n["value"] for n in payload["nodes"]] == pytest.approx(
        list(sem.node_values))
    assert payload["attention"] is False
    assert all("a" not in e for e in payload["edges"])
    roles = {n["name"]: n["role"] for n in payload["nodes"]}
    assert roles["SafeLanding"] == "goal"
    assert roles["WindRisk"] == "risk"          # more of it is worse
    assert roles["MarkerQuality"] == "support"  # more of it is better
    # Reading the values off the feature matrix must agree with passing them.
    assert [n["value"] for n in graph_payload(g)["nodes"]] == pytest.approx(
        list(sem.node_values))
    json.dumps(payload)                          # the dashboard serialises it


def test_payload_carries_attention_when_a_model_is_given(graph):
    pytest.importorskip("torch")
    cfg, sem, g = graph
    from ontology_rgat.rgat.model import build_potential

    model = build_potential(cfg, g)
    payload = graph_payload(g, sem.node_values, potential=model, source="epoch 3")
    assert payload["attention"] is True
    alpha = np.array([e["a"] for e in payload["edges"]])
    assert alpha.shape == np.asarray(g.rel).shape
    assert np.all(alpha >= 0.0) and np.all(alpha <= 1.0)
    # Attention is a softmax over the edges pooled together, so it cannot be
    # uniform across relations that carry different features.
    assert alpha.std() > 0.0
    means = [r["mean"] for r in payload["relations"]]
    assert len(means) == len(g.relation_names) and all(m >= 0.0 for m in means)
    json.dumps(payload)


def test_publisher_throttles_and_can_be_forced(graph):
    cfg, sem, g = graph
    cfg = default_config("quick")
    cfg.viz.graph3d.every = 4
    store = LiveStore()
    pub = GraphPublisher(cfg, store)
    for step in range(9):
        pub.publish(g, sem.node_values, source=f"step {step}")
    assert store.snapshot()["graph"]["source"] == "step 8"
    pub.publish(g, sem.node_values, source="skipped")
    assert store.snapshot()["graph"]["source"] == "step 8"
    pub.publish(g, sem.node_values, source="forced", force=True)
    assert store.snapshot()["graph"]["source"] == "forced"

    cfg.viz.graph3d.enabled = False
    off = GraphPublisher(cfg, LiveStore())
    off.publish(g, sem.node_values, source="never", force=True)
    assert off.store.snapshot()["graph"] is None


def test_dashboard_page_renders_the_view_without_a_cdn():
    """The page must stay self-contained: no script or style is fetched."""
    assert "cv-graph3d" in PAGE and "graphDraw" in PAGE
    for pattern in ("http://", "https://", "//cdn", "<script src", "<link "):
        assert pattern not in PAGE, f"the dashboard reaches out for {pattern!r}"
