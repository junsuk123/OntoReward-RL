"""The one proposed 8-node / 22-edge graph and its untyped control (guide 8.2)."""
from collections import Counter

import pytest

from ontology_rgat.current_semantics import features, graph
from ontology_rgat.current_semantics.contracts import NODE_ORDER, contract_manifest, load_config
from ontology_rgat.current_semantics.graph import (SELF, TYPED_GRAPH, UNTYPED_GRAPH, GraphSpec,
                                                   graph_for_arm, reachable_within)


Q, G, M, A, P, O, C, L = NODE_ORDER
PRIMITIVES = (Q, G, M, A)
DERIVED = (P, O, C)
GUIDE_EDGES = {
    "qualifiesEstimate": {(Q, P), (Q, O), (Q, C)},
    "informsApproach": {(G, P), (M, P)},
    "affectsVisibility": {(G, O), (M, O), (A, O)},
    "conditionsContact": {(G, C), (M, C), (A, C)},
    "informsLanding": {(P, L), (O, L), (C, L)},
    SELF: {(node, node) for node in NODE_ORDER},
}


def pairs(spec, relation=None):
    return {(s, t) for s, t, r in spec.edges if relation is None or r == relation}


def test_typed_graph_is_the_guide_table_exactly():
    assert TYPED_GRAPH.nodes == NODE_ORDER == features.NODE_ORDER
    assert len(TYPED_GRAPH.edges) == 22
    counts = Counter(r for _, _, r in TYPED_GRAPH.edges)
    assert counts[SELF] == 8 and sum(counts.values()) - counts[SELF] == 14
    for relation, expected in GUIDE_EDGES.items():
        assert pairs(TYPED_GRAPH, relation) == expected, relation
    assert set(TYPED_GRAPH.relations) == set(GUIDE_EDGES)
    assert TYPED_GRAPH.readout_node == L


def test_there_is_no_o_to_c_edge_and_nothing_leaves_the_landing_context():
    assert (O, C) not in pairs(TYPED_GRAPH)
    assert {t for s, t in pairs(TYPED_GRAPH) if s == L} == {L}


def test_primitives_reach_landing_context_in_exactly_two_hops():
    for spec in (TYPED_GRAPH, UNTYPED_GRAPH):
        one, two = reachable_within(spec, L, 1), reachable_within(spec, L, 2)
        assert set(DERIVED) <= one
        assert not set(PRIMITIVES) & one
        assert set(PRIMITIVES) <= two
        assert two == set(NODE_ORDER)


def test_untyped_control_erases_only_the_relation_names():
    assert UNTYPED_GRAPH.relations == ("related", SELF)
    assert pairs(UNTYPED_GRAPH) == pairs(TYPED_GRAPH)
    assert pairs(UNTYPED_GRAPH, SELF) == pairs(TYPED_GRAPH, SELF)
    assert Counter(r for _, _, r in UNTYPED_GRAPH.edges) == {"related": 14, SELF: 8}
    assert UNTYPED_GRAPH.graph_hash != TYPED_GRAPH.graph_hash
    assert graph_for_arm("ontology_rgat_ppo") is TYPED_GRAPH
    assert graph_for_arm("untyped_gnn_ppo") is UNTYPED_GRAPH
    with pytest.raises(KeyError):
        graph_for_arm("raw_ppo")


def test_edge_index_matches_the_named_edges():
    for spec in (TYPED_GRAPH, UNTYPED_GRAPH):
        source, target, relation = spec.edge_index()
        decoded = {(spec.nodes[s], spec.nodes[t], spec.relations[r])
                   for s, t, r in zip(source, target, relation)}
        assert decoded == set(spec.edges)


def test_spec_rejects_malformed_graphs():
    edges = TYPED_GRAPH.edges
    with pytest.raises(ValueError):
        GraphSpec(version="x", nodes=NODE_ORDER, relations=TYPED_GRAPH.relations,
                  edges=edges + ((O, C, "qualifiesEstimate"), (O, C, "informsApproach")))
    with pytest.raises(ValueError):
        GraphSpec(version="x", nodes=NODE_ORDER, relations=TYPED_GRAPH.relations + ("unused",),
                  edges=edges)
    with pytest.raises(ValueError):
        GraphSpec(version="x", nodes=NODE_ORDER, relations=TYPED_GRAPH.relations,
                  edges=edges + ((Q, Q, "informsApproach"),))


def test_graph_hash_enters_the_contract_manifest():
    manifest = contract_manifest(load_config())
    assert manifest["complete"] and manifest["graph_hash"]
    assert manifest["graph"]["graphs"]["typed"]["graph_hash"] == TYPED_GRAPH.graph_hash
    assert manifest["graph"]["graphs"]["untyped"]["graph_hash"] == UNTYPED_GRAPH.graph_hash
    assert graph.manifest_part(load_config())["readout_node"] == L
