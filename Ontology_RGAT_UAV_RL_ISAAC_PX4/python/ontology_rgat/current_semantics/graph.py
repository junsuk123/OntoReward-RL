"""The one proposed ontology graph of spatial_current_semantics_v1 (guide 8.2).

A candidate frozen before training, not a validated graph: eight nodes,
fourteen semantic edges and eight self loops.  An edge runs source -> target,
the direction its message travels.  Every primitive node (Q, G, M, A) reaches
LandingContext only through P, O or C, i.e. in exactly two hops, which is why
the encoder applies its shared layer twice; there is deliberately no O -> C
edge, which would change that path structure.  The untyped control keeps every
(source, target) pair and erases only the relation names, so the two graph
arms differ in relation typing alone.  Relation names select weights; they are
never executed as rules or gates.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from . import GRAPH_VERSION
from ..direct_policy.contracts import digest
from .contracts import ENCODER_RELATIONS, NODE_ORDER, CurrentSemanticsConfig


SELF = "self"
RELATED = "related"
TYPED_RELATIONS = ("qualifiesEstimate", "informsApproach", "affectsVisibility",
                   "conditionsContact", "informsLanding", SELF)
UNTYPED_RELATIONS = (RELATED, SELF)
READOUT_NODE = NODE_ORDER[-1]

_Q, _G, _M, _A, _P, _O, _C, _L = NODE_ORDER
SEMANTIC_EDGES = (
    (_Q, _P, "qualifiesEstimate"), (_Q, _O, "qualifiesEstimate"), (_Q, _C, "qualifiesEstimate"),
    (_G, _P, "informsApproach"), (_M, _P, "informsApproach"),
    (_G, _O, "affectsVisibility"), (_M, _O, "affectsVisibility"), (_A, _O, "affectsVisibility"),
    (_G, _C, "conditionsContact"), (_M, _C, "conditionsContact"), (_A, _C, "conditionsContact"),
    (_P, _L, "informsLanding"), (_O, _L, "informsLanding"), (_C, _L, "informsLanding"),
)
SELF_EDGES = tuple((node, node, SELF) for node in NODE_ORDER)


@dataclass(frozen=True)
class GraphSpec:
    """Nodes, relations and (source, target, relation) edges of one encoder graph.

    Every declared relation must label an edge: the encoder allocates one
    weight matrix per relation, and an unused one would be a dummy parameter.
    """
    version: str
    nodes: tuple[str, ...]
    relations: tuple[str, ...]
    edges: tuple[tuple[str, str, str], ...]
    readout_node: str = READOUT_NODE

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "relations", tuple(self.relations))
        object.__setattr__(self, "edges", tuple(tuple(edge) for edge in self.edges))
        if len(set(self.nodes)) != len(self.nodes) or len(set(self.relations)) != len(self.relations):
            raise ValueError("node and relation names must be unique")
        if self.readout_node not in self.nodes:
            raise ValueError(f"readout node {self.readout_node!r} is not a node")
        for edge in self.edges:
            if len(edge) != 3:
                raise ValueError(f"edge {edge!r} must be (source, target, relation)")
            source, target, relation = edge
            if source not in self.nodes or target not in self.nodes:
                raise ValueError(f"edge {edge!r} names an unknown node")
            if relation not in self.relations:
                raise ValueError(f"edge {edge!r} names an unknown relation")
            if (relation == SELF) != (source == target):
                raise ValueError(f"edge {edge!r}: self loops, and only they, carry {SELF!r}")
        pairs = [(source, target) for source, target, _ in self.edges]
        if len(set(pairs)) != len(pairs):
            raise ValueError("each (source, target) pair may carry one edge")
        unused = set(self.relations) - {relation for _, _, relation in self.edges}
        if unused:
            raise ValueError(f"relations {sorted(unused)} label no edge")

    def edge_index(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(source, target, relation) int64 indices into ``nodes`` and ``relations``."""
        node = {name: index for index, name in enumerate(self.nodes)}
        relation = {name: index for index, name in enumerate(self.relations)}
        return (np.array([node[s] for s, _, _ in self.edges], dtype=np.int64),
                np.array([node[t] for _, t, _ in self.edges], dtype=np.int64),
                np.array([relation[r] for _, _, r in self.edges], dtype=np.int64))

    def relation_counts(self) -> dict[str, int]:
        counts = Counter(relation for _, _, relation in self.edges)
        return {relation: counts[relation] for relation in self.relations}

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "nodes": list(self.nodes),
                "relations": list(self.relations),
                "edges": [list(edge) for edge in self.edges],
                "readout_node": self.readout_node}

    @property
    def graph_hash(self) -> str:
        return digest(self.to_dict())


def untyped(spec: GraphSpec) -> GraphSpec:
    """The same edges with every non-self relation renamed ``related``."""
    return GraphSpec(version=spec.version, nodes=spec.nodes, relations=UNTYPED_RELATIONS,
                     edges=tuple((s, t, r if r == SELF else RELATED) for s, t, r in spec.edges),
                     readout_node=spec.readout_node)


TYPED_GRAPH = GraphSpec(version=GRAPH_VERSION, nodes=NODE_ORDER, relations=TYPED_RELATIONS,
                        edges=SEMANTIC_EDGES + SELF_EDGES)
UNTYPED_GRAPH = untyped(TYPED_GRAPH)
ARM_GRAPHS = MappingProxyType({"untyped_gnn_ppo": "untyped", "ontology_rgat_ppo": "typed"})

if set(ARM_GRAPHS) != set(ENCODER_RELATIONS) or any(
        len({"typed": TYPED_GRAPH, "untyped": UNTYPED_GRAPH}[kind].relations)
        != ENCODER_RELATIONS[arm] for arm, kind in ARM_GRAPHS.items()):
    raise AssertionError("graph relations disagree with contracts.ENCODER_RELATIONS")


def graph_for_arm(arm: str) -> GraphSpec:
    if arm not in ARM_GRAPHS:
        raise KeyError(f"arm {arm!r} has no graph encoder; graph arms: {sorted(ARM_GRAPHS)}")
    return TYPED_GRAPH if ARM_GRAPHS[arm] == "typed" else UNTYPED_GRAPH


def reachable_within(spec: GraphSpec, target: str, hops: int) -> set[str]:
    """Nodes with a directed path of at most ``hops`` edges ending at ``target``."""
    if target not in spec.nodes:
        raise ValueError(f"unknown node {target!r}")
    if hops < 0:
        raise ValueError("hops must be non-negative")
    incoming: dict[str, set[str]] = {node: set() for node in spec.nodes}
    for source, destination, _ in spec.edges:
        incoming[destination].add(source)
    reached, frontier = {target}, {target}
    for _ in range(hops):
        frontier = {source for node in frontier for source in incoming[node]} - reached
        reached |= frontier
    return reached


def manifest_part(config: CurrentSemanticsConfig) -> dict[str, Any]:
    """Both encoder graphs for ``contracts.contract_manifest`` (hashed into graph_hash).

    ``config`` is the hook's signature; the graph does not depend on it.
    """
    graphs = {"typed": TYPED_GRAPH, "untyped": UNTYPED_GRAPH}
    return {
        "version": GRAPH_VERSION,
        "node_order": NODE_ORDER,
        "edge_direction": "source -> target, the direction a message travels",
        "readout_node": READOUT_NODE,
        "graphs": {kind: {**spec.to_dict(), "relation_counts": spec.relation_counts(),
                          "graph_hash": spec.graph_hash}
                   for kind, spec in graphs.items()},
        "arm_graphs": ARM_GRAPHS,
    }
