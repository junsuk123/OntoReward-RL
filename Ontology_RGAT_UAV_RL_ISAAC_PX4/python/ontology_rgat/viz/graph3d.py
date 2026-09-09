"""A 3D snapshot of the ontology graph with the R-GAT's own attention on it.

The dashboard already answers "is the run going somewhere"; this answers the
question a reviewer asks next, which is *what the potential learned to look
at*. The schema is fixed, so the interesting quantity is not the topology but
the second-layer attention over it: which relation carries the signal into
SafeLanding at this point in training, and how that changes once the markers
drop out or the deck starts moving.

Two things are worth being explicit about.

*The layout is computed, not drawn.* Nodes are placed by longest-path depth
towards the goal and then spread around a ring inside their layer, so the
picture follows the edge list rather than a hand-tuned table that would drift
the moment the schema gains a node. RViz's overlay keeps its flat table --
there the graph hangs beside a real vehicle and has to stay readable from one
viewpoint -- so the two views are deliberately laid out by different rules.

*Attention is learned importance, not causal proof.* A thick edge means the
layer weighted it, nothing stronger; the same caveat the RViz overlay and the
paper carry.
"""
from __future__ import annotations

import math
import time
from typing import Any, Sequence

import numpy as np

from ..semantic import GOAL_NODE, RISK_NODES
from .live import STORE, LiveStore

__all__ = ["layer_of", "layout_3d", "graph_payload", "GraphPublisher"]


def layer_of(src: Sequence[int], dst: Sequence[int], n_nodes: int) -> np.ndarray:
    """Longest-path depth of every node, ignoring self-loops.

    The ontology is a DAG once the self-relation is dropped, so a node's depth
    is one past its deepest predecessor. Sources -- the raw semantic channels --
    land at zero and the goal ends up furthest right, which is the reading
    order the schema was written in.
    """
    src = np.asarray(src, dtype=np.int64).reshape(-1)
    dst = np.asarray(dst, dtype=np.int64).reshape(-1)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    layer = np.zeros(int(n_nodes), dtype=np.int64)
    # n_nodes relaxations settle any DAG; a cycle (which the schema forbids)
    # would stop making progress rather than spin.
    for _ in range(int(n_nodes)):
        changed = False
        for s, d in zip(src, dst):
            if layer[d] < layer[s] + 1:
                layer[d] = layer[s] + 1
                changed = True
        if not changed:
            break
    return layer


def layout_3d(src: Sequence[int], dst: Sequence[int], n_nodes: int,
              goal_node: int = GOAL_NODE) -> np.ndarray:
    """``[n_nodes, 3]`` positions: depth along x, a ring in the y-z plane.

    The ring is what makes this worth rendering in three dimensions at all. A
    layer with eight raw channels drawn as a flat column crosses most of its
    outgoing edges over each other; spread around a circle, every edge into the
    next layer has its own line of sight, and rotating the view separates the
    few that still overlap.
    """
    layer = layer_of(src, dst, n_nodes)
    depth = int(layer.max()) + 1
    pos = np.zeros((int(n_nodes), 3), dtype=float)
    x_span = 4.0
    for level in range(depth):
        members = np.flatnonzero(layer == level)
        x = -x_span / 2.0 + x_span * (level / max(depth - 1, 1))
        if members.size == 1:
            pos[members[0]] = (x, 0.0, 0.0)
            continue
        # Roughly constant arc length between neighbours, so a crowded layer
        # opens out instead of packing its nodes together -- capped, or the
        # nine raw channels would push the ring wider than the graph is deep
        # and the depth axis would stop reading as depth.
        radius = min(1.7, max(0.75, 0.30 * members.size))
        # Golden-angle phase per layer: consecutive rings do not line up, so an
        # edge between them is never hidden exactly behind a node.
        phase = 2.399963 * level
        for i, node in enumerate(members):
            angle = phase + 2.0 * math.pi * i / members.size
            pos[node] = (x, radius * math.cos(angle), radius * math.sin(angle))
    # The goal sits on the axis whatever its ring would have said: it is the
    # read-out node, and every edge in the picture is on its way there.
    pos[int(goal_node)] = (x_span / 2.0, 0.0, 0.0)
    return pos


_GEOMETRY: dict[Any, tuple[np.ndarray, np.ndarray]] = {}


def _geometry(src, dst, n_nodes: int, goal_node: int) -> tuple[np.ndarray, np.ndarray]:
    """Positions and depths for a topology, computed once.

    The schema is fixed for a whole run and this is called from inside the
    50 Hz control loop's monitor, so re-deriving the same layout every few
    steps would be pure waste. Keyed on the edge list, so a schema change still
    produces a new layout rather than a stale one.
    """
    key = (int(n_nodes), int(goal_node),
           np.asarray(src, dtype=np.int64).tobytes(),
           np.asarray(dst, dtype=np.int64).tobytes())
    cached = _GEOMETRY.get(key)
    if cached is None:
        cached = _GEOMETRY[key] = (layout_3d(src, dst, n_nodes, goal_node),
                                   layer_of(src, dst, n_nodes))
    return cached


def _role(index: int, goal_node: int) -> str:
    if index == goal_node:
        return "goal"
    return "risk" if index in RISK_NODES else "support"


def graph_payload(graph, values: Sequence[float] | None = None, *,
                  potential=None, source: str = "", phi: float | None = None,
                  extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Nodes, edges and attention for one graph, ready to serialise.

    ``values`` is the node activation vector; when it is not given it is read
    back off the feature matrix, whose first row is exactly that (see
    :func:`ontology_rgat.semantic.node_features`). ``potential`` is optional --
    without it the picture is the schema alone, which is what the dashboard
    shows before the R-GAT has been trained.
    """
    src = np.asarray(graph.src, dtype=int).reshape(-1)
    dst = np.asarray(graph.dst, dtype=int).reshape(-1)
    rel = np.asarray(graph.rel, dtype=int).reshape(-1)
    n_nodes = len(graph.node_names)
    if values is None:
        values = np.asarray(graph.X, dtype=float)[0, :]
    values = np.asarray(values, dtype=float).reshape(-1)

    alpha = None
    relation_mean = None
    if potential is not None:
        try:
            explained = potential.explain(graph)
            alpha = np.asarray(explained["edge_alpha"], dtype=float).reshape(-1)
            relation_mean = np.asarray(explained["relation_mean"], dtype=float)
        except Exception as exc:                       # pragma: no cover - defensive
            # A view is never worth taking a run down with it.
            print(f"WARNING: attention read-out failed for the 3D graph: {exc}")
            alpha = None

    pos, layers = _geometry(src, dst, n_nodes, int(graph.goal_node))
    nodes = [{
        "name": str(name),
        "value": float(values[i]) if i < values.size else 0.0,
        "pos": [round(float(c), 4) for c in pos[i]],
        "layer": int(layer),
        "role": _role(i, int(graph.goal_node)),
    } for i, (name, layer) in enumerate(zip(graph.node_names, layers))]

    edges = []
    for e in range(src.size):
        edge = {"s": int(src[e]), "d": int(dst[e]), "r": int(rel[e])}
        if alpha is not None and e < alpha.size:
            edge["a"] = round(float(alpha[e]), 6)
        edges.append(edge)

    relations = []
    for r, name in enumerate(graph.relation_names):
        entry = {"name": str(name)}
        if relation_mean is not None and r < relation_mean.size:
            entry["mean"] = round(float(relation_mean[r]), 6)
        relations.append(entry)

    payload: dict[str, Any] = {
        "source": source,
        "attention": alpha is not None,
        "nodes": nodes,
        "edges": edges,
        "relations": relations,
        "goal_node": int(graph.goal_node),
        "time": time.time(),
    }
    if phi is not None and np.isfinite(phi):
        payload["phi"] = float(phi)
    if extra:
        payload.update(extra)
    return payload


class GraphPublisher:
    """Throttled writer of graph snapshots into the live store.

    The potential is an attribute rather than a constructor argument because
    the pipeline has an episode monitor running before the R-GAT exists: the
    dataset stage publishes the bare schema, and the same publisher starts
    carrying attention the moment training hands a model over.
    """

    def __init__(self, cfg, store: LiveStore | None = None, potential=None):
        self.cfg = cfg
        self.store = store or STORE
        self.potential = potential
        opt = getattr(cfg.viz, "graph3d", None)
        self.enabled = bool(getattr(opt, "enabled", True))
        self.every = max(1, int(getattr(opt, "every", 5)))
        self._counter = 0

    def publish(self, graph, values=None, *, source: str = "",
                phi: float | None = None, force: bool = False,
                extra: dict[str, Any] | None = None) -> None:
        if not self.enabled or graph is None:
            return
        index = self._counter
        self._counter += 1
        if index % self.every and not force:
            return
        self.store.graph(graph_payload(graph, values, potential=self.potential,
                                       source=source, phi=phi, extra=extra))

    def clear(self) -> None:
        self._counter = 0
