"""Edge index tables for the batched relational attention layer.

The ontology schema is fixed for a whole run, so everything that depends only
on ``(src, dst, rel)`` is computed once. Recomputing it per forward pass would
give back a good part of what batching wins.
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = ["Topology"]


class Topology(torch.nn.Module):
    """Index tensors for one fixed relational graph.

    Held as a module so ``model.to(device)`` moves the indices with the
    parameters; they are buffers, so they are not trained and not saved as
    weights that could drift out of step with the schema.
    """

    def __init__(self, src, dst, rel, n_nodes: int, n_relations: int, goal_node: int):
        super().__init__()
        src = np.asarray(src, dtype=np.int64).reshape(-1)
        dst = np.asarray(dst, dtype=np.int64).reshape(-1)
        rel = np.asarray(rel, dtype=np.int64).reshape(-1)
        if not (src.size == dst.size == rel.size):
            raise ValueError("src, dst and rel must be the same length")
        if src.min() < 0 or src.max() >= n_nodes or dst.min() < 0 or dst.max() >= n_nodes:
            raise ValueError("edge endpoints outside the node range")
        if rel.min() < 0 or rel.max() >= n_relations:
            raise ValueError("relation ids outside the relation range")

        self.n_nodes = int(n_nodes)
        self.n_relations = int(n_relations)
        self.n_edges = int(src.size)
        self.goal_node = int(goal_node)

        self.register_buffer("src", torch.as_tensor(src), persistent=False)
        self.register_buffer("dst", torch.as_tensor(dst), persistent=False)
        self.register_buffer("rel", torch.as_tensor(rel), persistent=False)
        # Flat index into a [relations, nodes] plane, which is how the
        # per-relation node projections are laid out.
        self.register_buffer("rel_src", torch.as_tensor(rel * n_nodes + src), persistent=False)
        self.register_buffer("rel_dst", torch.as_tensor(rel * n_nodes + dst), persistent=False)
        # Segment id the attention softmax normalises over. ARGAT pools every
        # incoming edge of a node; WIRGAT pools within each relation
        # separately (Busbridge et al. 2019, figures 1 and 2).
        self.register_buffer("seg_argat", torch.as_tensor(dst), persistent=False)
        self.register_buffer("seg_wirgat", torch.as_tensor(rel * n_nodes + dst),
                             persistent=False)

    def segments(self, mode: str) -> tuple[torch.Tensor, int]:
        if mode == "argat":
            return self.seg_argat, self.n_nodes
        if mode == "wirgat":
            return self.seg_wirgat, self.n_nodes * self.n_relations
        raise ValueError(f"attention_mode must be 'argat' or 'wirgat', got {mode!r}")

    @classmethod
    def from_graph(cls, graph, n_relations: int) -> "Topology":
        """Build from a :class:`ontology_rgat.semantic.OntologyGraph`."""
        return cls(graph.src, graph.dst, graph.rel, graph.X.shape[1],
                   n_relations, graph.goal_node)

    def extra_repr(self) -> str:
        return (f"nodes={self.n_nodes}, relations={self.n_relations}, "
                f"edges={self.n_edges}, goal={self.goal_node}")
