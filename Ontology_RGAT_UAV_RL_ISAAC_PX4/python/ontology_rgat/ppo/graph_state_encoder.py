"""R-GAT encoder that turns the ontology situation graph into policy state.

``G_t -> R-GAT -> H_t -> graph-level readout -> g_t``, and ``g_t`` is
concatenated onto the actor's and the critic's feature vector.

Three things distinguish this from the frozen FOV readout in
``rgat/fov_risk_model.py``, and all three follow from it being a *state*
rather than a reward:

* **It is trained by PPO.** There is no offline stage, no frozen checkpoint and
  no checksum gate. The gradient of the PPO objective reaches the relational
  kernels. Freezing it would be a different method.
* **It reads every node.** The reward readout takes the goal node of a
  one-unit second layer. A state has no reason to discard eight of nine node
  embeddings, so the readout is ``tanh(W [mean(H) ; max(H)])``. The reduced
  study measured the alternatives: mean alone washes the per-node differences
  out, and a goal-node readout is a one-node bottleneck.
* **Actor and critic hold one encoder each.** They already own separate
  optimizer state and learning rates; a shared encoder would have to merge two
  differently scaled gradients into one parameter. They receive the same
  ``G_t`` and have the same architecture.

Ablations
---------
``representation`` selects what the proposed arm actually gets, and all three
non-baseline settings run through this same class:

==============  ==========================================================
``ontology_rgat``  relations kept distinct -- the proposed model
``gat``            every declared edge collapsed onto one relation
``node_pool``      self-loops only: node features and readout, no messages
==============  ==========================================================
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..rgat.layers import RelationalGraphAttention
from ..rgat.state_graph import (STATE_GRAPH_INPUT_DIM, STATE_GRAPH_VERSION,
                                STATE_NODE_NAMES, STATE_RELATION_NAMES,
                                empty_state_graph)
from ..rgat.topology import Topology


__all__ = ["GRAPH_STATE_REPRESENTATIONS", "GraphStateEncoder",
           "graph_state_topology", "graph_feature_tensor"]


GRAPH_STATE_REPRESENTATIONS = ("ontology_rgat", "gat", "node_pool")

# Shared by both relational layers, fixed here so a checkpoint cannot quietly
# change the attention kernel between runs.
_LAYER_KWARGS = dict(
    head_aggregation="mean", attention_mode="argat", attention_style="sum",
    attention_units=1, leaky_relu_slope=0.2, kernel_basis_size=0,
    attn_kernel_basis_size=0, feature_dropout=0.0, support_dropout=0.0,
    softmax_floor=1e-12, stable_softmax=True)


def graph_state_topology(representation: str = "ontology_rgat") -> Topology:
    """The fixed edge tables for one representation.

    ``node_pool`` keeps only the self-loops, which is what makes it an ablation
    of message passing rather than of the ontology's node set. ``gat`` keeps
    every edge but merges the relation types, which is what makes it an
    ablation of *relational* attention rather than of attention.
    """
    if representation not in GRAPH_STATE_REPRESENTATIONS:
        raise ValueError(
            f"graph state representation must be one of {GRAPH_STATE_REPRESENTATIONS}")
    graph = empty_state_graph()
    nodes = graph.X.shape[1]
    self_relation = STATE_RELATION_NAMES.index("self")
    if representation == "node_pool":
        keep = np.asarray(graph.rel) == self_relation
        return Topology(np.asarray(graph.src)[keep], np.asarray(graph.dst)[keep],
                        np.zeros(int(keep.sum()), dtype=np.int64), nodes, 1,
                        graph.goal_node)
    if representation == "gat":
        return Topology(graph.src, graph.dst,
                        np.zeros(np.asarray(graph.rel).size, dtype=np.int64),
                        nodes, 1, graph.goal_node)
    return Topology.from_graph(graph, len(STATE_RELATION_NAMES))


def graph_feature_tensor(graphs, *, dtype=torch.float32, device=None) -> torch.Tensor:
    """Stack ``OntologyGraph.X`` matrices into ``[..., nodes, in_dim]``.

    ``OntologyGraph`` stores ``X`` as ``[in_dim, nodes]``; the layer wants the
    transpose, and doing it in one place keeps every caller from getting it
    wrong in its own way.
    """
    if hasattr(graphs, "X"):
        stacked = np.asarray(graphs.X, dtype=np.float32).T[None]
    else:
        stacked = np.stack([np.asarray(graph.X, dtype=np.float32).T
                            for graph in graphs], axis=0)
    return torch.as_tensor(stacked, dtype=dtype, device=device)


class GraphStateEncoder(nn.Module):
    """Two relational layers with a residual, then a graph-level readout."""

    graph_version = STATE_GRAPH_VERSION
    node_names = STATE_NODE_NAMES

    def __init__(self, *, hidden_dim: int = 32, graph_dim: int = 32,
                 relation_dim: int = 6, heads: int = 1,
                 representation: str = "ontology_rgat", seed: int = 42):
        super().__init__()
        if representation not in GRAPH_STATE_REPRESENTATIONS:
            raise ValueError(
                f"graph state representation must be one of {GRAPH_STATE_REPRESENTATIONS}")
        self.representation = str(representation)
        self.hidden_dim = int(hidden_dim)
        self.graph_dim = int(graph_dim)
        if self.hidden_dim <= 0 or self.graph_dim <= 0:
            raise ValueError("graph encoder widths must be positive")
        self.topology = graph_state_topology(self.representation)
        self.layer1 = RelationalGraphAttention(
            STATE_GRAPH_INPUT_DIM, self.hidden_dim, self.topology,
            relation_dim=int(relation_dim), heads=int(heads), **_LAYER_KWARGS)
        self.layer2 = RelationalGraphAttention(
            self.layer1.out_dim, self.hidden_dim, self.topology,
            relation_dim=int(relation_dim), heads=int(heads), **_LAYER_KWARGS)
        # mean and max over the node axis, so the readout sees both the
        # situation as a whole and whichever node currently dominates it.
        self.readout = nn.Linear(2 * self.layer2.out_dim, self.graph_dim)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.layer1.reset_parameters(generator, scheme="glorot")
        self.layer2.reset_parameters(generator, scheme="glorot")
        with torch.no_grad():
            bound = float(np.sqrt(1.0 / (2 * self.layer2.out_dim)))
            self.readout.weight.uniform_(-bound, bound, generator=generator)
            self.readout.bias.zero_()

    @property
    def output_dim(self) -> int:
        return self.graph_dim

    def node_embeddings(self, X: torch.Tensor) -> torch.Tensor:
        """``H_t`` for a batch of graphs, ``[..., nodes, hidden]``."""
        leading = X.shape[:-2]
        flat = X.reshape(-1, X.shape[-2], X.shape[-1])
        hidden = torch.tanh(self.layer1(flat))
        hidden = torch.tanh(hidden + self.layer2(hidden))
        return hidden.reshape(*leading, hidden.shape[-2], hidden.shape[-1])

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """``g_t`` for a batch of graphs, ``[..., graph_dim]``."""
        if X.shape[-1] != STATE_GRAPH_INPUT_DIM:
            raise ValueError(
                f"graph features must have width {STATE_GRAPH_INPUT_DIM}, "
                f"got {X.shape[-1]}")
        if X.shape[-2] != len(STATE_NODE_NAMES):
            raise ValueError(
                f"graph features must have {len(STATE_NODE_NAMES)} nodes, "
                f"got {X.shape[-2]}")
        hidden = self.node_embeddings(X)
        pooled = torch.cat((hidden.mean(-2), hidden.amax(-2)), dim=-1)
        return torch.tanh(self.readout(pooled))

    def describe(self) -> dict:
        return {
            "graph_version": self.graph_version,
            "representation": self.representation,
            "node_count": len(STATE_NODE_NAMES),
            "relation_count": int(self.topology.n_relations),
            "edge_count": int(self.topology.n_edges),
            "input_dim": STATE_GRAPH_INPUT_DIM,
            "hidden_dim": self.hidden_dim,
            "graph_dim": self.graph_dim,
            "readout": "tanh(W [mean(H) ; max(H)] + b)",
            "trained_by": "ppo",
            "frozen": False,
        }
