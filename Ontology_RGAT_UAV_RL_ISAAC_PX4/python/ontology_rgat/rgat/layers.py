"""Relational graph attention in PyTorch.

Follows the formulation of Busbridge, Sherburn, Cavallo and Hammerla,
*Relational Graph Attention Networks* (ICLR 2019 submission,
https://openreview.net/forum?id=Bklzkh0qFm), as implemented by the reference
release at https://github.com/babylonhealth/rgat (Apache-2.0). See NOTICE.

Why this is a reimplementation rather than a vendored dependency
----------------------------------------------------------------
The reference release is TensorFlow 1.x: it calls ``tf.sparse_reshape``,
``tf.sparse_transpose``, ``tf.layers.BatchNormalization``, ``tf.nn.dropout``
with ``keep_prob``, ``tensorflow.python.keras`` and ``TensorShape[-1].value``.
All six were removed in TensorFlow 2, and the last TensorFlow 1 release caps at
Python 3.7, while this workspace is Python 3.10 with ROS 2 Humble and Isaac Sim
5.1. Installing it is not possible here, and a second deep-learning runtime
beside the PyTorch that Isaac Sim already ships would be a liability even if it
were. What transfers is the model, and the model is what this file implements.

What it keeps from the reference
--------------------------------
* ``attention_mode``: ``argat`` normalises attention across every incoming edge
  of a node; ``wirgat`` normalises within each relation separately.
* ``attention_style``: ``sum`` is the additive GAT score, ``dot`` the
  transformer-style multiplicative one.
* ``heads`` with ``mean`` / ``sum`` / ``concat`` / ``projection`` aggregation.
* Basis decomposition of the relational kernels, ``W_r = sum_i c_{r,i} V_i``,
  which is how the reference keeps the parameter count off ``O(relations)``.
* Feature and support dropout.

What it adds
------------
* A dense batched formulation. The ontology graph is 14 nodes and 38 edges and
  never changes shape, so the sparse support tensor the reference needs for
  citation-network-sized graphs is pure overhead here. Batching over graphs
  instead of over nodes turns a training step into a handful of large kernels,
  which is what lets the GPU be worth using at all -- see rgat/benchmark.py.
* A learned per-relation embedding in the attention logits, which is the
  reference's ``attn_use_edge_features`` with a one-hot relation as the edge
  feature. The retired MATLAB layer had this term, so it is kept.

Defaults reproduce the retired MATLAB ``+rgat/relationLayer.m`` exactly: one
head, ARGAT, additive attention, leaky-ReLU slope 0.2, and an unshifted softmax
with a 1e-9 denominator floor. ``tests/test_rgat_equivalence.py`` checks the
batched result against a plain per-edge reference loop.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .topology import Topology

__all__ = ["RelationalGraphAttention", "RGAT", "segment_softmax"]

HEAD_AGGREGATIONS = ("mean", "sum", "concat", "projection")
ATTENTION_MODES = ("argat", "wirgat")
ATTENTION_STYLES = ("sum", "dot")


def segment_softmax(logits: torch.Tensor, segments: torch.Tensor, n_segments: int,
                    floor: float = 1e-9, stable: bool = False) -> torch.Tensor:
    """Softmax over edges grouped by ``segments``.

    ``logits`` is ``[..., E]``; the result has the same shape. ``floor`` is the
    denominator floor the retired MATLAB layer used in place of subtracting the
    per-segment maximum, and ``stable=True`` subtracts that maximum instead,
    which matters only if a score ever leaves float range.
    """
    shape = logits.shape
    flat = logits.reshape(-1, shape[-1])
    index = segments.expand_as(flat)
    if stable:
        neg_inf = torch.full((flat.shape[0], n_segments), float("-inf"),
                             dtype=flat.dtype, device=flat.device)
        peak = neg_inf.scatter_reduce(1, index, flat, reduce="amax", include_self=True)
        flat = flat - peak.gather(1, index)
    ex = torch.exp(flat)
    den = torch.zeros(flat.shape[0], n_segments, dtype=ex.dtype, device=ex.device)
    den.scatter_add_(1, index, ex)
    alpha = ex / (den.gather(1, index) + floor)
    return alpha.reshape(shape)


class _RelationalKernel(nn.Module):
    """``W_r`` for every relation and head, optionally basis-decomposed."""

    def __init__(self, relations: int, heads: int, in_dim: int, units: int,
                 basis_size: int | None):
        super().__init__()
        self.relations, self.heads = int(relations), int(heads)
        self.in_dim, self.units = int(in_dim), int(units)
        self.basis_size = int(basis_size) if basis_size else None
        if self.basis_size is None:
            self.weight = nn.Parameter(torch.empty(relations, heads, in_dim, units))
        else:
            # W_{r,h} = sum_i c_{(r,h),i} V_i, the reference's
            # BasisDecompositionDense with coefficients_size = relations*heads.
            self.basis = nn.Parameter(torch.empty(self.basis_size, in_dim, units))
            self.coefficients = nn.Parameter(
                torch.empty(relations * heads, self.basis_size))

    def forward(self) -> torch.Tensor:
        """``[relations, heads, in_dim, units]``."""
        if self.basis_size is None:
            return self.weight
        flat = self.coefficients @ self.basis.reshape(self.basis_size, -1)
        return flat.reshape(self.relations, self.heads, self.in_dim, self.units)


class RelationalGraphAttention(nn.Module):
    """One R-GAT layer over a fixed relational graph, batched across graphs.

    ``forward(H)`` takes ``[batch, nodes, in_dim]`` and returns
    ``[batch, nodes, out_dim]``, where ``out_dim`` is ``units`` for every head
    aggregation but ``concat``, which gives ``heads * units``.
    """

    def __init__(self, in_dim: int, units: int, topology: Topology, *,
                 heads: int = 1, head_aggregation: str = "mean",
                 attention_mode: str = "argat", attention_style: str = "sum",
                 attention_units: int = 1, relation_dim: int = 6,
                 leaky_relu_slope: float = 0.2,
                 kernel_basis_size: int | None = None,
                 attn_kernel_basis_size: int | None = None,
                 feature_dropout: float = 0.0, support_dropout: float = 0.0,
                 use_bias: bool = False, softmax_floor: float = 1e-9,
                 stable_softmax: bool = False):
        super().__init__()
        if head_aggregation not in HEAD_AGGREGATIONS:
            raise ValueError(f"head_aggregation must be one of {HEAD_AGGREGATIONS}")
        if attention_mode not in ATTENTION_MODES:
            raise ValueError(f"attention_mode must be one of {ATTENTION_MODES}")
        if attention_style not in ATTENTION_STYLES:
            raise ValueError(f"attention_style must be one of {ATTENTION_STYLES}")
        if attention_style == "sum" and attention_units != 1:
            raise ValueError("additive ('sum') attention requires attention_units=1")

        self.topology = topology
        self.in_dim, self.units, self.heads = int(in_dim), int(units), int(heads)
        self.head_aggregation = head_aggregation
        self.attention_mode = attention_mode
        self.attention_style = attention_style
        self.attention_units = int(attention_units)
        self.relation_dim = int(relation_dim)
        self.leaky_relu_slope = float(leaky_relu_slope)
        self.feature_dropout = float(feature_dropout)
        self.support_dropout = float(support_dropout)
        self.softmax_floor = float(softmax_floor)
        self.stable_softmax = bool(stable_softmax)

        relations = topology.n_relations
        self.kernel = _RelationalKernel(relations, heads, in_dim, units,
                                        kernel_basis_size)
        # Attention kernel. For the additive style the score is
        #   a_r . [W_r h_src ; W_r h_dst ; E_r]
        # which is one vector of width 2*units + relation_dim per relation and
        # head; for the multiplicative style the two halves are separate
        # projections into an attention space of ``attention_units``.
        attn_in = 2 * self.units + self.relation_dim
        self.attn_kernel = _RelationalKernel(relations, heads, attn_in,
                                            self.attention_units,
                                            attn_kernel_basis_size)
        self.relation_embedding = nn.Parameter(
            torch.empty(relations, self.relation_dim))

        self.out_dim = self.units * (self.heads if head_aggregation == "concat" else 1)
        if head_aggregation == "projection":
            self.projection = nn.Linear(self.heads * self.units, self.units, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.out_dim)) if use_bias else None
        self.last_attention: torch.Tensor | None = None

    # ------------------------------------------------------------------ init
    def reset_parameters(self, generator: torch.Generator | None = None,
                         scheme: str = "matlab", scale: float = 0.12) -> None:
        """Initialise the weights.

        ``matlab`` reproduces the retired ``rgat.initModel``: every parameter is
        ``scale * randn``. ``glorot`` is the reference implementation's default.
        The choice changes the optimisation trajectory, so it is explicit.
        """
        def fill(t: torch.Tensor, fan_in: int, fan_out: int) -> None:
            with torch.no_grad():
                if scheme == "matlab":
                    t.normal_(0.0, 1.0, generator=generator).mul_(scale)
                elif scheme == "glorot":
                    limit = math.sqrt(6.0 / max(fan_in + fan_out, 1))
                    t.uniform_(-limit, limit, generator=generator)
                else:
                    raise ValueError("scheme must be 'matlab' or 'glorot'")

        for module in (self.kernel, self.attn_kernel):
            if module.basis_size is None:
                fill(module.weight, module.in_dim, module.units)
            else:
                fill(module.basis, module.in_dim, module.units)
                fill(module.coefficients, module.basis_size, 1)
        fill(self.relation_embedding, self.relation_dim, 1)
        if self.head_aggregation == "projection":
            fill(self.projection.weight, self.heads * self.units, self.units)
        if self.bias is not None:
            with torch.no_grad():
                self.bias.zero_()

    # --------------------------------------------------------------- forward
    def forward(self, H: torch.Tensor, *, return_attention: bool = False):
        if H.dim() == 2:
            H = H.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        B, N, din = H.shape
        top = self.topology
        if N != top.n_nodes:
            raise ValueError(f"Feature matrix has {N} nodes, graph has {top.n_nodes}.")
        if din != self.in_dim:
            raise ValueError(f"Feature matrix has width {din}, layer expects {self.in_dim}.")

        R, Hd, U, E = top.n_relations, self.heads, self.units, top.n_edges
        if self.feature_dropout and self.training:
            H = torch.nn.functional.dropout(H, self.feature_dropout)

        # Project every node under every relation and head at once.
        # [B, R, Hd, N, U] -> flattened over (R, N) so an edge is one gather.
        W = self.kernel()                                        # [R, Hd, din, U]
        proj = torch.einsum("bnd,rhdu->brhnu", H, W)
        proj_flat = proj.permute(0, 2, 1, 3, 4).reshape(B, Hd, R * N, U)

        logits = self._logits(proj_flat, B, Hd, E)               # [B, Hd, E, A]
        # The multiplicative style has already contracted the attention space,
        # so it arrives with A == 1 whatever ``attention_units`` says; keying
        # off the tensor rather than the configuration keeps the two in step.
        logits = logits.squeeze(-1) if logits.shape[-1] == 1 else logits.mean(-1)
        segments, n_segments = top.segments(self.attention_mode)
        alpha = segment_softmax(logits, segments, n_segments,
                                floor=self.softmax_floor, stable=self.stable_softmax)
        self.last_attention = alpha.detach()
        if self.support_dropout and self.training:
            alpha = torch.nn.functional.dropout(alpha, self.support_dropout)

        # Messages are the source projections under the edge's own relation.
        messages = proj_flat.index_select(2, top.rel_src)         # [B, Hd, E, U]
        weighted = messages * alpha.unsqueeze(-1)
        out = torch.zeros(B, Hd, N, U, dtype=weighted.dtype, device=weighted.device)
        out.index_add_(2, top.dst, weighted)

        out = self._aggregate_heads(out)
        if self.bias is not None:
            out = out + self.bias
        if squeeze:
            out = out.squeeze(0)
        if return_attention:
            return out, alpha
        return out

    def _logits(self, proj_flat: torch.Tensor, B: int, Hd: int, E: int) -> torch.Tensor:
        """Per-edge attention logits, ``[B, heads, edges, attention_units]``."""
        top = self.topology
        A = self.attn_kernel()                     # [R, Hd, 2U+dr, attn_units]
        U, dr = self.units, self.relation_dim
        R, N = top.n_relations, top.n_nodes

        if self.attention_style == "sum":
            # Additive: split the kernel into its source, destination and
            # relation-embedding halves and contract each before gathering, so
            # the per-edge work is two index_selects and an add.
            a_src, a_dst, a_rel = A[..., :U, :], A[..., U:2 * U, :], A[..., 2 * U:, :]
            proj = proj_flat.reshape(B, Hd, R, N, U)
            s = torch.einsum("bhrnu,rhua->bhrna", proj, a_src).reshape(B, Hd, R * N, -1)
            d = torch.einsum("bhrnu,rhua->bhrna", proj, a_dst).reshape(B, Hd, R * N, -1)
            rel_term = torch.einsum("rd,rhda->rha", self.relation_embedding, a_rel)
            raw = (s.index_select(2, top.rel_src)
                   + d.index_select(2, top.rel_dst)
                   + rel_term.permute(1, 0, 2)[:, top.rel, :].unsqueeze(0))
            return torch.nn.functional.leaky_relu(raw, self.leaky_relu_slope)

        # Multiplicative: project source and destination into the attention
        # space and take their scaled dot product, transformer style. The
        # relation-embedding term stays, as a per-relation bias on the score --
        # the same role it plays in the additive style, and the reason a
        # relational attention is not just an attention over a merged graph.
        q_kernel, k_kernel, a_rel = A[..., :U, :], A[..., U:2 * U, :], A[..., 2 * U:, :]
        proj = proj_flat.reshape(B, Hd, R, N, U)
        q = torch.einsum("bhrnu,rhua->bhrna", proj, q_kernel).reshape(B, Hd, R * N, -1)
        k = torch.einsum("bhrnu,rhua->bhrna", proj, k_kernel).reshape(B, Hd, R * N, -1)
        raw = (q.index_select(2, top.rel_dst) * k.index_select(2, top.rel_src))
        raw = raw.sum(-1, keepdim=True) / math.sqrt(self.attention_units)
        rel_bias = torch.einsum("rd,rhda->rha", self.relation_embedding, a_rel)
        raw = raw + rel_bias.permute(1, 0, 2)[:, top.rel, :].unsqueeze(0).mean(
            -1, keepdim=True)
        if self.leaky_relu_slope:
            raw = torch.nn.functional.leaky_relu(raw, self.leaky_relu_slope)
        return raw

    def _aggregate_heads(self, out: torch.Tensor) -> torch.Tensor:
        """``[B, heads, N, U]`` -> ``[B, N, out_dim]``."""
        if self.head_aggregation == "mean":
            return out.mean(1)
        if self.head_aggregation == "sum":
            return out.sum(1)
        B, Hd, N, U = out.shape
        stacked = out.permute(0, 2, 1, 3).reshape(B, N, Hd * U)
        if self.head_aggregation == "concat":
            return stacked
        return self.projection(stacked)

    def extra_repr(self) -> str:
        return (f"in_dim={self.in_dim}, units={self.units}, heads={self.heads}, "
                f"mode={self.attention_mode}, style={self.attention_style}, "
                f"aggregation={self.head_aggregation}, out_dim={self.out_dim}")


RGAT = RelationalGraphAttention
