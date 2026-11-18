"""Relational bond equilibrium in front of the ontology R-GAT (``bond/1``, 2026-11-14).

The ablation of WMON v0.1 on its own benchmark (tools/wmon_ablation.py, every variant
re-trained) found that its gain is carried by ONE part, the L4-L5 relational energy:
position RMSE full 0.104 m, without the FFT wave 0.101, without the adaptive particle
confidence 0.103, bonds alone 0.102 -- and without the bonds 3.05 m.  What the bonds do
is reconstruct an atom whose evidence is missing (on-site weight 0) from the atoms it
is related to, as the minimiser of a typed energy.  This module transplants exactly
that, and nothing else, into the registered R-GAT:

    E(x) = sum_i c_i ||x_i - h0_i||^2 + sum_(s -r-> t) k_r ||x_t - a_r * x_s||^2 + eps ||x||^2

* h0 = tanh(input(nodes)), the R-GAT's own node projection;
* c_i = softplus(w_i) * (v_i + (1 - v_i) * sigmoid(rho_i)), the node's evidence weight:
  v_i is its registered validity (G, M, P, C: estimate valid AND navigation valid;
  O: front valid; Q, A: always), rho_i a learned residual trust of an invalid node
  (initialised at 0.1, because M and C also carry always-valid channels), and
  LandingContext has no evidence at all (c = 0): it is defined by its bonds, as a
  missing WMON particle is;
* k_r = softplus(kappa_r) > 0 and a_r (per feature) are learned per ontology relation
  type over the registered semantic edges (self loops carry no bond);
* the minimiser solves (diag(c) + eps I + sum_e k_r u_e u_e^T) x = c * h0 per feature,
  u_e = e_t - a_r e_s: one batched 8x8 SPD solve, exact and differentiable.

x then goes through the R-GAT's relational attention hops and LandingContext readout
unchanged.  The FFT wave (L0) and the adaptive confidence (L1-L2) are deliberately NOT
transplanted: they carried no gain, and the current-observation path has no history.
The actor head is narrowed (66 -> 64) to keep the parameter budget of the registered
arms (bond/1: 4,007 vs ontology_rgat_ppo 4,016).
"""
from __future__ import annotations

import math
from types import MappingProxyType
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ..direct_policy.contracts import digest
from .contracts import (ACTION_DIMENSION, MODEL_ARCHITECTURE, NODE_ORDER, OBSERVATION_CHANNELS,
                        OBSERVATION_DIMENSION, CurrentSemanticsConfig)
from .features import NODE_FEATURES
from .graph import SEMANTIC_EDGES, graph_for_arm
from .models import (CurrentActorCritic, GraphEncoder, _linear, _mlp,  # noqa: F401
                     component_generator)

BOND_VERSION = "bond/1"
ARM_PREFIX = "bond_rgat_ppo"
BASE_ARM = "ontology_rgat_ppo"
HEAD_WIDTH = 64            # ontology_rgat_ppo: 66; two units pay for the bond parameters
RIDGE = 1e-3
INVALID_TRUST_INIT = 0.1
STIFFNESS_INIT = 1.0
_Q, _G, _M, _A, _P, _O, _C, _L = NODE_ORDER
ESTIMATE, FRONT, ALWAYS, NONE = "estimate_and_navigation_valid", "front_valid", "always", "none"
EVIDENCE = MappingProxyType({_Q: ALWAYS, _G: ESTIMATE, _M: ESTIMATE, _A: ALWAYS,
                             _P: ESTIMATE, _O: FRONT, _C: ESTIMATE, _L: NONE})
BOND_RELATIONS = tuple(dict.fromkeys(r for _, _, r in SEMANTIC_EDGES))
_AGE = next(c for c in OBSERVATION_CHANNELS if c.name == "ugv_visionAge")
# estimate_valid := ugv_visionAge < 1 s (contracts.ESTIMATE_VALIDITY_RULE); age is encoded
# u = x / (x + s), monotone, so the rule is u < 1 / (1 + s) on the encoded value.
ESTIMATE_THRESHOLD = 1.0 / (1.0 + float(_AGE.scale))
_SLOT = MappingProxyType({
    "vision_age": (NODE_ORDER.index(_Q), NODE_FEATURES[_Q].index("ugv_visionAge")),
    "navigation_valid": (NODE_ORDER.index(_Q), NODE_FEATURES[_Q].index("drone_navigationValid")),
    "front_valid": (NODE_ORDER.index(_O), NODE_FEATURES[_O].index("view_front_valid"))})


def spec_dict() -> dict[str, Any]:
    return {"version": BOND_VERSION, "base_arm": BASE_ARM, "head_width": HEAD_WIDTH,
            "evidence": dict(EVIDENCE), "estimate_threshold_encoded": ESTIMATE_THRESHOLD,
            "relations": list(BOND_RELATIONS),
            "edges": [list(e) for e in SEMANTIC_EDGES], "ridge": RIDGE,
            "init": {"stiffness": STIFFNESS_INIT, "transform": 1.0, "evidence_weight": 1.0,
                     "invalid_trust": INVALID_TRUST_INIT},
            "energy": ("sum_i c_i ||x_i - h0_i||^2 + sum_(s-r->t) k_r ||x_t - a_r x_s||^2 "
                       "+ eps ||x||^2, solved exactly per feature")}


SPEC_HASH = digest(spec_dict())
ARM = f"{ARM_PREFIX}:{SPEC_HASH[:16]}"


def _inverse_softplus(y: float) -> float:
    return math.log(math.expm1(y))


def node_validity(nodes: torch.Tensor) -> torch.Tensor:
    """v (B, 8) in {0, 1} from the node input itself (raw channels sit bit-exact in
    their node's first slots, features.py)."""
    qa, sa = _SLOT["vision_age"]
    qn, sn = _SLOT["navigation_valid"]
    of, sf = _SLOT["front_valid"]
    estimate = (nodes[:, qa, sa] < ESTIMATE_THRESHOLD).to(nodes.dtype)
    navigation = (nodes[:, qn, sn] > 0.5).to(nodes.dtype)
    front = (nodes[:, of, sf] > 0.5).to(nodes.dtype)
    rule = {ALWAYS: torch.ones_like(estimate), ESTIMATE: estimate * navigation,
            FRONT: front, NONE: torch.zeros_like(estimate)}
    return torch.stack([rule[EVIDENCE[node]] for node in NODE_ORDER], dim=-1)


class RelationalBondLayer(nn.Module):
    """The equilibrium of the typed bond energy (module docstring), exact per feature."""

    def __init__(self, dim: int, *, generator: torch.Generator | None = None):
        super().__init__()
        self.dim = int(dim)
        nodes = NODE_ORDER
        self.evidence_nodes = tuple(n for n in nodes if EVIDENCE[n] != NONE)
        self.conditional_nodes = tuple(n for n in nodes if EVIDENCE[n] in (ESTIMATE, FRONT))
        self.evidence_weight = nn.Parameter(torch.full((len(self.evidence_nodes),),
                                                       _inverse_softplus(1.0)))
        p = INVALID_TRUST_INIT
        self.invalid_trust = nn.Parameter(torch.full((len(self.conditional_nodes),),
                                                     math.log(p / (1.0 - p))))
        self.stiffness = nn.Parameter(torch.full((len(BOND_RELATIONS),),
                                                 _inverse_softplus(STIFFNESS_INIT)))
        self.transform = nn.Parameter(torch.ones(len(BOND_RELATIONS), self.dim))
        index = {n: i for i, n in enumerate(nodes)}
        self.register_buffer("source", torch.tensor([index[s] for s, _, _ in SEMANTIC_EDGES]),
                             persistent=False)
        self.register_buffer("target", torch.tensor([index[t] for _, t, _ in SEMANTIC_EDGES]),
                             persistent=False)
        self.register_buffer("relation", torch.tensor(
            [BOND_RELATIONS.index(r) for _, _, r in SEMANTIC_EDGES]), persistent=False)
        self.register_buffer("evidence_index", torch.tensor(
            [index[n] for n in self.evidence_nodes]), persistent=False)
        self.register_buffer("conditional_index", torch.tensor(
            [index[n] for n in self.conditional_nodes]), persistent=False)

    def weights(self, validity: torch.Tensor) -> torch.Tensor:
        """c (B, 8): softplus(w) * (v + (1 - v) sigmoid(rho)); 0 for LandingContext."""
        trust = torch.ones_like(validity)
        conditional = validity[:, self.conditional_index]
        trust = trust.index_copy(
            1, self.conditional_index,
            conditional + (1.0 - conditional) * torch.sigmoid(self.invalid_trust))
        base = validity.new_zeros(validity.shape)
        base = base.index_copy(1, self.evidence_index,
                               F.softplus(self.evidence_weight).expand(validity.shape[0], -1))
        return base * trust

    def bond_matrix(self) -> torch.Tensor:
        """sum_e k_r u_e u_e^T per feature, (d, 8, 8); independent of the batch."""
        n = len(NODE_ORDER)
        k = F.softplus(self.stiffness)[self.relation]                     # (E,)
        a = self.transform[self.relation]                                 # (E, d)
        matrix = a.new_zeros(self.dim, n, n)
        t, s = self.target, self.source
        diag_t = k.unsqueeze(-1).expand(-1, self.dim)                     # k
        diag_s = k.unsqueeze(-1) * a * a                                  # k a^2
        cross = -k.unsqueeze(-1) * a                                      # -k a
        flat = matrix.view(self.dim, n * n)
        flat = flat.index_add(1, t * n + t, diag_t.T)
        flat = flat.index_add(1, s * n + s, diag_s.T)
        flat = flat.index_add(1, t * n + s, cross.T)
        flat = flat.index_add(1, s * n + t, cross.T)
        return flat.view(self.dim, n, n)

    def forward(self, h0: torch.Tensor, validity: torch.Tensor) -> torch.Tensor:
        batch, n, d = h0.shape
        c = self.weights(validity)                                         # (B, N)
        system = (self.bond_matrix().unsqueeze(0)
                  + torch.diag_embed(c).unsqueeze(1)
                  + RIDGE * torch.eye(n, dtype=h0.dtype, device=h0.device))  # (B, d, N, N)
        rhs = (c.unsqueeze(-1) * h0).transpose(1, 2).unsqueeze(-1)       # (B, d, N, 1)
        return torch.linalg.solve(system, rhs).squeeze(-1).transpose(1, 2)

    def energy(self, x: torch.Tensor, h0: torch.Tensor, validity: torch.Tensor) -> torch.Tensor:
        """E(x) per sample (B,), for tests and audits."""
        c = self.weights(validity)
        onsite = (c.unsqueeze(-1) * (x - h0) ** 2).sum((1, 2))
        k = F.softplus(self.stiffness)[self.relation]
        a = self.transform[self.relation]
        bond = (k.unsqueeze(-1) * (x[:, self.target] - a * x[:, self.source]) ** 2).sum((1, 2))
        return onsite + bond + RIDGE * (x ** 2).sum((1, 2))


class BondGraphEncoder(GraphEncoder):
    """The R-GAT encoder (same graph, layer, readout and init streams) with the bond
    equilibrium between the node projection and the relational hops."""

    def __init__(self, *, dim: int, hops: int, seed: int):
        super().__init__(graph_for_arm(BASE_ARM), dim=dim, hops=hops, seed=seed)
        self.bond = RelationalBondLayer(dim, generator=component_generator(seed, "encoder.bond"))

    def node_states(self, nodes: torch.Tensor) -> torch.Tensor:
        shape = (len(self.spec.nodes), self.node_input)
        if nodes.dim() != 3 or tuple(nodes.shape[1:]) != shape:
            raise ValueError(f"node input must be (B, {shape[0]}, {shape[1]}), "
                             f"got {tuple(nodes.shape)}")
        h = self.bond(torch.tanh(self.input(nodes)), node_validity(nodes))
        for _ in range(self.hops):
            h = self.layer(h, self.source, self.target, self.relation)
        return h


class BondActorCritic(CurrentActorCritic):
    """ontology_rgat_ppo with the bond equilibrium (raw21 + graph readout actor, the
    common raw21 critic and global log_std of every registered arm)."""

    def __init__(self, arm: str, config: CurrentSemanticsConfig, *, seed: int):
        nn.Module.__init__(self)
        if arm != ARM:
            raise ValueError(f"unknown bond arm {arm!r}; this code builds {ARM}")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative int")
        design = config.model
        self.arm, self.config, self.seed = arm, config, seed
        self.actor_input = "raw21+graph"
        self.encoder = BondGraphEncoder(dim=design.node_embedding, hops=design.hops, seed=seed)
        self.actor_head = _mlp(OBSERVATION_DIMENSION + design.node_embedding, HEAD_WIDTH,
                               ACTION_DIMENSION, component_generator(seed, "actor_head"))
        with torch.no_grad():
            self.actor_head[-1].weight.mul_(float(MODEL_ARCHITECTURE["actor_output_init_scale"]))
            self.actor_head[-1].bias.zero_()
        self.critic = _mlp(OBSERVATION_DIMENSION, design.critic_hidden, 1,
                           component_generator(seed, "critic"))
        self.log_std = nn.Parameter(torch.full((ACTION_DIMENSION,),
                                               float(design.initial_log_std)))
        self.register_buffer("action_maximum",
                             torch.tensor(config.action.maximum_m_s2, dtype=torch.float32),
                             persistent=False)


def bond_model(arm: str, config: CurrentSemanticsConfig, *, seed: int) -> BondActorCritic:
    return BondActorCritic(arm, config, seed=seed)
