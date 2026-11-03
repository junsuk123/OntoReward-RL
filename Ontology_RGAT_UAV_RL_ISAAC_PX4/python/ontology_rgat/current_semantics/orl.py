"""ORL -- ontology reinforcement learning: the ontology as the policy's inductive bias
(user, 2026-11-02; status doc 20).

A convolution makes a CNN local, weight-shared and translation-equivariant; a
recurrence makes an LSTM carry state through learned gates.  ORL takes the same
kind of bias from the ontology alone, represented by an R-GAT.  The registered
arms concatenate an 8-D graph readout to raw21 (or read phi44 through an MLP), so
the actor can ignore the graph; ORL removes that freedom:

1. Locality -- no raw bypass.  The actor reads phi44 only through ontology nodes,
   each node type encodes only its own slots (its receptive field), and
   information reaches an action only along declared relations.
2. Weight sharing -- the 3D task is the reference's landing plane once per
   horizontal axis (landing/plane_graph.py).  The x and y planes are two
   instances of the same concepts: they share node encoders, relation weights and
   the lateral readout, the way a kernel is shared across positions.
3. Equivariance -- every concept declares its parity under the two mirror
   reflections of the world (``ROLES``): lateral concepts of a plane flip under
   that plane's mirror, the view margins of a plane swap, heading flips under
   both, everything else is invariant.  The actor mean is the group average over
   those four reflections, so a mirrored state gets exactly the mirrored command
   (status doc 20.1 measures the reflections on build_features: exact except the
   registered half-pixel principal point, |d view| <= 4.2e-3).
4. A learned gate placed by the ontology -- the validation points of this study
   (touchdown speed/tilt/rate limits and pad radius in ContactCondition, the
   camera footprint in ViewGeometry, vision age in ObservationQuality) meet in
   DescentEligibility, and the vertical mean is ``hold + sigmoid(eligibility) *
   descend``.  Like an LSTM gate it is learned (initialised half open), never a
   fixed rule, and the action is still the plain ``a_max * tanh(z)``.

Budget: the actor has the registered R-GAT arm's 2,747 parameters; the critic and
log_std are the common ones, so ORL vs semantic_mlp_ppo (same phi44 input, same
budget, no structure) is the CNN-vs-MLP comparison.  ORL is NOT a registered arm
of the v1 contract (no v1 hash moves); its checkpoints bind to ``ORL_HASH`` through
the arm name.

Two versions, both buildable, each with its own hash and arm (``ARMS``):

- ``orl/1`` (orl-development/1, status doc 20.3): bias 4 never worked.  Every trained
  gate was a constant (sd <= 0.027) and hold/descend were collinear (status doc 20.4).
  Three reasons, all in the design.  (a) The gate's gradient is
  ``sigmoid' * descend(Z)``, and descend started at x0.01 with a state-dependent sign,
  so it was ~0 and incoherent.  (b) VerticalDecision also read ObservationQuality and
  DescentEligibility, so hold(Z) could express any vision dependence without the gate.
  (c) Nothing fixed what "open" means, so the gate could be absorbed into the other two
  terms.
- ``orl/2`` (status doc 20.5) makes the gate the ONLY path from vision, contact and view
  to the vertical command, and fixes its meaning.  The vertical command is
  ``hold(Z) - sigmoid(gate(E)) * softplus(sink(Z))``, where Z = VerticalDecision now reads
  VerticalState alone.  Opening the gate can only add descent.  A closed gate leaves
  ``hold(Z)``, a law of the vertical state alone, e.g. an altitude hold.  At
  initialisation the gate is half open and sink = softplus(0) = ln 2; the hold bias
  ln(2)/2 cancels the mean, so the initial vertical command is ~0, as in every arm
  (pilot 2's drift).  The gate's gradient is then ``-sigmoid' * ln 2 = -0.17`` in EVERY
  state, a coherent sign from the first update.  Same parameters, same budget.  The
  encoder and lateral path are orl/1's, with the same per-seed initialisation.
- ``orl/3`` (status doc 20.7) fixes what a CLOSED gate means.  In orl/2 the gate learned
  (pilot drop 0.76), but ``hold(Z)`` was free in sign and level and settled on a
  ~0.2 m/s sink, so a closed gate still descended (status doc 20.6).  orl/3's vertical
  command is ``-softplus(gain(Z)) * (vz + sigmoid(gate(E)) * softplus(sink(Z)))``, where
  ``vz`` is VerticalState's own phi44 slot ``drone_vz`` (B(vz; 1 m/s), odd, zero at
  zero).  A closed gate is a damper on the own vertical velocity, with its equilibrium at
  vz = 0 in every state.  An open gate sets a sink-speed reference, and the gate's
  gradient ``-gain * sigmoid' * sink`` is negative in every state and at every point of
  training.  No learned term can shift the closed-gate equilibrium, because nothing is
  added outside the bracket.  The price is that the initial mean is NOT zero: the gate
  starts half open with a full-open sink of ``INIT_SINK_M_S``, so the untrained policy
  settles into a 0.091 m/s descent instead of drifting.  The damper gain starts at
  ``HOLD_GAIN_INIT``, measured (see there).  Graph, encoder, lateral path and
  parameter count are orl/2's.
"""
from __future__ import annotations

import itertools
import math
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .contracts import ACTION_DIMENSION, OBSERVATION_DIMENSION, CurrentSemanticsConfig
from .features import FEATURE_NAMES, FEATURES
from .graph import SELF, GraphSpec, reachable_within
from .models import (MODEL_ARCHITECTURE, CurrentActorCritic, RelationalAttentionLayer,
                     _check_inputs, _linear, _mlp, component_generator)

ORL_VERSIONS = ("orl/1", "orl/2", "orl/3")
CURRENT_VERSION = "orl/3"
ARM_PREFIX = "orl_ppo"
X, Y = "x", "y"
PLANES = (X, Y)

# Concept -> features of the x plane and of the y plane, each with the sign that aligns
# the y plane with the x plane's convention (status doc 20.1): camera y is -body y, so
# the pad's image coordinate along y is -eta; a_x follows +sin(pitch) and a_y
# -sin(roll) (thrust of the scipy "xyz" ENU/FLU attitude), and so do their rates.
# ``margins`` are (B(1 + v), B(1 - v)) of the aligned image coordinate v: they swap
# under that plane's mirror.  Every other plane entry is odd under it.
PLANE_ROLES = MappingProxyType({
    "RelativeGeometry": ((("relative_x", 1), ("relative_y", 1)),),
    "MotionState": ((("relative_vx", 1), ("relative_vy", 1)),
                    (("ugv_vx", 1), ("ugv_vy", 1)),
                    (("own_vx", 1), ("own_vy", 1))),
    "AttitudeState": ((("drone_sinTheta", 1), ("drone_sinPhi", -1)),
                      (("drone_pitchRate", 1), ("drone_rollRate", -1))),
    "ApproachSituation": ((("approach_ex", 1), ("approach_ey", 1)),
                          (("approach_vx", 1), ("approach_vy", 1))),
    "ViewGeometry": ((("view_xi", 1), ("view_eta", -1)),
                     (("view_xi_rate", 1), ("view_eta_rate", -1))),
    "ContactCondition": ((("contact_ex", 1), ("contact_ey", 1)),
                         (("contact_vx", 1), ("contact_vy", 1))),
})
MARGINS = MappingProxyType({X: ("view_margin_left", "view_margin_right"),
                            Y: ("view_margin_bottom", "view_margin_top")})
# Shared concepts: (feature, parity) with parity "even" (invariant under both mirrors)
# or "odd" (flips under each mirror: the heading).
SHARED_ROLES = MappingProxyType({
    "ObservationQuality": (("ugv_visionUpdated", "even"), ("ugv_visionAge", "even"),
                           ("drone_navigationValid", "even"), ("drone_navigationAge", "even"),
                           ("view_front_valid", "even")),
    "VerticalState": (("relative_height", "even"), ("drone_vz", "even"),
                      ("contact_vz", "even"), ("contact_tilt", "even"),
                      ("contact_rate", "even"), ("drone_cosPhi", "even"),
                      ("drone_cosTheta", "even"), ("approach_closure", "even")),
    "HeadingState": (("drone_sinPsi", "odd"), ("drone_cosPsi", "even"),
                     ("drone_yawRate", "odd")),
})
DECISIONS = ("LateralDecision", "DescentEligibility", "VerticalDecision")


def _node(concept: str, plane: str | None = None) -> str:
    return concept if plane is None else f"{concept}_{plane}"


def input_dimensions() -> dict[str, int]:
    dims = {c: len(r) for c, r in PLANE_ROLES.items()}
    dims["ViewGeometry"] += 2  # the margin pair
    dims.update({c: len(r) for c, r in SHARED_ROLES.items()})
    return dims


NODES = tuple([_node(c) for c in SHARED_ROLES]
              + [_node(c, p) for p in PLANES for c in (*PLANE_ROLES, "LateralDecision")]
              + ["DescentEligibility", "VerticalDecision"])
RELATIONS = ("qualifiesEstimate", "informsApproach", "affectsVisibility", "conditionsContact",
             "informsLanding", "gatesDescent", "informsVertical", SELF)


def _edges(version: str) -> tuple[tuple[str, str, str], ...]:
    edges = []
    for p in PLANES:
        g, m, a = (_node(c, p) for c in ("RelativeGeometry", "MotionState", "AttitudeState"))
        ap, o, c, lat = (_node(n, p) for n in ("ApproachSituation", "ViewGeometry",
                                               "ContactCondition", "LateralDecision"))
        edges += [("ObservationQuality", t, "qualifiesEstimate") for t in (ap, o, c)]
        edges += [(g, ap, "informsApproach"), (m, ap, "informsApproach")]
        edges += [(s, o, "affectsVisibility") for s in (g, m, a, "VerticalState", "HeadingState")]
        edges += [(s, c, "conditionsContact") for s in (g, m, a, "VerticalState")]
        edges += [(s, lat, "informsLanding") for s in (ap, o, c)]
        edges += [(s, "DescentEligibility", "gatesDescent") for s in (c, o)]
    edges += [("ObservationQuality", "DescentEligibility", "gatesDescent")]
    # orl/2 and orl/3: the vertical law reads the vertical state alone; quality, contact
    # and view reach the vertical command only through the gate (DescentEligibility).
    vertical = (("VerticalState", "ObservationQuality", "DescentEligibility")
                if version == "orl/1" else ("VerticalState",))
    edges += [(s, "VerticalDecision", "informsVertical") for s in vertical]
    return tuple(edges) + tuple((n, n, SELF) for n in NODES)


GRAPHS = MappingProxyType({
    version: GraphSpec(version=version, nodes=NODES, relations=RELATIONS,
                       edges=_edges(version), readout_node="VerticalDecision")
    for version in ORL_VERSIONS})
ORL_GRAPH = GRAPHS[CURRENT_VERSION]
HOPS = 3            # every input concept reaches the vertical command and its LateralDecision
EMBEDDING = 14      # node state width
LATERAL_HIDDEN = 31  # lateral readout width; with EMBEDDING 14 the actor has 2,747 parameters


def _flat_index() -> dict[str, int]:
    return {name: i for i, name in enumerate(FEATURE_NAMES)}


def _check_roles(version: str) -> None:
    """Every phi44 feature has exactly one role; each LateralDecision sees its own plane's
    concepts and the shared ones within ``HOPS`` and NOTHING of the other plane; the
    vertical command sees every concept except the approach (lateral only).  On orl/1 it
    sees them through VerticalDecision; on orl/2 and orl/3 VerticalDecision reads
    VerticalState alone and everything else arrives through DescentEligibility, i.e. the
    gate."""
    graph = GRAPHS[version]
    used = [f for roles in PLANE_ROLES.values() for pair in roles for f, _ in pair]
    used += [f for pair in MARGINS.values() for f in pair]
    used += [f for roles in SHARED_ROLES.values() for f, _ in roles]
    if sorted(used) != sorted(FEATURE_NAMES):
        raise AssertionError("ORL roles must use every phi44 feature exactly once")
    shared = set(SHARED_ROLES)
    for plane, other in ((X, Y), (Y, X)):
        reach = reachable_within(graph, _node("LateralDecision", plane), HOPS)
        if not ({_node(c, plane) for c in PLANE_ROLES} | shared) <= reach:
            raise AssertionError(f"LateralDecision_{plane} misses a concept within {HOPS} hops")
        if any(n.endswith(f"_{other}") for n in reachable_within(
                graph, _node("LateralDecision", plane), len(NODES))):
            raise AssertionError(f"LateralDecision_{plane} must not read the {other} plane")
    needed = ({_node(c, p) for c in PLANE_ROLES if c != "ApproachSituation" for p in PLANES}
              | shared)
    if version == "orl/1":
        if not needed <= reachable_within(graph, "VerticalDecision", HOPS):
            raise AssertionError(f"VerticalDecision misses a concept within {HOPS} hops")
        return
    if reachable_within(graph, "VerticalDecision", len(NODES)) != {"VerticalDecision",
                                                                    "VerticalState"}:
        raise AssertionError(f"{version}: VerticalDecision must read VerticalState alone")
    if not needed - {"VerticalState"} <= reachable_within(graph, "DescentEligibility", HOPS):
        raise AssertionError(f"{version}: DescentEligibility misses a concept within "
                             f"{HOPS} hops")
    if version == "orl/3" and "drone_vz" not in dict(SHARED_ROLES["VerticalState"]):
        raise AssertionError("orl/3: the hold damps VerticalState's own drone_vz slot")


for _graph_version in ORL_VERSIONS:
    _check_roles(_graph_version)


def group_elements() -> tuple[tuple[int, int], ...]:
    """The four reflections (s_x, s_y): identity, mirror x, mirror y, both."""
    return tuple(itertools.product((1, -1), (1, -1)))


def group_matrix(sx: int, sy: int) -> np.ndarray:
    """The signed permutation of phi44 under the reflection (sx, sy), from the declared
    parities: ``flat @ M.T`` is the reflected decision's phi44."""
    index, m = _flat_index(), np.zeros((len(FEATURE_NAMES), len(FEATURE_NAMES)))
    sign = {X: sx, Y: sy}
    for roles in PLANE_ROLES.values():
        for pair in roles:
            for plane, (name, _) in zip(PLANES, pair):
                m[index[name], index[name]] = sign[plane]
    for plane, (plus, minus) in MARGINS.items():
        if sign[plane] == 1:
            m[index[plus], index[plus]] = m[index[minus], index[minus]] = 1
        else:
            m[index[plus], index[minus]] = m[index[minus], index[plus]] = 1
    for roles in SHARED_ROLES.values():
        for name, parity in roles:
            m[index[name], index[name]] = sx * sy if parity == "odd" else 1
    return m


HOLD_BIAS_V2 = 0.5 * math.log(2.0)  # sigmoid(0) * softplus(0): initial vertical mean 0

# orl/3 (status doc 20.7).  The damper reads drone_vz = B(vz; VZ_SCALE) = vz / (|vz| + s),
# and a vertical action of a_max * tanh(u) with u = -gain * B(vz).  Being memoryless, it
# leaves a residual velocity against the seeded constant force (U(-0.75, 0.75) N per axis
# on 1.5 kg, i.e. up to 0.5 m/s^2 vertically).  HOLD_GAIN_INIT was MEASURED, untrained,
# with the gate closed, at d = 1 on seeds 995001-995024 (orl3/design_checks):
# - residual |vz| p90 was 0.39 / 0.14 / 0.11 / 0.092 / 0.089 m/s at gain 1 / 2 / 2.5 / 3 / 4;
# - the per-decision velocity jitter rose 7x from gain 3 to 4.
# Gain 3 is the smallest whose worst residual (0.093 m/s) is inside probe 2's 0.10 m/s.
# INIT_SINK_M_S is the reward field's sink speed (spatial-current-reward: align, then sink
# at 0.2 m/s), so a fully open untrained gate sinks at it and the half-open start at
# 0.091 m/s (measured, independent of the gain).
VZ_SCALE = {spec.name: spec.scale for spec in FEATURES}["drone_vz"]
HOLD_GAIN_INIT = 3.0
INIT_SINK_M_S = 0.2


def _inverse_softplus(y: float) -> float:
    return math.log(math.expm1(y))


def sink_reference(speed_m_s: float) -> float:
    """The drone_vz feature value of a descent at ``speed_m_s`` (positive)."""
    return speed_m_s / (speed_m_s + VZ_SCALE)


def spec_dict(version: str = CURRENT_VERSION) -> dict[str, Any]:
    if version not in ORL_VERSIONS:
        raise ValueError(f"unknown ORL version {version!r}; known {ORL_VERSIONS}")
    spec = {"version": version, "plane_roles": {k: [list(map(list, p)) for p in v]
                                                for k, v in PLANE_ROLES.items()},
            "margins": dict(MARGINS),
            "shared_roles": {k: [list(r) for r in v] for k, v in SHARED_ROLES.items()},
            "graph": GRAPHS[version].to_dict(), "hops": HOPS, "embedding": EMBEDDING,
            "lateral_hidden": LATERAL_HIDDEN, "group": "Z2 x Z2 mirror average",
            "vertical": "hold + sigmoid(eligibility) * descend", "lateral_bias": False,
            "critic": "common raw21 critic"}
    if version == "orl/2":
        spec["vertical"] = ("hold(Z) - sigmoid(gate(E)) * softplus(sink(Z)); Z = "
                            "VerticalDecision reads VerticalState alone, E = "
                            "DescentEligibility")
        spec["vertical_init"] = {
            "output_weight_scale": float(MODEL_ARCHITECTURE["actor_output_init_scale"]),
            "scaled": ["lateral_out", "hold", "sink", "gate"], "gate_bias": 0.0,
            "sink_bias": 0.0, "hold_bias": "ln(2)/2 = sigmoid(0) * softplus(0)"}
    if version == "orl/3":
        spec["vertical"] = ("-softplus(gain(Z)) * (drone_vz + sigmoid(gate(E)) * "
                            "softplus(sink(Z))); Z = VerticalDecision reads VerticalState "
                            "alone, E = DescentEligibility, drone_vz = VerticalState's own "
                            "phi44 slot; a closed gate damps vz to 0 in every state")
        spec["vertical_init"] = {
            "output_weight_scale": float(MODEL_ARCHITECTURE["actor_output_init_scale"]),
            "scaled": ["lateral_out", "gain", "sink", "gate"], "gate_bias": 0.0,
            "gain_bias": f"softplus^-1({HOLD_GAIN_INIT})",
            "sink_bias": f"softplus^-1(B({INIT_SINK_M_S} m/s; {VZ_SCALE} m/s))",
            "drone_vz_scale_m_s": VZ_SCALE}
    return spec


def _digest(version: str) -> str:
    from ..direct_policy.contracts import digest
    return digest(spec_dict(version))


HASHES = MappingProxyType({version: _digest(version) for version in ORL_VERSIONS})
ARMS = MappingProxyType({version: f"{ARM_PREFIX}:{HASHES[version][:16]}"
                         for version in ORL_VERSIONS})
ORL_VERSION = CURRENT_VERSION
ORL_HASH = HASHES[CURRENT_VERSION]
ORL_ARM = ARMS[CURRENT_VERSION]


def version_of(arm: str) -> str:
    """The ORL version an arm name binds to; a stale or foreign hash is refused."""
    for version, name in ARMS.items():
        if arm == name:
            return version
    raise ValueError(f"unknown arm {arm!r}; the ORL arms of this code are "
                     f"{dict(ARMS)}")


# ---------------------------------------------------------------------- model

class ORLEncoder(nn.Module):
    """Typed node encoders (shared across planes), decision-node initial states, and ONE
    relational layer applied ``HOPS`` times over the version's graph (``GRAPHS``)."""

    def __init__(self, *, seed: int, version: str = CURRENT_VERSION):
        super().__init__()
        dims = input_dimensions()
        self.encoders = nn.ModuleDict({
            concept: _linear(dims[concept], EMBEDDING,
                             component_generator(seed, f"orl.encoder.{concept}"))
            for concept in dims})
        self.decision = nn.Parameter(torch.zeros(len(DECISIONS), EMBEDDING))
        self.layer = RelationalAttentionLayer(EMBEDDING, len(RELATIONS),
                                              generator=component_generator(seed, "orl.layer"))
        for name, values in zip(("source", "target", "relation"), GRAPHS[version].edge_index()):
            self.register_buffer(name, torch.as_tensor(values, dtype=torch.long),
                                 persistent=False)
        index = _flat_index()
        for concept, roles in PLANE_ROLES.items():
            for plane_i, plane in enumerate(PLANES):
                names = [pair[plane_i][0] for pair in roles]
                signs = [float(pair[plane_i][1]) for pair in roles]
                if concept == "ViewGeometry":
                    names += list(MARGINS[plane]); signs += [1.0, 1.0]
                self.register_buffer(f"idx_{concept}_{plane}",
                                     torch.tensor([index[n] for n in names]), persistent=False)
                self.register_buffer(f"sgn_{concept}_{plane}", torch.tensor(signs),
                                     persistent=False)
        for concept, roles in SHARED_ROLES.items():
            self.register_buffer(f"idx_{concept}", torch.tensor([index[n] for n, _ in roles]),
                                 persistent=False)

    def node_states(self, flat44: torch.Tensor) -> torch.Tensor:
        """(B, len(NODES), EMBEDDING) after ``HOPS`` rounds of message passing."""
        batch = flat44.shape[0]
        states = {}
        for concept in SHARED_ROLES:
            states[concept] = torch.tanh(self.encoders[concept](
                flat44[:, getattr(self, f"idx_{concept}")]))
        for concept in PLANE_ROLES:
            for plane in PLANES:
                inputs = flat44[:, getattr(self, f"idx_{concept}_{plane}")] \
                    * getattr(self, f"sgn_{concept}_{plane}")
                states[_node(concept, plane)] = torch.tanh(self.encoders[concept](inputs))
        decision = torch.tanh(self.decision)
        for plane in PLANES:
            states[_node("LateralDecision", plane)] = decision[0].expand(batch, -1)
        states["DescentEligibility"] = decision[1].expand(batch, -1)
        states["VerticalDecision"] = decision[2].expand(batch, -1)
        h = torch.stack([states[n] for n in NODES], dim=1)
        for _ in range(HOPS):
            h = self.layer(h, self.source, self.target, self.relation)
        return h


class ORLReadout(nn.Module):
    """Lateral: one bias-free MLP shared by both LateralDecision nodes (the group average
    cancels any lateral output bias, so it would be a dead parameter).  Vertical, with
    Z = VerticalDecision and E = DescentEligibility:

    - orl/1: ``hold(Z) + sigmoid(gate(E)) * descend(Z)``;
    - orl/2: ``hold(Z) - sigmoid(gate(E)) * softplus(descend(Z))`` -- ``descend`` is the
      sink, so an open gate can only add descent (module docstring);
    - orl/3: ``-softplus(hold(Z)) * (vz + sigmoid(gate(E)) * softplus(descend(Z)))`` --
      ``hold`` is the damper's gain, and a closed gate holds vz = 0.
    """

    def __init__(self, *, seed: int, output_scale: float, version: str = CURRENT_VERSION):
        super().__init__()
        self.version = version
        self.lateral_hidden = _linear(EMBEDDING, LATERAL_HIDDEN,
                                      component_generator(seed, "orl.lateral.hidden"))
        self.lateral_out = _linear(LATERAL_HIDDEN, 1,
                                   component_generator(seed, "orl.lateral.out"), bias=False)
        self.hold = _linear(EMBEDDING, 1, component_generator(seed, "orl.vertical.hold"))
        self.descend = _linear(EMBEDDING, 1, component_generator(seed, "orl.vertical.descend"))
        self.gate = _linear(EMBEDDING, 1, component_generator(seed, "orl.vertical.gate"))
        with torch.no_grad():  # near-zero initial mean, gate half open (MODEL_ARCHITECTURE)
            scaled = (self.lateral_out, self.hold, self.descend)
            for layer in scaled if version == "orl/1" else (*scaled, self.gate):
                layer.weight.mul_(output_scale)
            for layer in (self.hold, self.descend, self.gate):
                layer.bias.zero_()
            if version == "orl/2":  # sigmoid(0) * softplus(0) cancelled: initial mean 0
                self.hold.bias.fill_(HOLD_BIAS_V2)
            if version == "orl/3":  # damper gain HOLD_GAIN_INIT, full-open sink INIT_SINK_M_S
                self.hold.bias.fill_(_inverse_softplus(HOLD_GAIN_INIT))
                self.descend.bias.fill_(_inverse_softplus(sink_reference(INIT_SINK_M_S)))
        self.lateral_index = [NODES.index(_node("LateralDecision", p)) for p in PLANES]
        self.eligibility_index = NODES.index("DescentEligibility")
        self.vertical_index = NODES.index("VerticalDecision")

    def vertical_terms(self, h: torch.Tensor, vz: torch.Tensor | None = None
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(hold, gate, gated term) of the vertical output, each (B, 1); the output is
        ``hold + gated``.  orl/3 needs ``vz`` (B, 1), the drone_vz feature."""
        z, e = h[:, self.vertical_index], h[:, self.eligibility_index]
        gate = torch.sigmoid(self.gate(e))
        if self.version == "orl/1":
            return self.hold(z), gate, gate * self.descend(z)
        if self.version == "orl/2":
            return self.hold(z), gate, -gate * F.softplus(self.descend(z))
        if vz is None:
            raise ValueError("orl/3's hold damps drone_vz; pass it")
        gain = F.softplus(self.hold(z))
        return -gain * vz, gate, -gain * gate * F.softplus(self.descend(z))

    def forward(self, h: torch.Tensor, vz: torch.Tensor | None = None) -> torch.Tensor:
        lateral = self.lateral_out(torch.tanh(self.lateral_hidden(h[:, self.lateral_index])))
        hold, _, gated = self.vertical_terms(h, vz)
        return torch.cat((lateral.squeeze(-1), hold + gated), dim=-1)


class ORLActorCritic(CurrentActorCritic):
    """The ORL actor (phi44 through the ontology only, mirror-averaged) with the common
    raw21 critic and global log_std of every registered arm."""

    def __init__(self, arm: str, config: CurrentSemanticsConfig, *, seed: int):
        nn.Module.__init__(self)
        self.version = version_of(arm)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative int")
        design = config.model
        self.arm, self.config, self.seed = arm, config, seed
        self.actor_input = "flat44"
        self.encoder = ORLEncoder(seed=seed, version=self.version)
        self.actor_head = ORLReadout(
            seed=seed, output_scale=float(MODEL_ARCHITECTURE["actor_output_init_scale"]),
            version=self.version)
        self.critic = _mlp(OBSERVATION_DIMENSION, design.critic_hidden, 1,
                           component_generator(seed, "critic"))
        self.log_std = nn.Parameter(torch.full((ACTION_DIMENSION,),
                                               float(design.initial_log_std)))
        self.register_buffer("action_maximum",
                             torch.tensor(config.action.maximum_m_s2, dtype=torch.float32),
                             persistent=False)
        group = group_elements()
        self.register_buffer("group", torch.tensor(
            np.stack([group_matrix(*g) for g in group]), dtype=torch.float32),
            persistent=False)
        self.register_buffer("action_signs", torch.tensor(
            [[sx, sy, 1] for sx, sy in group], dtype=torch.float32), persistent=False)
        self.vz_index = _flat_index()["drone_vz"]  # even under every mirror

    def graph_readout(self, nodes: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("ORL reads phi44; use encoder.node_states(flat44)")

    def _vz(self, flat44: torch.Tensor) -> torch.Tensor:
        return flat44[:, self.vz_index:self.vz_index + 1]

    def unaveraged(self, flat44: torch.Tensor) -> torch.Tensor:
        """The ontology network before the mirror average (diagnostics)."""
        return self.actor_head(self.encoder.node_states(flat44), self._vz(flat44))

    def _reflected(self, flat44: torch.Tensor) -> torch.Tensor:
        """(members * B, 44): every mirror image of every row, member-major."""
        members, batch = self.group.shape[0], flat44.shape[0]
        return torch.einsum("gij,bj->gbi", self.group, flat44).reshape(members * batch, -1)

    def forward(self, raw21: torch.Tensor, flat44: torch.Tensor,
                nodes: torch.Tensor) -> torch.Tensor:
        _check_inputs(raw21, flat44, nodes)
        batch, members = flat44.shape[0], self.group.shape[0]
        out = self.unaveraged(self._reflected(flat44)).reshape(members, batch, ACTION_DIMENSION)
        return (out * self.action_signs[:, None, :]).mean(0)

    def vertical_terms(self, flat44: torch.Tensor) -> dict[str, torch.Tensor]:
        """Mirror-averaged (B,) hold, gate and gated term of the vertical mean (before
        tanh), as the actor computes them; the vertical mean is ``hold + gated``."""
        batch, members = flat44.shape[0], self.group.shape[0]
        reflected = self._reflected(flat44)
        terms = self.actor_head.vertical_terms(self.encoder.node_states(reflected),
                                               self._vz(reflected))
        return {name: term.reshape(members, batch).mean(0)
                for name, term in zip(("hold", "gate", "gated"), terms)}


def orl_model(arm: str, config: CurrentSemanticsConfig, *, seed: int) -> ORLActorCritic:
    return ORLActorCritic(arm, config, seed=seed)
