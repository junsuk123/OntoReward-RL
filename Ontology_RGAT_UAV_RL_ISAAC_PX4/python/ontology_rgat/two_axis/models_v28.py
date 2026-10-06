"""Raw-semantic bypass + typed/grouped relational residual (upstream v2.8).

Actor and critic have independent encoders. Identical component RNG streams
make flat and graph raw paths exactly equal initially; graph context starts
at zero, NOT its residual output weight (which would block first gradients).
"""
from __future__ import annotations

import torch
from torch import nn
from ..landing.plane_graph import FEATURE_CHANNELS
from .ontology_v28 import GRAPH_EDGES, NODE_NAMES, RELATION_NAMES, READOUT_GROUPS


class GroupedEncoder(nn.Module):
    """Typed relational encoder over however many nodes the task declares.

    Defaults to the planar nine so the ported contract is untouched; the
    spatial task passes a schema with its extension nodes appended. The four
    grouped readouts are preserved whatever the node count, because an
    extension joins an existing group rather than adding a fifth.
    """

    def __init__(self, hidden=8, relation_dim=4, schema=None):
        super().__init__()
        from ..landing.ontology import schema as ontology_schema

        schema = schema or ontology_schema()
        node_names = schema.node_names
        edges = schema.edges
        relations = schema.relation_names
        self.node_count = len(node_names)
        self.readout_groups = schema.readout_groups
        r = len(relations)
        self.kernel = nn.Parameter(.1 * torch.randn(r, 12, hidden))
        self.attention = nn.Parameter(.1 * torch.randn(r, 2*hidden + relation_dim))
        self.relation_embedding = nn.Parameter(.1 * torch.randn(r, relation_dim))
        self.local = nn.Linear(12, hidden)
        self.readout = nn.Linear(len(self.readout_groups)*hidden,
                                 len(self.readout_groups))
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        self.register_buffer("src", torch.tensor([node_names.index(s) for s,_,_ in edges]))
        self.register_buffer("dst", torch.tensor([node_names.index(d) for _,_,d in edges]))
        self.register_buffer("rel", torch.tensor([relations.index(r) for _,r,_ in edges]))

    def node_embeddings(self, x):
        k = self.kernel[self.rel]
        source = torch.einsum("bei,eih->beh", x[:,self.src], k)
        target = torch.einsum("bei,eih->beh", x[:,self.dst], k)
        embedded = self.relation_embedding[self.rel].expand(x.shape[0], -1, -1)
        score = torch.nn.functional.leaky_relu(
            (torch.cat((source,target,embedded),-1)*self.attention[self.rel]).sum(-1), .2)
        # One softmax over ALL incoming edges, not one independently per type.
        messages = []
        for node in range(self.node_count):
            mask = self.dst == node
            weights = torch.softmax(score[:,mask], -1)
            messages.append((weights.unsqueeze(-1)*source[:,mask]).sum(1))
        return torch.tanh(torch.stack(messages,1) + self.local(x))

    def forward(self, x):
        h = self.node_embeddings(x)
        grouped = torch.cat([h[:,list(group)].mean(1)
                             for group in self.readout_groups], -1)
        return torch.tanh(self.readout(grouped))


def _raw_net(input_dim, output_dim, hidden):
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, output_dim))


class ReferenceHead(nn.Module):
    def __init__(self, mode, role, config, seed, initial_log_std=-.7, *,
                 packet_dim=26, action_dim=2, descent_axis=1,
                 initialization="torch-default", minimum_log_std=-5., graph_planes=1,
                 ontology=None, state_dependent_log_std=False):
        super().__init__()
        from ..landing.ontology import schema as ontology_schema

        # Default is the planar nine, so every existing caller is unchanged.
        self.ontology = ontology or ontology_schema()
        self.node_count = self.ontology.node_count
        self.mode, self.role = mode, role
        size = action_dim if role == "actor" else 1
        self.descent_axis = descent_axis
        self.minimum_log_std = float(minimum_log_std)
        if graph_planes not in (1,2):
            raise ValueError('graph planes must be 1 or 2')
        self.graph_planes = graph_planes
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + (31001 if role == "actor" else 31002))
            graph_dim = self.node_count * len(FEATURE_CHANNELS) * graph_planes
            self.raw = _raw_net(packet_dim if mode == "ppo_vector_canonical" else graph_dim,
                                size, config.policy_hidden_dimension)
            if initialization == "reference-v28-gaussian":
                # mlpInit.m: N(0, 1/fan_in), final-layer gain .1,
                # and zero bias. Preserve historical callers by opt-in.
                layers = [layer for layer in self.raw if isinstance(layer, nn.Linear)]
                for i, layer in enumerate(layers):
                    gain = .1 if i == len(layers)-1 else 1.
                    nn.init.normal_(layer.weight, std=gain / layer.in_features**.5)
                    nn.init.zeros_(layer.bias)
            elif initialization != "torch-default":
                raise ValueError("unknown reference head initialization")
            if mode == "ppo_ontology_rgat":
                torch.manual_seed(seed + (31003 if role == "actor" else 31004))
                self.encoder = GroupedEncoder(config.hidden_dimension,
                                              config.relation_dimension,
                                              schema=self.ontology)
                self.residual = nn.Linear(4*graph_planes, size, bias=False)
                nn.init.normal_(self.residual.weight, std=.05)
        if role == "actor":
            self.log_std = nn.Parameter(torch.full((action_dim,), float(initial_log_std)))
            # Scheduled ceiling on exploration. The shipped 1.0 is inert; a
            # trainer that anneals writes this per iteration. Measured
            # 2026-10-06: a clone whose MEAN lands 95.8 % at difficulty 0.0
            # lands 0.0 % sampled at the shipped sigma 0.333, so every PPO
            # batch is success-free and nothing points at a landing.
            self.maximum_log_std = 1.0
            self.state_dependent_log_std = bool(state_dependent_log_std)
            if self.state_dependent_log_std:
                # One global sigma cannot be both exploratory at 2.5 m and
                # precise inside the last 0.5 m, which is what this task
                # needs. Final layer starts at zero, so a fresh head is
                # numerically identical to the global-sigma policy.
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed + 31005)
                    self.log_std_net = _raw_net(
                        packet_dim if mode == "ppo_vector_canonical" else graph_dim,
                        action_dim, config.policy_hidden_dimension)
                final = [l for l in self.log_std_net if isinstance(l, nn.Linear)][-1]
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)

    def relational_delta(self, graphs):
        readout = self.encoder.readout
        frozen = not readout.weight.requires_grad and not readout.bias.requires_grad
        if (not torch.is_grad_enabled() or frozen) and (
                torch.count_nonzero(readout.weight).item() == 0
                and torch.count_nonzero(readout.bias).item() == 0):
            # Exactly zero context gives exactly zero residual. Skip its
            # expensive encoding in warmup, but NEVER skip a trainable zero
            # readout in backward: that gradient activates the relation path.
            return graphs.new_zeros((graphs.shape[0], self.residual.out_features))
        if self.graph_planes == 2:
            context = self.encoder(
                graphs.reshape(-1, self.node_count, len(FEATURE_CHANNELS))
            ).reshape(graphs.shape[0], 8)
        else:
            context = self.encoder(graphs)
        delta = self.residual(context)
        if self.role == "actor":
            gate = self.ontology.node_names.index("DescentEligibility")
            eligibility = (graphs[:,:,gate,0].amin(1) if self.graph_planes == 2
                           else graphs[:,gate,0]).clamp(0,1)
            # Public inhibit is already included in DescentEligibility. The
            # raw policy remains intact; the common supervisor is still final.
            axis = self.descent_axis
            z = torch.where(delta[:,axis] < 0, delta[:,axis]*eligibility, delta[:,axis])
            delta = torch.cat((delta[:,:axis], z[:,None], delta[:,axis+1:]), -1)
        return delta

    def forward(self, packets, graphs):
        output = self.raw(packets if self.mode == "ppo_vector_canonical" else graphs.flatten(1))
        if self.mode == "ppo_ontology_rgat":
            output = output + self.relational_delta(graphs)
        if self.role == "critic":
            return output.squeeze(-1)
        log_std = self.log_std.expand_as(output)
        if self.state_dependent_log_std:
            source = packets if self.mode == "ppo_vector_canonical" else graphs.flatten(1)
            log_std = log_std + self.log_std_net(source)
        return output, log_std.clamp(
            self.minimum_log_std, self.maximum_log_std).exp()

    @staticmethod
    def raw_log_probability(raw, mu, std):
        from .models import TwoAxisActor
        return TwoAxisActor.raw_log_probability(raw, mu, std)

    def set_adaptation(self, config, enabled):
        if self.mode != "ppo_ontology_rgat":
            return
        for p in self.encoder.parameters():
            p.requires_grad_(enabled)
        for p in self.residual.parameters():
            p.requires_grad_(enabled)
        if config.freeze_static_backbone:
            self.encoder.relation_embedding.requires_grad_(False)
            for p in self.encoder.local.parameters():
                p.requires_grad_(False)
        for p in self.raw.parameters():
            p.requires_grad_(not (enabled and config.preserve_raw_during_adaptation))
        if self.role == "actor":
            self.log_std.requires_grad_(not (enabled and config.preserve_raw_during_adaptation))


@torch.no_grad()
def relational_contribution(agent, packets, graphs):
    """Measure the output contribution, not just nonzero attention weights."""
    if agent.mode != "ppo_ontology_rgat":
        return {"applicable": False}
    actor_delta = agent.actor.relational_delta(graphs)
    critic_delta = agent.critic.relational_delta(graphs)
    return {"applicable": True, "samples": int(graphs.shape[0]),
            "actor_readout_norm": float(agent.actor.encoder.readout.weight.norm()),
            "critic_readout_norm": float(agent.critic.encoder.readout.weight.norm()),
            "actor_delta_max_abs": float(actor_delta.abs().max()),
            "critic_delta_max_abs": float(critic_delta.abs().max()),
            "relation_active_on_probe": bool(actor_delta.abs().max() > 1e-8),
            "warning": "A zero output is flat-equivalent; nonzero output alone proves no benefit."}
