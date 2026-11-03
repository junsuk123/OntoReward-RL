"""The control arm for orl/4: the registered semantic MLP with the SAME view rule
(status doc 20.9).

orl/4 hard-codes a rule the task declares (view_rule.py).  Against the plain semantic MLP
any gain could be the rule's alone, so orl-development/4's primary comparison is against
this arm instead: ``semantic_mlp_ppo`` (same phi44 input, same initialisation, bit for
bit) whose vertical mean is blended into orl/4's closed-gate law as the rule closes:

    mean_z = (1 - w) * mlp_z + w * (-softplus(g) * drone_vz)

with w the view rule and g ONE learned scalar, initialised to orl/4's damper gain
(``orl.HOLD_GAIN_INIT``).  Where the rule is open the MLP is untouched; where it is closed
both arms run the same damper on the same feature.  The lateral means are the MLP's.
The arm name carries the hash of this spec, so a checkpoint never binds to another.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import CurrentSemanticsConfig
from .features import FEATURE_NAMES
from .models import CurrentActorCritic
from .orl import HOLD_GAIN_INIT, _inverse_softplus
from .view_rule import check_config, closed_weight, rule_dict

ARM_PREFIX = "semantic_mlp_rule_ppo"
BASE_ARM = "semantic_mlp_ppo"
VZ_INDEX = FEATURE_NAMES.index("drone_vz")


def spec_dict() -> dict[str, Any]:
    return {"version": "semantic-mlp-rule/1", "base_arm": BASE_ARM,
            "vertical": "(1 - w) * mlp_z + w * (-softplus(g) * drone_vz)",
            "gain": f"one learned scalar g, init softplus^-1({HOLD_GAIN_INIT}) "
                    "(orl.HOLD_GAIN_INIT)",
            "lateral": "the MLP's own lateral means",
            "init": f"{BASE_ARM}'s registered initialisation, unchanged",
            "view_rule": rule_dict()}


def _digest() -> str:
    from ..direct_policy.contracts import digest
    return digest(spec_dict())


CONTROL_HASH = _digest()
CONTROL_ARM = f"{ARM_PREFIX}:{CONTROL_HASH[:16]}"
SPEC = MappingProxyType(spec_dict())


class RuledSemanticMLP(CurrentActorCritic):
    """semantic_mlp_ppo whose vertical mean follows the view rule's damper when closed."""

    def __init__(self, arm: str, config: CurrentSemanticsConfig, *, seed: int):
        if arm != CONTROL_ARM:
            raise ValueError(f"unknown arm {arm!r}; this code's rule control is {CONTROL_ARM}")
        super().__init__(BASE_ARM, config, seed=seed)
        check_config(config)
        self.arm = arm
        self.rule_gain = nn.Parameter(torch.tensor([_inverse_softplus(HOLD_GAIN_INIT)]))

    def _terms(self, mean: torch.Tensor, flat44: torch.Tensor):
        w = closed_weight(flat44)[:, 0]
        damp = -F.softplus(self.rule_gain) * flat44[:, VZ_INDEX]
        return w, damp, mean[:, 2]

    def forward(self, raw21: torch.Tensor, flat44: torch.Tensor,
                nodes: torch.Tensor) -> torch.Tensor:
        mean = super().forward(raw21, flat44, nodes)
        w, damp, mlp_z = self._terms(mean, flat44)
        vertical = (1.0 - w) * mlp_z + w * damp
        return torch.cat((mean[:, :2], vertical.unsqueeze(-1)), dim=-1)

    def vertical_terms(self, flat44: torch.Tensor) -> dict[str, torch.Tensor]:
        """(B,) terms in the gate probe's vocabulary: ``gate`` is the rule's open fraction
        1 - w, ``gated`` the MLP's share and ``hold`` the damper's share; the vertical mean
        is ``hold + gated``.  The MLP reads flat44 alone (its registered actor input)."""
        w, damp, mlp_z = self._terms(self.actor_head(flat44), flat44)
        return {"hold": w * damp, "gate": 1.0 - w, "gated": (1.0 - w) * mlp_z}

    def _parameter_groups(self):
        groups = []
        for name, params in super()._parameter_groups():
            groups.append((name, params + [self.rule_gain] if name == "actor_head" else params))
        return tuple(groups)


def control_model(arm: str, config: CurrentSemanticsConfig, *, seed: int) -> RuledSemanticMLP:
    return RuledSemanticMLP(arm, config, seed=seed)
