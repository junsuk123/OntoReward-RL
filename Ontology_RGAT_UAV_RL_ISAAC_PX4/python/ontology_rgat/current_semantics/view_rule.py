"""The view rule: no descent without a recent view of the pad (status doc 20.9).

orl/3's gate could close, and a closed gate held altitude, but PPO left it half open when
blind: under the sampled training distribution closing was worth +1.9 discounted per
episode (SE ~3), too little to find (status doc 20.8).  The rule therefore stops being
learned.  It is the task's own declaration -- reward /4's lost-view weight
(``reward.lost_view_weight``, ``task.reward.field.lost_view``) -- evaluated on the
policy's OWN estimate instead of the truth, so no constant is chosen here:

    w = ramp((age - 0.5) / 0.5) * max(ramp((h - 0.5) / 0.25), ramp((d - 0.75) / 0.25))

with age the policy's vision age, h its estimated height above the contact height and d
its estimated horizontal distance to the pad.  w = 1 closes the rule (the view is lost
away from the last stretch of a landing); below 0.5 m AND within 0.75 m of the centre a
blind touchdown stays allowed, because every local landing goes blind there.  Without an
estimate (vision age sentinel) there is nothing to descend on and w = 1.

Two arms read it (status doc 20.9): orl/4 multiplies its learned gate by (1 - w), and the
control ``semantic_mlp_rule_ppo`` blends the registered semantic MLP's vertical output
into the same damper.  The rule is truth-free: phi44's raw slots only.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Any

import torch

from .features import FEATURE_NAMES, FEATURES

VIEW_RULE = MappingProxyType({
    "version": "view-rule/1",
    "source": "reward /4 lost_view_weight on the policy's own decoded estimate",
    "after_s": 0.5, "full_after_s": 1.0, "terminal_height_m": 0.5,
    "terminal_radius_m": 0.75, "ramp_m": 0.25, "contact_height_m": 0.0672,
    "no_estimate": "closed (w = 1)",
    "height": "max(relative_height - contact_height_m, 0)",
    "distance": "hypot(relative_x, relative_y)"})
_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}
_SCALE = {spec.name: spec.scale for spec in FEATURES}
AGE, X, Y, H = "ugv_visionAge", "relative_x", "relative_y", "relative_height"
for _name in (AGE, X, Y, H):
    if _SCALE[_name] is None:
        raise AssertionError(f"view rule: {_name} needs a fixed phi44 scale")


def check_config(config: Any) -> None:
    """The rule restates the task's lost-view constants and the local contact height so
    its hash is self-contained; refuse a config that says otherwise."""
    lost = config.task.reward.field.lost_view
    declared = {"after_s": lost.after_s, "full_after_s": lost.full_after_s,
                "terminal_height_m": lost.terminal_height_m,
                "terminal_radius_m": lost.terminal_radius_m, "ramp_m": lost.ramp_m,
                "contact_height_m": config.local_plant.contact_height_m}
    differs = {k: (v, VIEW_RULE[k]) for k, v in declared.items() if float(v) != VIEW_RULE[k]}
    if differs:
        raise ValueError(f"view rule: the config's lost-view / contact constants differ from "
                         f"{VIEW_RULE['version']}: {differs}")


def _signed(u: torch.Tensor, scale: float) -> torch.Tensor:
    return scale * u / (1.0 - u.abs())


def closed_weight(flat44: torch.Tensor) -> torch.Tensor:
    """w in [0, 1], shape (B, 1): how far the rule has closed descent for each decision.
    Invariant under both mirrors (it reads an age, a height and a distance)."""
    u_age = flat44[:, _INDEX[AGE]]
    valid = u_age < 1.0
    safe = torch.where(valid, u_age, torch.zeros_like(u_age))
    age = _SCALE[AGE] * safe / (1.0 - safe)
    ramp = lambda x: x.clamp(0.0, 1.0)
    w_age = ramp((age - VIEW_RULE["after_s"])
                 / (VIEW_RULE["full_after_s"] - VIEW_RULE["after_s"]))
    h = (_signed(flat44[:, _INDEX[H]], _SCALE[H]) - VIEW_RULE["contact_height_m"]).clamp_min(0.0)
    d = torch.hypot(_signed(flat44[:, _INDEX[X]], _SCALE[X]),
                    _signed(flat44[:, _INDEX[Y]], _SCALE[Y]))
    where = torch.maximum(ramp((h - VIEW_RULE["terminal_height_m"]) / VIEW_RULE["ramp_m"]),
                          ramp((d - VIEW_RULE["terminal_radius_m"]) / VIEW_RULE["ramp_m"]))
    w = torch.where(valid, w_age * where, torch.ones_like(u_age))
    return w.unsqueeze(-1)


def rule_dict() -> dict[str, Any]:
    return dict(VIEW_RULE)
