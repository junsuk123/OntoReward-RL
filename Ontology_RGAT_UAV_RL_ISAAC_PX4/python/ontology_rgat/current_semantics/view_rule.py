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

view-rule/2 (status doc 20.12, orl/5) keeps w and adds the rest of the same field's
vertical law, still on the policy's own estimate.  orl-development/4's motion_ood failures
(orl4/diagnostics_motion) lost the pad at ~1 m while 0.85 m behind an accelerating pad and
then descended on a stale estimate: the closed rule only HELD altitude, below 0.5 m the
blind-touchdown exemption reopened descent on an estimate 0.6 m off, and 9 of 122
re-acquired.  The field itself asks for two more things, and view-rule/2 states both:

- where the view is lost it asks for a CLIMB toward the search height, so the closed law
  tracks ``vz_ref = w * min(climb_speed, climb_gain * max(search_height - h, 0))``
  (0 without an estimate: with no height there is no search height to climb to);
- (``ALIGNMENT_EDGE``) it asks for no descent outside the alignment cone
  r = r_0 + k_r * h, so the rule also closes there:
  ``W = max(w, ramp((d - r) / ramp_m))``, with the field's own lost-view ramp.

``rule_terms`` returns (W, vz_ref); view-rule/1 (``closed_weight``) is unchanged.
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


# ------------------------------------------------------------------ view-rule/2
# Chosen by the variant rule declared in orl4/diagnostics_motion/selection_rule.txt BEFORE
# its run (climb_diag.json, diagnostic seeds 995001-995048, both arms' orl/4-era weights, ten
# cell conditions): mean success climb 0.861, climb + cone 0.874 (>= +0.010 required), and
# neither arm's ID mean lower (status doc 20.12).
ALIGNMENT_EDGE = True
VIEW_RULE_V2 = MappingProxyType({
    **VIEW_RULE, "version": "view-rule/2",
    "source": "reward /4's lost-view field on the policy's own decoded estimate: its weight "
              "w AND its climb" + (" AND its alignment cone" if ALIGNMENT_EDGE else ""),
    "climb_speed_m_s": 0.5, "climb_gain_per_s": 1.0, "search_height_m": 2.5,
    "reference": "vz_ref = w * min(climb_speed, climb_gain * max(search_height - h, 0)); "
                 "0 without an estimate",
    **({"alignment_radius_m": 0.25, "alignment_radius_per_m": 0.5,
        "closing": "W = max(w, clip((d - (alignment_radius_m + alignment_radius_per_m * h)) "
                   "/ ramp_m, 0, 1))"} if ALIGNMENT_EDGE else {"closing": "W = w"})})
RULES = MappingProxyType({"view-rule/1": VIEW_RULE, "view-rule/2": VIEW_RULE_V2})


def check_config_v2(config: Any) -> None:
    """view-rule/1's binding plus the climb (and cone) constants of the same field."""
    check_config(config)
    field = config.task.reward.field
    declared = {"climb_speed_m_s": field.lost_view.climb_speed_m_s,
                "climb_gain_per_s": field.lost_view.climb_gain_per_s,
                "search_height_m": field.lost_view.search_height_m}
    if ALIGNMENT_EDGE:
        declared.update(alignment_radius_m=field.alignment_radius_m,
                        alignment_radius_per_m=field.alignment_radius_per_m)
    differs = {k: (v, VIEW_RULE_V2[k]) for k, v in declared.items()
               if float(v) != VIEW_RULE_V2[k]}
    if differs:
        raise ValueError(f"view rule: the config's field constants differ from "
                         f"{VIEW_RULE_V2['version']}: {differs}")


def rule_terms(flat44: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """view-rule/2: (W, vz_ref), each (B, 1).  W in [0, 1] is how far descent is closed;
    vz_ref (m/s, up positive) is what the closed law tracks.  Both read an age, a height
    and a distance only, so both are invariant under the two mirrors."""
    w = closed_weight(flat44)
    valid = flat44[:, _INDEX[AGE]] < 1.0
    h = (_signed(flat44[:, _INDEX[H]], _SCALE[H]) - VIEW_RULE["contact_height_m"]).clamp_min(0.0)
    climb = (VIEW_RULE_V2["climb_gain_per_s"]
             * (VIEW_RULE_V2["search_height_m"] - h).clamp_min(0.0)
             ).clamp(max=VIEW_RULE_V2["climb_speed_m_s"])
    climb = torch.where(valid, climb, torch.zeros_like(climb))
    reference = w * climb.unsqueeze(-1)
    if not ALIGNMENT_EDGE:
        return w, reference
    d = torch.hypot(_signed(flat44[:, _INDEX[X]], _SCALE[X]),
                    _signed(flat44[:, _INDEX[Y]], _SCALE[Y]))
    radius = VIEW_RULE_V2["alignment_radius_m"] + VIEW_RULE_V2["alignment_radius_per_m"] * h
    edge = ((d - radius) / VIEW_RULE["ramp_m"]).clamp(0.0, 1.0).unsqueeze(-1)
    return torch.maximum(w, edge), reference


def rule_dict_for(version: str) -> dict[str, Any]:
    return dict(RULES[version])
