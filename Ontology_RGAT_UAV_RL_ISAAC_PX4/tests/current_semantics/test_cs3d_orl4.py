"""orl/4, its rule control and orl-development/4 (status doc 20.9).

The view rule is the task's own lost-view weight on the policy's own estimate, so it is
tested against ``reward.lost_view_weight`` on decoded observations, for mirror invariance
and for its config binding.  orl/4 must be orl/3 where the rule is open and the closed-gate
damper where it is closed.  The control must be the registered semantic MLP, bit for bit,
except where the rule acts.  orl-development/4 must compare orl/4 with the control and
register only after both arms' pilots and probes pass.
"""
from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from ontology_rgat.current_semantics import orl, orl_development as od, rule_control as rc
from ontology_rgat.current_semantics import view_rule as vr
from ontology_rgat.current_semantics.contracts import (DEFAULT_CONFIG_PATH, contract_manifest,
                                                       load_config)
from ontology_rgat.current_semantics.features import FEATURE_NAMES
from ontology_rgat.current_semantics.models import (CurrentActorCritic, build_model,
                                                    load_checkpoint, save_checkpoint)
from ontology_rgat.current_semantics.observation import decode
from ontology_rgat.current_semantics.reward import lost_view_weight

from test_cs3d_models import batch

CONFIG = load_config(DEFAULT_CONFIG_PATH)
INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def _set(flat, *, age=None, h=None, ex=None, ey=None):
    """phi44 rows with the rule's inputs replaced (registered encodings)."""
    out = flat.clone()
    enc = lambda v, s: v / (abs(v) + s)
    if age is not None:
        out[:, INDEX[vr.AGE]] = 1.0 if age == float("inf") else age / (age + 3.0)
    for name, value, scale in ((vr.H, h, 2.0), (vr.X, ex, 1.0), (vr.Y, ey, 1.0)):
        if value is not None:
            out[:, INDEX[name]] = enc(value, scale)
    return out


def test_the_rule_is_the_tasks_lost_view_weight_on_the_policys_own_estimate():
    raw, flat, _ = batch(50, 256)
    w = vr.closed_weight(flat)[:, 0].numpy()
    lost = CONFIG.task.reward.field.lost_view
    ref = []
    for row in raw.numpy().astype(np.float64):
        seen = decode(row)
        ref.append(1.0 if not seen.estimate_valid else lost_view_weight(
            seen.vision_age_s, max(seen.h - vr.VIEW_RULE["contact_height_m"], 0.0),
            float(np.hypot(seen.ex, seen.ey)), lost))
    np.testing.assert_allclose(w, ref, atol=2e-6)
    assert 0.0 < w.mean() < 1.0  # the batch exercises both sides


@pytest.mark.parametrize("age, h, d, expected", [
    (0.2, 2.0, 0.0, 0.0),    # fresh: open at any height
    (0.75, 2.0, 0.0, 0.5),   # half way through the age ramp
    (1.5, 2.0, 0.0, 1.0),    # stale, high: closed
    (1.5, 0.3, 0.3, 0.0),    # stale in the last stretch of a landing: open
    (1.5, 0.3, 1.5, 1.0),    # stale, low but far off the pad: closed
])
def test_the_rule_closes_descent_only_when_the_view_is_lost_away_from_touchdown(age, h, d,
                                                                                expected):
    _, flat, _ = batch(51, 4)
    flat = _set(flat, age=age, h=h + vr.VIEW_RULE["contact_height_m"], ex=d, ey=0.0)
    torch.testing.assert_close(vr.closed_weight(flat)[:, 0],
                               torch.full((4,), expected), atol=1e-5, rtol=0)


def test_no_estimate_closes_the_rule():
    _, flat, _ = batch(52, 8)
    none = _set(flat, age=float("inf"))
    for name in ("relative_x", "relative_y", "relative_height"):
        none[:, INDEX[name]] = 0.0  # the sentinel's zeroed estimate
    assert (vr.closed_weight(none) == 1.0).all()


def test_the_rule_is_mirror_invariant_and_bound_to_the_task_constants():
    net = build_model(orl.ARMS["orl/4"], CONFIG, seed=0)
    _, flat, _ = batch(53, 64)
    for g in range(4):
        torch.testing.assert_close(vr.closed_weight(flat @ net.group[g].T),
                                   vr.closed_weight(flat), atol=0, rtol=0)
    lost = CONFIG.task.reward.field.lost_view
    moved = replace(CONFIG, task=replace(CONFIG.task, reward=replace(
        CONFIG.task.reward, field=replace(CONFIG.task.reward.field,
                                          lost_view=replace(lost, after_s=0.7)))))
    with pytest.raises(ValueError, match="view rule"):
        build_model(orl.ARMS["orl/4"], moved, seed=0)
    with pytest.raises(ValueError, match="view rule"):
        build_model(rc.CONTROL_ARM, moved, seed=0)


@pytest.mark.parametrize("seed", [0, 1])
def test_orl4_is_orl3_where_the_rule_is_open_and_the_damper_where_it_is_closed(seed):
    torch.manual_seed(seed)
    orl3 = build_model(orl.ARMS["orl/3"], CONFIG, seed=seed)
    with torch.no_grad():
        for p in orl3.actor_head.parameters():
            p.add_(0.3 * torch.randn_like(p))
    orl4 = build_model(orl.ARMS["orl/4"], CONFIG, seed=seed)
    orl4.load_state_dict(orl3.state_dict())  # same parameters, same names
    raw, flat, nodes = batch(54 + seed, 32)
    fresh, lost = _set(flat, age=0.1), _set(flat, age=2.0, h=2.0)
    with torch.no_grad():
        torch.testing.assert_close(orl4(raw, fresh, nodes), orl3(raw, fresh, nodes))
        closed = orl4(raw, lost, nodes)
        orl3.actor_head.gate.bias.fill_(-50.0)
        torch.testing.assert_close(closed, orl3(raw, lost, nodes), atol=1e-6, rtol=0)
        level = lost.clone()
        level[:, INDEX["drone_vz"]] = 0.0
        assert orl4(raw, level, nodes)[:, 2].abs().max() < 1e-6  # holds, whatever the gate


def test_orl4_gate_gradient_is_never_positive_and_vanishes_only_where_the_rule_closes():
    net = build_model(orl.ARMS["orl/4"], CONFIG, seed=0)
    raw, flat, nodes = batch(56, 24)
    flat = torch.cat((_set(flat[:12], age=0.1), _set(flat[12:], age=2.0, h=2.0)))
    grads = []
    for i in range(len(flat)):
        net.zero_grad()
        net(raw[i:i + 1], flat[i:i + 1], nodes[i:i + 1])[0, 2].backward()
        grads.append(net.actor_head.gate.bias.grad.item())
    grads = torch.tensor(grads)
    assert (grads[:12] < -0.05).all() and (grads[12:].abs() < 1e-7).all()


def test_the_control_is_the_semantic_mlp_except_where_the_rule_acts():
    control = build_model(rc.CONTROL_ARM, CONFIG, seed=3)
    mlp = CurrentActorCritic("semantic_mlp_ppo", CONFIG, seed=3)
    assert all(torch.equal(control.state_dict()[k], v) for k, v in mlp.state_dict().items())
    assert control.parameter_report()["total"] == mlp.parameter_report()["total"] + 1 == 4009
    raw, flat, nodes = batch(57, 32)
    with torch.no_grad():
        for p in (*control.actor_head.parameters(),):
            p.add_(0.3 * torch.randn_like(p))
        mlp.load_state_dict({k: v for k, v in control.state_dict().items() if k != "rule_gain"})
        fresh, lost = _set(flat, age=0.1), _set(flat, age=2.0, h=2.0)
        torch.testing.assert_close(control(raw, fresh, nodes), mlp(raw, fresh, nodes))
        closed, free = control(raw, lost, nodes), mlp(raw, lost, nodes)
        torch.testing.assert_close(closed[:, :2], free[:, :2])  # lateral untouched
        damp = -torch.nn.functional.softplus(control.rule_gain) * lost[:, INDEX["drone_vz"]]
        torch.testing.assert_close(closed[:, 2], damp, atol=1e-6, rtol=0)
        terms = control.vertical_terms(flat)
        torch.testing.assert_close(terms["hold"] + terms["gated"],
                                   control(raw, flat, nodes)[:, 2], atol=1e-6, rtol=0)
    assert torch.nn.functional.softplus(build_model(rc.CONTROL_ARM, CONFIG, seed=0)
                                        .rule_gain).item() == pytest.approx(orl.HOLD_GAIN_INIT)


def test_the_control_trains_every_parameter_and_binds_its_own_checkpoints(tmp_path):
    control = build_model(rc.CONTROL_ARM, CONFIG, seed=0)
    raw, flat, nodes = batch(58, 64)
    flat = torch.cat((_set(flat[:32], age=0.1), _set(flat[32:], age=2.0, h=2.0)))
    control(raw, flat, nodes).square().sum().backward()
    groups = {id(p) for g in control.optimizer_groups() for p in g["params"]}
    assert id(control.rule_gain) in groups and control.rule_gain.grad.abs().item() > 0
    manifest = contract_manifest(CONFIG)
    path = save_checkpoint(tmp_path / "control.pt", control, manifest=manifest)
    load_checkpoint(path, build_model(rc.CONTROL_ARM, CONFIG, seed=0), manifest=manifest)
    with pytest.raises(ValueError):
        load_checkpoint(path, CurrentActorCritic("semantic_mlp_ppo", CONFIG, seed=0),
                        manifest=manifest)
    with pytest.raises(ValueError, match="unknown arm"):
        build_model("semantic_mlp_rule_ppo:0000000000000000", CONFIG, seed=0)


def test_orl_development_4_binds_both_specs_and_needs_both_pilots(tmp_path):
    plan, development = od.load_plan(od.DEFAULT_PLAN_PATHS["orl/4"], CONFIG)
    assert od.DEFAULT_PLAN_PATHS["orl/4"].name == "orl_development_v4.yaml"
    assert plan.protocol == "orl-development/4" and plan.arm == orl.ARMS["orl/4"]
    assert od.CONTROLS["orl/4"] == rc.CONTROL_ARM
    resolved = od.resolved(plan, development, CONFIG)
    assert resolved["primary_comparison"] == rc.CONTROL_ARM
    assert resolved["control"]["hash"] == rc.CONTROL_HASH
    assert od.part_dir(tmp_path, "orl/4", "control") == tmp_path / "orl4" / "control"
    with pytest.raises(ValueError):
        od.part_dir(tmp_path, "orl/3", "control")
    assert od.probe_version(rc.CONTROL_ARM) == "orl/4"
    assert od.probe_hash("orl/4") != od.probe_hash("orl/3")
    assert od.GATE_PROBES["orl/4"]["max_blind_sink_m_s"] == 0.10
    assert "no ORL G4 learnability pilot" in od.pilot_blocker(CONFIG, tmp_path, "orl/4")


def test_the_v4_verdict_is_orl_minus_the_rule_control():
    _, development = od.load_plan(od.DEFAULT_PLAN_PATHS["orl/4"], CONFIG)
    from ontology_rgat.current_semantics.development import CELLS
    seeds = development.training_seeds
    summary = lambda v, active=True: {"cells": {c: {"success": v, "unsafe": 0.05}
                                                for c in CELLS},
                                      "gate_probe": {"gate_active": active}}
    stored = {arm: {s: summary(0.6) for s in seeds} for arm in (od.PRIMARY, *od.SECONDARY)}
    control = {s: summary(0.8) for s in seeds}
    better = od.judge(development, {s: summary(0.9 + 0.01 * i) for i, s in enumerate(seeds)},
                      stored, "orl-development/4", control)
    assert better["orl_vs_rule_control"] == "better" and better["rule"] == od.RULE_V4
    assert better["primary"]["mean"] == pytest.approx(0.12)
    assert better["metrics"]["all_cells"]["control_minus_semantic_mlp_ppo"]["mean"] == \
        pytest.approx(0.2)
    same = od.judge(development, {s: summary(0.8) for s in seeds}, stored,
                    "orl-development/4", {s: summary(0.8, s != 12) for s in seeds})
    assert same["orl_vs_rule_control"] == "not established"
    assert same["gate_active_models"] == "9/10" and not same["reads_as_test_of_the_gate"]
    with pytest.raises(ValueError, match="needs"):
        od.judge(development, control, stored, "orl-development/4")
    with pytest.raises(ValueError, match="has no"):
        od.judge(development, control, stored, "orl-development/3", control)


def test_orl_develop_refuses_an_unknown_part(tmp_path):
    status, payload = od.run(DEFAULT_CONFIG_PATH, od.DEFAULT_PLAN_PATHS["orl/4"], tmp_path,
                             part="control")
    assert status == 3 and payload["status"] == "REFUSED"  # not registered yet


def test_a_control_probe_episode_records_the_rule():
    from ontology_rgat.current_semantics.pilot import _init_worker
    _init_worker(str(DEFAULT_CONFIG_PATH))
    state = build_model(rc.CONTROL_ARM, CONFIG, seed=0).state_dict()
    episode = od._probe_work((rc.CONTROL_ARM, state, od.GATE_PROBE["seeds"][0]))
    assert episode["refused"] is None
    gate = np.asarray(episode["gate"])
    assert gate.size > 0 and ((gate >= 0) & (gate <= 1)).all()
