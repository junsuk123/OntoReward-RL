"""orl/5, its rule control and orl-development/5 (status doc 20.12).

view-rule/2 is the task field's lost-view law on the policy's own estimate: view-rule/1's
closing weight plus the field's climb (and, with ``ALIGNMENT_EDGE``, its alignment cone),
so it is tested against the reward's own constants and ``desired_velocity``.  orl/5 must be
orl/4 where the rule is open and a damper toward the climb where it is closed; the control
must be the registered semantic MLP except where the rule acts.  Nothing registered before
(orl/1-/4, control v1, the orl-development/4 plan and probe) may move.
"""
from __future__ import annotations

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
from ontology_rgat.current_semantics.reward import desired_velocity, lost_view_weight

from test_cs3d_models import batch

CONFIG = load_config(DEFAULT_CONFIG_PATH)
INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}
FIELD = CONFIG.task.reward.field
CONTACT = vr.VIEW_RULE["contact_height_m"]


def _set(flat, *, age=None, h=None, ex=None, ey=None, vz=None):
    """phi44 rows with the rule's inputs replaced (registered encodings); ``h`` is the
    height ABOVE the contact height."""
    out = flat.clone()
    enc = lambda v, s: v / (abs(v) + s)
    if age is not None:
        out[:, INDEX[vr.AGE]] = 1.0 if age == float("inf") else age / (age + 3.0)
    for name, value, scale in ((vr.H, None if h is None else h + CONTACT, 2.0),
                               (vr.X, ex, 1.0), (vr.Y, ey, 1.0), ("drone_vz", vz, 1.0)):
        if value is not None:
            out[:, INDEX[name]] = enc(value, scale)
    return out


def climb(h: float) -> float:
    lost = FIELD.lost_view
    return min(lost.climb_speed_m_s, lost.climb_gain_per_s * max(lost.search_height_m - h, 0.0))


def test_nothing_registered_before_moves():
    assert orl.ARMS["orl/4"] == "orl_ppo:af6f4fb0994a5042"  # orl-development/4 binds it
    assert rc.CONTROL_ARM == "semantic_mlp_rule_ppo:46967b8a9be973e4"
    assert rc.CONTROL_ARMS["semantic-mlp-rule/1"] == rc.CONTROL_ARM
    plan, development = od.load_plan(od.DEFAULT_PLAN_PATHS["orl/4"], CONFIG)
    assert od.plan_hash(plan, development, CONFIG).startswith("528727af37cdf27e")
    assert od.probe_hash("orl/4").startswith("b4cc19767446b43e")
    assert orl.CURRENT_VERSION == "orl/5" and orl.ORL_ARM == orl.ARMS["orl/5"]
    assert od.DEFAULT_PLAN_PATH == od.DEFAULT_PLAN_PATHS["orl/5"]
    assert vr.ALIGNMENT_EDGE in (True, False)  # decided by the declared variant rule
    assert len(set(orl.ARMS.values())) == len(orl.ORL_VERSIONS)


def test_view_rule_2_is_the_fields_lost_view_law_on_the_policys_own_estimate():
    raw, flat, _ = batch(60, 256)
    closing, reference = (x[:, 0].numpy() for x in vr.rule_terms(flat))
    for row, W, ref in zip(raw.numpy().astype(np.float64), closing, reference):
        seen = decode(row)
        if not seen.estimate_valid:
            assert W == 1.0 and ref == 0.0
            continue
        h, d = max(seen.h - CONTACT, 0.0), float(np.hypot(seen.ex, seen.ey))
        w = lost_view_weight(seen.vision_age_s, h, d, FIELD.lost_view)
        assert ref == pytest.approx(w * climb(h), abs=2e-5)
        if not vr.ALIGNMENT_EDGE:
            assert W == pytest.approx(w, abs=2e-6)
            continue
        assert W >= w - 2e-6
        if W > w + 1e-4:  # the cone closed it: exactly where the field asks for no descent
            assert desired_velocity(np.array([d, 0.0]), h, FIELD, 0.0)[2] == 0.0
    assert 0.0 < closing.mean() < 1.0 and reference.max() > 0.0


@pytest.mark.parametrize("age, h, d, expected_w, expected_ref", [
    (0.2, 2.0, 0.0, 0.0, 0.0),     # fresh, aligned: open, nothing to track
    (1.5, 1.0, 0.0, 1.0, 0.5),     # stale, high: closed, climbs at the field's 0.5 m/s
    (1.5, 2.3, 0.0, 1.0, 0.2),     # near the search height: the climb slows down
    (1.5, 0.3, 0.3, 0.0, 0.0),     # the blind last stretch of a landing: open
    (0.2, 1.0, 1.5, None, 0.0),    # fresh but outside the alignment cone (r = 0.75 m)
])
def test_rule_2_cases(age, h, d, expected_w, expected_ref):
    _, flat, _ = batch(61, 4)
    W, ref = vr.rule_terms(_set(flat, age=age, h=h, ex=d, ey=0.0))
    if expected_w is None:
        expected_w = 1.0 if vr.ALIGNMENT_EDGE else 0.0
    torch.testing.assert_close(W[:, 0], torch.full((4,), expected_w), atol=1e-5, rtol=0)
    torch.testing.assert_close(ref[:, 0], torch.full((4,), expected_ref), atol=1e-5, rtol=0)


def test_no_estimate_closes_the_rule_and_holds():
    _, flat, _ = batch(62, 8)
    none = _set(flat, age=float("inf"))
    for name in ("relative_x", "relative_y", "relative_height"):
        none[:, INDEX[name]] = 0.0
    W, ref = vr.rule_terms(none)
    assert (W == 1.0).all() and (ref == 0.0).all()


def test_rule_2_is_mirror_invariant_and_bound_to_the_field():
    net = build_model(orl.ARMS["orl/5"], CONFIG, seed=0)
    _, flat, _ = batch(63, 64)
    for g in range(4):
        for a, b in zip(vr.rule_terms(flat @ net.group[g].T), vr.rule_terms(flat)):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    from dataclasses import replace
    moved = replace(CONFIG, task=replace(CONFIG.task, reward=replace(
        CONFIG.task.reward, field=replace(FIELD, lost_view=replace(
            FIELD.lost_view, search_height_m=3.0)))))
    with pytest.raises(ValueError, match="view rule"):
        build_model(orl.ARMS["orl/5"], moved, seed=0)
    with pytest.raises(ValueError, match="view rule"):
        build_model(rc.CONTROL_ARM_V2, moved, seed=0)
    build_model(orl.ARMS["orl/4"], moved, seed=0)  # view-rule/1 never read the climb


@pytest.mark.parametrize("seed", [0, 1])
def test_orl5_is_orl4_where_the_rule_is_open_and_climbs_where_it_is_closed(seed):
    torch.manual_seed(seed)
    orl4 = build_model(orl.ARMS["orl/4"], CONFIG, seed=seed)
    with torch.no_grad():
        for p in orl4.actor_head.parameters():
            p.add_(0.3 * torch.randn_like(p))
    orl5 = build_model(orl.ARMS["orl/5"], CONFIG, seed=seed)
    orl5.load_state_dict(orl4.state_dict())  # same parameters, same names
    raw, flat, nodes = batch(64 + seed, 32)
    fresh = _set(flat, age=0.1, ex=0.0, ey=0.0)
    lost = _set(flat, age=2.0, h=1.0, ex=0.0, ey=0.0)
    with torch.no_grad():
        torch.testing.assert_close(orl5(raw, fresh, nodes), orl4(raw, fresh, nodes),
                                   atol=2e-6, rtol=0)
        torch.testing.assert_close(orl5(raw, lost, nodes)[:, :2], orl4(raw, lost, nodes)[:, :2])
        at_climb = orl5(raw, _set(lost, vz=climb(1.0)), nodes)[:, 2]
        assert at_climb.abs().max() < 1e-5  # its equilibrium is the field's climb
        assert (orl5(raw, _set(lost, vz=0.0), nodes)[:, 2] > 0.05).all()  # hovering: climb
        assert orl4(raw, _set(lost, vz=0.0), nodes)[:, 2].abs().max() < 1e-6  # orl/4 held
        terms = orl5.vertical_terms(lost)
        torch.testing.assert_close(terms["hold"] + terms["gated"], orl5(raw, lost, nodes)[:, 2],
                                   atol=1e-6, rtol=0)


def test_orl5_is_exactly_mirror_equivariant_with_the_same_budget():
    net = build_model(orl.ARMS["orl/5"], CONFIG, seed=2)
    with torch.no_grad():
        for p in net.actor_head.parameters():
            p.add_(0.2 * torch.randn_like(p))
    raw, flat, nodes = batch(66, 32)
    with torch.no_grad():
        base = net(raw, flat, nodes)
        for g in range(4):
            mirrored = net(raw, flat @ net.group[g].T, nodes)
            torch.testing.assert_close(mirrored, base * net.action_signs[g], atol=1e-6, rtol=0)
    assert net.parameter_report()["total"] == \
        build_model(orl.ARMS["orl/4"], CONFIG, seed=2).parameter_report()["total"] == 4016


def test_the_v2_control_is_the_semantic_mlp_except_where_the_rule_acts():
    control = build_model(rc.CONTROL_ARM_V2, CONFIG, seed=3)
    mlp = CurrentActorCritic("semantic_mlp_ppo", CONFIG, seed=3)
    v1 = build_model(rc.CONTROL_ARM, CONFIG, seed=3)
    assert all(torch.equal(control.state_dict()[k], v) for k, v in mlp.state_dict().items())
    assert all(torch.equal(control.state_dict()[k], v) for k, v in v1.state_dict().items())
    assert control.parameter_report()["total"] == 4009
    raw, flat, nodes = batch(67, 32)
    with torch.no_grad():
        for p in control.actor_head.parameters():
            p.add_(0.3 * torch.randn_like(p))
        mlp.load_state_dict({k: v for k, v in control.state_dict().items() if k != "rule_gain"})
        fresh = _set(flat, age=0.1, ex=0.0, ey=0.0)
        lost = _set(flat, age=2.0, h=1.0, ex=0.0, ey=0.0, vz=0.0)
        torch.testing.assert_close(control(raw, fresh, nodes), mlp(raw, fresh, nodes),
                                   atol=2e-6, rtol=0)
        closed, free = control(raw, lost, nodes), mlp(raw, lost, nodes)
        torch.testing.assert_close(closed[:, :2], free[:, :2])  # lateral untouched
        error = -climb(1.0)
        expected = -torch.nn.functional.softplus(control.rule_gain) * error / (abs(error) + 1.0)
        torch.testing.assert_close(closed[:, 2], expected.expand(32), atol=1e-5, rtol=0)
        terms = control.vertical_terms(flat)
        torch.testing.assert_close(terms["hold"] + terms["gated"],
                                   control(raw, flat, nodes)[:, 2], atol=1e-6, rtol=0)


def test_v5_checkpoints_bind_their_own_specs(tmp_path):
    manifest = contract_manifest(CONFIG)
    for new, old in ((orl.ARMS["orl/5"], orl.ARMS["orl/4"]),
                     (rc.CONTROL_ARM_V2, rc.CONTROL_ARM)):
        path = save_checkpoint(tmp_path / "m.pt", build_model(new, CONFIG, seed=0),
                               manifest=manifest)
        load_checkpoint(path, build_model(new, CONFIG, seed=0), manifest=manifest)
        with pytest.raises(ValueError):
            load_checkpoint(path, build_model(old, CONFIG, seed=0), manifest=manifest)


def test_orl_development_5_binds_orl5_and_the_v2_control(tmp_path):
    plan, development = od.load_plan(od.DEFAULT_PLAN_PATHS["orl/5"], CONFIG)
    assert plan.protocol == "orl-development/5" and plan.arm == orl.ARMS["orl/5"]
    assert od.CONTROLS["orl/5"] == rc.CONTROL_ARM_V2
    resolved = od.resolved(plan, development, CONFIG)
    assert resolved["primary_comparison"] == rc.CONTROL_ARM_V2
    assert resolved["control"]["hash"] == rc.CONTROL_HASHES["semantic-mlp-rule/2"]
    assert resolved["rule"] == od.RULE_V5
    assert od.part_dir(tmp_path, "orl/5", "control") == tmp_path / "orl5" / "control"
    assert od.probe_version(rc.CONTROL_ARM_V2) == "orl/5"
    assert od.probe_version(rc.CONTROL_ARM) == "orl/4"
    assert od.probe_hash("orl/5") != od.probe_hash("orl/4")
    assert od.GATE_PROBES["orl/5"]["max_blind_sink_m_s"] == 0.10
    plan4, _ = od.load_plan(od.DEFAULT_PLAN_PATHS["orl/4"], CONFIG)
    assert od.plan_hash(plan, development, CONFIG) != od.plan_hash(plan4, development, CONFIG)
    assert "no ORL G4 learnability pilot" in od.pilot_blocker(CONFIG, tmp_path, "orl/5")
    key, record = od.pilot_record("orl/5", "control")
    assert key == "control" and record["hash"] == rc.CONTROL_HASHES["semantic-mlp-rule/2"]


def test_a_v2_control_probe_episode_records_the_rule():
    from ontology_rgat.current_semantics.pilot import _init_worker
    _init_worker(str(DEFAULT_CONFIG_PATH))
    state = build_model(rc.CONTROL_ARM_V2, CONFIG, seed=0).state_dict()
    episode = od._probe_work((rc.CONTROL_ARM_V2, state, od.GATE_PROBE["seeds"][0]))
    assert episode["refused"] is None
    gate = np.asarray(episode["gate"])
    assert gate.size > 0 and ((gate >= 0) & (gate <= 1)).all()
