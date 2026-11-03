"""ORL (orl.py, status doc 20): the ontology as the policy's inductive bias.

Each bias is a property of the function, tested as such: mirror equivariance (on
phi44 and on the physical state), lateral locality, weight sharing across the two
planes, a learned descent gate, the registered R-GAT budget with no dead parameter,
and checkpoints that never alias a registered arm.  orl/2 (status doc 20.5) adds the
gate's own guarantees: a coherent gradient from the first update, an open gate that can
only add descent, and quality/contact/view reaching the vertical command only through
the gate; orl/1 stays buildable and frozen.  orl/3 (status doc 20.7) adds what a CLOSED
gate means: a damper whose equilibrium is vz = 0 in every state, a gate gradient that is
negative in every state, and a slow initial sink; orl/2 stays buildable and frozen.
orl/4 (status doc 20.9) adds the view rule (view_rule.py): the task's lost-view weight on
the policy's own estimate, multiplying the gate; it is orl/3 where the rule is open and the
damper where it is closed, and orl/3 stays buildable and frozen.  The rule control
(rule_control.py) and orl-development/4 are tested in test_cs3d_orl4.py.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from ontology_rgat.current_semantics import orl
from ontology_rgat.current_semantics.contracts import (DEFAULT_CONFIG_PATH, contract_manifest,
                                                       load_config)
from ontology_rgat.current_semantics.features import FEATURE_NAMES, build_feature_batch
from ontology_rgat.current_semantics.models import (CurrentActorCritic, build_model,
                                                    load_checkpoint, save_checkpoint,
                                                    tensors_from_batch)
from ontology_rgat.current_semantics.observation import CurrentPhysical, encode
from ontology_rgat.current_semantics.pilot import _model

from test_cs3d_models import batch

CONFIG = load_config(DEFAULT_CONFIG_PATH)
INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def model(seed: int = 0, version: str = orl.CURRENT_VERSION):
    torch.manual_seed(seed)
    net = build_model(orl.ARMS[version], CONFIG, seed=seed)
    with torch.no_grad():  # move the output layers off their near-zero init
        for p in net.actor_head.parameters():
            p.add_(0.3 * torch.randn_like(p))
    return net


@pytest.mark.parametrize("version", orl.ORL_VERSIONS)
def test_budget_equals_the_registered_rgat_arm_and_no_parameter_is_dead(version):
    net = model(0, version)
    rgat = CurrentActorCritic("ontology_rgat_ppo", CONFIG, seed=0)
    assert net.parameter_report()["total"] == rgat.parameter_report()["total"] == 4016
    assert torch.equal(net.critic[0].weight, rgat.critic[0].weight)  # the common critic
    raw, flat, nodes = batch(3, 64)
    net(raw, flat, nodes).square().sum().backward()
    dead = [name for name, p in net.named_parameters()
            if not name.startswith(("critic", "log_std")) and not p.grad.abs().sum() > 0]
    assert dead == []


@pytest.mark.parametrize("version", orl.ORL_VERSIONS)
@pytest.mark.parametrize("sx, sy", [(1, -1), (-1, 1), (-1, -1)])
def test_the_policy_is_exactly_mirror_equivariant(sx, sy, version):
    net = model(1, version)
    raw, flat, nodes = batch(5, 48)
    mirrored = flat @ torch.tensor(orl.group_matrix(sx, sy), dtype=flat.dtype).T
    with torch.no_grad():
        out, back = net(raw, flat, nodes), net(raw, mirrored, nodes)
    torch.testing.assert_close(back, out * torch.tensor([sx, sy, 1.0]), atol=1e-6, rtol=0)
    with torch.no_grad():  # the network before the average is NOT equivariant by itself
        raw_net = net.unaveraged(flat), net.unaveraged(mirrored)
    assert not torch.allclose(raw_net[1], raw_net[0] * torch.tensor([sx, sy, 1.0]), atol=1e-4)


def _state(rng) -> CurrentPhysical:
    return CurrentPhysical(
        ex=rng.uniform(-2, 2), ey=rng.uniform(-2, 2), h=rng.uniform(0.4, 3.0),
        vrx=rng.uniform(-1.5, 1.5), vry=rng.uniform(-1.5, 1.5), vgx=rng.uniform(-1.5, 1.5),
        vgy=rng.uniform(-1.5, 1.5), vz=rng.uniform(-0.8, 0.8), roll=rng.uniform(-0.25, 0.25),
        pitch=rng.uniform(-0.25, 0.25), yaw=rng.uniform(-0.08, 0.08), p=rng.uniform(-1, 1),
        q=rng.uniform(-1, 1), r=rng.uniform(-0.3, 0.3), vision_updated=bool(rng.integers(2)),
        vision_age_s=rng.uniform(0, 2), nav_valid=True, nav_age_s=rng.uniform(0, 0.2),
        estimate_valid=bool(rng.random() > 0.1))


def _mirror(s: CurrentPhysical, sx: int, sy: int) -> CurrentPhysical:
    """The world reflected (ENU positions/velocities; scipy "xyz" roll/pitch/yaw; FLU rates)."""
    if sx < 0:
        s = dataclasses.replace(s, ex=-s.ex, vrx=-s.vrx, vgx=-s.vgx, pitch=-s.pitch,
                                yaw=-s.yaw, q=-s.q, r=-s.r)
    if sy < 0:
        s = dataclasses.replace(s, ey=-s.ey, vry=-s.vry, vgy=-s.vgy, roll=-s.roll,
                                yaw=-s.yaw, p=-s.p, r=-s.r)
    return s


@pytest.mark.parametrize("sx, sy", [(1, -1), (-1, 1), (-1, -1)])
def test_the_declared_parities_are_the_physical_mirrors(sx, sy):
    """The ontology's parity table equals what build_features does to a reflected world,
    exactly except the registered half-pixel principal point (view entries only)."""
    rng = np.random.default_rng(11)
    states = [_state(rng) for _ in range(300)]
    flat = build_feature_batch(np.stack([encode(s) for s in states]), CONFIG)[0]
    reflected = build_feature_batch(np.stack([encode(_mirror(s, sx, sy)) for s in states]),
                                    CONFIG)[0]
    error = np.abs(reflected - flat @ orl.group_matrix(sx, sy).T).max(axis=0)
    view = [INDEX[n] for n in FEATURE_NAMES if n.startswith("view_")]
    assert error[[i for i in range(len(FEATURE_NAMES)) if i not in view]].max() == 0.0
    assert error[view].max() < 5e-3


def test_the_planes_are_aligned_so_shared_weights_mean_the_same_physics():
    """Pad offset, closing speed and the tilt that accelerates toward +axis carry the
    same sign in both planes' aligned inputs (thrust of the scipy "xyz" attitude)."""
    def aligned(**kw):
        values = dict(ex=0.0, ey=0.0, h=1.5, vrx=0.0, vry=0.0, vgx=0.0, vgy=0.0, vz=0.0,
                      roll=0.0, pitch=0.0, yaw=0.0, p=0.0, q=0.0, r=0.0, vision_updated=True,
                      vision_age_s=0.05, nav_valid=True, nav_age_s=0.02, estimate_valid=True)
        values.update(kw)
        flat = build_feature_batch(encode(CurrentPhysical(**values))[None], CONFIG)[0][0]
        out = {}
        for concept, roles in orl.PLANE_ROLES.items():
            for i, plane in enumerate(orl.PLANES):
                out[(concept, plane)] = [sign * flat[INDEX[pair[i][0]]] for pair in roles
                                         for sign in [pair[i][1]]]
        return out
    x = aligned(ex=0.6, vrx=0.4, pitch=0.1, q=0.3)
    y = aligned(ey=0.6, vry=0.4, roll=-0.1, p=-0.3)
    for concept in ("RelativeGeometry", "MotionState", "AttitudeState", "ApproachSituation",
                    "ContactCondition"):
        np.testing.assert_allclose(x[(concept, "x")], y[(concept, "y")], atol=1e-12)
    assert np.sign(x[("ViewGeometry", "x")][0]) == np.sign(y[("ViewGeometry", "y")][0]) != 0
    from scipy.spatial.transform import Rotation
    thrust = lambda roll, pitch: Rotation.from_euler("xyz", [roll, pitch, 0]).apply([0, 0, 1])
    assert thrust(0.0, 0.1)[0] > 0 and thrust(-0.1, 0.0)[1] > 0


def _swap_planes(flat: torch.Tensor) -> torch.Tensor:
    out = flat.clone()
    for roles in orl.PLANE_ROLES.values():
        for (fx, sx), (fy, sy) in roles:
            out[:, INDEX[fy]] = sx * sy * flat[:, INDEX[fx]]
            out[:, INDEX[fx]] = sx * sy * flat[:, INDEX[fy]]
    (lx, rx), (by, ty) = orl.MARGINS["x"], orl.MARGINS["y"]
    for a, b in ((lx, by), (rx, ty)):
        out[:, INDEX[a]], out[:, INDEX[b]] = flat[:, INDEX[b]], flat[:, INDEX[a]]
    return out


@pytest.mark.parametrize("version", orl.ORL_VERSIONS)
def test_the_two_planes_share_every_weight(version):
    net = model(2, version)
    raw, flat, nodes = batch(9, 32)
    with torch.no_grad():
        out, swapped = net(raw, flat, nodes), net(raw, _swap_planes(flat), nodes)
    torch.testing.assert_close(swapped, out[:, [1, 0, 2]], atol=1e-6, rtol=0)


@pytest.mark.parametrize("version", orl.ORL_VERSIONS)
def test_lateral_x_never_reads_the_y_plane(version):
    net = model(3, version)
    raw, flat, nodes = batch(4, 32)
    poked = flat.clone()
    for roles in orl.PLANE_ROLES.values():
        for _, (fy, _) in roles:
            poked[:, INDEX[fy]] = torch.empty(len(flat)).uniform_(-0.9, 0.9)
    for name in orl.MARGINS["y"]:
        poked[:, INDEX[name]] = torch.empty(len(flat)).uniform_(0.1, 0.9)
    with torch.no_grad():
        out, moved = net(raw, flat, nodes), net(raw, poked, nodes)
    torch.testing.assert_close(moved[:, 0], out[:, 0], atol=1e-7, rtol=0)
    assert not torch.allclose(moved[:, 1], out[:, 1])
    assert not torch.allclose(moved[:, 2], out[:, 2])


@pytest.mark.parametrize("version", orl.ORL_VERSIONS)
def test_descent_passes_a_learned_gate_that_reads_only_its_concepts(version):
    sources = {s for s, t, r in orl.GRAPHS[version].edges
               if t == "DescentEligibility" and s != t}
    assert sources == {"ObservationQuality", "ContactCondition_x", "ContactCondition_y",
                       "ViewGeometry_x", "ViewGeometry_y"}
    net = model(4, version)
    raw, flat, nodes = batch(6, 16)
    head = net.actor_head
    with torch.no_grad():
        head.gate.bias.fill_(-50.0)
        closed = net(raw, flat, nodes)[:, 2]
        if version in orl.DAMPED:  # the damper alone: -softplus(gain(Z)) * drone_vz
            gains = torch.stack([torch.nn.functional.softplus(head.hold(
                net.encoder.node_states(flat @ net.group[g].T)[:, head.vertical_index]))
                .squeeze(-1) for g in range(4)]).mean(0)
            hold_only = -gains * flat[:, INDEX["drone_vz"]]
        else:
            hold_only = torch.stack([head.hold(net.encoder.node_states(
                flat @ net.group[g].T)[:, head.vertical_index]).squeeze(-1)
                for g in range(4)]).mean(0)
    torch.testing.assert_close(closed, hold_only, atol=1e-6, rtol=0)


def test_the_arm_never_aliases_a_registered_arm(tmp_path):
    manifest = contract_manifest(CONFIG)
    path = save_checkpoint(tmp_path / "orl.pt", model(), manifest=manifest)
    load_checkpoint(path, build_model(orl.ORL_ARM, CONFIG, seed=0), manifest=manifest)
    with pytest.raises(ValueError):
        load_checkpoint(path, CurrentActorCritic("ontology_rgat_ppo", CONFIG, seed=0),
                        manifest=manifest)
    with pytest.raises(ValueError, match="unknown arm"):
        build_model("orl_ppo:0000000000000000", CONFIG, seed=0)
    rebuilt = _model(CONFIG, orl.ORL_ARM, model(5).state_dict())
    assert rebuilt.arm == orl.ORL_ARM and not rebuilt.training


def test_orl_does_not_move_the_v1_contract():
    manifest = contract_manifest(CONFIG)
    assert manifest["graph_hash"].startswith("66e4881a316372ad")
    assert manifest["algorithm_hash"].startswith("70ebb7e0833ef441")
    assert orl.ORL_ARM == f"orl_ppo:{orl.ORL_HASH[:16]}"


def test_orl1_is_frozen_and_stays_loadable():
    # orl-development/1's checkpoints and registration bind this hash (status doc 20.3-20.4).
    assert orl.ARMS["orl/1"] == "orl_ppo:eb68eb947a6a49a2"
    assert len(set(orl.ARMS.values())) == len(orl.ORL_VERSIONS)
    assert orl.ORL_ARM == orl.ARMS[orl.CURRENT_VERSION] == orl.ARMS["orl/4"]
    assert orl.ARMS["orl/3"] == "orl_ppo:80a8e23a39e9b0f4"  # the orl/3 pilot binds it
    assert orl.ARMS["orl/4"] == "orl_ppo:af6f4fb0994a5042"  # orl-development/4 binds it
    assert len(orl.GRAPHS["orl/1"].edges) == 61
    assert len(orl.GRAPHS["orl/2"].edges) == len(orl.GRAPHS["orl/3"].edges) == 59
    assert len(orl.GRAPHS["orl/4"].edges) == 59
    net = build_model(orl.ARMS["orl/1"], CONFIG, seed=0)
    assert net.version == "orl/1" and net.actor_head.hold.bias.abs().max() == 0


def test_orl2_is_frozen_and_stays_loadable():
    # The orl/2 pilot (status doc 20.6) and its gate probe bind these.
    assert orl.ARMS["orl/2"] == "orl_ppo:0d34c49904e549b3"
    net = build_model(orl.ARMS["orl/2"], CONFIG, seed=0)
    assert net.version == "orl/2"
    assert net.actor_head.hold.bias.item() == pytest.approx(np.log(2.0) / 2)


# ------------------------------------------------------------------ orl/2's gate

def _vertical_and_gate_bias_grad(net, raw, flat, nodes):
    grads = []
    for i in range(len(flat)):
        net.zero_grad()
        net(raw[i:i + 1], flat[i:i + 1], nodes[i:i + 1])[0, 2].backward()
        grads.append(net.actor_head.gate.bias.grad.item())
    with torch.no_grad():
        return net(raw, flat, nodes)[:, 2], torch.tensor(grads)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_orl2_gate_gets_a_coherent_gradient_from_the_first_update(seed):
    """orl/1's gate gradient was sigmoid' * descend(Z): x0.01 and of either sign, i.e. ~0
    (status doc 20.4).  orl/2's is -sigmoid'(0) * softplus(0) = -ln(2)/4 in EVERY state,
    and the initial vertical mean is still ~0 (pilot 2's drift)."""
    raw, flat, nodes = batch(7, 24)
    mean2, grad2 = _vertical_and_gate_bias_grad(
        build_model(orl.ARMS["orl/2"], CONFIG, seed=seed), raw, flat, nodes)
    mean1, grad1 = _vertical_and_gate_bias_grad(
        build_model(orl.ARMS["orl/1"], CONFIG, seed=seed), raw, flat, nodes)
    torch.testing.assert_close(grad2, torch.full_like(grad2, -np.log(2.0) / 4), atol=2e-3,
                               rtol=0)
    assert grad1.abs().max() < 2e-3
    assert mean2.abs().max() < 5e-3 and mean1.abs().max() < 5e-3


@pytest.mark.parametrize("version", ["orl/2", "orl/3", "orl/4"])
def test_orl2_an_open_gate_can_only_add_descent(version):
    net = model(6, version)
    raw, flat, nodes = batch(8, 64)
    head = net.actor_head
    with torch.no_grad():
        head.gate.bias.fill_(50.0)
        opened = net(raw, flat, nodes)[:, 2]
        head.gate.bias.fill_(-50.0)
        closed = net(raw, flat, nodes)[:, 2]
    assert (opened <= closed).all() and (opened < closed - 1e-3).any()


VERTICAL_STATE = {name for name, _ in orl.SHARED_ROLES["VerticalState"]}


@pytest.mark.parametrize("version", ["orl/2", "orl/3"])
@pytest.mark.parametrize("bias", [50.0, -50.0])
def test_orl2_quality_contact_and_view_reach_the_vertical_command_only_through_the_gate(
        bias, version):
    net = model(7, version)
    raw, flat, nodes = batch(10, 48)
    poked = flat.clone()
    for name in FEATURE_NAMES:
        if name not in VERTICAL_STATE:
            poked[:, INDEX[name]] = torch.empty(len(flat)).uniform_(0.05, 0.95)
    with torch.no_grad():
        net.actor_head.gate.bias.fill_(bias)  # saturated gate: no path but the gate is left
        torch.testing.assert_close(net(raw, poked, nodes)[:, 2], net(raw, flat, nodes)[:, 2],
                                   atol=1e-6, rtol=0)
        net.actor_head.gate.bias.zero_()
        assert not torch.allclose(net(raw, poked, nodes)[:, 2], net(raw, flat, nodes)[:, 2],
                                  atol=1e-4)
    orl1 = model(7, "orl/1")  # orl/1 let them in around the gate, which is why it idled
    with torch.no_grad():
        orl1.actor_head.gate.bias.fill_(bias)
        assert not torch.allclose(orl1(raw, poked, nodes)[:, 2], orl1(raw, flat, nodes)[:, 2],
                                  atol=1e-4)


def test_vertical_terms_are_the_actor_vertical_mean():
    for version in orl.ORL_VERSIONS:
        net = model(8, version)
        raw, flat, nodes = batch(11, 32)
        with torch.no_grad():
            terms = net.vertical_terms(flat)
            torch.testing.assert_close(terms["hold"] + terms["gated"],
                                       net(raw, flat, nodes)[:, 2], atol=1e-6, rtol=0)
        low = terms["gate"] >= 0 if version in orl.RULED else terms["gate"] > 0  # w may be 1
        assert (low & (terms["gate"] < 1)).all()


# ------------------------------------------------------------- orl/3's closed gate

@pytest.mark.parametrize("version", orl.DAMPED)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_orl3_a_closed_gate_holds_altitude_in_every_state(seed, version):
    """orl/2's closed gate left hold(Z), which learned a ~0.2 m/s sink (status doc 20.6).
    orl/3's closed gate is -gain(Z) * drone_vz: zero at vz = 0 and opposing vz, whatever
    the other 43 features and the trained weights are."""
    net = model(seed, version)
    raw, flat, nodes = batch(20 + seed, 64)
    level, rising, sinking = flat.clone(), flat.clone(), flat.clone()
    level[:, INDEX["drone_vz"]] = 0.0
    rising[:, INDEX["drone_vz"]] = torch.empty(len(flat)).uniform_(0.02, 0.6)
    sinking[:, INDEX["drone_vz"]] = -rising[:, INDEX["drone_vz"]]
    with torch.no_grad():
        net.actor_head.gate.bias.fill_(-50.0)
        assert net(raw, level, nodes)[:, 2].abs().max() < 1e-6
        assert (net(raw, rising, nodes)[:, 2] < 0).all()
        assert (net(raw, sinking, nodes)[:, 2] > 0).all()
        orl2 = model(seed, "orl/2")  # orl/2's closed gate has no such zero
        orl2.actor_head.gate.bias.fill_(-50.0)
        assert orl2(raw, level, nodes)[:, 2].abs().max() > 1e-2


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_orl3_the_gate_gradient_is_negative_in_every_state_trained_or_not(seed):
    raw, flat, nodes = batch(30 + seed, 24)
    for net in (build_model(orl.ARMS["orl/3"], CONFIG, seed=seed), model(seed, "orl/3")):
        _, grad = _vertical_and_gate_bias_grad(net, raw, flat, nodes)
        assert (grad < 0).all()
    _, init = _vertical_and_gate_bias_grad(build_model(orl.ARMS["orl/3"], CONFIG, seed=seed),
                                           raw, flat, nodes)
    expected = -orl.HOLD_GAIN_INIT * 0.25 * orl.sink_reference(orl.INIT_SINK_M_S)
    torch.testing.assert_close(init, torch.full_like(init, expected), atol=2e-3, rtol=0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_orl3_the_untrained_policy_settles_into_a_slow_sink_without_lateral_drift(seed):
    """The price of a meaningful closed gate: the initial vertical mean is not 0.  The
    half-open gate's equilibrium is half the reward field's sink speed, a bounded descent,
    not pilot 2's integrating drift."""
    net = build_model(orl.ARMS["orl/3"], CONFIG, seed=seed)
    raw, flat, nodes = batch(40 + seed, 64)
    settled = flat.clone()
    settled[:, INDEX["drone_vz"]] = -0.5 * orl.sink_reference(orl.INIT_SINK_M_S)
    with torch.no_grad():
        out = net(raw, settled, nodes)
        assert out[:, 2].abs().max() < 2e-3
        assert out[:, :2].abs().max() < 5e-3
    speed = 0.5 * orl.sink_reference(orl.INIT_SINK_M_S)
    assert orl.VZ_SCALE * speed / (1 - speed) == pytest.approx(0.0909, abs=1e-3)  # m/s


@pytest.mark.parametrize("version", orl.ORL_VERSIONS)
def test_the_development_plan_pins_this_orl_spec_and_waits_for_learnability(tmp_path, version):
    from pathlib import Path
    import yaml
    from ontology_rgat.current_semantics import orl_development as od
    from ontology_rgat.current_semantics.contracts import TASK_CONFIG_PATHS
    path = od.DEFAULT_PLAN_PATHS[version]
    assert od.plan_version(path) == version
    plan, development = od.load_plan(path, CONFIG)
    assert plan.arm == orl.ARMS[version] and plan.version == version
    assert development.training_seeds == (10, 11, 12, 13, 14)
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    data["arm"] = "orl_ppo:0000000000000000"
    stale = tmp_path / "stale.yaml"
    stale.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="not this code's ORL arm"):
        od.load_plan(stale, CONFIG)
    with pytest.raises(ValueError, match="extends development/1"):
        od.load_plan(path, load_config(TASK_CONFIG_PATHS["spatial-current-task/6"]))
    status, payload = od.register(DEFAULT_CONFIG_PATH, path, tmp_path)
    assert status == 3 and "not registered" in payload["reason"]
    status, payload = od.run(DEFAULT_CONFIG_PATH, path, tmp_path)
    assert status == 3 and payload["status"] == "REFUSED"


def test_the_orl1_registration_and_paths_did_not_move():
    from ontology_rgat.current_semantics import orl_development as od
    plan, development = od.load_plan(od.DEFAULT_PLAN_PATHS["orl/1"], CONFIG)
    assert od.plan_hash(plan, development, CONFIG).startswith("18c3f32ec5154566")
    plan, development = od.load_plan(od.DEFAULT_PLAN_PATHS["orl/2"], CONFIG)
    assert od.plan_hash(plan, development, CONFIG).startswith("c0d68961b9b5f1b6")
    assert od.probe_hash("orl/2").startswith("be8c22eee0bb0469")  # the orl/2 pilot's probe
    plan, development = od.load_plan(od.DEFAULT_PLAN_PATHS["orl/3"], CONFIG)
    assert od.plan_hash(plan, development, CONFIG).startswith("498a7b31280c6382")
    assert od.probe_hash("orl/3").startswith("722c0edc0ee1800f")  # the orl/3 pilot's probe
    assert od.probe_hash("orl/3") != od.probe_hash("orl/2")
    assert dict(od.OUTS) == {"orl/1": "orl", "orl/2": "orl2", "orl/3": "orl3",
                             "orl/4": "orl4"}
    assert od.DEFAULT_PLAN_PATH == od.DEFAULT_PLAN_PATHS[orl.CURRENT_VERSION]


@pytest.mark.parametrize("version", orl.ORL_VERSIONS)
def test_a_passed_orl_pilot_is_accepted_under_the_hashes_run_pilot_writes(tmp_path, version):
    # run_pilot(arm=ORL_ARM) replaces g4.arm, so its stored g4_hash is NOT the Raw
    # protocol's; the first live registration was refused on exactly that mismatch.
    from dataclasses import replace
    import json
    from ontology_rgat.current_semantics import orl_development as od
    arm = orl.ARMS[version]
    swapped = contract_manifest(replace(CONFIG, g4=replace(CONFIG.g4, arm=arm)))
    assert swapped["g4_hash"] != contract_manifest(CONFIG)["g4_hash"]
    keys = ("g4_hash", "algorithm_hash", "task_hash", "calibration_hash", "g3_hash")
    report = {"arm": arm, "orl": {"hash": orl.HASHES[version]},
              "hashes": {k: swapped[k] for k in keys},
              "protocol_complete": True, "learnability_passed": True}
    if version != "orl/1":
        report["gate_probe"] = {"probe_hash": od.probe_hash(version), "gate_active": True}
    path = od.out_dir(tmp_path, version) / od.LEARNABILITY
    path.parent.mkdir(parents=True)
    write = lambda: path.write_text(json.dumps(report), encoding="utf-8")
    write()
    if version in od.CONTROLS:  # orl/4 also needs its rule control's pilot (test_cs3d_orl4)
        assert "rule control" in od.pilot_blocker(CONFIG, tmp_path, version)
        control = od.part_dir(tmp_path, version, "control") / od.LEARNABILITY
        control.parent.mkdir(parents=True)
        cswap = contract_manifest(replace(CONFIG, g4=replace(CONFIG.g4,
                                                             arm=od.CONTROLS[version])))
        key, record = od.pilot_record(version, "control")
        control.write_text(json.dumps({**report, "arm": od.CONTROLS[version], key: record,
                                       "hashes": {k: cswap[k] for k in keys}}),
                           encoding="utf-8")
    assert od.pilot_blocker(CONFIG, tmp_path, version) is None
    other = [v for v in orl.ORL_VERSIONS if v != version][0]
    assert od.pilot_blocker(CONFIG, tmp_path, other) is not None  # its own directory only
    report["hashes"]["g4_hash"] = contract_manifest(CONFIG)["g4_hash"]
    write()
    assert "another contract" in od.pilot_blocker(CONFIG, tmp_path, version)
    report["hashes"] = {k: swapped[k] for k in keys}
    report["learnability_passed"] = False
    write()
    assert "did not pass" in od.pilot_blocker(CONFIG, tmp_path, version)
    if version != "orl/1":  # a passed pilot whose gate stayed inert never registers /2
        report["learnability_passed"] = True
        report["gate_probe"]["gate_active"] = False
        write()
        assert "inert" in od.pilot_blocker(CONFIG, tmp_path, version)
        report["gate_probe"] = {"probe_hash": "0" * 64, "gate_active": True}
        write()
        assert "not probed" in od.pilot_blocker(CONFIG, tmp_path, version)


def test_the_development_verdict_is_orl_minus_the_semantic_mlp():
    from ontology_rgat.current_semantics import orl_development as od
    from ontology_rgat.current_semantics.development import CELLS
    _, development = od.load_plan(od.DEFAULT_PLAN_PATH, CONFIG)
    seeds = development.training_seeds
    summary = lambda v: {"cells": {c: {"success": v, "unsafe": 0.05} for c in CELLS}}
    stored = {arm: {s: summary(0.8) for s in seeds} for arm in (od.PRIMARY, *od.SECONDARY)}
    better = od.judge(development, {s: summary(0.9 + 0.01 * i) for i, s in enumerate(seeds)},
                      stored)
    assert better["orl_vs_semantic_mlp"] == "better"
    assert better["primary"]["mean"] == pytest.approx(0.12)
    noisy = od.judge(development, {s: summary(v) for s, v in zip(seeds, (.9, .7, .9, .7, .85))},
                     stored)
    assert noisy["orl_vs_semantic_mlp"] == "not established"


def test_the_development_verdict_v2_reports_how_many_gates_are_active():
    from ontology_rgat.current_semantics import orl_development as od
    from ontology_rgat.current_semantics.development import CELLS
    _, development = od.load_plan(od.DEFAULT_PLAN_PATHS["orl/2"], CONFIG)
    seeds = development.training_seeds
    summary = lambda v, active: {"cells": {c: {"success": v, "unsafe": 0.05} for c in CELLS},
                                 "gate_probe": {"gate_active": active}}
    stored = {arm: {s: summary(0.8, None) for s in seeds} for arm in (od.PRIMARY, *od.SECONDARY)}
    orl2 = {s: summary(0.9, i != 2) for i, s in enumerate(seeds)}
    verdict = od.judge(development, orl2, stored, "orl-development/2")
    assert verdict["gate_active_models"] == "4/5" and not verdict["reads_as_test_of_the_gate"]
    assert verdict["rule"] == od.RULE_V2
    assert "gate_active_models" not in od.judge(development, orl2, stored)
    verdict3 = od.judge(development, orl2, stored, "orl-development/3")
    assert verdict3["gate_active_models"] == "4/5" and verdict3["rule"] == od.RULE_V3


def _probe_episode(gate_fresh, gate_blind, fresh=40, blind=40, low=10, vz_blind=None):
    """fresh decisions above 0.5 m, blind ones above 0.5 m, then low ones (ignored)."""
    gate = [gate_fresh] * fresh + [gate_blind] * blind + [0.0] * low
    age = [0.1] * fresh + [1.2] * blind + [1.2] * low
    h = [1.5] * (fresh + blind) + [0.3] * low
    episode = {"refused": None, "status": "SUCCESS", "gate": gate,
               "gated": [-g for g in gate], "valid": [True] * len(gate),
               "vision_age_s": age, "h": h}
    if vz_blind is not None:  # sinking 0.3 m/s while fresh; low ones touching down
        episode["vz"] = [-0.3] * fresh + [vz_blind] * blind + [-0.5] * low
    return episode


def test_the_gate_probe_rule():
    from ontology_rgat.current_semantics import orl_development as od
    active = od.summarize_probe([_probe_episode(0.8, 0.3), _probe_episode(0.8, 0.3)])
    assert active["gate_active"] and active["gate_drop"] == pytest.approx(0.5)
    assert active["fresh_decisions"] == active["blind_decisions"] == 80
    inert = od.summarize_probe([_probe_episode(0.62, 0.6), _probe_episode(0.62, 0.6)])
    assert not inert["gate_active"] and inert["gate_drop"] == pytest.approx(0.02)
    too_few = od.summarize_probe([_probe_episode(0.9, 0.1, blind=20)])
    assert too_few["blind_decisions"] == 20 and too_few["gate_drop"] is None
    assert not too_few["gate_active"]
    assert not od.summarize_probe([{"refused": "setup"}])["gate_active"]


def test_gate_probe_2_also_requires_the_closed_gate_to_stop_the_descent():
    from ontology_rgat.current_semantics import orl_development as od
    limit = od.GATE_PROBES["orl/3"]["max_blind_sink_m_s"]
    holds = od.summarize_probe([_probe_episode(0.8, 0.3, vz_blind=-0.05)] * 2, "orl/3")
    assert holds["gate_active"] and holds["blind_sink_ok"]
    assert holds["vz_blind"] == pytest.approx(-0.05) and holds["vz_fresh"] == pytest.approx(-0.3)
    # orl/2's pilot seed 0: the gate dropped 0.76 and the vehicle still sank 0.17-0.20 m/s
    sinks = od.summarize_probe([_probe_episode(0.91, 0.15, vz_blind=-0.18)] * 2, "orl/3")
    assert sinks["gate_drop"] >= 0.2 and not sinks["blind_sink_ok"]
    assert not sinks["gate_active"]
    assert od.summarize_probe([_probe_episode(0.91, 0.15, vz_blind=-0.18)] * 2)["gate_active"]
    inert = od.summarize_probe([_probe_episode(0.6, 0.6, vz_blind=0.0)] * 2, "orl/3")
    assert inert["blind_sink_ok"] and not inert["gate_active"]
    edge = od.summarize_probe([_probe_episode(0.8, 0.3, vz_blind=-limit)] * 2, "orl/3")
    assert edge["gate_active"]
    no_vz = od.summarize_probe([_probe_episode(0.8, 0.3)] * 2, "orl/3")
    assert not no_vz["gate_active"]  # probe 2 never passes without the own vz
    assert od.GATE_PROBES["orl/3"]["seeds"] == od.GATE_PROBE["seeds"]


def test_the_gate_probe_seeds_were_never_used():
    from ontology_rgat.current_semantics import orl_development as od
    from ontology_rgat.current_semantics.contracts import (AUDITED_SEED_RANGES,
                                                           DIAGNOSTIC_SEED_RANGES)
    seeds = od.GATE_PROBE["seeds"]
    assert len(seeds) == len(set(seeds)) == 24
    for seed in seeds:
        assert not any(first <= seed <= last for _, first, last, _ in AUDITED_SEED_RANGES)
        assert not any(first <= seed <= last for _, first, last in DIAGNOSTIC_SEED_RANGES)
    g4 = CONFIG.g4
    assert not set(seeds) & (set(g4.validation_seeds) | set(g4.test_seeds))
    assert max(seeds) < g4.train_seed_base


def test_a_probe_episode_records_the_gate_per_decision():
    from ontology_rgat.current_semantics import orl_development as od
    from ontology_rgat.current_semantics.pilot import _init_worker
    _init_worker(str(DEFAULT_CONFIG_PATH))
    state = build_model(orl.ORL_ARM, CONFIG, seed=0).state_dict()
    episode = od._probe_work((orl.ORL_ARM, state, od.GATE_PROBE["seeds"][0]))
    assert episode["refused"] is None and episode["blackout_s"] is not None
    n = len(episode["gate"])
    assert n > 0 and all(len(episode[k]) == n
                         for k in ("gated", "valid", "vision_age_s", "h", "vz"))
