"""ORL (orl.py, status doc 20): the ontology as the policy's inductive bias.

Each bias is a property of the function, tested as such: mirror equivariance (on
phi44 and on the physical state), lateral locality, weight sharing across the two
planes, a learned descent gate, the registered R-GAT budget with no dead parameter,
and checkpoints that never alias a registered arm.
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


def model(seed: int = 0):
    torch.manual_seed(seed)
    net = build_model(orl.ORL_ARM, CONFIG, seed=seed)
    with torch.no_grad():  # move the output layers off their near-zero init
        for p in net.actor_head.parameters():
            p.add_(0.3 * torch.randn_like(p))
    return net


def test_budget_equals_the_registered_rgat_arm_and_no_parameter_is_dead():
    net = model()
    rgat = CurrentActorCritic("ontology_rgat_ppo", CONFIG, seed=0)
    assert net.parameter_report()["total"] == rgat.parameter_report()["total"] == 4016
    assert torch.equal(net.critic[0].weight, rgat.critic[0].weight)  # the common critic
    raw, flat, nodes = batch(3, 64)
    net(raw, flat, nodes).square().sum().backward()
    dead = [name for name, p in net.named_parameters()
            if not name.startswith(("critic", "log_std")) and not p.grad.abs().sum() > 0]
    assert dead == []


@pytest.mark.parametrize("sx, sy", [(1, -1), (-1, 1), (-1, -1)])
def test_the_policy_is_exactly_mirror_equivariant(sx, sy):
    net = model(1)
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


def test_the_two_planes_share_every_weight():
    net = model(2)
    raw, flat, nodes = batch(9, 32)
    with torch.no_grad():
        out, swapped = net(raw, flat, nodes), net(raw, _swap_planes(flat), nodes)
    torch.testing.assert_close(swapped, out[:, [1, 0, 2]], atol=1e-6, rtol=0)


def test_lateral_x_never_reads_the_y_plane():
    net = model(3)
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


def test_descent_passes_a_learned_gate_that_reads_only_its_concepts():
    sources = {s for s, t, r in orl.ORL_GRAPH.edges if t == "DescentEligibility" and s != t}
    assert sources == {"ObservationQuality", "ContactCondition_x", "ContactCondition_y",
                       "ViewGeometry_x", "ViewGeometry_y"}
    net = model(4)
    raw, flat, nodes = batch(6, 16)
    head = net.actor_head
    with torch.no_grad():
        head.gate.bias.fill_(-50.0)
        closed = net(raw, flat, nodes)[:, 2]
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


def test_the_development_plan_pins_this_orl_spec_and_waits_for_learnability(tmp_path):
    from pathlib import Path
    import yaml
    from ontology_rgat.current_semantics import orl_development as od
    from ontology_rgat.current_semantics.contracts import TASK_CONFIG_PATHS
    plan, development = od.load_plan(od.DEFAULT_PLAN_PATH, CONFIG)
    assert plan.arm == orl.ORL_ARM and development.training_seeds == (10, 11, 12, 13, 14)
    data = yaml.safe_load(Path(od.DEFAULT_PLAN_PATH).read_text(encoding="utf-8"))
    data["arm"] = "orl_ppo:0000000000000000"
    stale = tmp_path / "stale.yaml"
    stale.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="not this code's ORL arm"):
        od.load_plan(stale, CONFIG)
    with pytest.raises(ValueError, match="extends development/1"):
        od.load_plan(od.DEFAULT_PLAN_PATH, load_config(TASK_CONFIG_PATHS["spatial-current-task/6"]))
    status, payload = od.register(DEFAULT_CONFIG_PATH, od.DEFAULT_PLAN_PATH, tmp_path)
    assert status == 3 and "not registered" in payload["reason"]
    status, payload = od.run(DEFAULT_CONFIG_PATH, od.DEFAULT_PLAN_PATH, tmp_path)
    assert status == 3 and payload["status"] == "REFUSED"


def test_a_passed_orl_pilot_is_accepted_under_the_hashes_run_pilot_writes(tmp_path):
    # run_pilot(arm=ORL_ARM) replaces g4.arm, so its stored g4_hash is NOT the Raw
    # protocol's; the first live registration was refused on exactly that mismatch.
    from dataclasses import replace
    import json
    from ontology_rgat.current_semantics import orl_development as od
    swapped = contract_manifest(replace(CONFIG, g4=replace(CONFIG.g4, arm=orl.ORL_ARM)))
    assert swapped["g4_hash"] != contract_manifest(CONFIG)["g4_hash"]
    keys = ("g4_hash", "algorithm_hash", "task_hash", "calibration_hash", "g3_hash")
    report = {"arm": orl.ORL_ARM, "orl": {"hash": orl.ORL_HASH},
              "hashes": {k: swapped[k] for k in keys},
              "protocol_complete": True, "learnability_passed": True}
    path = tmp_path / od.OUT / od.LEARNABILITY
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(report), encoding="utf-8")
    assert od.pilot_blocker(CONFIG, tmp_path) is None
    report["hashes"]["g4_hash"] = contract_manifest(CONFIG)["g4_hash"]
    path.write_text(json.dumps(report), encoding="utf-8")
    assert "another contract" in od.pilot_blocker(CONFIG, tmp_path)
    report["hashes"] = {k: swapped[k] for k in keys}
    report["learnability_passed"] = False
    path.write_text(json.dumps(report), encoding="utf-8")
    assert "did not pass" in od.pilot_blocker(CONFIG, tmp_path)


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
