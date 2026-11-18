"""bond/1: WMON's relational bond equilibrium in front of the ontology R-GAT."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from ontology_rgat.current_semantics import bond
from ontology_rgat.current_semantics.contracts import (DESIGN_PARAMETER_TOTALS, NODE_ORDER,
                                                       contract_manifest, load_config)
from ontology_rgat.current_semantics.features import build_features
from ontology_rgat.current_semantics.models import (build_model, load_checkpoint,
                                                    save_checkpoint, tensors_from_frames)

L = NODE_ORDER.index("LandingContext")


@pytest.fixture(scope="module")
def config():
    return load_config()


def random_layer(seed=0, dim=8):
    torch.manual_seed(seed)
    layer = bond.RelationalBondLayer(dim).double()
    with torch.no_grad():
        for p in layer.parameters():
            p.add_(0.5 * torch.randn_like(p))
    return layer


def test_closed_form_is_the_energy_minimiser():
    layer = random_layer()
    torch.manual_seed(1)
    h0 = torch.randn(5, 8, 8, dtype=torch.float64)
    validity = (torch.rand(5, 8) > 0.4).double()
    x = layer(h0, validity)
    # dense reference: the full (8*d) quadratic, minimised by one big solve
    c = layer.weights(validity)
    k = torch.nn.functional.softplus(layer.stiffness)[layer.relation]
    a = layer.transform[layer.relation]
    for b in range(5):
        for m in range(8):
            M = torch.diag(c[b]) + bond.RIDGE * torch.eye(8, dtype=torch.float64)
            for e in range(len(k)):
                u = torch.zeros(8, dtype=torch.float64)
                u[layer.target[e]] += 1.0
                u[layer.source[e]] -= a[e, m]
                M = M + k[e] * torch.outer(u, u)
            ref = torch.linalg.solve(M, c[b] * h0[b, :, m])
            torch.testing.assert_close(x[b, :, m], ref, rtol=1e-10, atol=1e-10)
    e0 = layer.energy(x, h0, validity)
    for _ in range(20):  # any perturbation raises the energy
        assert torch.all(layer.energy(x + 1e-3 * torch.randn_like(x), h0, validity) > e0)


def test_landing_context_is_defined_by_its_bonds_alone():
    layer = bond.RelationalBondLayer(8).double()          # k = 1, a = 1 at init
    h0 = torch.randn(2, 8, 8, dtype=torch.float64)
    validity = torch.ones(2, 8, dtype=torch.float64)
    x = layer(h0, validity)
    assert layer.weights(validity)[:, L].abs().max() == 0.0
    sources = [NODE_ORDER.index(n) for n in ("ApproachSituation", "ViewGeometry",
                                             "ContactCondition")]
    # L has no evidence: its value is the ridge-damped mean of its informsLanding sources
    expected = x[:, sources].sum(1) / (3.0 + bond.RIDGE)
    torch.testing.assert_close(x[:, L], expected, rtol=1e-9, atol=1e-9)
    changed = h0.clone()
    changed[:, L] += 100.0                                 # L's own projection is ignored
    torch.testing.assert_close(layer(changed, validity), x)


def test_an_invalid_node_leans_on_its_relations():
    layer = bond.RelationalBondLayer(8).double()
    h0 = torch.randn(1, 8, 8, dtype=torch.float64)
    valid = torch.ones(1, 8, dtype=torch.float64)
    invalid = valid.clone()
    G = NODE_ORDER.index("RelativeGeometry")
    invalid[:, G] = 0.0
    w = layer.weights(invalid)
    assert w[0, G].item() == pytest.approx(bond.INVALID_TRUST_INIT, rel=1e-6)  # float32 init
    dist_valid = (layer(h0, valid)[0, G] - h0[0, G]).norm()
    dist_invalid = (layer(h0, invalid)[0, G] - h0[0, G]).norm()
    assert dist_invalid > dist_valid                       # pulled toward its neighbours


def test_validity_is_read_from_the_node_input(config):
    from ontology_rgat.current_semantics.contracts import OBSERVATION_NAMES
    vec = np.zeros(len(OBSERVATION_NAMES))
    i = {n: k for k, n in enumerate(OBSERVATION_NAMES)}
    for name in ("drone_cosPhi", "drone_cosTheta", "drone_cosPsi"):
        vec[i[name]] = 1.0
    vec[i["drone_navigationValid"]] = 1.0
    fresh, stale, lost = vec.copy(), vec.copy(), vec.copy()
    s = next(c for c in __import__("ontology_rgat.current_semantics.contracts", fromlist=["x"])
             .OBSERVATION_CHANNELS if c.name == "ugv_visionAge").scale
    fresh[i["ugv_visionAge"]] = 0.2 / (0.2 + s)
    stale[i["ugv_visionAge"]] = 1.5 / (1.5 + s)
    lost[i["ugv_visionAge"]] = 0.2 / (0.2 + s)
    lost[i["drone_navigationValid"]] = 0.0
    fresh[i["relative_height"]] = stale[i["relative_height"]] = lost[i["relative_height"]] = 0.5
    _, _, nodes = tensors_from_frames([build_features(v, config) for v in (fresh, stale, lost)])
    v = bond.node_validity(nodes)
    est = [NODE_ORDER.index(n) for n in ("RelativeGeometry", "MotionState",
                                         "ApproachSituation", "ContactCondition")]
    assert v[0, est].min() == 1 and v[1, est].max() == 0 and v[2, est].max() == 0
    assert v[:, NODE_ORDER.index("AttitudeState")].min() == 1 and v[:, L].max() == 0


def test_arm_builds_trains_and_keeps_the_budget(config, tmp_path):
    model = build_model(bond.ARM, config, seed=3)
    report = model.parameter_report()
    rgat = DESIGN_PARAMETER_TOTALS["ontology_rgat_ppo"]
    assert abs(report["total"] - rgat) <= 0.005 * rgat
    assert bond.ARM.startswith("bond_rgat_ppo:") and len(bond.SPEC_HASH) == 64
    raw = torch.randn(4, 21) * 0.1
    flat = torch.zeros(4, 44)
    nodes = torch.zeros(4, 8, 10)
    nodes[..., -1] = torch.arange(1, 9) / 8
    out = model.act(raw, flat, nodes, deterministic=False,
                    generator=torch.Generator().manual_seed(0))
    out["log_prob"].sum().backward()
    grads = {n: p.grad for n, p in model.named_parameters() if "bond" in n}
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads.values())
    # the same seed rebuilds the same network; a checkpoint binds to the bond arm
    acfg = replace(config, g4=replace(config.g4, arm=bond.ARM))
    model = build_model(bond.ARM, acfg, seed=3)
    path = save_checkpoint(tmp_path / "b.pt", model, manifest=contract_manifest(acfg))
    other = build_model(bond.ARM, acfg, seed=9)
    load_checkpoint(path, other, manifest=contract_manifest(acfg))
    for (n, p), (_, q) in zip(model.state_dict().items(), other.state_dict().items()):
        torch.testing.assert_close(p, q, msg=n)
    rgat_cfg = replace(config, g4=replace(config.g4, arm="ontology_rgat_ppo"))
    with pytest.raises(ValueError):
        load_checkpoint(path, build_model("ontology_rgat_ppo", rgat_cfg, seed=3),
                        manifest=contract_manifest(rgat_cfg))


def test_bond_development_rule_and_interval(config):
    from types import SimpleNamespace
    from ontology_rgat.current_semantics import bond_development as bd
    from ontology_rgat.current_semantics import development as dev
    plan = SimpleNamespace(training_seeds=(10, 11, 12, 13, 14),
                           arms=("raw_ppo", "ontology_rgat_ppo"))

    def cells(x):
        return {c: {"success": x, "unsafe": 0.1} for c in dev.CELLS}

    stored = {"models": {f"{s}/{a}": {"cells": cells(0.70)}
                         for s in plan.training_seeds for a in plan.arms}}
    up = {s: {"cells": cells(0.75 + 0.01 * i)} for i, s in enumerate(plan.training_seeds)}
    down = {s: {"cells": cells(0.65 - 0.01 * i)} for i, s in enumerate(plan.training_seeds)}
    mixed = {s: {"cells": cells(v)} for s, v in zip(plan.training_seeds,
                                                   (0.9, 0.5, 0.8, 0.6, 0.7))}
    assert bd.judge(plan, up, stored)["verdict"] == "SUPPORTED"
    assert bd.judge(plan, down, stored)["verdict"] == "NOT SUPPORTED"
    assert bd.judge(plan, mixed, stored)["verdict"] == "INCONCLUSIVE"
    primary = bd.judge(plan, up, stored)["primary"]
    assert primary["mean"] == pytest.approx(0.07) and primary["low"] < 0.07 < primary["high"]
