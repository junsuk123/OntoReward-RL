"""confirmation/1 (guide 11.4-11.6): fresh seeds, a registered decision rule, the training
seed as the unit.  Nothing here trains or flies."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from ontology_rgat.current_semantics import confirmation as cf
from ontology_rgat.current_semantics import development as dev
from ontology_rgat.current_semantics.contracts import load_config

PROJECT = Path(__file__).resolve().parents[2]
WMON = PROJECT / "config" / "current_semantics" / "spatial_view_contact_v1_nav4_wmon.yaml"


@pytest.fixture(scope="module")
def config():
    return load_config(WMON)


@pytest.fixture(scope="module")
def plan(config):
    return cf.load_plan(cf.FULL_PATH, config)


@pytest.fixture(scope="module")
def mini(config):
    return cf.load_plan(cf.DEFAULT_PATH, config)


def test_plan_repeats_the_development_procedure_on_fresh_seeds(plan, config):
    development = dev.load_development(dev.default_plan_path(config), config)
    m = plan.models
    assert m.cells == development.cells and m.gates == development.gates
    assert m.episodes_per_cell == development.episodes_per_cell == 100
    assert m.dropout_start_s == development.dropout_start_s and m.arms == development.arms
    assert len(m.training_seeds) == 20
    assert not set(m.training_seeds) & set(development.training_seeds)
    assert not set(m.validation_seeds) & set(development.validation_seeds)
    assert plan.analysis.primary_comparisons == ("semantic_mlp_ppo", "untyped_gnn_ppo")
    assert plan.analysis.practical_comparison == "raw_ppo"
    assert [c for c, _ in plan.isaac.conditions] == ["composition_id", "composition_ood",
                                                     "combined_id"]
    assert len(plan.isaac.seeds) == 30 and min(plan.isaac.seeds) > 994130


def _write(tmp_path, **changes):
    data = yaml.safe_load(cf.FULL_PATH.read_text(encoding="utf-8"))
    data.update(changes)
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.mark.parametrize("changes, text", [
    ({"training_seeds": [10, 21, 22]}, "reuse"),
    ({"env_seed_base": 50_000_000}, "overlap"),
    ({"validation_seeds": [992_101, 993_102]}, "overlap"),
    ({"train_seed_base": 20_000_000}, "overlap"),
])
def test_seed_audit_refuses_spent_seeds(tmp_path, config, changes, text):
    with pytest.raises(ValueError, match=text):
        cf.load_plan(_write(tmp_path, **changes), config)


def test_isaac_seeds_must_be_fresh(tmp_path, config):
    data = yaml.safe_load(cf.FULL_PATH.read_text(encoding="utf-8"))
    data["isaac"]["conditions"]["composition_id"][0] = 994_101
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        cf.load_plan(path, config)


def test_the_comparison_family_cannot_be_rewritten(tmp_path, config):
    data = yaml.safe_load(cf.FULL_PATH.read_text(encoding="utf-8"))
    data["analysis"]["primary_comparisons"] = ["raw_ppo", "untyped_gnn_ppo"]
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="guide 11.3"):
        cf.load_plan(path, config)


def test_shards_partition_the_models(plan):
    shards = [cf.shard_keys(plan, (k, 3)) for k in range(3)]
    flat = [k for s in shards for k in s]
    assert sorted(flat) == sorted(cf.model_keys(plan)) and len(set(flat)) == len(flat)


def test_paired_interval_matches_the_t_distribution():
    from scipy import stats
    d = [0.1, 0.2, -0.05, 0.15, 0.3]
    out = cf.paired_interval(d, 0.975)
    lo, hi = stats.t.interval(0.975, 4, loc=np.mean(d), scale=stats.sem(d))
    assert out["low"] == pytest.approx(lo) and out["high"] == pytest.approx(hi)
    assert out["p"] == pytest.approx(stats.ttest_1samp(d, 0.0).pvalue)


def test_holm_steps_down():
    assert cf.holm({"a": 0.01, "b": 0.02, "c": 0.04}, 0.05) == {"a": True, "b": True,
                                                                  "c": True}
    assert cf.holm({"a": 0.01, "b": 0.03, "c": 0.04}, 0.05) == {"a": True, "b": False,
                                                                  "c": False}


def _models(plan, success, unsafe=0.1):
    rng = np.random.default_rng(0)
    out = {}
    for seed in plan.models.training_seeds:
        for arm in plan.models.arms:
            base = success[arm] + rng.normal(0, 0.02)
            out[f"{seed}/{arm}"] = {
                "validation": {"deterministic_success": 0.9, "sampled_success": 0.8},
                "checkpoint": "x",
                "cells": {c: {"success": float(np.clip(base, 0, 1)), "unsafe": unsafe}
                          for c in dev.CELLS}}
    return out


def test_primaries_gate_the_practical_comparison(plan):
    arms = {"raw_ppo": 0.70, "semantic_mlp_ppo": 0.70, "untyped_gnn_ppo": 0.60,
            "ontology_rgat_ppo": 0.80}
    verdict = cf.judge_local(plan, _models(plan, arms))
    assert verdict["primary_passed"] and verdict["confirmation_passed"]
    assert verdict["practical"]["raw_ppo"]["opened"] and verdict["practical_passed"]
    tied = cf.judge_local(plan, _models(plan, {**arms, "semantic_mlp_ppo": 0.80}))
    assert not tied["primary_passed"] and not tied["confirmation_passed"]
    assert not tied["practical"]["raw_ppo"]["opened"] and not tied["practical_passed"]


def test_a_guard_failure_withholds_confirmation(plan):
    arms = {"raw_ppo": 0.70, "semantic_mlp_ppo": 0.70, "untyped_gnn_ppo": 0.60,
            "ontology_rgat_ppo": 0.80}
    models = _models(plan, arms)
    for seed in plan.models.training_seeds:
        for cell in dev.OOD_CELLS:
            models[f"{seed}/ontology_rgat_ppo"]["cells"][cell]["unsafe"] = 0.3
    verdict = cf.judge_local(plan, models)
    assert verdict["primary_passed"] and not verdict["guards"]["ood_unsafe_vs_raw"]["passed"]
    assert not verdict["confirmation_passed"]


def test_isaac_judge_counts_models_and_risk(plan):
    rows, excluded = [], {"39/semantic_mlp_ppo": "no best.pt"}
    for seed in plan.models.training_seeds:
        for arm in plan.models.arms:
            key = f"{seed}/{arm}"
            if key in excluded:
                continue
            for i, (cell, env, _) in enumerate(cf.seed_order(plan, [seed], {})[0][:30 * 4:4]):
                good = arm == "ontology_rgat_ppo" or i % 3
                rows.append({"condition": cell, "seed": env, "model": key, "arm": arm,
                             "status": "SUCCESS" if good else "UNSAFE_CONTACT"})
    verdict = cf.judge_isaac(plan, rows, excluded)
    assert verdict["means"]["ontology_rgat_ppo"] == 1.0
    assert verdict["primary_passed"] and verdict["practical_passed"]
    assert all(v["lower_risk"] for v in verdict["risk_family"].values())
    with pytest.raises(ValueError, match="Isaac verdicts"):
        cf.judge_isaac(plan, rows[:-1], excluded)


def test_seed_order_is_environment_major_within_a_seed(plan):
    order, keys = cf.seed_order(plan, [20, 21], {"21/raw_ppo": "x"})
    assert len(keys) == 7 and len(order) == 30 * 7
    first = order[:4]
    assert {k for _, _, k in first} == {f"20/{a}" for a in plan.models.arms}
    assert len({(c, e) for c, e, _ in first}) == 1


def test_checkpoint_binding_refuses_a_changed_checkpoint(tmp_path, plan):
    for arm in plan.models.arms:
        d = cf.model_root(tmp_path, plan, f"20/{arm}")
        d.mkdir(parents=True)
        if arm != "raw_ppo":
            (d / "best.pt").write_bytes(arm.encode())
    bound, excluded = cf.bind_checkpoints(plan, tmp_path, [20])
    assert set(bound) == {f"20/{a}" for a in plan.models.arms if a != "raw_ppo"}
    assert set(excluded) == {"20/raw_ppo"}
    (cf.model_root(tmp_path, plan, "20/ontology_rgat_ppo") / "best.pt").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="changed"):
        cf.bind_checkpoints(plan, tmp_path, [20])


def test_registration_needs_a_passed_development_gate(tmp_path):
    config = load_config(WMON)
    assert "registration" in cf.blocker(config, tmp_path)
    status, payload = cf.register(WMON, cf.DEFAULT_PATH, tmp_path)
    assert status == 3 and payload["status"] == "REFUSED"
    assert not (tmp_path / "confirmation").exists()
    assert not (tmp_path / "confirmation_mini").exists()


def test_the_miniature_plan_only_shrinks_sizes(mini, plan, config):
    assert mini.protocol == "confirmation-mini/1" and mini.folder == "confirmation_mini"
    assert mini.isaac.protocol == "isaac-confirmation-mini/1"
    assert mini.updates == 300 < config.g4.updates == plan.updates
    assert len(mini.models.training_seeds) == 6 and mini.models.episodes_per_cell == 30
    assert len(mini.isaac.seeds) == 9 and set(mini.isaac.seeds) <= set(plan.isaac.seeds)
    m, f = mini.models, plan.models
    assert (m.cells, m.gates, m.arms, m.dropout_start_s) == (f.cells, f.gates, f.arms,
                                                             f.dropout_start_s)
    assert mini.analysis == plan.analysis
    budget = cf.budget_config(config, mini)
    assert budget.g4.updates == 300
    assert replace(budget.g4, updates=config.g4.updates) == config.g4
    assert cf.budget_config(config, plan) is config
    assert cf.plan_hash(mini, config) != cf.plan_hash(plan, config)


def test_four_shards_are_one_arm_each(mini):
    for k, arm in enumerate(mini.models.arms):
        keys = cf.shard_keys(mini, (k, 4))
        assert {key.split("/")[1] for key in keys} == {arm} and len(keys) == 6


def test_a_plan_cannot_extend_the_budget(tmp_path, config):
    data = yaml.safe_load(cf.DEFAULT_PATH.read_text(encoding="utf-8"))
    data["updates"] = config.g4.updates + 1
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="never extend"):
        cf.load_plan(path, config)


def test_claims_are_exclusive_and_stale_claims_are_cleared(tmp_path):
    import os
    d = tmp_path / "seed20" / "raw_ppo"
    assert cf.claim(d) and cf.claim(d)                    # own claim is re-entrant
    cf.release(d)
    (d / cf.CLAIM).write_text(f"{os.getppid()} x\n")      # a live other process
    assert not cf.claim(d)
    (d / cf.CLAIM).write_text("999999999 x\n")             # a dead one
    (d / "best.pt").write_bytes(b"stale")
    assert cf.claim(d) and not (d / "best.pt").exists()
    assert (d / cf.CLAIM).read_text().split()[0] == str(os.getpid())


def test_amendment_narrows_arms_and_keeps_the_registered_hash(mini, config):
    body = cf.load_amendment(cf.AMENDMENT_PATH, mini)
    assert body["arms"] == ["raw_ppo", "untyped_gnn_ppo", "ontology_rgat_ppo"]
    analysis = cf.Analysis(**{k: (tuple(v) if isinstance(v, list) else v)
                              for k, v in body["analysis"].items()})
    narrowed = replace(mini, models=replace(mini.models, arms=tuple(body["arms"])),
                       analysis=analysis, registered_hash=cf.plan_hash(mini, config))
    assert cf.plan_hash(narrowed, config) == cf.plan_hash(mini, config)
    assert len(cf.model_keys(narrowed)) == 18
    arms = {"raw_ppo": 0.70, "untyped_gnn_ppo": 0.60, "ontology_rgat_ppo": 0.80}
    verdict = cf.judge_local(narrowed, _models(narrowed, arms))
    assert set(verdict["primary"]) == {"untyped_gnn_ppo", "raw_ppo"}
    assert verdict["primary"]["raw_ppo"]["confidence"] == pytest.approx(0.975)
    assert verdict["practical"] == {} and verdict["practical_passed"] is None
    assert verdict["confirmation_passed"]


def test_an_amendment_cannot_drop_raw_or_move_guards(tmp_path, mini):
    data = yaml.safe_load(cf.AMENDMENT_PATH.read_text(encoding="utf-8"))
    bad = dict(data, arms=["semantic_mlp_ppo", "ontology_rgat_ppo"])
    (tmp_path / "a.yaml").write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="raw_ppo"):
        cf.load_amendment(tmp_path / "a.yaml", mini)
    moved = yaml.safe_load(cf.AMENDMENT_PATH.read_text(encoding="utf-8"))
    moved["analysis"]["ood_unsafe_ci_high_max"] = 0.05
    (tmp_path / "b.yaml").write_text(yaml.safe_dump(moved), encoding="utf-8")
    with pytest.raises(ValueError, match="guards"):
        cf.load_amendment(tmp_path / "b.yaml", mini)


def test_models_fly_as_soon_as_they_are_evaluated(tmp_path, plan):
    done = cf.model_root(tmp_path, plan, "20/raw_ppo")
    done.mkdir(parents=True)
    (done / "summary.json").write_text("{}")
    (done / "best.pt").write_bytes(b"x")
    training = cf.model_root(tmp_path, plan, "20/ontology_rgat_ppo")
    training.mkdir(parents=True)
    (training / "best.pt").write_bytes(b"still changing")
    bound, excluded = cf.bind_models(plan, tmp_path, ["20/raw_ppo"])
    assert set(bound) == {"20/raw_ppo"} and not excluded
    with pytest.raises(RuntimeError, match="not complete"):
        cf.bind_models(plan, tmp_path, ["20/ontology_rgat_ppo"])
    order = cf.model_order(plan, ["20/raw_ppo"])
    assert len(order) == len(plan.isaac.seeds) and {k for _, _, k in order} == {"20/raw_ppo"}
