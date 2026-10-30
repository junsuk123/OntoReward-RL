"""Guide 11: the development stage (development.py): plan, registration, cells, gate."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import multiprocessing
from pathlib import Path

import pytest
import yaml

from ontology_rgat.current_semantics import development as dev
from ontology_rgat.current_semantics.contracts import (DEFAULT_CONFIG_PATH, contract_manifest,
                                                       load_config)
from ontology_rgat.current_semantics.models import ARMS, CurrentActorCritic
from ontology_rgat.current_semantics.pilot import _init_worker

CONFIG = load_config()
PLAN = dev.load_development(dev.DEFAULT_DEVELOPMENT_PATH, CONFIG)


def _plan_file(tmp_path, edit):
    data = yaml.safe_load(Path(dev.DEFAULT_DEVELOPMENT_PATH).read_text(encoding="utf-8"))
    edit(data)
    out = tmp_path / "plan.yaml"
    out.write_text(yaml.safe_dump(data), encoding="utf-8")
    return out


def test_registered_plan_is_the_guide_design():
    assert PLAN.arms == ARMS and PLAN.proposed_arm == "ontology_rgat_ppo"
    assert len(PLAN.training_seeds) == 5 and not set(PLAN.training_seeds) & set(CONFIG.g4.model_seeds)
    assert tuple(c.name for c in PLAN.cells) == dev.CELLS and len(dev.CELLS) == 10
    assert PLAN.episodes_per_cell == 100
    assert PLAN.cell("composition_ood").split == "composition"
    plan = dev.resolved(PLAN, CONFIG)
    assert plan["models"] == 20 and plan["evaluation_episodes"] == 20_000
    assert plan["starts_isaac"] is False


@pytest.mark.parametrize("edit, match", [
    (lambda d: d["evaluation"]["cells"]["motion_ood"].update(pad_acceleration_m_s2=[0.2, 0.9]),
     "beyond the trained range"),
    (lambda d: d["evaluation"]["cells"]["dropout_ood"].update(dropout_s=[2.0, 5.0]),
     "beyond the trained range"),
    (lambda d: d["evaluation"]["cells"]["dropout_id"].update(dropout_s=[1.0, 2.0]),
     "trained blackout range"),
    (lambda d: d["evaluation"]["cells"]["combined_ood"].update(optical_noise_scale=3.0),
     "combine exactly"),
    (lambda d: d["evaluation"]["cells"]["noise_id"].update(optical_noise_scale=2.0),
     "nominal condition"),
    (lambda d: d["evaluation"]["cells"]["composition_ood"].update(split="in_distribution"),
     "composition split"),
    (lambda d: d.update(training_seeds=[0, 11, 12, 13, 14]), "G4 pilot model seed"),
    (lambda d: d.update(validation_seeds=[992001] + d["validation_seeds"][1:]),
     "registered seed"),
    (lambda d: d.update(train_seed_base=40_000_000), "overlap"),
    (lambda d: d.update(train_seed_base=-7_000_000), "train_seed_base"),
    (lambda d: d.update(proposed_arm="raw_ppo"), "ontology_rgat_ppo"),
    (lambda d: d.update(extra=1), "unknown keys"),
])
def test_plan_refuses_what_guide_11_forbids(tmp_path, edit, match):
    with pytest.raises(ValueError, match=match):
        dev.load_development(_plan_file(tmp_path, edit), CONFIG)


def test_seed_blocks_do_not_overlap_each_other_or_recorded_use():
    per = CONFIG.g4.episodes_per_update
    last_train = PLAN.train_seed(max(PLAN.training_seeds), CONFIG.g4.updates - 1, per - 1, per)
    first_eval = PLAN.eval_seed(0, 0, 0)
    assert last_train < first_eval
    assert PLAN.eval_seed(4, 9, 99) - first_eval < 1_000_000


def test_arm_config_changes_only_the_arm_and_the_seeds():
    acfg = dev.arm_config(CONFIG, PLAN, "untyped_gnn_ppo")
    a, b = contract_manifest(CONFIG), contract_manifest(acfg)
    for key in ("algorithm_hash", "task_hash", "feature_hash", "graph_hash", "calibration_hash"):
        assert a[key] == b[key], key
    g4 = acfg.g4
    assert g4.arm == "untyped_gnn_ppo" and g4.validation_seeds == PLAN.validation_seeds
    assert (g4.updates, g4.episodes_per_update, g4.ppo, g4.curriculum,
            g4.training_randomization) == (CONFIG.g4.updates, CONFIG.g4.episodes_per_update,
                                           CONFIG.g4.ppo, CONFIG.g4.curriculum,
                                           CONFIG.g4.training_randomization)
    assert g4.train_seed(10, 0, 0) == PLAN.train_seed(10, 0, 0, g4.episodes_per_update)


def _models(rates, unsafe=0.0, validation=(1.0, 1.0)):
    out = {}
    for s in PLAN.training_seeds:
        for arm in ARMS:
            cells = {c: {"success": rates(arm, c), "unsafe": unsafe if arm != "raw_ppo" else 0.0}
                     for c in dev.CELLS}
            out[f"{s}/{arm}"] = {"validation": {"deterministic_success": validation[0],
                                                "sampled_success": validation[1]},
                                 "cells": cells}
    return out


def test_gate_passes_only_when_every_criterion_holds():
    better = lambda arm, cell: 0.8 if arm == "ontology_rgat_ppo" else 0.7
    assert dev.judge(PLAN, _models(better))["development_gate_passed"] is True
    # Composition must beat EVERY comparison arm.
    tie = lambda arm, cell: (0.7 if cell == "composition_ood" and arm == "semantic_mlp_ppo"
                             else better(arm, cell))
    tie_mlp = lambda arm, cell: 0.8 if arm in ("ontology_rgat_ppo", "semantic_mlp_ppo") \
        and cell == "composition_ood" else better(arm, cell)
    verdict = dev.judge(PLAN, _models(tie_mlp))
    assert verdict["composition_margins"]["mlp"]["passed"] is False
    assert verdict["development_gate_passed"] is False
    # ID guard: a 3 pp drop against Raw fails even with a composition win.
    drop = lambda arm, cell: (0.67 if arm == "ontology_rgat_ppo" and cell.endswith("_id")
                              else better(arm, cell))
    assert dev.judge(PLAN, _models(drop))["guards"]["id_success_vs_raw"]["passed"] is False
    # Unsafe guard: +2 pp over Raw on the OOD cells fails.
    assert dev.judge(PLAN, _models(better, unsafe=0.02))["guards"]["ood_unsafe_vs_raw"][
        "passed"] is False
    # Any seed below the validation criterion fails learnability; no seed is dropped.
    v = dev.judge(PLAN, _models(better, validation=(0.4, 1.0)))
    assert v["learnability_passed"] is False and v["development_gate_passed"] is False


def test_register_needs_a_passed_g4_and_never_overwrites(tmp_path):
    status, payload = dev.register(DEFAULT_CONFIG_PATH, dev.DEFAULT_DEVELOPMENT_PATH, tmp_path)
    assert status == 3 and "G4" in payload["reason"]
    manifest = contract_manifest(CONFIG)
    (tmp_path / "g4_pilot.json").write_text(json.dumps({
        "protocol_complete": True, "learnability_passed": True,
        "hashes": {k: manifest[k] for k in ("g4_hash", "algorithm_hash", "task_hash",
                                            "calibration_hash")}}))
    status, payload = dev.register(DEFAULT_CONFIG_PATH, dev.DEFAULT_DEVELOPMENT_PATH, tmp_path)
    assert status == 0 and payload["status"] == "REGISTERED"
    (tmp_path / dev.REGISTRATION).write_text(json.dumps(payload))
    assert dev.check_registration(CONFIG, PLAN, tmp_path) is None
    assert dev.register(DEFAULT_CONFIG_PATH, dev.DEFAULT_DEVELOPMENT_PATH,
                        tmp_path)[1]["status"] == "ALREADY_REGISTERED"
    other = _plan_file(tmp_path, lambda d: d["gates"].update(id_drop_max=0.05))
    assert dev.register(DEFAULT_CONFIG_PATH, other, tmp_path)[0] == 3
    changed = dev.load_development(other, CONFIG)
    assert "differs" in dev.check_registration(CONFIG, changed, tmp_path)
    status, payload = dev.run_development(DEFAULT_CONFIG_PATH, other, tmp_path)
    assert status == 3 and payload["status"] == "REFUSED"


def test_one_model_flies_every_cell_with_its_conditions():
    model = CurrentActorCritic("ontology_rgat_ppo", CONFIG, seed=3)
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    plan = replace(PLAN, episodes_per_cell=1)
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_init_worker, initargs=(str(DEFAULT_CONFIG_PATH),)) as pool:
        cells = dev.evaluate_model(pool, plan, "ontology_rgat_ppo", state, seed_index=0)
    assert set(cells) == set(dev.CELLS)
    for name, cell in cells.items():
        assert cell["episodes"] + cell["refused"] == 1, name
        episode = cell["per_seed"][0]
        if not episode.get("refused"):
            assert episode["scenario"]["split"] == plan.cell(name).split
    assert cells["composition_ood"]["per_seed"][0]["seed"] == plan.eval_seed(0, 1, 0)
