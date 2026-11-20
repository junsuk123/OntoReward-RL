"""equal-budget/1 and isaac-equal-budget/1 (current_semantics/budget.py): fresh seeds, the
development cells, weights after exactly B updates, and the registered fixed-sequence rule."""
from __future__ import annotations

from dataclasses import replace
import inspect
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from ontology_rgat.current_semantics import budget, development as dev, orl, pilot, speed
from ontology_rgat.current_semantics.contracts import PROJECT_ROOT, load_config

CONFIGS = PROJECT_ROOT / "config" / "current_semantics"
BASE = CONFIGS / "spatial_view_contact_v1.yaml"
CLI = PROJECT_ROOT / "python" / "run_current_semantics_3d.py"


@pytest.fixture(scope="module")
def config():
    return load_config(BASE)


@pytest.fixture(scope="module")
def plan(config):
    return budget.load_plan(budget.DEFAULT_PATH, config)


def _plan_file(tmp_path: Path, source: Path = budget.DEFAULT_PATH, **changes) -> Path:
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    data.update(changes)
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_the_registered_plan_is_the_learning_speed_arms_on_fresh_seeds(plan):
    assert plan.proposed_arm == orl.ARMS["orl/1"]
    assert plan.comparisons == ("untyped_gnn_ppo", "raw_ppo")
    assert plan.training_seeds == tuple(range(50, 60))
    assert not set(plan.training_seeds) & budget.SPENT_MODEL_SEEDS
    assert set(range(40, 50)) <= budget.SPENT_MODEL_SEEDS     # learning-speed/1's seeds
    assert plan.budgets == (100, 200, 600) and plan.updates == 600


def test_the_cells_are_the_development_cells(config, plan):
    development = dev.load_development(dev.DEFAULT_DEVELOPMENT_PATH, config)
    assert plan.cells == development.cells
    assert plan.dropout_start_s == development.dropout_start_s
    assert [c.name for c in plan.cells] == list(dev.CELLS)


def test_spent_or_overlapping_seeds_are_refused(config, tmp_path):
    with pytest.raises(ValueError, match="model seed"):
        budget.load_plan(_plan_file(tmp_path, training_seeds=[45, 51, 52]), config)
    with pytest.raises(ValueError, match="overlap"):   # 300e6 + 50e6 = learning-speed/1's block
        budget.load_plan(_plan_file(tmp_path, train_seed_base=290_000_000), config)
    evaluation = {"episodes_per_cell": 50, "env_seed_base": 50_000_000,
                  "cells_from": "development/1"}           # development/1's environments
    with pytest.raises(ValueError, match="overlap"):
        budget.load_plan(_plan_file(tmp_path, evaluation=evaluation), config)


def test_budgets_must_increase_and_stay_within_the_recipe(config, tmp_path):
    with pytest.raises(ValueError, match="increasing"):
        budget.load_plan(_plan_file(tmp_path, budgets=[200, 100]), config)
    with pytest.raises(ValueError, match="exceeds"):
        budget.load_plan(_plan_file(tmp_path, budgets=[100, 900]), config)


def test_each_arm_trains_the_base_recipe_for_the_largest_budget(config, plan):
    acfg = budget.arm_config(config, plan, "raw_ppo")
    assert acfg.g4.arm == "raw_ppo" and acfg.g4.updates == 600
    assert acfg.g4.train_seed(50, 0, 0) == 450_000_000
    assert plan.eval_seed(0, 0, 0) == 70_000_000


def test_train_seed_keeps_budget_weights_without_changing_training():
    """Budget weights are written like live.pt (state-dict copy, no generator read), and
    validation is optional; the smoke run in status doc 24 compared the final weights."""
    params = inspect.signature(pilot.train_seed).parameters
    assert params["snapshots"].default == () and params["validate"].default is True
    source = inspect.getsource(pilot.train_seed)
    assert 'f"budget_{update + 1}.pt"' in source and "if validate and" in source


def test_the_plan_hash_binds_the_plan(config, plan, tmp_path):
    digest = budget.plan_hash(plan, config)
    changed = budget.load_plan(_plan_file(tmp_path, budgets=[100, 300, 600]), config)
    assert budget.plan_hash(changed, config) != digest


def _summaries(seeds, budgets, values):
    return {s: {"budgets": {str(b): {"mean_success": v[i]} for i, b in enumerate(budgets)}}
            for s, v in zip(seeds, values)}


def test_the_fixed_sequence_stops_at_the_first_unestablished_budget(plan):
    seeds = (50, 51, 52, 53)
    small = replace(plan, training_seeds=seeds)
    proposed = _summaries(seeds, plan.budgets, [(0.8, 0.85, 0.9), (0.82, 0.86, 0.88),
                                                (0.79, 0.84, 0.87), (0.81, 0.83, 0.9)])
    other = _summaries(seeds, plan.budgets, [(0.3, 0.84, 0.5), (0.35, 0.86, 0.5),
                                             (0.28, 0.85, 0.5), (0.4, 0.84, 0.5)])
    out = budget.compare(small, proposed, other)
    assert out["by_budget"]["100"]["verdict"] == "better"
    assert out["by_budget"]["200"]["verdict"] == "not established"
    third = out["by_budget"]["600"]
    assert third["tested"] is False and third["verdict"] is None
    assert third["descriptive_verdict"] == "better"           # reported, never claimed
    assert out["sequence_stopped_at"] == 200
    assert out["confidence"] == pytest.approx(0.975)


def test_worse_also_continues_the_sequence(plan):
    seeds = (50, 51, 52)
    small = replace(plan, training_seeds=seeds)
    proposed = _summaries(seeds, plan.budgets, [(0.1, 0.9, 0.9), (0.12, 0.9, 0.9),
                                                (0.11, 0.9, 0.9)])
    other = _summaries(seeds, plan.budgets, [(0.8, 0.5, 0.9), (0.81, 0.52, 0.9),
                                             (0.79, 0.49, 0.9)])
    out = budget.compare(small, proposed, other)
    assert [out["by_budget"][b]["verdict"] for b in ("100", "200", "600")] == \
        ["worse", "better", "not established"]


def test_the_isaac_plan_flies_fresh_seeds_at_a_registered_budget(config, plan, tmp_path):
    iplan = budget.load_isaac_plan(budget.ISAAC_DEFAULT_PATH, config, plan)
    assert iplan.budget == 600 and len(iplan.seeds) == 15
    assert [c for c, _ in iplan.conditions] == ["composition_id", "composition_ood",
                                               "combined_id"]
    assert len(budget.isaac_order(iplan, budget.isaac_keys(plan))) == 450
    with pytest.raises(ValueError, match="already used"):      # isaac-development's cases
        budget.load_isaac_plan(_plan_file(tmp_path, budget.ISAAC_DEFAULT_PATH,
                                          conditions={"composition_id": [994101]}),
                               config, plan)
    with pytest.raises(ValueError, match="optical noise"):
        budget.load_isaac_plan(_plan_file(tmp_path, budget.ISAAC_DEFAULT_PATH,
                                          conditions={"noise_ood": [994601]}), config, plan)
    with pytest.raises(ValueError, match="budget"):
        budget.load_isaac_plan(_plan_file(tmp_path, budget.ISAAC_DEFAULT_PATH, budget=300),
                               config, plan)


def _cli(*args, cwd):
    return subprocess.run([sys.executable, str(CLI), *args], capture_output=True, text=True,
                          cwd=cwd, timeout=300)


def test_the_cli_stages(tmp_path):
    done = _cli("--stage", "register-budget", "--dry-run", "--run-root", str(tmp_path),
                cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)["evaluation_episodes"] == 45_000
    assert not any(tmp_path.iterdir())
    done = _cli("--stage", "budget", "--run-root", str(tmp_path), cwd=tmp_path)
    assert done.returncode == 3 and "not registered" in done.stdout
    done = _cli("--stage", "isaac-budget", "--run-root", str(tmp_path), cwd=tmp_path)
    assert done.returncode == 3 and "allow-isaac" in done.stdout
    done = _cli("--stage", "register-isaac-budget", "--run-root", str(tmp_path), cwd=tmp_path)
    assert done.returncode == 3 and "not registered" in done.stdout
