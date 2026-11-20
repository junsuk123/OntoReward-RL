"""learning-speed/1 (current_semantics/speed.py): base navigation only, fresh seeds, and the
registered decision rule on the time to difficulty 1.0."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ontology_rgat.current_semantics import orl, speed
from ontology_rgat.current_semantics.contracts import PROJECT_ROOT, load_config

CONFIGS = PROJECT_ROOT / "config" / "current_semantics"
BASE = CONFIGS / "spatial_view_contact_v1.yaml"


@pytest.fixture(scope="module")
def config():
    return load_config(BASE)


def _plan_file(tmp_path: Path, **changes) -> Path:
    data = yaml.safe_load(speed.DEFAULT_PATH.read_text(encoding="utf-8"))
    data.update(changes)
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_the_registered_plan_is_orl_against_untyped_and_raw_on_fresh_seeds(config):
    plan = speed.load_plan(speed.DEFAULT_PATH, config)
    assert plan.proposed_arm == orl.ARMS["orl/1"]
    assert plan.comparisons == ("untyped_gnn_ppo", "raw_ppo")
    assert plan.training_seeds == tuple(range(40, 50)) and plan.cap == 600
    assert not set(plan.training_seeds) & speed.SPENT_MODEL_SEEDS


@pytest.mark.parametrize("name", ["spatial_view_contact_v1_nav4_wmon.yaml",
                                  "spatial_view_contact_v1_nav4_kalman.yaml"])
def test_a_navigation_estimator_config_is_refused(name):
    with pytest.raises(ValueError, match="base navigation"):
        speed.load_plan(speed.DEFAULT_PATH, load_config(CONFIGS / name))


def test_spent_model_or_episode_seeds_are_refused(config, tmp_path):
    with pytest.raises(ValueError, match="model seed"):
        speed.load_plan(_plan_file(tmp_path, training_seeds=[20, 41, 42]), config)
    with pytest.raises(ValueError, match="overlap"):   # 80e6 + 40e6 = confirmation's block
        speed.load_plan(_plan_file(tmp_path, train_seed_base=80_000_000), config)


def test_each_arm_trains_the_base_recipe_up_to_the_cap(config):
    plan = speed.load_plan(speed.DEFAULT_PATH, config)
    acfg = speed.arm_config(config, plan, "raw_ppo")
    assert acfg.g4.arm == "raw_ppo" and acfg.g4.updates == plan.cap
    assert acfg.g4.train_seed(40, 0, 0) == 340_000_000
    assert acfg.g4.ppo == config.g4.ppo and acfg.g4.curriculum == config.g4.curriculum


def test_the_plan_hash_binds_the_plan(config, tmp_path):
    plan = speed.load_plan(speed.DEFAULT_PATH, config)
    other = speed.load_plan(_plan_file(tmp_path, cap=500), config)
    assert speed.plan_hash(plan, config) != speed.plan_hash(other, config)


def _times(values, censored=()):
    return {i: {"first_full_update": None if i in censored else v, "censored": i in censored}
            for i, v in enumerate(values)}


def test_the_decision_rule(config):
    plan = speed.load_plan(speed.DEFAULT_PATH, config)
    fast = _times([60, 70, 80, 65, 75, 70, 66, 72, 68, 74])
    slow = _times([100, 110, 130, 95, 120, 105, 99, 140, 101, 115])
    assert speed.compare(plan, fast, slow)["verdict"] == "faster"
    assert speed.compare(plan, slow, fast)["verdict"] == "slower"
    # a censored comparison model only strengthens the case; a censored ORL model blocks it
    assert speed.compare(plan, fast, _times([100] * 10, censored={3}))["verdict"] == "faster"
    blocked = speed.compare(plan, _times([60] * 10, censored={3}), slow)
    assert blocked["verdict"] == "not established" and blocked["proposed_censored"] == 1
    mixed = speed.compare(plan, _times([60, 140] * 5), _times([100] * 10))
    assert mixed["verdict"] == "not established"
