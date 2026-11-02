"""isaac-handover-fix/1 re-flies isaac-development/1 cases chosen by a rule, not by outcome.

The rule (first N environments of the reference flight, in seed order, whose one-tick
handover kick reached PX4 wrapped; environment i meets all four arms of one training
seed, rotating) is recomputed from the configs on every load, so the registered cases
cannot drift toward environments that happened to go well or badly.
"""
import collections
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from ontology_rgat.current_semantics import development as dev  # noqa: E402
from ontology_rgat.current_semantics import isaac_development as idev  # noqa: E402
from ontology_rgat.current_semantics import isaac_handover_fix as fix  # noqa: E402
from ontology_rgat.current_semantics.contracts import DEFAULT_CONFIG_PATH, load_config  # noqa: E402


@pytest.fixture(scope="module")
def cases():
    config = load_config(DEFAULT_CONFIG_PATH)
    dev_plan = dev.load_development(dev.DEFAULT_DEVELOPMENT_PATH, config)
    reference = idev.load_plan(idev.DEFAULT_PATH, config, dev_plan)
    return fix.select_cases(fix.load_plan(fix.DEFAULT_PATH), reference, dev_plan), dev_plan


def test_cases_are_the_first_wrapped_environments_in_seed_order(cases):
    chosen, _ = cases
    physics_dt, accel_range = fix.imu_limits()
    assert accel_range * physics_dt == pytest.approx(0.628, abs=0.001)
    environments = list(dict.fromkeys(seed for _, seed, _ in chosen))
    wrapped = [s for s in range(994101, 994131) if fix.kick_wrapped(s, physics_dt, accel_range)]
    assert environments == wrapped[:10]
    assert environments == [994101, 994103, 994104, 994105, 994106,
                            994108, 994109, 994110, 994112, 994114]
    # A kick the IMU could measure in one tick never qualifies.
    assert not fix.kick_wrapped(994107, physics_dt, accel_range)


def test_each_environment_meets_every_arm_of_one_training_seed(cases):
    chosen, dev_plan = cases
    assert len(chosen) == 40 and len(set(chosen)) == 40
    per_environment = collections.defaultdict(set)
    for _, seed, key in chosen:
        per_environment[seed].add(key)
    for i, (seed, keys) in enumerate(per_environment.items()):
        training = dev_plan.training_seeds[i % len(dev_plan.training_seeds)]
        assert keys == {f"{training}/{arm}" for arm in dev_plan.arms}
    flights = collections.Counter(key for _, _, key in chosen)
    assert set(flights) == set(idev.model_keys(dev_plan)) and set(flights.values()) == {2}


def test_plan_rejects_unregistered_keys(tmp_path):
    text = fix.DEFAULT_PATH.read_text(encoding="utf-8")
    bad = tmp_path / "plan.yaml"
    bad.write_text(text + "extra_seed: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        fix.load_plan(bad)
    bad.write_text(text.replace("mode: deterministic", "mode: sampled"), encoding="utf-8")
    with pytest.raises(ValueError, match="deterministic"):
        fix.load_plan(bad)


def test_second_protocol_flies_exactly_the_environments_the_first_did_not(cases):
    first, dev_plan = cases
    config = load_config(DEFAULT_CONFIG_PATH)
    reference = idev.load_plan(idev.DEFAULT_PATH, config, dev_plan)
    plan = fix.load_plan(fix.DEFAULT_PATH.with_name("isaac_handover_fix_v2.yaml"))
    assert plan.protocol == "isaac-handover-fix/2"
    second = fix.select_cases(plan, reference, dev_plan)
    one = {seed for _, seed, _ in first}
    two = list(dict.fromkeys(seed for _, seed, _ in second))
    every = sorted(seed for _, seeds in reference.conditions for seed in seeds)
    assert not one & set(two) and sorted(one | set(two)) == every and two == sorted(two)
    assert len(second) == 80 and len(set(second)) == 80
    # Same rotation as /1: after both, every development model has flown six times.
    flights = collections.Counter(key for _, _, key in first + second)
    assert set(flights) == set(idev.model_keys(dev_plan)) and set(flights.values()) == {6}
    # /1 keeps its file names; /2 writes beside them, never over them.
    assert fix.files("isaac-handover-fix/1") == (fix.REGISTRATION, fix.RESULT, fix.OUT_DIR)
    assert not set(fix.files("isaac-handover-fix/2")) & set(fix.files("isaac-handover-fix/1"))
