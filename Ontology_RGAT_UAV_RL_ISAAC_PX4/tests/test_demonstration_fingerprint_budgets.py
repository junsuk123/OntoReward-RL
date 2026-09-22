"""Transport budgets must not decide whether stored teacher flights are reused.

Keying the demonstration fingerprint on the whole ``system.benchmark`` block
re-flew the four-flight teacher set three times on 2026-09-21, each time a
gateway or entry budget was widened. A budget only decides when an attempt is
abandoned as infrastructure failure; a flight that completed under one budget
is the same demonstration under another.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "isaac_sim"))

from config_loader import load_config as load_system  # noqa: E402

from ontology_rgat.benchmarks.experiment import load_experiment  # noqa: E402
from run_three_pipeline import (_DEMONSTRATION_BUDGET_KEYS,  # noqa: E402
                                behavior_cloning_settings,
                                demonstration_fingerprint)

CFG = SimpleNamespace(sim=SimpleNamespace(max_steps=300))


def _fingerprint(system):
    from conftest import default_experiment_config

    config = load_experiment(default_experiment_config())
    return demonstration_fingerprint(
        config, behavior_cloning_settings(config), cfg=CFG, system=system)


def test_budget_keys_do_not_change_the_demonstration_fingerprint():
    system = load_system(ROOT / "config/shin2026-minimal-system.yaml")
    assert set(_DEMONSTRATION_BUDGET_KEYS) <= set(system["benchmark"]), (
        "the shipped benchmark block should declare every excluded budget")
    baseline = _fingerprint(system)
    widened = deepcopy(system)
    widened["benchmark"]["gateway_timeout_s"] = 99.0
    widened["benchmark"]["setup_timeout_s"] = 9999.0
    widened["benchmark"]["reset_recoveries"] = 7
    widened["benchmark"]["entry_timeout_s"] = 4321.0
    assert _fingerprint(widened) == baseline
    stripped = deepcopy(system)
    for key in _DEMONSTRATION_BUDGET_KEYS:
        stripped["benchmark"].pop(key, None)
    assert _fingerprint(stripped) == baseline


def test_flight_relevant_benchmark_keys_still_change_it():
    system = load_system(ROOT / "config/shin2026-minimal-system.yaml")
    baseline = _fingerprint(system)
    changed = deepcopy(system)
    changed["benchmark"]["entry_speed_tolerance_m_s"] = float(
        changed["benchmark"]["entry_speed_tolerance_m_s"]) + 0.5
    assert _fingerprint(changed) != baseline
    changed = deepcopy(system)
    changed["benchmark"]["entry_sim_budget_s"] = float(
        changed["benchmark"]["entry_sim_budget_s"]) + 30.0
    assert _fingerprint(changed) != baseline


def test_the_servo_stabilisation_knobs_are_part_of_the_fingerprint():
    """Demonstrations flown with the raw derivative are a different set.

    ``rate_filter_s`` and ``anti_windup`` change what the image servo flies,
    not how long it is given to fly it, so a stored set must not be reused
    across them (2026-09-21: the unfiltered loop landed 1 of 160 offline
    flights and 0 of 26 real ones).
    """
    from run_three_pipeline import _DEMONSTRATION_FLIGHT_KEYS

    config = load_experiment(
        ROOT / "config/experiments/two_pipeline_comparison.yaml")
    settings = behavior_cloning_settings(config)
    system = load_system(ROOT / "config/shin2026-minimal-system.yaml")
    baseline = demonstration_fingerprint(config, settings, cfg=CFG, system=system)
    for key, value in (("rate_filter_s", 0.0), ("anti_windup", False),
                       ("integral_leak_s", 12.0)):
        assert key in _DEMONSTRATION_FLIGHT_KEYS, key
        changed = dict(settings)
        changed[key] = value
        assert demonstration_fingerprint(
            config, changed, cfg=CFG, system=system) != baseline, key


def test_servo_only_knobs_stay_out_of_a_privileged_teacher_configuration():
    """A knob the selected teacher cannot read must not key its flights.

    The stabilisation values ship as code defaults. Writing them into the
    experiment while the privileged PD is the teacher would change the
    fingerprint of demonstrations no servo knob can affect -- the same
    mistake the transport budgets were excluded for, and it re-flies a stored
    set. They belong in the file when the servo is the teacher.
    """
    from run_three_pipeline import PRIVILEGED_VELOCITY_TEACHER

    from conftest import default_experiment_config

    for path in {default_experiment_config(),
                 ROOT / "config/experiments/two_pipeline_comparison.yaml"}:
        settings = behavior_cloning_settings(load_experiment(path))
        if str(settings.get("teacher")) != PRIVILEGED_VELOCITY_TEACHER:
            continue
        for key in ("rate_filter_s", "integral_leak_s", "anti_windup"):
            assert key not in settings, (
                f"{path.name}: {key} is a servo knob; recorded here it would "
                "re-fly the PD set")
