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
    config = load_experiment(
        ROOT / "config/experiments/two_pipeline_comparison.yaml")
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
