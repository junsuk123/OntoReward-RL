"""The dashboard gates panels on an arm's ontology role, so the role must arrive.

``_arm_manifest`` computes ``ontology_role`` from the pipeline spec -- its
docstring says it exists so the dashboard does not have to guess how the
ontology takes part -- and ``BenchmarkMonitor.configure`` then rebuilt every
entry from three keys and dropped it. Every role reached the browser as
``None``, ``hasRole`` was permanently false, and the three panels built for the
current method (the situation graph's node values, its support nodes, its
embedding norm) had never been displayed once. A panel nobody can see is
indistinguishable from a panel nobody wrote.
"""
from __future__ import annotations

from pathlib import Path
import re
import sys
import threading

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "isaac_sim"))

from ontology_rgat.viz.live import BenchmarkMonitor       # noqa: E402


class _Store:
    """Just enough of the live store for ``configure`` to publish into."""

    def __init__(self):
        self.published = {}

    def set(self, **values):
        self.published.update(values)


def _configured(arms):
    monitor = BenchmarkMonitor.__new__(BenchmarkMonitor)
    monitor._state_lock = threading.RLock()
    monitor._phase_episode_counts = {}
    monitor._teacher_flights = {}
    monitor.store = _Store()
    BenchmarkMonitor.configure(
        monitor, methods=[a["method"] for a in arms], mode="full",
        config_hash="deadbeef", training_total=10, evaluation_total=10,
        arms=arms)
    # What the browser receives is the published copy, not the attribute.
    return monitor.store.published["benchmark_arms"]


def test_the_manifest_reaches_the_dashboard_whole():
    arms = _configured([
        {"method": "shin_se_fixed", "label": "Baseline", "learned": True,
         "ontology": False, "ontology_role": None,
         "graph_state_representation": None},
        {"method": "shin_se_onto_rgat_state", "label": "Proposed", "learned": True,
         "ontology": True, "ontology_role": "state_representation",
         "graph_state_representation": "situation_graph"},
    ])
    proposed = next(a for a in arms if a["method"] == "shin_se_onto_rgat_state")
    assert proposed["ontology_role"] == "state_representation"
    assert proposed["graph_state_representation"] == "situation_graph"
    assert proposed["ontology"] is True
    # The three fields the narrow rebuild did keep are still normalised.
    assert proposed["label"] == "Proposed" and proposed["learned"] is True


def test_a_bare_method_list_still_gets_usable_defaults():
    arms = _configured([{"method": "shin_se_fixed"}])
    assert arms == [{"method": "shin_se_fixed", "label": "shin_se_fixed",
                     "learned": True}]


def test_every_role_the_dashboard_gates_on_is_one_the_runner_can_emit():
    """A gate on a role no pipeline produces hides its panel forever."""
    dashboard = (ROOT / "python/ontology_rgat/viz/dashboard.py").read_text(
        encoding="utf-8")
    runner = (ROOT / "python/run_three_pipeline.py").read_text(encoding="utf-8")
    gated = set(re.findall(r"requiresRole:'([a-z_]+)'", dashboard))
    assert gated, "the gate exists; if it is removed, remove this test with it"
    emitted = set(re.findall(r'return "([a-z_]+)"', runner))
    missing = gated - emitted
    assert not missing, f"panels gated on roles nothing emits: {sorted(missing)}"
