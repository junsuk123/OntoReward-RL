"""The panels must match what the run actually produces, now.

Three failure modes this guards, all of them found by auditing the page
against the runner on 2026-09-22:

* a panel that shows fewer things than the run has (the live pair plots were
  hardcoded to two while the stage sizes itself to four);
* a panel that does not exist for analysis the run does produce (the
  collection stage and its datasets had no panel at all);
* a panel wired to a literal that the run now publishes (the ontology arm was
  identified by comparing method ids against one hardcoded name).
"""
from __future__ import annotations

from pathlib import Path
import re

import pytest

from ontology_rgat.viz.dashboard import PAGE
from ontology_rgat.viz.live import BenchmarkMonitor, LiveStore
from run_three_pipeline import _arm_manifest, _baseline_arms

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------- the live pair panels follow N

def test_the_live_pair_plots_are_built_per_physical_pair():
    """Two hardcoded plots hid pairs 3 and 4 for a whole run."""
    assert "pair-plot-grid-${c.id}" in PAGE
    assert "parallel_pair_count" in PAGE
    # the old fixed-width construction must not survive
    assert "+[0,1].map(index=>`<div class=\"pair-plot\">" not in PAGE
    body = PAGE[PAGE.index("function drawPairPlots"):]
    body = body[:body.index("\nfunction ")]
    assert "for(let index=0;index<2;index++)" not in body
    assert "index<count" in body


def test_every_wide_live_panel_reads_the_same_pair_count():
    for function in ("drawTrajectories", "pairPanel", "drawPairPlots"):
        body = PAGE[PAGE.index(f"function {function}"):]
        body = body[:body.index("\nfunction ")]
        assert "parallel_pair_count" in body, function


# ------------------------------------------------ the collection stage panel

def test_the_collection_stage_has_a_panel():
    assert "id:'collection_stage'" in PAGE
    assert "collection-stage" in PAGE and "function collectionPanel" in PAGE
    # It is rendered, not merely declared.
    assert "collectionPanel(state);" in PAGE
    # and it names the two stages and the command that fills a gap
    assert "--stage collect" in PAGE and "--stage train" in PAGE


def test_the_monitor_publishes_what_that_panel_draws():
    store = LiveStore()
    monitor = BenchmarkMonitor(store)
    monitor.collection_stage(
        stage="collect", pairs=[0, 1, 2, 3], complete=True,
        datasets=[{"name": "FOV-risk", "episodes": 42,
                   "reused_episodes": 30, "flown_episodes": 12}])
    scalars = store.snapshot()["scalars"]
    assert scalars["collection_stage"] == "collect"
    assert scalars["collection_complete"] is True
    assert scalars["collection_pairs"] == [0, 1, 2, 3]
    assert scalars["collection_datasets"] == [
        {"name": "FOV-risk", "episodes": 42,
         "reused_episodes": 30, "flown_episodes": 12}]


def test_the_panel_reads_every_key_the_monitor_writes():
    body = PAGE[PAGE.index("function collectionPanel"):]
    body = body[:body.index("\nfunction ")]
    for key in ("collection_stage", "collection_datasets", "collection_pairs",
                "collection_complete"):
        assert key in body, key


# --------------------------------------- the ontology arm comes from the run

def test_the_run_declares_which_arm_carries_the_rgat_reward():
    config = {"pipelines": ["shin_se_fixed", "shin_se_onto_rgat_recovery"]}
    arms = _arm_manifest(config["pipelines"], _baseline_arms(config), {})
    by_method = {arm["method"]: arm for arm in arms}
    assert by_method["shin_se_fixed"]["ontology"] is False
    assert by_method["shin_se_onto_rgat_recovery"]["ontology"] is True


@pytest.mark.parametrize("name", ["onto_rgat_potential_pbrs_no_se",
                                  "onto_rgat_adaptive_weight_no_se"])
def test_other_ontology_pipelines_are_declared_too(name):
    """The literal the page used to compare against named exactly one of these."""
    arms = _arm_manifest([name], [], {})
    assert arms[0]["ontology"] is True


def test_the_page_identifies_the_ontology_arm_from_the_run_not_a_literal():
    assert "function isOntologyArm" in PAGE and "function ontologyArms" in PAGE
    assert "a.ontology===true" in PAGE
    assert "const PROPOSED_STEP=" in PAGE
    assert "series:PROPOSED_STEP" in PAGE
    # Literals may remain only as pre-configure fallbacks -- a frozen default,
    # a constant that syncArms refills, a label map, an ``||`` alternative --
    # never as something the page decides an arm's role by. Counting them is
    # too fragile to be worth asserting; what they are used FOR is not.
    for line in PAGE.splitlines():
        if "shin_se_onto_rgat_recovery" not in line:
            continue
        assert not re.search(r"==+\s*'shin_se_onto_rgat_recovery'", line), line
        assert not re.search(r"'shin_se_onto_rgat_recovery'\s*==+", line), line
    # and the two constants that name it are refilled from the run
    sync = PAGE[PAGE.index("function syncArms"):]
    sync = sync[:sync.index("\nfunction ")]
    assert "refill(PROPOSED_TRAIN" in sync and "refill(PROPOSED_STEP" in sync


def test_the_head_to_head_uses_the_ontology_arm_not_list_order():
    body = PAGE[PAGE.index("const learned=learnedArms();"):]
    body = body[:body.index("const evalRows")]
    assert "ontologyArms()[0]" in body
    assert "(learned[1]||{}).method" not in body
