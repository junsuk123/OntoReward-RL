"""Gate table and separated report states (guide 10, 12)."""
import json

import pytest

from ontology_rgat.current_semantics.contracts import (DEFAULT_CONFIG_PATH, contract_manifest,
                                                       load_config)
from ontology_rgat.current_semantics.experiment import (GATES, fixture_checks, gate_table,
                                                       report_states, resolved_plan,
                                                       run_smoke, write_json)


CONFIG = load_config()


@pytest.fixture(scope="module")
def fixtures():
    return fixture_checks(CONFIG)


def test_every_fixture_check_passes(fixtures):
    assert [c["name"] for c in fixtures if not c["passed"]] == []
    assert {c["name"] for c in fixtures} >= {"observation_round_trip", "phi44_layout",
                                            "graph_22_edges_two_hop_reach", "parameter_budget",
                                            "common_raw_critic",
                                            "encoder_gradient_every_relation",
                                            "manifest_complete"}


def test_gates_without_artifacts(fixtures):
    gates = gate_table(CONFIG, None, fixtures=fixtures)
    assert tuple(gates) == GATES
    assert {g: gates[g]["status"] for g in GATES} == {
        "G0": "waived", "G1": "waived", "G2": "passed", "G3": "not_started",
        "G4": "not_started"}
    # A waiver is recorded with what it leaves unresolved, never as evidence.
    unresolved = " ".join(gates["G0"]["unresolved"])
    assert "5853781" in unresolved and "evidence bundle" in unresolved
    assert all("waived by the user 2026-10-28" in gates[g]["waiver"] for g in ("G0", "G1", "G2"))
    assert gates["G0"]["evidence"] and not gates["G1"]["evidence"]
    assert any("TaskContract" in item for item in gates["G2"]["evidence"])
    assert gates["G2"]["blocking"] == []
    assert any("Isaac half" in item for item in gates["G3"]["blocking"])


def coupling_artifact(root, passed, g3_hash=None, *, isaac=False):
    manifest = contract_manifest(CONFIG)
    name, key = (("coupling_isaac.json", "isaac_checks_passed") if isaac
                 else ("coupling_local.json", "local_checks_passed"))
    write_json(root / name, {
        "schema_id": "spatial_current_semantics_v1", key: passed,
        "hashes": {"g3_hash": g3_hash or manifest["g3_hash"]}})


def test_g3_turns_partial_only_with_a_passing_current_local_report(tmp_path, fixtures):
    coupling_artifact(tmp_path, False)
    assert gate_table(CONFIG, tmp_path, fixtures=fixtures)["G3"]["status"] == "not_started"
    coupling_artifact(tmp_path, True, g3_hash="0" * 64)
    stale = gate_table(CONFIG, tmp_path, fixtures=fixtures)["G3"]
    assert stale["status"] == "not_started" and "different g3_hash" in stale["blocking"][0]
    coupling_artifact(tmp_path, True)
    g3 = gate_table(CONFIG, tmp_path, fixtures=fixtures)["G3"]
    assert g3["status"] == "partial" and any("Isaac half" in b for b in g3["blocking"])


def test_g3_passes_only_with_both_halves_current(tmp_path, fixtures):
    coupling_artifact(tmp_path, True, isaac=True)
    assert gate_table(CONFIG, tmp_path, fixtures=fixtures)["G3"]["status"] == "partial"
    coupling_artifact(tmp_path, True)
    assert gate_table(CONFIG, tmp_path, fixtures=fixtures)["G3"]["status"] == "passed"
    coupling_artifact(tmp_path, True, g3_hash="1" * 64, isaac=True)
    assert gate_table(CONFIG, tmp_path, fixtures=fixtures)["G3"]["status"] == "partial"


def test_a_failing_fixture_blocks_g2(fixtures):
    broken = [dict(fixtures[0], passed=False), *fixtures[1:]]
    assert gate_table(CONFIG, None, fixtures=broken)["G2"]["status"] == "blocked"


def test_report_states_are_separate_and_none_is_true_without_artifacts():
    states = report_states(gate_table(CONFIG, None))
    assert set(states) == {"procedure_complete", "implementation_acceptance",
                           "learnability_passed", "development_gate_passed",
                           "confirmation_opened", "scientific_goal_complete",
                           "isaac_integration_passed"}
    assert all(state["value"] is not True and state["reason"] for state in states.values())
    assert "PASSED" not in json.dumps(states)


def test_resolved_plan_is_strict_json_and_refuses_gated_stages(tmp_path):
    plan = resolved_plan(DEFAULT_CONFIG_PATH, "audit", tmp_path, dry_run=True)
    json.dumps(plan, allow_nan=False)
    assert plan["will_write"] == [str(tmp_path / "audit.json")] and not plan["refused"]
    refused = resolved_plan(DEFAULT_CONFIG_PATH, "develop", tmp_path, dry_run=True)
    assert refused["refused"] and refused["will_write"] == []
    reasons = refused["refusal_reasons"]
    assert not any(r.startswith(("G0", "G1", "G2")) for r in reasons)
    assert any(r.startswith("G3 not_started") for r in reasons)
    assert any(r.startswith("G4 not_started") for r in reasons)


def test_smoke_is_local_untrained_and_never_scientific():
    smoke = run_smoke(CONFIG, seed=3, steps=5)
    json.dumps(smoke, allow_nan=False)
    assert smoke["backend"] == "spatial_local_v12" and smoke["scientific_use"] is False
    assert smoke["untrained"] and smoke["steps_run"] == 5 and smoke["all_finite"]
    again = run_smoke(CONFIG, seed=3, steps=5)
    assert again["trace"] == smoke["trace"]
    with pytest.raises(ValueError):
        run_smoke(CONFIG, seed=3, steps=0)


def pilot_artifact(root, *, passed, complete=True, stale=False):
    manifest = contract_manifest(CONFIG)
    hashes = {key: manifest[key] for key in ("g4_hash", "algorithm_hash", "task_hash",
                                             "calibration_hash", "g3_hash")}
    if stale:
        hashes["task_hash"] = "2" * 64
    verdict = {"deterministic_success": 0.6 if passed else 0.0,
               "sampled_success": 0.3 if passed else 0.0, "passed": passed}
    write_json(root / "g4_pilot.json", {
        "schema_id": "spatial_current_semantics_v1", "hashes": hashes,
        "protocol_complete": complete, "verdicts": {"0": verdict, "1": verdict},
        "learnability_passed": complete and passed})


@pytest.mark.parametrize("passed, complete, stale, expected", [
    (True, True, False, "passed"), (False, True, False, "failed"),
    (True, False, False, "not_started"), (True, True, True, "not_started")])
def test_g4_reads_only_a_complete_current_pilot(tmp_path, fixtures, passed, complete, stale,
                                                expected):
    pilot_artifact(tmp_path, passed=passed, complete=complete, stale=stale)
    gates = gate_table(CONFIG, tmp_path, fixtures=fixtures)
    assert gates["G4"]["status"] == expected
    learn = report_states(gates)["learnability_passed"]["value"]
    assert learn == {"passed": True, "failed": False}.get(expected)
    if expected == "failed":
        assert "no representation conclusion" in gates["G4"]["blocking"][0]


def test_g4_names_a_seed_that_never_produced_a_testable_checkpoint(tmp_path, fixtures):
    pilot_artifact(tmp_path, passed=False)
    pilot = json.loads((tmp_path / "g4_pilot.json").read_text())
    pilot["verdicts"]["0"].update(deterministic_success=None, sampled_success=None)
    pilot["tests"] = {"0": {"checkpoint": None, "note": "no checkpoint reached difficulty 1.0"},
                      "1": {"checkpoint": "best.pt"}}
    write_json(tmp_path / "g4_pilot.json", pilot)
    gates = gate_table(CONFIG, tmp_path, fixtures=fixtures)
    assert gates["G4"]["status"] == "failed"
    assert gates["G4"]["evidence"] == [
        "model seed 0: not tested (no checkpoint reached difficulty 1.0)",
        "model seed 1: deterministic 0.0, sampled 0.0 on the test seeds"]


def test_implementation_acceptance_needs_g2_and_both_g3_halves(tmp_path, fixtures):
    assert report_states(gate_table(CONFIG, tmp_path, fixtures=fixtures))[
        "implementation_acceptance"]["value"] is False
    coupling_artifact(tmp_path, True)
    coupling_artifact(tmp_path, True, isaac=True)
    states = report_states(gate_table(CONFIG, tmp_path, fixtures=fixtures))
    assert states["implementation_acceptance"]["value"] is True
    assert states["scientific_goal_complete"]["value"] is False
