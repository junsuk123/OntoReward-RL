"""``./run.sh current-semantics-3d``: dry runs create nothing, gated stages refuse.

Every invocation passes an explicit stage and writes only under tmp_path; no
stage here trains, and none can reach a flight stack (coupling --backend isaac
is refused before anything is imported).
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from ontology_rgat.current_semantics.experiment import LOCK_NAME, owned


PROJECT = Path(__file__).resolve().parents[2]
REPOSITORY = PROJECT.parent
RUN_SH = REPOSITORY / "run.sh"
ROUTE = "current-semantics-3d"
GATED = ("register-development", "develop", "register-confirmation", "confirm")


def run(*args, timeout=180):
    return subprocess.run(["bash", str(RUN_SH), ROUTE, *args], capture_output=True,
                          text=True, timeout=timeout, cwd=PROJECT)


def payload(done):
    start = done.stdout.index("{")
    return json.loads(done.stdout[start:done.stdout.rindex("}") + 1])


def test_help_lists_the_route_and_keeps_the_guarded_strings():
    done = subprocess.run(["bash", str(RUN_SH), "--help"], capture_output=True, text=True,
                          timeout=30)
    assert done.returncode == 0
    for text in (ROUTE, "WHOLE minimal-observation system", "TRAINS", "FLIES",
                 "reference-smoke"):
        assert text in done.stdout


def test_every_pre_existing_route_is_still_dispatched():
    body = RUN_SH.read_text()
    for name in ("system", "all", "reference-smoke", "reference", "status", "spatial",
                 "matlab-port", "matlab-port-all", "matlab-port-final",
                 "matlab-port-dashboard", "isaac-legacy", ROUTE):
        assert re.search(rf"^\s+{re.escape(name)}\)\s*$", body, re.M), name
    assert "--takeover" not in body.split("case")[0]


@pytest.mark.parametrize("stage, extra", [("audit", ()), ("fixtures", ()),
                                          ("smoke", ("--steps", "4")),
                                          ("coupling", ("--backend", "local")),
                                          ("report", ())])
def test_dry_run_prints_the_plan_and_creates_nothing(tmp_path, stage, extra):
    root = tmp_path / "x"
    done = run("--stage", stage, "--dry-run", "--run-root", str(root), *extra)
    assert done.returncode == 0, done.stderr
    plan = payload(done)
    assert plan["schema_id"] == "spatial_current_semantics_v1" and plan["dry_run"]
    assert plan["manifest"]["calibration_hash"] and plan["manifest"]["algorithm_hash"]
    assert set(plan["gates"]) == {"G0", "G1", "G2", "G3", "G4"}
    assert plan["gates"]["G0"]["status"] == "waived" and "2026-10-28" in plan["gates"]["G0"]["waiver"]
    assert not root.exists()


@pytest.mark.parametrize("stage", GATED)
@pytest.mark.parametrize("dry", [False, True])
def test_development_and_confirmation_stages_are_refused(tmp_path, stage, dry):
    root = tmp_path / "x"
    done = run("--stage", stage, "--run-root", str(root), *(["--dry-run"] if dry else []))
    assert done.returncode == 3
    assert payload(done)["status"] == "REFUSED"
    assert not root.exists()


@pytest.mark.parametrize("dry", [False, True])
def test_isaac_coupling_without_allow_isaac_is_refused_and_contacts_nothing(tmp_path, dry):
    root = tmp_path / "x"
    done = run("--stage", "coupling", "--backend", "isaac", "--run-root", str(root),
               *(["--dry-run"] if dry else []))
    assert done.returncode == 3
    assert "--allow-isaac" in payload(done)["reason"]
    assert not root.exists()


def test_isaac_coupling_dry_run_shows_the_owned_stack_plan_and_starts_nothing(tmp_path):
    # NEVER drop --dry-run here: with --allow-isaac the stage starts Isaac Sim + PX4.
    root = tmp_path / "x"
    done = run("--stage", "coupling", "--backend", "isaac", "--allow-isaac", "--dry-run",
               "--run-root", str(root))
    assert done.returncode == 0, done.stderr
    plan = payload(done)
    assert plan["backend"] == "isaac" and plan["starts_owned_stack"] is True
    assert plan["adopts_running_isaac"] is False and plan["isaac_seeds"] == [990001, 990002]
    assert plan["will_write"] == [str(root / "coupling_isaac.json")]
    assert not root.exists()


@pytest.mark.parametrize("args", [("--stage", "audit", "--seed", "1"),
                                  ("--stage", "fixtures", "--steps", "3"),
                                  ("--stage", "smoke", "--backend", "local"),
                                  ("--stage", "report", "--allow-isaac"),
                                  ("--stage", "coupling", "--backend", "local",
                                   "--allow-isaac"),
                                  ("--stage", "coupling", "--backend", "local", "--headless"),
                                  ("--stage", "coupling"),
                                  ("--stage", "develop", "--seed", "1")])
def test_stage_specific_options_are_rejected_elsewhere(tmp_path, args):
    done = run(*args, "--run-root", str(tmp_path / "x"))
    assert done.returncode == 2
    assert not (tmp_path / "x").exists()


def test_fixtures_smoke_coupling_then_report_shows_g3_partial(tmp_path):
    root = tmp_path / "run"
    done = run("--stage", "fixtures", "--run-root", str(root))
    assert done.returncode == 0, done.stdout + done.stderr
    fixtures = json.loads((root / "fixtures.json").read_text())
    assert fixtures["all_passed"] and all(c["passed"] for c in fixtures["checks"])

    done = run("--stage", "smoke", "--steps", "8", "--run-root", str(root))
    assert done.returncode == 0, done.stdout + done.stderr
    smoke = json.loads((root / "smoke.json").read_text())
    assert smoke["scientific_use"] is False and smoke["backend"] == "spatial_local_v12"
    assert smoke["steps_run"] == 8 and smoke["all_finite"]

    done = run("--stage", "report", "--run-root", str(root))
    assert payload(done)["gates"]["G3"]["status"] == "not_started"

    done = run("--stage", "coupling", "--backend", "local", "--run-root", str(root),
               timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
    assert json.loads((root / "coupling_local.json").read_text())["local_checks_passed"]

    before = sorted(p.name for p in root.iterdir())
    report = payload(run("--stage", "report", "--run-root", str(root)))
    assert report["gates"]["G3"]["status"] == "partial"
    assert sorted(p.name for p in root.iterdir()) == before
    assert not (root / LOCK_NAME).exists()


def test_audit_writes_one_atomic_json(tmp_path):
    root = tmp_path / "run"
    done = run("--stage", "audit", "--run-root", str(root))
    assert done.returncode == 0, done.stderr
    audit = json.loads((root / "audit.json").read_text())
    assert audit["stage"] == "audit" and not audit["dry_run"]
    assert audit["provenance"]["git_head"]
    assert sorted(p.name for p in root.iterdir()) == ["audit.json"]


def test_relative_run_root_resolves_against_the_launch_directory(tmp_path):
    done = subprocess.run(["bash", str(RUN_SH), ROUTE, "--stage", "audit", "--dry-run",
                           "--run-root", "rel/x"], capture_output=True, text=True,
                          timeout=180, cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    assert payload(done)["run_root"] == str(tmp_path / "rel" / "x")
    assert not (tmp_path / "rel").exists()


def test_lock_refuses_a_live_owner_and_clears_a_dead_one(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    lock = root / LOCK_NAME
    lock.write_text(f"{os.getpid()} now\n")
    with pytest.raises(SystemExit):
        with owned(root):
            pass
    assert lock.read_text().startswith(str(os.getpid()))

    child = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"],
                           capture_output=True, text=True, check=True)
    lock.write_text(f"{child.stdout.strip()} then\n")
    with owned(root):
        assert lock.read_text().split()[0] == str(os.getpid())
    assert not lock.exists()

    lock.write_text("garbage\n")
    with owned(root):
        pass
    assert not lock.exists()
