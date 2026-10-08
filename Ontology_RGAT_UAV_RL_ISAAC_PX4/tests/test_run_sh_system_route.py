"""Bare ./run.sh runs the whole minimal-observation system (user decision 2026-10-08).

Never executes the working route (it trains for hours): only --help and
--dry-run, which must build, train, lock and fly nothing.
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT.parent / "run.sh"


def run(*args):
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, timeout=120)


def test_bare_route_is_the_system_and_dry_run_touches_nothing(tmp_path):
    root = tmp_path / "system"
    plan = run("--dry-run", "--run-root", str(root))
    assert plan.returncode == 0, plan.stdout + plan.stderr
    assert "nothing is built, trained or flown" in plan.stdout
    assert '"train_scenario": "dr"' in plan.stdout
    assert "828, 829, 830, 831, 832" in plan.stdout
    assert not root.exists()


def test_no_isaac_keeps_the_system_local(tmp_path):
    plan = run("--no-isaac", "--dry-run", "--run-root", str(tmp_path / "x"))
    assert plan.returncode == 0
    assert "Isaac/PX4: off" in plan.stdout and "pass --isaac" in plan.stdout


def test_the_launcher_never_takes_over_a_running_isaac():
    body = (ROOT / "scripts/run_minimal_system.sh").read_text()
    assert "already running (not taken over)" in body
    assert "--adopt-stack" not in body and "--takeover" not in body


def test_help_names_the_system_route_and_that_it_flies():
    text = run("--help").stdout
    assert "WHOLE minimal-observation system" in text and "FLIES" in text


def test_zero_arguments_reach_the_system_launcher(tmp_path):
    """`set -u` plus `A || B && C` once read an unset $1 and exited at line 12
    before anything ran. Exercise the no-argument branch without the work:
    a stub launcher on a copied run.sh records what it was handed."""
    import shutil
    root = tmp_path / "repo"
    (root / "Ontology_RGAT_UAV_RL_ISAAC_PX4" / "scripts").mkdir(parents=True)
    shutil.copy(SCRIPT, root / "run.sh")
    stub = root / "Ontology_RGAT_UAV_RL_ISAAC_PX4" / "scripts" / "run_minimal_system.sh"
    stub.write_text('#!/usr/bin/env bash\necho "SYSTEM LAUNCHER args=[$*]"\n')
    out = subprocess.run(["bash", str(root / "run.sh")], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    assert "SYSTEM LAUNCHER args=[]" in out.stdout
    out = subprocess.run(["bash", str(root / "run.sh"), "--no-isaac"], capture_output=True, text=True, timeout=30)
    assert "SYSTEM LAUNCHER args=[--no-isaac]" in out.stdout


def test_a_bc_stage_killed_mid_dagger_is_redone(tmp_path, monkeypatch):
    """2026-10-08: the DAgger run died in round 2 with all 15 .pt files on
    disk, half from round 1. Resume must refit them, not hand them to PPO."""
    import importlib.util
    import types
    spec = importlib.util.spec_from_file_location(
        "minimal_full_pipeline", ROOT / "tools" / "minimal_full_pipeline.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    calls = []
    monkeypatch.setattr(driver.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    (tmp_path / "logs").mkdir()
    bc = tmp_path / "bc"
    bc.mkdir()
    for n in driver.ARMS:
        (bc / f"{n}__seed828.pt").write_bytes(b"")
    args = types.SimpleNamespace(seeds=[828], sweep=tmp_path / "s.json", workers=1,
                                 bc_episodes=1, train_scenario=None, dagger_rounds=2)
    driver.stage_bc(tmp_path, args)
    assert len(calls) == 1 and calls[0][calls[0].index("--out") + 1] == str(bc)
    (bc / "summary.json").write_text("{}")
    calls.clear()
    driver.stage_bc(tmp_path, args)
    assert calls == []
