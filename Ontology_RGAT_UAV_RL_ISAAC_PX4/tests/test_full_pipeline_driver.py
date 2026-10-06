"""`./run.sh` with no arguments runs the whole local pipeline. Two hazards.

Once the bare route does real work, anything that merely executes `run.sh` to
look at it starts a multi-hour job: a 5 s unit test did exactly that, and the
behaviour-cloning child outlived the SIGKILL and ran to completion in the
background. And this project has lost runs before to a second process
relaunching `run.sh` on a root the first one was still writing.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "tools" / "run_full_pipeline.py"


def drive(*arguments, timeout=120):
    return subprocess.run([sys.executable, str(DRIVER), *arguments],
                          capture_output=True, text=True, timeout=timeout)


def test_dry_run_plans_without_touching_anything(tmp_path):
    root = tmp_path / "run"
    result = drive("--run-root", str(root), "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Isaac/PX4 : off" in result.stdout
    assert not root.exists(), "a dry run must not create the run root"


def test_isaac_is_off_by_default_and_named_when_on(tmp_path):
    """Flying by default would fail the run at its last step, every time.

    `--stage isaac` raises on an ineligible checkpoint and a cell that lands
    0 % never produces an eligible one, so Isaac is opt-in until a cell
    actually lands. It is NOT excluded because of takeover: `live_stack`
    already refuses an Isaac that is running unless --adopt-stack.
    """
    off = drive("--run-root", str(tmp_path / "a"), "--dry-run")
    assert "pass --isaac" in off.stdout
    on = drive("--run-root", str(tmp_path / "b"), "--dry-run", "--isaac")
    assert "will start a stack" in on.stdout


def test_cells_change_exactly_one_thing_each():
    """A cell that moves two controls cannot attribute its own result."""
    from tools.run_full_pipeline import CELLS

    names = [cell["name"] for cell in CELLS]
    assert names[0] == "baseline" and not CELLS[0]["flags"]
    assert names[-1] == "combined", "the stacked cell is read last"
    controls = set()
    for cell in CELLS[1:-1]:
        flags = {flag for flag in cell["flags"] if flag.startswith("--")}
        assert flags, cell
        controls |= flags
    # Every single-control cell is a subset of the stacked one, so the stack
    # is the union of what was measured separately and nothing else.
    assert controls <= {flag for flag in CELLS[-1]["flags"] if flag.startswith("--")}


def test_an_unknown_cell_is_refused_rather_than_silently_dropped(tmp_path):
    result = drive("--run-root", str(tmp_path / "run"), "--cells", "nonsense")
    assert result.returncode != 0
    assert "unknown cells" in (result.stdout + result.stderr)


def test_a_live_lock_refuses_a_second_writer(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "pipeline.lock").write_text(f"{os.getpid()} held\n")
    result = drive("--run-root", str(root), "--cells", "baseline")
    assert result.returncode != 0
    assert "another run owns" in (result.stdout + result.stderr)


def test_a_stale_lock_is_cleared_rather_than_blocking_forever(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    # A pid that cannot be running: the owner died without unlinking.
    (root / "pipeline.lock").write_text("999999 stale\n")
    result = drive("--run-root", str(root), "--cells", "baseline", "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_stage_counts_as_done_only_when_every_run_wrote_a_summary(tmp_path):
    from tools.run_full_pipeline import ARMS, complete

    seeds = [828, 829]
    root = tmp_path / "cell"
    assert not complete(root, seeds)
    for arm in ARMS:
        for seed in seeds:
            directory = root / "runs" / f"{arm}__seed{seed}"
            directory.mkdir(parents=True)
            (directory / "summary.json").write_text("{}")
    assert complete(root, seeds)
    # One missing summary must not read as a finished stage, or resume skips
    # a cell that never produced a result.
    (root / "runs" / f"{ARMS[0]}__seed828" / "summary.json").unlink()
    assert not complete(root, seeds)
