#!/usr/bin/env python3
"""The whole local spatial experiment, end to end, with no flags to assemble.

`./run.sh` with no arguments lands here. It runs, in order:

  1. clone     behaviour-clone the swept teacher into each arm. This is the
               only thing in the repository that reliably lands, so it is both
               the fine-tuning start and the reference every cell is read
               against.
  2. train     one PPO cell per exploration/objective control, all three arms
               and all seeds, every cell initialised from the SAME clones so
               the only difference is the control under test.
  3. evaluate  every selected checkpoint -- the clones and each cell -- on ONE
               wide held-out seed range (tools/compare_spatial_arms.py), so
               the table has resolution. The in-run validation that selects
               checkpoints uses two seeds and can only read 0, 0.5 or 1.
  4. report    one table, written to the run root as summary.json.

Isaac/PX4 is opt-in behind --isaac, for three reasons, the first decisive:

  * `--stage isaac` refuses an ineligible checkpoint, and a cell that lands
    0 % never produces an eligible one. Flying by default would fail the run
    at its last step every time until a cell actually lands. This is
    self-correcting: --isaac flies whichever cells became eligible and says so
    about the rest.
  * It would START an Isaac Sim + PX4 SITL stack. That is a large external
    side effect for a bare command, though NOT a takeover -- `live_stack`
    already refuses an Isaac that is already running unless --adopt-stack.
  * Isaac is serialized to one worker, so it cannot overlap the training.

Resumable: a stage whose artifacts already exist is skipped, so an interrupted
run continues where it stopped. Pass --fresh to start over.

Single-writer: the run root holds a lock naming the owning pid, because a bare
`run.sh` is easy to start twice -- a 5 s unit test did exactly that once -- and
two writers on one root interleave their artifacts.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import errno
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent
PYTHON = sys.executable or "python3"

#: Each cell is the shipped configuration plus ONE control, so a difference
#: between cells is attributable. `combined` is the only stacked cell and is
#: read last, because a stacked cell cannot attribute anything on its own.
CELLS: tuple[dict, ...] = (
    dict(name="baseline", clone="plain", flags=[]),
    dict(name="masking", clone="plain",
         flags=["--intervention-masking", "per_axis"]),
    dict(name="anneal", clone="plain",
         flags=["--final-log-std", "-3.5", "--log-std-anneal-fraction", "0.5"]),
    # Same endpoint as `anneal`, reached by iteration 5 instead of 100. The
    # clone's sampled batches are success-free until sigma is below ~0.08, and
    # every actor update taken on a success-free batch teaches hovering
    # (TASK_TIMEOUT 12/12 by iteration 61 in results/spatial_finetune_20261006).
    # `anneal` spends its first ~60 actor updates there; this cell spends two.
    # On its own it is NOT enough: with sigma at 0.030 the first (unchecked)
    # actor step of each iteration carries KL 0.5-3.4 and both graph arms
    # collapse to 12/12 SAFE_ABORT within two iterations
    # (results/full_pipeline_20261006_lowsigma_unconstrained).
    dict(name="low_sigma", clone="plain",
         flags=["--final-log-std", "-3.5", "--log-std-anneal-fraction", "0.02"]),
    # The trust region actually enforced, so a small sigma is survivable.
    dict(name="low_sigma_kl", clone="plain",
         flags=["--final-log-std", "-3.5", "--log-std-anneal-fraction", "0.02",
                "--enforce-target-kl"]),
    dict(name="low_sigma_kl_masked", clone="plain",
         flags=["--final-log-std", "-3.5", "--log-std-anneal-fraction", "0.02",
                "--enforce-target-kl", "--intervention-masking", "per_axis"]),
    dict(name="state_sigma", clone="state", flags=["--state-dependent-log-std"]),
    dict(name="combined", clone="state",
         flags=["--state-dependent-log-std", "--intervention-masking", "per_axis",
                "--final-log-std", "-3.5", "--log-std-anneal-fraction", "0.02",
                "--enforce-target-kl"]),
)

ARMS = ("ppo_ontology_rgat", "ppo_semantic_flat", "ppo_vector_canonical")


@contextlib.contextmanager
def owned(root: Path):
    """Refuse to start while another process is writing this run root."""
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "pipeline.lock"
    try:
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError as error:
        if error.errno != errno.EEXIST:
            raise
        owner = lock.read_text().strip()
        pid = int(owner.split()[0]) if owner[:1].isdigit() else -1
        alive = True
        try:
            os.kill(pid, 0)
        except (OSError, ValueError):
            alive = False
        if alive:
            raise SystemExit(
                f"another run owns {root} (pid {pid}). Wait for it, or stop it "
                f"first; two writers on one root interleave artifacts.")
        print(f"clearing a stale lock from pid {pid}")
        lock.unlink()
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(handle, f"{os.getpid()} {dt.datetime.now().isoformat()}\n".encode())
    os.close(handle)
    try:
        yield
    finally:
        with contextlib.suppress(FileNotFoundError):
            lock.unlink()


def run(command: list[str], log: Path, *, cwd: Path = ROOT,
        tolerated: tuple[int, ...] = ()) -> int:
    """Run one stage; a return code outside {0} + `tolerated` ends the run."""
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"  $ {' '.join(str(part) for part in command)}", flush=True)
    print(f"    log -> {log}", flush=True)
    with log.open("w") as handle:
        completed = subprocess.run(command, cwd=cwd, stdout=handle,
                                   stderr=subprocess.STDOUT)
    if completed.returncode != 0 and completed.returncode not in tolerated:
        tail = log.read_text().splitlines()[-25:]
        raise SystemExit("stage failed ({}):\n{}".format(
            completed.returncode, "\n".join(tail)))
    return completed.returncode


def complete(directory: Path, seeds: list[int]) -> bool:
    """A stage counts as done only when every run wrote its summary."""
    return all((directory / "runs" / f"{arm}__seed{seed}" / "summary.json").exists()
               for arm in ARMS for seed in seeds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[828, 829])
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--episodes-per-iteration", type=int, default=12)
    parser.add_argument("--evaluation-every", type=int, default=20,
                        help="validate every N iterations; the shipped spatial "
                             "default of iterations//2 gives two points per run")
    parser.add_argument("--held-out-seeds", type=int, nargs=2, default=[4100, 4148],
                        metavar=("START", "STOP"),
                        help="seed range every selected checkpoint is read on; "
                             "never used for selection or training")
    parser.add_argument("--workers", type=int,
                        default=max(1, min(6, (os.cpu_count() or 4) // 3)))
    parser.add_argument("--cells", nargs="+", default=[cell["name"] for cell in CELLS],
                        help="subset of: " + ", ".join(c["name"] for c in CELLS))
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--fresh", action="store_true",
                        help="ignore existing artifacts and rebuild every stage")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--isaac", action="store_true",
                        help="after training, fly every cell that produced an "
                             "eligible checkpoint (starts an Isaac/PX4 stack)")
    parser.add_argument("--isaac-episodes", type=int, default=1)
    # The /7 five-seed flight died at run 7 of 15 when PX4 refused to arm
    # ("Preflight Fail: High Accelerometer Bias", a degraded SITL EKF that
    # needs a fresh simulator). The stage has an explicit owned-stack restart
    # budget for exactly that (`--reset-recoveries`, recorded per restart in
    # the lifecycle events), but the driver never passed it, so one refusal
    # ended a 30-episode matrix with 18 episodes unflown. Every restart is
    # logged; none collects a transition.
    parser.add_argument("--isaac-reset-recoveries", type=int, default=2,
                        choices=range(6),
                        help="owned-stack restarts the Isaac stage may spend on "
                             "arming refusals before giving up (0 disables)")
    args = parser.parse_args()

    selected = [cell for cell in CELLS if cell["name"] in args.cells]
    unknown = set(args.cells) - {cell["name"] for cell in CELLS}
    if unknown:
        parser.error(f"unknown cells: {sorted(unknown)}")

    stamp = dt.datetime.now().strftime("%Y%m%d")
    # Stages run with cwd=ROOT (the project directory) while the driver's own
    # reads use this path from wherever it was launched; a relative --run-root
    # therefore split one run across two `results/` trees on 2026-10-06
    # (clones under the project's results/, lock and summary under the
    # repository's). Resolve it once, against the launch cwd, before anyone
    # sees it.
    root = (args.run_root.resolve() if args.run_root
            else REPOSITORY / "results" / f"full_pipeline_{stamp}")
    logs = REPOSITORY / "results" / "logs"
    seeds = [str(seed) for seed in args.seeds]
    print(f"run root : {root}")
    print(f"cells    : {', '.join(cell['name'] for cell in selected)}")
    print(f"arms     : {len(ARMS)}   seeds: {' '.join(seeds)}   "
          f"iterations: {args.iterations}   workers: {args.workers}")
    print("Isaac/PX4 : " + ("ON -- will start a stack and fly eligible cells"
                            if args.isaac else
                            "off (pass --isaac to fly the cells that qualify)") + "\n")
    if args.dry_run:
        return 0

    with owned(root):
        return _execute(args, selected, root, logs, seeds)


def _execute(args, selected, root: Path, logs: Path, seeds: list[str]) -> int:
    # ---- 1. clones -------------------------------------------------------
    clones: dict[str, Path] = {}
    for kind in sorted({cell["clone"] for cell in selected}):
        directory = root / f"clone_{kind}"
        clones[kind] = directory
        if not args.fresh and complete(directory, args.seeds):
            print(f"[clone:{kind}] already complete, skipping")
            continue
        print(f"[clone:{kind}] behaviour-cloning the teacher")
        command = [PYTHON, str(ROOT / "tools" / "clone_spatial_teacher.py"),
                   "--output", str(directory), "--seeds", *seeds]
        if kind == "state":
            command.append("--state-dependent-log-std")
        run(command, logs / f"full_clone_{kind}.log")

    # ---- 2. training cells ----------------------------------------------
    for cell in selected:
        directory = root / cell["name"]
        if not args.fresh and complete(directory, args.seeds):
            print(f"[train:{cell['name']}] already complete, skipping")
            continue
        print(f"[train:{cell['name']}] {' '.join(cell['flags']) or '(shipped settings)'}")
        run([PYTHON, str(ROOT / "python" / "run_spatial_pipeline.py"),
             "--stage", "train", "--output", str(directory),
             "--initialize-from", str(clones[cell["clone"]]),
             "--seeds", *seeds,
             "--iterations", str(args.iterations),
             "--episodes-per-iteration", str(args.episodes_per_iteration),
             "--evaluation-every", str(args.evaluation_every),
             "--ppo-preset", "reference-v28-episodic",
             "--activation-iterations", "25", "--activation-decisions", "2048",
             "--workers", str(args.workers), *cell["flags"]],
            logs / f"full_train_{cell['name']}.log")

    # ---- 3. one wide held-out read of every selected checkpoint ----------
    # Clones and cells alike, same seeds, so "BC start" and "after PPO" are
    # the same measurement. `--stage evaluate` and the in-run validation use
    # two seeds each and cannot order anything.
    held_out = root / "held_out.json"
    if args.fresh or not held_out.exists():
        run([PYTHON, str(ROOT / "tools" / "compare_spatial_arms.py"),
             *[str(directory) for directory in clones.values()],
             *[str(root / cell["name"]) for cell in selected],
             "--seeds", *map(str, args.held_out_seeds), "--report", str(held_out)],
            logs / "full_held_out.log")
    read = {}
    for row in json.loads(held_out.read_text())["rows"]:
        read[(row["output"], row["run"])] = row

    # ---- 4. read every cell against its own clone ------------------------
    report = {"root": str(root), "seeds": args.seeds, "iterations": args.iterations,
              "held_out_seeds": args.held_out_seeds, "cells": {}}
    for kind, directory in clones.items():
        report.setdefault("clones", {})[kind] = {
            path.parent.name: read.get((directory.name, path.parent.name), {})
            for path in sorted(directory.glob("runs/*/summary.json"))}
    for cell in selected:
        directory = root / cell["name"]
        rows = {}
        for path in sorted(directory.glob("runs/*/summary.json")):
            summary = json.loads(path.read_text())
            history = summary.get("history") or [{}]
            outcomes: dict[str, int] = {}
            for entry in history:
                for episode in entry.get("episode_outcomes", []):
                    outcomes[episode["status"]] = outcomes.get(episode["status"], 0) + 1
            evaluated = read.get((directory.name, path.parent.name), {})
            health = summary.get("relational_health") or {}
            rows[path.parent.name] = {
                "landing_rate": evaluated.get("landing_rate"),
                "unsafe_rate": evaluated.get("unsafe_rate"),
                "mean_return": evaluated.get("mean_return"),
                "selected_checkpoint": summary.get("selected_checkpoint"),
                "relational_active": health.get("active") if health.get("applicable") else None,
                "training_success": outcomes.get("SUCCESS", 0),
                "training_episodes": sum(outcomes.values()),
            }
        report["cells"][cell["name"]] = {"flags": cell["flags"], "runs": rows}

    # ---- 5. optional actual flight, eligible cells only ------------------
    if args.isaac:
        flown, skipped = {}, {}
        for cell in selected:
            directory = root / cell["name"]
            eligible = [path.parent.name
                        for path in sorted(directory.glob("runs/*/summary.json"))
                        if json.loads(path.read_text()).get("selected_checkpoint")]
            landed = [name for name in eligible
                      if (report["cells"][cell["name"]]["runs"]
                          .get(name, {}).get("landing_rate") or 0) > 0]
            if not landed:
                skipped[cell["name"]] = "no run produced a landing checkpoint"
                continue
            print(f"[isaac:{cell['name']}] flying {len(landed)} run(s)")
            # Exit code 2 is "flew every episode, acceptance failed" (for
            # example one unsafe contact, which acceptance forbids). That is
            # a measured result, not a broken stage, so the run goes on and
            # the verdict is read from isaac_acceptance.json below.
            code = run([PYTHON, str(ROOT / "python" / "run_spatial_pipeline.py"),
                        "--stage", "isaac", "--output", str(directory),
                        "--seeds", *seeds, "--workers", "1",
                        "--isaac-episodes", str(args.isaac_episodes),
                        "--reset-recoveries", str(args.isaac_reset_recoveries)],
                       logs / f"full_isaac_{cell['name']}.log", tolerated=(2,))
            acceptance_path = directory / "isaac_acceptance.json"
            acceptance = (json.loads(acceptance_path.read_text())
                          if acceptance_path.exists() else {})
            evaluation_path = directory / "isaac_evaluation.json"
            flights = (json.loads(evaluation_path.read_text())
                       if evaluation_path.exists() else [])
            flown[cell["name"]] = {
                "runs": landed, "exit_code": code,
                "acceptance_passes": acceptance.get("passes"),
                "acceptance": {key: value for key, value in acceptance.items()
                               if key != "rows"},
                "episodes": [{"mode": row["mode"], "seed": row["seed"],
                              "statuses": [e["status"] for e in row["metrics"]["rows"]],
                              "landing_rate": row["metrics"]["landing_rate"],
                              "unsafe_rate": row["metrics"]["unsafe_rate"]}
                             for row in flights],
            }
        report["isaac"] = {"flown": flown, "skipped": skipped,
                           "episodes_per_run": args.isaac_episodes}
        for name, reason in skipped.items():
            print(f"[isaac:{name}] skipped -- {reason}")
        for name, outcome in flown.items():
            verdict = ("acceptance PASSED" if outcome["acceptance_passes"]
                       else "acceptance FAILED (read isaac_acceptance.json)")
            print(f"[isaac:{name}] {verdict}")
            for episode in outcome["episodes"]:
                print(f"    {episode['mode']:22s} seed{episode['seed']} "
                      f"{' '.join(episode['statuses'])}")

    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    width = args.held_out_seeds[1] - args.held_out_seeds[0]
    print(f"\nheld-out landing on {width} seeds {args.held_out_seeds[0]}-"
          f"{args.held_out_seeds[1] - 1}, deterministic; 'relation' is whether "
          "the proposed arm's relational path was live in the deployed checkpoint")
    print(f"\n{'cell':20s} {'run':30s} {'BC start':>9s} {'after PPO':>10s} "
          f"{'unsafe':>7s} {'return':>8s} {'train SUCCESS':>14s} {'relation':>9s}")
    for cell in selected:
        start = report.get("clones", {}).get(cell["clone"], {})
        for name, row in report["cells"][cell["name"]]["runs"].items():
            began = start.get(name, {}).get("landing_rate")
            landed = row["landing_rate"]
            unsafe = row["unsafe_rate"]
            relation = row["relational_active"]
            print(f"{cell['name']:20s} {name:30s} "
                  f"{'n/a' if began is None else f'{began*100:8.1f}%':>9s} "
                  f"{'n/a' if landed is None else f'{landed*100:9.1f}%':>10s} "
                  f"{'n/a' if unsafe is None else f'{unsafe*100:6.1f}%':>7s} "
                  f"{row['mean_return'] if row['mean_return'] is not None else float('nan'):8.2f} "
                  f"{row['training_success']:6d}/{row['training_episodes']:<7d} "
                  f"{'-' if relation is None else ('live' if relation else 'INERT'):>9s}")
    print(f"\nwrote {root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
