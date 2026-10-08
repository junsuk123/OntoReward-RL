#!/usr/bin/env python3
"""The whole minimal-contract pipeline: teacher -> BC -> PPO -> held-out -> Isaac.

Stages (each skipped when its outputs exist, so the run is resumable):

  teacher   the swept teacher on the 48 held-out seeds (solvability on record)
  bc        tools/minimal_clone_and_evaluate.py: demos, three arms x fit seeds
  ppo       one fine-tuning run per arm x seed, from that seed's clone
            (python/ontology_rgat/minimal/ppo.py), in parallel processes
  evaluate  every BC clone and every PPO best checkpoint on the 48 held-out
            seeds, deterministic AND sampled at the arm's own sigma
  stress    the same checkpoints and the teacher under the evaluation-only
            stress scenarios (minimal/stress.py), deterministic
  isaac     opt-in (--isaac): the PPO checkpoints in Isaac/PX4 through the
            gateway, plus the R-GAT arm flown with the ROS chain as the control
            path; starts an Isaac Sim + PX4 SITL stack via live_stack

    python tools/minimal_full_pipeline.py --run-root results/minimal_pipeline_<date> [--isaac]
    python tools/minimal_full_pipeline.py --run-root ... --dry-run
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
ARMS = ("ppo_ontology_rgat", "ppo_semantic_flat", "ppo_vector_canonical")
HELDOUT = list(range(4100, 4148))
SCRATCH_INSTALL = os.environ.get("MINIMAL_ROS_INSTALL", "")


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _teacher_episode(seed):
    from ontology_rgat.minimal.local_env import MinimalLandingEnv
    from ontology_rgat.minimal.teacher import MinimalTeacher
    env = MinimalLandingEnv()
    obs, _ = env.reset(seed=seed)
    teacher, done, info = MinimalTeacher(), False, {}
    while not done:
        obs, _, done, info = env.step(teacher.act(obs))
    return info["status"]


def stage_teacher(root: Path, workers: int):
    path = root / "teacher_heldout.json"
    if path.exists():
        return json.loads(path.read_text())
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        statuses = list(pool.map(_teacher_episode, HELDOUT))
    result = {"seeds": [HELDOUT[0], HELDOUT[-1]],
              "statuses": {s: statuses.count(s) for s in set(statuses)},
              "landing": statuses.count("SUCCESS") / len(statuses)}
    path.write_text(json.dumps(result, indent=1))
    return result


def stage_bc(root: Path, args):
    """Clone every requested seed; seeds added later are fitted into their own
    directory (same demos: the demo seeds are fixed) and linked into bc/."""
    out = root / "bc"

    def cloned(n, s):
        # A checkpoint counts only once the clone tool finished its directory:
        # DAgger refits overwrite the .pt files round by round, so a run killed
        # mid-DAgger (2026-10-08, round 2) leaves every file present and half
        # of them from the wrong round. summary.json is written last.
        p = out / f"{n}__seed{s}.pt"
        return p.exists() and (p.resolve().parent / "summary.json").exists()

    missing = [s for s in args.seeds if not all(cloned(n, s) for n in ARMS)]
    if not missing:
        return
    complete = (out / "summary.json").exists()
    target = root / f"bc_seeds_{'_'.join(map(str, missing))}" if complete else out
    cmd = [sys.executable, str(ROOT / "tools/minimal_clone_and_evaluate.py"),
           "--sweep", str(args.sweep), "--gains", json.dumps({"kp": 0.4, "kd": 0.8}),
           "--out", str(target), "--workers", str(args.workers),
           "--train-seeds", str(args.bc_episodes), "--fit-seeds", *map(str, missing)]
    if args.train_scenario:
        cmd += ["--demo-scenario", args.train_scenario]
    if args.dagger_rounds:
        cmd += ["--dagger-rounds", str(args.dagger_rounds)]
    with open(root / f"logs/bc_{target.name}.log", "w") as handle:
        subprocess.run(cmd, check=True, stdout=handle, stderr=subprocess.STDOUT)
    if target != out:
        for n in ARMS:
            for s in missing:
                # Relative to the link's own directory, so the run root can move.
                (out / f"{n}__seed{s}.pt").symlink_to(Path("..") / target.name / f"{n}__seed{s}.pt")


def _ppo_run(job):
    name, seed, init, out, config = job
    from ontology_rgat.minimal.ppo import MinimalPPOConfig, train
    return train(name, init, out, MinimalPPOConfig(**config), seed=seed)


def stage_ppo(root: Path, args):
    jobs = []
    for name in ARMS:
        for seed in args.seeds:
            out = root / "ppo" / f"{name}__seed{seed}"
            if (out / "summary.json").exists():
                continue
            config = dict(iterations=args.iterations, workers=args.rollout_workers,
                          episodes_per_iteration=args.episodes_per_iteration,
                          train_scenario=args.train_scenario)
            jobs.append((name, seed, str(root / "bc" / f"{name}__seed{seed}.pt"), str(out), config))
    if jobs:
        log(f"ppo: {len(jobs)} run(s), {args.iterations} iterations x "
            f"{args.episodes_per_iteration} episodes each")
        # Each run spawns its own rollout pool; runs themselves are separate
        # spawned processes so a crash in one cannot take the others down.
        procs = []
        for job in jobs:
            name, seed, init, out, config = job
            Path(out).mkdir(parents=True, exist_ok=True)
            # The main guard is required: the run's rollout pool is spawned,
            # and a spawned child re-imports this file as __mp_main__.
            code = (f"import sys\nsys.path.insert(0, {str(ROOT / 'python')!r})\n"
                    f"from ontology_rgat.minimal.ppo import MinimalPPOConfig, train\n"
                    f"if __name__ == '__main__':\n"
                    f"    train({name!r}, {init!r}, {out!r}, MinimalPPOConfig(**{config!r}), seed={seed})")
            script = Path(out) / "run.py"
            script.write_text(code + "\n")
            procs.append((job, subprocess.Popen(
                [sys.executable, str(script)], stdout=open(Path(out) / "train.log", "w"),
                stderr=subprocess.STDOUT)))
        failed = []
        for job, proc in procs:
            if proc.wait() != 0:
                failed.append(f"{job[0]} seed {job[1]}")
        if failed:
            raise RuntimeError(f"PPO run(s) failed: {failed}; see ppo/*/train.log")
    return {f"{name}__seed{seed}": json.loads((root / "ppo" / f"{name}__seed{seed}" / "summary.json").read_text())
            for name in ARMS for seed in args.seeds}


def _evaluate_one(job):
    import torch
    torch.set_num_threads(1)
    key, path, mode, scenario = job
    from ontology_rgat.minimal.rollout import evaluate, make_env
    if path == "teacher":
        from ontology_rgat.minimal.teacher import MinimalTeacher
        from ontology_rgat.minimal.rollout import UNSAFE
        from collections import Counter
        env, statuses = make_env(scenario), Counter()
        for seed in HELDOUT:
            obs, _ = env.reset(seed=seed)
            teacher, done, info = MinimalTeacher(), False, {}
            while not done:
                obs, _, done, info = env.step(teacher.act(obs))
            statuses[info["status"]] += 1
        n = len(HELDOUT)
        return key, mode, {"episodes": n, "statuses": dict(statuses),
                           "landing": statuses["SUCCESS"] / n,
                           "unsafe": sum(statuses[u] for u in UNSAFE) / n,
                           "abort": statuses["SAFE_ABORT"] / n,
                           "timeout": statuses["TASK_TIMEOUT"] / n}
    from ontology_rgat.minimal.arms import load_arm
    name, arm, _ = load_arm(path)
    arm.eval()
    return key, mode, evaluate(arm, HELDOUT, deterministic=(mode != "sampled"),
                               scenario=scenario)


def _checkpoints(root, args):
    checkpoints = {}
    for name in ARMS:
        for seed in args.seeds:
            checkpoints[f"bc/{name}__seed{seed}"] = root / "bc" / f"{name}__seed{seed}.pt"
            checkpoints[f"ppo/{name}__seed{seed}"] = root / "ppo" / f"{name}__seed{seed}" / "checkpoint_best.pt"
    return checkpoints


def _run_jobs(path: Path, jobs, args):
    """Run only the (key, mode) cells not already in ``path``; resumable."""
    results = json.loads(path.read_text()) if path.exists() else {}
    todo = [j for j in jobs if j[2] not in results.get(j[0], {})]
    if todo:
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for key, mode, result in pool.map(_evaluate_one, todo):
                results.setdefault(key, {})[mode] = result
                path.write_text(json.dumps(results, indent=1))
    return results


def stage_evaluate(root: Path, args):
    jobs = [(k, str(p), mode, None) for k, p in _checkpoints(root, args).items()
            for mode in ("deterministic", "sampled")]
    return _run_jobs(root / "heldout_evaluation.json", jobs, args)


def stage_stress(root: Path, args):
    """Evaluation-only stress scenarios (stress.py), deterministic, teacher too."""
    from ontology_rgat.minimal.stress import SCENARIOS
    entries = {"teacher": "teacher", **{k: str(p) for k, p in _checkpoints(root, args).items()}}
    jobs = [(k, p, scenario, scenario) for k, p in entries.items() for scenario in SCENARIOS]
    return _run_jobs(root / "stress_evaluation.json", jobs, args)


def stage_isaac(root: Path, args, evaluation):
    base = root / ("isaac" if args.isaac_seed_start == 12000 else f"isaac_{args.isaac_seed_start}")
    path = base / "summary.json"
    if path.exists():
        return json.loads(path.read_text())
    if not SCRATCH_INSTALL:
        raise RuntimeError("set MINIMAL_ROS_INSTALL to the colcon install holding "
                           "ontology_rgat_interfaces/_landing")
    # Fly each arm's PPO checkpoint whose held-out run was best (landing, then
    # fewer unsafe) -- chosen on held-out numbers already on record, so the
    # Isaac seeds 12000+ stay untouched by selection.
    chosen = {}
    for name in ARMS:
        candidates = [(evaluation[f"ppo/{name}__seed{s}"]["deterministic"], s) for s in args.seeds]
        best = max(candidates, key=lambda c: (c[0]["landing"], -c[0]["unsafe"]))
        if best[0]["unsafe"] > 0:
            log(f"isaac: {name} seed {best[1]} has held-out unsafe {best[0]['unsafe']:.3f}; flown anyway "
                "(the supervisor is the same for every arm), noted in the summary")
        chosen[name] = str(root / "ppo" / f"{name}__seed{best[1]}" / "checkpoint_best.pt")
    ascii_ws = Path.home() / ".local/share/ontology_rgat_uav_rl/ros2_ws"
    env_prefix = (f"set +u; source /opt/ros/humble/setup.bash; "
                  f"source '{ascii_ws}/install/local_setup.bash'; source '{SCRATCH_INSTALL}/local_setup.bash'; "
                  "export RMW_IMPLEMENTATION=rmw_fastrtps_cpp; unset CYCLONEDDS_URI; "
                  f"export ONTOLOGY_RGAT_ROOT='{ROOT}'; ")
    flights = {}
    for tag, extra in (
            ("inprocess", ["--controllers", *chosen.values(), "--episodes", str(args.isaac_episodes)]),
            ("ros_control", ["--control", "ros", "--episodes", str(args.isaac_episodes)])):
        out = base / tag
        if not (out / "summary.json").exists():
            cmd = ([sys.executable, str(ROOT / "tools/minimal_isaac_flight.py"), "--out", str(out),
                    "--ros-chain", "--ros-install", SCRATCH_INSTALL,
                    "--chain-checkpoint", chosen["ppo_ontology_rgat"],
                    "--seed-start", str(args.isaac_seed_start)] + extra)
            log(f"isaac: {tag}")
            with open(root / f"logs/{base.name}_{tag}.log", "w") as handle:
                code = subprocess.run(["bash", "-c", env_prefix + " ".join(f"'{c}'" for c in cmd)],
                                      stdout=handle, stderr=subprocess.STDOUT).returncode
            if code != 0:
                raise RuntimeError(f"isaac {tag} exited {code}; see logs/{base.name}_{tag}.log")
        flights[tag] = json.loads((out / "summary.json").read_text())
    summary = {"chosen": chosen, "flights": flights}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=1))
    return summary


def table(evaluation, args):
    lines = [f"{'checkpoint':34s}  det land/unsafe/abort/timeout   sampled land/unsafe"]
    for name in ARMS:
        for stage in ("bc", "ppo"):
            for seed in args.seeds:
                row = evaluation[f"{stage}/{name}__seed{seed}"]
                d, s = row["deterministic"], row["sampled"]
                lines.append(f"{stage + '/' + name + '__' + str(seed):34s}  {d['landing']:.3f}/{d['unsafe']:.3f}/"
                             f"{d['abort']:.3f}/{d['timeout']:.3f}   {s['landing']:.3f}/{s['unsafe']:.3f}")
    return "\n".join(lines)


def stress_table(stress, args):
    from ontology_rgat.minimal.stress import SCENARIOS
    names = list(SCENARIOS)
    lines = [f"{'checkpoint':34s}  " + "  ".join(f"{n:>13s}" for n in names) + "   (landing/unsafe)"]
    keys = ["teacher"] + [f"{st}/{a}__seed{s}" for a in ARMS for st in ("bc", "ppo") for s in args.seeds]
    for key in keys:
        row = stress.get(key, {})
        lines.append(f"{key:34s}  " + "  ".join(
            f"{row[n]['landing']:.2f}/{row[n]['unsafe']:.2f}".rjust(13) if n in row else " " * 13
            for n in names))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--sweep", type=Path,
                        default=ROOT / "results/minimal_contract_20261007/teacher_sweep_r3.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[828, 829, 830])
    parser.add_argument("--bc-episodes", type=int, default=160)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--episodes-per-iteration", type=int, default=12)
    parser.add_argument("--rollout-workers", type=int, default=2)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--isaac", action="store_true")
    parser.add_argument("--isaac-episodes", type=int, default=2)
    parser.add_argument("--isaac-seed-start", type=int, default=12000)
    parser.add_argument("--skip-stress", action="store_true")
    parser.add_argument("--dagger-rounds", type=int, default=0,
                        help="DAgger rounds after the first BC fit (teacher-labelled clone states)")
    parser.add_argument("--train-scenario", default=None, choices=[None, "dr"],
                        help="dr: BC demos and PPO rollouts under per-episode randomized "
                             "stress; validation, selection and held-out stay nominal")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.run_root
    plan = {"teacher": "48 held-out seeds", "bc": f"{args.bc_episodes} demos, fit seeds {args.seeds}",
            "ppo": f"{len(ARMS) * len(args.seeds)} runs x {args.iterations} it x {args.episodes_per_iteration} ep",
            "train_scenario": args.train_scenario or "nominal",
            "dagger_rounds": args.dagger_rounds,
            "evaluate": "every clone and PPO best on 48 held-out seeds, det + sampled",
            "isaac": f"{'yes' if args.isaac else 'no (pass --isaac)'}, {args.isaac_episodes} episodes per controller"}
    if args.dry_run:
        print(json.dumps(plan, indent=1))
        return 0
    (root / "logs").mkdir(parents=True, exist_ok=True)
    lock = open(root / "pipeline.lock", "a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another pipeline owns {root}")
    (root / "plan.json").write_text(json.dumps({**plan, "argv": sys.argv}, indent=1))
    log("teacher"); teacher = stage_teacher(root, args.workers); log(f"teacher held-out {teacher['statuses']}")
    log("bc"); stage_bc(root, args); log("bc done")
    log("ppo"); ppo = stage_ppo(root, args); log("ppo done")
    log("evaluate"); evaluation = stage_evaluate(root, args)
    print(table(evaluation, args), flush=True)
    stress = None
    if not args.skip_stress:
        log("stress"); stress = stage_stress(root, args); print(stress_table(stress, args), flush=True)
    isaac = stage_isaac(root, args, evaluation) if args.isaac else None
    summary = {"teacher": teacher, "ppo": {k: {"best_iteration": v["best_iteration"],
                                               "best_validation": v["best_validation"]["landing"],
                                               "train_success_last10": v["train_success_last10"]}
                                           for k, v in ppo.items()},
               "heldout": evaluation, "stress": stress, "isaac": isaac}
    (root / "summary.json").write_text(json.dumps(summary, indent=1))
    log("pipeline complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
