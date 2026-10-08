#!/usr/bin/env python3
"""Behaviour-clone the minimal-contract teacher into all three arms, then evaluate.

1. Fly the teacher (gains from a sweep summary) on TRAINING seeds and record,
   at every decision, every arm's input (built by the same ArmController the
   evaluation uses) and the teacher's requested acceleration.
2. Fit each arm in ACTION space (MSE on m/s^2 / authority), per training seed.
3. Evaluate every clone on HELD-OUT seeds, deterministically AND sampled at
   the arm's own sigma (AGENTS.md: a policy that lands deterministically can
   land 0 % sampled, and PPO rollouts are sampled).

    python tools/minimal_clone_and_evaluate.py --sweep results/.../teacher_sweep.json \
        --out results/minimal_contract_20261007/bc
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def collect(args):
    gains, seeds, *rest = args
    scenario = rest[0] if rest else None
    from ontology_rgat.minimal.arms import FlatGraphArm, VectorArm
    from ontology_rgat.minimal.rollout import ArmController, make_env
    from ontology_rgat.minimal.teacher import MinimalTeacher, TeacherGains
    env = make_env(scenario)
    rows = {"vector": [], "graph": [], "action": []}
    statuses = []
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        teacher, ctl = MinimalTeacher(TeacherGains(**gains)), ArmController()
        done, info = False, {}
        while not done:
            stacked, graph = ctl.inputs(obs)
            action = teacher.act(obs)
            rows["vector"].append(VectorArm.inputs(stacked, graph))
            rows["graph"].append(FlatGraphArm.inputs(stacked, graph))
            rows["action"].append(np.asarray(action, np.float32))
            obs, _, done, info = env.step(action)
        statuses.append(info["status"])
    return {k: np.asarray(v, np.float32) for k, v in rows.items()}, statuses


def dagger_collect(args):
    """Fly a clone (teacher with probability beta per step) and label every
    visited state with a SHADOW teacher's action on the same observations.

    The teacher's whole state -- pad memory and TrackingBias -- is the
    ontology's own, so the label is a function of what the clone sees; this is
    what makes DAgger valid here, where it collapsed on the spatial route
    (AGENTS.md: that teacher's integral was not in the input).
    """
    path, seeds, scenario, beta, gains, sample_seed = args
    import torch
    torch.set_num_threads(1)
    from ontology_rgat.minimal.arms import FlatGraphArm, VectorArm, load_arm
    from ontology_rgat.minimal.rollout import ArmController, make_env
    from ontology_rgat.minimal.teacher import MinimalTeacher, TeacherGains
    _, arm, _ = load_arm(path)
    arm.eval()
    rng = np.random.default_rng(sample_seed)
    env = make_env(scenario)
    rows = {"vector": [], "graph": [], "action": []}
    statuses = []
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        teacher, ctl = MinimalTeacher(TeacherGains(**gains)), ArmController(arm)
        done, info = False, {}
        while not done:
            stacked, graph = ctl.inputs(obs)
            label = teacher.act(obs)
            with torch.no_grad():
                own = arm(ctl.last_input)[0].numpy().astype(float)
            rows["vector"].append(VectorArm.inputs(stacked, graph))
            rows["graph"].append(FlatGraphArm.inputs(stacked, graph))
            rows["action"].append(np.asarray(label, np.float32))
            obs, _, done, info = env.step(label if rng.uniform() < beta else own)
        statuses.append(info["status"])
    return path, {k: np.asarray(v, np.float32) for k, v in rows.items()}, statuses


def fit(name, seed, data, epochs, out_dir):
    import torch
    from ontology_rgat.minimal.arms import build_arm, save_arm
    torch.manual_seed(seed)
    arm = build_arm(name, seed=seed)
    x = torch.as_tensor(data["vector"] if name == "ppo_vector_canonical" else data["graph"])
    scale = torch.as_tensor(arm.action_scale if hasattr(arm, "action_scale")
                            else arm.policy.action_scale)
    y = torch.as_tensor(data["action"]).clamp(-scale, scale)
    params = [p for n, p in arm.named_parameters() if "log_std" not in n]
    opt = torch.optim.Adam(params, lr=1e-3)
    gen = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(epochs):
        order = torch.randperm(len(x), generator=gen)
        total = 0.0
        for start in range(0, len(x), 256):
            idx = order[start:start + 256]
            mean, _, _ = arm(x[idx])
            loss = (((mean - y[idx]) / scale) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        losses.append(total / len(x))
    path = out_dir / f"{name}__seed{seed}.pt"
    save_arm(arm, name, path, bc_loss=losses[-1], samples=len(x))
    return str(path), losses


def evaluate_one(args):
    path, seeds, deterministic, sigma = args
    import torch
    torch.set_num_threads(1)
    from ontology_rgat.minimal.arms import load_arm
    from ontology_rgat.minimal.rollout import evaluate
    name, arm, payload = load_arm(path)
    arm.eval()
    return path, deterministic, sigma, evaluate(arm, seeds, deterministic=deterministic,
                                                sigma=sigma)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--gains", type=json.loads, default=None,
                        help="teacher gains overriding the sweep's best cell (e.g. an interior plateau cell)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-seeds", type=int, default=160)
    parser.add_argument("--train-seed-base", type=int, default=1000)
    parser.add_argument("--heldout-seeds", type=int, default=48)
    parser.add_argument("--heldout-seed-base", type=int, default=4100)
    parser.add_argument("--fit-seeds", type=int, nargs="+", default=[828, 829])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--sigmas", type=float, nargs="*", default=[],
                        help="extra fixed exploration sigmas to evaluate sampled at")
    parser.add_argument("--demo-scenario", default=None,
                        help="teacher demos under this scenario ('dr' = randomized stress); "
                             "held-out evaluation stays nominal")
    parser.add_argument("--dagger-rounds", type=int, default=0)
    parser.add_argument("--dagger-episodes", type=int, default=64,
                        help="clone-driven episodes per checkpoint per round")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="reuse the checkpoints already in --out")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    gains = args.gains or json.loads(args.sweep.read_text())["best"]["gains"]
    started = time.time()

    train = list(range(args.train_seed_base, args.train_seed_base + args.train_seeds))
    chunks = [train[i::args.workers] for i in range(args.workers)]
    with ProcessPoolExecutor(args.workers) as pool:
        parts = list(pool.map(collect, [(gains, c, args.demo_scenario) for c in chunks if c]))
    data = {k: np.concatenate([p[0][k] for p in parts]) for k in parts[0][0]}
    teacher_statuses = [s for p in parts for s in p[1]]
    print(f"demos: {len(data['action'])} decisions from {len(train)} episodes, "
          f"teacher landed {teacher_statuses.count('SUCCESS')}/{len(train)}", flush=True)

    from ontology_rgat.minimal.arms import ARMS
    checkpoints = {}
    for name in ARMS:
        for seed in args.fit_seeds:
            if args.evaluate_only:
                import torch
                path = str(args.out / f"{name}__seed{seed}.pt")
                loss = torch.load(path, weights_only=False)["bc_loss"]
                checkpoints[path] = {"arm": name, "seed": seed, "bc_loss": loss}
                continue
            path, losses = fit(name, seed, data, args.epochs, args.out)
            checkpoints[path] = {"arm": name, "seed": seed, "bc_loss": losses[-1]}
            print(f"{name} seed {seed}: bc loss {losses[0]:.4f} -> {losses[-1]:.4f}", flush=True)

    dagger_log = []
    if args.dagger_rounds and not args.evaluate_only:
        extra = {path: [] for path in checkpoints}
        for r in range(args.dagger_rounds):
            beta = 0.5 ** (r + 1)
            base = 20_000 + 1_000 * r
            jobs = [(path, list(range(base + 97 * k, base + 97 * k + args.dagger_episodes)),
                     args.demo_scenario, beta, gains, 7 * r + k)
                    for k, path in enumerate(checkpoints)]
            with ProcessPoolExecutor(args.workers,
                                     mp_context=multiprocessing.get_context("spawn")) as pool:
                for path, rows, statuses in pool.map(dagger_collect, jobs):
                    extra[path].append(rows)
                    dagger_log.append({"round": r + 1, "beta": beta, "checkpoint": Path(path).name,
                                       "decisions": int(len(rows["action"])),
                                       "landing": statuses.count("SUCCESS") / len(statuses)})
            for path, row in checkpoints.items():
                merged = {k: np.concatenate([data[k]] + [e[k] for e in extra[path]]) for k in data}
                _, losses = fit(row["arm"], row["seed"], merged, args.epochs, args.out)
                row["bc_loss"] = losses[-1]
                row["dagger_decisions"] = int(len(merged["action"]) - len(data["action"]))
            print(f"dagger round {r + 1} (beta {beta}): " + ", ".join(
                f"{d['checkpoint'].split('__')[0][4:]}:{d['landing']:.2f}"
                for d in dagger_log if d["round"] == r + 1), flush=True)

    heldout = list(range(args.heldout_seed_base, args.heldout_seed_base + args.heldout_seeds))
    jobs = ([(p, heldout, det, None) for p in checkpoints for det in (True, False)]
            + [(p, heldout, False, sg) for p in checkpoints for sg in args.sigmas])
    # Spawned, not forked: the parent has run torch, and a forked child can
    # inherit a held thread-pool lock and hang (measured: idle for an hour).
    with ProcessPoolExecutor(args.workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        for path, det, sigma, result in pool.map(evaluate_one, jobs):
            key = ("deterministic" if det else "sampled") if sigma is None else f"sigma_{sigma:g}"
            checkpoints[path][key] = result

    heldout_teacher = collect((gains, heldout, None))[1]
    summary = {
        "teacher_heldout_statuses": {s: heldout_teacher.count(s) for s in set(heldout_teacher)},
        "contract": "minimal-landing-obs/2 + minimal-landing-ontology/2, local SpatialConfig() plant",
        "teacher_gains": gains,
        "teacher_train_landing": teacher_statuses.count("SUCCESS") / len(train),
        "train_seeds": [train[0], train[-1]], "heldout_seeds": [heldout[0], heldout[-1]],
        "decisions": int(len(data["action"])), "epochs": args.epochs,
        "checkpoints": checkpoints, "elapsed_s": time.time() - started,
        "dagger": dagger_log,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\n{'arm':24s} seed  det land/unsafe/abort/timeout   sampled land/unsafe   rel.act")
    for path, row in checkpoints.items():
        d, s = row["deterministic"], row["sampled"]
        act = d["relational_activity"]
        print(f"{row['arm']:24s} {row['seed']}  {d['landing']:.3f}/{d['unsafe']:.3f}/"
              f"{d['abort']:.3f}/{d['timeout']:.3f}   {s['landing']:.3f}/{s['unsafe']:.3f}"
              f"   {'' if act is None else f'{act:.3f}'}   "
              + " ".join(f"s{sg:g}:{row[f'sigma_{sg:g}']['landing']:.2f}/{row[f'sigma_{sg:g}']['unsafe']:.2f}"
                         for sg in args.sigmas))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
