#!/usr/bin/env python3
"""Behaviour-clone the swept causal controller into each arm. NOT a PPO result.

What this is for
----------------
Actual Isaac/PX4 evaluation refuses to fly without a checkpoint that is
``eligible`` and that passes a nominal local preflight with zero unsafe
outcomes (``run_spatial_pipeline.py``). End-to-end PPO has never produced one:
measured 2026-10-05 over nine controlled cells, every arm lands 0 % and the best
cell still carries unsafe 0.000-0.292. So no learned policy has ever flown in
Isaac -- all 40 recorded Isaac terminals are ``learned_policy: false``.

Supervised imitation does produce landing policies: 38-71 % on held-out seeds
for all three arms across two seeds, against 0 % for PPO. This tool writes those
clones as ordinary checkpoints so the Isaac path has something admissible to
fly, which turns "can a learned policy fly the real stack at all" into a
question that can be answered separately from "does PPO learn".

What this is NOT
----------------
* NOT a PPO result, and never to be reported as one. Every artifact it writes
  carries ``learned_by: behaviour_cloning``, ``ppo_success_claim: false`` and
  ``publication_claim_allowed: false``.
* NOT a teacher for PPO. Nothing here writes a demonstration cache or seeds a
  policy that is later trained. It is an endpoint.
* NOT evidence about the proposed representation. The three arms are cloned
  from the same teacher with the same budget, so differences between them
  measure fitting capacity, not what the ontology buys under reinforcement.

Method notes, both learned the hard way
---------------------------------------
* The loss is in ACTION space, ``mean((tanh(mu) - a_teacher)**2)``. Regressing
  ``mu`` onto ``atanh(clip(a, -0.999, 0.999))`` instead makes the objective
  track the clipping constant, because the teacher saturates often and atanh
  sends those targets to +/-3.8.
* No DAgger. Aggregating relabelled learner states collapsed every arm to ~0
  landings with the training error rising 5-10x, because the teacher carries an
  integral term and its label on the learner's own trajectory depends on a
  history the teacher would never have produced.
"""
import argparse
import collections
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from ontology_rgat.spatial.core import SpatialConfig, schema_for
from ontology_rgat.spatial.runtime_contract import deployment_profile
from ontology_rgat.spatial.environment import SpatialLandingEnv
from ontology_rgat.spatial.training import (
    SpatialAgent, VALIDATION_SEEDS, evaluate, save_agent)
from ontology_rgat.two_axis.artifacts import json_text
from ontology_rgat.two_axis.models import POLICY_MODES, observation_arrays
from ontology_rgat.two_axis.models_v28 import relational_contribution

#: Swept gains PER RUNG, reading the estimator only -- no simulator truth.
#:
#: /2 (and every instant-response rung): the 96-cell sweep's plateau, 95.8 %
#: with zero unsafe or unauthorized contacts.
#:
#: /3 carries Isaac's 0.10 s actuation latency, and the /2 set lands 0 of 24
#: on it (7 unsafe, 15 timeouts): a derivative gain of 0.8 against a delayed
#: plant oscillates, exactly as the policies tuned on /2 did in Isaac. Swept
#: again on /3 (2026-10-06, 144 + 192 cells x 12 seeds, then the plateau
#: re-scored on 48 fresh seeds 7001-7048): kd above 0.6 lands nothing, ki 0
#: lands nothing, and the best interior cell is kp 0.4 / kd 0.3 / ki 0.1 at
#: 85.4 % with zero unsafe contacts (41 SUCCESS, 5 TASK_TIMEOUT, 2 SAFE_ABORT).
TEACHER_GAINS_BY_SCHEMA = {
    "spatial-reference/3": dict(kp=0.4, kd=0.3, ki=0.1, kz=1.0, vdes=0.15,
                                align=0.15, anorm=0.3),
    # /4 shares the latency and adds a faster servo plus 0.18 m of gear; the
    # /3 set re-scored on 48 seeds there is recorded in the plan.json of the
    # run that used it.
    "spatial-reference/4": dict(kp=0.4, kd=0.3, ki=0.1, kz=1.0, vdes=0.15,
                                align=0.15, anorm=0.3),
    "spatial-reference/5": dict(kp=0.4, kd=0.3, ki=0.1, kz=1.0, vdes=0.15,
                                align=0.15, anorm=0.3),
    "spatial-reference/6": dict(kp=0.4, kd=0.3, ki=0.1, kz=1.0, vdes=0.15,
                                align=0.15, anorm=0.3),
    # /7 changes the contact verdict only; the plant is /6's, so are the gains.
    "spatial-reference/7": dict(kp=0.4, kd=0.3, ki=0.1, kz=1.0, vdes=0.15,
                                align=0.15, anorm=0.3),
    # /8 removes the legs; the /3 plant had none and landed on this same set.
    "spatial-reference/8": dict(kp=0.4, kd=0.3, ki=0.1, kz=1.0, vdes=0.15,
                                align=0.15, anorm=0.3),
    # /9 changes the corridor brake only; the teacher sinks at 0.15 m/s,
    # under the 0.24 m/s the brake now holds, so it never meets it.
    "spatial-reference/9": dict(kp=0.4, kd=0.3, ki=0.1, kz=1.0, vdes=0.15,
                                align=0.15, anorm=0.3),
    # /10 widens the corridor gate near the pad; the plant is /9's.
    "spatial-reference/10": dict(kp=0.4, kd=0.3, ki=0.1, kz=1.0, vdes=0.15,
                                 align=0.15, anorm=0.3),
    # /11 limits the post-handover lateral command; the teacher's anorm 0.3
    # (0.75 m/s^2) never reaches the 2.0 cap.
    "spatial-reference/11": dict(kp=0.4, kd=0.3, ki=0.1, kz=1.0, vdes=0.15,
                                 align=0.15, anorm=0.3),
}
TEACHER_GAINS = dict(kp=0.4, kd=0.8, ki=0.4, kz=1.0, vdes=0.15, align=0.15,
                     anorm=0.3)


def teacher_gains(cfg):
    """The swept set for this rung; the /2 plateau for every other schema."""
    return TEACHER_GAINS_BY_SCHEMA.get(cfg.schema, TEACHER_GAINS)


def teacher_action(env, cfg, integral):
    gains, estimate = teacher_gains(cfg), env.estimator
    r, rv = estimate.r.copy(), estimate.rv.copy()
    integral[:] = np.clip(integral + cfg.dt * r[:2], -3.0, 3.0)
    desired = np.asarray(estimate.pad_v, dtype=float).copy()
    desired[:2] += (-gains["kp"] * r[:2] - gains["kd"] * rv[:2]
                    - gains["ki"] * integral)
    aligned = (np.linalg.norm(r[:2]) < gains["align"]
               and np.linalg.norm(rv[:2]) < 0.12)
    desired[2] = -gains["vdes"] if aligned else gains["kz"] * (1.0 - r[2])
    limits = gains["anorm"] * np.asarray(cfg.max_acceleration)
    return np.clip((desired - env.backend.velocity) / limits, -1.0, 1.0)


def demonstrate(cfg, seeds, difficulties):
    env = SpatialLandingEnv(cfg)
    observations, actions, outcomes = [], [], collections.Counter()
    try:
        for seed in seeds:
            for difficulty in difficulties:
                observation, _ = env.reset(seed=seed, difficulty=difficulty)
                integral = np.zeros(2)
                for _ in range(int(cfg.horizon / cfg.dt) + 2):
                    action = teacher_action(env, cfg, integral)
                    observations.append(observation)
                    actions.append(action)
                    observation, _r, done, _t, info = env.step(action)
                    if done:
                        outcomes[info["status"]] += 1
                        break
    finally:
        env.close()
    return observations, np.asarray(actions, dtype=np.float32), outcomes


def clone(mode, cfg, observations, actions, *, seed, epochs, batch, lr,
          state_dependent_log_std=False):
    agent = SpatialAgent(mode, cfg, seed, -1.1,
                         state_dependent_log_std=state_dependent_log_std)
    packets, graphs = observation_arrays(observations)
    packets, graphs = torch.as_tensor(packets), torch.as_tensor(graphs)
    target = torch.as_tensor(actions)
    optimizer = torch.optim.Adam(agent.actor.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    history = []
    for _epoch in range(epochs):
        order = torch.randperm(len(target), generator=generator)
        total = 0.0
        for start in range(0, len(order), batch):
            index = order[start:start + batch]
            mu, _std = agent.actor(packets[index], graphs[index])
            loss = ((torch.tanh(mu) - target[index]) ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss) * len(index)
        history.append(total / len(order))
    return agent, history


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[828, 829])
    parser.add_argument("--demonstration-seeds", type=int, nargs=2,
                        default=[6001, 6041], metavar=("START", "STOP"))
    parser.add_argument("--difficulties", type=float, nargs="+",
                        default=[0.0, 0.5, 1.0])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--holdout-seeds", type=int, nargs=2,
                        default=[3001, 3025], metavar=("START", "STOP"))
    parser.add_argument("--state-dependent-log-std", action="store_true",
                        help="clone into the state-dependent-sigma actor")
    parser.add_argument("--contract-version",
                        choices=["5", "6", "7", "8", "9", "10",
                                 "spatial-reference/1", "spatial-reference/2",
                                 "spatial-reference/3", "spatial-reference/4",
                                 "spatial-reference/5", "spatial-reference/6",
                                 "spatial-reference/7", "spatial-reference/8",
                                 "spatial-reference/9", "spatial-reference/10", "spatial-reference/11", "reference"],
                        default="reference")
    args = parser.parse_args()
    if any(not 0.0 <= d <= 1.0 for d in args.difficulties):
        parser.error("difficulties must be in [0,1]")
    if args.epochs < 1 or args.batch < 2:
        parser.error("positive epochs and a batch of at least two are required")

    # Same deployment profile the pipeline stamps into its config, or the
    # clone's config_sha256 differs and `--initialize-from` refuses the
    # checkpoint as "not compatible with the exact spatial contract".
    schema = schema_for(args.contract_version)
    cfg = replace(SpatialConfig(), schema=schema,
                  isaac_profile_sha256=deployment_profile(schema)["sha256"])
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    demonstration_seeds = list(range(*args.demonstration_seeds))
    holdout = tuple(range(*args.holdout_seeds))

    observations, actions, teacher_outcomes = demonstrate(
        cfg, demonstration_seeds, args.difficulties)
    print(f"[clone] teacher: {len(observations)} steps from "
          f"{len(demonstration_seeds) * len(args.difficulties)} episodes, "
          f"{dict(teacher_outcomes)}", flush=True)

    plan = {
        "tool": "clone_spatial_teacher/1",
        "signature": cfg.signature,
        "learned_by": "behaviour_cloning",
        "ppo_success_claim": False,
        "publication_claim_allowed": False,
        "teacher": {"controller": "swept-causal-pd-with-bounded-integral",
                    "gains": teacher_gains(cfg),
                    "reads": "estimator only, never simulator truth",
                    "outcomes": dict(teacher_outcomes)},
        "demonstration_seeds": demonstration_seeds,
        "difficulties": list(args.difficulties),
        "holdout_seeds": list(holdout),
        "objective": "mean((tanh(mu) - a_teacher)**2), actor only",
        "note": "cloned checkpoints are admissible to the Isaac stage; they are "
                "NOT a PPO result and carry no claim about the representation.",
    }
    (output / "plan.json").write_text(json_text(plan, indent=2) + "\n")

    rows = []
    for seed in args.seeds:
        for mode in POLICY_MODES:
            agent, history = clone(mode, cfg, observations, actions, seed=seed,
                                   epochs=args.epochs, batch=args.batch, lr=args.lr,
                                   state_dependent_log_std=args.state_dependent_log_std)
            held = evaluate(agent, cfg, seeds=holdout)
            preflight = evaluate(agent, cfg, seeds=VALIDATION_SEEDS)
            # Did the relational path move off zero at all? Its readout is
            # zero-initialised, so an arm whose readout norm stays ~0 is
            # numerically the flat baseline however it is labelled.
            sample = slice(0, min(512, len(observations)))
            probe_packets, probe_graphs = observation_arrays(observations[sample])
            relational = relational_contribution(
                agent, torch.as_tensor(probe_packets), torch.as_tensor(probe_graphs))
            admissible = preflight["unsafe_rate"] == 0
            directory = output / "runs" / f"{mode}__seed{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            save_agent(directory / "checkpoint_cloned.pt", agent, cfg,
                       eligible=bool(admissible),
                       completed_episodes=0,
                       training_backend="behaviour-cloning",
                       learned_by="behaviour_cloning",
                       ppo_success_claim=False,
                       validation=held)
            summary = {
                "mode": mode, "seed": seed, "signature": cfg.signature,
                "learned_by": "behaviour_cloning",
                "ppo_success_claim": False,
                "publication_claim_allowed": False,
                "selected_checkpoint": "checkpoint_cloned.pt" if admissible else None,
                "isaac_preflight": {
                    "seeds": list(VALIDATION_SEEDS),
                    "unsafe_rate": preflight["unsafe_rate"],
                    "statuses": [r["status"] for r in preflight["rows"]],
                    "admissible": bool(admissible),
                },
                "holdout": {key: held[key] for key in (
                    "episodes", "landing_rate", "unsafe_rate", "safe_abort_rate",
                    "task_timeout_rate", "mean_return")},
                "relational_contribution": relational,
                "first_epoch_action_mse": history[0],
                "last_epoch_action_mse": history[-1],
                "demonstration_steps": len(observations),
                "hyperparameters": {"epochs": args.epochs, "batch": args.batch,
                                    "lr": args.lr,
                                    "state_dependent_log_std":
                                        bool(args.state_dependent_log_std)},
                "completed_nominal_episodes": 0,
                "environment_steps": 0,
            }
            (directory / "summary.json").write_text(
                json_text(summary, indent=2) + "\n")
            rows.append(summary)
            print(f"[clone] {mode} seed={seed} mse {history[0]:.5f}->{history[-1]:.5f} "
                  f"holdout land={held['landing_rate']:.3f} unsafe={held['unsafe_rate']:.3f} "
                  f"| isaac preflight unsafe={preflight['unsafe_rate']:.3f} "
                  f"{'ADMISSIBLE' if admissible else 'REFUSED'}"
                  + (f" | actor readout {relational['actor_readout_norm']:.4f}"
                     f" delta|max| {relational['actor_delta_max_abs']:.4f}"
                     if relational.get("applicable") else ""), flush=True)
            (output / "clone_report.json").write_text(
                json_text(rows, indent=2) + "\n")

    admitted = [r for r in rows if r["isaac_preflight"]["admissible"]]
    print(f"\n[clone] {len(admitted)}/{len(rows)} checkpoints pass the Isaac preflight")
    for row in rows:
        mark = "ok " if row["isaac_preflight"]["admissible"] else "NO "
        print(f"   {mark}{row['mode']:22s} seed{row['seed']} "
              f"{row['isaac_preflight']['statuses']}")


if __name__ == "__main__":
    main()
