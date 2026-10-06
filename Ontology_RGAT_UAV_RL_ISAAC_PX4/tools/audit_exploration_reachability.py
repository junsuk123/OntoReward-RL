#!/usr/bin/env python3
"""Is a landing reachable from the distribution the policy actually samples?

A ceiling measured with a tuned controller says the TASK is solvable. It says
nothing about whether the policy's own action distribution can ever stumble
into a landing, and that is the quantity PPO depends on. Running this on
2026-10-05 is what showed that 838 episodes at the easiest curriculum rung
produced zero SUCCESS on a rung where a constant action lands 40 out of 40.

Three numbers per rung, none of them derived:

reachability
    Outcomes of a fixed open-loop descent and of the untrained policy, on the
    same seeds. A large gap means exploration, not the task, is the obstacle.

vertical bias
    Mean requested against mean APPLIED vertical acceleration. The supervisors
    rewrite only negative vertical commands while descent is inhibited
    (``spatial/safety.py``, ``two_axis/safety.py``), so a zero-mean policy can
    come out with a positive applied mean -- an altitude ratchet, since the
    direct-acceleration plant has no restoring force on altitude.

override fraction
    Share of steps whose action the supervisor replaced before it reached the
    plant. PPO regresses on the action the policy PROPOSED, so this is the
    share of the policy gradient that describes something that never ran.

Read-only: no training, no checkpoint selection, no reward shaping.
"""
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from ontology_rgat.spatial.core import SpatialConfig, REFERENCE_SCHEMA
from ontology_rgat.spatial.environment import SpatialLandingEnv
from ontology_rgat.spatial.training import SpatialAgent


def _episode(env, cfg, seed, difficulty, act):
    obs, _ = env.reset(seed=seed, difficulty=difficulty)
    requested, applied, overridden, heights = [], [], 0, []
    steps = int(cfg.horizon / cfg.dt) + 2
    for _ in range(steps):
        obs, _reward, done, _truncated, info = env.step(act(obs))
        requested.append(info['requested_acceleration_m_s2'][2])
        applied.append(info['applied_acceleration_m_s2'][2])
        overridden += bool(info['safety_intervened'])
        heights.append(info['truth_relative_position'][2])
        if done:
            return dict(status=info['status'], steps=len(applied),
                        requested=requested, applied=applied,
                        overridden=overridden,
                        climb=max(heights) - heights[0])
    return dict(status='NO_TERMINAL', steps=len(applied), requested=requested,
                applied=applied, overridden=overridden,
                climb=max(heights) - heights[0])


def _summarize(rows):
    requested = [value for row in rows for value in row['requested']]
    applied = [value for row in rows for value in row['applied']]
    steps = sum(row['steps'] for row in rows)
    return dict(
        episodes=len(rows),
        outcomes=dict(Counter(row['status'] for row in rows)),
        requested_vertical_mean_m_s2=round(float(np.mean(requested)), 4),
        applied_vertical_mean_m_s2=round(float(np.mean(applied)), 4),
        overridden_step_fraction=round(
            sum(row['overridden'] for row in rows) / max(steps, 1), 4),
        median_climb_above_start_m=round(
            float(np.median([row['climb'] for row in rows])), 3),
    )


def audit(cfg, *, seeds, difficulties, arm, policy_seed, descent):
    env = SpatialLandingEnv(cfg)
    try:
        rungs = {}
        for difficulty in difficulties:
            agent = SpatialAgent(arm, cfg, policy_seed, -1.1)
            sampled = [_episode(env, cfg, seed, difficulty,
                                lambda obs: agent.act(obs)[1]) for seed in seeds]
            fixed = np.array([0.0, 0.0, float(descent)])
            open_loop = [_episode(env, cfg, seed, difficulty, lambda _obs: fixed)
                         for seed in seeds]
            rungs[f'difficulty_{difficulty}'] = dict(
                sampled_policy=_summarize(sampled),
                open_loop_descent=_summarize(open_loop))
        return dict(audit='exploration-reachability/1', contract=cfg.signature,
                    arm=arm, policy_seed=policy_seed, seeds=list(seeds),
                    open_loop_vertical_action=descent, rungs=rungs)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=list(range(5001, 5041)))
    parser.add_argument('--difficulties', type=float, nargs='+',
                        default=[0.0, 1.0])
    parser.add_argument('--arm', default='ppo_vector_canonical')
    parser.add_argument('--policy-seed', type=int, default=828)
    parser.add_argument('--descent', type=float, default=-0.12,
                        help='fixed normalized vertical action for the open-loop arm')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not -1.0 <= args.descent < 0.0:
        parser.error('the open-loop arm must command a descent in [-1,0)')
    if any(not 0.0 <= d <= 1.0 for d in args.difficulties):
        parser.error('difficulties must be in [0,1]')
    cfg = replace(SpatialConfig(), schema=REFERENCE_SCHEMA)
    report = audit(cfg, seeds=args.seeds, difficulties=args.difficulties,
                   arm=args.arm, policy_seed=args.policy_seed,
                   descent=args.descent)
    text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output:
        # Evidence files are immutable; a new audit is an explicit new file.
        with args.output.open('x') as stream:
            stream.write(text)
    print(text)


if __name__ == '__main__':
    main()
