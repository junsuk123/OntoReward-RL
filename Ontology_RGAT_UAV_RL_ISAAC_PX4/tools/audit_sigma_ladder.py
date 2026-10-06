#!/usr/bin/env python3
"""Why a sampled policy cannot land: acceleration noise -> body rate -> no corridor.

A checkpoint whose MEAN lands is replayed at several exploration sigmas on the
same nominal seeds, recording per step what the exploration noise does to the
quantities the supervisor gates on. Measured 2026-10-06 on the behaviour-cloned
`ppo_ontology_rgat` seed 828 (12 seeds, 7001-7012):

===========  =====  =======  =========  ==========  ========  ========
sigma        land   return   rate<=lim  corridor    interv.   dA_xy
===========  =====  =======  =========  ==========  ========  ========
determin.    0.50     5.57      0.852       0.624     0.196    0.160
0.333        0.00   -17.89      0.108       0.007     0.712    1.125
0.082        0.00   -20.40      0.447       0.214     0.437    0.390
0.041        0.50     2.75      0.781       0.496     0.234    0.250
0.030        0.42    -5.15      0.808       0.574     0.216    0.227
0.010        0.67    12.37      0.871       0.625     0.154    0.157
===========  =====  =======  =========  ==========  ========  ========

`rate<=lim` is the share of steps with |w_xy| inside `touchdown_rate`,
`corridor` the share with the terminal-descent corridor armed, `interv.` the
share the supervisor rewrote, `dA_xy` the mean step-to-step change of the
requested lateral acceleration in m/s^2. At the shipped sigma 0.333 the
requested lateral acceleration jumps 1.1 m/s^2 per 0.1 s decision; the
derived attitude follows, the body rate sits inside its 10 deg/s limit on
one step in nine, and the corridor -- which arms only while `settled` --
opens on 0.7 % of steps. Nothing in the reward can be learned from a batch
that physically cannot contain a landing; this is the quantity to measure
before touching the reward table again.

Read-only: no training, no selection, no reward shaping.
"""
import argparse
from collections import Counter
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from ontology_rgat.spatial.core import SpatialConfig, schema_for
from ontology_rgat.spatial.environment import SpatialLandingEnv
from ontology_rgat.spatial.runtime_contract import deployment_profile
from ontology_rgat.spatial.training import load_agent


def replay(agent, cfg, seeds, sigma):
    """One row of the ladder; `sigma=None` is the deterministic mean."""
    if sigma is not None:
        agent.actor.maximum_log_std = math.log(sigma)
        agent.actor.minimum_log_std = min(agent.actor.minimum_log_std, math.log(sigma))
    torch.manual_seed(0)
    env = SpatialLandingEnv(cfg)
    outcomes, returns, jerk = Counter(), [], []
    steps = rate_ok = tilt_ok = corridor = margin = intervened = 0
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            total, previous = 0.0, None
            for _ in range(int(cfg.horizon / cfg.dt) * 3 + 5):
                _, action, _, _ = agent.act(observation, deterministic=sigma is None)
                observation, reward, done, _, info = env.step(action)
                total += reward
                steps += 1
                rate_ok += np.linalg.norm(info["truth_angular_rate"][:2]) <= cfg.touchdown_rate
                tilt_ok += np.linalg.norm(info["truth_roll_pitch"]) <= cfg.touchdown_tilt
                corridor += "terminal_descent_corridor" in info["safety_reasons"]
                margin += "vertical_stopping_margin" in info["safety_reasons"]
                intervened += bool(info["safety_intervened"])
                requested = np.asarray(info["requested_acceleration_m_s2"][:2])
                if previous is not None:
                    jerk.append(float(np.linalg.norm(requested - previous)))
                previous = requested
                if done:
                    break
            outcomes[info["status"]] += 1
            returns.append(total)
    finally:
        env.close()
    return dict(
        sigma="deterministic" if sigma is None else sigma,
        landing_rate=outcomes["SUCCESS"] / len(seeds), outcomes=dict(outcomes),
        mean_return=round(float(np.mean(returns)), 3),
        body_rate_within_limit=round(rate_ok / steps, 4),
        tilt_within_limit=round(tilt_ok / steps, 4),
        corridor_armed=round(corridor / steps, 4),
        stopping_margin_inhibit=round(margin / steps, 4),
        intervened=round(intervened / steps, 4),
        lateral_acceleration_step_change_m_s2=round(float(np.mean(jerk)), 4))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help="an eligible spatial checkpoint")
    parser.add_argument("--contract-version", default="reference")
    parser.add_argument("--seeds", type=int, nargs=2, default=[7001, 7013],
                        metavar=("START", "STOP"))
    parser.add_argument("--sigmas", type=float, nargs="+",
                        default=[0.333, 0.082, 0.041, 0.030, 0.010])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(not 0 < sigma <= 1 for sigma in args.sigmas):
        parser.error("sigmas must be in (0, 1]")
    torch.set_num_threads(1)
    schema = schema_for(args.contract_version)
    cfg = replace(SpatialConfig(), schema=schema,
                  isaac_profile_sha256=deployment_profile(schema)["sha256"])
    seeds = list(range(*args.seeds))
    rows = []
    for sigma in [None, *args.sigmas]:
        agent, _meta = load_agent(args.checkpoint, cfg)
        rows.append(replay(agent, cfg, seeds, sigma))
        print(json.dumps(rows[-1]), flush=True)
    report = dict(audit="exploration-sigma-ladder/1", contract=cfg.signature,
                  checkpoint=str(args.checkpoint), seeds=seeds, rows=rows)
    if args.output:
        # Evidence files are immutable; a new audit is an explicit new file.
        with args.output.open("x") as stream:
            stream.write(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
