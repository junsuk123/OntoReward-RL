#!/usr/bin/env python3
"""Does the local plant answer a command the way the Isaac vehicle does?

Fits one finite impulse response per plant, ``dv/dt[t] = sum_k h_k a[t-k]``
over 0.1 s lags, from the first decisions of every Isaac episode in a run
directory (``isaac_*.jsonl`` traces) and from the same recorded commands
replayed open-loop through the local plant of each schema given. Measured
2026-10-06 on the first learned-policy Isaac flight (12 episodes, 0 landings):

====================================  ==============================  =====
plant                                 h (lags 0..3)                   sum
====================================  ==============================  =====
Isaac/PX4, ground-truth velocity      [-0.02, 0.74, 0.24, -0.01]      0.91
Isaac/PX4, EKF own velocity (policy)  [ 0.32, 0.43, 0.22, -0.12]      0.88
spatial-reference/2 (instant)         [ 0.44, 0.15, 0.18, -0.04]      0.76
spatial-reference/3 (0.10 s latency)  [ 0.05, 0.35, 0.24,  0.14]      0.79
====================================  ==============================  =====

Isaac has no response in the step a command is issued; the /2 plant answers
inside it, and a policy tuned on /2 over-commands the real vehicle until the
pad leaves the camera. Run this after every Isaac flight: if the local row
drifts from the Isaac row again, the plant, not the policy, is the suspect.

Read-only: no training, no selection.
"""
import argparse
from dataclasses import replace
import glob
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from ontology_rgat.spatial.core import SpatialConfig, schema_for
from ontology_rgat.spatial.environment import SpatialLandingEnv
from ontology_rgat.spatial.runtime_contract import deployment_profile


def impulse_response(series, lags):
    rows, targets = [], []
    for episode in series:
        command = np.array([s[0] for s in episode])
        velocity = np.array([s[1] for s in episode])
        dt = np.array([s[2] for s in episode])
        realized = (velocity[1:] - velocity[:-1]) / dt[:-1, None]
        for t in range(lags - 1, len(realized)):
            for axis in range(command.shape[1]):
                rows.append([command[t - k, axis] for k in range(lags)])
                targets.append(realized[t, axis])
    X, y = np.asarray(rows), np.asarray(targets)
    h, *_ = np.linalg.lstsq(X, y, rcond=None)
    fit = 1.0 - ((y - X @ h) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
    return h, float(fit), len(y)


def isaac_episodes(directory, steps):
    recorded, truth, ekf = [], [], []
    for path in sorted(glob.glob(str(directory / "isaac_*.jsonl"))):
        rows = [json.loads(line) for line in open(path)]
        for seed in sorted({row["seed"] for row in rows}):
            episode = [row for row in rows
                       if row["seed"] == seed and not row["info"]["abort_requested"]][:steps]
            if len(episode) < 8:
                continue
            recorded.append((seed, [np.asarray(row["action"]) for row in episode]))
            applied = [np.asarray(row["info"]["applied_acceleration_m_s2"][:2]) for row in episode]
            dts = [row["info"]["dt_s"] for row in episode]
            truth.append(list(zip(applied, [np.asarray(row["info"]["truth_relative_velocity"][:2]) for row in episode], dts)))
            ekf.append(list(zip(applied, [np.asarray(row["info"]["own_velocity_enu_m_s"][:2]) for row in episode], dts)))
    return recorded, truth, ekf


def local_episodes(schema, recorded):
    cfg = replace(SpatialConfig(), schema=schema,
                  isaac_profile_sha256=deployment_profile(schema)["sha256"])
    series = []
    for seed, actions in recorded:
        env = SpatialLandingEnv(cfg)
        try:
            env.reset(seed=seed)
            episode = []
            for action in actions:
                _obs, _r, done, _t, info = env.step(action)
                episode.append((np.asarray(info["applied_acceleration_m_s2"][:2]),
                                np.asarray(info["truth_relative_velocity"][:2]),
                                info["dt_s"]))
                if done:
                    break
        finally:
            env.close()
        series.append(episode)
    return series


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("isaac_output", type=Path,
                        help="a pipeline output directory holding isaac_*.jsonl traces")
    parser.add_argument("--schemas", nargs="+", default=["spatial-reference/2", "reference"])
    parser.add_argument("--lags", type=int, default=6)
    parser.add_argument("--steps", type=int, default=20,
                        help="decisions per episode to fit, before any abort hold")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    recorded, truth, ekf = isaac_episodes(args.isaac_output, args.steps)
    if not recorded:
        raise SystemExit(f"no usable isaac_*.jsonl episodes under {args.isaac_output}")
    report = {"audit": "actuation-response/1", "isaac_output": str(args.isaac_output),
              "episodes": len(recorded), "lags": args.lags, "rows": []}
    for label, series in (("isaac truth_relative_velocity", truth),
                          ("isaac own_velocity (EKF, policy input)", ekf)):
        h, fit, n = impulse_response(series, args.lags)
        report["rows"].append({"plant": label, "h": h.tolist(), "sum": float(h.sum()), "r2": fit, "n": n})
    target = np.asarray(report["rows"][0]["h"])
    for version in args.schemas:
        schema = schema_for(version)
        h, fit, n = impulse_response(local_episodes(schema, recorded), args.lags)
        report["rows"].append({"plant": f"local {schema}", "h": h.tolist(), "sum": float(h.sum()),
                               "r2": fit, "n": n,
                               "mean_abs_error_vs_isaac_truth": float(np.abs(h - target).mean())})
    for row in report["rows"]:
        error = row.get("mean_abs_error_vs_isaac_truth")
        print(f"{row['plant']:42s} h={np.round(row['h'][:4], 2)} sum={row['sum']:.2f} r2={row['r2']:.2f}"
              + (f" err={error:.3f}" if error is not None else ""))
    if args.output:
        with args.output.open("x") as stream:
            stream.write(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
