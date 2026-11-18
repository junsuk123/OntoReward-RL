#!/usr/bin/env python3
"""WMON vs a constant-acceleration Kalman filter on WMON's own benchmark.

Episodes: run_WMON_demo.m's ``simulateEpisode`` run in MATLAB (tools/wmon_episodes.m):
train 1..12, validation 501..503, MATLAB held-out 504..507, extended held-out
508..557.  Both estimators see the SAME sensors (position with dropouts,
velocity, acceleration) and are tuned on the SAME train + validation episodes:

* WMON   -- the MATLAB-trained theta, and optionally a re-training here
            (SPSA + Adam, --retrain-epochs, three restarts, best validation);
* Kalman -- R from simulateEpisode's training noise, jerk PSD q and an R scale
            picked by grid search on train + validation (2 tuned numbers).

    python tools/wmon_vs_kalman_benchmark.py --retrain-epochs 300
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from ontology_rgat.spatial import wmon  # noqa: E402
from ontology_rgat.spatial.kinematic_kf import KalmanNoise, kalman_episode  # noqa: E402

DEFAULT = ROOT / "results" / "wmon_vs_kalman_20261110" / "wmon_episodes.mat"
DT, FIRST = 0.1, 11          # MATLAB evaluates cfg.win:cfg.T (1-based)
TRAIN_NOISE = (0.26, 0.16, 0.19)


def _corrupt(o, kind, rng):
    o = o.copy()
    T = o.shape[1]
    if kind == "position noise burst (+0.78 m sigma, 3 s)":
        s = rng.integers(20, 50)
        o[:2, s:s + 30] += rng.normal(0, 0.78, (2, 30))
    elif kind == "position outliers 5% (+-2 m)":
        m = rng.uniform(size=T) < 0.05
        o[:2, m] += rng.choice([-2.0, 2.0], size=(2, int(m.sum())))
    elif kind == "velocity noise burst (+0.48 m/s sigma, 3 s)":
        s = rng.integers(20, 50)
        o[2:4, s:s + 30] += rng.normal(0, 0.48, (2, 30))
    elif kind == "IMU bias 0.5 m/s^2":
        o[4:6] += 0.5
    return o


def robustness(obs, xt, pv, idx, params, kalman):
    out = {}
    for kind in ("clean", "position noise burst (+0.78 m sigma, 3 s)",
                 "position outliers 5% (+-2 m)", "velocity noise burst (+0.48 m/s sigma, 3 s)",
                 "IMU bias 0.5 m/s^2"):
        rng = np.random.default_rng(7)
        w, k = [], []
        for i in idx:
            o = _corrupt(obs[i], kind, rng)
            sw = wmon.infer_episode(params, o, pv[i], DT).state
            sk = kalman_episode(kalman, o, pv[i], DT)
            w.append(np.sqrt(np.mean((sw[:2, FIRST:] - xt[i][:2, FIRST:]) ** 2)))
            k.append(np.sqrt(np.mean((sk[:2, FIRST:] - xt[i][:2, FIRST:]) ** 2)))
        w, k = np.array(w), np.array(k)
        out[kind] = {"wmon_position_rmse_m": float(w.mean()),
                     "kalman_position_rmse_m": float(k.mean()),
                     "wmon_better_episodes": int((w < k).sum()), "episodes": len(idx)}
        print(f"  {kind:44s} WMON {w.mean():.4f}  Kalman {k.mean():.4f}  "
              f"WMON better in {(w < k).sum()}/{len(idx)}")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--episodes", type=Path, default=DEFAULT)
    parser.add_argument("--retrain-epochs", type=int, default=0)
    parser.add_argument("--robustness", action="store_true",
                        help="also inject noise bursts / outliers / IMU bias into the extended "
                             "held-out episodes (estimators keep their clean tuning)")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    d = sio.loadmat(args.episodes)
    seeds = d["seeds"].ravel()
    obs, xt, matlab_state = d["obs"], d["xTrue"], d["wmonState"]
    pv = d["posValid"].astype(bool)
    split = {"train": [i for i, s in enumerate(seeds) if s <= 12],
             "validation": [i for i, s in enumerate(seeds) if 501 <= s <= 503],
             "matlab_heldout": [i for i, s in enumerate(seeds) if 504 <= s <= 507],
             "extended_heldout": [i for i, s in enumerate(seeds) if s >= 508]}
    tune = split["train"] + split["validation"]
    theta = d["theta"].ravel()
    params = wmon.WMONParams.from_theta(theta)
    parity = max(float(np.abs(wmon.infer_episode(params, obs[i], pv[i], DT).state
                              - matlab_state[i]).max()) for i in range(len(seeds)))
    print(f"python WMON vs MATLAB WMON on {len(seeds)} episodes: max |diff| {parity:.1e}")

    def metrics(state, i):
        ep = state[:2, FIRST:] - xt[i][:2, FIRST:]
        ev = state[2:4, FIRST:] - xt[i][2:4, FIRST:]
        dark = ~pv[i][FIRST:]
        return (float(np.mean(ep ** 2) + wmon.VELOCITY_LOSS_WEIGHT * np.mean(ev ** 2)),
                float(np.sqrt(np.mean(ep ** 2))), float(np.sqrt(np.mean(ev ** 2))),
                float(np.sqrt(np.mean(ep[:, dark] ** 2))) if dark.any() else float("nan"))

    def wmon_loss(th, idx):
        p = wmon.WMONParams.from_theta(th)
        return float(np.mean([wmon.episode_loss(p, obs[i], pv[i], xt[i], DT) for i in idx]))

    best = None
    for q in np.logspace(-3, 2, 26):
        for scale in (0.5, 0.75, 1.0, 1.5, 2.0):
            noise = KalmanNoise(*(scale * s for s in TRAIN_NOISE), float(q))
            loss = float(np.mean([metrics(kalman_episode(noise, obs[i], pv[i], DT), i)[0]
                                  for i in tune]))
            if best is None or loss < best[0]:
                best = (loss, noise)
    kalman = best[1]
    print("Kalman selected", kalman.to_dict())

    estimators = {
        "WMON (MATLAB theta)": lambda i: matlab_state[i],
        "Kalman (constant acceleration)": lambda i: kalman_episode(kalman, obs[i], pv[i], DT),
        "raw / hold": lambda i: wmon.hold_missing_position(obs[i], pv[i]),
    }
    retrained = None
    if args.retrain_epochs:
        runs = []
        for restart, (start, lr) in enumerate([(wmon.MATLAB_INITIAL_THETA, 0.045),
                                               (wmon.MATLAB_TRAINED_THETA, 0.02),
                                               (wmon.MATLAB_INITIAL_THETA, 0.08)]):
            r = wmon.spsa_adam(lambda th, ids: wmon_loss(th, [split["train"][j] for j in ids]),
                               len(split["train"]), start, epochs=args.retrain_epochs,
                               batch_size=4, learning_rate=lr,
                               validation_loss=lambda th: wmon_loss(th, split["validation"]),
                               rng=np.random.default_rng(100 + restart))
            runs.append((wmon_loss(r.theta, split["validation"]), r.theta, r.best_epoch))
            print(f"WMON restart {restart}: best epoch {r.best_epoch}, validation {runs[-1][0]:.5f}")
        retrained = min(runs, key=lambda t: t[0])
        p2 = wmon.WMONParams.from_theta(retrained[1])
        estimators["WMON (re-trained here)"] = (
            lambda i: wmon.infer_episode(p2, obs[i], pv[i], DT).state)

    summary = {"episodes": str(args.episodes), "python_matlab_wmon_parity": parity,
               "kalman": kalman.to_dict(), "kalman_tune_loss": best[0], "splits": {}}
    if retrained is not None:
        summary["wmon_retrained_theta"] = retrained[1].tolist()
    for name in ("matlab_heldout", "extended_heldout"):
        idx = split[name]
        rows = {k: np.array([metrics(f(i), i) for i in idx]) for k, f in estimators.items()}
        k_pos, w_pos = rows["Kalman (constant acceleration)"][:, 1], rows["WMON (MATLAB theta)"][:, 1]
        summary["splits"][name] = {
            "episodes": len(idx),
            "estimators": {k: {"loss": float(r[:, 0].mean()),
                               "position_rmse_m": float(r[:, 1].mean()),
                               "velocity_rmse_m_s": float(r[:, 2].mean()),
                               "dropout_position_rmse_m": float(np.nanmean(r[:, 3]))}
                           for k, r in rows.items()},
            "kalman_better_episodes": int((k_pos < w_pos).sum())}
        print(f"--- {name} ({len(idx)} episodes): loss | pos RMSE | vel RMSE | pos RMSE in dropout")
        for k, e in summary["splits"][name]["estimators"].items():
            print(f"  {k:32s} {e['loss']:.5f} | {e['position_rmse_m']:.4f} | "
                  f"{e['velocity_rmse_m_s']:.4f} | {e['dropout_position_rmse_m']:.4f}")
        print(f"  paired: Kalman lower position RMSE than MATLAB WMON in "
              f"{summary['splits'][name]['kalman_better_episodes']}/{len(idx)}")
    if args.robustness:
        summary["robustness"] = robustness(obs, xt, pv, split["extended_heldout"], params, kalman)
    out = args.output or args.episodes.parent / "summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
