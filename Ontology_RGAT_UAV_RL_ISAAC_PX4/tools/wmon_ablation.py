#!/usr/bin/env python3
"""Which part of WMON carries its gain?  Component ablation on WMON's own benchmark.

Episodes: tools/wmon_episodes.m (run_WMON_demo.m simulateEpisode; train 1-12,
validation 501-503, held-out 508-557).  Every variant is RE-TRAINED with the MATLAB
rule (SPSA + Adam, best validation) on the same data, so a removed component is not
penalised by coefficients tuned for its presence.

    L0     FFT wave mixing          fft_mix                      -> "no FFT mix": 0
    L1-L2  particle confidence       1/(1 + s * hf_ratio)         -> "constant confidence": s -> 0
    L4-L5  relational (bond) energy  k_pv, k_va                   -> "no bonds": 0

    python tools/wmon_ablation.py --epochs 300
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from ontology_rgat.spatial import wmon  # noqa: E402

EPISODES = ROOT / "results" / "wmon_vs_kalman_20261110" / "wmon_episodes.mat"
DT, FIRST = 0.1, 11
VARIANTS = {
    "full": dict(bonds=True, fft=True, adaptive=True),
    "no FFT mix (L0)": dict(bonds=True, fft=False, adaptive=True),
    "constant confidence (L1-L2)": dict(bonds=True, fft=True, adaptive=False),
    "bonds only (no L0, no L1-L2)": dict(bonds=True, fft=False, adaptive=False),
    "no bonds (L4-L5)": dict(bonds=False, fft=True, adaptive=True),
    "no bonds, no FFT, constant conf.": dict(bonds=False, fft=False, adaptive=False),
}


def params_for(theta, variant) -> wmon.WMONParams:
    p = wmon.WMONParams.from_theta(theta)
    if not variant["fft"]:
        p = replace(p, fft_mix=0.0)
    if not variant["adaptive"]:
        p = replace(p, noise_sensitivity=1e-12)   # confidence == 1 (0 for a missing position)
    return p


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--output", type=Path, default=EPISODES.parent / "ablation.json")
    args = parser.parse_args(argv)
    d = sio.loadmat(EPISODES)
    seeds = d["seeds"].ravel()
    obs, xt = d["obs"], d["xTrue"]
    pv = d["posValid"].astype(bool)
    train = [i for i, s in enumerate(seeds) if s <= 12]
    val = [i for i, s in enumerate(seeds) if 501 <= s <= 503]
    test = [i for i, s in enumerate(seeds) if s >= 508]

    def state(theta, variant, i):
        return wmon.infer_episode(params_for(theta, variant), obs[i], pv[i], DT,
                                  use_bonds=variant["bonds"]).state

    def loss(theta, variant, idx):
        out = []
        for i in idx:
            s = state(theta, variant, i)
            ep = s[:2, FIRST:] - xt[i][:2, FIRST:]
            ev = s[2:4, FIRST:] - xt[i][2:4, FIRST:]
            out.append(np.mean(ep ** 2) + wmon.VELOCITY_LOSS_WEIGHT * np.mean(ev ** 2))
        return float(np.mean(out))

    def metrics(theta, variant, idx):
        pos, dark, vel = [], [], []
        for i in idx:
            s = state(theta, variant, i)
            ep = s[:2, FIRST:] - xt[i][:2, FIRST:]
            m = ~pv[i][FIRST:]
            pos.append(np.sqrt(np.mean(ep ** 2)))
            vel.append(np.sqrt(np.mean((s[2:4, FIRST:] - xt[i][2:4, FIRST:]) ** 2)))
            if m.any():
                dark.append(np.sqrt(np.mean(ep[:, m] ** 2)))
        return np.array(pos), float(np.mean(dark)), float(np.mean(vel))

    summary, per_episode = {}, {}
    for name, variant in VARIANTS.items():
        runs = []
        for restart, (start, lr) in enumerate([(wmon.MATLAB_TRAINED_THETA, 0.02),
                                               (wmon.MATLAB_INITIAL_THETA, 0.045)]):
            r = wmon.spsa_adam(lambda th, ids: loss(th, variant, [train[j] for j in ids]),
                               len(train), start, epochs=args.epochs, batch_size=4,
                               learning_rate=lr,
                               validation_loss=lambda th: loss(th, variant, val),
                               rng=np.random.default_rng(300 + restart))
            runs.append((loss(r.theta, variant, val), r.theta))
        best_val, theta = min(runs, key=lambda t: t[0])
        pos, dark, vel = metrics(theta, variant, test)
        per_episode[name] = pos
        summary[name] = {"validation_loss": best_val, "test_loss": loss(theta, variant, test),
                         "position_rmse_m": float(pos.mean()),
                         "dropout_position_rmse_m": dark, "velocity_rmse_m_s": vel,
                         "params": params_for(theta, variant).to_dict(),
                         "bonds": variant["bonds"]}
        print(f"{name:34s} test loss {summary[name]['test_loss']:.5f}  pos {pos.mean():.4f}  "
              f"dropout pos {dark:.4f}  vel {vel:.4f}", flush=True)
    full = per_episode["full"]
    for name, pos in per_episode.items():
        summary[name]["worse_than_full_episodes"] = int((pos > full).sum())
        summary[name]["delta_position_rmse_vs_full_m"] = float((pos - full).mean())
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
