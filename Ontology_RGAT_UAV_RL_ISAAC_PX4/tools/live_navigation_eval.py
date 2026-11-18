#!/usr/bin/env python3
"""Live own navigation (observation profile /4): hold vs Kalman vs WMON under GNSS outages.

Opt-in study tool for ``current_semantics.navigation``.  Never runs Isaac.

1. record     -- local /12 flights (task-v5 split) with a scripted PD on the
                 registered observation; the recorded Measurements carry the
                 plant's exact own state, which is the navigation reference.
2. tune       -- on the training flights: Kalman jerk PSD by grid (R = the
                 true sensor sigmas); WMON = MATLAB theta, and SPSA + Adam
                 re-tuned (--tune-epochs), both on own-navigation error.
3. replay     -- each estimator on the SAME streams and the SAME GNSS noise and
                 outages (drawn from the episode seed): own-navigation error and
                 the error of the 21D relative state, by phase.
4. closed loop (--closed-loop) -- the PD flies the held-out seeds with each
                 navigation in the loop; verdicts are counted.

    python tools/live_navigation_eval.py --episodes 36 --tune-epochs 40 --closed-loop
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import datetime as dt
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from ontology_rgat.current_semantics.backend import (  # noqa: E402
    CurrentObserver, LocalOpticalCadence, make_local_backend)
from ontology_rgat.current_semantics.contracts import (  # noqa: E402
    contract_manifest, load_config)
from ontology_rgat.current_semantics.navigation import (  # noqa: E402
    LiveNavigation, NavigationDegradation, NavigationEstimator)
from ontology_rgat.current_semantics.observation import decode  # noqa: E402
from ontology_rgat.spatial.kinematic_kf import KalmanNoise  # noqa: E402
from ontology_rgat.spatial.wmon import MATLAB_TRAINED_THETA, spsa_adam  # noqa: E402
import wmon_pad_track_eval as pad  # noqa: E402

PHASES = ("all", "gnss_ok", "gnss_outage", "after_outage")
AFTER_S = 1.5


def record(config, seed: int, split: str, navigation=None) -> pad.Flight | str:
    if navigation is None:
        return pad.record(config, seed, split, dropout=None)
    backend = make_local_backend(config, scenario=split, navigation=navigation)
    policy = pad.pd_policy(np.asarray(config.action.maximum_m_s2, dtype=np.float64))
    try:
        vector, info = backend.reset(seed)
        while True:
            step = backend.step(policy(decode(vector)))
            vector, info = step.observation, step.info
            if step.done:
                return str(info["termination"]["verdict"])
    finally:
        backend.close()


def own_reference(flight: pad.Flight) -> tuple[np.ndarray, np.ndarray]:
    """Plant own position/velocity at each decision (the navigation reference)."""
    ms = [flight.measurements[i - 1] for i in flight.decision_index]
    return (np.array([m.own_position for m in ms], float),
            np.array([m.own_velocity for m in ms], float))


def nav_only(flight: pad.Flight, spec: NavigationDegradation, est: NavigationEstimator):
    """Own-navigation estimate at every decision, without the observer (for tuning)."""
    nav = LiveNavigation(spec, est)
    nav.reset(flight.seed)
    pos, vel, fed = [], [], 0
    for upto in flight.decision_index:
        while fed < upto:
            nav(flight.measurements[fed])
            fed += 1
        pos.append(nav.position.copy())
        vel.append(nav.velocity.copy())
    return np.array(pos), np.array(vel), nav


def nav_loss(flight, spec, est) -> float:
    p, v, _ = nav_only(flight, spec, est)
    rp, rv = own_reference(flight)
    return float(np.mean((p - rp) ** 2) + 0.35 * np.mean((v - rv)[:, :2] ** 2))


def replay(flight: pad.Flight, spec, est) -> dict[str, np.ndarray]:
    cadence = None if flight.cadence is None else LocalOpticalCadence(*flight.cadence)
    nav = None if est is None else LiveNavigation(spec, est)
    observer = CurrentObserver(cadence, navigation=nav)
    observer.reset("replay", seed=flight.seed)
    if nav is not None:
        nav.reset(flight.seed)
    n = len(flight.decision_index)
    out = {k: np.full((n, 3), np.nan) for k in ("rel_p", "rel_v", "own_p", "own_v")}
    outage = np.zeros(n, dtype=bool)
    since = np.full(n, np.inf)
    fed = 0
    for row, upto in enumerate(flight.decision_index):
        while fed < upto:
            m = flight.measurements[fed]
            observer.ingest(m if nav is None else nav(m))
            fed += 1
        out["rel_p"][row], out["rel_v"][row], _ = pad._relative(observer)
        m = observer.last_measurement
        out["own_p"][row], out["own_v"][row] = m.own_position, m.own_velocity
        if nav is not None:
            t = float(m.time_s)
            outage[row] = nav.in_outage(t)
            if nav.outage is not None:
                end = nav.t0 + nav.outage[0] + nav.outage[1]
                since[row] = t - end if t >= end else np.inf
    out["outage"], out["since_outage"] = outage, since
    return out


def phase_masks(result) -> dict[str, np.ndarray]:
    every = np.ones(len(result["outage"]), dtype=bool)
    after = (result["since_outage"] >= 0) & (result["since_outage"] < AFTER_S)
    return {"all": every, "gnss_ok": ~result["outage"] & ~after,
            "gnss_outage": result["outage"], "after_outage": after}


def summarize(flights, results, masks_from) -> dict[str, Any]:
    out = {}
    for phase in PHASES:
        acc = {k: [] for k in ("own_p", "own_v", "rel_p", "rel_v")}
        for f, r, ref in zip(flights, results, masks_from):
            mask = phase_masks(ref)[phase]
            rp, rv = own_reference(f)
            acc["own_p"].append(np.linalg.norm(r["own_p"] - rp, axis=1)[mask])
            acc["own_v"].append(np.linalg.norm((r["own_v"] - rv)[:, :2], axis=1)[mask])
            acc["rel_p"].append(np.linalg.norm(r["rel_p"] - f.truth_position, axis=1)[mask])
            acc["rel_v"].append(np.linalg.norm((r["rel_v"] - f.truth_velocity)[:, :2],
                                               axis=1)[mask])
        row = {"decisions": int(sum(a.size for a in acc["own_p"]))}
        for k, parts in acc.items():
            x = np.concatenate(parts)
            x = x[np.isfinite(x)]
            row[k + "_rmse"] = float(np.sqrt(np.mean(x ** 2))) if x.size else None
        out[phase] = row
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--episodes", type=int, default=36)
    parser.add_argument("--seed0", type=int, default=920_000)
    parser.add_argument("--split", default="in_distribution")
    parser.add_argument("--tune-epochs", type=int, default=0)
    parser.add_argument("--closed-loop", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    started = time.time()
    config = load_config()
    spec = NavigationDegradation()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output or ROOT / "results" / f"live_navigation_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.seed0, args.seed0 + args.episodes))
    print(f"recording {len(seeds)} flights ...", flush=True)
    flights = [record(config, s, args.split) for s in seeds]
    n_train, n_val = len(flights) // 2, max(1, len(flights) // 6)
    train, val, test = flights[:n_train], flights[n_train:n_train + n_val], flights[n_train + n_val:]

    sig = spec.gnss_sigma_m
    grid = [float(q) for q in np.logspace(-2, 3, 11)]
    q = min(grid, key=lambda q: np.mean([nav_loss(f, spec, NavigationEstimator(
        "kalman", kalman=KalmanNoise(float(np.mean(sig)), spec.velocity_sigma_m_s,
                                     spec.imu_sigma_m_s2, q))) for f in train + val]))
    kalman = NavigationEstimator("kalman", kalman=KalmanNoise(
        float(np.mean(sig)), spec.velocity_sigma_m_s, spec.imu_sigma_m_s2, q))
    estimators = {"truth navigation (registered /3)": None,
                  "hold (last GNSS fix)": NavigationEstimator("hold"),
                  f"Kalman (q={q:.3g})": kalman,
                  "WMON (MATLAB theta)": NavigationEstimator("wmon")}
    if args.tune_epochs:
        def batch(theta, ids):
            est = NavigationEstimator("wmon", wmon_theta=tuple(theta))
            return float(np.mean([nav_loss(train[i], spec, est) for i in ids]))
        result = spsa_adam(batch, len(train), MATLAB_TRAINED_THETA, epochs=args.tune_epochs,
                           batch_size=6, learning_rate=0.045,
                           validation_loss=lambda th: float(np.mean([nav_loss(
                               f, spec, NavigationEstimator("wmon", wmon_theta=tuple(th)))
                               for f in val])),
                           rng=np.random.default_rng(20261110),
                           log=lambda s: print(s, flush=True))
        estimators["WMON (SPSA re-tuned)"] = NavigationEstimator(
            "wmon", wmon_theta=tuple(float(v) for v in result.theta))

    # phases come from the navigation draw, identical for every estimator
    reference = [replay(f, spec, NavigationEstimator("hold")) for f in test]
    summary: dict[str, Any] = {
        "created": stamp, "split": args.split, "degradation": spec.to_dict(),
        "seeds": {"train": [f.seed for f in train], "validation": [f.seed for f in val],
                  "test": [f.seed for f in test]},
        "observation_hash": {
            "registered_3": contract_manifest(config)["observation_hash"],
            "live_navigation_4": contract_manifest(
                config, navigation=LiveNavigation(spec, NavigationEstimator()).manifest_part()
            )["observation_hash"]},
        "estimators": {}}
    for name, est in estimators.items():
        results = [replay(f, spec, est) for f in test]
        summary["estimators"][name] = {"settings": None if est is None else est.to_dict(),
                                       "test": summarize(test, results, reference)}
    if args.closed_loop:
        print("closed loop on the held-out seeds ...", flush=True)
        summary["closed_loop"] = {}
        for name, est in estimators.items():
            verdicts: dict[str, int] = {}
            for f in test:
                v = (f.verdict if est is None else
                     record(config, f.seed, args.split, LiveNavigation(spec, est)))
                verdicts[v] = verdicts.get(v, 0) + 1
            summary["closed_loop"][name] = verdicts
    summary["wall_s"] = time.time() - started
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nheld-out {len(test)} episodes -> {out_dir}")
    print("RMSE own position m | own velocity xy m/s | 21D relative position m | relative velocity xy m/s")
    for name, entry in summary["estimators"].items():
        print(f"  {name}")
        for phase in PHASES:
            r = entry["test"][phase]
            fmt = lambda x: "  -  " if x is None else f"{x:.3f}"
            print(f"    {phase:12s} n={r['decisions']:5d}  own_p {fmt(r['own_p_rmse'])}  "
                  f"own_v {fmt(r['own_v_rmse'])}  rel_p {fmt(r['rel_p_rmse'])}  "
                  f"rel_v {fmt(r['rel_v_rmse'])}")
    for name, verdicts in summary.get("closed_loop", {}).items():
        print(f"  closed loop {name:32s} {verdicts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
