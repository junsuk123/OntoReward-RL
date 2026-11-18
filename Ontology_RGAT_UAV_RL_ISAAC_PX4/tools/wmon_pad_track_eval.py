#!/usr/bin/env python3
"""WMON pad track vs the registered ABG track on spatial-reference/12 flights.

Opt-in study tool for ``spatial.wmon_track`` (WMON v0.1, run_WMON_demo.m).  It
does NOT change the registered observation contract and never runs Isaac.

1. record   -- fly local-plant episodes (task-v5 split, evaluation vision
               blackout) with a scripted PD on the REGISTERED observation and
               record every ``Measurement`` the observer received, plus the
               evaluator-only truth at each decision.
2. replay   -- feed each recorded stream, unchanged, through
               ``CurrentObserver()`` (ABG + re-acquire) and
               ``CurrentObserver(track_factory=WMON)``: same sensor data, so the
               difference is the estimator alone (open loop).  The ABG replay
               must reproduce the online observer bit for bit (checked).
3. calibrate (--calibrate) -- the WMON method itself: SPSA + Adam on the seven
               coefficients over the training episodes, best-validation
               selection, report on held-out episodes only.
4. closed loop (--closed-loop) -- fly the held-out seeds again with the PD fed
               by each estimator and count verdicts.

Errors are of the RELATIVE state the 21D vector carries (own minus pad, from
the measured own state), against the evaluator's truth at each decision.

    python tools/wmon_pad_track_eval.py --episodes 36 --calibrate --closed-loop
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import datetime as dt
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from ontology_rgat.current_semantics.backend import (  # noqa: E402
    CurrentObserver, LocalOpticalCadence, make_local_backend)
from ontology_rgat.current_semantics.contracts import VisionBlackout, load_config  # noqa: E402
from ontology_rgat.current_semantics.observation import decode  # noqa: E402
from ontology_rgat.spatial.wmon import (MATLAB_TRAINED, THETA_NAMES,  # noqa: E402
                                        VELOCITY_LOSS_WEIGHT, WMONParams, spsa_adam)
from ontology_rgat.spatial.wmon_track import (WMONTrackSettings,  # noqa: E402
                                              wmon_track_factory)

RECOVERY_S = 1.5   # "after blackout" window used in the phase metrics
LOST_VIEW_AGE_S = 0.5   # sensor-level frame age beyond which the pad counts as lost
PHASES = ("all", "tracking", "lost_view", "blackout", "recovery")


# ------------------------------------------------------------------ policy

def pd_policy(maximum: np.ndarray) -> Callable[[Any], np.ndarray]:
    """Scripted PD on the decoded 21D vector (pad-minus-own error and velocity).

    Not a registered teacher: it only has to keep the pad in view long enough
    for the estimators to have something to track, and to descend.
    """
    def policy(ph: Any) -> np.ndarray:
        if not ph.estimate_valid:
            return np.array([0.0, 0.0, min(0.25, maximum[2])])
        ax = 1.2 * ph.ex + 1.8 * ph.vrx
        ay = 1.2 * ph.ey + 1.8 * ph.vry
        horizontal = math.hypot(ph.ex, ph.ey)
        if ph.vision_age_s > 0.3:
            target_vz = 0.0
        elif horizontal < 0.25:
            target_vz = -0.35
        elif horizontal < 0.8:
            target_vz = -0.1
        else:
            target_vz = 0.05
        az = 1.5 * (target_vz - ph.vz)
        return np.clip(np.array([ax, ay, az]), -maximum, maximum)
    return policy


# --------------------------------------------------------------- recording

@dataclass(frozen=True, eq=False)
class Flight:
    seed: int
    measurements: tuple            # every Measurement handed to the observer
    decision_index: np.ndarray     # (N,) measurements ingested before decision n
    decision_time: np.ndarray      # (N,)
    truth_position: np.ndarray     # (N, 3) own minus pad
    truth_velocity: np.ndarray     # (N, 3) own minus pad
    online_position: np.ndarray    # (N, 3) relative estimate the online ABG observer had
    sensor_age: np.ndarray         # (N,) decision time minus the last delivered capture
    blackout: tuple[float, float] | None   # absolute (start, end) time
    verdict: str
    cadence: tuple | None          # (OpticalCadence, camera_dt) of the observer


def _relative(observer: CurrentObserver) -> tuple[np.ndarray, np.ndarray, bool]:
    m, track = observer.last_measurement, observer.track
    if not track.initialized:
        return np.full(3, np.nan), np.full(3, np.nan), False
    own_p = np.asarray(m.own_position, dtype=np.float64)
    own_v = np.asarray(m.own_velocity, dtype=np.float64)
    return own_p - track.position, own_v - track.velocity, True


def record(config: Any, seed: int, split: str, dropout: VisionBlackout,
           track_factory: Callable[[], Any] | None = None) -> Flight:
    backend = make_local_backend(config, scenario=split, dropout=dropout,
                                 track_factory=track_factory)
    observer = backend.observer
    stream: list = []
    ingest = observer.ingest
    last_capture = [-math.inf]

    def tee(measurement):
        stream.append(measurement)
        if measurement.optical_position is not None and measurement.optical_time_s is not None:
            last_capture[0] = max(last_capture[0], float(measurement.optical_time_s))
        return ingest(measurement)

    observer.ingest = tee
    policy = pd_policy(np.asarray(config.action.maximum_m_s2, dtype=np.float64))
    index, times, truth_p, truth_v, online, age = [], [], [], [], [], []
    try:
        vector, info = backend.reset(seed)
        start = backend.start_s
        verdict = "TIMEOUT"
        while True:
            evaluator = info["evaluator_only"]
            index.append(len(stream))
            times.append(float(observer.last_measurement.time_s))
            truth_p.append(evaluator["relative_position_own_minus_pad_m"])
            truth_v.append(evaluator["relative_velocity_own_minus_pad_m_s"])
            online.append(_relative(observer)[0])
            age.append(times[-1] - last_capture[0])
            step = backend.step(policy(decode(vector)))
            vector, info = step.observation, step.info
            if step.done:
                verdict = str(info["termination"]["verdict"])
                break
        blackout = None
        if backend.blackout is not None:
            b0, duration = backend.blackout
            blackout = (start + b0, start + b0 + duration)
        cadence = None
        if observer.cadence is not None:
            cadence = (config.local_plant.optical_cadence, observer.cadence.camera_dt)
    finally:
        backend.close()
    return Flight(seed=seed, measurements=tuple(stream), decision_index=np.array(index),
                  decision_time=np.array(times), truth_position=np.array(truth_p, float),
                  truth_velocity=np.array(truth_v, float), online_position=np.array(online),
                  sensor_age=np.array(age, float), blackout=blackout, verdict=verdict, cadence=cadence)


# ------------------------------------------------------------------ replay

@dataclass(frozen=True, eq=False)
class Replay:
    position: np.ndarray       # (N, 3) relative estimate at each decision
    velocity: np.ndarray       # (N, 3)
    valid: np.ndarray          # (N,) track initialized
    reacquisitions: int
    confidence: np.ndarray | None   # (N, 3) WMON mean particle confidence (p, v, a)


def replay(flight: Flight, track_factory: Callable[[], Any] | None) -> Replay:
    cadence = None if flight.cadence is None else LocalOpticalCadence(*flight.cadence)
    observer = CurrentObserver(cadence, track_factory=track_factory)
    observer.reset("replay", seed=flight.seed)
    n = len(flight.decision_index)
    position, velocity = np.full((n, 3), np.nan), np.full((n, 3), np.nan)
    valid = np.zeros(n, dtype=bool)
    confidence = np.full((n, 3), np.nan) if track_factory is not None else None
    fed = 0
    for row, upto in enumerate(flight.decision_index):
        while fed < upto:
            observer.ingest(flight.measurements[fed])
            fed += 1
        position[row], velocity[row], valid[row] = _relative(observer)
        if confidence is not None and hasattr(observer.track, "particle_confidence"):
            confidence[row] = observer.track.particle_confidence.mean(axis=1)
    return Replay(position, velocity, valid, observer.reacquisitions, confidence)


def phases(flight: Flight) -> dict[str, np.ndarray]:
    t = flight.decision_time
    if flight.blackout is None:
        dark = np.zeros_like(t, dtype=bool)
        recovery = np.zeros_like(t, dtype=bool)
    else:
        start, end = flight.blackout
        dark = (t >= start) & (t < end)
        recovery = (t >= end) & (t < end + RECOVERY_S)
    every = np.ones_like(t, dtype=bool)
    lost = (flight.sensor_age > LOST_VIEW_AGE_S) & ~dark
    return {"all": every, "tracking": every & ~dark & ~recovery & ~lost, "lost_view": lost,
            "blackout": dark, "recovery": recovery & ~lost}


def errors(flight: Flight, result: Replay) -> tuple[np.ndarray, np.ndarray]:
    """Per-decision |relative position error| (3D) and |relative velocity error| (xy)."""
    ep = np.linalg.norm(result.position - flight.truth_position, axis=1)
    ev = np.linalg.norm((result.velocity - flight.truth_velocity)[:, :2], axis=1)
    ep[~result.valid] = np.nan
    ev[~result.valid] = np.nan
    return ep, ev


def loss(flight: Flight, result: Replay) -> float:
    """MATLAB batchLoss form: position MSE + 0.35 velocity MSE (per axis, valid decisions)."""
    ok = result.valid
    p = (result.position - flight.truth_position)[ok]
    v = (result.velocity - flight.truth_velocity)[ok][:, :2]
    return float(np.mean(p ** 2) + VELOCITY_LOSS_WEIGHT * np.mean(v ** 2))


def summarize(flights: list[Flight], results: list[Replay]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for phase in PHASES:
        ep_all, ev_all = [], []
        for flight, result in zip(flights, results):
            mask = phases(flight)[phase]
            ep, ev = errors(flight, result)
            ep_all.append(ep[mask])
            ev_all.append(ev[mask])
        ep, ev = np.concatenate(ep_all), np.concatenate(ev_all)
        ep, ev = ep[np.isfinite(ep)], ev[np.isfinite(ev)]
        out[phase] = {"decisions": int(ep.size),
                      "position_rmse_m": float(np.sqrt(np.mean(ep ** 2))) if ep.size else None,
                      "position_p95_m": float(np.percentile(ep, 95)) if ep.size else None,
                      "velocity_xy_rmse_m_s": float(np.sqrt(np.mean(ev ** 2))) if ev.size
                      else None}
    out["loss"] = float(np.mean([loss(f, r) for f, r in zip(flights, results)]))
    out["reacquisitions"] = int(sum(r.reacquisitions for r in results))
    return out


# ---------------------------------------------------------------- figures

def figure(path: Path, flights: list[Flight], abg: list[Replay], wmon: list[Replay],
           summary: dict[str, Any], training: Any | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    ax = axes[0, 0]
    if training is not None:
        ax.plot(training.train_loss, label="train")
        ax.plot(training.validation_loss, label="validation")
        ax.axvline(training.best_epoch - 1, color="k", ls=":", lw=1, label="selected")
        ax.set_xlabel("epoch")
        ax.set_ylabel("pos MSE + 0.35 vel MSE")
        ax.set_title("WMON calibration (SPSA + Adam)")
        ax.legend()
    else:
        ax.axis("off")
        ax.text(0.05, 0.5, "no calibration (--calibrate not given)", transform=ax.transAxes)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    names = [n for n in summary["estimators"]]
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        values = [summary["estimators"][name]["test"][phase]["position_rmse_m"] or 0.0
                  for phase in PHASES]
        ax.bar(np.arange(len(PHASES)) + i * width, values, width, label=name)
    ax.set_xticks(np.arange(len(PHASES)) + width * (len(names) - 1) / 2, PHASES)
    ax.set_ylabel("relative position RMSE [m]")
    ax.set_title("Held-out episodes, by phase")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # the held-out episode with a blackout and the longest flight
    candidates = [i for i, f in enumerate(flights) if f.blackout is not None] or \
        list(range(len(flights)))
    k = max(candidates, key=lambda i: len(flights[i].decision_time))
    f = flights[k]
    t = f.decision_time - f.decision_time[0]
    for ax, quantity in ((axes[1, 0], "position"), (axes[1, 1], "confidence")):
        if f.blackout is not None:
            ax.axvspan(f.blackout[0] - f.decision_time[0], f.blackout[1] - f.decision_time[0],
                       color="tab:orange", alpha=0.15, label="vision blackout")
        ax.grid(alpha=0.3)
        ax.set_xlabel("time since handover [s]")
    ep_abg, _ = errors(f, abg[k])
    ep_wmon, _ = errors(f, wmon[k])
    axes[1, 0].plot(t, ep_abg, label="ABG (registered)")
    axes[1, 0].plot(t, ep_wmon, label="WMON")
    axes[1, 0].set_ylabel("|relative position error| [m]")
    axes[1, 0].set_title(f"seed {f.seed} ({f.verdict})")
    axes[1, 0].legend()
    if wmon[k].confidence is not None:
        for j, label in enumerate(("position particle", "velocity particle", "acceleration")):
            axes[1, 1].plot(t, wmon[k].confidence[:, j], label=label)
    axes[1, 1].set_ylim(-0.05, 1.05)
    axes[1, 1].set_ylabel("WMON mean particle confidence")
    axes[1, 1].set_title("Formation / dissolution of information particles")
    axes[1, 1].legend()
    fig.suptitle("WMON pad track vs registered ABG track (open-loop replay, local plant /12)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# -------------------------------------------------------------------- main

def closed_loop(config: Any, seeds: list[int], split: str, dropout: VisionBlackout,
                factories: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for name, factory in factories.items():
        verdicts: dict[str, int] = {}
        for seed in seeds:
            verdict = record(config, seed, split, dropout, track_factory=factory).verdict
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
        out[name] = {"verdicts": verdicts, "success": verdicts.get("SUCCESS", 0),
                     "episodes": len(seeds)}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--episodes", type=int, default=36)
    parser.add_argument("--seed0", type=int, default=910_000,
                        help="first seed; episodes use seed0 .. seed0+episodes-1")
    parser.add_argument("--split", default="in_distribution",
                        choices=("in_distribution", "composition"))
    parser.add_argument("--blackout", default="1.0,3.0,6.0,0.5,2.0",
                        help="probability,start_min,start_max,duration_min,duration_max (s)")
    parser.add_argument("--settings", type=Path, default=None,
                        help="WMONTrackSettings JSON (default: MATLAB-trained coefficients)")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--validation-fraction", type=float, default=1 / 6)
    parser.add_argument("--closed-loop", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    started = time.time()
    config = load_config()
    values = [float(v) for v in args.blackout.split(",")]
    if len(values) != 5:
        parser.error("--blackout needs five comma-separated numbers")
    dropout = VisionBlackout(*values)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output or ROOT / "results" / f"wmon_pad_track_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    base = WMONTrackSettings.load(args.settings) if args.settings else WMONTrackSettings()

    seeds = list(range(args.seed0, args.seed0 + args.episodes))
    print(f"recording {len(seeds)} local-plant flights ({args.split}, blackout {values}) ...")
    flights = [record(config, seed, args.split, dropout) for seed in seeds]
    n_train = max(1, int(round(args.train_fraction * len(flights))))
    n_val = max(1, int(round(args.validation_fraction * len(flights))))
    train, val, test = (flights[:n_train], flights[n_train:n_train + n_val],
                        flights[n_train + n_val:])
    if not test:
        parser.error("no held-out episodes left; raise --episodes")

    abg_test = [replay(f, None) for f in test]
    mismatch = max(float(np.nanmax(np.abs(r.position - f.online_position)))
                   for f, r in zip(test, abg_test))
    if mismatch > 1e-12:
        raise RuntimeError(f"ABG replay differs from the online observer by {mismatch:g} m")
    print(f"ABG replay reproduces the online observer (max |diff| {mismatch:.1e} m)")

    estimators: dict[str, WMONTrackSettings] = {"WMON (MATLAB coefficients)": base}
    training = None
    if args.calibrate:
        def batch_loss(theta: np.ndarray, ids) -> float:
            settings = replace(base, params=WMONParams.from_theta(theta))
            factory = wmon_track_factory(settings)
            return float(np.mean([loss(train[i], replay(train[i], factory)) for i in ids]))

        def validation_loss(theta: np.ndarray) -> float:
            factory = wmon_track_factory(replace(base, params=WMONParams.from_theta(theta)))
            return float(np.mean([loss(f, replay(f, factory)) for f in val]))

        print(f"calibrating on {len(train)} train / {len(val)} validation episodes ...")
        training = spsa_adam(batch_loss, len(train), base.params.to_theta(),
                             epochs=args.epochs, batch_size=args.batch_size,
                             validation_loss=validation_loss,
                             rng=np.random.default_rng(20261110), log=print)
        calibrated = replace(base, params=WMONParams.from_theta(training.theta))
        estimators["WMON (calibrated)"] = calibrated
        (out / "wmon_track_settings_calibrated.json").write_text(json.dumps({
            "settings": calibrated.to_dict(),
            "provenance": {"tool": "tools/wmon_pad_track_eval.py", "created": stamp,
                           "flight": "spatial_local_v12 (spatial-reference/12)",
                           "split": args.split, "blackout": values,
                           "train_seeds": [f.seed for f in train],
                           "validation_seeds": [f.seed for f in val],
                           "best_epoch": training.best_epoch, "epochs": args.epochs,
                           "theta_names": THETA_NAMES}}, indent=2), encoding="utf-8")

    results = {"ABG (registered)": abg_test}
    for name, settings in estimators.items():
        factory = wmon_track_factory(settings)
        results[name] = [replay(f, factory) for f in test]
    summary: dict[str, Any] = {
        "created": stamp, "split": args.split, "blackout": values,
        "seeds": {"train": [f.seed for f in train], "validation": [f.seed for f in val],
                  "test": [f.seed for f in test]},
        "recorded_verdicts": {v: sum(f.verdict == v for f in flights)
                              for v in sorted({f.verdict for f in flights})},
        "estimators": {name: {"test": summarize(test, res)} for name, res in results.items()},
        "settings": {name: s.to_dict() for name, s in estimators.items()},
        "note": ("open-loop replay of the same Measurement streams; the flights were "
                 "driven by a scripted PD on the registered ABG observation"),
    }
    if training is not None:
        summary["calibration"] = {"best_epoch": training.best_epoch,
                                  "train_loss": training.train_loss.tolist(),
                                  "validation_loss": training.validation_loss.tolist()}
    if args.closed_loop:
        print("closed loop on the held-out seeds ...")
        factories = {"ABG (registered)": None}
        factories.update({name: wmon_track_factory(s) for name, s in estimators.items()})
        summary["closed_loop"] = closed_loop(config, [f.seed for f in test], args.split,
                                             dropout, factories)
    best_wmon = list(estimators)[-1]
    figure(out / "wmon_pad_track.png", test, results["ABG (registered)"], results[best_wmon],
           summary, training)
    summary["wall_s"] = time.time() - started
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nheld-out episodes: {len(test)}  (output {out})")
    for name, entry in summary["estimators"].items():
        row = entry["test"]
        print(f"  {name:28s} " + "  ".join(
            f"{phase} {row[phase]['position_rmse_m']:.3f} m" if row[phase]["position_rmse_m"]
            is not None else f"{phase} -" for phase in PHASES)
            + f"  | vel(all) {row['all']['velocity_xy_rmse_m_s']:.3f} m/s"
            + f"  reacq {row['reacquisitions']}")
    if "closed_loop" in summary:
        for name, entry in summary["closed_loop"].items():
            print(f"  closed loop {name:28s} {entry['verdicts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
