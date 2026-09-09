"""Generalization sweeps over the disturbances the experiment varies."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..config import Config
from ..env import run_episode
from ..expert import PolicySpec
from .compare import LABELS, write_table

__all__ = ["wind_sweep", "pad_sweep", "gnss_sweep", "battery_sweep"]


def _run(agent, potential, seed: int, cfg: Config, episode_monitor=None):
    policy = PolicySpec("ppo", agent=agent, deterministic=True)
    return run_episode(policy, "sparse", potential, seed, cfg, monitor=episode_monitor)


def wind_sweep(baseline_agent, proposed_agent, potential, cfg: Config,
               *, episode_monitor=None) -> list[dict[str, Any]]:
    """Isaac wind-intensity generalization."""
    rows: list[dict[str, Any]] = []
    n = int(cfg.eval.sweep_episodes)
    for si, scale in enumerate(cfg.eval.wind_scales, start=1):
        scaled = cfg.derive(**{"external.wind_scale": float(scale)})
        for agent, label in zip((baseline_agent, proposed_agent), LABELS):
            metrics = [_run(agent, potential, int(cfg.eval.seed0) + 10000 * si + i,
                            scaled, episode_monitor).metrics for i in range(n)]
            row = {
                "Policy": label, "WindScale": float(scale),
                "SuccessRate": float(np.mean([m["success"] for m in metrics])),
                "UnsafeRate": float(np.mean([m["unsafe"] for m in metrics])),
                "TimeoutRate": float(np.mean([m["timeout"] for m in metrics])),
                "MeanXY": float(np.mean([m["touchdown_xy"] for m in metrics])),
                "MeanMaxTiltDeg": float(np.degrees(np.mean([m["max_tilt"] for m in metrics]))),
            }
            rows.append(row)
            print(f"Isaac wind x{scale:.2f} | {label:<14} | "
                  f"success {100 * row['SuccessRate']:.1f}% | "
                  f"unsafe {100 * row['UnsafeRate']:.1f}% | "
                  f"timeout {100 * row['TimeoutRate']:.1f}%")
    write_table(rows, Path(cfg.paths.results) / "wind_generalization_sweep.csv")
    return rows


def pad_sweep(baseline_agent, proposed_agent, potential, cfg: Config,
              *, episode_monitor=None) -> list[dict[str, Any]]:
    """How fast a deck each policy can still land on.

    The seed picks the same point in the deck-motion distribution at every
    scale, so the comparison is paired across speeds as well as across
    policies. Scale 0.0 is the static-deck control condition.
    """
    rows: list[dict[str, Any]] = []
    n = int(cfg.eval.sweep_episodes)
    for si, scale in enumerate(cfg.eval.pad_scales, start=1):
        scaled = cfg.derive(**{"external.pad_scale": float(scale)})
        for agent, label in zip((baseline_agent, proposed_agent), LABELS):
            metrics = [_run(agent, potential, int(cfg.eval.seed0) + 20000 * si + i,
                            scaled, episode_monitor).metrics for i in range(n)]
            row = {
                "Policy": label, "PadScale": float(scale),
                "MeanPadSpeed": float(np.mean([m["pad_speed_mean"] for m in metrics])),
                "SuccessRate": float(np.mean([m["success"] for m in metrics])),
                "UnsafeRate": float(np.mean([m["unsafe"] for m in metrics])),
                "TimeoutRate": float(np.mean([m["timeout"] for m in metrics])),
                "DepletedRate": float(np.mean([m["depleted"] for m in metrics])),
                "MeanXY": float(np.mean([m["touchdown_xy"] for m in metrics])),
                "MeanRelSpeedXY": float(np.mean([m["touchdown_rel_speed_xy"]
                                                 for m in metrics])),
            }
            rows.append(row)
            print(f"Pad x{scale:.2f} ({row['MeanPadSpeed']:.2f} m/s) | {label:<14} | "
                  f"success {100 * row['SuccessRate']:.1f}% | "
                  f"unsafe {100 * row['UnsafeRate']:.1f}% | "
                  f"timeout {100 * row['TimeoutRate']:.1f}% | "
                  f"depleted {100 * row['DepletedRate']:.1f}% | "
                  f"rel speed {row['MeanRelSpeedXY']:.2f} m/s")
    write_table(rows, Path(cfg.paths.results) / "pad_speed_sweep.csv")
    return rows


def gnss_sweep(baseline_agent, proposed_agent, potential, cfg: Config,
               *, episode_monitor=None) -> list[dict[str, Any]]:
    """How much of the canyon's GNSS degradation each policy survives.

    The scale multiplies the error mechanisms -- how many blocked satellites
    are still tracked through a reflection, how much excess delay they carry,
    how strong the diffuse multipath is -- and moves no building. Scale 0.0 is
    therefore open sky in the *same* city: the facades still hide the markers
    and still channel the wind, so what this isolates is the fix and nothing
    else. Which is the whole question: how much of the landing was being flown
    on GNSS that nobody had checked.
    """
    rows: list[dict[str, Any]] = []
    n = int(cfg.eval.sweep_episodes)
    for si, scale in enumerate(cfg.eval.gnss_scales, start=1):
        scaled = cfg.derive(**{"external.gnss_scale": float(scale)})
        for agent, label in zip((baseline_agent, proposed_agent), LABELS):
            metrics = [_run(agent, potential, int(cfg.eval.seed0) + 40000 * si + i,
                            scaled, episode_monitor).metrics for i in range(n)]
            row = {
                "Policy": label, "GnssScale": float(scale),
                "MeanGnssQuality": float(np.mean([m["gnss_quality_mean"] for m in metrics])),
                "MeanNlosFraction": float(np.mean([m["gnss_nlos_mean"] for m in metrics])),
                # What the degradation actually did to the pose the policy flew
                # on, measured against the simulator's own geometry.
                "MeanEstimateErrorM": float(np.mean([m["estimate_error_mean_m"]
                                                     for m in metrics])),
                "MaxEstimateErrorM": float(np.max([m["estimate_error_max_m"]
                                                   for m in metrics])),
                "SuccessRate": float(np.mean([m["success"] for m in metrics])),
                "UnsafeRate": float(np.mean([m["unsafe"] for m in metrics])),
                "TimeoutRate": float(np.mean([m["timeout"] for m in metrics])),
                "MeanXY": float(np.mean([m["touchdown_xy"] for m in metrics])),
            }
            rows.append(row)
            print(f"GNSS x{scale:.2f} (q {row['MeanGnssQuality']:.2f}, "
                  f"err {row['MeanEstimateErrorM']:.1f} m) | {label:<14} | "
                  f"success {100 * row['SuccessRate']:.1f}% | "
                  f"unsafe {100 * row['UnsafeRate']:.1f}% | "
                  f"timeout {100 * row['TimeoutRate']:.1f}%")
    write_table(rows, Path(cfg.paths.results) / "gnss_degradation_sweep.csv")
    return rows


def battery_sweep(baseline_agent, proposed_agent, potential, cfg: Config,
                  *, episode_monitor=None) -> list[dict[str, Any]]:
    """Success against how much reserve the episode started with.

    The reserve is seeded per episode by Isaac, not commanded, so this bins
    finished episodes by their starting reserve instead of sweeping a scale --
    which is the honest way to read a factor the initial condition owns.
    """
    edges = [float(v) for v in cfg.eval.battery_bins_s]
    n = int(cfg.eval.sweep_episodes) * len(cfg.eval.pad_scales)
    rows: list[dict[str, Any]] = []
    for agent, label in zip((baseline_agent, proposed_agent), LABELS):
        logs = [_run(agent, potential, int(cfg.eval.seed0) + 30000 + i, cfg,
                     episode_monitor) for i in range(n)]
        # Starting reserve, reconstructed from the pack's own currency.
        start = np.asarray([log.hover_seconds_left[0] for log in logs])
        metrics = [log.metrics for log in logs]
        success = np.asarray([m["success"] for m in metrics])
        depleted = np.asarray([m["depleted"] for m in metrics])
        unsafe = np.asarray([m["unsafe"] for m in metrics])
        energy = np.asarray([m["energy_j"] for m in metrics])
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (start >= lo) & (start < hi)
            row = {"Policy": label, "ReserveLowS": lo, "ReserveHighS": hi,
                   "Episodes": int(mask.sum())}
            for name, source in (("SuccessRate", success), ("UnsafeRate", unsafe),
                                 ("DepletedRate", depleted), ("MeanEnergyJ", energy)):
                row[name] = float(source[mask].mean()) if mask.any() else float("nan")
            rows.append(row)
            print(f"Reserve {lo:4.1f}-{hi:4.1f} s | {label:<14} | n={row['Episodes']:3d} "
                  f"| success {100 * row['SuccessRate']:5.1f}% "
                  f"| depleted {100 * row['DepletedRate']:5.1f}%")
    write_table(rows, Path(cfg.paths.results) / "battery_reserve_bins.csv")
    return rows
