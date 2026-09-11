"""Paired common-random-number comparison on the external stack."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..config import Config
from ..env import run_episode
from ..expert import PolicySpec

__all__ = ["evaluate_policy", "compare_policies", "write_table", "wilson_interval",
           "METRIC_FIELDS"]

# Reported name -> field in EpisodeLog.metrics. Beyond the fixed-pad metrics
# this carries horizontal touchdown speed relative to the moving deck, the deck
# speed, and the measured episode energy.
METRIC_FIELDS = (
    ("Success", "success"), ("Unsafe", "unsafe"), ("Timeout", "timeout"),
    ("Depleted", "depleted"), ("TouchdownXY", "touchdown_xy"),
    ("TouchdownVz", "touchdown_vz"),
    ("TouchdownRelSpeedXY", "touchdown_rel_speed_xy"),
    ("MaxTiltDeg", "max_tilt"), ("EnergyJ", "energy_j"),
    ("MaxAeroForce", "max_aero_force"), ("PadSpeedMean", "pad_speed_mean"),
    ("BatteryReserveFinal", "battery_reserve_final"),
)
RATE_METRICS = {"Success", "Unsafe", "Timeout", "Depleted"}
LABELS = ("Manual", "Ontology-RGAT")


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Binomial confidence interval that remains meaningful near 0% and 100%."""
    if total <= 0:
        return 0.0, 1.0
    p = float(successes) / float(total)
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    radius = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return float(centre - radius / denominator), float(centre + radius / denominator)


def write_table(rows: Sequence[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return path
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _metric_row(metrics: dict[str, Any]) -> np.ndarray:
    values = []
    for name, field in METRIC_FIELDS:
        value = float(metrics[field])
        if name == "MaxTiltDeg":
            value = float(np.degrees(value))
        values.append(value)
    return np.asarray(values)


def evaluate_policy(agent, label: str, potential, cfg: Config, *,
                    episode_monitor=None) -> dict[str, Any]:
    """Deterministic paired-seed Monte Carlo under the common sparse task reward.

    Both arms are scored by the same reward, whatever they were trained on:
    comparing a shaped return against an unshaped one would compare two
    different quantities.
    """
    n = int(cfg.eval.episodes)
    policy = PolicySpec("ppo", agent=agent, deterministic=True)
    metrics: list[dict[str, Any]] = []
    representative = None
    for i in range(n):
        seed = int(cfg.eval.seed0) + i
        if episode_monitor is not None:
            episode_monitor.reset(f"eval {label} {i + 1}/{n}")
        log = run_episode(policy, "sparse", potential, seed, cfg,
                          monitor=episode_monitor)
        metrics.append(log.metrics)
        if representative is None:
            representative = log
        print(f"Eval {label:<14} {i + 1:3d}/{n} | success={int(log.metrics['success'])} "
              f"| xy={log.metrics['touchdown_xy']:.3f} "
              f"| vz={log.metrics['touchdown_vz']:.3f} "
              f"| rel={log.metrics['touchdown_rel_speed_xy']:.3f}")
    return {"label": label, "metrics": metrics, "representative": representative}


def compare_policies(baseline_agent, proposed_agent, potential, cfg: Config, *,
                     episode_monitor=None) -> dict[str, Any]:
    """Run both arms on the same seeds and report the paired difference.

    Common random numbers: episode ``i`` of each arm gets the same seed, so the
    deck trajectory, the wind and the starting energy are shared and the
    difference is attributable to the policy rather than to the draw.
    """
    baseline = evaluate_policy(baseline_agent, LABELS[0], potential, cfg,
                               episode_monitor=episode_monitor)
    proposed = evaluate_policy(proposed_agent, LABELS[1], potential, cfg,
                               episode_monitor=episode_monitor)

    per_episode: list[dict[str, Any]] = []
    values = {}
    for arm in (baseline, proposed):
        rows = np.asarray([_metric_row(m) for m in arm["metrics"]])
        values[arm["label"]] = rows
        for metrics, row in zip(arm["metrics"], rows):
            entry = {"Policy": arm["label"], "Seed": int(metrics["seed"])}
            entry.update({name: float(v) for (name, _), v in zip(METRIC_FIELDS, row)})
            per_episode.append(entry)

    summary: list[dict[str, Any]] = []
    for label in LABELS:
        row: dict[str, Any] = {"Policy": label}
        for index, (name, _) in enumerate(METRIC_FIELDS):
            key = f"{name}Rate" if name in RATE_METRICS else f"Mean{name}"
            row[key] = float(values[label][:, index].mean())
        row["Episodes"] = int(values[label].shape[0])
        successes = int(np.count_nonzero(values[label][:, 0] > 0.5))
        low, high = wilson_interval(successes, row["Episodes"])
        row["SuccessCount"] = successes
        row["SuccessCI95Low"] = low
        row["SuccessCI95High"] = high
        summary.append(row)

    # Paired differences, proposed minus manual, with a normal-approximation
    # interval on the mean of the per-seed difference.
    paired: list[dict[str, Any]] = []
    manual, ontology = values[LABELS[0]], values[LABELS[1]]
    n = min(manual.shape[0], ontology.shape[0])
    for index, (name, _) in enumerate(METRIC_FIELDS):
        difference = ontology[:n, index] - manual[:n, index]
        mu = float(difference.mean())
        se = float(difference.std(ddof=0) / np.sqrt(max(1, difference.size)))
        paired.append({"Metric": name, "MeanDifference": mu,
                       "CI95Low": mu - 1.96 * se, "CI95High": mu + 1.96 * se})

    results = {"baseline": baseline, "proposed": proposed,
               "per_episode": per_episode, "summary": summary, "paired": paired,
               "potential": potential}
    write_table(paired, Path(cfg.paths.results) / "paired_difference_ci.csv")
    return results
