"""Paired statistics and publication outputs for the Shin benchmark."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


METRICS = (
    "paper_success", "strict_success", "pad_contact", "unsafe_pad_contact",
    "landing_gate_contact", "landing_gate_position",
    "landing_gate_vertical_speed", "landing_gate_relative_horizontal_speed",
    "landing_gate_attitude", "landing_gate_angular_rate",
    "position_rmse", "velocity_rmse",
    "touchdown_lateral_error", "touchdown_vertical_velocity",
    "touchdown_relative_horizontal_velocity", "touchdown_tilt",
    "touchdown_angular_rate", "fov_loss_fraction", "longest_visual_loss_s",
    "visual_loss_estimation_error", "episode_return", "touchdown_time_s",
    "training_sample_efficiency", "curriculum_level",
)


def _write_csv(path: Path, rows: list[dict], fieldnames=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fieldnames or (rows[0].keys() if rows else ()))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _ci(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    half = 1.96 * std / np.sqrt(values.size)
    return mean, std, mean - half, mean + half


def summarize(records: list[dict]) -> list[dict]:
    groups = {}
    for row in records:
        groups.setdefault((row["method"], row["scenario"]), []).append(row)
    output = []
    for (method, scenario), rows in sorted(groups.items()):
        item = {"method": method, "scenario": scenario, "episodes": len(rows)}
        for metric in METRICS:
            values = [float(row[metric]) for row in rows if metric in row]
            mean, std, low, high = _ci(values)
            item.update({f"{metric}_mean": mean, f"{metric}_std": std,
                         f"{metric}_ci95_low": low, f"{metric}_ci95_high": high})
        output.append(item)
    return output


def paired_bootstrap(records: list[dict], baseline="shin2026", draws=10000,
                     seed=8675309) -> list[dict]:
    index = {(row["method"], row["scenario"], int(row["seed"])): row
             for row in records}
    methods = sorted({row["method"] for row in records if row["method"] != baseline})
    scenarios = sorted({row["scenario"] for row in records})
    rng = np.random.default_rng(seed)
    output = []
    for method in methods:
        for scenario in scenarios:
            seeds = sorted({key[2] for key in index if key[:2] == (baseline, scenario)}
                           & {key[2] for key in index if key[:2] == (method, scenario)})
            for metric in METRICS:
                differences = np.array([
                    float(index[(method, scenario, value)][metric])
                    - float(index[(baseline, scenario, value)][metric])
                    for value in seeds
                    if metric in index[(method, scenario, value)]
                    and metric in index[(baseline, scenario, value)]], dtype=float)
                if not differences.size:
                    continue
                # Full mode has 10,000 paired random-walk episodes. Allocating
                # draws x pairs at once would require ~800 MB per metric.
                bootstrap = np.empty(draws, dtype=float)
                for start in range(0, draws, 256):
                    stop = min(start + 256, draws)
                    selections = rng.integers(
                        0, differences.size, size=(stop - start, differences.size))
                    bootstrap[start:stop] = differences[selections].mean(axis=1)
                output.append({
                    "baseline": baseline, "method": method, "scenario": scenario,
                    "metric": metric, "pairs": int(differences.size),
                    "mean_difference": float(differences.mean()),
                    "ci95_low": float(np.quantile(bootstrap, 0.025)),
                    "ci95_high": float(np.quantile(bootstrap, 0.975)),
                    "two_sided_p": float(2 * min(np.mean(bootstrap <= 0),
                                                  np.mean(bootstrap >= 0))),
                })
    return output


def write_benchmark_outputs(records: list[dict], output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(records)
    paired = paired_bootstrap(records)
    _write_csv(output_dir / "scenario_success_rates.csv", summary)
    _write_csv(output_dir / "paired_comparison.csv", paired)
    families = {
        "training_curves.csv": ("episode", "episode_return", "paper_success",
                                "training_sample_efficiency", "curriculum_level"),
        "state_estimation_metrics.csv": ("position_rmse", "velocity_rmse",
                                         "visual_loss_estimation_error"),
        "touchdown_metrics.csv": ("paper_success", "strict_success",
                                  "pad_contact", "unsafe_pad_contact",
                                  "landing_gate_contact", "landing_gate_position",
                                  "landing_gate_vertical_speed",
                                  "landing_gate_relative_horizontal_speed",
                                  "landing_gate_attitude",
                                  "landing_gate_angular_rate",
                                  "touchdown_lateral_error", "touchdown_vertical_velocity",
                                  "touchdown_relative_horizontal_velocity", "touchdown_tilt",
                                  "touchdown_angular_rate", "touchdown_time_s"),
        "visual_observability_metrics.csv": ("fov_loss_fraction",
                                             "longest_visual_loss_s",
                                             "visual_loss_estimation_error"),
    }
    identity = ("method", "scenario", "seed")
    for filename, fields in families.items():
        rows = [{key: row.get(key, "") for key in identity + fields}
                for row in records]
        _write_csv(output_dir / filename, rows, identity + fields)
    _write_publication_table(summary, output_dir)
    figures = _write_plots(records, output_dir)
    return {"summary": summary, "paired": paired, "figures": figures}


def _write_publication_table(summary, output_dir: Path) -> None:
    methods = ("shin2026", "sparse", "manual_no_active", "ontoreward",
               "ontoreward_plus_active")
    rows = []
    for method in methods:
        values = [row for row in summary if row["method"] == method]
        def avg(field):
            data = [float(row[field]) for row in values if np.isfinite(float(row[field]))]
            return float(np.mean(data)) if data else float("nan")
        rows.append({
            "Method": method, "RL": "PPO", "Perception": "keypoint approximation",
            "Temporal estimator": "LSTM", "Reward": method,
            "UGV communication": "none", "Action": "velocity+yaw-rate",
            "Success rate": avg("paper_success_mean"),
            "Position RMSE": avg("position_rmse_mean"),
            "Velocity RMSE": avg("velocity_rmse_mean"),
            "Visual-loss robustness": avg("visual_loss_estimation_error_mean"),
            "Strict safe-landing rate": avg("strict_success_mean"),
        })
    _write_csv(output_dir / "publication_table.csv", rows)
    header = list(rows[0])
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(str(row[key]) for key in header) + " |" for row in rows)
    (output_dir / "publication_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plots(records, output_dir: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    specifications = {
        "training_success_curve.png": ("episode", "paper_success"),
        "curriculum_progression.png": ("episode", "curriculum_level"),
        "scenario_success_rates.png": ("seed", "paper_success"),
        "relative_state_estimation.png": ("seed", "position_rmse"),
        "estimation_error_during_visual_loss.png": ("seed", "visual_loss_estimation_error"),
        "landing_trajectories.png": ("seed", "touchdown_lateral_error"),
        "touchdown_error_distribution.png": ("seed", "touchdown_vertical_velocity"),
        "fov_target_heatmap.png": ("seed", "fov_loss_fraction"),
        "reward_component_breakdown.png": ("seed", "episode_return"),
        "shin_vs_ontoreward_learning_curve.png": ("episode", "paper_success"),
    }
    paths = []
    for filename, (x_key, y_key) in specifications.items():
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for method in sorted({row["method"] for row in records}):
            rows = [row for row in records if row["method"] == method
                    and x_key in row and y_key in row]
            if rows:
                ax.plot([float(row[x_key]) for row in rows],
                        [float(row[y_key]) for row in rows], ".", alpha=0.6,
                        label=method)
        ax.set(xlabel=x_key, ylabel=y_key)
        if ax.lines:
            ax.legend(fontsize=7)
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path))
    return paths
