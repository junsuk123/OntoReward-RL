"""Reward-independent reporting for the primary three-pipeline comparison."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


PHYSICAL_METRICS = (
    "paper_success", "strict_success", "crash_failure",
    "touchdown_lateral_error", "touchdown_vertical_velocity",
    "touchdown_relative_horizontal_velocity", "touchdown_tilt",
    "touchdown_angular_rate", "fov_loss_fraction",
    "longest_visual_loss_s", "touchdown_time_s",
)
PRIMARY_PIPELINES = ("shin_se", "no_se", "onto_no_se")


def _write_csv(path: Path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    names = list(fieldnames or sorted({key for row in rows for key in row}))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap_mean(values, *, draws=10000, seed=90421):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 256):
        stop = min(draws, start + 256)
        selected = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[selected].mean(axis=1)
    return (float(values.mean()), float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)))


def physical_summary(records):
    grouped = {}
    for row in records:
        grouped.setdefault((row["pipeline"], row["scenario"]), []).append(row)
    result = []
    for (pipeline, scenario), rows in sorted(grouped.items()):
        summary = {"pipeline": pipeline, "scenario": scenario,
                   "episodes": len(rows)}
        for metric in PHYSICAL_METRICS:
            values = [float(row[metric]) for row in rows if row.get(metric, "") != ""]
            mean, low, high = _bootstrap_mean(values)
            summary.update({f"{metric}_mean": mean,
                            f"{metric}_ci95_low": low,
                            f"{metric}_ci95_high": high})
        result.append(summary)
    return result


def paired_confidence_intervals(records, *, draws=10000, seed=8128):
    index = {(row["pipeline"], row["scenario"], int(row["seed"])): row
             for row in records}
    comparisons = (("onto_no_se", "shin_se"), ("onto_no_se", "no_se"))
    scenarios = sorted({row["scenario"] for row in records})
    output = []
    for proposed, baseline in comparisons:
        for scenario in scenarios:
            proposed_seeds = {key[2] for key in index if key[:2] == (proposed, scenario)}
            baseline_seeds = {key[2] for key in index if key[:2] == (baseline, scenario)}
            seeds = sorted(proposed_seeds & baseline_seeds)
            for metric in PHYSICAL_METRICS:
                difference = [
                    float(index[(proposed, scenario, item)][metric])
                    - float(index[(baseline, scenario, item)][metric])
                    for item in seeds
                    if metric in index[(proposed, scenario, item)]
                    and metric in index[(baseline, scenario, item)]
                ]
                if not difference:
                    continue
                mean, low, high = _bootstrap_mean(
                    difference, draws=draws, seed=seed + len(output))
                output.append({
                    "comparison": f"{proposed} - {baseline}",
                    "proposed": proposed, "baseline": baseline,
                    "scenario": scenario, "metric": metric,
                    "paired_episodes": len(difference),
                    "mean_difference": mean, "ci95_low": low,
                    "ci95_high": high,
                })
    return output


def _rolling_success(rows, window=20):
    rows = [row for row in rows
            if row.get("optimization_phase", "ppo") == "ppo"]
    ordered = sorted(rows, key=lambda row: int(float(
        row.get("ppo_episode", row["episode"]))))
    successes = np.asarray([float(row["paper_success"]) for row in ordered])
    episodes = np.asarray([int(float(row.get("ppo_episode", row["episode"])))
                           for row in ordered])
    steps = np.asarray([int(float(row.get("ppo_environment_steps", 0)))
                        for row in ordered])
    average = np.asarray([
        successes[max(0, index - window + 1):index + 1].mean()
        for index in range(successes.size)])
    return episodes, steps, average


def learning_efficiency(training_records, *, reward_design_episodes=0,
                        reward_design_steps=0, estimator_warmup_episodes=0,
                        estimator_warmup_steps=0):
    output = []
    for pipeline in PRIMARY_PIPELINES:
        rows = [row for row in training_records if row.get("pipeline") == pipeline]
        if not rows:
            continue
        episodes, steps, success = _rolling_success(rows)
        if episodes.size == 0:
            continue
        auc = float(np.trapz(success, episodes) / max(float(episodes[-1] - episodes[0]), 1.0))
        extra_episodes = (
            int(reward_design_episodes) if pipeline == "onto_no_se" else
            int(estimator_warmup_episodes) if pipeline == "shin_se" else 0)
        extra_steps = (
            int(reward_design_steps) if pipeline == "onto_no_se" else
            int(estimator_warmup_steps) if pipeline == "shin_se" else 0)
        item = {
            "pipeline": pipeline, "ppo_episodes": int(episodes[-1]),
            "ppo_environment_steps": int(steps[-1]),
            "reward_design_episodes": (int(reward_design_episodes)
                                       if pipeline == "onto_no_se" else 0),
            "reward_design_environment_steps": (int(reward_design_steps)
                                                  if pipeline == "onto_no_se" else 0),
            "estimator_warmup_episodes": (int(estimator_warmup_episodes)
                                           if pipeline == "shin_se" else 0),
            "estimator_warmup_environment_steps": (int(estimator_warmup_steps)
                if pipeline == "shin_se" else 0),
            "total_environment_episodes": int(episodes[-1]) + extra_episodes,
            "total_environment_steps": int(steps[-1]) + extra_steps,
            "success_curve_auc": auc,
        }
        for threshold in (0.7, 0.8, 0.9):
            reached = np.flatnonzero(success >= threshold)
            item[f"episodes_to_{int(threshold * 100)}pct"] = (
                int(episodes[reached[0]]) if reached.size else "not reached")
        output.append(item)
    return output


def _publication_table(summary, efficiency):
    rows = []
    costs = {row["pipeline"]: row for row in efficiency}
    for pipeline in PRIMARY_PIPELINES:
        values = [row for row in summary if row["pipeline"] == pipeline]
        if not values:
            continue
        def average(field):
            numbers = [float(row[field]) for row in values
                       if field in row and np.isfinite(float(row[field]))]
            return float(np.mean(numbers)) if numbers else np.nan
        row = {
            "pipeline": pipeline,
            "state_estimation_supervision": pipeline == "shin_se",
            "direct_semantic_rgat": pipeline == "onto_no_se",
            "success_rate": average("paper_success_mean"),
            "strict_success_rate": average("strict_success_mean"),
            "crash_rate": average("crash_failure_mean"),
            "touchdown_lateral_error_m": average("touchdown_lateral_error_mean"),
            "touchdown_vertical_velocity_m_s": average(
                "touchdown_vertical_velocity_mean"),
            "fov_loss_fraction": average("fov_loss_fraction_mean"),
            "landing_time_s": average("touchdown_time_s_mean"),
        }
        row.update({key: value for key, value in costs.get(pipeline, {}).items()
                    if key != "pipeline"})
        rows.append(row)
    return rows


def _write_markdown(path: Path, rows):
    if not rows:
        path.write_text("No completed evaluation records.\n", encoding="utf-8")
        return
    fields = list(rows[0])
    lines = ["| " + " | ".join(fields) + " |",
             "| " + " | ".join("---" for _ in fields) + " |"]
    lines.extend("| " + " | ".join(str(row.get(key, "")) for key in fields) + " |"
                 for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_figures(records, training_records, figures_dir: Path,
                   reward_design_episodes=0, reward_design_steps=0,
                   estimator_warmup_episodes=0, estimator_warmup_steps=0):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    colors = ("#0072BD", "#D95319", "#EDB120")

    def save(name):
        path = figures_dir / name
        plt.gcf().tight_layout()
        plt.gcf().savefig(path, dpi=180)
        plt.close(plt.gcf())
        paths.append(str(path))

    for name, metric, ylabel in (
            ("success_rate.png", "paper_success", "Landing success"),
            ("touchdown_lateral_error.png", "touchdown_lateral_error", "Lateral error (m)"),
            ("touchdown_velocity.png", "touchdown_vertical_velocity", "Vertical velocity (m/s)"),
            ("fov_loss.png", "fov_loss_fraction", "FOV loss fraction"),
            ("landing_time.png", "touchdown_time_s", "Landing time (s)")):
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        data = [[float(row[metric]) for row in records if row["pipeline"] == pipeline]
                for pipeline in PRIMARY_PIPELINES]
        ax.boxplot(data, labels=PRIMARY_PIPELINES, showmeans=True)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=.45)
        save(name)

    for filename, x_kind in (("success_learning_curve.png", "episode"),
                             ("sample_efficiency_ppo_only.png", "steps"),
                             ("sample_efficiency_total_interactions.png", "total")):
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for color, pipeline in zip(colors, PRIMARY_PIPELINES):
            rows = [row for row in training_records if row.get("pipeline") == pipeline]
            if not rows:
                continue
            episodes, steps, success = _rolling_success(rows)
            if x_kind == "episode":
                x = episodes
                xlabel = "PPO episodes"
            elif x_kind == "steps":
                x = steps
                xlabel = "PPO environment steps"
            else:
                shift = (reward_design_steps if pipeline == "onto_no_se" else
                         estimator_warmup_steps if pipeline == "shin_se" else 0)
                x = steps + shift
                xlabel = "Total environment steps (pre-training + PPO)"
            ax.plot(x, success, color=color, label=pipeline)
        ax.set(xlabel=xlabel, ylabel="Rolling landing success")
        ax.grid(True, linestyle=":", alpha=.45)
        ax.legend()
        save(filename)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    index = {(row["pipeline"], row["scenario"], int(row["seed"])): row
             for row in records}
    labels, values = [], []
    for baseline in ("shin_se", "no_se"):
        diff = []
        for key, row in index.items():
            if key[0] != "onto_no_se":
                continue
            other = index.get((baseline, key[1], key[2]))
            if other is not None:
                diff.append(float(row["paper_success"]) - float(other["paper_success"]))
        labels.append(f"Onto-NoSE - {baseline}")
        values.append(float(np.mean(diff)) if diff else np.nan)
    ax.bar(labels, values, color=colors[1:])
    ax.axhline(0.0, color="black", linewidth=.8)
    ax.set_ylabel("Paired success difference")
    save("paired_success_difference.png")
    return paths


def write_three_pipeline_outputs(records, training_records, output_dir, *,
                                 reward_design_episodes=0,
                                 reward_design_steps=0,
                                 estimator_warmup_episodes=0,
                                 estimator_warmup_steps=0):
    output_dir = Path(output_dir)
    evaluation_dir = output_dir / "evaluation"
    tables_dir = output_dir / "tables"
    _write_csv(evaluation_dir / "per_episode.csv", records)
    summary = physical_summary(records)
    paired = paired_confidence_intervals(records)
    efficiency = learning_efficiency(
        training_records, reward_design_episodes=reward_design_episodes,
        reward_design_steps=reward_design_steps,
        estimator_warmup_episodes=estimator_warmup_episodes,
        estimator_warmup_steps=estimator_warmup_steps)
    table = _publication_table(summary, efficiency)
    _write_csv(evaluation_dir / "paired_summary.csv", summary)
    _write_csv(evaluation_dir / "confidence_intervals.csv", paired)
    _write_csv(tables_dir / "primary_comparison.csv", table)
    tables_dir.mkdir(parents=True, exist_ok=True)
    _write_markdown(tables_dir / "primary_comparison.md", table)
    _write_csv(tables_dir / "sample_efficiency.csv", efficiency)
    figures = _write_figures(
        records, training_records, output_dir / "figures",
        reward_design_episodes=reward_design_episodes,
        reward_design_steps=reward_design_steps,
        estimator_warmup_episodes=estimator_warmup_episodes,
        estimator_warmup_steps=estimator_warmup_steps)
    return {"summary": summary, "paired": paired,
            "sample_efficiency": efficiency, "figures": figures}
