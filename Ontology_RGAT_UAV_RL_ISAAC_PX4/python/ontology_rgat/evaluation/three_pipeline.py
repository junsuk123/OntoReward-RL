"""Reward-independent reporting for the primary three-pipeline comparison."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


PHYSICAL_METRICS = (
    "paper_success", "strict_success", "pad_contact", "unsafe_pad_contact",
    "landing_gate_contact", "landing_gate_position",
    "landing_gate_vertical_speed", "landing_gate_relative_horizontal_speed",
    "landing_gate_attitude", "landing_gate_angular_rate", "crash_failure",
    "touchdown_lateral_error", "touchdown_vertical_velocity",
    "touchdown_relative_horizontal_velocity", "touchdown_tilt",
    "touchdown_roll", "touchdown_pitch", "touchdown_angular_rate",
    "collision_rate", "excessive_drift_rate", "fov_loss_fraction",
    "longest_visual_loss_s", "visual_loss_events",
    "visual_reacquisition_events", "visual_reacquisition_rate",
    "mean_visual_reacquisition_time_s", "recovery_climb_fraction",
    "unsafe_descent_low_visibility_fraction", "recovery_landing_opportunity",
    "successful_recovery_landing", "touchdown_time_s",
    "adaptive_rgat_inference_latency_ms_mean",
    "adaptive_rgat_parameter_count",
)
PRIMARY_PIPELINES = ("shin_se", "no_se", "onto_no_se")


def _replicate(row) -> str:
    return str(row.get("training_replicate", "0"))


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


def _hierarchical_bootstrap_mean(grouped_values, *, draws=10000, seed=90421):
    """Equal-weight replicate bootstrap, then episode bootstrap within each."""
    groups = [np.asarray(values, dtype=np.float64)
              for values in grouped_values if len(values)]
    if not groups:
        return np.nan, np.nan, np.nan, "none"
    if len(groups) == 1:
        mean, low, high = _bootstrap_mean(groups[0], draws=draws, seed=seed)
        return mean, low, high, "episode"
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        chosen = rng.integers(0, len(groups), size=len(groups))
        replicate_means = []
        for group_index in chosen:
            values = groups[int(group_index)]
            sample = values[rng.integers(0, values.size, size=values.size)]
            replicate_means.append(float(sample.mean()))
        estimates[draw] = float(np.mean(replicate_means))
    point = float(np.mean([values.mean() for values in groups]))
    return (point, float(np.quantile(estimates, 0.025)),
            float(np.quantile(estimates, 0.975)), "training_replicate_then_episode")


def physical_summary(records):
    grouped = {}
    for row in records:
        grouped.setdefault((row["pipeline"], row["scenario"]), []).append(row)
    result = []
    for (pipeline, scenario), rows in sorted(grouped.items()):
        summary = {"pipeline": pipeline, "scenario": scenario,
                   "episodes": len(rows),
                   "training_replicates": len({_replicate(row) for row in rows})}
        for metric in PHYSICAL_METRICS:
            by_replicate = {}
            for row in rows:
                if row.get(metric, "") != "":
                    by_replicate.setdefault(_replicate(row), []).append(
                        float(row[metric]))
            mean, low, high, unit = _hierarchical_bootstrap_mean(
                list(by_replicate.values()))
            replicate_means = [float(np.mean(values))
                               for values in by_replicate.values()]
            standard_deviation = (float(np.std(replicate_means, ddof=1))
                                  if len(replicate_means) > 1 else
                                  float(np.std([value for values in by_replicate.values()
                                                for value in values], ddof=1))
                                  if sum(len(values) for values in by_replicate.values()) > 1
                                  else 0.0)
            summary.update({f"{metric}_mean": mean,
                            f"{metric}_std": standard_deviation,
                            f"{metric}_ci95_low": low,
                            f"{metric}_ci95_high": high,
                            f"{metric}_bootstrap_unit": unit})
        result.append(summary)
    return result


def paired_differences(records):
    """Raw within-replicate, within-scenario, within-seed differences."""
    index = {(_replicate(row), row["pipeline"], row["scenario"], int(row["seed"])): row
             for row in records}
    present = {row["pipeline"] for row in records}
    if "onto_rgat_adaptive_weight_no_se" in present:
        proposed = "onto_rgat_adaptive_weight_no_se"
        baselines = [name for name in (
            "shin_se_fixed", "shin_se_rgat_weight", "no_se_fixed",
            "onto_rgat_potential_pbrs_no_se") if name in present]
        comparisons = tuple((proposed, baseline) for baseline in baselines)
    else:
        comparisons = (("onto_no_se", "shin_se"), ("onto_no_se", "no_se"))
    scenarios = sorted({row["scenario"] for row in records})
    output = []
    for proposed, baseline in comparisons:
        for scenario in scenarios:
            replicates = sorted({key[0] for key in index})
            for replicate in replicates:
                proposed_seeds = {key[3] for key in index
                                  if key[:3] == (replicate, proposed, scenario)}
                baseline_seeds = {key[3] for key in index
                                  if key[:3] == (replicate, baseline, scenario)}
                for item in sorted(proposed_seeds & baseline_seeds):
                    proposed_row = index[(replicate, proposed, scenario, item)]
                    baseline_row = index[(replicate, baseline, scenario, item)]
                    for metric in PHYSICAL_METRICS:
                        if metric not in proposed_row or metric not in baseline_row:
                            continue
                        output.append({
                            "comparison": f"{proposed} - {baseline}",
                            "proposed": proposed, "baseline": baseline,
                            "training_replicate": replicate,
                            "scenario": scenario, "seed": item, "metric": metric,
                            "difference": (float(proposed_row[metric])
                                           - float(baseline_row[metric])),
                        })
    return output


def paired_confidence_intervals(records, *, draws=10000, seed=8128):
    raw = paired_differences(records)
    grouped = {}
    for row in raw:
        key = (row["comparison"], row["proposed"], row["baseline"],
               row["scenario"], row["metric"])
        grouped.setdefault(key, {}).setdefault(
            row["training_replicate"], []).append(float(row["difference"]))
    output = []
    for key, by_replicate in sorted(grouped.items()):
        mean, low, high, unit = _hierarchical_bootstrap_mean(
            list(by_replicate.values()), draws=draws, seed=seed + len(output))
        comparison, proposed, baseline, scenario, metric = key
        output.append({
            "comparison": comparison, "proposed": proposed,
            "baseline": baseline, "scenario": scenario, "metric": metric,
            "training_replicates": len(by_replicate),
            "paired_episodes": sum(len(values) for values in by_replicate.values()),
            "mean_difference": mean, "ci95_low": low, "ci95_high": high,
            "bootstrap_unit": unit,
        })
    return output


def _rolling_success(rows, window=20):
    rows = [row for row in rows
            if row.get("optimization_phase", "ppo") == "ppo"]
    grouped = {}
    for row in rows:
        grouped.setdefault(_replicate(row), []).append(row)
    curves = []
    for replicate_rows in grouped.values():
        ordered = sorted(replicate_rows, key=lambda row: int(float(
            row.get("ppo_episode", row["episode"]))))
        successes = np.asarray([float(row["paper_success"]) for row in ordered])
        episodes = np.asarray([
            int(float(row.get("ppo_episode", row["episode"]))) for row in ordered])
        steps = np.asarray([int(float(row.get("ppo_environment_steps", 0)))
                            for row in ordered])
        average = np.asarray([
            successes[max(0, index - window + 1):index + 1].mean()
            for index in range(successes.size)])
        curves.append({episode: (step, value)
                       for episode, step, value in zip(episodes, steps, average)})
    if not curves:
        return np.asarray([]), np.asarray([]), np.asarray([])
    common = sorted(set.intersection(*(set(curve) for curve in curves)))
    episodes = np.asarray(common, dtype=np.int64)
    steps = np.asarray([
        int(round(np.mean([curve[episode][0] for curve in curves])))
        for episode in common], dtype=np.int64)
    average = np.asarray([
        float(np.mean([curve[episode][1] for curve in curves]))
        for episode in common], dtype=np.float64)
    return episodes, steps, average


def learning_efficiency(training_records, *, reward_design_episodes=0,
                        reward_design_steps=0, estimator_warmup_episodes=0,
                        estimator_warmup_steps=0,
                        behavior_cloning_episodes=0,
                        behavior_cloning_steps=0,
                        reward_design_costs=None):
    output = []
    pipelines = list(dict.fromkeys(
        row.get("pipeline") for row in training_records if row.get("pipeline")))
    for pipeline in pipelines:
        rows = [row for row in training_records if row.get("pipeline") == pipeline]
        if not rows:
            continue
        episodes, steps, success = _rolling_success(rows)
        if episodes.size == 0:
            continue
        auc = float(np.trapz(success, episodes) / max(float(episodes[-1] - episodes[0]), 1.0))
        uses_reward_design = (pipeline == "onto_no_se"
                              or "rgat" in pipeline or "pbrs" in pipeline)
        uses_estimator_warmup = pipeline.startswith("shin_se")
        explicit_cost = dict((reward_design_costs or {}).get(pipeline) or {})
        reward_extra_episodes = (int(explicit_cost.get(
            "episodes", reward_design_episodes)) if uses_reward_design else 0)
        reward_extra_steps = (int(explicit_cost.get(
            "steps", reward_design_steps)) if uses_reward_design else 0)
        estimator_extra_episodes = (int(estimator_warmup_episodes)
                                    if uses_estimator_warmup else 0)
        estimator_extra_steps = (int(estimator_warmup_steps)
                                 if uses_estimator_warmup else 0)
        extra_episodes = (int(behavior_cloning_episodes)
                          + reward_extra_episodes + estimator_extra_episodes)
        extra_steps = (int(behavior_cloning_steps)
                       + reward_extra_steps + estimator_extra_steps)
        item = {
            "pipeline": pipeline, "ppo_episodes": int(episodes[-1]),
            "ppo_environment_steps": int(steps[-1]),
            "behavior_cloning_episodes": int(behavior_cloning_episodes),
            "behavior_cloning_environment_steps": int(behavior_cloning_steps),
            "reward_design_episodes": reward_extra_episodes,
            "reward_design_environment_steps": reward_extra_steps,
            "estimator_warmup_episodes": estimator_extra_episodes,
            "estimator_warmup_environment_steps": estimator_extra_steps,
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
    pipelines = list(dict.fromkeys(row["pipeline"] for row in summary))
    for pipeline in pipelines:
        values = [row for row in summary if row["pipeline"] == pipeline]
        if not values:
            continue
        def average(field):
            numbers = [float(row[field]) for row in values
                       if field in row and np.isfinite(float(row[field]))]
            return float(np.mean(numbers)) if numbers else np.nan
        row = {
            "pipeline": pipeline,
            "state_estimation_supervision": pipeline.startswith("shin_se"),
            "direct_semantic_rgat_potential": (
                pipeline == "onto_no_se" or "potential_pbrs" in pipeline),
            "adaptive_rgat_weighting": (
                "adaptive_weight" in pipeline or "rgat_weight" in pipeline),
            "success_rate": average("paper_success_mean"),
            "strict_success_rate": average("strict_success_mean"),
            "crash_rate": average("crash_failure_mean"),
            # This is the final endpoint error for timeouts and touchdown error
            # only when contact occurred.  The old label overstated precision.
            "final_lateral_error_m": average("touchdown_lateral_error_mean"),
            "touchdown_vertical_velocity_m_s": average(
                "touchdown_vertical_velocity_mean"),
            "fov_loss_fraction": average("fov_loss_fraction_mean"),
            "visual_reacquisition_rate": average(
                "visual_reacquisition_rate_mean"),
            "successful_recovery_landing_rate": average(
                "successful_recovery_landing_mean"),
            "unsafe_descent_low_visibility_fraction": average(
                "unsafe_descent_low_visibility_fraction_mean"),
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
                   estimator_warmup_episodes=0, estimator_warmup_steps=0,
                   behavior_cloning_episodes=0, behavior_cloning_steps=0,
                   reward_design_costs=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    colors = ("#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30",
              "#4DBEEE", "#A2142F")
    pipelines = list(dict.fromkeys(row["pipeline"] for row in records))

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
            ("visual_reacquisition_rate.png", "visual_reacquisition_rate",
             "Reacquisition / visual-loss event"),
            ("successful_recovery_landing.png", "successful_recovery_landing",
             "Loss-reacquisition-landing success"),
            ("unsafe_blind_descent.png", "unsafe_descent_low_visibility_fraction",
             "Unsafe low-visibility descent fraction"),
            ("landing_time.png", "touchdown_time_s", "Landing time (s)")):
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        data = [[float(row[metric]) for row in records if row["pipeline"] == pipeline]
                for pipeline in pipelines]
        ax.boxplot(data, labels=pipelines, showmeans=True)
        ax.tick_params(axis="x", labelrotation=18)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=.45)
        save(name)

    for filename, x_kind in (("success_learning_curve.png", "episode"),
                             ("sample_efficiency_ppo_only.png", "steps"),
                             ("sample_efficiency_total_interactions.png", "total")):
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for color, pipeline in zip(colors, pipelines):
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
                shift = int(behavior_cloning_steps)
                if pipeline == "onto_no_se" or "rgat" in pipeline or "pbrs" in pipeline:
                    shift += int(dict((reward_design_costs or {}).get(
                        pipeline) or {}).get("steps", reward_design_steps))
                if pipeline.startswith("shin_se"):
                    shift += int(estimator_warmup_steps)
                x = steps + shift
                xlabel = "Total environment steps (pre-training + PPO)"
            ax.plot(x, success, color=color, label=pipeline)
        ax.set(xlabel=xlabel, ylabel="Rolling landing success")
        ax.grid(True, linestyle=":", alpha=.45)
        ax.legend()
        save(filename)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    index = {(_replicate(row), row["pipeline"], row["scenario"], int(row["seed"])): row
             for row in records}
    labels, values = [], []
    proposed = ("onto_rgat_adaptive_weight_no_se"
                if "onto_rgat_adaptive_weight_no_se" in pipelines else "onto_no_se")
    baselines = ([name for name in (
        "shin_se_fixed", "shin_se_rgat_weight", "no_se_fixed",
        "onto_rgat_potential_pbrs_no_se") if name in pipelines]
        if proposed != "onto_no_se" else ["shin_se", "no_se"])
    for baseline in baselines:
        diff = []
        for key, row in index.items():
            if key[1] != proposed:
                continue
            other = index.get((key[0], baseline, key[2], key[3]))
            if other is not None:
                diff.append(float(row["paper_success"]) - float(other["paper_success"]))
        labels.append(f"{proposed} - {baseline}")
        values.append(float(np.mean(diff)) if diff else np.nan)
    ax.bar(labels, values, color=[colors[(i + 1) % len(colors)]
                                  for i in range(len(labels))])
    ax.tick_params(axis="x", labelrotation=18)
    ax.axhline(0.0, color="black", linewidth=.8)
    ax.set_ylabel("Paired success difference")
    save("paired_success_difference.png")
    return paths


def write_three_pipeline_outputs(records, training_records, output_dir, *,
                                 reward_design_episodes=0,
                                 reward_design_steps=0,
                                 estimator_warmup_episodes=0,
                                 estimator_warmup_steps=0,
                                 behavior_cloning_episodes=0,
                                 behavior_cloning_steps=0,
                                 reward_design_costs=None):
    output_dir = Path(output_dir)
    evaluation_dir = output_dir / "evaluation"
    tables_dir = output_dir / "tables"
    _write_csv(evaluation_dir / "per_episode.csv", records)
    summary = physical_summary(records)
    raw_paired = paired_differences(records)
    paired = paired_confidence_intervals(records)
    efficiency = learning_efficiency(
        training_records, reward_design_episodes=reward_design_episodes,
        reward_design_steps=reward_design_steps,
        estimator_warmup_episodes=estimator_warmup_episodes,
        estimator_warmup_steps=estimator_warmup_steps,
        behavior_cloning_episodes=behavior_cloning_episodes,
        behavior_cloning_steps=behavior_cloning_steps,
        reward_design_costs=reward_design_costs)
    table = _publication_table(summary, efficiency)
    _write_csv(evaluation_dir / "paired_summary.csv", summary)
    nominal = {(row["pipeline"]): row for row in summary
               if row["scenario"] == "training_random_walk"}
    degradation = []
    for row in summary:
        reference = nominal.get(row["pipeline"])
        if reference is None:
            continue
        degradation.append({
            "pipeline": row["pipeline"], "scenario": row["scenario"],
            "success_rate_degradation": (
                float(reference["paper_success_mean"])
                - float(row["paper_success_mean"])),
            "touchdown_error_increase_m": (
                float(row["touchdown_lateral_error_mean"])
                - float(reference["touchdown_lateral_error_mean"])),
            "fov_loss_increase": (
                float(row["fov_loss_fraction_mean"])
                - float(reference["fov_loss_fraction_mean"])),
        })
    _write_csv(evaluation_dir / "disturbance_degradation.csv", degradation)
    _write_csv(evaluation_dir / "confidence_intervals.csv", paired)
    _write_csv(evaluation_dir / "paired_differences.csv", raw_paired)
    _write_csv(tables_dir / "primary_comparison.csv", table)
    tables_dir.mkdir(parents=True, exist_ok=True)
    _write_markdown(tables_dir / "primary_comparison.md", table)
    _write_csv(tables_dir / "sample_efficiency.csv", efficiency)
    figures = _write_figures(
        records, training_records, output_dir / "figures",
        reward_design_episodes=reward_design_episodes,
        reward_design_steps=reward_design_steps,
        estimator_warmup_episodes=estimator_warmup_episodes,
        estimator_warmup_steps=estimator_warmup_steps,
        behavior_cloning_episodes=behavior_cloning_episodes,
        behavior_cloning_steps=behavior_cloning_steps,
        reward_design_costs=reward_design_costs)
    return {"summary": summary, "paired": paired,
            "disturbance_degradation": degradation,
            "sample_efficiency": efficiency, "figures": figures}
