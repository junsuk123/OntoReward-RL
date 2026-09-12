"""적응 보상 가중치 전용 MATLAB 계열 진단 플롯."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


MATLAB_COLORS = ("#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30")
COMPONENT_LABELS = ("lateral", "vertical", "vz safety", "undershoot", "yaw")


def _read_traces(output_dir: Path):
    rows = []
    for path in sorted((output_dir / "models").glob("*/*_reward_steps.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_adaptive_reward_figures(output_dir) -> list[str]:
    """가중치/기여도/attention을 서로 혼동하지 않는 별도 그림으로 저장한다."""
    output_dir = Path(output_dir)
    rows = _read_traces(output_dir)
    adaptive = [row for row in rows
                if row.get("weight_source") == "frozen_rgat_adaptive"]
    if not adaptive:
        return []
    proposed = [row for row in adaptive
                if row.get("method") == "onto_rgat_adaptive_weight_no_se"]
    if proposed:
        adaptive = proposed
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return []
    figure_dir = output_dir / "figures/adaptive_reward"
    figure_dir.mkdir(parents=True, exist_ok=True)
    output = []

    def save(name):
        path = figure_dir / name
        plt.gcf().tight_layout()
        plt.gcf().savefig(path, dpi=180)
        plt.close(plt.gcf())
        output.append(str(path))

    groups = {}
    for row in adaptive:
        groups.setdefault((row["method"], int(row["episode"])), []).append(row)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    episodes = sorted({key[1] for key in groups})
    for component in range(1, 6):
        values = [np.mean([item[f"weight_{component}"]
                           for key, group in groups.items() if key[1] == episode
                           for item in group]) for episode in episodes]
        ax.plot(episodes, values, color=MATLAB_COLORS[component - 1],
                label=COMPONENT_LABELS[component - 1])
    ax.set(xlabel="Episode", ylabel="Mean adaptive weight",
           title="State-adaptive reward weights by episode")
    ax.grid(True, linestyle=":", alpha=.45)
    ax.legend(ncols=3, fontsize=8)
    save("weights_by_episode.png")

    phases = list(dict.fromkeys(str(row.get("landing_phase", "unknown"))
                               for row in adaptive))
    matrix = np.asarray([[np.mean([float(row[f"weight_{k}"])
                                  for row in adaptive
                                  if row.get("landing_phase", "unknown") == phase])
                          for k in range(1, 6)] for phase in phases])
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(5), COMPONENT_LABELS, rotation=18)
    ax.set_yticks(range(len(phases)), phases)
    ax.set_title("Mean weight by flight/training phase")
    fig.colorbar(image, ax=ax, label="weight")
    save("weights_by_phase.png")

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    x = np.arange(5)
    for offset, (outcome, label) in zip((-.18, .18), ((1, "success"), (0, "failure"))):
        value = [np.mean([float(row[f"weight_{k}"]) for row in adaptive
                          if int(row["success"]) == outcome]) for k in range(1, 6)]
        ax.bar(x + offset, value, width=.36, label=label)
    ax.set_xticks(x, COMPONENT_LABELS, rotation=18)
    ax.set(ylabel="Mean weight", title="Weight distribution by terminal outcome")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=.45)
    save("weights_success_failure.png")

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    disturbance = np.asarray([float(row.get("disturbance_level", 0.0))
                              for row in adaptive])
    order = np.argsort(disturbance)
    for k in range(1, 6):
        ax.plot(disturbance[order], np.asarray(
            [float(row[f"weight_{k}"]) for row in adaptive])[order], ".",
            color=MATLAB_COLORS[k - 1], alpha=.25,
            label=COMPONENT_LABELS[k - 1])
    ax.set(xlabel="External-force disturbance [N]", ylabel="Adaptive weight",
           title="Weight response to disturbance")
    ax.grid(True, linestyle=":", alpha=.45)
    ax.legend(ncols=3, fontsize=8)
    save("weights_by_disturbance.png")

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    contributions = [sum(float(row.get(f"weighted_{k}", 0.0)) for row in adaptive)
                     for k in range(1, 6)]
    ax.bar(range(5), contributions, color=MATLAB_COLORS)
    ax.axhline(0.0, color="black", linewidth=.8)
    ax.set_xticks(range(5), COMPONENT_LABELS, rotation=18)
    ax.set(ylabel="Cumulative reward contribution",
           title="Cumulative adaptive component contributions")
    ax.grid(axis="y", linestyle=":", alpha=.45)
    save("cumulative_component_contributions.png")

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    fixed_names = ("lateral_progress", "vertical_progress",
                   "vertical_speed_penalty", "undershoot_penalty",
                   "yaw_rate_penalty")
    values = [np.mean([float(row.get("adaptive_shaping",
                                     sum(float(row.get(name, 0.0))
                                         for name in fixed_names)))
                       for row in rows if row["method"] == method])
              for method in methods]
    ax.bar(range(len(methods)), values,
           color=[MATLAB_COLORS[i % 5] for i in range(len(methods))])
    ax.set_xticks(range(len(methods)), methods, rotation=20, ha="right")
    ax.set(ylabel="Mean shaping reward", title="Fixed vs adaptive reward")
    ax.grid(axis="y", linestyle=":", alpha=.45)
    save("fixed_vs_adaptive_reward.png")

    attention_keys = sorted(key for key in adaptive[0]
                            if key.startswith("attention_relation_"))
    if attention_keys:
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        values = [np.mean([float(row.get(key, 0.0)) for row in adaptive])
                  for key in attention_keys]
        ax.bar(range(len(values)), values, color="#4DBEEE")
        ax.set_xticks(range(len(values)), [key.rsplit("_", 1)[-1]
                                           for key in attention_keys])
        ax.set(xlabel="Relation ID", ylabel="Mean attention",
               title="R-GAT relation attention (not reward weights)")
        ax.grid(axis="y", linestyle=":", alpha=.45)
        save("relation_attention_separate_from_weights.png")
    return output
