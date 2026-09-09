"""Publication figures.

Replaces the retired ``+evaluation/makePlots.m``. Nothing here recomputes
physics: every number was measured by Isaac, PX4 or the evaluation, and this
module only draws it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..config import Config
from ..semantic import GOAL_NODE, RISK_NODES
from .compare import LABELS, METRIC_FIELDS, RATE_METRICS
from .style import (GRID, INK, MUTED, POLICY_COLORS, SERIES, STATUS, SURFACE,
                    annotate_bars, figure)

__all__ = ["make_plots"]

# Node layout for the printed ontology figure, shared with the RViz overlay so
# the paper and the live view show the same graph.
GRAPH_LAYOUT = (
    (0, 3.5), (0, 2.5), (0, 1.5), (0, 0.5), (0, -0.5), (0, -1.5),
    (1, -1.5), (1, 1.0), (1, -0.2), (2, 0.4), (0, -2.5), (0, -3.5), (3, 0.4),
)


def _moving_mean(values: Sequence[float], window: int) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return v
    w = max(1, min(int(window), v.size))
    padded = np.concatenate([np.full(w - 1, v[0]), v])
    return np.convolve(padded, np.ones(w) / w, mode="valid")


def _label_last(ax, x, y, text: str, color: str) -> None:
    """Direct-label a line at its right end, so identity is not colour alone."""
    if len(x) == 0:
        return
    ax.annotate(text, (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                color=color, fontsize=8, va="center", fontweight="semibold")


def _save(fig, figdir: Path, name: str) -> Path:
    path = figdir / f"{name}.png"
    fig.savefig(path)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


# --------------------------------------------------------------------- panels
def _training_curves(histories: dict[str, Any], figdir: Path) -> Path:
    fig, axes = figure(2, 1, (7.0, 5.4), sharex=True)
    for label, history in histories.items():
        color = POLICY_COLORS[label]
        episodes = np.asarray(history["episode"])
        window = max(1, len(episodes) // 10)
        returns = _moving_mean(history["ret"], window)
        axes[0].plot(episodes, history["ret"], color=color, lw=0.7, alpha=0.28)
        axes[0].plot(episodes, returns, color=color, lw=2.0, label=label)
        _label_last(axes[0], episodes, returns, label, color)
        success = _moving_mean(history["success"], window)
        axes[1].plot(episodes, success, color=color, lw=2.0, label=label)
        _label_last(axes[1], episodes, success, label, color)
    axes[0].set(ylabel="Return", title="PPO training return (Isaac Sim + PX4)")
    axes[1].set(xlabel="Episode", ylabel="Success rate", ylim=(0, 1),
                title="Moving success rate")
    axes[0].legend(loc="lower right")
    return _save(fig, figdir, "training_curves")


def _evaluation_summary(results: dict[str, Any], figdir: Path) -> Path:
    """Small multiples, one measure per panel, one axis each.

    The retired version put episode energy and depletion rate on a single plot
    with two y-scales. Two scales on one plot imply a relationship the data
    does not contain, so each measure gets its own panel here.
    """
    summary = {row["Policy"]: row for row in results["summary"]}
    panels = [
        ("Landing success", "SuccessRate", 100.0, "%", "{:.1f}"),
        ("Unsafe touchdown", "UnsafeRate", 100.0, "%", "{:.1f}"),
        ("Timeout", "TimeoutRate", 100.0, "%", "{:.1f}"),
        ("Battery depleted", "DepletedRate", 100.0, "%", "{:.1f}"),
        ("Touchdown XY error", "MeanTouchdownXY", 1.0, "m", "{:.3f}"),
        ("Relative touchdown speed", "MeanTouchdownRelSpeedXY", 1.0, "m/s", "{:.3f}"),
        ("Maximum tilt", "MeanMaxTiltDeg", 1.0, "deg", "{:.1f}"),
        ("Peak Isaac aerodynamic force", "MeanMaxAeroForce", 1.0, "N", "{:.2f}"),
        ("Episode energy", "MeanEnergyJ", 1.0, "J", "{:.0f}"),
    ]
    fig, axes = figure(3, 3, (10.5, 8.0))
    flat = np.asarray(axes).ravel()
    for ax, (title, key, gain, unit, fmt) in zip(flat, panels):
        values = [summary[label][key] * gain for label in LABELS]
        bars = ax.bar(range(len(LABELS)), values, width=0.55,
                      color=[POLICY_COLORS[label] for label in LABELS])
        annotate_bars(ax, bars, values, fmt)
        ax.set_xticks(range(len(LABELS)))
        ax.set_xticklabels(["Manual", "Ontology\nR-GAT"], fontsize=8)
        ax.set(ylabel=unit, title=title)
        peak = max(values)
        # A panel where both arms are zero has no scale of its own; giving it
        # one makes matplotlib print a 1e-9 exponent over an empty plot.
        ax.set_ylim(0, peak * 1.28 if peak > 0 else 1.0)
        ax.grid(axis="x", visible=False)
    for ax in flat[len(panels):]:
        ax.remove()
    fig.suptitle(f"Paired evaluation over {summary[LABELS[0]]['Episodes']} common seeds",
                 fontsize=11, color=INK, x=0.01, ha="left")
    return _save(fig, figdir, "evaluation_summary")


def _paired_difference(results: dict[str, Any], figdir: Path) -> Path:
    """Proposed minus manual, per seed, with a 95% interval.

    This is the figure the comparison actually turns on and the retired version
    only ever wrote to CSV. Rates and continuous measures are faceted because
    they do not share a unit -- putting them on one axis would be the same
    mistake as a second y-scale.
    """
    paired = {row["Metric"]: row for row in results["paired"]}
    summary = {row["Policy"]: row for row in results["summary"]}
    rate_names = [name for name, _ in METRIC_FIELDS if name in RATE_METRICS]
    other_names = [name for name, _ in METRIC_FIELDS if name not in RATE_METRICS]

    # The continuous measures span joules to metres, so an absolute axis would
    # squash every one of them against the energy difference and show nothing.
    # Each is scaled by the manual arm's own mean, which puts them on one
    # dimensionless axis without implying a shared unit.
    scale = {}
    for name in other_names:
        reference = abs(summary[LABELS[0]].get(f"Mean{name}", 0.0))
        scale[name] = 100.0 / reference if reference > 1e-12 else float("nan")
    other_names = [n for n in other_names if np.isfinite(scale[n])]

    fig, axes = figure(1, 2, (10.4, 4.2))
    for ax, names, gains, unit in (
            (axes[0], rate_names, {n: 100.0 for n in rate_names}, "percentage points"),
            (axes[1], other_names, scale, "% of the Manual mean")):
        y = np.arange(len(names))
        mu = np.asarray([paired[n]["MeanDifference"] * gains[n] for n in names])
        lo = np.asarray([paired[n]["CI95Low"] * gains[n] for n in names])
        hi = np.asarray([paired[n]["CI95High"] * gains[n] for n in names])
        ax.axvline(0.0, color=STATUS["reference"], lw=1.0, ls="--")
        ax.hlines(y, lo, hi, color=SERIES[0], lw=2.0, alpha=0.55)
        ax.plot(mu, y, "o", color=SERIES[0], markersize=6)
        for yi, m, h in zip(y, mu, hi):
            ax.annotate(f"{m:+.1f}" if abs(m) >= 0.05 else "0.0",
                        (h, yi), xytext=(6, 0), textcoords="offset points",
                        fontsize=8, va="center", color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(f"Ontology-R-GAT minus Manual [{unit}]")
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.22)
    axes[0].set_title("Outcome rates")
    axes[1].set_title("Continuous measures, relative")
    fig.suptitle("Paired difference with a 95% interval (common random numbers)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    return _save(fig, figdir, "paired_difference")


def _axes3d_available() -> bool:
    """Whether this interpreter can draw a 3D axes.

    A machine with matplotlib installed twice -- the distribution package and a
    newer pip one -- resolves ``mpl_toolkits`` to the older tree, which then
    fails to import against the newer core. That is an environment fault and
    not something a figure should die of, so the trajectory figure falls back
    to two 2D views that carry the same information.
    """
    try:
        # Imported for the side effect: it registers the "3d" projection.
        import mpl_toolkits.mplot3d  # noqa: F401
        return True
    except Exception:
        return False


def _trajectories(results: dict[str, Any], cfg: Config, figdir: Path) -> Path:
    import matplotlib.pyplot as plt

    from .style import apply_style
    apply_style()
    baseline = results["baseline"]["representative"]
    proposed = results["proposed"]["representative"]
    use_3d = _axes3d_available()

    fig = plt.figure(figsize=(10.0, 4.4))
    left = fig.add_subplot(1, 2, 1, projection="3d") if use_3d else fig.add_subplot(1, 2, 1)
    right = fig.add_subplot(1, 2, 2)

    for log, label in ((baseline, LABELS[0]), (proposed, LABELS[1])):
        x = np.asarray(log.x)
        color = POLICY_COLORS[label]
        if use_3d:
            left.plot(x[:, 0], x[:, 1], x[:, 2], color=color, lw=2.0, label=label)
        else:
            left.plot(x[:, 0], x[:, 1], color=color, lw=2.0, label=label)
            left.plot(x[-1, 0], x[-1, 1], "o", color=color, markersize=7)
        right.plot(np.linalg.norm(x[:, :2], axis=1), x[:, 2], color=color,
                   lw=2.0, label=label)

    if use_3d:
        for axis in (left.xaxis, left.yaxis, left.zaxis):
            axis.pane.set_facecolor(SURFACE)
            axis.pane.set_edgecolor(GRID)
            axis._axinfo["grid"]["color"] = GRID
        left.scatter([0], [0], [0], color=INK, s=28, label="Deck centre")
        left.set(xlabel="East [m]", ylabel="North [m]", zlabel="Up [m]")
        left.set_title("Representative trajectories, pad-relative ENU", loc="left",
                       fontsize=10, color=INK)
    else:
        circle = plt.Circle((0, 0), cfg.criteria.xy, fill=False, lw=1.0,
                            ls="--", color=STATUS["reference"])
        left.add_patch(circle)
        left.plot(0, 0, "+", color=INK, markersize=10)
        left.set(xlabel="East [m]", ylabel="North [m]",
                 title="Representative tracks, pad-relative ENU (top down)")
        left.set_aspect("equal", adjustable="datalim")
    left.legend(loc="upper left", fontsize=8)

    right.axvline(cfg.criteria.xy, color=STATUS["reference"], ls="--", lw=1.0)
    right.annotate("success radius", (cfg.criteria.xy, right.get_ylim()[1]),
                   xytext=(4, -12), textcoords="offset points", fontsize=8, color=MUTED)
    right.set(xlabel="Horizontal distance to deck centre [m]",
              ylabel="Height above deck [m]", title="Descent profile")
    fig.tight_layout()
    return _save(fig, figdir, "representative_trajectory")


def _disturbance_response(results: dict[str, Any], figdir: Path) -> Path:
    """The proposed arm's representative episode, panel per measure.

    Six panels sharing one time axis, rather than the retired version's two
    dual-axis plots. Energy is two panels because hover seconds and electrical
    power are two different quantities.
    """
    log = results["proposed"]["representative"]
    t = np.asarray(log.t)
    wind = np.asarray(log.wind)
    fig, axes = figure(6, 1, (7.4, 11.0), sharex=True)
    for index, (name, color) in enumerate(zip(("East", "North", "Up"), SERIES)):
        axes[0].plot(t, wind[:, index], color=color, lw=1.6, label=name)
    axes[0].set(ylabel="m/s", title="Isaac wind field at the vehicle")
    axes[0].legend(ncols=3, loc="upper right")

    axes[1].plot(t, log.aero_mag, color=SERIES[0], lw=1.8)
    axes[1].set(ylabel="N", title="Isaac aerodynamic resultant |F|")

    axes[2].plot(t, np.degrees(log.tilt), color=SERIES[0], lw=1.8)
    axes[2].set(ylabel="deg", title="PX4 attitude response (tilt)")

    axes[3].plot(t, log.pad_speed, color=SERIES[0], lw=1.8, label="Deck speed")
    axes[3].plot(t, log.closing_speed, color=SERIES[1], lw=1.8, label="Relative UAV speed")
    axes[3].set(ylabel="m/s", title="Moving-target velocity")
    axes[3].legend(loc="upper right")

    axes[4].plot(t, log.hover_seconds_left, color=SERIES[0], lw=1.8)
    axes[4].set(ylabel="s", title="Onboard energy: hover seconds remaining")

    axes[5].plot(t, log.battery_power_w, color=SERIES[1], lw=1.8)
    axes[5].set(xlabel="Time [s]", ylabel="W", title="Onboard energy: electrical power")
    return _save(fig, figdir, "proposed_disturbance_response")


def _relation_attention(results: dict[str, Any], cfg: Config, figdir: Path) -> Path | None:
    """Mean attention per relation over the representative rollout.

    One series over four named categories, so one colour: a value ramp here
    would double-encode bar length as hue and say nothing new.
    """
    potential = results.get("potential")
    log = results["proposed"]["representative"]
    if potential is None or log.graph_template is None:
        return None
    template = log.graph_template
    totals = np.zeros(cfg.ontology.n_relations)
    samples = 0
    for k in range(0, len(log.graph_x), 5):
        explained = potential.explain(template.with_features(log.graph_x[k]))
        totals += explained["relation_mean"]
        samples += 1
    if samples == 0:
        return None
    means = totals / samples
    fig, ax = figure(1, 1, (6.4, 3.6))
    bars = ax.bar(range(len(means)), means, width=0.6, color=SERIES[0])
    annotate_bars(ax, bars, means, "{:.3f}")
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(cfg.ontology.relation_names)
    ax.set(ylabel="Mean attention", title="R-GAT relation attention on an Isaac/PX4 rollout")
    ax.set_ylim(0, float(means.max()) * 1.25)
    ax.grid(axis="x", visible=False)
    fig.text(0.01, -0.02, "Attention is learned importance, not causal proof.",
             fontsize=7.5, color=MUTED)
    return _save(fig, figdir, "rgat_relation_attention")


def _ontology_graph(results: dict[str, Any], cfg: Config, figdir: Path) -> Path | None:
    """The ontology, drawn with the potential's attention on its edges.

    The same node layout the RViz 2 overlay uses, so the printed figure and the
    live view are the same picture. Labels go outside their column rather than
    under their node: eight nodes stacked in the input column leave no room
    underneath, and a label that lands on the next node is worse than no label.
    """
    potential = results.get("potential")
    log = results["proposed"]["representative"]
    if potential is None or log.graph_template is None:
        return None
    # The step just before touchdown: the moment the potential is judged on.
    features = log.graph_x[-1]
    graph = log.graph_template.with_features(features)
    explained = potential.explain(graph)
    alpha = explained["edge_alpha"]
    values = features[0, :]

    positions = np.asarray([[2.4 * col, 1.25 * row] for col, row in GRAPH_LAYOUT],
                           dtype=float)
    fig, ax = figure(1, 1, (9.4, 5.8))
    peak = float(alpha.max()) or 1.0
    for e, (s, d, r) in enumerate(zip(graph.src, graph.dst, graph.rel)):
        if s == d:
            continue                                   # self-loops would be dots
        weight = float(alpha[e]) / peak
        ax.annotate("", xy=positions[int(d)], xytext=positions[int(s)],
                    arrowprops=dict(arrowstyle="-|>", color=SERIES[0],
                                    alpha=0.16 + 0.74 * weight,
                                    lw=0.5 + 3.0 * weight,
                                    shrinkA=14, shrinkB=16))
    for i, name in enumerate(graph.node_names):
        value = float(np.clip(values[i], 0.0, 1.0))
        if i == GOAL_NODE:
            face, edge = "#eda100", INK
        elif i in RISK_NODES:
            face, edge = SERIES[1], "none"
        else:
            face, edge = SERIES[2], "none"
        ax.scatter(*positions[i], s=150 + 520 * value, color=face,
                   edgecolors=edge, linewidths=1.2, zorder=3)
        column = GRAPH_LAYOUT[i][0]
        if column == 0:                                # the crowded input stack
            offset, ha, va = (-22, 0), "right", "center"
        elif i == GOAL_NODE:
            offset, ha, va = (24, 0), "left", "center"
        else:
            offset, ha, va = (0, -24), "center", "top"
        ax.annotate(name if i == GOAL_NODE else f"{name}  {value:.2f}",
                    positions[i], xytext=offset, textcoords="offset points",
                    ha=ha, va=va, fontsize=8, color=INK)
    ax.set_axis_off()
    ax.margins(x=0.30, y=0.10)
    ax.set_title("Ontology at touchdown: node activation (size) and R-GAT attention "
                 "(edge width)", loc="left", fontsize=10, color=INK)
    handles = [
        ax.scatter([], [], s=110, color=SERIES[1], label="risk node"),
        ax.scatter([], [], s=110, color=SERIES[2], label="support node"),
        ax.scatter([], [], s=110, color="#eda100", edgecolors=INK,
                   label="goal (SafeLanding)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    fig.text(0.01, 0.01, "Attention is learned importance, not causal proof.",
             fontsize=7.5, color=MUTED)
    return _save(fig, figdir, "ontology_attention_graph")


def _sweep_lines(ax, rows: Sequence[dict[str, Any]], x_key: str, y_key: str,
                 gain: float = 1.0) -> None:
    for label in LABELS:
        selected = [r for r in rows if r["Policy"] == label]
        if not selected:
            continue
        x = np.asarray([r[x_key] for r in selected], dtype=float)
        y = np.asarray([r[y_key] for r in selected], dtype=float) * gain
        order = np.argsort(x)
        color = POLICY_COLORS[label]
        ax.plot(x[order], y[order], "-o", color=color, lw=2.0, label=label)
        _label_last(ax, x[order], y[order], label, color)


def _wind_sweep(rows, figdir: Path) -> Path:
    fig, ax = figure(1, 1, (6.6, 3.8))
    _sweep_lines(ax, rows, "WindScale", "SuccessRate", 100.0)
    ax.set(xlabel="Isaac wind intensity scale", ylabel="Success [%]", ylim=(0, 100),
           title="External disturbance generalization")
    ax.legend(loc="lower left")
    ax.margins(x=0.14)
    return _save(fig, figdir, "wind_generalization_success")


def _pad_sweep(rows, cfg: Config, figdir: Path) -> Path:
    fig, axes = figure(2, 1, (6.8, 6.2), sharex=True)
    _sweep_lines(axes[0], rows, "MeanPadSpeed", "SuccessRate", 100.0)
    axes[0].set(ylabel="Success [%]", ylim=(0, 100),
                title="Success versus measured deck speed")
    axes[0].legend(loc="lower left")
    _sweep_lines(axes[1], rows, "MeanPadSpeed", "MeanRelSpeedXY")
    axes[1].axhline(cfg.criteria.rel_speed_xy, color=STATUS["reference"], ls="--", lw=1.0)
    axes[1].annotate("success limit", (axes[1].get_xlim()[0], cfg.criteria.rel_speed_xy),
                     xytext=(4, 4), textcoords="offset points", fontsize=8, color=MUTED)
    axes[1].set(xlabel="Mean deck speed [m/s]", ylabel="Relative touchdown speed [m/s]",
                title="Velocity still unmatched at contact")
    for ax in axes:
        ax.margins(x=0.14)
    return _save(fig, figdir, "pad_speed_generalization")


def _gnss_sweep(rows, figdir: Path) -> Path:
    """Success against the canyon, and what the canyon did to the estimate.

    The lower panel is what makes the upper one readable: the scale is a
    multiplier on the error mechanisms, so the honest x-axis for "how hard was
    this" is the metres of position error it actually produced.
    """
    fig, axes = figure(2, 1, (6.8, 6.2), sharex=True)
    _sweep_lines(axes[0], rows, "GnssScale", "SuccessRate", 100.0)
    axes[0].set(ylabel="Success [%]", ylim=(0, 100),
                title="Landing success under urban GNSS degradation")
    axes[0].legend(loc="lower left")
    _sweep_lines(axes[1], rows, "GnssScale", "MeanEstimateErrorM")
    axes[1].set(xlabel="GNSS degradation scale (0 = open sky, same city)",
                ylabel="Pad-relative estimate error [m]",
                title="Error in the pose the landing was flown on")
    for ax in axes:
        ax.margins(x=0.14)
    return _save(fig, figdir, "gnss_degradation_success")


def _battery_bins(rows, figdir: Path) -> Path:
    centred = [dict(r, Centre=0.5 * (r["ReserveLowS"] + r["ReserveHighS"])) for r in rows]
    fig, ax = figure(1, 1, (6.6, 3.8))
    _sweep_lines(ax, centred, "Centre", "SuccessRate", 100.0)
    ax.set(xlabel="Starting reserve [hover s]", ylabel="Success [%]", ylim=(0, 100),
           title="Landing success versus onboard energy reserve")
    ax.legend(loc="lower right")
    ax.margins(x=0.14)
    return _save(fig, figdir, "battery_reserve_success")


# ---------------------------------------------------------------------- entry
def make_plots(results: dict[str, Any], histories: dict[str, Any],
               cfg: Config) -> list[Path]:
    """Write every publication figure and return the paths."""
    figdir = Path(cfg.paths.figures)
    figdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if histories:
        written.append(_training_curves(histories, figdir))
    written.append(_evaluation_summary(results, figdir))
    written.append(_paired_difference(results, figdir))
    written.append(_trajectories(results, cfg, figdir))
    written.append(_disturbance_response(results, figdir))
    for maker in (_relation_attention, _ontology_graph):
        path = maker(results, cfg, figdir)
        if path is not None:
            written.append(path)
    if results.get("wind_sweep"):
        written.append(_wind_sweep(results["wind_sweep"], figdir))
    if results.get("pad_sweep"):
        written.append(_pad_sweep(results["pad_sweep"], cfg, figdir))
    if results.get("gnss_sweep"):
        written.append(_gnss_sweep(results["gnss_sweep"], figdir))
    if results.get("battery_bins"):
        written.append(_battery_bins(results["battery_bins"], figdir))
    print(f"Wrote {len(written)} figures to {figdir}")
    return written
