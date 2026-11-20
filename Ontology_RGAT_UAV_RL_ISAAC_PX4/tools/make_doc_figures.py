"""README figures for the ORL landing policy (docs/figures/*.png, *.svg).

    python3 tools/make_doc_figures.py all         # every figure
    python3 tools/make_doc_figures.py overall | architecture | graph | field

Diagrams are drawn from the code: the proposed ORL policy's nodes, edges, slots and sizes from
current_semantics/orl.py (version ORL_VERSION), the approach field from current_semantics/reward.py.
White background, English labels, PNG 200 dpi + SVG.  The result figures of the earlier
navigation-degradation study (registered R-GAT arm) were archived on 2026-11-20
(archive/non_orl_20261120.tar.zst).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import (Circle, Ellipse, FancyArrowPatch, FancyBboxPatch,  # noqa: E402
                                Rectangle, Wedge)
from matplotlib.path import Path as MplPath  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parent
sys.path.insert(0, str(PACKAGE / "python"))
OUT = REPO / "docs" / "figures"
RESULTS = PACKAGE / "results" / "current_semantics_3d"
CONFIG = PACKAGE / "config" / "current_semantics" / "spatial_view_contact_v1.yaml"

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 10, "svg.fonttype": "path",
                     "figure.facecolor": "white", "savefig.facecolor": "white"})

# paper-figure palette: pastel fill, darker edge of the same hue (one hue per role)
C = {"sim": ("#EEF1F3", "#2B2B2B"), "est": ("#CEDCF6", "#6E98DA"), "know": ("#DDF5E7", "#4FA862"),
     "pol": ("#E2C2B6", "#C0634B"), "off": ("#FBF8F0", "#7A7A7A"), "gt": ("#FCEBE2", "#B5470A"),
     "kg": ("#B9EEF4", "#2B2B2B")}
DASH = "#2F6EA6"          # dashed containers and zoom connectors
INK, MUTED, GRID = "#1F1F1F", "#5A5A5A", "#E4E4E4"
PROPOSED, OTHER = "#C0634B", "#9A9A9A"
# relation types: validated categorical slots 1-5 (scripts/validate_palette.js, light surface);
# every relation is also named in the legend and drawn with its own line style
RELATION_STYLE = {
    "qualifiesEstimate": ("#2a78d6", "-"),
    "informsApproach": ("#eb6834", (0, (5, 2))),
    "affectsVisibility": ("#1baf7a", (0, (1.2, 1.6))),
    "conditionsContact": ("#eda100", (0, (6, 2, 1.5, 2))),
    "informsLanding": ("#e87ba4", "-"),
    # ORL's two decision relations: categorical slots 6-7 (same validator, 7-slot palette passes)
    "gatesDescent": ("#008300", (0, (8, 2))),
    "informsVertical": ("#6250d6", (0, (2, 1.2, 6, 1.2))),
}


# ----------------------------------------------------------------------------- drawing helpers
def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200 if ext == "png" else None, bbox_inches="tight",
                    pad_inches=0.12)
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def canvas(w, h):
    fig = plt.figure(figsize=(w / 10, h / 10))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title, lines=(), role="sim", tsize=10, bsize=8.6, dashed=False):
    fc, ec = C[role]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec,
                                lw=1.8, ls=(0, (4, 3)) if dashed else "-", zorder=3))
    if lines:
        n = len(lines)
        ty = y + h / 2 + (n * 1.55) / 2 + 0.25
        ax.text(x + w / 2, ty, title, ha="center", va="center", fontsize=tsize, weight="bold",
                color=INK, zorder=4)
        ax.text(x + w / 2, ty - 1.55, "\n".join(lines), ha="center", va="top", fontsize=bsize,
                color=MUTED, zorder=4, linespacing=1.32)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=tsize, weight="bold",
                color=INK, zorder=4)


def panel(ax, x, y, w, h, title, role=None, fill="white"):
    """A dashed blue container with a bold title (the paper-figure grouping box)."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6", fc=fill,
                                ec=DASH, lw=2.0, ls=(0, (6, 3)), zorder=1))
    ax.text(x + 1.4, y + h - 2.2, title, ha="left", va="center", fontsize=12, weight="bold",
            color=INK, zorder=2)


def arrow(ax, p, q, color=INK, lw=1.4, ls="-", rad=0.0, z=2, size=10):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=size, color=color, lw=lw,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=1,
                                 shrinkB=1))


def route(ax, pts, color=INK, lw=1.3, ls="-", label=None, lseg=1, lsize=8.2):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls, zorder=2, solid_capstyle="butt")
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw, ls=ls)
    if label:
        (x0, y0), (x1, y1) = pts[lseg], pts[lseg + 1]
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, label, ha="center", va="center", fontsize=lsize, color=color,
                zorder=5, bbox=dict(fc="white", ec="none", pad=0.4))


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#BDBDBD")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------------------- data extraction
def _rect(ax, x, y, w, h, text, role, size=11.5, weight="normal", z=3):
    fc, ec = C[role]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc=fc, ec=ec,
                                lw=2.0, zorder=z))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, weight=weight,
            color=INK, zorder=z + 1, linespacing=1.25)


def _ellipse(ax, cx, cy, w, h, text, size=11):
    ax.add_patch(Ellipse((cx, cy), w, h, fc="white", ec=INK, lw=1.4, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=size, color=INK, zorder=4, linespacing=1.15)


def _line(ax, pts, color=INK, lw=2.0, ls="-", head=True, size=12, z=2):
    xs, ys = zip(*pts)
    if head:
        ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls, zorder=z, solid_capstyle="butt")
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=size, color=color,
                                     lw=lw, linestyle=ls, zorder=z, shrinkA=0, shrinkB=0))
    else:
        ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=z)


def _circles(ax, x0, y, n, r, fc, ec=INK, gap=0.55, frame=True, ellipsis_at=None):
    xs = []
    x = x0 + r
    for i in range(n):
        if ellipsis_at is not None and i == ellipsis_at:
            ax.text(x + r * 0.6, y, "...", ha="center", va="center", fontsize=11, color=INK, zorder=5)
            x += 2.4 * r
        ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=1.2, zorder=4))
        xs.append(x)
        x += 2 * r + gap
    if frame:
        ax.add_patch(FancyBboxPatch((x0 - 0.6, y - r - 0.6), x - x0 - gap + 1.2, 2 * r + 1.2,
                                    boxstyle="round,pad=0,rounding_size=0.8", fc="white", ec=INK, lw=1.4,
                                    zorder=3))
    return xs, x - gap


def _stack(ax, x, y, s, fc, n=3, hatch=None):
    for k in range(n - 1, -1, -1):
        ax.add_patch(Rectangle((x + 0.45 * k, y + 0.45 * k), s, s, fc=fc, ec=INK, lw=0.9, hatch=hatch,
                               zorder=4 + (n - k)))


def _oplus(ax, x, y, r=1.9):
    ax.add_patch(Circle((x, y), r, fc="white", ec=INK, lw=2.0, zorder=5))
    ax.plot([x - r * 0.7, x + r * 0.7], [y, y], color=INK, lw=1.8, zorder=6)
    ax.plot([x, x], [y - r * 0.7, y + r * 0.7], color=INK, lw=1.8, zorder=6)


def _zoom(ax, a, b):
    ax.plot([a[0], b[0]], [a[1], b[1]], color=DASH, lw=2.0, ls=(0, (5, 3)), zorder=1)


ORL_VERSION = "orl/1"     # the proposed model of learning-speed/1 (status doc 21)
ORL_SHORT = {"ObservationQuality": "Q", "VerticalState": "V", "HeadingState": "H",
             "RelativeGeometry": "G", "MotionState": "M", "AttitudeState": "A",
             "ApproachSituation": "P", "ViewGeometry": "O", "ContactCondition": "C",
             "LateralDecision": "Lat", "DescentEligibility": "E", "VerticalDecision": "Z"}


def _orl():
    from ontology_rgat.current_semantics import orl
    return orl


def overall():
    """The whole method in one figure, laid out like an encoder paper figure (ORL, orl/1)."""
    orl = _orl()
    graph = orl.GRAPHS[ORL_VERSION]
    semantic = [e for e in graph.edges if e[2] != orl.SELF]
    dims = orl.input_dimensions()
    fig, ax = canvas(150, 172)
    # ---- top band: scene graph -> ORL actor -> policy -> PX4
    kg = {"UAV": (16.5, 154.0), "Moving\npad": (5.2, 165.5), "Camera": (28.6, 166.5),
          "Navigation\n(PX4 EKF2)": (5.2, 142.5), "Attitude": (28.6, 142.0)}
    edge_lab = {"Moving\npad": "relative x, y, h", "Camera": "vision age",
                "Navigation\n(PX4 EKF2)": "own state, valid", "Attitude": "roll, pitch, rate"}
    for name, (x, y) in kg.items():
        if name != "UAV":
            ax.plot([kg["UAV"][0], x], [kg["UAV"][1], y], color=INK, lw=1.2, zorder=2)
            mx, my = (kg["UAV"][0] + x) / 2, (kg["UAV"][1] + y) / 2
            ax.text(mx + (1.4 if x > 16 else -1.4), my, edge_lab[name], fontsize=9.5,
                    ha="left" if x > 16 else "right", va="center", color=INK,
                    bbox=dict(fc="white", ec="none", pad=0.1), zorder=3)
        r = 5.2 if name == "UAV" else 4.3
        ax.add_patch(Circle((x, y), r, fc=C["kg"][0], ec=C["kg"][1], lw=1.2, zorder=4))
        ax.text(x, y, name, ha="center", va="center", fontsize=8.6 if "\n(" in name else 10.5,
                zorder=5, linespacing=1.0)
    _line(ax, [(34.5, 154.0), (39.0, 154.0)])
    ax.add_patch(FancyBboxPatch((37.5, 138.5), 109.5, 33.0, boxstyle="round,pad=0,rounding_size=3.5",
                                fc=C["off"][0], ec="#555555", lw=1.6, ls=(0, (4, 3)), zorder=1))
    ax.text(41.0, 166.6, "ORL Actor", fontsize=14,
            weight="bold", color=INK, va="center")
    _rect(ax, 39.5, 150.0, 15.5, 8.0, "Semantic Feature\nBuilder  " + r"$\phi$", "sim", size=10.5)
    _ellipse(ax, 62.5, 154.0, 12.0, 7.0, "Typed Node\nEncoders", size=10.5)
    _rect(ax, 71.5, 144.0, 8.5, 24.5, "Relational\nGraph\nAttention\n(hop 1)", "est", size=10.5)
    _rect(ax, 84.0, 144.0, 8.5, 20.5, "Relational\nGraph\nAttention\n(hops 2–3,\nshared)", "know", size=10.5)
    _rect(ax, 96.5, 144.0, 9.0, 24.5, "Decision\nReadouts\n(lateral,\ndescent\ngate)", "pol", size=10.5)
    _ellipse(ax, 124.5, 154.0, 25.0, 8.0, "Mirror average\n" + r"$\frac{1}{4}\sum_g S_g\,f(M_g\phi)$",
             size=10.5)
    for a, b in [((55.0, 154), (56.5, 154)), ((68.5, 154), (71.5, 154)), ((80.0, 154), (84.0, 154)),
                 ((92.5, 154), (96.5, 154)), ((105.5, 154), (112.0, 154))]:
        _line(ax, [a, b])
    # the four mirrored copies share one set of weights (no raw-observation bypass)
    ax.plot([57.0, 57.0, 105.0, 105.0], [143.0, 141.6, 141.6, 143.0], color=DASH, lw=1.4, zorder=2)
    ax.text(81.0, 140.9, r"run on $M_g\phi$ for the 4 reflections $g\in\mathbb{Z}_2\times\mathbb{Z}_2$, "
            "one set of weights; no raw-observation bypass", fontsize=9.8, color=INK, va="top",
            ha="center")
    _line(ax, [(137.0, 154.0), (148.5, 154.0), (148.5, 134.2)])
    _rect(ax, 129.0, 126.5, 19.0, 7.5, "Policy (PPO)", "sim", size=12)
    ax.add_patch(FancyBboxPatch((1.0, 127.0), 123.5, 6.5, boxstyle="round,pad=0", fc=C["kg"][0], ec=INK,
                                lw=1.2, zorder=3))
    ax.text(2.0, 130.25, r"$a_t=(a_x,a_y,a_z)$: world ENU net acceleration $\rightarrow$ roll, pitch and "
            r"thrust (yaw held, tilt $\leq 20^{\circ}$) $\rightarrow$ PX4 OFFBOARD at about 10 Hz $\rightarrow$ "
            "Isaac Sim + Pegasus quadrotor", fontsize=11, va="center", zorder=4)
    _line(ax, [(129.0, 129.0), (124.5, 129.0)])
    _line(ax, [(124.5, 131.6), (129.0, 131.6)], color=MUTED, lw=1.2, size=9)
    _line(ax, [(9.0, 133.5), (9.0, 137.6)], lw=1.4, size=10)
    ax.text(10.0, 135.6, "next decision (0.1 s)", fontsize=9.5, va="center", color=INK)

    # ---- bottom left: the ORL block, drawn as four mirrored copies of one network
    ax.text(1.5, 121.0, "ORL Encoder Block", fontsize=16, weight="bold", color=INK, va="center")
    for k in (2, 1):
        ax.add_patch(FancyBboxPatch((2.0 + 0.8 * k, 10.0 + 0.8 * k), 56.0, 90.0,
                                    boxstyle="round,pad=0,rounding_size=0.3", fc="white", ec=DASH, lw=1.0,
                                    ls=(0, (3, 3)), alpha=0.6, zorder=0.5))
    ax.add_patch(FancyBboxPatch((2.0, 10.0), 56.0, 90.0, boxstyle="round,pad=0,rounding_size=0.3",
                                fc="white", ec=DASH, lw=2.6, ls=(0, (8, 4)), zorder=1))
    ax.text(4.0, 96.4, r"$\times 4$ mirrored copies," + "\none set of weights", fontsize=10.5,
            ha="left", va="center", color=DASH, linespacing=1.2)
    _rect(ax, 8.0, 14.0, 44.0, 14.0, "Semantic Features\n+ Typed Node Encoders", "est", size=15)
    _rect(ax, 8.0, 46.0, 44.0, 14.0, "Relational Graph Attention", "know", size=15)
    _rect(ax, 8.0, 78.0, 44.0, 14.0, "Decision Readouts\n(lateral MLP, descent gate)", "pol", size=15)
    _line(ax, [(30.0, 1.0), (30.0, 14.0)], lw=2.6, size=15)
    ax.text(31.5, 4.0, r"Current observation $O_t$" + "\n(21 channels)", fontsize=13, va="center")
    for a, b in [((30.0, 28.0), (30.0, 46.0)), ((30.0, 60.0), (30.0, 78.0)), ((30.0, 92.0), (30.0, 103.6))]:
        _line(ax, [a, b], lw=2.6, size=15)
    ax.add_patch(FancyArrowPatch((52.0, 57.0), (52.0, 49.0), arrowstyle="-|>", mutation_scale=14,
                                 connectionstyle="arc3,rad=-1.4", color=INK, lw=2.2, zorder=3))
    ax.text(54.6, 41.0, f"{orl.HOPS} hops,\nshared", fontsize=11, ha="center", va="center")
    ax.add_patch(Circle((30.0, 106.0), 2.4, fc="white", ec=INK, lw=2.0, zorder=5))
    ax.text(30.0, 106.0, r"$\frac{1}{4}\Sigma$", fontsize=12, ha="center", va="center", zorder=6)
    ax.text(33.2, 106.0, r"mirror average over $g$", fontsize=12, va="center")
    ax.text(5.6, 56.0, r"input $M_g\,\phi(O_t)$, output mapped back by $S_g$", fontsize=11.5, rotation=90,
            va="center", ha="center")
    _line(ax, [(30.0, 108.4), (30.0, 124.5)], lw=2.6, size=15)
    ax.text(31.5, 113.5, r"$\mu_t\ \rightarrow$ Policy (no raw $O_t$ bypass)", fontsize=13, va="center")

    # ---- right: detail panels with zoom connectors
    panel(ax, 61.5, 86.0, 87.0, 37.5, "Decision Readouts, Mirror Average and Policy")
    panel(ax, 61.5, 45.0, 87.0, 38.5, "Relational Graph Attention (R-GAT)")
    panel(ax, 61.5, 3.0, 87.0, 39.5, "Semantic Features and Typed Node Inputs")
    _zoom(ax, (52.0, 92.0), (61.5, 123.5)); _zoom(ax, (52.0, 78.0), (61.5, 86.0))
    _zoom(ax, (52.0, 60.0), (61.5, 83.5)); _zoom(ax, (52.0, 46.0), (61.5, 45.0))
    _zoom(ax, (52.0, 28.0), (61.5, 42.5)); _zoom(ax, (52.0, 14.0), (61.5, 3.0))

    # panel A: decision readouts -> mirror average -> policy
    e = orl.EMBEDDING
    groups = [(64.5, "LateralDecision x"), (80.0, "LateralDecision y"), (97.0, "VerticalDecision Z"),
              (111.0, "Eligibility E")]
    centres = []
    for x0, name in groups:
        xs, x_end = _circles(ax, x0, 94.1, 4, 0.95, C["pol"][0], ellipsis_at=3)
        ax.text((x0 + x_end) / 2, 91.1, name, fontsize=8.8, ha="center", va="center", color=MUTED)
        centres.append((x0 + x_end) / 2)
    for c in centres:
        _line(ax, [(c, 95.7), (c, 97.5)], lw=1.4, size=9)
    _rect(ax, 64.5, 97.5, 30.0, 4.6, f"Lateral MLP {e}→{orl.LATERAL_HIDDEN}→1\nshared by x and y",
          "pol", size=9.6)
    _rect(ax, 97.0, 97.5, 27.0, 4.6, r"$\mathrm{hold}(Z)+\sigma(\mathrm{gate}(E))\cdot\mathrm{descend}(Z)$",
          "pol", size=9.6)
    for c in (centres[0], centres[1], 110.5):
        _line(ax, [(c, 102.1), (c, 104.1)], lw=1.4, size=9)
    _rect(ax, 64.5, 104.1, 59.5, 4.2, r"$f(M_g\phi)=(\mu_x,\ \mu_y,\ \mu_z)$ of one mirrored copy", "sim",
          size=10)
    _line(ax, [(94.0, 108.3), (94.0, 110.5)], lw=1.4, size=9)
    _rect(ax, 64.5, 110.5, 59.5, 4.8, r"$\mu_t=\frac{1}{4}\sum_{g} S_g\,f(M_g\phi)$: exact mirror equivariance",
          "pol", size=10.2)
    _line(ax, [(124.0, 112.9), (126.4, 112.9)], lw=1.4, size=9)
    ax.text(126.8, 112.9, r"$z_t\sim\mathcal{N}(\mu_t,\mathrm{diag}\,\sigma^2)$", fontsize=10.5, va="center")
    ax.text(126.8, 109.1, r"$a_t=a_{max}\odot\tanh(z_t)$", fontsize=10.5, va="center")
    ax.text(126.8, 105.4, r"$a_{max}=(2.5,2.5,2.0)$ m/s$^2$", fontsize=10, va="center")
    ax.text(126.8, 101.1, "critic $V(O_t)$: MLP (55)\non raw $O_t$, training only", fontsize=9.6,
            va="center", linespacing=1.25)
    ax.text(126.8, 95.6, "actor 2,747 parameters,\n4,016 in total", fontsize=9.6, va="center",
            color=MUTED, linespacing=1.25)

    # panel B: relational attention over the ORL graph
    relations = [r for r in orl.RELATIONS if r != orl.SELF]
    rel_colors = [RELATION_STYLE[r][0] for r in relations] + ["#9A9A9A"]
    for k, col in enumerate(rel_colors):
        ax.add_patch(Rectangle((66.5 + k * 3.2, 48.6), 2.4, 2.4, fc=col, ec=INK, lw=0.8, zorder=4))
    ax.text(66.0, 53.0, r"neighbour states $h_j$ (one colour per relation type, 7 + self)", fontsize=10,
            va="bottom")
    _line(ax, [(73.5, 56.0), (73.5, 59.2)], lw=1.4, size=9)
    ax.text(75.0, 57.6, r"relation-specific projection $W_{r(ij)}$", fontsize=10, va="center")
    for k, col in enumerate(rel_colors[:5]):
        _stack(ax, 66.5 + k * 5.6, 60.0, 3.0, col)
    ax.text(95.4, 61.8, "...", fontsize=12, va="center")
    ax.text(66.0, 65.8, r"messages $m_{ij}=W_{r(ij)}\,h_j$", fontsize=10.5, va="bottom")
    ax.add_patch(Rectangle((140.0, 48.6), 3.0, 3.0, fc="white", ec=INK, lw=1.0, zorder=4))
    ax.text(139.2, 50.1, r"target state $h_i$", fontsize=10.5, ha="right", va="center")
    _line(ax, [(141.5, 51.6), (141.5, 60.2)], lw=1.4, size=9)
    ax.add_patch(FancyBboxPatch((99.0, 60.2), 47.0, 5.6, boxstyle="round,pad=0,rounding_size=0.8",
                                fc="white", ec=INK, lw=1.2, zorder=3))
    ax.text(122.5, 63.0, r"$e_{ij}=\mathrm{LeakyReLU}(a_s^\top m_{ij}+a_d^\top h_i+b_{r(ij)})$", fontsize=11,
            ha="center", va="center", zorder=4)
    ax.plot([80.5, 80.5, 122.5, 122.5], [68.8, 71.2, 71.2, 65.8], color=INK, lw=1.4, zorder=3)
    _line(ax, [(80.5, 71.2), (80.5, 75.7)], lw=1.4, size=9)
    ax.text(84.0, 73.0, "softmax over the edges into $i$, weighted sum", fontsize=10.5, va="center")
    _circles(ax, 70.0, 77.2, 8, 1.1, C["know"][0], ellipsis_at=6)
    ax.text(96.0, 77.2, r"$h_i'=\tanh(\sum_j \alpha_{ij}\,m_{ij}+b)\in\mathbb{R}^{%d}$" % e, fontsize=11.5,
            va="center")
    ax.text(110.5, 53.4, f"self loop on every node; the same layer is\napplied {orl.HOPS} times; "
            "x and y planes share it", fontsize=10, color=MUTED, va="center")

    # panel C: features and typed node inputs
    groups = [("#FFC20E", 4), ("#6E98DA", 3), ("#4FA862", 7), ("#B59AD8", 7)]
    k = 0
    for col, n in groups:
        for _ in range(n):
            ax.add_patch(Rectangle((66.0 + (k % 7) * 2.8, 27.0 - (k // 7) * 2.8), 2.4, 2.4, fc=col, ec=INK,
                                   lw=0.7, zorder=4))
            k += 1
    ax.text(66.0, 31.6, r"current observation $O_t$", fontsize=10.5)
    ax.text(66.0, 16.0, "quality · geometry ·\nmotion · attitude", fontsize=9.5, va="top", color=MUTED)
    _line(ax, [(86.5, 25.0), (90.0, 25.0)], lw=1.6, size=10)
    ax.add_patch(FancyBboxPatch((90.0, 15.5), 26.0, 19.0, boxstyle="round,pad=0,rounding_size=0.8",
                                fc=C["know"][0], ec=C["know"][1], lw=1.6, zorder=3))
    ax.text(103.0, 31.4, r"semantic features $\phi(O_t)$ (44)", fontsize=10.5, ha="center", va="center",
            zorder=4)
    ax.text(103.0, 26.8, r"$B(x;s)=x/(|x|+s)$", fontsize=11.5, ha="center", va="center", zorder=4)
    ax.text(103.0, 22.0, "y plane aligned to x\nby declared signs", fontsize=9.5, ha="center",
            va="center", zorder=4)
    ax.text(103.0, 17.6, r"invalid $\rightarrow$ 0 + flag", fontsize=10, ha="center", va="center", zorder=4)
    _line(ax, [(116.0, 25.0), (119.5, 25.0)], lw=1.6, size=10)
    shared = list(orl.SHARED_ROLES)
    planar = list(orl.PLANE_ROLES)
    ax.text(129.0, 37.2, "x plane", fontsize=9, ha="center", color=MUTED)
    ax.text(140.0, 37.2, "y plane", fontsize=9, ha="center", color=MUTED)
    rows = [(c, "shared") for c in shared] + [(c, "plane") for c in planar] + [("decisions", "decision")]
    for i, (concept, kind) in enumerate(rows):
        yy = 34.0 - i * 2.55
        label = "D" if kind == "decision" else ORL_SHORT[concept]
        ax.text(122.0, yy + 1.0, label, fontsize=9.2, ha="center", va="center")
        if kind == "shared":
            for j in range(dims[concept]):
                ax.add_patch(Rectangle((124.0 + j * 2.05, yy), 1.8, 2.1, fc=C["est"][0], ec=INK, lw=0.5,
                                       zorder=4))
            if concept == "VerticalState":
                ax.text(141.3, yy + 1.0, "shared", fontsize=8.4, color=MUTED, va="center")
        elif kind == "plane":
            for base in (124.0, 135.0):
                for j in range(dims[concept]):
                    ax.add_patch(Rectangle((base + j * 2.05, yy), 1.8, 2.1, fc=C["know"][0], ec=INK,
                                           lw=0.5, zorder=4))
        else:
            for j, name in enumerate(("Lat x", "Lat y", "E", "Z")):
                ax.add_patch(Rectangle((124.0 + j * 5.6, yy), 5.0, 2.1, fc=C["pol"][0], ec=INK, lw=0.5,
                                       zorder=4))
                ax.text(126.5 + j * 5.6, yy + 1.05, name, fontsize=7.6, ha="center", va="center", zorder=5)
    ax.text(134.5, 6.4, f"{len(orl.NODES)} nodes; decision nodes\nstart from a learned state",
            fontsize=9.6, ha="center", va="center", linespacing=1.25)
    ax.text(66.0, 7.0, f"{len(dims)} typed encoders (one per concept, shared by both planes)\n"
            f"{len(semantic)} semantic edges, {len(relations)} relation types",
            fontsize=10, color=MUTED, va="center", linespacing=1.3)
    save(fig, "overall_architecture")


# ----------------------------------------------------------------------------- 1 architecture
def architecture():
    orl = _orl()
    graph = orl.GRAPHS[ORL_VERSION]
    semantic = [e for e in graph.edges if e[2] != orl.SELF]
    relations = [r for r in orl.RELATIONS if r != orl.SELF]
    fig, ax = canvas(140, 86)
    ax.text(1, 83.6, "ORL landing policy on a moving platform: system pipeline",
            fontsize=13.5, weight="bold", color=INK, va="center")
    PW, GAP, X0 = 29.5, 6.0, 1.0
    PX = [X0 + i * (PW + GAP) for i in range(4)]
    py, ph = 28.0, 52.5
    rows = [64.8, 53.2, 41.6, 30.0]
    bh = 9.8
    content = {
        0: ("1  Isaac Sim + Pegasus", "sim", [
            ("Moving landing pad", ["UGV deck with 17-tag ArUco board", "any bearing x heading, accel."]),
            ("Quadrotor + PX4 SITL", ["lockstep physics · OFFBOARD", "attitude + thrust setpoints"]),
            ("Sensors", ["downward camera · IMU · GNSS", "seeded wind and handover kick"]),
            ("Degradation", ["vision dropout · optical noise", "pad acceleration"])]),
        1: ("2  Perception & navigation", "est", [
            ("ArUco detection + PnP", ["ROS 2 camera stream", "capture-time aligned"]),
            ("Pad track", ["relative position, velocity", "re-acquired after rejections"]),
            ("Own navigation state", ["PX4 EKF2 output", "no additional estimator"]),
            (r"Current observation  $O_t$", ["21 channels, no history", "quality · geometry · motion · att."])]),
        2: ("3  Ontology semantics", "know", [
            ("Concepts and relations", [f"{len(orl.input_dimensions())} concepts x 2 planes · "
                                        f"{len(relations)} relation types",
                                        f"{len(semantic)} semantic edges + self loops"]),
            ("Semantic features", ["approach, view geometry,", "contact condition (44 values)"]),
            ("Validity rules", ["no estimate or camera behind", "-> neutral 0 + explicit flag"]),
            (f"Typed node inputs  ({len(orl.NODES)} nodes)", ["x / y planes share encoders",
                                                               "y aligned to x by declared signs"])]),
        3: ("4  Policy & flight control", "pol", [
            ("ORL relational encoder", ["one typed attention layer", f"shared over {orl.HOPS} hops"]),
            ("Decision readouts", ["lateral MLP shared by x and y", "descent gate on eligibility E"]),
            ("Mirror-equivariant actor (PPO)",
             [r"$\mu_t=\frac{1}{4}\sum_g S_g f(M_g\phi)\rightarrow a_t$", "no raw-observation bypass"]),
            ("Attitude adapter", ["roll / pitch / thrust, yaw held", "20 deg command limit -> PX4"])]),
    }
    for i, (title, role, items) in content.items():
        panel(ax, PX[i], py, PW, ph, title, role)
        for (t, lines), y in zip(items, rows):
            box(ax, PX[i] + 1.2, y, PW - 2.4, bh, t, lines, role=role)
    ymid = py + ph / 2 - 2
    arrow(ax, (PX[0] + PW, ymid), (PX[1], ymid), color=C["sim"][1], lw=2.4, size=14)
    arrow(ax, (PX[1] + PW, ymid), (PX[2], ymid), color=C["est"][1], lw=2.4, size=14)
    ax.text((PX[1] + PW + PX[2]) / 2, ymid + 1.6, r"$O_t$", ha="center", fontsize=10, color=C["est"][1])
    k = C["know"][1]
    arrow(ax, (PX[2] + PW - 1.2, rows[0] + bh / 2), (PX[3] + 1.2, rows[0] + bh / 2 + 1.5), color=k, lw=1.7)
    arrow(ax, (PX[2] + PW - 1.2, rows[3] + bh / 2), (PX[3] + 1.2, rows[0] + bh / 2 - 1.5), color=k, lw=1.7)
    ax.text(PX[2] + PW + 0.4, rows[0] + bh / 2 + 2.6, "edges", fontsize=7.8, color=k)
    ax.text(PX[2] + PW + 1.2, 55.5, "nodes", fontsize=7.8, color=k, rotation=78, ha="center", va="center")
    p = C["pol"][1]
    for a, b in zip(rows[:-1], rows[1:]):
        arrow(ax, (PX[3] + PW / 2, a), (PX[3] + PW / 2, b + bh), color=p, lw=1.4)
    # closed loop back to the vehicle
    route(ax, [(PX[3] + PW * 0.75, py), (PX[3] + PW * 0.75, py - 4.6), (PX[0] + PW * 0.5, py - 4.6),
               (PX[0] + PW * 0.5, py)], color=p, lw=1.5, label="setpoints to PX4 (closed loop, ~10 Hz)", lseg=1)

    # offline training band
    by, bh2 = 1.5, 18.0
    ax.add_patch(FancyBboxPatch((1, by), 138, bh2, boxstyle="round,pad=0,rounding_size=1.0", fc="white",
                                ec=C["off"][1], lw=1.5, zorder=1))
    ax.text(2.4, by + bh2 - 1.9, "Training  ·  local plant calibrated to Isaac/PX4 response, then evaluated in "
            "Isaac Sim + PX4 SITL", fontsize=10, weight="bold", color=C["off"][1], va="center")
    ty, th, tw = by + 3.0, 10.0, 24.0
    xs = [3.0, 30.5, 58.0, 85.5, 113.0]
    box(ax, xs[0], ty, tw, th, "Ground truth", ["reward and verdict only;", "never a policy input"], role="gt")
    box(ax, xs[1], ty, tw, th, "Scenario + curriculum", ["bearing x heading draws", "success-gated difficulty"],
        role="off")
    box(ax, xs[2], ty, tw, th, "Rollouts", ["12 complete episodes", "per update, every policy"], role="off")
    box(ax, xs[3], ty, tw, th, "Reward", [r"$\Phi(s')-\Phi(s)$, approach field", "+ terminal verdict table"],
        role="off")
    box(ax, xs[4], ty, tw, th, "PPO update", ["clip 0.2 · GAE 0.95", r"critic $V(O_t)$ on raw $O_t$"],
        role="off")
    for a, b in zip(xs[1:-1], xs[2:]):
        arrow(ax, (a + tw, ty + th / 2), (b, ty + th / 2), color=C["off"][1])
    route(ax, [(xs[0] + tw / 2, ty), (xs[0] + tw / 2, ty - 1.6), (xs[3] + tw / 2, ty - 1.6), (xs[3] + tw / 2, ty)],
          color=C["gt"][1], ls=(0, (3, 2)))
    # the raw observation reaches the critic only (training), never the actor
    route(ax, [(PX[1] + PW * 0.82, py), (PX[1] + PW * 0.82, py - 2.2), (xs[4] + 1.6, py - 2.2),
               (xs[4] + 1.6, ty + th)], color=C["est"][1], lw=1.3, ls=(0, (4, 2)),
          label=r"raw $O_t$ reaches the critic only (training)", lseg=1)
    route(ax, [(xs[4] + tw * 0.6, ty + th), (xs[4] + tw * 0.6, py - 7.0), (PX[3] + PW * 0.35, py - 7.0),
               (PX[3] + PW * 0.35, py)], color=p, ls=(0, (3, 2)), label="learned weights", lseg=1)
    save(fig, "system_architecture")


# ----------------------------------------------------------------------------- 2 model
# ----------------------------------------------------------------------------- 3 graph
def _qpt(p, c, q, t):
    u = 1.0 - t
    return (u * u * p[0] + 2 * u * t * c[0] + t * t * q[0], u * u * p[1] + 2 * u * t * c[1] + t * t * q[1])


def _qcut(p, c, q, centre, radius, t_from, t_to):
    """The curve parameter, moving from ``t_from`` to ``t_to``, where it leaves the circle."""
    lo, hi, prev = t_from, t_to, t_from
    for i in range(1, 41):
        t = t_from + (t_to - t_from) * i / 40
        x = _qpt(p, c, q, t)
        if math.hypot(x[0] - centre[0], x[1] - centre[1]) >= radius:
            lo, hi = prev, t
            break
        prev = t
    for _ in range(24):
        mid = (lo + hi) / 2
        x = _qpt(p, c, q, mid)
        if math.hypot(x[0] - centre[0], x[1] - centre[1]) >= radius:
            hi = mid
        else:
            lo = mid
    return hi


def _route_edge(p, q, obstacles, radius, *, clearance, step, cut_source, cut_target, tries=24):
    """A quadratic edge from node centre ``p`` to ``q``, bent just enough to keep every other
    node centre at least ``clearance`` away (the most clearance found if none does), trimmed
    to the two node circles: (start, control, end) of the drawn curve.  The monitor's encoding
    view routes its edges the same way (monitor.py ``routeEdges``)."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    length = math.hypot(dx, dy)
    nx, ny = -dy / length, dx / length
    mid = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
    best, best_clear = None, -1.0
    for j in range(tries + 1):
        k = 0 if j == 0 else (1 if j % 2 else -1) * math.ceil(j / 2)
        c = (mid[0] + nx * k * step, mid[1] + ny * k * step)
        clear = min((math.hypot(x[0] - o[0], x[1] - o[1])
                     for x in (_qpt(p, c, q, i / 32) for i in range(1, 32)) for o in obstacles),
                    default=math.inf)
        if clear >= clearance:
            best = c
            break
        if clear > best_clear:
            best, best_clear = c, clear
    c = best
    t0 = _qcut(p, c, q, p, cut_source, 0.0, 1.0)
    t1 = _qcut(p, c, q, q, cut_target, 1.0, 0.0)
    start, end = _qpt(p, c, q, t0), _qpt(p, c, q, t1)
    d0 = (2 * (1 - t0) * (c[0] - p[0]) + 2 * t0 * (q[0] - c[0]),
          2 * (1 - t0) * (c[1] - p[1]) + 2 * t0 * (q[1] - c[1]))
    half = (t1 - t0) / 2
    return start, (start[0] + half * d0[0], start[1] + half * d0[1]), end


def graph():
    """The ORL ontology graph: both planes, the shared state concepts and the decision nodes."""
    orl = _orl()
    spec = orl.GRAPHS[ORL_VERSION]
    semantic = [e for e in spec.edges if e[2] != orl.SELF]
    dims = orl.input_dimensions()
    fig, ax = canvas(136, 96)
    ax.text(2, 93.0, f"ORL ontology graph  ({len(orl.NODES)} nodes, {len(set(r for *_, r in semantic))} "
            "relation types; the x and y planes share every weight)", fontsize=14.5, weight="bold",
            color=INK, va="center")
    X1, X2, X3 = 24.0, 62.0, 100.0
    # x plane on top, its mirror image (y plane) below; the shared state concepts sit between
    # the planes on the left, the decision nodes on the right
    pos = {"RelativeGeometry_x": (X1, 79.0), "MotionState_x": (X1, 68.5), "AttitudeState_x": (X1, 58.0),
           "RelativeGeometry_y": (X1, 12.0), "MotionState_y": (X1, 22.5), "AttitudeState_y": (X1, 33.0),
           "ObservationQuality": (X1 - 14.0, 56.0), "VerticalState": (X1 - 14.0, 45.5),
           "HeadingState": (X1 - 14.0, 35.0),
           "ApproachSituation_x": (X2, 78.0), "ViewGeometry_x": (X2, 66.5), "ContactCondition_x": (X2, 55.0),
           "ApproachSituation_y": (X2, 13.0), "ViewGeometry_y": (X2, 24.5), "ContactCondition_y": (X2, 36.0),
           "LateralDecision_x": (X3, 70.0), "LateralDecision_y": (X3, 21.0),
           "DescentEligibility": (X3, 50.5), "VerticalDecision": (X3 + 14.0, 41.0)}
    names = {"ObservationQuality": "Observation\nQuality", "VerticalState": "Vertical\nState",
             "HeadingState": "Heading\nState", "RelativeGeometry": "Relative\nGeometry",
             "MotionState": "Motion\nState", "AttitudeState": "Attitude\nState",
             "ApproachSituation": "Approach\nSituation", "ViewGeometry": "View\nGeometry",
             "ContactCondition": "Contact\nCondition", "LateralDecision": "Lateral\nDecision",
             "DescentEligibility": "Descent\nEligibility", "VerticalDecision": "Vertical\nDecision"}

    def concept(node):
        return node.rsplit("_", 1)[0] if node.endswith(("_x", "_y")) else node

    def role(node):
        c = concept(node)
        if c in orl.SHARED_ROLES:
            return "est"
        if c in ("RelativeGeometry", "MotionState", "AttitudeState"):
            return "kg"
        if c in ("ApproachSituation", "ViewGeometry", "ContactCondition"):
            return "know"
        return "pol"

    R = 4.0
    for x, title in ((X1 - 7.0, "observed state"), (X2, "landing situations"), (X3 + 7.0, "decisions")):
        ax.text(x, 86.6, title, ha="center", fontsize=12.5, weight="bold", color=INK)
    # plane bands
    for (y0, y1, name) in ((50.2, 84.0, "x plane"), (7.0, 41.2, "y plane")):
        ax.add_patch(FancyBboxPatch((X1 - 6.0, y0), X3 + 6.0 - (X1 - 6.0), y1 - y0,
                                    boxstyle="round,pad=0,rounding_size=1.2", fc="#FAFAFA", ec=GRID,
                                    lw=1.0, zorder=0))
        ax.text(X3 + 5.2, y1 - 1.6, name, ha="right", va="center", fontsize=10.5, color=MUTED,
                style="italic")
    for (src, dst, rel) in semantic:
        others = [pos[n] for n in pos if n not in (src, dst)]
        start, control, end = _route_edge(pos[src], pos[dst], others, R, clearance=R + 1.6,
                                          step=1.4, cut_source=R + 0.2, cut_target=R + 0.5)
        color, ls = RELATION_STYLE[rel]
        ax.add_patch(FancyArrowPatch(path=MplPath([start, control, end],
                                                  [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]),
                                     arrowstyle="-|>", mutation_scale=10, color=color,
                                     lw=1.9 if rel in ("informsLanding", "informsVertical") else 1.3,
                                     linestyle=ls, zorder=2, alpha=0.95))
    for n, (x, y) in pos.items():
        fc, ec = C[role(n)]
        ax.add_patch(Circle((x, y), R, fc=fc, ec=ec if role(n) != "kg" else INK, lw=1.6, zorder=4))
        ax.add_patch(Circle((x, y), R + 0.8, fc="none", ec=MUTED, lw=0.8, ls=(0, (1.5, 1.5)), zorder=3))
        c = concept(n)
        text = names[c] + (f" {n[-1]}" if n.endswith(("_x", "_y")) else "")
        ax.text(x, y, text, ha="center", va="center", fontsize=8.2, color=INK, zorder=5, linespacing=1.0)
        if c in dims:
            ax.text(x + R * 0.78, y + R * 0.78, str(dims[c]), fontsize=7.8, color=INK, zorder=6, ha="center",
                    va="center", bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec=INK, lw=0.7))
    counts = defaultdict(int)
    for _, _, r in semantic:
        counts[r] += 1
    lx, ly = 109.5, 30.0
    ax.add_patch(FancyBboxPatch((lx - 1.6, 2.0), 25.5, 31.0, boxstyle="round,pad=0,rounding_size=1.2",
                                fc="white", ec=DASH, lw=1.6, ls=(0, (5, 3)), zorder=1))
    for i, r in enumerate(orl.RELATIONS[:-1]):
        color, ls = RELATION_STYLE[r]
        yy = ly - i * 3.0
        ax.plot([lx, lx + 4.0], [yy, yy], color=color, lw=2.2, ls=ls)
        ax.text(lx + 5.0, yy, f"{r}  ({counts[r]})", fontsize=9.2, color=INK, va="center")
    ax.text(lx, 5.6, "dotted ring = self loop on every\nnode; one attention layer\n"
            f"applied {orl.HOPS} times", fontsize=8.4, color=MUTED, va="center", linespacing=1.25)
    ax.text(2, 1.2, f"Edges read from current_semantics/orl.py ({ORL_VERSION}); the number on a concept is its "
            "input slot count (44 in total). One encoder per concept type serves both planes; decision nodes "
            "have no input.", fontsize=9.2, color=MUTED)
    save(fig, "ontology_graph")


# ----------------------------------------------------------------------------- 4 task + field
def field():
    from ontology_rgat.current_semantics.contracts import load_config
    from ontology_rgat.current_semantics.reward import desired_velocity
    vf = load_config(CONFIG).task.reward.field
    fig = plt.figure(figsize=(14.5, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 1.25], wspace=0.28)
    # (a) scenario geometry, top view
    ax = fig.add_subplot(gs[0])
    ax.set_aspect("equal")
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.85, 1.6)
    ax.set_anchor("N")
    for k in range(4):
        ax.add_patch(Wedge((0, 0), 1.45, 45 + 90 * k, 135 + 90 * k, width=0.0, fc="none", ec=GRID, lw=0.8))
        a = math.radians(45 + 90 * k)
        ax.plot([0, 1.45 * math.cos(a)], [0, 1.45 * math.sin(a)], color=GRID, lw=0.9, ls=(0, (3, 2)))
    ax.add_patch(Circle((0, 0), 0.5, fc=C["est"][0], ec=C["est"][1], lw=1.4))
    ax.add_patch(Rectangle((-0.26, -0.26), 0.52, 0.52, fc="white", ec=C["est"][1], lw=1.0))
    for (dx, dy) in [(-0.13, -0.13), (0.13, -0.13), (-0.13, 0.13), (0.13, 0.13), (0, 0)]:
        ax.add_patch(Rectangle((dx - 0.05, dy - 0.05), 0.1, 0.1, fc=INK, ec="none"))
    B = math.radians(60)
    ax.annotate("", xy=(1.05 * math.cos(B), 1.05 * math.sin(B)), xytext=(0.5 * math.cos(B), 0.5 * math.sin(B)),
                arrowprops=dict(arrowstyle="-|>", color=C["est"][1], lw=1.8))
    ax.text(1.08 * math.cos(B) + 0.04, 1.08 * math.sin(B), "pad heading B\n(+ acceleration)", fontsize=8.4,
            color=C["est"][1], va="bottom")
    A = math.radians(200)
    r = 0.85
    dx, dy = -r * math.cos(A), -r * math.sin(A)
    ax.plot(dx, dy, marker="X", ms=12, color=PROPOSED, mec="white", mew=1.2, zorder=5)
    ax.annotate("", xy=(0, 0), xytext=(dx, dy), arrowprops=dict(arrowstyle="-|>", color=PROPOSED, lw=1.4,
                                                                 ls=(0, (3, 2))))
    ax.text(dx - 0.12, dy - 0.27, "drone at handover", ha="left", fontsize=8.4, color=PROPOSED)
    ax.text(0.15, -0.95, "bearing A to the pad\noffset r in [0.3, 0.9] m", fontsize=8.4, color=PROPOSED)
    ax.set_title("(a) Scenario draw, top view", fontsize=10.5, color=INK, loc="left")
    ax.text(-1.55, -1.82, "A, B: 4 sectors each; 4 of the 16 (A, B)\ncombinations are held out for testing",
            fontsize=7.8, color=MUTED, va="bottom")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    # (b, c) approach field in the vertical plane through the pad (offset along x, pad frame)
    xs = np.linspace(-2.4, 2.4, 17)
    hs = np.linspace(0.1, 3.0, 12)
    for idx, (age, title) in enumerate([(0.0, "(b) Approach field, pad in view"),
                                        (1.2, "(c) Approach field, view lost for >= 1 s")]):
        ax = fig.add_subplot(gs[idx + 1])
        U = np.zeros((len(hs), len(xs)))
        V = np.zeros_like(U)
        for i, h in enumerate(hs):
            for j, x in enumerate(xs):
                v = desired_velocity([x, 0.0], h, vf, vision_age_s=age)
                U[i, j], V[i, j] = v[0], v[2]
        X, Hh = np.meshgrid(xs, hs)
        ax.quiver(X, Hh, U, V, color=PROPOSED if idx == 0 else C["sim"][1], angles="xy", scale_units="xy",
                  scale=4.0, width=0.0042, headwidth=3.6)
        hh = np.linspace(0, 3.1, 50)
        rad = vf.alignment_radius_m + vf.alignment_radius_per_m * hh
        ax.fill_betweenx(hh, -rad, rad, color=C["know"][0], zorder=0)
        ax.plot(rad, hh, color=C["know"][1], lw=1.0, ls=(0, (4, 2)))
        ax.plot(-rad, hh, color=C["know"][1], lw=1.0, ls=(0, (4, 2)))
        ax.add_patch(Rectangle((-0.5, -0.12), 1.0, 0.12, fc=C["est"][1], ec="none"))
        if idx == 0:
            ax.text(0.0, 2.95, "descent cone\n" + r"$r = 0.25 + 0.5\,h$", ha="center", va="top", fontsize=8.4,
                    color=C["know"][1], bbox=dict(fc=C["know"][0], ec="none", pad=0.2))
            ax.text(2.35, 0.25, "align first, then sink\n0.2 m/s at touchdown", ha="right", fontsize=8.4,
                    color=MUTED, bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.3))
        else:
            ax.axhline(vf.lost_view.search_height_m, color=C["sim"][1], lw=1.0, ls=(0, (2, 2)))
            ax.text(-2.35, vf.lost_view.search_height_m + 0.07, "search height 2.5 m", fontsize=8.4,
                    color=C["sim"][1])
            ax.add_patch(Rectangle((-0.75, 0.0), 1.5, 0.5, fc="none", ec=MUTED, lw=1.0, ls=(0, (2, 2))))
            ax.text(0.0, 0.62, "blind touchdown\nallowed", ha="center", fontsize=7.8, color=MUTED,
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.2))
        ax.set_xlim(-2.6, 2.6)
        ax.set_ylim(-0.15, 3.15)
        ax.set_xlabel("horizontal offset from the pad [m]", fontsize=9, color=MUTED)
        ax.set_ylabel("height above contact [m]", fontsize=9, color=MUTED)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        style_axes(ax)
        ax.yaxis.grid(False)
    fig.text(0.5, -0.03, "Arrows: desired drone-minus-pad velocity v_des of the shaping potential "
             "(current_semantics/reward.py). The field shapes the reward only; the policy is not told it.",
             ha="center", fontsize=8.4, color=MUTED)
    save(fig, "task_and_field")


# ----------------------------------------------------------------------------- 5 results
FIGURES = {"overall": overall, "architecture": architecture, "graph": graph, "field": field}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["all", *FIGURES])
    args = parser.parse_args()
    if args.what == "all":
        for f in FIGURES.values():
            f()
    else:
        FIGURES[args.what]()


if __name__ == "__main__":
    main()
