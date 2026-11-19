"""README figures for the ontology R-GAT landing policy (docs/figures/*.png, *.svg).

    python3 tools/make_doc_figures.py extract     # results/ -> docs/figures/figure_data.json
    python3 tools/make_doc_figures.py all         # every figure from figure_data.json + code
    python3 tools/make_doc_figures.py overall | architecture | graph | field | effects | results | robustness

Diagrams are drawn from the code (graph edges from current_semantics/graph.py, node slots from
current_semantics/features.py, the approach field from current_semantics/reward.py).  Result
figures read only docs/figures/figure_data.json, which ``extract`` builds from the confirmation
report (20 fresh training seeds), the per-episode Isaac records and the local evaluation summaries
under results/current_semantics_3d (results/ is not versioned, so the extracted numbers are).  White background, English labels, PNG 200 dpi + SVG.
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
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle, Wedge  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parent
sys.path.insert(0, str(PACKAGE / "python"))
OUT = REPO / "docs" / "figures"
DATA = OUT / "figure_data.json"
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
}
ARM_LABEL = {"ontology_rgat_ppo": "Ontology R-GAT\n(proposed)", "semantic_mlp_ppo": "Semantic MLP",
             "raw_ppo": "PPO (raw obs.)", "untyped_gnn_ppo": "Untyped GNN"}
SHORT_LABEL = {"ontology_rgat_ppo": "Ontology\nR-GAT", "semantic_mlp_ppo": "Semantic\nMLP",
               "raw_ppo": "PPO\n(raw obs.)", "untyped_gnn_ppo": "Untyped\nGNN"}
ARMS = ("ontology_rgat_ppo", "semantic_mlp_ppo", "raw_ppo", "untyped_gnn_ppo")
UNSAFE = {"UNSAFE_CONTACT", "SAFETY_ENVELOPE_VIOLATION"}
SEEDS = (10, 11, 12, 13, 14)
WMON_ROOT = RESULTS / "spatial_current_semantics_v1_nav4_5aa4091b"
CONFIRMATION = WMON_ROOT / "confirmation_mini" / "confirmation.json"
COMPARED = ("semantic_mlp_ppo", "untyped_gnn_ppo", "raw_ppo")


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
def _episodes(path):
    rows = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        rows[(r["condition"], r["seed"], r["training_seed"], r["arm"])] = r["status"]
    return rows


def extract():
    isaac_src = {
        "truth": RESULTS / "spatial_current_semantics_v1/isaac_development_2/episodes.jsonl",
        "kalman": RESULTS / "spatial_current_semantics_v1_nav4_489abc2d/isaac_development_2/episodes.jsonl",
        "wmon": RESULTS / "spatial_current_semantics_v1_nav4_5aa4091b/isaac_development/episodes.jsonl",
    }
    local_src = {
        "truth": RESULTS / "spatial_current_semantics_v1/development.json",
        "kalman": RESULTS / "spatial_current_semantics_v1_nav4_489abc2d/development.json",
        "wmon": RESULTS / "spatial_current_semantics_v1_nav4_5aa4091b/development.json",
    }
    rows = {k: _episodes(p) for k, p in isaac_src.items()}
    common = set.intersection(*(set(r) for r in rows.values()))
    isaac = {}
    for nav, rec in rows.items():
        per_arm = {}
        for arm in ARMS:
            keys = [k for k in rec if k[3] == arm]
            statuses = defaultdict(int)
            for k in keys:
                statuses[rec[k]] += 1
            seeds = {}
            for s in SEEDS:
                ks = [k for k in keys if k[2] == s]
                if ks:
                    seeds[str(s)] = sum(rec[k] == "SUCCESS" for k in ks) / len(ks)
            ck = [k for k in common if k[3] == arm]
            per_arm[arm] = {"episodes": len(keys), "success": statuses["SUCCESS"],
                            "unsafe": sum(v for s, v in statuses.items() if s in UNSAFE),
                            "statuses": dict(statuses), "seed_success": seeds,
                            "common_episodes": len(ck),
                            "common_success": sum(rec[k] == "SUCCESS" for k in ck)}
        isaac[nav] = per_arm
    local = {}
    for nav, path in local_src.items():
        models = json.load(open(path, encoding="utf-8"))["models"]
        per_arm = {}
        for arm in ARMS:
            vals, trained = [], []
            for s in SEEDS:
                m = models[f"{s}/{arm}"]
                cells = m.get("cells") or {}
                trained.append(bool(cells))
                vals.append(float(np.mean([c["success"] for c in cells.values()])) if cells else 0.0)
            per_arm[arm] = {"ten_cell_success_per_seed": vals, "trained": trained}
        local[nav] = per_arm
    # development stage, WMON root: the same paired-t intervals the confirmation registers, for
    # comparison only (5 seeds; the confirmation is the test)
    from ontology_rgat.current_semantics.confirmation import paired_interval
    development = {}
    for family, values in (("local_success", {a: local["wmon"][a]["ten_cell_success_per_seed"]
                                              for a in ARMS}),
                           ("isaac_success", {a: [isaac["wmon"][a]["seed_success"][str(s)] for s in SEEDS]
                                              for a in ARMS})):
        development[family] = {a: paired_interval([p - q for p, q in zip(values["ontology_rgat_ppo"],
                                                                         values[a])],
                                                  0.975 if a != "raw_ppo" else 0.95)
                               for a in COMPARED}
    data = {"note": "extracted by tools/make_doc_figures.py extract; development: 5 training seeds "
                    "(10-14) per policy; confirmation: 20 fresh training seeds (20-39)",
            "isaac": isaac, "local": local, "development_intervals": development,
            "confirmation": _confirmation()}
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print("wrote", DATA)


def _confirmation():
    """The registered confirmation report, reduced to what the README and figures quote."""
    if not CONFIRMATION.exists():
        return None
    c = json.load(open(CONFIRMATION, encoding="utf-8"))
    keep = ("n", "mean", "sd", "confidence", "low", "high", "t", "p", "passed", "opened",
            "holm_rejected", "lower_risk")
    trim = lambda d: {k: v for k, v in d.items() if k in keep}
    out = {"plan_hash": c["plan_hash"], "local_complete": c["local"]["complete"],
           "isaac_complete": c["isaac"]["complete"]}
    lv = c["local"]["verdict"]
    if lv:
        out["local"] = {"means": lv["means"], "confirmation_passed": lv["confirmation_passed"],
                        "practical_passed": lv["practical_passed"],
                        "primary": {a: trim(v) for a, v in lv["primary"].items()},
                        "practical": {a: trim(v) for a, v in lv["practical"].items()},
                        "guards": {g: trim(v) for g, v in lv["guards"].items()},
                        "composition_ood": {
                            "means": lv["secondary"]["composition_ood_success"]["means"],
                            "primary": {a: trim(v) for a, v in
                                        lv["secondary"]["composition_ood_success"]["primary"].items()},
                            "practical": {a: trim(v) for a, v in
                                          lv["secondary"]["composition_ood_success"]["practical"].items()}},
                        "learnability": {k: lv["learnability"][k] for k in
                                         ("validated", "with_checkpoint", "models")},
                        "success_by_arm_and_cell": lv["success_by_arm_and_cell"],
                        "unsafe_by_arm_and_cell": lv["unsafe_by_arm_and_cell"],
                        "ten_cell_success_by_seed": lv["ten_cell_success_by_seed"]}
    iv = c["isaac"]["verdict"]
    if iv:
        out["isaac"] = {"means": iv["means"], "isaac_passed": iv["isaac_passed"],
                        "practical_passed": iv["practical_passed"],
                        "primary": {a: trim(v) for a, v in iv["primary"].items()},
                        "practical": {a: trim(v) for a, v in iv["practical"].items()},
                        "risk_family": {a: trim(v) for a, v in iv["risk_family"].items()},
                        "risk_means": iv["risk_means"], "statuses_by_arm": iv["statuses_by_arm"],
                        "flights_per_arm": iv["flights_per_arm"],
                        "success_by_condition": iv["success_by_condition"],
                        "success_by_seed": iv["success_by_seed"], "risk_by_seed": iv["risk_by_seed"],
                        "excluded_models": iv["excluded_models"],
                        "paired_with_local": c["isaac"]["paired_with_local"]}
    return out


def load_data():
    return json.loads(DATA.read_text(encoding="utf-8"))



# ----------------------------------------------------------------------------- 0 overall (paper style)
from matplotlib.patches import Ellipse  # noqa: E402


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


def overall():
    """The whole method in one figure, laid out like an encoder paper figure."""
    from ontology_rgat.current_semantics.contracts import NODE_ORDER
    from ontology_rgat.current_semantics.graph import SEMANTIC_EDGES
    fig, ax = canvas(150, 172)
    # ---- top band: scene graph -> encoder -> actor -> PX4
    kg = {"UAV": (16.5, 154.0), "Moving\npad": (5.2, 165.5), "Camera": (28.6, 166.5),
          "WMON\nnav.": (5.2, 142.5), "Attitude": (28.6, 142.0)}
    edge_lab = {"Moving\npad": "relative x, y, h", "Camera": "vision age", "WMON\nnav.": "valid, age",
                "Attitude": "roll, pitch, rate"}
    for name, (x, y) in kg.items():
        if name != "UAV":
            ax.plot([kg["UAV"][0], x], [kg["UAV"][1], y], color=INK, lw=1.2, zorder=2)
            mx, my = (kg["UAV"][0] + x) / 2, (kg["UAV"][1] + y) / 2
            ax.text(mx + (1.4 if x > 16 else -1.4), my, edge_lab[name], fontsize=9.5,
                    ha="left" if x > 16 else "right", va="center", color=INK,
                    bbox=dict(fc="white", ec="none", pad=0.1), zorder=3)
        r = 5.2 if name == "UAV" else 4.3
        ax.add_patch(Circle((x, y), r, fc=C["kg"][0], ec=C["kg"][1], lw=1.2, zorder=4))
        ax.text(x, y, name, ha="center", va="center", fontsize=10.5, zorder=5, linespacing=1.0)
    _line(ax, [(34.5, 154.0), (39.0, 154.0)])
    ax.add_patch(FancyBboxPatch((37.5, 138.5), 109.5, 33.0, boxstyle="round,pad=0,rounding_size=3.5",
                                fc=C["off"][0], ec="#555555", lw=1.6, ls=(0, (4, 3)), zorder=1))
    ax.text(44.0, 166.6, "Ontology R-GAT Encoder", fontsize=14, weight="bold", color=INK, va="center")
    _rect(ax, 39.5, 150.0, 17.0, 8.0, "Semantic Feature\nBuilder  " + r"$\phi$", "sim", size=11)
    _ellipse(ax, 65.5, 154.0, 13.0, 7.0, "Node\nProjection", size=10.5)
    _rect(ax, 75.5, 144.0, 9.0, 24.5, "Relational\nGraph\nAttention\n(hop 1)", "est", size=10.5)
    _rect(ax, 90.5, 144.0, 9.0, 20.5, "Relational\nGraph\nAttention\n(hop 2,\nshared)", "know", size=10.5)
    _rect(ax, 106.0, 144.0, 9.0, 24.5, "Landing-\nContext\nReadout", "pol", size=10.5)
    _ellipse(ax, 131.0, 154.0, 20.0, 7.0, r"Concat $[\,O_t\,;\,g_t\,]$", size=11)
    for a, b in [((56.5, 154), (59.0, 154)), ((72.0, 154), (75.5, 154)), ((84.5, 154), (90.5, 154)),
                 ((99.5, 154), (106.0, 154)), ((115.0, 154), (121.0, 154))]:
        _line(ax, [a, b])
    _line(ax, [(48.0, 150.0), (48.0, 140.4), (131.0, 140.4), (131.0, 150.5)])
    ax.text(58.0, 140.9, r"raw current observation $O_t$ (21 channels), bypassing the graph", fontsize=10,
            color=INK, va="bottom")
    _line(ax, [(141.0, 154.0), (148.5, 154.0), (148.5, 134.2)])
    _rect(ax, 129.0, 126.5, 19.0, 7.5, "Actor (PPO)", "sim", size=12)
    ax.add_patch(FancyBboxPatch((1.0, 127.0), 123.5, 6.5, boxstyle="round,pad=0", fc=C["kg"][0], ec=INK,
                                lw=1.2, zorder=3))
    ax.text(2.0, 130.25, r"$a_t=(a_x,a_y,a_z)$: world ENU net acceleration $\rightarrow$ roll, pitch and "
            r"thrust (yaw held, tilt $\leq 20^{\circ}$) $\rightarrow$ PX4 OFFBOARD at about 10 Hz $\rightarrow$ "
            "Isaac Sim + Pegasus quadrotor", fontsize=11, va="center", zorder=4)
    _line(ax, [(129.0, 129.0), (124.5, 129.0)])
    _line(ax, [(124.5, 131.6), (129.0, 131.6)], color=MUTED, lw=1.2, size=9)
    _line(ax, [(9.0, 133.5), (9.0, 137.6)], lw=1.4, size=10)
    ax.text(10.0, 135.6, "next decision (0.1 s)", fontsize=9.5, va="center", color=INK)

    # ---- bottom left: the encoder block
    ax.text(1.5, 121.0, "R-GAT Encoder Block", fontsize=16, weight="bold", color=INK, va="center")
    ax.add_patch(FancyBboxPatch((2.0, 10.0), 56.0, 106.5, boxstyle="round,pad=0,rounding_size=0.3",
                                fc="none", ec=DASH, lw=2.6, ls=(0, (8, 4)), zorder=1))
    _rect(ax, 8.0, 14.0, 44.0, 14.0, "Semantic Features\n+ Node Projection", "est", size=15)
    _rect(ax, 8.0, 46.0, 44.0, 14.0, "Relational Graph Attention", "know", size=15)
    _rect(ax, 8.0, 78.0, 44.0, 14.0, "LandingContext Readout", "pol", size=15)
    _line(ax, [(30.0, 1.0), (30.0, 14.0)], lw=2.6, size=15)
    ax.text(31.5, 4.0, r"Current observation $O_t$" + "\n(21 channels)", fontsize=13, va="center")
    for a, b in [((30.0, 28.0), (30.0, 46.0)), ((30.0, 60.0), (30.0, 78.0)), ((30.0, 92.0), (30.0, 102.1))]:
        _line(ax, [a, b], lw=2.6, size=15)
    ax.add_patch(FancyArrowPatch((52.0, 57.0), (52.0, 49.0), arrowstyle="-|>", mutation_scale=14,
                                 connectionstyle="arc3,rad=-1.4", color=INK, lw=2.2, zorder=3))
    ax.text(54.6, 41.0, "2 hops,\nshared", fontsize=11, ha="center", va="center")
    _oplus(ax, 30.0, 104.0)
    _line(ax, [(5.0, 1.0), (5.0, 104.0), (28.1, 104.0)], lw=2.6, size=15)
    ax.text(6.2, 66.0, r"raw $O_t$ bypass", fontsize=12, rotation=90, va="center")
    _line(ax, [(30.0, 105.9), (30.0, 124.5)], lw=2.6, size=15)
    ax.text(31.5, 112.0, r"$[\,O_t\,;\,g_t\,]\ \rightarrow$ Actor", fontsize=13, va="center")

    # ---- right: detail panels with zoom connectors
    panel(ax, 61.5, 86.0, 87.0, 37.5, "LandingContext Readout and Actor")
    panel(ax, 61.5, 45.0, 87.0, 38.5, "Relational Graph Attention (R-GAT)")
    panel(ax, 61.5, 3.0, 87.0, 39.5, "Semantic Feature Builder and Node Inputs")
    _zoom(ax, (52.0, 92.0), (61.5, 123.5)); _zoom(ax, (52.0, 78.0), (61.5, 86.0))
    _zoom(ax, (52.0, 60.0), (61.5, 83.5)); _zoom(ax, (52.0, 46.0), (61.5, 45.0))
    _zoom(ax, (52.0, 28.0), (61.5, 42.5)); _zoom(ax, (52.0, 14.0), (61.5, 3.0))

    # panel A: readout and actor
    _circles(ax, 70.0, 90.8, 8, 1.15, C["pol"][0])
    ax.text(94.4, 90.8, r"state of L after 2 hops, $h_L^{(2)}\in\mathbb{R}^8$", fontsize=10.5, va="center")
    _line(ax, [(79.5, 92.6), (79.5, 95.6)], lw=1.6, size=10)
    _rect(ax, 68.0, 95.6, 23.0, 4.4, r"Readout $g_t=W_o\,h_L^{(2)}$", "pol", size=10.5)
    gx, gend = _circles(ax, 66.0, 104.6, 8, 1.1, "#F4D3C8")
    ox, oend = _circles(ax, 90.0, 104.6, 5, 1.1, C["kg"][0], ellipsis_at=3)
    ax.text(66.0, 107.6, r"$g_t$ (8)", fontsize=10.5); ax.text(90.0, 107.6, r"$O_t$ (21)", fontsize=10.5)
    _line(ax, [(79.5, 100.0), (79.5, 103.0)], lw=1.6, size=10)
    ax.plot([76.0, 76.0, 98.0, 98.0], [106.2, 109.5, 109.5, 106.2], color="#6E98DA", lw=1.2, zorder=3)
    ax.text(99.0, 110.2, "concat", fontsize=10, color="#3A6FC4")
    _line(ax, [(87.0, 109.5), (87.0, 112.0)], lw=1.6, size=10)
    _rect(ax, 77.0, 112.0, 20.0, 5.0, "Actor MLP (66)", "sim", size=10.5)
    _line(ax, [(97.0, 114.5), (110.0, 114.5)], lw=1.6, size=10)
    ax.text(110.6, 116.6, r"$\mu_t \rightarrow z_t\sim\mathcal{N}(\mu_t,\mathrm{diag}\,\sigma^2)$", fontsize=11,
            va="center")
    ax.text(110.6, 112.3, r"$a_t=a_{max}\odot\tanh(z_t)$", fontsize=11, va="center")
    ax.text(110.6, 108.3, r"$a_{max}=(2.5,\,2.5,\,2.0)$ m/s$^2$", fontsize=10.5, va="center")
    ax.text(110.6, 104.4, r"critic $V(O_t)$: MLP (55)", fontsize=10.5, va="center")
    ax.text(110.6, 100.5, "4,016 parameters in total", fontsize=10.5, va="center", color=MUTED)

    # panel B: relational attention
    rel_colors = [c for c, _ in RELATION_STYLE.values()]
    for k, col in enumerate(rel_colors):
        ax.add_patch(Rectangle((66.5 + k * 3.2, 48.6), 2.4, 2.4, fc=col, ec=INK, lw=0.8, zorder=4))
    ax.text(66.0, 53.0, r"neighbour states $h_j$ (one colour per relation type)", fontsize=10, va="bottom")
    _line(ax, [(73.5, 56.0), (73.5, 59.2)], lw=1.4, size=9)
    ax.text(75.0, 57.6, r"relation-specific projection $W_{r(ij)}$", fontsize=10, va="center")
    for k, col in enumerate(rel_colors[:4]):
        _stack(ax, 66.5 + k * 5.6, 60.0, 3.0, col)
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
    _circles(ax, 70.0, 77.2, 8, 1.1, C["know"][0])
    ax.text(93.5, 77.2, r"$h_i'=\tanh(\sum_j \alpha_{ij}\,m_{ij}+b)$", fontsize=11.5, va="center")
    ax.text(120.5, 53.4, "self loop on every node;\nthe same layer is applied twice", fontsize=10, color=MUTED,
            va="center")

    # panel C: features and node inputs
    groups = [("#FFC20E", 4), ("#6E98DA", 3), ("#4FA862", 7), ("#B59AD8", 7)]
    k = 0
    for col, n in groups:
        for _ in range(n):
            ax.add_patch(Rectangle((66.0 + (k % 7) * 2.8, 27.0 - (k // 7) * 2.8), 2.4, 2.4, fc=col, ec=INK,
                                   lw=0.7, zorder=4))
            k += 1
    ax.text(66.0, 31.6, r"current observation $O_t$", fontsize=10.5)
    ax.text(66.0, 16.0, "quality · geometry ·\nmotion · attitude", fontsize=9.5, va="top", color=MUTED)
    _line(ax, [(86.5, 25.0), (91.0, 25.0)], lw=1.6, size=10)
    ax.add_patch(FancyBboxPatch((91.0, 15.5), 26.0, 19.0, boxstyle="round,pad=0,rounding_size=0.8",
                                fc=C["know"][0], ec=C["know"][1], lw=1.6, zorder=3))
    ax.text(104.0, 31.4, r"semantic features $\phi(O_t)$ (44)", fontsize=10.5, ha="center", va="center", zorder=4)
    ax.text(104.0, 26.6, r"$B(x;s)=x/(|x|+s)$", fontsize=11.5, ha="center", va="center", zorder=4)
    ax.text(104.0, 21.6, "approach, view geometry,\ncontact condition", fontsize=9.5, ha="center",
            va="center", zorder=4)
    ax.text(104.0, 17.3, r"invalid $\rightarrow$ 0 + flag", fontsize=10, ha="center", va="center", zorder=4)
    _line(ax, [(117.0, 25.0), (121.5, 25.0)], lw=1.6, size=10)
    short = {"ObservationQuality": "Q", "RelativeGeometry": "G", "MotionState": "M", "AttitudeState": "A",
             "ApproachSituation": "P", "ViewGeometry": "O", "ContactCondition": "C", "LandingContext": "L"}
    for i, node in enumerate(NODE_ORDER):
        yy = 34.0 - i * 2.75
        ax.text(123.6, yy + 1.0, short[node], fontsize=9.5, ha="center", va="center")
        for j in range(10):
            col = C["pol"][0] if node == "LandingContext" else (C["est"][0] if i < 4 else C["know"][0])
            ax.add_patch(Rectangle((125.0 + j * 2.1, yy), 1.85, 2.2, fc=col, ec=INK, lw=0.5, zorder=4))
    ax.text(135.5, 9.5, "node inputs (8 × 10)", fontsize=10.5, ha="center")
    ax.text(66.0, 7.0, f"{len(SEMANTIC_EDGES)} semantic edges, 5 relation types; L has no input slots",
            fontsize=10, color=MUTED)
    save(fig, "overall_architecture")


# ----------------------------------------------------------------------------- 1 architecture
def architecture():
    fig, ax = canvas(140, 86)
    ax.text(1, 83.6, "Ontology R-GAT landing policy on a moving platform: system pipeline",
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
            ("Degradation", ["GNSS outage · vision dropout", "optical noise · pad acceleration"])]),
        1: ("2  Perception & navigation", "est", [
            ("ArUco detection + PnP", ["ROS 2 camera stream", "capture-time aligned"]),
            ("Pad track", ["relative position, velocity", "re-acquired after rejections"]),
            ("Own navigation", ["GNSS + velocity + IMU fusion", "WMON or Kalman, outage-aware"]),
            (r"Current observation  $O_t$", ["21 channels, no history", "quality · geometry · motion · att."])]),
        2: ("3  Ontology semantics", "know", [
            ("Concepts and relations", ["8 concepts · 5 relation types", "14 semantic edges + self loops"]),
            ("Semantic features", ["approach, view geometry,", "contact condition (44 values)"]),
            ("Validity rules", ["no estimate or camera behind", "-> neutral 0 + explicit flag"]),
            ("Node inputs  8 x 10", ["raw slots copied exactly", "+ derived slots + node id"])]),
        3: ("4  Policy & flight control", "pol", [
            ("Relational GAT encoder", ["one typed attention layer", "shared over 2 hops"]),
            ("LandingContext embedding", [r"$g_t \in \mathbb{R}^8$ read from node L", "situation summary"]),
            ("Actor (PPO)", [r"$[\,O_t\,;\,g_t\,] \rightarrow a_t=(a_x,a_y,a_z)$", "world ENU net acceleration"]),
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
    # raw bypass of the current observation into the actor
    route(ax, [(PX[1] + PW * 0.82, py), (PX[1] + PW * 0.82, py - 2.2), (PX[3] - 2.2, py - 2.2),
               (PX[3] - 2.2, rows[2] + bh / 2), (PX[3] + 1.2, rows[2] + bh / 2)],
          color=C["est"][1], lw=1.3, ls=(0, (4, 2)), label=r"raw $O_t$ also enters the actor", lseg=1)
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
    box(ax, xs[4], ty, tw, th, "PPO update", ["clip 0.2 · GAE 0.95", "target KL 0.02"], role="off")
    for a, b in zip(xs[1:-1], xs[2:]):
        arrow(ax, (a + tw, ty + th / 2), (b, ty + th / 2), color=C["off"][1])
    route(ax, [(xs[0] + tw / 2, ty), (xs[0] + tw / 2, ty - 1.6), (xs[3] + tw / 2, ty - 1.6), (xs[3] + tw / 2, ty)],
          color=C["gt"][1], ls=(0, (3, 2)))
    route(ax, [(xs[4] + tw * 0.6, ty + th), (xs[4] + tw * 0.6, py - 7.0), (PX[3] + PW * 0.35, py - 7.0),
               (PX[3] + PW * 0.35, py)], color=p, ls=(0, (3, 2)), label="learned weights", lseg=1)
    save(fig, "system_architecture")


# ----------------------------------------------------------------------------- 2 model
# ----------------------------------------------------------------------------- 3 graph
def graph():
    """The landing ontology as a concept graph: circles for concepts, one colour per relation."""
    from ontology_rgat.current_semantics.features import FEATURES
    from ontology_rgat.current_semantics.graph import SEMANTIC_EDGES
    slots = defaultdict(int)
    for f in FEATURES:
        slots[f.node] += 1
    fig, ax = canvas(124, 76)
    ax.text(2, 73.0, "Landing ontology read by the R-GAT  (8 concepts, 5 relation types)", fontsize=15,
            weight="bold", color=INK, va="center")
    pos = {"ObservationQuality": (28, 58), "RelativeGeometry": (28, 43), "MotionState": (28, 28),
           "AttitudeState": (28, 13), "ApproachSituation": (68, 50.5), "ViewGeometry": (68, 35.5),
           "ContactCondition": (68, 20.5), "LandingContext": (102, 35.5)}
    label = {"ObservationQuality": "Observation\nQuality", "RelativeGeometry": "Relative\nGeometry",
             "MotionState": "Motion\nState", "AttitudeState": "Attitude\nState",
             "ApproachSituation": "Approach\nSituation", "ViewGeometry": "View\nGeometry",
             "ContactCondition": "Contact\nCondition", "LandingContext": "Landing\nContext"}
    sub = {"ObservationQuality": "vision / navigation age, validity", "RelativeGeometry": "relative x, y, height",
           "MotionState": "relative, pad and own velocity", "AttitudeState": "roll, pitch, yaw, body rates",
           "ApproachSituation": "approach error and closure", "ViewGeometry": "image position, rate, margins",
           "ContactCondition": "touchdown speed, tilt, rate", "LandingContext": "readout g (8-D), no input"}
    role = {n: ("kg" if p[0] == 28 else "know" if p[0] == 68 else "pol") for n, p in pos.items()}
    R = 5.4
    for x, title in ((28, "observed state"), (68, "landing situations"), (102, "decision context")):
        ax.text(x, 66.5, title, ha="center", fontsize=12.5, weight="bold", color=INK)
    for (src, dst, rel) in SEMANTIC_EDGES:
        (x0, y0), (x1, y1) = pos[src], pos[dst]
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy)
        a = (x0 + dx / L * R, y0 + dy / L * R)
        b = (x1 - dx / L * (R + 0.3), y1 - dy / L * (R + 0.3))
        color, ls = RELATION_STYLE[rel]
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, color=color,
                                     lw=2.2 if rel == "informsLanding" else 1.7, linestyle=ls,
                                     connectionstyle="arc3,rad=0.0", zorder=2, shrinkA=0, shrinkB=0))
    for n, (x, y) in pos.items():
        fc, ec = C[role[n]]
        ax.add_patch(Circle((x, y), R, fc=fc, ec=ec if role[n] != "kg" else INK, lw=1.8, zorder=4))
        ax.add_patch(Circle((x, y), R + 1.0, fc="none", ec=MUTED, lw=0.9, ls=(0, (1.5, 1.5)), zorder=3))
        ax.text(x, y, label[n], ha="center", va="center", fontsize=10.5, color=INK, zorder=5, linespacing=1.05)
        if slots.get(n):
            ax.text(x + R * 0.78, y + R * 0.78, str(slots[n]), fontsize=9, color=INK, zorder=6, ha="center",
                    va="center", bbox=dict(boxstyle="circle,pad=0.18", fc="white", ec=INK, lw=0.8))
        if role[n] == "kg":
            ax.text(x - R - 1.6, y, sub[n], ha="right", va="center", fontsize=9.5, color=MUTED)
        else:
            ax.text(x, y - R - 1.9, sub[n], ha="center", va="center", fontsize=9.5, color=MUTED,
                    bbox=dict(fc="white", ec="none", pad=0.1), zorder=5)
    lx, ly = 88.0, 20.5
    ax.add_patch(FancyBboxPatch((lx - 1.6, ly - 14.0), 36.5, 17.5, boxstyle="round,pad=0,rounding_size=1.2",
                                fc="white", ec=DASH, lw=1.6, ls=(0, (5, 3)), zorder=1))
    counts = defaultdict(int)
    for _, _, r in SEMANTIC_EDGES:
        counts[r] += 1
    for i, (r, (color, ls)) in enumerate(RELATION_STYLE.items()):
        yy = ly + 1.6 - i * 2.6
        ax.plot([lx, lx + 5.5], [yy, yy], color=color, lw=2.2, ls=ls)
        ax.text(lx + 6.6, yy, f"{r}  ({counts[r]})", fontsize=10, color=INK, va="center")
    ax.text(lx, ly - 11.6, "dotted ring = self loop on every node;\none attention layer applied twice",
            fontsize=9, color=MUTED, va="center", linespacing=1.3)
    ax.text(2, 1.8, "Edges read from current_semantics/graph.py; the number on a concept is its input slot count "
            "(44 in total). The untyped GNN keeps every edge and erases only the relation names.",
            fontsize=9.5, color=MUTED)
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
def results():
    """Isaac Sim + PX4 under WMON navigation: the confirmation (20 fresh seeds) as bars and seed
    dots, the development stage (5 seeds) as hollow diamonds."""
    data = load_data()
    conf = (data.get("confirmation") or {}).get("isaac")
    dev = data["isaac"]["wmon"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.8, 4.5), gridspec_kw=dict(width_ratios=[1.25, 1.0],
                                                                                wspace=0.30))
    x = np.arange(len(ARMS))
    colors = [PROPOSED if a == "ontology_rgat_ppo" else OTHER for a in ARMS]
    dev_succ = [100 * dev[a]["success"] / dev[a]["episodes"] for a in ARMS]
    dev_risk = [100 * dev[a]["unsafe"] / dev[a]["episodes"] for a in ARMS]
    if conf:
        succ = [100 * conf["means"][a] for a in ARMS]
        risk = [100 * conf["risk_means"][a] for a in ARMS]
        flights = conf["flights_per_arm"]["ontology_rgat_ppo"]
        n_seeds = len(conf["success_by_seed"]["ontology_rgat_ppo"])
    else:
        succ, risk, flights, n_seeds = dev_succ, dev_risk, dev["ontology_rgat_ppo"]["episodes"], len(SEEDS)
    a1.bar(x, succ, width=0.56, color=colors, zorder=2)
    rng = np.random.default_rng(3)
    for i, a in enumerate(ARMS):
        ys = [100 * v for v in (conf["success_by_seed"][a].values() if conf else dev[a]["seed_success"].values())]
        a1.scatter(i + rng.uniform(-0.16, 0.16, len(ys)), ys, s=13, color="white", edgecolor=INK, lw=0.7,
                   zorder=3)
        a1.text(i, 3.0, f"{succ[i]:.1f} %", ha="center", va="bottom", fontsize=9.8, color="white",
                weight="bold")
        if conf:
            a1.scatter([i + 0.36], [dev_succ[i]], marker="D", s=30, color="white", edgecolor=INK, lw=1.1,
                       zorder=4)
    a1.set_xticks(x, [SHORT_LABEL[a] for a in ARMS], fontsize=9)
    a1.set_xlim(-0.5, len(ARMS) - 0.35)
    a1.set_ylim(0, 108)
    a1.set_ylabel("landing success [%]", fontsize=9.5, color=MUTED)
    a1.set_title(f"Landing success  (o = one training seed, {flights // n_seeds} flights)", fontsize=10.5,
                 color=INK, loc="left")
    style_axes(a1)
    a2.bar(x, risk, width=0.56, color=colors, zorder=2)
    for i, a in enumerate(ARMS):
        a2.text(i, risk[i] + 0.5, f"{risk[i]:.1f} %", ha="center", va="bottom", fontsize=9.4, color=INK,
                weight="bold" if a == "ontology_rgat_ppo" else "normal")
        if conf:
            a2.scatter([i + 0.36], [dev_risk[i]], marker="D", s=30, color="white", edgecolor=INK, lw=1.1,
                       zorder=4)
    a2.set_xticks(x, [SHORT_LABEL[a] for a in ARMS], fontsize=9)
    a2.set_xlim(-0.5, len(ARMS) - 0.35)
    a2.set_ylim(0, max(risk + dev_risk) * 1.25 + 1)
    a2.set_ylabel("flights with a risk verdict [%]", fontsize=9.5, color=MUTED)
    a2.set_title("Unsafe contact or envelope violation", fontsize=10.5, color=INK, loc="left")
    style_axes(a2)
    fig.suptitle("Isaac Sim + PX4 SITL, own navigation from the WMON estimator with GNSS outages",
                 x=0.06, y=1.04, ha="left", fontsize=12, weight="bold", color=INK)
    if conf:
        fig.text(0.06, -0.06, f"Bars: confirmation, {n_seeds} fresh training seeds x {flights // n_seeds} "
                 f"flights = {flights} flights per policy.   "
                 "Hollow diamond: development stage, 5 seeds x 30 flights (other seeds and environments).",
                 fontsize=8.4, color=MUTED)
    save(fig, "isaac_wmon_results")


def effects():
    """Registered decision of the confirmation: R-GAT minus each policy, per-seed paired-t intervals."""
    data = load_data()
    conf = data["confirmation"]
    devi = data["development_intervals"]
    names = {"semantic_mlp_ppo": "vs Semantic MLP", "untyped_gnn_ppo": "vs Untyped GNN",
             "raw_ppo": "vs PPO (raw obs.)"}
    fig = plt.figure(figsize=(11.0, 7.4))
    a1 = fig.add_axes([0.25, 0.42, 0.40, 0.50])
    a2 = fig.add_axes([0.25, 0.10, 0.40, 0.20])

    def verdict(v, a):
        if v.get("passed"):
            return "confirmed"
        if a == "raw_ppo" and not v.get("opened", True):
            return "not opened"
        return "not confirmed"

    def draw(ax, y, v, good, dev=None):
        col = PROPOSED if good else OTHER
        ax.plot([100 * v["low"], 100 * v["high"]], [y, y], color=col, lw=3.0, solid_capstyle="round", zorder=3)
        ax.scatter([100 * v["mean"]], [y], s=46, color=col, edgecolor="white", lw=1.0, zorder=4)
        if dev is not None:
            ax.scatter([100 * dev["mean"]], [y + 0.3], marker="D", s=22, color="white", edgecolor=INK, lw=0.9,
                       zorder=4)

    def finish(ax, ticks, labels, xlabel, title):
        ax.axvline(0, color=INK, lw=1.0, zorder=2)
        ax.set_yticks(ticks, labels, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=9.2, color=MUTED)
        style_axes(ax)
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")

    n = (conf.get("local") or conf.get("isaac"))["primary"]["semantic_mlp_ppo"]["n"]
    y, ticks, labels = 0.0, [], []
    for title, fam, dkey in (("Local plant: ten-cell mean success", conf.get("local"), "local_success"),
                             ("Isaac + PX4: landing success", conf.get("isaac"), "isaac_success")):
        if fam is None:
            continue
        a1.text(-0.02, y, title, transform=a1.get_yaxis_transform(), ha="right", va="center",
                fontsize=9.6, weight="bold", color=INK)
        y += 1
        for a in COMPARED:
            v = fam["primary"][a] if a in fam["primary"] else fam["practical"][a]
            good = bool(v.get("passed"))
            draw(a1, y, v, good, devi[dkey][a])
            level = f"{100 * v['confidence']:.1f}".rstrip("0").rstrip(".")
            a1.text(1.03, y, f"{100 * v['mean']:+.1f} pp  [{100 * v['low']:+.1f}, {100 * v['high']:+.1f}]  "
                    f"{level} %   {verdict(v, a)}", transform=a1.get_yaxis_transform(), va="center",
                    fontsize=8.6, color=PROPOSED if good else INK, weight="bold" if good else "normal")
            ticks.append(y)
            labels.append(names[a])
            y += 1
        y += 0.5
    a1.set_ylim(y - 0.6, -0.7)
    finish(a1, ticks, labels, "R-GAT minus compared policy, landing success [pp]",
           f"Landing success: registered comparisons, {n} fresh training seeds, paired by seed")
    iv = conf.get("isaac")
    if iv:
        for i, a in enumerate(COMPARED):
            v = iv["risk_family"][a]
            good = bool(v.get("lower_risk"))
            draw(a2, i, v, good)
            a2.text(1.03, i, f"{100 * v['mean']:+.1f} pp  [{100 * v['low']:+.1f}, {100 * v['high']:+.1f}]  95 %   "
                    + ("fewer (Holm)" if good else "not established"), transform=a2.get_yaxis_transform(),
                    va="center", fontsize=8.6, color=PROPOSED if good else INK,
                    weight="bold" if good else "normal")
        a2.set_ylim(len(COMPARED) - 0.5, -0.5)
        finish(a2, list(range(len(COMPARED))), [names[a] for a in COMPARED],
               "R-GAT minus compared policy, risk verdicts [pp]  (negative = fewer)",
               "Isaac + PX4: unsafe contact or envelope violation")
    fig.text(0.02, -0.02, f"Line: paired-t interval of the mean over the {n} per-seed differences (97.5 % for the two "
             "primary comparisons, Bonferroni; 95 % for PPO,\nread only after both primaries pass; risk family "
             "Holm-adjusted).  Dot: mean.  Hollow diamond: development-stage estimate (5 other seeds).\n"
             "Rule and seeds were registered before training: config/current_semantics/confirmation_mini_v1.yaml.",
             fontsize=8.2, color=MUTED, linespacing=1.45, va="top")
    save(fig, "confirmation_effects")


def robustness():
    d = load_data()["isaac"]
    navs = [("truth", "reference\nnavigation"), ("kalman", "Kalman\n(GNSS outages)"),
            ("wmon", "WMON\n(GNSS outages)")]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    markers = {"ontology_rgat_ppo": "o", "semantic_mlp_ppo": "s", "raw_ppo": "^", "untyped_gnn_ppo": "D"}
    x = np.arange(len(navs))
    ends = []
    for a in ARMS:
        ys = [100 * d[n][a]["common_success"] / d[n][a]["common_episodes"] for n, _ in navs]
        prop = a == "ontology_rgat_ppo"
        ax.plot(x, ys, color=PROPOSED if prop else OTHER, lw=2.6 if prop else 1.5, marker=markers[a],
                ms=8.5 if prop else 7, mec="white", mew=1.2, zorder=4 if prop else 3)
        ends.append((ys[-1], a, ys))
    ends.sort()
    last = -99.0
    for y, a, ys in ends:
        yl = max(y, last + 3.4)
        last = yl
        name = ARM_LABEL[a].replace("\n", " ")
        ax.text(x[-1] + 0.08, yl, f"{name}  {y:.1f} %", va="center", fontsize=9,
                color=PROPOSED if a == "ontology_rgat_ppo" else INK,
                weight="bold" if a == "ontology_rgat_ppo" else "normal")
    ax.set_xticks(x, [lab for _, lab in navs], fontsize=9)
    ax.set_xlim(-0.2, 2.95)
    ax.set_ylim(64, 101)
    ax.set_ylabel("landing success in Isaac Sim + PX4 [%]", fontsize=9.5, color=MUTED)
    ax.set_title("Same flights, degrading own-navigation source", fontsize=11.5, color=INK, loc="left",
                 weight="bold")
    style_axes(ax)
    n = d["wmon"]["ontology_rgat_ppo"]["common_episodes"]
    m = d["wmon"]["semantic_mlp_ppo"]["common_episodes"]
    fig.text(0.02, -0.04, f"Identical cases in every column: 30 environments x 5 training seeds = {n} flights per "
             f"policy ({m} for the MLP,\nwhose seeds common to all three profiles are 3).", fontsize=8.2,
             color=MUTED)
    save(fig, "navigation_robustness")


FIGURES = {"overall": overall, "architecture": architecture, "graph": graph, "field": field, "effects": effects,
           "results": results, "robustness": robustness}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["extract", "all", *FIGURES])
    args = parser.parse_args()
    if args.what == "extract":
        extract()
    elif args.what == "all":
        for name, f in FIGURES.items():
            if name == "effects" and not load_data().get("confirmation"):
                print("skip effects: no confirmation report yet")
                continue
            f()
    else:
        FIGURES[args.what]()


if __name__ == "__main__":
    main()
