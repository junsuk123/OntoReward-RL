"""README figures for the ontology R-GAT landing policy (docs/figures/*.png, *.svg).

    python3 tools/make_doc_figures.py extract     # results/ -> docs/figures/figure_data.json
    python3 tools/make_doc_figures.py all         # every figure from figure_data.json + code
    python3 tools/make_doc_figures.py architecture | model | graph | field | effects | results | robustness

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

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "path",
                     "figure.facecolor": "white", "savefig.facecolor": "white"})

# one hue per role: light fill, dark edge
C = {"sim": ("#EAF2FB", "#0B5E9C"), "est": ("#FDF0DA", "#A86A00"), "know": ("#E3F4EC", "#00704F"),
     "pol": ("#F6E8F0", "#8E3F6C"), "off": ("#F2F2F2", "#4D4D4D"), "gt": ("#FDECE3", "#B5470A")}
INK, MUTED, GRID = "#1F1F1F", "#5A5A5A", "#E4E4E4"
PROPOSED, OTHER = "#8E3F6C", "#9A9A9A"
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
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.9", fc=fc, ec=ec,
                                lw=1.2, ls=(0, (4, 3)) if dashed else "-", zorder=3))
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


def panel(ax, x, y, w, h, title, role):
    fc, ec = C[role]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc="white", ec=ec,
                                lw=1.5, zorder=1))
    ax.add_patch(FancyBboxPatch((x + 0.25, y + h - 4.0), w - 0.5, 3.75,
                                boxstyle="round,pad=0,rounding_size=0.8", fc=fc, ec="none", zorder=1.5))
    ax.text(x + 1.3, y + h - 2.1, title, ha="left", va="center", fontsize=11, weight="bold", color=ec,
            zorder=2)


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
    box(ax, xs[2], ty, tw, th, "Rollouts", ["12 complete episodes", "per update, 4 policies"], role="off")
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
def model():
    from ontology_rgat.current_semantics.contracts import NODE_ORDER
    from ontology_rgat.current_semantics.features import FEATURES
    slots = defaultdict(int)
    for f in FEATURES:
        slots[f.node] += 1
    fig, ax = canvas(150, 92)
    ax.text(1, 89.4, "Ontology R-GAT actor-critic at one decision step (0.1 s)", fontsize=13.5,
            weight="bold", color=INK, va="center")
    # node inputs
    panel(ax, 1, 34, 33, 52, "Node inputs  (8 x 10)", "know")
    short = {"ObservationQuality": "Q", "RelativeGeometry": "G", "MotionState": "M",
             "AttitudeState": "A", "ApproachSituation": "P", "ViewGeometry": "O",
             "ContactCondition": "C", "LandingContext": "L"}
    for i, node in enumerate(NODE_ORDER):
        y = 76.0 - i * 5.0
        role = "know" if i < 7 else "pol"
        fc, ec = C[role]
        ax.add_patch(FancyBboxPatch((2.5, y), 30, 4.1, boxstyle="round,pad=0,rounding_size=0.6", fc=fc,
                                    ec=ec, lw=1.0, zorder=3))
        ax.text(4.0, y + 2.05, short[node], fontsize=10, weight="bold", color=ec, va="center", zorder=4)
        ax.text(7.0, y + 2.05, node, fontsize=8.6, color=INK, va="center", zorder=4)
        n = slots.get(node, 0)
        ax.text(31.2, y + 2.05, f"{n} slots" if n else "readout", fontsize=8.0, color=MUTED, va="center",
                ha="right", zorder=4)
    ax.text(17.5, 37.4, "44 semantic values in fixed slots;\nmissing estimate -> 0 + validity flag",
            ha="center", va="center", fontsize=8.0, color=C["know"][1])
    # encoder
    panel(ax, 38, 34, 44, 52, "Relational GAT encoder", "pol")
    box(ax, 40, 70, 40, 9.5, "Node projection", [r"$h_i^{(0)}=\tanh(W_{in}\,x_i),\ \ h\in\mathbb{R}^8$"], role="pol")
    box(ax, 40, 48.5, 40, 18, "Typed relational attention  (x2, shared)", [
        r"$m_{ij}=W_{r(ij)}\,h_j$   (one $W_r$ per relation type)",
        r"$e_{ij}=\mathrm{LeakyReLU}(a_s^\top m_{ij}+a_d^\top h_i+b_{r})$",
        r"$\alpha_{ij}=\mathrm{softmax}_{j\in\mathcal{N}(i)}(e_{ij})$",
        r"$h_i'=\tanh(\sum_j \alpha_{ij}\, m_{ij}+b)$"], role="pol", bsize=8.8)
    box(ax, 40, 36.5, 40, 9.0, "LandingContext readout", [r"$g_t=W_o\,h_L^{(2)}\in\mathbb{R}^8$"], role="pol")
    arrow(ax, (60, 70), (60, 66.5), color=C["pol"][1])
    arrow(ax, (60, 48.5), (60, 45.5), color=C["pol"][1])
    arrow(ax, (34, 60), (38, 60), color=C["know"][1], lw=2.2, size=13)
    # actor-critic
    panel(ax, 86, 34, 38, 52, "Actor-critic", "pol")
    box(ax, 88, 63, 34, 17, "Actor", [r"$\mu_t=\mathrm{MLP}_{66}([\,O_t\,;\,g_t\,])$",
                                      r"$z_t\sim\mathcal{N}(\mu_t,\mathrm{diag}\,\sigma^2)$",
                                      r"$a_t=a_{max}\odot\tanh(z_t)$",
                                      r"$a_{max}=(2.5,\,2.5,\,2.0)\ \mathrm{m/s^2}$"], role="pol")
    box(ax, 88, 48, 34, 11, "Critic", [r"$V_t=\mathrm{MLP}_{55}(O_t)$", "same design for every policy"],
        role="pol")
    ax.text(105, 41.0, "4,016 parameters\n(compared policies: 4,008-4,022)", ha="center", va="center",
            fontsize=8.4, color=MUTED)
    arrow(ax, (82, 41), (88, 66), color=C["pol"][1], lw=1.6)
    ax.text(84.3, 52.5, r"$g_t$", fontsize=10, color=C["pol"][1])
    # output
    box(ax, 128, 56, 21, 22, "Attitude adapter", ["net acceleration ->", "roll, pitch, thrust", "yaw held",
                                                   "tilt <= 20 deg", "-> PX4 OFFBOARD"], role="sim")
    arrow(ax, (122, 71.5), (128, 67), color=C["pol"][1], lw=1.6)
    box(ax, 128, 36, 21, 14, "Raw observation", [r"$O_t\in\mathbb{R}^{21}$", "actor + critic input"], role="est")
    arrow(ax, (128, 43), (122.2, 51), color=C["est"][1], lw=1.4)
    arrow(ax, (132, 50), (122.2, 64.5), color=C["est"][1], lw=1.4)
    # training
    panel(ax, 1, 1.5, 148, 28.5, "Training  (ground truth enters only the reward and the terminal verdict)", "gt")
    ax.text(3.5, 20.0, r"$r_t=\Phi(s_{t+1})-\Phi(s_t)\;-\;(0.3+0.5\,\overline{u_t^2})\,\Delta t\;-\;"
                       r"20\,\kappa_t^2\,\Delta t\;+\;R_{term}$", fontsize=11.5, color=INK, va="center")
    ax.text(3.5, 14.2, r"$\Phi(s)=-(\,2\,\|(e,h)\|+4\,\|v-v_{des}(e,h,\tau_{vis})\|\,)$", fontsize=11.5,
            color=INK, va="center")
    ax.text(3.5, 7.4, r"$v_{des}$: align laterally, sink at 0.2 m/s inside the cone $0.25+0.5h$, "
                      "climb to 2.5 m when the view is lost\n"
                      r"$\kappa_t$: tilt above 10 deg, normalised to the 21 deg envelope;  "
                      r"$u_t=a_t/a_{max}$", fontsize=9.0, color=MUTED, va="center", linespacing=1.5)
    ax.text(92, 20.0, r"$R_{term}$:  SUCCESS +25 · TIMEOUT -12 · ABORT -15 · unsafe -40", fontsize=9.6,
            color=INK, va="center")
    ax.text(92, 14.2, r"PPO: clip 0.2, 8 epochs, GAE $\lambda$ 0.95, $\gamma=e^{-\Delta t/70\,s}$", fontsize=9.6,
            color=INK, va="center")
    ax.text(92, 8.4, "12 complete episodes per update, target KL 0.02,\ncurriculum promoted at 50 % success",
            fontsize=9.0, color=MUTED, va="center", linespacing=1.4)
    save(fig, "policy_model")


# ----------------------------------------------------------------------------- 3 graph
def graph():
    from ontology_rgat.current_semantics.features import FEATURES
    from ontology_rgat.current_semantics.graph import SEMANTIC_EDGES
    slots = defaultdict(int)
    for f in FEATURES:
        slots[f.node] += 1
    fig, ax = canvas(124, 72)
    ax.text(2, 69.5, "Landing ontology graph read by the R-GAT  (8 concepts, 5 relation types)", fontsize=13.5,
            weight="bold", color=INK, va="center")
    pos = {"ObservationQuality": (14, 55), "RelativeGeometry": (14, 41), "MotionState": (14, 27),
           "AttitudeState": (14, 13), "ApproachSituation": (56, 48), "ViewGeometry": (56, 34),
           "ContactCondition": (56, 20), "LandingContext": (96, 34)}
    sub = {"ObservationQuality": "vision / navigation age, validity", "RelativeGeometry": "relative x, y, height",
           "MotionState": "relative, pad and own velocity", "AttitudeState": "roll, pitch, yaw, body rates",
           "ApproachSituation": "approach error and closure", "ViewGeometry": "image position, rate, margins",
           "ContactCondition": "touchdown speed, tilt, rate", "LandingContext": "readout g (8-D), no input"}
    role = {n: ("sim" if p[0] == 14 else "know" if p[0] == 56 else "pol") for n, p in pos.items()}
    W, H = 22.0, 8.6
    for col, (x, title) in {14: (14, "observed state"), 56: (56, "landing situations"),
                            96: (96, "decision context")}.items():
        ax.text(x, 63.0, title, ha="center", fontsize=11, weight="bold", color=C[role[next(
            n for n, p in pos.items() if p[0] == x)]][1])
    for (s, t, r) in SEMANTIC_EDGES:
        (x0, y0), (x1, y1) = pos[s], pos[t]
        color, ls = RELATION_STYLE[r]
        rad = 0.0 if abs(y1 - y0) < 1 else (0.08 if y1 > y0 else -0.08)
        ax.add_patch(FancyArrowPatch((x0 + W / 2, y0), (x1 - W / 2, y1), arrowstyle="-|>", mutation_scale=12,
                                     color=color, lw=2.0 if r == "informsLanding" else 1.6, linestyle=ls,
                                     connectionstyle=f"arc3,rad={rad}", zorder=2, shrinkA=2, shrinkB=2))
    for n, (x, y) in pos.items():
        fc, ec = C[role[n]]
        ax.add_patch(FancyBboxPatch((x - W / 2, y - H / 2), W, H, boxstyle="round,pad=0,rounding_size=1.0",
                                    fc=fc, ec=ec, lw=1.6, zorder=3))
        ax.text(x, y + 1.3, n, ha="center", va="center", fontsize=10, weight="bold", color=INK, zorder=4)
        ax.text(x, y - 1.8, sub[n], ha="center", va="center", fontsize=7.8, color=MUTED, zorder=4)
        if slots.get(n):
            ax.text(x - W / 2 + 0.6, y + H / 2 - 0.9, str(slots[n]), fontsize=7.6, color=ec, zorder=4,
                    va="center")
    # legend
    lx, ly = 83, 21.0
    ax.add_patch(FancyBboxPatch((lx - 1.5, ly - 16.5), 40, 21.5, boxstyle="round,pad=0,rounding_size=1.0",
                                fc="white", ec="#BDBDBD", lw=1.0, zorder=1))
    counts = defaultdict(int)
    for _, _, r in SEMANTIC_EDGES:
        counts[r] += 1
    for i, (r, (color, ls)) in enumerate(RELATION_STYLE.items()):
        yy = ly + 2.2 - i * 3.0
        ax.plot([lx, lx + 6], [yy, yy], color=color, lw=2.0, ls=ls)
        ax.text(lx + 7.2, yy, f"{r}  ({counts[r]})", fontsize=8.6, color=INK, va="center")
    ax.text(lx, ly - 13.6, "+ a self loop on every node (own relation type)\n"
            "one attention layer applied twice: every observed\nconcept reaches LandingContext in exactly 2 hops",
            fontsize=7.8, color=MUTED, va="center", linespacing=1.35)
    ax.text(2, 1.6, "Edges are read from current_semantics/graph.py; the corner number is the node's slot count "
            "(44 in total). The untyped control keeps every edge and erases only the relation names.",
            fontsize=8.0, color=MUTED)
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


FIGURES = {"architecture": architecture, "model": model, "graph": graph, "field": field, "effects": effects,
           "results": results, "robustness": robustness}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["extract", "all", *FIGURES])
    args = parser.parse_args()
    if args.what == "extract":
        extract()
    elif args.what == "all":
        for f in FIGURES.values():
            f()
    else:
        FIGURES[args.what]()


if __name__ == "__main__":
    main()
