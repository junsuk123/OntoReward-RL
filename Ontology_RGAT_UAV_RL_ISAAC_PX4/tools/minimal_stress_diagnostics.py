#!/usr/bin/env python3
"""Why do the arms differ under stress? Behaviour, imitation error, input shift.

For the teacher and each arm's BC clones, under nominal and the evaluation-only
stress scenarios (minimal/stress.py), on the same held-out seeds:

* outcomes, and for failures the height / offset at the FIRST pad loss
* time share of every supervisor mode
* imitation error |a_arm - a_teacher| where the teacher is run as a SHADOW on
  the arm's own observations (its pad memory and TrackingBias follow the same
  stream), split by whether the pad is visible
* command magnitude and step-to-step change (jerk) of the lateral command,
  and the vertical command
* input shift: fraction of graph feature entries outside the [0.5, 99.5]
  percentile band of NOMINAL teacher flights, per node
* per-node sensitivity: mean |delta a| when one node's 8 features are
  perturbed by N(0, 0.05) -- what each graph reader actually reacts to

    python tools/minimal_stress_diagnostics.py --bc results/minimal_pipeline_20261007/bc \
        --out results/minimal_pipeline_20261007/stress_diagnostics.json
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
ARMS = ("ppo_ontology_rgat", "ppo_semantic_flat", "ppo_vector_canonical")


def episode(job):
    import torch
    torch.set_num_threads(1)
    controller, path, scenario, seed = job
    from ontology_rgat.minimal.arms import load_arm
    from ontology_rgat.minimal.ontology import NODES
    from ontology_rgat.minimal.pad_loss import CLASS_NAMES, LOST, TERMINAL_OCCLUSION, VISIBLE
    from ontology_rgat.minimal.rollout import ArmController, make_env
    from ontology_rgat.minimal.supervisor import MODE_NAMES
    from ontology_rgat.minimal.teacher import MinimalTeacher
    env = make_env(None if scenario == "nominal" else scenario)
    obs, _ = env.reset(seed=seed)
    shadow = MinimalTeacher()
    arm = None
    if path != "teacher":
        _, arm, _ = load_arm(path)
        arm.eval()
    ctl = ArmController(arm)
    rows, done, info = [], False, {}
    first_loss = None
    rng = np.random.default_rng(seed)
    sensitivity = defaultdict(list)
    while not done:
        teacher_action = shadow.act(obs)
        ctl.inputs(obs)
        graph = ctl.last_graph
        if arm is None:
            action = teacher_action
        else:
            with torch.no_grad():
                action = arm(ctl.last_input)[0].numpy().astype(float)
            # Per-node sensitivity on a subsample of states (graph arms only).
            if arm.input_kind in ("graph", "graph_flat") and rng.uniform() < 0.1:
                base = torch.as_tensor(ctl.last_input)
                n_feat = len(NODES) * 8
                with torch.no_grad():
                    for i, node in enumerate(NODES):
                        x = base.clone()
                        x[i * 8:(i + 1) * 8] += torch.as_tensor(rng.normal(0, 0.05, 8), dtype=x.dtype)
                        sensitivity[node].append(float(torch.linalg.norm(arm(x)[0] - arm(base)[0])))
                    x = base.clone()
                    x[n_feat:] = (x[n_feat:] + torch.as_tensor(rng.normal(0, 0.05, len(x) - n_feat),
                                                              dtype=x.dtype)).clamp(0, 1)
                    sensitivity["edge_weights"].append(float(torch.linalg.norm(arm(x)[0] - arm(base)[0])))
        obs, _, done, info = env.step(action)
        truth = np.asarray(info["truth_pad_minus_body"])
        if first_loss is None and info["pad_class"] == LOST:
            first_loss = {"height": -float(truth[2]), "offset": float(np.hypot(*truth[:2])),
                          "t": info["elapsed_s"]}
        rows.append({"mode": MODE_NAMES[info["mode"]], "class": CLASS_NAMES[info["pad_class"]],
                     "visible": bool(obs.pad_detected),
                     "error": float(np.linalg.norm(np.asarray(action) - teacher_action)),
                     "lateral": float(np.hypot(action[0], action[1])), "vertical": float(action[2]),
                     "features": graph.features.tolist()})
    lateral = np.array([r["lateral"] for r in rows])
    actions_xy = np.array([[0.0, 0.0]] + [[r["lateral"], 0.0] for r in rows])
    return {"controller": controller, "scenario": scenario, "seed": seed,
            "status": info["status"], "steps": len(rows), "first_loss": first_loss,
            "modes": dict(Counter(r["mode"] for r in rows)),
            "classes": dict(Counter(r["class"] for r in rows)),
            "error_visible": [r["error"] for r in rows if r["visible"]],
            "error_hidden": [r["error"] for r in rows if not r["visible"]],
            "lateral_mean": float(lateral.mean()),
            "lateral_jerk": float(np.mean(np.abs(np.diff(lateral)))) if len(lateral) > 1 else 0.0,
            "vertical_mean": float(np.mean([r["vertical"] for r in rows])),
            "features": [r["features"] for r in rows[::5]],
            "sensitivity": {k: float(np.mean(v)) for k, v in sensitivity.items()}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bc", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[828, 829, 830])
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--scenarios", nargs="+", default=["nominal", "poor_vision", "strong_wind"])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from ontology_rgat.minimal.ontology import NODES
    episodes = list(range(4100, 4100 + args.episodes))
    controllers = [("teacher", "teacher")] + [
        (f"{arm}__seed{s}", str(args.bc / f"{arm}__seed{s}.pt")) for arm in ARMS for s in args.seeds]
    jobs = [(c, p, sc, e) for c, p in controllers for sc in args.scenarios for e in episodes]
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(episode, jobs, chunksize=4))

    # Nominal input band from the teacher's own nominal flights.
    nominal = np.array([f for r in results if r["controller"] == "teacher" and r["scenario"] == "nominal"
                        for f in r["features"]])
    band = (None if len(nominal) == 0 else
            (np.percentile(nominal, 0.5, axis=0), np.percentile(nominal, 99.5, axis=0)))

    def group(r):
        return ("teacher" if r["controller"] == "teacher" else r["controller"].split("__")[0], r["scenario"])

    table = {}
    for key in sorted({group(r) for r in results}):
        rs = [r for r in results if group(r) == key]
        statuses = Counter(r["status"] for r in rs)
        failures = [r for r in rs if r["status"] != "SUCCESS" and r["first_loss"]]
        modes = Counter()
        for r in rs:
            modes.update(r["modes"])
        total = sum(modes.values())
        feats = np.array([f for r in rs for f in r["features"]])
        outside = (((feats < band[0]) | (feats > band[1])).mean(axis=(0, 2))
                   if len(feats) and band is not None else None)
        sens = defaultdict(list)
        for r in rs:
            for k, v in r["sensitivity"].items():
                sens[k].append(v)
        table[f"{key[0]}|{key[1]}"] = {
            "episodes": len(rs), "statuses": dict(statuses),
            "landing": statuses["SUCCESS"] / len(rs),
            "mode_share": {m: modes[m] / total for m in modes},
            "first_loss_of_failures": {
                "n": len(failures),
                "height_median": float(np.median([f["first_loss"]["height"] for f in failures])) if failures else None,
                "offset_median": float(np.median([f["first_loss"]["offset"] for f in failures])) if failures else None},
            "imitation_error_visible": float(np.mean([e for r in rs for e in r["error_visible"]] or [0])),
            "imitation_error_hidden": float(np.mean([e for r in rs for e in r["error_hidden"]] or [0])),
            "hidden_share": float(np.mean([len(r["error_hidden"]) / max(r["steps"], 1) for r in rs])),
            "lateral_mean": float(np.mean([r["lateral_mean"] for r in rs])),
            "lateral_jerk": float(np.mean([r["lateral_jerk"] for r in rs])),
            "vertical_mean": float(np.mean([r["vertical_mean"] for r in rs])),
            "input_outside_nominal_band": (dict(zip(NODES, map(float, outside)))
                                           if outside is not None else None),
            "sensitivity_per_node": {k: float(np.mean(v)) for k, v in sens.items()} or None,
        }
    args.out.write_text(json.dumps({"seeds": args.seeds, "episodes": [episodes[0], episodes[-1]],
                                    "table": table}, indent=1))
    for key, row in table.items():
        print(f"{key:36s} land {row['landing']:.2f}  err vis/hid {row['imitation_error_visible']:.2f}/"
              f"{row['imitation_error_hidden']:.2f}  hidden {row['hidden_share']:.2f}  lat {row['lateral_mean']:.2f} "
              f"jerk {row['lateral_jerk']:.2f}  vz {row['vertical_mean']:+.2f}  "
              f"modes {', '.join(f'{m[:4]}={v:.2f}' for m, v in sorted(row['mode_share'].items()))}  "
              f"fail-loss h/off {row['first_loss_of_failures']['height_median']}/"
              f"{row['first_loss_of_failures']['offset_median']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
