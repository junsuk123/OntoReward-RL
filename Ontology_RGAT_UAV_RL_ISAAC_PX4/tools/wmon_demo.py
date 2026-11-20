#!/usr/bin/env python3
"""Fly a trained current-semantics policy and watch it in RViz2 / rqt (``./run.sh wmon-demo``).

Pick the observation profile by config (/3 truth navigation, /4 WMON or Kalman
navigation), a trained model of that root (``SEED/ARM`` from development/1, or
``SEED/bond`` for the bond/1 arm), a development cell (its blackout / pad-acceleration
conditions), and a backend: ``local`` (calibrated plant, seconds, no Isaac) or ``isaac``
(an OWNED Isaac Sim + PX4 SITL + XRCE-DDS + ROS 2 gateway stack, ``--allow-isaac``).
Every decision is published under /wmon_demo (current_semantics/ros_viz.py).

    python tools/wmon_demo.py --profile wmon --model 10/ontology_rgat_ppo --cell dropout_id
    python tools/wmon_demo.py --profile wmon --model 10/bond --backend isaac --allow-isaac
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from ontology_rgat.current_semantics import bond  # noqa: E402
from ontology_rgat.current_semantics import development as dev  # noqa: E402
from ontology_rgat.current_semantics.contracts import contract_manifest, load_config  # noqa: E402
from ontology_rgat.current_semantics.experiment import run_root_for  # noqa: E402
from ontology_rgat.current_semantics.features import build_features  # noqa: E402
from ontology_rgat.current_semantics.models import (build_model, load_checkpoint,  # noqa: E402
                                                    tensors_from_frames)
from ontology_rgat.current_semantics.observation import decode  # noqa: E402
from ontology_rgat.current_semantics.ros_viz import DemoPublisher  # noqa: E402

CONFIGS = {"truth": "spatial_view_contact_v1.yaml",
           "wmon": "spatial_view_contact_v1_nav4_wmon.yaml",
           "kalman": "spatial_view_contact_v1_nav4_kalman.yaml"}
ISAAC_CASES = ROOT / "config" / "current_semantics" / "isaac_development_v1.yaml"


def load_policy(config, plan, root: Path, spec: str):
    seed, arm = spec.split("/")
    if arm == "bond":
        arm, path = bond.ARM, root / "bond" / "development" / f"seed{seed}" / "best.pt"
    else:
        if arm not in plan.arms:
            raise SystemExit(f"--model arm must be one of {plan.arms} or 'bond'")
        path = root / "development" / f"seed{seed}" / arm / "best.pt"
    if not path.exists():
        raise SystemExit(f"no trained checkpoint at {path}")
    acfg = dev.arm_config(config, plan, arm)
    model = build_model(arm, acfg, seed=0)
    load_checkpoint(path, model, manifest=contract_manifest(acfg))
    model.eval()
    return arm, model, path


def default_seeds(plan, cell: str, count: int) -> list[int]:
    import yaml
    cases = yaml.safe_load(ISAAC_CASES.read_text(encoding="utf-8"))["conditions"]
    if cell in cases:
        return list(cases[cell])[:count]
    index = dev.CELLS.index(cell)
    return [plan.eval_seed(0, index, k) for k in range(count)]


def fly(backend, model, config, arm: str, seeds, viz, *, realtime: float, label: str) -> list[str]:
    statuses = []
    for n, seed in enumerate(seeds, 1):
        viz.new_episode()
        observation, info = backend.reset(int(seed))
        t0 = time.monotonic()
        while True:
            raw, flat, nodes = tensors_from_frames([build_features(observation, config)])
            with torch.no_grad():
                out = model.act(raw, flat, nodes, deterministic=True)
                validity = bond.node_validity(nodes)
                evidence = (model.encoder.bond.weights(validity)
                            if arm == bond.ARM else validity)[0].tolist()
            action = out["action"][0].double().numpy()
            step = backend.step(action, latent=out["latent"][0].double().numpy())
            viz.publish(state(backend, observation, step, action, evidence, label,
                              n, len(seeds), statuses))
            if realtime > 0:
                time.sleep(max(0.0, step.transition_s / realtime
                               - (time.monotonic() - t0)))
                t0 = time.monotonic()
            observation = step.observation
            if step.done:
                statuses.append(step.status)
                print(f"wmon-demo: [{n}/{len(seeds)}] seed {seed} -> {step.status} "
                      f"in {step.elapsed_s:.1f} s", flush=True)
                break
    return statuses


def state(backend, observation, step, action, evidence, label, n, total, statuses) -> dict:
    observer = backend.observer
    m = observer.last_measurement
    uav = np.asarray(m.own_position, float)
    own_v = np.asarray(m.own_velocity, float)
    track = observer.track
    pad_est = np.asarray(track.position, float) if track.initialized else uav.copy()
    truth = step.info["evaluator_only"]
    truth_r = np.asarray(truth["relative_position_own_minus_pad_m"], float)
    truth_v = np.asarray(truth["relative_velocity_own_minus_pad_m_s"], float)
    est_v = own_v - np.asarray(track.velocity, float) if track.initialized else own_v
    seen = decode(step.observation)
    vision_age = seen.vision_age_s if math.isfinite(seen.vision_age_s) else 10.0
    nav_age = seen.nav_age_s if math.isfinite(seen.nav_age_s) else 10.0
    nav = getattr(backend, "navigation", None)
    confidence = None
    stream = getattr(getattr(nav, "step", None), "stream", None)
    if stream is not None:
        confidence = [float(v) for v in stream.confidence]
    status = step.status if step.done else ""
    done = sum(s == "SUCCESS" for s in statuses)
    hud = (f"{label}\nepisode {n}/{total}  t={step.elapsed_s:5.1f}s  {status}\n"
           f"vision age {vision_age:4.2f}s  nav {'OK' if seen.nav_valid else 'LOST'} "
           f"({nav_age:4.2f}s)\nlanded {done}/{len(statuses)}")
    rel_err = float(np.linalg.norm((uav - pad_est) - truth_r))
    vel_err = float(np.linalg.norm((est_v - truth_v)[:2]))
    return {"uav": uav, "quat_wxyz": np.asarray(m.quaternion, float), "pad_estimate": pad_est,
            "pad_truth": uav - truth_r, "action": action, "status": status, "hud": hud,
            "rel_pos_error": rel_err, "rel_vel_error": vel_err, "vision_age": vision_age,
            "nav_valid": float(seen.nav_valid), "nav_age": nav_age,
            "wmon_confidence": confidence, "evidence": evidence,
            "telemetry": {"t": float(step.elapsed_s), "status": status,
                          "rel_pos_error_m": rel_err, "rel_vel_error_m_s": vel_err,
                          "vision_age_s": vision_age, "nav_valid": bool(seen.nav_valid),
                          "nav_age_s": nav_age, "action": [float(a) for a in action],
                          "evidence": evidence, "wmon_confidence": confidence,
                          "reward": float(step.reward)}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--profile", choices=tuple(CONFIGS), default="wmon",
                        help="observation: reference navigation (/3), /4 FFT-assisted kinematic WLS (internal id wmon) or /4 constant-acceleration Kalman (CA-KF)")
    parser.add_argument("--model", default="10/ontology_rgat_ppo",
                        help="SEED/ARM of the root's development/1, or SEED/bond")
    parser.add_argument("--cell", default="composition_id", choices=dev.CELLS)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seeds", default=None, help="comma-separated environment seeds")
    parser.add_argument("--backend", choices=("local", "isaac"), default="local")
    parser.add_argument("--allow-isaac", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--realtime", type=float, default=1.0,
                        help="local only: playback speed (1 = real time, 0 = as fast as possible)")
    parser.add_argument("--no-ros", action="store_true")
    args = parser.parse_args(argv)
    config_path = ROOT / "config" / "current_semantics" / CONFIGS[args.profile]
    config = load_config(config_path)
    plan = dev.load_development(dev.default_plan_path(config), config)
    root = run_root_for(config, None)
    arm, model, path = load_policy(config, plan, root, args.model)
    cell = plan.cell(args.cell)
    factors, dropout = dev.cell_conditions(config, asdict(cell), plan.dropout_start_s)
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else default_seeds(plan, args.cell, args.episodes))
    label = f"{args.profile} | {args.model} | {args.cell} | {args.backend}"
    print(f"wmon-demo: {label}\n  checkpoint {path}\n  seeds {seeds}", flush=True)
    from ontology_rgat.current_semantics.ros_viz import _Disabled
    viz = _Disabled() if args.no_ros else DemoPublisher.create()
    try:
        if args.backend == "local":
            from ontology_rgat.current_semantics.backend import make_local_backend
            backend = make_local_backend(config, scenario=cell.split, factors=factors,
                                         dropout=dropout)
            try:
                statuses = fly(backend, model, config, arm, seeds, viz,
                               realtime=args.realtime, label=label)
            finally:
                backend.close()
        else:
            if not args.allow_isaac:
                raise SystemExit("--backend isaac starts an owned Isaac/PX4 stack: "
                                 "add --allow-isaac")
            from ontology_rgat.current_semantics.backend import make_isaac_backend
            from ontology_rgat.spatial.runtime_contract import deployment_profile
            sys.path.insert(0, str(ROOT / "python"))
            from run_spatial_pipeline import live_stack
            stack = ROOT / "results" / "wmon_demo" / "isaac_stack"
            stack.mkdir(parents=True, exist_ok=True)
            with live_stack(stack, adopt=False, headless=args.headless,
                            schema="spatial-reference/12", reset_recoveries=2,
                            profile_path=Path(deployment_profile("spatial-reference/12")["path"])):
                backend = make_isaac_backend(config, allow_isaac=True, split=cell.split,
                                             factors=factors, dropout=dropout)
                try:
                    statuses = fly(backend, model, config, arm, seeds, viz, realtime=0.0,
                                   label=label)
                finally:
                    backend.close()
    finally:
        viz.close()
    print(f"wmon-demo: {sum(s == 'SUCCESS' for s in statuses)}/{len(statuses)} landed "
          f"{statuses}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
