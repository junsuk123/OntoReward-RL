#!/usr/bin/env python3
"""One-command live Shin/OntoReward training, paired evaluation and reports."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ontology_rgat import stack as stack_module
from ontology_rgat.benchmarks.experiment import (METHODS, configuration_hash,
                                                 load_experiment, paired_seed_plan)
from ontology_rgat.benchmarks.live_env import LiveShinEnvironment
from ontology_rgat.cli import ensure_fastdds
from ontology_rgat.config import default_config
from ontology_rgat.evaluation.shin2026 import write_benchmark_outputs
from ontology_rgat.perception import RosGrayscaleSource
from ontology_rgat.ppo.recurrent import ShinRecurrentActorCritic
from ontology_rgat.ppo.recurrent_train import collect_episode, train_live
from ontology_rgat.reward_modes import (FrozenControlledPotential,
                                        prepare_controlled_rgat_artifact)
from ontology_rgat.stack import ExternalStack
from ontology_rgat.viz.dashboard import Dashboard
from ontology_rgat.viz.live import BenchmarkMonitor, STORE


ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = sorted({key for row in rows for key in row}) if rows else []
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _live_config(mode, results_dir, system_config):
    cfg = default_config(mode, "sitl")
    cfg.sim.dt = 0.1
    cfg.sim.max_time = 30.0
    cfg.sim.max_steps = 300
    cfg.sim.world_xy_limit = 60.0
    cfg.external.control_hz = 10.0
    # Table I does not require the target to begin in view.
    cfg.external.require_pad_in_view = False
    cfg.ppo.gamma = 0.99
    cfg.reward.pbrs.gamma = 0.99
    cfg.paths.system_yaml = str(Path(system_config).resolve())
    cfg.paths.results = str(Path(results_dir).resolve())
    cfg.paths.models = str((Path(results_dir) / "models").resolve())
    cfg.paths.live = str((Path(results_dir) / "live").resolve())
    return cfg


def _build_model(config, device):
    estimator = config.get("estimator") or {}
    torch_device = torch.device(device)
    return ShinRecurrentActorCritic(
        image_embedding=int(estimator.get("image_embedding", 512)),
        lstm_hidden=int(estimator.get("lstm_hidden", 512)),
        latent_dim=int(estimator.get("latent_dimension", 256)),
        actor_hidden=int((config.get("ppo") or {}).get("hidden", 256)),
        critic_hidden=int((config.get("ppo") or {}).get("hidden", 256)),
    ).to(torch_device)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                        default=["shin2026", "ontoreward"])
    parser.add_argument("--reward", choices=METHODS,
                        help="single-method shorthand")
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--config", type=Path,
                        default=ROOT / "config/experiments/shin2026_ablation.yaml")
    parser.add_argument("--system-config", type=Path,
                        default=ROOT / "config/shin2026-system.yaml")
    parser.add_argument("--results-dir", type=Path,
                        help="defaults to results/shin2026/MODE")
    parser.add_argument("--reward-design", type=Path,
                        help="defaults inside RESULTS_DIR/models")
    parser.add_argument("--no-prepare-reward-design", action="store_true",
                        help="require an existing controlled R-GAT artifact")
    parser.add_argument("--train-episodes", type=int,
                        help="override episodes per method")
    parser.add_argument("--eval-episodes", type=int,
                        help="override episodes per scenario and method")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--use-running-stack", action="store_true")
    parser.add_argument("--keep-stack", action="store_true")
    parser.add_argument("--isaac-sim-path")
    parser.add_argument("--isaac-timeout", type=float)
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--dashboard-port", type=int)
    args = parser.parse_args()
    if args.reward:
        args.methods = [args.reward]
    if args.results_dir is None:
        args.results_dir = ROOT / "results/shin2026" / args.mode
    if args.reward_design is None:
        args.reward_design = args.results_dir / "models/rgat_fixed_reward_controlled.json"

    config = load_experiment(args.config)
    config_hash = configuration_hash(config)
    needs_potential = any(name.startswith("ontoreward") for name in args.methods)
    potential = None
    if needs_potential:
        if not args.reward_design.is_file():
            if args.no_prepare_reward_design:
                parser.error(
                    "OntoReward requires a frozen controlled_landing R-GAT artifact: "
                    f"{args.reward_design}. The legacy urban artifact is intentionally rejected.")
            print("Preparing the missing controlled OntoReward R-GAT artifact...")
            prepare_controlled_rgat_artifact(
                args.reward_design, mode=args.mode,
                seed=int((config.get("seeds") or {}).get("model_initialization", 42)))
        potential = FrozenControlledPotential(args.reward_design)

    training_config = config.get("training") or {}
    configured_train_count = int(training_config.get(
        f"episodes_{args.mode}", 8 if args.mode == "quick" else 40960))
    train_count = args.train_episodes or configured_train_count
    scenarios = dict(config.get("evaluation") or {})
    if args.eval_episodes is not None:
        scenarios = {name: args.eval_episodes for name in scenarios}
    elif args.mode == "quick":
        scenarios = {name: (4 if name == "training_random_walk" else 2)
                     for name in scenarios}
    seed0 = int((config.get("seeds") or {}).get("evaluation_start", 5000))
    plan = paired_seed_plan(args.methods, scenarios, seed0)
    seed_config = config.get("seeds") or {}
    model_seed = int(seed_config.get("model_initialization", 42))
    training_seed0 = int(seed_config.get("training_start", 20000))
    args.results_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": str(args.config.resolve()), "config_hash": config_hash,
        "system_config": str(args.system_config.resolve()), "mode": args.mode,
        "methods": args.methods, "training_episodes_per_method": train_count,
        "evaluation": scenarios, "paired_seeds": True,
        "reward_design_id": getattr(potential, "design_id", None),
        "reward_design_sha256": getattr(potential, "sha256", None),
        "trajectory_note": "named evaluation trajectories are documented approximations",
        "table_ii_runtime_application": "incomplete; see docs/SHIN2026_BASELINE.md",
        "execution_status": "live Isaac/Pegasus/PX4 pipeline",
    }
    manifest_path = args.results_dir / "manifest.json"
    episode_path = args.results_dir / "per_episode.csv"
    if episode_path.is_file() and manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("config_hash") != config_hash:
            parser.error("existing evaluation records use a different configuration; "
                         "select a new --results-dir")
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    _write_csv(args.results_dir / "paired_plan.csv", plan)

    ensure_fastdds()
    cfg = _live_config(args.mode, args.results_dir, args.system_config)
    cfg.viz.dashboard.enabled = not args.no_dashboard
    if args.dashboard_port is not None:
        cfg.viz.dashboard.port = int(args.dashboard_port)
    ppo_config = dict(config.get("ppo") or {})
    cfg.reward.pbrs.gamma = float(ppo_config.get("gamma", .99))
    cfg.reward.pbrs["lambda"] = float(ppo_config.get("shaping_lambda", 1.0))
    monitor = BenchmarkMonitor(STORE)
    monitor.configure(
        methods=args.methods, mode=args.mode, config_hash=config_hash,
        training_total=train_count * len(args.methods),
        evaluation_total=len(plan),
        reward_design_id=getattr(potential, "design_id", None),
        reward_design_sha256=getattr(potential, "sha256", None))
    dashboard = Dashboard(cfg, STORE).start()
    owned = None
    stack_module.current(None)
    try:
        monitor.stage("stack startup", "DDS · Isaac Sim · Pegasus · PX4")
        if not args.use_running_stack:
            owned = ExternalStack(
                cfg, isaac_sim_path=args.isaac_sim_path,
                headless=args.headless, config_path=args.system_config,
                isaac_timeout=(args.isaac_timeout if args.isaac_timeout is not None
                               else (600.0 if args.headless else 1200.0)))
            owned.start()
            stack_module.current(owned)
        with RosGrayscaleSource() as camera:
            models = {}
            training_rows = []
            curriculum_raw = dict(config.get("curriculum") or {})
            curriculum_config = {
                "levels": int(curriculum_raw.get("levels", 80)),
                "episodes_per_update": int(curriculum_raw.get("update_every_episodes", 512)),
            }
            for method in args.methods:
                monitor.stage("training", f"recurrent PPO · {method}")
                # Identical initialization is part of the paired comparison.
                torch.manual_seed(model_seed)
                model = _build_model(config, args.device)
                history = train_live(
                    lambda: LiveShinEnvironment(cfg, camera, horizon_steps=300),
                    model, method, range(training_seed0, training_seed0 + train_count),
                    args.results_dir / "models", config_hash=config_hash,
                    potential=potential, ppo=ppo_config,
                    curriculum_config=curriculum_config, monitor=monitor)
                models[method] = model
                training_rows.extend(history)
            _write_csv(args.results_dir / "training_curves.csv", training_rows)

            evaluation_rows = []
            if episode_path.is_file():
                with episode_path.open(newline="", encoding="utf-8") as stream:
                    evaluation_rows = list(csv.DictReader(stream))
            monitor.restore_evaluation(evaluation_rows)
            completed_evaluation = {
                (row["method"], row["scenario"], int(row["seed"]))
                for row in evaluation_rows
            }
            for method, model in models.items():
                monitor.stage("paired evaluation", method)
                with LiveShinEnvironment(cfg, camera, horizon_steps=300) as env:
                    for item in plan:
                        if item["method"] != method:
                            continue
                        key = (method, item["scenario"], int(item["seed"]))
                        if key in completed_evaluation:
                            continue
                        _, metric = collect_episode(
                            env, model, method, int(item["seed"]), curriculum=1.0,
                            potential=potential, deterministic=True,
                            gamma=float(ppo_config.get("gamma", .99)),
                            scenario=item["scenario"], monitor=monitor,
                            phase="evaluation")
                        metric.update({"method": method, "scenario": item["scenario"],
                                       "episode": int(item["seed"]),
                                       "curriculum_level": 80,
                                       "training_sample_efficiency": train_count})
                        evaluation_rows.append(metric)
                        monitor.evaluation_update(method, metric)
                        _write_csv(episode_path, evaluation_rows)
            monitor.stage("reporting", "paired bootstrap · tables · figures")
            outputs = write_benchmark_outputs(evaluation_rows, args.results_dir)
            # The evaluator creates an evaluation-view training file; restore
            # the actual optimization history generated above.
            _write_csv(args.results_dir / "training_curves.csv", training_rows)
            manifest["execution_status"] = "complete"
            manifest["evaluation_episodes"] = len(evaluation_rows)
            manifest["figures"] = outputs["figures"]
            manifest_path.write_text(
                json.dumps(manifest, indent=2), encoding="utf-8")
            monitor.stage("complete", str(args.results_dir.resolve()))
        return 0
    except Exception as exc:
        monitor.stage("failed", f"{type(exc).__name__}: {exc}")
        raise
    finally:
        stack_module.current(None)
        if owned is not None and not args.keep_stack:
            owned.stop()
        if dashboard is not None:
            dashboard.stop()


if __name__ == "__main__":
    raise SystemExit(main())
