#!/usr/bin/env python3
"""One-command live Shin/OntoReward training, paired evaluation and reports."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "isaac_sim"))

from config_loader import load_config as load_system_config
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
                                        episode_rollout_dataset,
                                        load_rollout_dataset,
                                        merge_rollout_datasets,
                                        prepare_controlled_rgat_artifact,
                                        save_rollout_dataset)
from ontology_rgat.stack import ExternalStack
from ontology_rgat.viz.dashboard import Dashboard
from ontology_rgat.viz.live import BenchmarkMonitor, STORE
from ontology_rgat.viz.rviz import RvizPublisher

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
    simulator_config = load_system_config(Path(system_config))
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
    # RViz is a learner-side process and does not read Isaac's YAML itself.
    # Copy only presentation geometry from the authoritative merged simulator
    # profile so its deck and surveyed campus route match the live world.
    pad = simulator_config.get("pad") or {}
    cfg.viz.rviz.deck_size_m = list(pad.get("deck_size_m", (1.5, 1.5)))
    cfg.viz.rviz.deck_height_m = float(pad.get("deck_height_m", 0.0))
    cfg.viz.rviz.route_waypoints_enu_m = list(
        pad.get("route_waypoints_enu_m", ()))
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


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _start_rviz(enabled: bool) -> tuple[subprocess.Popen | None, object | None]:
    """Open the repository RViz layout and retain its diagnostic log."""
    if not enabled:
        return None, None
    script = ROOT / "scripts/run_rviz.sh"
    if not os.environ.get("DISPLAY"):
        print("WARNING: DISPLAY is unset; continuing without the RViz 2 window.")
        return None, None
    if not script.is_file() or shutil.which("rviz2") is None:
        print("WARNING: RViz 2 is unavailable; continuing without its window.")
        return None, None
    log_path = Path("/tmp/ontology_rgat_stack/rviz.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("wb")
    process = subprocess.Popen(
        [str(script)], cwd=str(ROOT), stdout=stream, stderr=subprocess.STDOUT)
    # Environment/Qt loader errors surface immediately. Do not let a broken
    # optional window abort a publication-scale flight run.
    time.sleep(2.0)
    if process.poll() is not None:
        stream.close()
        print(f"WARNING: RViz 2 exited immediately; see {log_path}.")
        return None, None
    print("RViz 2 opened with the landing camera and flight layout.")
    return process, stream


def _collect_empirical_rgat_data(*, cfg, camera, model, config, config_hash,
                                 checkpoint_path, results_dir, mode, monitor,
                                 episodes_override=None):
    """Fly a trained Shin policy and checkpoint an empirical R-GAT dataset."""
    design = dict(config.get("rgat_design") or {})
    configured_count = design.get(
        f"episodes_{mode}", 8 if mode == "quick" else 400)
    count = int(episodes_override if episodes_override is not None
                else configured_count)
    if count < 2:
        raise ValueError("empirical R-GAT collection needs at least two episodes")
    stride = int(design.get("sample_stride", 3))
    gamma = float(design.get("outcome_discount", (config.get("ppo") or {}).get(
        "gamma", .99)))
    seed_start = int((config.get("seeds") or {}).get("rgat_dataset_start", 70000))
    seeds = list(range(seed_start, seed_start + count))
    checkpoint_sha = _sha256_file(checkpoint_path)
    data_dir = Path(results_dir) / "data"
    dataset_path = data_dir / "rgat_design_rollouts.npz"
    episode_path = data_dir / "rgat_design_episodes.csv"
    dataset = None
    dataset_manifest = None
    episode_rows = []
    completed_seeds: list[int] = []
    if dataset_path.is_file() and dataset_path.with_suffix(".manifest.json").is_file():
        try:
            dataset, dataset_manifest = load_rollout_dataset(
                dataset_path, config_hash=config_hash,
                source_checkpoint_sha256=checkpoint_sha)
            completed_seeds = [int(seed) for seed in
                               dataset_manifest.get("completed_seeds", [])]
            if not set(completed_seeds).issubset(seeds):
                raise ValueError("R-GAT rollout cache uses a different requested seed range")
            if episode_path.is_file():
                with episode_path.open(newline="", encoding="utf-8") as stream:
                    episode_rows = list(csv.DictReader(stream))
            print(f"Resuming empirical R-GAT data: {len(completed_seeds)}/{count} "
                  f"episodes, {len(dataset['y'])} samples.")
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            print(f"Ignoring incompatible R-GAT rollout cache: {exc}")
            dataset = None
            dataset_manifest = None
            completed_seeds = []
            episode_rows = []

    pending = [(index, seed) for index, seed in enumerate(seeds, start=1)
               if seed not in completed_seeds]
    if pending:
        monitor.stage("R-GAT data", "actual Shin-policy Isaac/PX4 rollouts")
        with LiveShinEnvironment(cfg, camera, horizon_steps=300) as env:
            for index, seed in pending:
                curriculum = 1.0 if count == 1 else (index - 1) / (count - 1)
                torch.manual_seed(seed)
                rows, metric = collect_episode(
                    env, model, "shin2026", seed, curriculum=curriculum,
                    deterministic=False, gamma=gamma, monitor=monitor,
                    phase="R-GAT data")
                batch = episode_rollout_dataset(
                    rows, metric, episode=index, seed=seed, gamma=gamma,
                    sample_stride=stride)
                dataset = merge_rollout_datasets(dataset, batch)
                completed_seeds.append(seed)
                metric.update({
                    "method": "rgat_design_shin2026", "episode": index,
                    "scenario": "training_random_walk", "curriculum": curriculum,
                    "rgat_samples": len(batch["y"]),
                })
                episode_rows.append(metric)
                _write_csv(episode_path, episode_rows)
                dataset_manifest = save_rollout_dataset(
                    dataset, dataset_path, config_hash=config_hash,
                    source_checkpoint_sha256=checkpoint_sha,
                    completed_seeds=completed_seeds)
                STORE.set(
                    rgat_dataset_episodes=dataset_manifest["episodes"],
                    rgat_dataset_samples=dataset_manifest["samples"],
                    rgat_dataset_successes=dataset_manifest["successful_episodes"])
                print(f"R-GAT data episode {index}/{count} | "
                      f"contact={int(metric['paper_success'])} | "
                      f"samples={len(batch['y'])} | c={curriculum:.3f}")
    if dataset is None or dataset_manifest is None:
        raise RuntimeError("empirical R-GAT dataset collection produced no data")
    print(f"Empirical R-GAT dataset: {dataset_manifest['samples']} samples from "
          f"{dataset_manifest['episodes']} actual flights; physical contacts "
          f"{dataset_manifest['successful_episodes']}/{dataset_manifest['episodes']}.")
    return dataset, dataset_manifest, dataset_path, checkpoint_sha


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
                        help="require an existing empirical controlled R-GAT artifact")
    parser.add_argument("--rgat-data-episodes", type=int,
                        help="override actual Shin-policy rollouts for R-GAT design")
    parser.add_argument("--rgat-epochs", type=int,
                        help="override empirical R-GAT training epochs")
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
    parser.add_argument("--no-rviz", action="store_true")
    parser.add_argument("--dashboard-port", type=int)
    args = parser.parse_args()
    if args.reward:
        args.methods = [args.reward]
    for option, value in (("--train-episodes", args.train_episodes),
                          ("--eval-episodes", args.eval_episodes),
                          ("--rgat-epochs", args.rgat_epochs)):
        if value is not None and value < 1:
            parser.error(f"{option} must be at least 1")
    if args.rgat_data_episodes is not None and args.rgat_data_episodes < 2:
        parser.error("--rgat-data-episodes must be at least 2")
    if args.results_dir is None:
        args.results_dir = ROOT / "results/shin2026" / args.mode
    if args.reward_design is None:
        args.reward_design = args.results_dir / "models/rgat_fixed_reward_controlled.json"

    config = load_experiment(args.config)
    config_hash = configuration_hash(config)
    needs_potential = any(name.startswith("ontoreward") for name in args.methods)
    potential = None
    artifact_error = None
    if needs_potential and args.reward_design.is_file():
        try:
            potential = FrozenControlledPotential(
                args.reward_design, expected_config_hash=config_hash)
            print(f"Using empirical frozen R-GAT reward design {potential.design_id}.")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            artifact_error = str(exc)
            print(f"Existing reward artifact is not reusable: {exc}")
    if needs_potential and potential is None and args.no_prepare_reward_design:
        parser.error(
            "OntoReward requires a finite empirical Isaac/PX4 R-GAT artifact at "
            f"{args.reward_design}: {artifact_error or 'file not found'}")

    training_config = config.get("training") or {}
    configured_train_count = int(training_config.get(
        f"episodes_{args.mode}", 8 if args.mode == "quick" else 40960))
    train_count = (args.train_episodes if args.train_episodes is not None
                   else configured_train_count)
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
    rviz_publisher = RvizPublisher.create(cfg) if not args.no_rviz else None
    monitor = BenchmarkMonitor(STORE, rviz=rviz_publisher)
    monitor.configure(
        methods=args.methods, mode=args.mode, config_hash=config_hash,
        training_total=train_count * len(args.methods),
        evaluation_total=len(plan),
        reward_design_id=getattr(potential, "design_id", None),
        reward_design_sha256=getattr(potential, "sha256", None))
    dashboard = Dashboard(cfg, STORE).start()
    rviz_process, rviz_log = _start_rviz(
        cfg.viz.rviz.enabled and not args.no_rviz and not args.headless)
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
            training_by_method = {}
            curriculum_raw = dict(config.get("curriculum") or {})
            curriculum_config = {
                "levels": int(curriculum_raw.get("levels", 80)),
                "episodes_per_update": int(curriculum_raw.get("update_every_episodes", 512)),
            }

            def train_requested(method):
                monitor.stage("training", f"recurrent PPO · {method}")
                # Identical initialization is part of the paired comparison.
                torch.manual_seed(model_seed)
                model = _build_model(config, args.device)
                method_potential = potential if method.startswith("ontoreward") else None
                history = train_live(
                    lambda: LiveShinEnvironment(cfg, camera, horizon_steps=300),
                    model, method, range(training_seed0, training_seed0 + train_count),
                    args.results_dir / "models", config_hash=config_hash,
                    potential=method_potential, ppo=ppo_config,
                    curriculum_config=curriculum_config, monitor=monitor)
                models[method] = model
                training_by_method[method] = history
                return model

            if needs_potential and potential is None:
                # Reward design is learned only after a trained Shin policy has
                # generated real camera/PX4 trajectories and physical outcomes.
                if "shin2026" in args.methods:
                    source_model = train_requested("shin2026")
                    source_checkpoint = args.results_dir / "models/shin2026.pt"
                else:
                    monitor.stage("design-source training", "Shin recurrent PPO")
                    torch.manual_seed(model_seed)
                    source_model = _build_model(config, args.device)
                    source_dir = args.results_dir / "models/rgat_design_source"
                    train_live(
                        lambda: LiveShinEnvironment(cfg, camera, horizon_steps=300),
                        source_model, "shin2026",
                        range(training_seed0, training_seed0 + train_count),
                        source_dir, config_hash=config_hash, potential=None,
                        ppo=ppo_config, curriculum_config=curriculum_config,
                        monitor=None)
                    source_checkpoint = source_dir / "shin2026.pt"
                dataset, dataset_manifest, dataset_path, source_sha = (
                    _collect_empirical_rgat_data(
                        cfg=cfg, camera=camera, model=source_model, config=config,
                        config_hash=config_hash, checkpoint_path=source_checkpoint,
                        results_dir=args.results_dir, mode=args.mode,
                        monitor=monitor,
                        episodes_override=args.rgat_data_episodes))
                monitor.stage("R-GAT training", "empirical discounted outcomes")
                _, design_metadata = prepare_controlled_rgat_artifact(
                    args.reward_design, dataset, config_hash=config_hash,
                    source_checkpoint_sha256=source_sha,
                    dataset_path=dataset_path, mode=args.mode, seed=model_seed,
                    epochs=args.rgat_epochs)
                potential = FrozenControlledPotential(
                    args.reward_design, expected_config_hash=config_hash)
                manifest.update(
                    reward_design_id=potential.design_id,
                    reward_design_sha256=potential.sha256,
                    rgat_dataset=dataset_manifest,
                    rgat_design=design_metadata)
                manifest_path.write_text(
                    json.dumps(manifest, indent=2), encoding="utf-8")
                STORE.set(
                    reward_design_id=potential.design_id,
                    reward_design_sha256=potential.sha256,
                    rgat_dataset_episodes=dataset_manifest["episodes"],
                    rgat_dataset_samples=dataset_manifest["samples"],
                    rgat_dataset_successes=dataset_manifest["successful_episodes"])

            for method in args.methods:
                if method not in models:
                    train_requested(method)
            training_rows = [row for method in args.methods
                             for row in training_by_method[method]]
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
        if rviz_process is not None and rviz_process.poll() is None:
            rviz_process.terminate()
            try:
                rviz_process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                rviz_process.kill()
        if rviz_log is not None:
            rviz_log.close()
        if rviz_publisher is not None:
            rviz_publisher.close()
        if dashboard is not None:
            dashboard.stop()


if __name__ == "__main__":
    raise SystemExit(main())
