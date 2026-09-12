#!/usr/bin/env python3
"""One-command controlled shin_se/no_se/onto_no_se live experiment."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "isaac_sim"))

from config_loader import load_config as load_system_config
from ontology_rgat import stack as stack_module
from ontology_rgat.benchmarks.experiment import (
    configuration_hash, controlled_training_seeds, episodes_per_method,
    load_experiment, paired_seed_plan)
from ontology_rgat.benchmarks.live_env import LiveShinEnvironment
from ontology_rgat.cli import ensure_fastdds
from ontology_rgat.curriculum import fitted_update_interval
from ontology_rgat.evaluation import write_three_pipeline_outputs
from ontology_rgat.perception import RosGrayscaleSource, prepare_keypoint_encoder
from ontology_rgat.pipelines import (get_pipeline, primary_pipeline_ids,
                                     validate_pipeline_configuration)
from ontology_rgat.ppo.recurrent_train import collect_episode, train_live
from ontology_rgat.rgat import (
    FrozenSemanticRGATPotential, load_semantic_dataset,
    merge_semantic_datasets, prepare_semantic_rgat_artifact,
    save_semantic_dataset, semantic_episode_dataset)
from ontology_rgat.stack import ExternalStack
from ontology_rgat.viz.dashboard import Dashboard
from ontology_rgat.viz.live import BenchmarkMonitor, STORE
from ontology_rgat.viz.rviz import RvizPublisher
from run_shin2026_pipeline import (_build_model, _live_config, _sha256_file,
                                   _start_rviz, _write_csv)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False),
                         encoding="utf-8")
    os.replace(temporary, path)


def _behavior_transform(variant: int):
    """Estimator-free mixture: no_se action, image servo and bounded noise."""
    def transform(step, policy_action, semantic, rng):
        action = np.asarray(policy_action, dtype=np.float64).copy()
        if variant == 0:
            # Direct image-plane servo. For the configured forward/down camera,
            # optical +x projects primarily onto body-forward and optical +y
            # onto body-right. No metric depth/pose is reconstructed.
            cx, cy = semantic.centroid_xy
            action[0] = np.clip(0.35 * action[0] + 0.75 * cx, -0.8, 0.8)
            action[1] = np.clip(0.35 * action[1] - 0.75 * cy, -0.8, 0.8)
            if semantic.image_alignment > 0.65 and semantic.keypoint_confidence > 0.01:
                scale = semantic.apparent_target_scale
                descent = -0.55 if scale < 0.45 else -0.28 if scale < 0.75 else -0.10
                action[2] = min(0.25 * action[2], descent)
            else:
                action[2] = max(0.25 * action[2], 0.0)
            action[3] *= 0.25
            action += rng.normal(0.0, 0.04, 4)
        elif variant == 1:
            action += rng.normal(0.0, 0.18, 4)
        else:
            # Bounded exploration supplies clear failure-side graph states.
            action = 0.55 * action + 0.45 * rng.uniform(-1.0, 1.0, 4)
        return np.clip(action, -0.9, 0.9)
    return transform


def _read_csv(path: Path):
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _collect_semantic_data(*, cfg, camera, model, config, config_hash,
                           checkpoint_path, results_dir, mode, monitor,
                           episodes_override=None,
                           source_training_episodes=0,
                           source_training_environment_steps=0):
    design = dict(config.get("rgat_design") or {})
    count = int(episodes_override if episodes_override is not None else
                design.get(f"episodes_{mode}", 8 if mode == "quick" else 40))
    if count < 2:
        raise ValueError("semantic R-GAT collection requires at least two episodes")
    stride = int(design.get("sample_stride", 3))
    gamma = float(design.get("outcome_discount", (config.get("ppo") or {}).get(
        "gamma", .99)))
    seed0 = int((config.get("seeds") or {}).get("rgat_dataset_start", 70000))
    requested_seeds = list(range(seed0, seed0 + count))
    dataset_path = Path(results_dir) / "rgat/semantic_rollouts.npz"
    episodes_path = Path(results_dir) / "rgat/semantic_rollout_episodes.csv"
    checkpoint_sha = _sha256_file(checkpoint_path)
    behavior = {
        "name": str(design.get(
            "behavior_policy", "semantic_visual_servo_noisy_no_se_mixture_v1")),
        "source_pipeline": "no_se",
        "state_estimation_enabled": False,
        "components": ["trained_no_se", "image_plane_servo",
                       "bounded_random_exploration"],
        "source_checkpoint_sha256": checkpoint_sha,
        "additional_source_training_episodes": int(source_training_episodes),
        "additional_source_training_environment_steps": int(
            source_training_environment_steps),
        "deterministic_seed_rule": "environment_seed + 9187",
    }
    dataset = None
    manifest = None
    episode_rows = _read_csv(episodes_path)
    completed = []
    if dataset_path.is_file() and dataset_path.with_suffix(".manifest.json").is_file():
        try:
            dataset, manifest = load_semantic_dataset(
                dataset_path, config_hash=config_hash)
            old_behavior = manifest.get("source_behavior_policy") or {}
            if old_behavior != behavior:
                raise ValueError("semantic rollout behavior-policy provenance mismatch")
            completed = [int(seed) for seed in manifest.get("completed_seeds", [])]
            if not set(completed).issubset(requested_seeds):
                raise ValueError("semantic rollout cache uses a different seed range")
            print(f"Resuming semantic R-GAT data: {len(completed)}/{count} episodes.")
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            print(f"Ignoring incompatible semantic rollout cache: {exc}")
            dataset, manifest, episode_rows, completed = None, None, [], []

    pending = [(index, seed) for index, seed in enumerate(requested_seeds, start=1)
               if seed not in completed]
    if pending:
        monitor.stage("R-GAT data", "estimator-free semantic behavior mixture")
        with LiveShinEnvironment(
                cfg, camera, horizon_steps=int(cfg.sim.max_steps)) as environment:
            for episode, seed in pending:
                rows, metric = collect_episode(
                    environment, model, "no_se", seed, curriculum=1.0,
                    deterministic=False, gamma=gamma,
                    scenario="training_random_walk", monitor=monitor,
                    phase="semantic reward-design data",
                    action_transform=_behavior_transform((episode - 1) % 3))
                # Only the explicitly whitelisted semantic matrices cross into
                # the dataset builder. PPO critic truth remains in ``rows`` but
                # is neither passed nor serialised here.
                semantic_samples = [
                    {"graph_X": row["semantic_graph_X"], "step_id": index}
                    for index, row in enumerate(rows)
                ]
                current = semantic_episode_dataset(
                    semantic_samples, success=bool(metric["paper_success"]),
                    episode_id=episode, seed=seed, gamma_design=gamma,
                    sample_stride=stride)
                dataset = merge_semantic_datasets(dataset, current)
                completed.append(seed)
                episode_rows.append({
                    "episode": episode, "seed": seed,
                    "behavior_component": (episode - 1) % 3,
                    "paper_success": metric["paper_success"],
                    "status": metric["status"], "steps": metric["steps"],
                })
                manifest = save_semantic_dataset(
                    dataset, dataset_path, config_hash=config_hash,
                    source_behavior_policy=behavior, completed_seeds=completed,
                    environment_steps=sum(
                        int(float(row.get("steps", 0))) for row in episode_rows))
                _write_csv(episodes_path, episode_rows)
                print(f"semantic data {len(completed)}/{count}: "
                      f"success={int(metric['paper_success'])} samples={len(current['y'])}")
    if dataset is None or manifest is None:
        raise RuntimeError("semantic reward-design dataset is unavailable")
    successes = int(manifest["successful_episodes"])
    if successes == 0 or successes == int(manifest["episodes"]):
        raise RuntimeError(
            "semantic reward-design data contains only one terminal class; "
            "collect additional estimator-free behavior trajectories")
    total_steps = sum(int(float(row.get("steps", 0))) for row in episode_rows)
    return dataset, manifest, dataset_path, total_steps


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--experiment", choices=("three_pipeline",),
                        default="three_pipeline")
    parser.add_argument("--pipelines", nargs="+", choices=primary_pipeline_ids(),
                        default=list(primary_pipeline_ids()))
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--config", type=Path,
                        default=ROOT / "config/experiments/three_pipeline_comparison.yaml")
    parser.add_argument("--system-config", type=Path,
                        default=ROOT / "config/shin2026-system.yaml")
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--reward-design", type=Path)
    parser.add_argument("--no-prepare-reward-design", action="store_true")
    parser.add_argument("--rgat-data-episodes", type=int)
    parser.add_argument("--rgat-epochs", type=int)
    parser.add_argument("--train-episodes", type=int)
    parser.add_argument("--total-train-episodes", type=int)
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument(
        "--training-replicate", type=int, default=0,
        help="index into training.independent_seeds_full for publication runs")
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
    if args.train_episodes is not None and args.total_train_episodes is not None:
        parser.error("use either --train-episodes or --total-train-episodes")
    for name in ("train_episodes", "total_train_episodes", "eval_episodes",
                 "rgat_epochs"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.rgat_data_episodes is not None and args.rgat_data_episodes < 2:
        parser.error("--rgat-data-episodes must be at least two")
    if args.training_replicate < 0:
        parser.error("--training-replicate must be non-negative")

    config = load_experiment(args.config)
    validate_pipeline_configuration(config)
    configured = tuple(config.get("pipelines") or ())
    if any(name not in configured for name in args.pipelines):
        parser.error("selected pipeline is absent from experiment configuration")
    for name in args.pipelines:
        get_pipeline(name)
    if args.results_dir is None:
        args.results_dir = ROOT / "results/three_pipeline" / args.mode
        if args.training_replicate:
            args.results_dir /= f"replicate_{args.training_replicate}"
    if args.reward_design is None:
        args.reward_design = args.results_dir / "rgat/rgat_model.pt"
    args.results_dir.mkdir(parents=True, exist_ok=True)

    system = load_system_config(args.system_config)
    training_cfg = dict(config.get("training") or {})
    seed_cfg = dict(config.get("seeds") or {})
    base_model_seed = int(seed_cfg.get("model_initialization", 42))
    base_training_seed = int(seed_cfg.get("training_start", 20000))
    if args.mode == "full":
        independent_model_seeds = list(training_cfg.get(
            "independent_seeds_full", [base_model_seed]))
        independent_training_starts = list(seed_cfg.get(
            "independent_training_starts_full",
            [base_training_seed + 100000 * index
             for index in range(len(independent_model_seeds))]))
    else:
        independent_model_seeds = [base_model_seed]
        independent_training_starts = [base_training_seed]
    if len(independent_training_starts) != len(independent_model_seeds):
        parser.error("independent model/training seed lists must have equal length")
    if args.training_replicate >= len(independent_model_seeds):
        parser.error(
            f"--training-replicate {args.training_replicate} is unavailable in "
            f"{args.mode} mode (configured replicates: {len(independent_model_seeds)})")
    model_seed = int(independent_model_seeds[args.training_replicate])
    training_seed0 = int(independent_training_starts[args.training_replicate])
    config_hash = configuration_hash({
        "experiment": config, "system": system,
        "training_replicate": args.training_replicate,
        "model_seed": model_seed, "training_seed_start": training_seed0,
    })
    configured_count = int(training_cfg.get(
        f"episodes_{args.mode}", 8 if args.mode == "quick" else 40960))
    configured_ppo = dict(config.get("ppo") or {})
    selected_warmup = (
        int(configured_ppo.get(
            f"perception_warmup_episodes_{args.mode}",
            2 if args.mode == "quick" else 8))
        if "shin_se" in args.pipelines else 0)
    if args.total_train_episodes is not None:
        try:
            train_count = episodes_per_method(
                args.total_train_episodes, len(args.pipelines), selected_warmup)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        train_count = int(args.train_episodes or configured_count)
    evaluation_cfg = dict(config.get("evaluation") or {})
    if args.eval_episodes is not None:
        evaluation_cfg = {name: args.eval_episodes for name in evaluation_cfg}
    elif args.mode == "quick":
        evaluation_cfg = {name: (4 if name == "training_random_walk" else 2)
                          for name in evaluation_cfg}
    warmup_seed0 = int(seed_cfg.get(
        "estimator_warmup_start", base_training_seed - selected_warmup))
    warmup_seed0 += 100000 * args.training_replicate
    plan = paired_seed_plan(args.pipelines, evaluation_cfg,
                            int(seed_cfg.get("evaluation_start", 5000)))

    keypoint_pretraining = prepare_keypoint_encoder(
        args.results_dir / "models/shared/keypoint_encoder.pt",
        config_hash=config_hash, experiment=config, system=system,
        mode=args.mode, device=args.device)
    needs_potential = "onto_no_se" in args.pipelines
    potential = None
    artifact_error = None
    if needs_potential and args.reward_design.is_file():
        try:
            potential = FrozenSemanticRGATPotential(
                args.reward_design, expected_config_hash=config_hash)
            print(f"Using frozen direct semantic R-GAT {potential.design_id}.")
        except (OSError, ValueError, KeyError) as exc:
            artifact_error = str(exc)
            print(f"Existing semantic R-GAT is not reusable: {exc}")
    if needs_potential and potential is None and args.no_prepare_reward_design:
        parser.error(f"onto_no_se requires a valid semantic R-GAT: {artifact_error}")

    controlled_fields = {
        name: config.get(name) for name in
        ("camera", "estimator", "control", "ppo", "curriculum", "seeds")
    }
    manifest = {
        "format": "ontology_rgat.three_pipeline_experiment/1",
        "experiment": args.experiment, "mode": args.mode,
        "training_replicate": args.training_replicate,
        "configured_independent_model_seeds": independent_model_seeds,
        "model_initialization_seed": model_seed,
        "config": str(args.config.resolve()), "config_hash": config_hash,
        "system_config": str(args.system_config.resolve()),
        "pipelines": args.pipelines,
        "pipeline_specs": {name: get_pipeline(name).to_manifest()
                           for name in args.pipelines},
        "controlled_fields": controlled_fields,
        "ppo_episodes_per_pipeline": train_count,
        "N_PPO": train_count * len(args.pipelines),
        "N_estimator_warmup": selected_warmup,
        "N_training_environment_episodes": (
            train_count * len(args.pipelines) + selected_warmup),
        "training_seed_contract": {
            "ppo_seed_start": training_seed0,
            "ppo_seed_stop_exclusive": training_seed0 + train_count,
            "identical_ppo_seeds_for_all_selected_pipelines": True,
            "shin_warmup_seed_start": (warmup_seed0 if selected_warmup else None),
            "shin_warmup_seed_stop_exclusive": (
                warmup_seed0 + selected_warmup if selected_warmup else None),
        },
        "evaluation": evaluation_cfg, "paired_seeds": True,
        "primary_comparison_metric_family": "reward-independent physical task metrics",
        "episode_return_role": "debugging only; never used for cross-pipeline ranking",
        "checkpoint_selection_rule": (
            "latest completed PPO episode; no reward-return model selection"),
        "reward_design_id": getattr(potential, "design_id", None),
        "execution_status": "configured; real Isaac/Pegasus/PX4 results pending",
    }
    manifest_path = args.results_dir / "manifest.json"
    existing_eval = args.results_dir / "evaluation/per_episode.csv"
    if existing_eval.is_file() and manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("config_hash") != config_hash:
            parser.error("existing results use a different experiment configuration")
    _write_json(manifest_path, manifest)
    _write_csv(args.results_dir / "evaluation/paired_plan.csv", plan)

    ensure_fastdds()
    cfg = _live_config(args.mode, args.results_dir, args.system_config)
    control = dict(config.get("control") or {})
    cfg.benchmark_control = control
    cfg.sim.dt = float(control.get("dt_seconds", .1))
    cfg.sim.max_steps = int(control.get("horizon_steps", 300))
    cfg.sim.max_time = cfg.sim.dt * cfg.sim.max_steps
    cfg.external.control_hz = 1.0 / cfg.sim.dt
    cfg.viz.dashboard.enabled = not args.no_dashboard
    if args.dashboard_port is not None:
        cfg.viz.dashboard.port = int(args.dashboard_port)
    ppo = dict(config.get("ppo") or {})
    ppo["perception_warmup_episodes"] = int(ppo.get(
        f"perception_warmup_episodes_{args.mode}",
        2 if args.mode == "quick" else 8))
    cfg.reward.pbrs.gamma = float(ppo.get("gamma", .99))
    cfg.reward.pbrs["lambda"] = float(ppo.get("shaping_lambda", 1.0))
    curriculum_raw = dict(config.get("curriculum") or {})
    levels = int(curriculum_raw.get("levels", 80))
    interval = int(curriculum_raw.get("update_every_episodes", 512))
    if args.total_train_episodes is not None or args.train_episodes is not None:
        interval = fitted_update_interval(train_count, levels)
        ppo["allow_curriculum_interval_migration"] = True
    curriculum = {"levels": levels, "episodes_per_update": interval}

    rviz = RvizPublisher.create(cfg) if not args.no_rviz else None
    monitor = BenchmarkMonitor(STORE, rviz=rviz)
    monitor.configure(
        methods=args.pipelines, mode=args.mode, config_hash=config_hash,
        training_total=(train_count * len(args.pipelines)
                        + (int(ppo["perception_warmup_episodes"])
                           if "shin_se" in args.pipelines else 0)),
        evaluation_total=len(plan),
        reward_design_id=getattr(potential, "design_id", None),
        reward_design_sha256=getattr(potential, "sha256", None))
    dashboard = Dashboard(cfg, STORE).start()
    rviz_process, rviz_log = _start_rviz(
        cfg.viz.rviz.enabled and not args.no_rviz and not args.headless)
    owned = None
    stack_module.current(None)
    design_episodes = 0
    design_steps = 0
    source_training_episodes = 0
    source_training_steps = 0
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
            histories = {}

            def train_pipeline(name, *, primary=True):
                monitor.stage("training", f"recurrent PPO · {name}")
                torch.manual_seed(model_seed)
                model = _build_model(
                    config, args.device, keypoint_pretraining, pipeline=name)
                target_dir = (args.results_dir / f"models/{name}" if primary else
                              args.results_dir / "models/reward_design_source")
                warmup_count = (int(ppo["perception_warmup_episodes"])
                                if get_pipeline(name).state_estimation_enabled else 0)
                training_seeds = controlled_training_seeds(
                    training_seed0, train_count, warmup_episodes=warmup_count,
                    warmup_seed0=warmup_seed0)
                history = train_live(
                    lambda: LiveShinEnvironment(
                        cfg, camera, horizon_steps=int(cfg.sim.max_steps)),
                    model, name,
                    training_seeds,
                    target_dir, config_hash=config_hash,
                    potential=(potential if name == "onto_no_se" else None),
                    ppo=ppo, curriculum_config=curriculum, monitor=monitor if primary else None,
                    restart_incompatible=True)
                if primary:
                    models[name] = model
                    histories[name] = history
                    _write_csv(args.results_dir / f"training/{name}.csv", history)
                return model, history, target_dir / f"{name}.pt"

            for name in args.pipelines:
                if name != "onto_no_se":
                    train_pipeline(name)

            if needs_potential and potential is None:
                if "no_se" in models:
                    source_model = models["no_se"]
                    source_checkpoint = args.results_dir / "models/no_se/no_se.pt"
                    source_training_episodes = 0
                else:
                    source_model, source_history, source_checkpoint = train_pipeline(
                        "no_se", primary=False)
                    source_training_episodes = train_count
                    source_training_steps = sum(
                        int(float(row.get("steps", 0))) for row in source_history
                        if row.get("optimization_phase", "ppo") == "ppo")
                dataset, dataset_manifest, dataset_path, design_steps = (
                    _collect_semantic_data(
                        cfg=cfg, camera=camera, model=source_model, config=config,
                        config_hash=config_hash, checkpoint_path=source_checkpoint,
                        results_dir=args.results_dir, mode=args.mode,
                        monitor=monitor, episodes_override=args.rgat_data_episodes,
                        source_training_episodes=source_training_episodes,
                        source_training_environment_steps=source_training_steps))
                design_episodes = int(dataset_manifest["episodes"])
                design_settings = dict(config.get("rgat_design") or {})
                design_settings["epochs"] = int(
                    args.rgat_epochs or design_settings.get(
                        f"epochs_{args.mode}", 10 if args.mode == "quick" else 80))
                monitor.stage("R-GAT training", "direct semantic discounted outcome")
                _, design_metadata = prepare_semantic_rgat_artifact(
                    args.reward_design, dataset, dataset_path=dataset_path,
                    config_hash=config_hash, mode=args.mode, seed=model_seed,
                    settings=design_settings)
                potential = FrozenSemanticRGATPotential(
                    args.reward_design, expected_config_hash=config_hash)
                monitor.potential = potential
                manifest.update({
                    "reward_design_id": potential.design_id,
                    "reward_design_sha256": potential.sha256,
                    "rgat_dataset": dataset_manifest,
                    "rgat_model": design_metadata,
                    "reward_design_source_ppo_episodes": source_training_episodes,
                    "N_reward_design": design_episodes + source_training_episodes,
                })
                _write_json(manifest_path, manifest)
                STORE.set(reward_design_id=potential.design_id,
                          reward_design_sha256=potential.sha256)
            elif needs_potential:
                monitor.potential = potential
                model_manifest = potential.metadata
                data_manifest = model_manifest.get("dataset_manifest") or {}
                design_episodes = int(data_manifest.get("episodes", 0))
                design_steps = int(data_manifest.get("environment_steps") or 0)
                source_provenance = data_manifest.get("source_behavior_policy") or {}
                source_training_episodes = int(source_provenance.get(
                    "additional_source_training_episodes", 0))
                source_training_steps = int(source_provenance.get(
                    "additional_source_training_environment_steps", 0))

            if "onto_no_se" in args.pipelines:
                train_pipeline("onto_no_se")

            training_records = [row for name in args.pipelines
                                for row in histories.get(name, [])]
            evaluation_rows = _read_csv(existing_eval)
            completed = {(row["pipeline"], row["scenario"], int(row["seed"]))
                         for row in evaluation_rows}
            monitor.restore_evaluation(evaluation_rows)
            for name in args.pipelines:
                model = models[name]
                monitor.stage("paired evaluation", name)
                with LiveShinEnvironment(
                        cfg, camera, horizon_steps=int(cfg.sim.max_steps)) as environment:
                    for item in plan:
                        if item["method"] != name:
                            continue
                        key = (name, item["scenario"], int(item["seed"]))
                        if key in completed:
                            continue
                        _, metric = collect_episode(
                            environment, model, name, int(item["seed"]),
                            curriculum=1.0,
                            potential=(potential if name == "onto_no_se" else None),
                            deterministic=True, gamma=float(ppo.get("gamma", .99)),
                            shaping_lambda=float(ppo.get("shaping_lambda", 1.0)),
                            scenario=item["scenario"], monitor=monitor,
                            phase="evaluation")
                        metric.update({"method": name, "pipeline": name,
                                       "scenario": item["scenario"]})
                        evaluation_rows.append(metric)
                        _write_csv(existing_eval, evaluation_rows)
                        monitor.evaluation_update(name, metric)

            reports = write_three_pipeline_outputs(
                evaluation_rows, training_records, args.results_dir,
                reward_design_episodes=design_episodes + source_training_episodes,
                reward_design_steps=design_steps + source_training_steps,
                estimator_warmup_episodes=selected_warmup,
                estimator_warmup_steps=sum(
                    int(float(row.get("steps", 0)))
                    for row in histories.get("shin_se", [])
                    if row.get("optimization_phase") == "perception_warmup"))
            reward_design_cost = design_episodes + source_training_episodes
            reward_design_step_cost = design_steps + source_training_steps
            warmup_step_cost = sum(
                int(float(row.get("steps", 0)))
                for row in histories.get("shin_se", [])
                if row.get("optimization_phase") == "perception_warmup")
            manifest.update({
                "execution_status": "complete real Isaac/Pegasus/PX4 run",
                "N_PPO": train_count * len(args.pipelines),
                "N_estimator_warmup": selected_warmup,
                "estimator_warmup_environment_steps": warmup_step_cost,
                "N_reward_design": reward_design_cost,
                "N_total": train_count * len(args.pipelines) + reward_design_cost,
                "N_total_including_estimator_warmup": (
                    train_count * len(args.pipelines) + reward_design_cost
                    + selected_warmup),
                "reward_design_environment_steps": reward_design_step_cost,
                "reports": reports,
            })
            _write_json(manifest_path, manifest)
    finally:
        stack_module.current(None)
        if owned is not None and not args.keep_stack:
            owned.stop()
        if rviz_process is not None and rviz_process.poll() is None:
            rviz_process.terminate()
        if rviz_log is not None and not rviz_log.closed:
            rviz_log.close()
        dashboard.stop()
    print(f"Three-pipeline experiment complete: {args.results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
