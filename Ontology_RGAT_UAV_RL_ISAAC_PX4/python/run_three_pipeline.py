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
from ontology_rgat.evaluation import (write_adaptive_reward_figures,
                                      write_three_pipeline_outputs)
from ontology_rgat.perception import (RosGrayscaleSource,
                                      calibrate_keypoint_encoder,
                                      prepare_keypoint_encoder)
from ontology_rgat.pipelines import (available_pipeline_ids, get_pipeline,
                                     primary_pipeline_ids,
                                     validate_pipeline_configuration)
from ontology_rgat.ppo.behavior_cloning import (
    behavior_clone, encoded_demonstration_episode,
    load_encoded_demonstrations, merge_encoded_demonstrations,
    save_encoded_demonstrations)
from ontology_rgat.ppo.recurrent_train import (collect_episode_resilient,
                                               train_live)
from ontology_rgat.reward_modes import RewardComponentNormalizer
from ontology_rgat.rgat import (
    FrozenAdaptiveRewardWeights, FrozenSemanticRGATPotential,
    adaptive_episode_records, build_adaptive_dataset, load_adaptive_dataset,
    load_semantic_dataset, prepare_adaptive_reward_artifact,
    merge_semantic_datasets, prepare_semantic_rgat_artifact,
    save_adaptive_dataset, save_semantic_dataset, semantic_episode_dataset)
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


def _behavior_transform(variant: int, *, policy_blend: float = .35,
                        servo_gain: float = 1.0, noise_std: float | None = None,
                        yaw_blend: float = .25):
    """Estimator-free mixture: no_se action, image servo and bounded noise."""
    def transform(step, policy_action, semantic, rng):
        action = np.asarray(policy_action, dtype=np.float64).copy()
        if variant == 0:
            # Direct image-plane servo. For the configured forward/down camera,
            # optical +x projects primarily onto body-forward and optical +y
            # onto body-right. No metric depth/pose is reconstructed.
            cx, cy = semantic.centroid_xy
            recovering = (semantic.visible_keypoint_fraction < 0.5
                          or semantic.visual_loss_risk > 0.0)
            correction = float(servo_gain) * (0.45 if recovering else 0.75)
            action[0] = np.clip(
                float(policy_blend) * action[0] + correction * cx, -0.8, 0.8)
            action[1] = np.clip(
                float(policy_blend) * action[1] - correction * cy, -0.8, 0.8)
            if recovering:
                # Preserve the last trustworthy image direction, climb to
                # widen the footprint, and suppress yaw until keypoints return.
                action[2] = 0.45
                action[3] *= min(float(yaw_blend), 0.15)
            elif (semantic.image_alignment > 0.65
                  and semantic.keypoint_confidence > 0.01):
                scale = semantic.apparent_target_scale
                descent = -0.55 if scale < 0.45 else -0.28 if scale < 0.75 else -0.10
                action[2] = min(0.25 * action[2], descent)
            else:
                action[2] = max(0.25 * action[2], 0.0)
            action[3] *= float(yaw_blend)
            action += rng.normal(
                0.0, 0.04 if noise_std is None else float(noise_std), 4)
        elif variant == 1:
            action += rng.normal(0.0, 0.18, 4)
        else:
            # Bounded exploration supplies clear failure-side graph states.
            action = 0.55 * action + 0.45 * rng.uniform(-1.0, 1.0, 4)
        return np.clip(action, -0.9, 0.9)
    return transform


def _privileged_velocity_teacher_action(
        relative_state, body_velocity, semantic, velocity_limit,
        *, position_gain: float = .35, velocity_gain: float = .75,
        horizontal_speed_limit: float = .60,
        noise_std: float = .01, rng=None):
    """Return a stable training-only velocity label for deadline warm starts.

    ``relative_state`` is platform-in-UAV-body simulator truth and is never an
    actor input. The teacher reconstructs the deck's horizontal velocity from
    ``v_uav + v_relative`` and adds a damped position correction. A
    position-only image servo commands zero at image centre, lets a moving UGV
    escape, and repeatedly crosses the target.
    """
    relative = np.asarray(relative_state, dtype=np.float64).reshape(-1)
    velocity = np.asarray(body_velocity, dtype=np.float64).reshape(-1)
    limit = np.asarray(velocity_limit, dtype=np.float64).reshape(-1)
    if relative.shape != (6,) or velocity.shape != (3,) or limit.shape != (3,):
        raise ValueError("privileged teacher expects 6-D truth and 3-D velocities")
    if (not np.isfinite(relative).all() or not np.isfinite(velocity).all()
            or not np.isfinite(limit).all() or np.any(limit <= 0.0)):
        raise ValueError("privileged teacher inputs must be finite with positive limits")

    position_xy = relative[:2]
    relative_velocity_xy = relative[3:5]
    deck_velocity_xy = velocity[:2] + relative_velocity_xy
    target_xy = (deck_velocity_xy
                 + float(position_gain) * position_xy
                 + float(velocity_gain) * relative_velocity_xy)
    speed_limit = float(horizontal_speed_limit)
    if not np.isfinite(speed_limit) or speed_limit <= 0.0:
        raise ValueError("privileged teacher speed limit must be positive")
    speed = float(np.linalg.norm(target_xy))
    if speed > speed_limit:
        target_xy *= speed_limit / speed

    lateral_error = float(np.linalg.norm(position_xy))
    relative_speed = float(np.linalg.norm(relative_velocity_xy))
    altitude = max(0.0, -float(relative[2]))
    visual_lost = (float(semantic.visible_keypoint_fraction) < 0.5
                   or float(semantic.visual_loss_risk) > 0.0)
    # The multi-scale marker naturally fills and then leaves the downward
    # camera at the end of a correct flare. Treating that expected low-altitude
    # disappearance as a recovery event traps the vehicle centimetres above
    # the deck. Only climb on loss while still high or laterally displaced.
    if visual_lost and (altitude > 0.80 or lateral_error > 0.40):
        target_vz = 0.22
    elif lateral_error > 0.50 or relative_speed > 0.45:
        target_vz = 0.0
    elif altitude > 1.10:
        target_vz = -0.35
    elif altitude > 0.60:
        target_vz = -0.14
    else:
        target_vz = -0.04

    target_velocity = np.r_[target_xy, target_vz]
    action = np.r_[target_velocity / limit, 0.0]
    if float(noise_std) > 0.0:
        generator = rng if rng is not None else np.random.default_rng()
        action += generator.normal(0.0, float(noise_std), 4)
    return np.clip(action, -0.90, 0.90)


def _privileged_velocity_teacher(environment, *, settings):
    """Bind the training-only teacher to the environment's current live step."""
    def transform(step, policy_action, semantic, rng):
        current = environment.last_step
        if current is None:
            raise RuntimeError("privileged teacher requires a reset live environment")
        controller = environment.adapter.controller
        return _privileged_velocity_teacher_action(
            current.critic.true_relative_state,
            current.actor.body_velocity,
            semantic,
            controller.max_velocity * controller.action_scale,
            position_gain=float(settings.get("position_gain", .35)),
            velocity_gain=float(settings.get("velocity_gain", .75)),
            horizontal_speed_limit=float(settings.get(
                "horizontal_speed_limit_m_s", .60)),
            noise_std=float(settings.get("noise_std", .01)),
            rng=rng)
    return transform


def _read_csv(path: Path):
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _prepare_fast_demonstrations(*, cfg, camera, config, config_hash,
                                 keypoint_pretraining, results_dir, device,
                                 model_seed, monitor):
    """Collect successful teacher flights once and store compact actor inputs."""
    fast = dict(config.get("seminar_fast") or {})
    settings = dict(fast.get("behavior_cloning") or {})
    if not bool(settings.get("enabled", False)):
        return None
    required = max(1, int(settings.get("successful_episodes", 6)))
    maximum = max(required, int(settings.get("max_attempts", 12)))
    source_pipeline = str(settings.get("source_pipeline", "no_se_fixed"))
    if get_pipeline(source_pipeline).state_estimation_enabled:
        raise ValueError("the behavior-teacher encoder source must be estimator-free")
    teacher_id = str(settings.get(
        "teacher", "privileged_relative_state_velocity_pd_v4"))
    if teacher_id != "privileged_relative_state_velocity_pd_v4":
        raise ValueError(f"unknown seminar-fast behavior teacher: {teacher_id}")
    artifact_path = (Path(results_dir) / "models/shared"
                     / f"teacher_demonstrations_{config_hash[:12]}.pt")
    attempts_path = (Path(results_dir) / "training"
                     / f"teacher_attempts_{config_hash[:12]}.csv")
    encoder_path = Path(results_dir) / "models/shared/keypoint_encoder.pt"
    encoder_sha = _sha256_file(encoder_path)
    payload = None
    dataset = None
    attempts = _read_csv(attempts_path)
    attempted_seeds = [int(float(row["seed"])) for row in attempts]
    environment_steps = sum(int(float(row.get("steps", 0))) for row in attempts)
    if artifact_path.is_file():
        try:
            payload = load_encoded_demonstrations(
                artifact_path, config_hash=config_hash,
                encoder_sha256=encoder_sha)
            dataset = payload["dataset"]
            attempted_seeds = list(dict.fromkeys(
                [*payload.get("attempted_seeds", ()), *attempted_seeds]))
            environment_steps = max(
                int(payload.get("environment_steps", 0)), environment_steps)
            print(
                "Resuming shared training-teacher demonstrations: "
                f"{payload['successful_episodes']}/{required} successes from "
                f"{len(attempted_seeds)}/{maximum} attempts.")
        except (OSError, ValueError, KeyError) as exc:
            print(f"Ignoring incompatible training-teacher demonstrations: {exc}")
            payload, dataset, attempts, attempted_seeds, environment_steps = (
                None, None, [], [], 0)
    successes = (0 if dataset is None else
                 int(torch.unique(dataset["episode_id"]).numel()))
    if successes < required and len(attempted_seeds) < maximum:
        torch.manual_seed(int(model_seed))
        teacher_model = _build_model(
            config, device, keypoint_pretraining, pipeline=source_pipeline)
        seed0 = int((config.get("seeds") or {}).get(
            "behavior_cloning_start", 90000))
        monitor.stage(
            "training-only teacher demonstrations",
            f"successful real Isaac/PX4 flights {successes}/{required}")
        with LiveShinEnvironment(
                cfg, camera, horizon_steps=int(cfg.sim.max_steps)) as environment:
            teacher = _privileged_velocity_teacher(
                environment, settings=settings)
            for attempt in range(maximum):
                seed = seed0 + attempt
                if seed in attempted_seeds:
                    continue
                rows, metric = collect_episode_resilient(
                    environment, teacher_model, source_pipeline, seed,
                    curriculum=float(settings.get("curriculum", 1.0)),
                    deterministic=True, scenario="training_random_walk",
                    monitor=monitor, phase="training-only teacher demonstration",
                    action_transform=teacher)
                attempted_seeds.append(seed)
                environment_steps += int(metric["steps"])
                metric.update({
                    "method": "privileged_teacher", "pipeline": "shared_warm_start",
                    "episode": len(attempted_seeds),
                    "accepted_for_cloning": float(metric["paper_success"]),
                    "teacher": teacher_id,
                    "config_hash": config_hash,
                    "teacher_information": (
                        "training-only simulator relative state for action labels; "
                        "stored/deployed actor inputs are image embedding and UAV "
                        "proprioception only"),
                })
                attempts.append(metric)
                _write_csv(attempts_path, attempts)
                if bool(metric["paper_success"]):
                    successes += 1
                    episode = encoded_demonstration_episode(
                        teacher_model, rows, episode_id=successes,
                        batch_size=int(settings.get("encoding_batch_size", 64)))
                    dataset = merge_encoded_demonstrations(dataset, episode)
                if dataset is not None:
                    payload = save_encoded_demonstrations(
                        artifact_path, dataset, config_hash=config_hash,
                        encoder_sha256=encoder_sha,
                        attempted_seeds=attempted_seeds,
                        environment_steps=environment_steps,
                        teacher=teacher_id)
                print(
                    f"training teacher attempt {len(attempted_seeds)}/{maximum} "
                    f"success={int(metric['paper_success'])} "
                    f"accepted={successes}/{required}")
                if successes >= required:
                    break
        del teacher_model
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    if payload is None or int(payload["successful_episodes"]) < required:
        raise RuntimeError(
            "training teacher did not produce enough real successful landings "
            f"({successes}/{required}) after {len(attempted_seeds)}/{maximum} attempts")
    return payload


def _reward_design_collection_contract(design, mode, minimum, maximum):
    """Build the manifest contract before any external process is started."""
    recoveries = int(design.get(
        f"minimum_successful_recovery_episodes_{mode}", 1))
    if recoveries < 1:
        raise ValueError("semantic R-GAT data requires a positive recovery minimum")
    return {
        "minimum_episodes": int(minimum),
        "maximum_episodes": int(maximum),
        "minimum_successful_recovery_episodes": recoveries,
        "stop_condition": (
            "minimum reached, both terminal classes observed, and successful "
            "loss-to-reacquisition-to-landing trajectories observed"),
        "synthetic_outcomes_allowed": False,
    }


def _collect_semantic_data(*, cfg, camera, model, config, config_hash,
                           checkpoint_path, results_dir, mode, monitor,
                           episodes_override=None,
                           max_episodes_override=None,
                           source_training_episodes=0,
                           source_training_environment_steps=0,
                           source_pipeline="no_se"):
    design = dict(config.get("rgat_design") or {})
    count = int(episodes_override if episodes_override is not None else
                design.get(f"episodes_{mode}", 8 if mode == "quick" else 40))
    if count < 2:
        raise ValueError("semantic R-GAT collection requires at least two episodes")
    max_count = int(
        max_episodes_override if max_episodes_override is not None else
        max(3 * count, int(design.get(
            f"max_episodes_{mode}", 3 * count))))
    if max_count < count:
        raise ValueError("semantic R-GAT maximum episodes cannot be below its minimum")
    stride = int(design.get("sample_stride", 3))
    minimum_recoveries = int(design.get(
        f"minimum_successful_recovery_episodes_{mode}", 1))
    if minimum_recoveries < 1:
        raise ValueError("semantic R-GAT data requires a positive recovery minimum")
    gamma = float(design.get("outcome_discount", (config.get("ppo") or {}).get(
        "gamma", .99)))
    seed0 = int((config.get("seeds") or {}).get("rgat_dataset_start", 70000))
    requested_seeds = list(range(seed0, seed0 + max_count))
    dataset_path = Path(results_dir) / "rgat/semantic_rollouts.npz"
    episodes_path = Path(results_dir) / "rgat/semantic_rollout_episodes.csv"
    checkpoint_sha = _sha256_file(checkpoint_path)
    behavior = {
        "name": str(design.get(
            "behavior_policy", "semantic_visual_servo_recovery_noisy_no_se_mixture_v2")),
        "source_pipeline": str(source_pipeline),
        "state_estimation_enabled": False,
        "components": ["trained_no_se", "image_plane_servo",
                       "explicit_visibility_recovery_climb",
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
            print(f"Resuming semantic R-GAT data: {len(completed)}/{count} minimum "
                  f"({max_count} hard cap).")
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            print(f"Ignoring incompatible semantic rollout cache: {exc}")
            dataset, manifest, episode_rows, completed = None, None, [], []

    def recovery_statistics(rows):
        return {
            "episodes_with_visual_loss": sum(
                float(row.get("visual_loss_events", 0)) > 0 for row in rows),
            "episodes_with_reacquisition": sum(
                float(row.get("visual_reacquisition_events", 0)) > 0
                for row in rows),
            "successful_recovery_episodes": sum(
                float(row.get("successful_recovery_landing", 0)) > 0
                for row in rows),
            "required_successful_recovery_episodes": minimum_recoveries,
        }

    def requirements_met(current_manifest, rows):
        if current_manifest is None:
            return False
        episodes = int(current_manifest.get("episodes", 0))
        successes = int(current_manifest.get("successful_episodes", 0))
        recoveries = recovery_statistics(rows)["successful_recovery_episodes"]
        return (episodes >= count and 0 < successes < episodes
                and recoveries >= minimum_recoveries)

    pending = [(index, seed) for index, seed in enumerate(requested_seeds, start=1)
               if seed not in completed]
    if pending and not requirements_met(manifest, episode_rows):
        monitor.stage("R-GAT data", "estimator-free semantic behavior mixture")
        with LiveShinEnvironment(
                cfg, camera, horizon_steps=int(cfg.sim.max_steps)) as environment:
            for episode, seed in pending:
                if requirements_met(manifest, episode_rows):
                    break
                rows, metric = collect_episode_resilient(
                    environment, model, source_pipeline, seed, curriculum=1.0,
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
                    "visual_loss_events": metric["visual_loss_events"],
                    "visual_reacquisition_events": metric[
                        "visual_reacquisition_events"],
                    "visual_reacquisition_rate": metric[
                        "visual_reacquisition_rate"],
                    "mean_visual_reacquisition_time_s": metric[
                        "mean_visual_reacquisition_time_s"],
                    "recovery_climb_fraction": metric["recovery_climb_fraction"],
                    "unsafe_descent_low_visibility_fraction": metric[
                        "unsafe_descent_low_visibility_fraction"],
                    "successful_recovery_landing": metric[
                        "successful_recovery_landing"],
                })
                recovery = recovery_statistics(episode_rows)
                manifest = save_semantic_dataset(
                    dataset, dataset_path, config_hash=config_hash,
                    source_behavior_policy=behavior, completed_seeds=completed,
                    environment_steps=sum(
                        int(float(row.get("steps", 0))) for row in episode_rows),
                    recovery_statistics=recovery)
                _write_csv(episodes_path, episode_rows)
                print(f"semantic data {len(completed)}/{count} minimum "
                      f"(cap {max_count}): "
                      f"success={int(metric['paper_success'])} samples={len(current['y'])}")
    if dataset is None or manifest is None:
        raise RuntimeError("semantic reward-design dataset is unavailable")
    successes = int(manifest["successful_episodes"])
    if successes == 0 or successes == int(manifest["episodes"]):
        raise RuntimeError(
            f"semantic reward-design data still contains only one terminal class "
            f"after its {max_count}-episode hard cap; increase "
            "--rgat-max-data-episodes or improve the estimator-free behavior policy")
    recovery = recovery_statistics(episode_rows)
    if recovery["successful_recovery_episodes"] < minimum_recoveries:
        raise RuntimeError(
            "semantic reward-design data lacks successful loss/reacquisition/landing "
            f"trajectories ({recovery['successful_recovery_episodes']}/"
            f"{minimum_recoveries}) after its {max_count}-episode hard cap")
    total_steps = sum(int(float(row.get("steps", 0))) for row in episode_rows)
    if total_steps == 0:
        total_steps = int(manifest.get("environment_steps") or 0)
    return dataset, manifest, dataset_path, total_steps


def _records_from_adaptive_dataset(dataset):
    records = []
    for index in range(len(dataset["episode_id"])):
        records.append({
            "graph_X": dataset["X"][index].T,
            "rho_raw": dataset["rho_raw"][index],
            **{name: dataset[name][index].item()
               for name in (
                   "episode_id", "time_index", "success", "failure_type",
                   "touchdown_error", "touchdown_vertical_speed",
                   "touchdown_roll", "touchdown_pitch", "duration",
                   "terminal_reason", "scenario", "seed", "phase",
                   "disturbance_level")},
        })
    return records


def _collect_adaptive_data(*, cfg, camera, model, source_pipeline, config,
                           config_hash, results_dir, mode, monitor,
                           episodes_override=None, max_episodes_override=None):
    """실제 Isaac/PX4 전이로 5성분 adaptive reward dataset을 만든다."""
    design = dict(config.get("adaptive_reward_design") or {})
    reward_cfg = dict(config.get("adaptive_reward") or {})
    count = int(episodes_override if episodes_override is not None else
                design.get(f"episodes_{mode}", 8 if mode == "quick" else 40))
    maximum = int(max_episodes_override if max_episodes_override is not None else
                  design.get(f"max_episodes_{mode}", max(3 * count, count)))
    if count < 2 or maximum < count:
        raise ValueError("adaptive reward data needs at least two episodes and a valid cap")
    seed0 = int((config.get("seeds") or {}).get("adaptive_dataset_start", 80000))
    path = Path(results_dir) / "rgat/adaptive_reward_rollouts.npz"
    records = []
    if path.is_file() and path.with_suffix(".manifest.json").is_file():
        try:
            cached, cached_manifest = load_adaptive_dataset(
                path, config_hash=config_hash)
            records = _records_from_adaptive_dataset(cached)
            print(f"Resuming adaptive reward data: "
                  f"{len(np.unique(cached['episode_id']))}/{count} minimum.")
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            print(f"Ignoring incompatible adaptive rollout cache: {exc}")
            records = []

    scenarios = tuple(design.get("scenarios") or (
        "training_random_walk", "zigzag", "vertical_heave_boat"))
    fast_settings = dict(
        ((config.get("seminar_fast") or {}).get("behavior_cloning") or {}))
    deadline_teacher_enabled = bool(fast_settings.get("enabled", False))
    completed_episodes = set(int(row["episode_id"]) for row in records)

    def requirements_met(dataset):
        if dataset is None:
            return False
        episodes = np.unique(dataset["episode_id"])
        outcomes = set(np.asarray(dataset["success"], dtype=int).tolist())
        return len(episodes) >= count and outcomes == {0, 1}

    dataset = None
    manifest = None
    if records:
        normalization = dict(reward_cfg.get("component_normalization") or {})
        dataset = build_adaptive_dataset(
            records, seed=int(config.get("seed", 42)),
            validation_fraction=float(design.get("validation_fraction", .2)),
            test_fraction=float(design.get("test_fraction", 0.0)),
            physical_scales=normalization.get("scales"),
            normalization_quantile=float(normalization.get("quantile", .99)),
            exact_paper_raw=bool(reward_cfg.get("exact_paper_raw", False)))
    behavior = {
        "name": str(design.get(
            "behavior_policy",
            "mixed_random_fixed_success_collision_drift_near_miss")),
        "source_pipeline": str(source_pipeline),
        "components": [
            "trained_fixed_policy",
            ("training_only_privileged_velocity_teacher"
             if deadline_teacher_enabled else "visual_servo_success_recovery"),
            "moderate_noise_near_miss", "bounded_random_exploration",
            "empirical_collision_or_drift_failures"],
        "deadline_teacher": (
            str(fast_settings.get("teacher"))
            if deadline_teacher_enabled else None),
        "scenario_cycle": list(scenarios),
        "synthetic_transitions_allowed": False,
    }
    pending = [(episode, seed0 + episode - 1)
               for episode in range(1, maximum + 1)
               if episode not in completed_episodes]
    if pending and not requirements_met(dataset):
        monitor.stage("adaptive reward data", "real transition outcome mixture")
        with LiveShinEnvironment(
                cfg, camera, horizon_steps=int(cfg.sim.max_steps)) as environment:
            for episode, seed in pending:
                if requirements_met(dataset):
                    break
                scenario = scenarios[(episode - 1) % len(scenarios)]
                variant = (episode - 1) % 3
                transform = (
                    _privileged_velocity_teacher(
                        environment, settings=fast_settings)
                    if deadline_teacher_enabled and variant == 0 else
                    _behavior_transform(variant))
                rows, metric = collect_episode_resilient(
                    environment, model, source_pipeline, seed, curriculum=1.0,
                    deterministic=False, scenario=scenario, monitor=monitor,
                    phase="adaptive reward-design data",
                    action_transform=transform)
                records.extend(adaptive_episode_records(
                    rows, metric, episode_id=episode, seed=seed,
                    scenario=scenario))
                if len(set(int(row["episode_id"]) for row in records)) >= 2:
                    normalization = dict(
                        reward_cfg.get("component_normalization") or {})
                    dataset = build_adaptive_dataset(
                        records, seed=int(config.get("seed", 42)),
                        validation_fraction=float(
                            design.get("validation_fraction", .2)),
                        test_fraction=float(design.get("test_fraction", 0.0)),
                        physical_scales=normalization.get("scales"),
                        normalization_quantile=float(
                            normalization.get("quantile", .99)),
                        exact_paper_raw=bool(
                            reward_cfg.get("exact_paper_raw", False)))
                    manifest = save_adaptive_dataset(
                        dataset, path, config_hash=config_hash,
                        source_behavior_policy=behavior)
                print(f"adaptive data episode {episode}/{count} minimum "
                      f"success={int(metric['paper_success'])} steps={len(rows)}")
    if dataset is None:
        raise RuntimeError("adaptive reward-design dataset is unavailable")
    if not requirements_met(dataset):
        raise RuntimeError(
            "adaptive reward data lacks both success and failure outcomes at its hard cap")
    if manifest is None:
        manifest = save_adaptive_dataset(
            dataset, path, config_hash=config_hash,
            source_behavior_policy=behavior)
    steps = int(len(dataset["episode_id"]))
    return dataset, manifest, path, steps


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--experiment", choices=(
        "three_pipeline", "adaptive_reward_weight_comparison"),
                        default=None)
    parser.add_argument("--pipelines", nargs="+", choices=available_pipeline_ids())
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--config", type=Path,
                        default=ROOT / "config/experiments/three_pipeline_comparison.yaml")
    parser.add_argument("--system-config", type=Path,
                        default=ROOT / "config/shin2026-system.yaml")
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--reward-design", type=Path)
    parser.add_argument("--adaptive-reward-design", type=Path)
    parser.add_argument("--no-prepare-reward-design", action="store_true")
    parser.add_argument("--rgat-data-episodes", type=int)
    parser.add_argument(
        "--rgat-max-data-episodes", type=int,
        help="hard cap for automatic real rollout extension when one class is missing")
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
    if args.rgat_max_data_episodes is not None and args.rgat_max_data_episodes < 2:
        parser.error("--rgat-max-data-episodes must be at least two")
    if args.training_replicate < 0:
        parser.error("--training-replicate must be non-negative")

    config = load_experiment(args.config)
    seminar_fast = dict(config.get("seminar_fast") or {})
    if args.experiment is None:
        args.experiment = str(config.get("experiment", "three_pipeline"))
    validate_pipeline_configuration(config)
    configured = tuple(config.get("pipelines") or ())
    if args.pipelines is None:
        args.pipelines = list(configured)
    if any(name not in configured for name in args.pipelines):
        parser.error("selected pipeline is absent from experiment configuration")
    for name in args.pipelines:
        get_pipeline(name)
    if args.results_dir is None:
        experiment_dir = ("adaptive_reward_weight" if
                          config.get("experiment") == "adaptive_reward_weight_comparison"
                          else "three_pipeline")
        args.results_dir = ROOT / "results" / experiment_dir / args.mode
        if args.training_replicate:
            args.results_dir /= f"replicate_{args.training_replicate}"
    if args.reward_design is None:
        args.reward_design = args.results_dir / "rgat/rgat_model.pt"
    if args.adaptive_reward_design is None:
        args.adaptive_reward_design = (
            args.results_dir / "rgat/adaptive_reward_weights.pt")
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
        "outcome_contract": (
            "safe_landing_contact_position_velocity_attitude_rate_v2"),
    })
    configured_count = int(training_cfg.get(
        f"episodes_{args.mode}", 8 if args.mode == "quick" else 40960))
    configured_ppo = dict(config.get("ppo") or {})
    se_pipeline_count = sum(
        get_pipeline(name).state_estimation_enabled for name in args.pipelines)
    warmup_each = (int(configured_ppo.get(
        f"perception_warmup_episodes_{args.mode}",
        2 if args.mode == "quick" else 8)) if se_pipeline_count else 0)
    selected_warmup = warmup_each * se_pipeline_count
    if args.total_train_episodes is not None:
        try:
            train_count = episodes_per_method(
                args.total_train_episodes, len(args.pipelines), selected_warmup)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        train_count = int(args.train_episodes or configured_count)
    design_cfg = dict(config.get("rgat_design") or {})
    design_minimum = int(
        args.rgat_data_episodes if args.rgat_data_episodes is not None else
        design_cfg.get(f"episodes_{args.mode}", 8 if args.mode == "quick" else 40))
    design_maximum = int(
        args.rgat_max_data_episodes
        if args.rgat_max_data_episodes is not None else
        max(3 * design_minimum, int(design_cfg.get(
            f"max_episodes_{args.mode}", 3 * design_minimum))))
    if design_maximum < design_minimum:
        parser.error("R-GAT maximum data episodes cannot be below its minimum")
    evaluation_cfg = dict(config.get("evaluation") or {})
    if args.eval_episodes is not None:
        evaluation_cfg = {name: args.eval_episodes for name in evaluation_cfg}
    elif args.mode == "quick":
        evaluation_cfg = {name: (4 if name == "training_random_walk" else 2)
                          for name in evaluation_cfg}
    selected_scenarios = seminar_fast.get("evaluation_scenarios")
    if selected_scenarios:
        unknown_scenarios = set(selected_scenarios) - set(evaluation_cfg)
        if unknown_scenarios:
            parser.error(
                f"seminar-fast evaluation has unknown scenarios: "
                f"{sorted(unknown_scenarios)}")
        evaluation_cfg = {
            name: evaluation_cfg[name] for name in selected_scenarios}
    warmup_seed0 = int(seed_cfg.get(
        "estimator_warmup_start", base_training_seed - selected_warmup))
    warmup_seed0 += 100000 * args.training_replicate
    plan = paired_seed_plan(args.pipelines, evaluation_cfg,
                            int(seed_cfg.get("evaluation_start", 5000)))

    keypoint_pretraining = prepare_keypoint_encoder(
        args.results_dir / "models/shared/keypoint_encoder.pt",
        config_hash=config_hash, experiment=config, system=system,
        mode=args.mode, device=args.device)
    needs_potential = any(
        get_pipeline(name).use_direct_rgat_potential for name in args.pipelines)
    needs_adaptive = any(
        get_pipeline(name).use_adaptive_reward_weights for name in args.pipelines)
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
        parser.error(f"PBRS mode requires a valid semantic R-GAT: {artifact_error}")
    adaptive_weights = None
    adaptive_error = None
    if needs_adaptive and args.adaptive_reward_design.is_file():
        try:
            adaptive_weights = FrozenAdaptiveRewardWeights(
                args.adaptive_reward_design, expected_config_hash=config_hash)
            expected_architectures = {
                get_pipeline(name).adaptive_reward_architecture
                for name in args.pipelines
                if get_pipeline(name).use_adaptive_reward_weights}
            artifact_architecture = adaptive_weights.metadata[
                "model_config"]["architecture"]
            if expected_architectures != {artifact_architecture}:
                raise ValueError(
                    "one run may compare adaptive arms only with the same frozen architecture")
            print(f"Using frozen adaptive reward R-GAT {adaptive_weights.design_id}.")
        except (OSError, ValueError, KeyError) as exc:
            adaptive_error = str(exc)
            adaptive_weights = None
            print(f"Existing adaptive reward model is not reusable: {exc}")
    if needs_adaptive and adaptive_weights is None and args.no_prepare_reward_design:
        parser.error(
            f"adaptive reward mode requires a valid frozen model: {adaptive_error}")

    controlled_fields = {
        name: config.get(name) for name in
        ("camera", "estimator", "control", "ppo", "curriculum", "seeds")
    }
    landing_contract = dict(system.get("landing") or {})
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
        "domain_randomization": dict(system.get("domain_randomization") or {}),
        "ppo_episodes_per_pipeline": train_count,
        "N_PPO": train_count * len(args.pipelines),
        "N_estimator_warmup": selected_warmup,
        "N_training_environment_episodes": (
            train_count * len(args.pipelines) + selected_warmup),
        "reward_design_collection_contract": _reward_design_collection_contract(
            design_cfg, args.mode, design_minimum, design_maximum),
        "training_seed_contract": {
            "ppo_seed_start": training_seed0,
            "ppo_seed_stop_exclusive": training_seed0 + train_count,
            "identical_ppo_seeds_for_all_selected_pipelines": True,
            "shin_warmup_seed_start": (warmup_seed0 if selected_warmup else None),
            "shin_warmup_seed_stop_exclusive": (
                warmup_seed0 + warmup_each if selected_warmup else None),
            "warmup_episodes_per_se_pipeline": warmup_each,
            "se_pipeline_count": se_pipeline_count,
            "identical_warmup_seeds_for_se_arms": True,
        },
        "evaluation": evaluation_cfg, "paired_seeds": True,
        "primary_comparison_metric_family": "reward-independent physical task metrics",
        "landing_success_contract": {
            "version": "safe_landing_v2",
            "all_required": True,
            "pad_contact": True,
            "maximum_lateral_error_m": float(landing_contract.get(
                "success_xy_m", 0.35)),
            "maximum_vertical_speed_m_s": float(landing_contract.get(
                "success_vz_m_s", 0.55)),
            "maximum_relative_horizontal_speed_m_s": float(
                landing_contract.get("success_rel_speed_xy_m_s", 0.45)),
            "maximum_tilt_deg": float(landing_contract.get(
                "success_tilt_deg", 10.0)),
            "maximum_angular_rate_deg_s": float(landing_contract.get(
                "success_rate_deg_s", 45.0)),
            "unsafe_contact_is_failure": True,
        },
        "episode_return_role": "debugging only; never used for cross-pipeline ranking",
        "checkpoint_selection_rule": (
            "latest completed PPO episode; no reward-return model selection"),
        "reward_design_id": getattr(potential, "design_id", None),
        "adaptive_reward_design_id": getattr(adaptive_weights, "design_id", None),
        "adaptive_reward": config.get("adaptive_reward"),
        "seminar_fast": seminar_fast or None,
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
    runtime_reward_normalizer = None
    adaptive_runtime = dict(config.get("adaptive_reward") or {})
    normalization_runtime = dict(
        adaptive_runtime.get("component_normalization") or {})
    if adaptive_runtime.get("enabled") and normalization_runtime.get("scales"):
        ppo["reward_component_scales"] = list(normalization_runtime["scales"])
        ppo["exact_paper_raw_reward"] = bool(
            adaptive_runtime.get("exact_paper_raw", False))
        runtime_reward_normalizer = RewardComponentNormalizer(
            scales=tuple(ppo["reward_component_scales"]),
            exact_paper_raw=ppo["exact_paper_raw_reward"],
            source="shared_runtime_configuration")
    ppo["perception_warmup_episodes"] = int(ppo.get(
        f"perception_warmup_episodes_{args.mode}",
        2 if args.mode == "quick" else 8))
    cfg.reward.pbrs.gamma = float(ppo.get("gamma", .99))
    cfg.reward.pbrs["lambda"] = float(ppo.get("shaping_lambda", 1.0))
    curriculum_raw = dict(config.get("curriculum") or {})
    levels = int(curriculum_raw.get("levels", 80))
    interval = int(curriculum_raw.get("update_every_episodes", 512))
    curriculum = {
        "levels": levels,
        "episodes_per_update": interval,
        "initial_level": int(curriculum_raw.get("initial_level", 1)),
        "performance_gated": bool(curriculum_raw.get("performance_gated", False)),
        "assessment_window": int(curriculum_raw.get("assessment_window", 20)),
        "minimum_episodes_at_level": int(curriculum_raw.get(
            "minimum_episodes_at_level", 20)),
        "success_rate_threshold": float(curriculum_raw.get(
            "success_rate_threshold", 0.20)),
        "max_position_rmse_m": float(curriculum_raw.get(
            "max_position_rmse_m", 2.0)),
        "max_fov_loss_fraction": float(curriculum_raw.get(
            "max_fov_loss_fraction", 0.50)),
    }

    rviz = RvizPublisher.create(cfg) if not args.no_rviz else None
    monitor = BenchmarkMonitor(STORE, rviz=rviz)
    monitor.configure(
        methods=args.pipelines, mode=args.mode, config_hash=config_hash,
        training_total=(train_count * len(args.pipelines) + selected_warmup),
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
            monitor.stage("keypoint validation", "live Isaac camera · held-out labels")
            keypoint_pretraining = calibrate_keypoint_encoder(
                args.results_dir / "models/shared/keypoint_encoder.pt",
                keypoint_pretraining, camera, system=system, experiment=config,
                mode=args.mode, device=args.device)
            manifest["keypoint_pretraining"] = (
                None if keypoint_pretraining is None else {
                    "format": keypoint_pretraining["format"],
                    "implementation": keypoint_pretraining["implementation"],
                    "frozen_for_ppo": keypoint_pretraining["frozen_for_ppo"],
                    "training_source": keypoint_pretraining["training_source"],
                    "metrics": keypoint_pretraining["metrics"],
                    "empirical_calibration": keypoint_pretraining.get(
                        "empirical_calibration"),
                    "empirical_dataset": keypoint_pretraining.get(
                        "empirical_dataset"),
                })
            _write_json(manifest_path, manifest)
            demonstrations = _prepare_fast_demonstrations(
                cfg=cfg, camera=camera, config=config,
                config_hash=config_hash,
                keypoint_pretraining=keypoint_pretraining,
                results_dir=args.results_dir, device=args.device,
                model_seed=model_seed, monitor=monitor)
            cloning_metrics = {}
            if demonstrations is not None:
                manifest["behavior_cloning_demonstrations"] = {
                    key: demonstrations[key] for key in (
                        "teacher", "successful_episodes", "attempted_seeds",
                        "environment_steps", "transitions")}
                _write_json(manifest_path, manifest)
            models = {}
            histories = {}

            def train_pipeline(name, *, primary=True):
                monitor.stage("training", f"recurrent PPO · {name}")
                torch.manual_seed(model_seed)
                model = _build_model(
                    config, args.device, keypoint_pretraining, pipeline=name)
                target_dir = (args.results_dir / f"models/{name}" if primary else
                              args.results_dir / "models/reward_design_source")
                if demonstrations is not None:
                    cloning = dict((seminar_fast.get("behavior_cloning") or {}))
                    monitor.stage("behavior cloning", f"shared training teacher · {name}")
                    cloning_metrics[name] = behavior_clone(
                        model, demonstrations["dataset"],
                        epochs=int(cloning.get("epochs", 12)),
                        learning_rate=float(cloning.get("learning_rate", 3e-4)),
                        sequence_length=int(cloning.get("sequence_length", 48)),
                        auxiliary_coefficient=float(cloning.get(
                            "auxiliary_coefficient", .20)),
                        post_log_std=float(cloning.get("post_log_std", -1.8)))
                    manifest["behavior_cloning_by_pipeline"] = cloning_metrics
                    _write_json(manifest_path, manifest)
                    print(
                        f"{name} teacher warm start: action loss "
                        f"{cloning_metrics[name]['action_loss_before']:.4f} -> "
                        f"{cloning_metrics[name]['action_loss_after']:.4f}, "
                        f"std={cloning_metrics[name]['post_action_std']:.3f}")
                warmup_count = (int(ppo["perception_warmup_episodes"])
                                if get_pipeline(name).state_estimation_enabled else 0)
                training_seeds = controlled_training_seeds(
                    training_seed0, train_count, warmup_episodes=warmup_count,
                    warmup_seed0=warmup_seed0)
                spec = get_pipeline(name)
                reward_design = (adaptive_weights
                                 if spec.use_adaptive_reward_weights else
                                 potential if spec.use_direct_rgat_potential else None)
                if spec.use_adaptive_reward_weights:
                    if reward_design is None:
                        raise RuntimeError(f"{name} requires adaptive reward weights")
                    artifact_architecture = reward_design.metadata[
                        "model_config"]["architecture"]
                    if artifact_architecture != spec.adaptive_reward_architecture:
                        raise RuntimeError(
                            f"{name} requires {spec.adaptive_reward_architecture} "
                            f"weights, artifact is {artifact_architecture}")
                history = train_live(
                    lambda: LiveShinEnvironment(
                        cfg, camera, horizon_steps=int(cfg.sim.max_steps)),
                    model, name,
                    training_seeds,
                    target_dir, config_hash=config_hash,
                    potential=reward_design,
                    ppo=ppo, curriculum_config=curriculum, monitor=monitor if primary else None,
                    restart_incompatible=True)
                if primary:
                    for row in history:
                        row["training_replicate"] = args.training_replicate
                    models[name] = model
                    histories[name] = history
                    _write_csv(args.results_dir / f"training/{name}.csv", history)
                return model, history, target_dir / f"{name}.pt"

            # 먼저 고정 보상 arm을 학습한다. 두 reward-design dataset 모두
            # 이 실제 정책/visual-servo/noise 혼합 rollout을 출발점으로 쓴다.
            for name in args.pipelines:
                spec = get_pipeline(name)
                if not (spec.use_direct_rgat_potential
                        or spec.use_adaptive_reward_weights):
                    train_pipeline(name)

            if needs_potential and potential is None:
                preferred_source = str((config.get("rgat_design") or {}).get(
                    "source_pipeline", "no_se"))
                source_name = (preferred_source if preferred_source in available_pipeline_ids()
                               else "no_se")
                if source_name in models:
                    source_model = models[source_name]
                    source_checkpoint = (args.results_dir
                                         / f"models/{source_name}/{source_name}.pt")
                    source_training_episodes = 0
                else:
                    source_model, source_history, source_checkpoint = train_pipeline(
                        source_name, primary=False)
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
                        max_episodes_override=args.rgat_max_data_episodes,
                        source_pipeline=source_name,
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

            adaptive_design_episodes = 0
            adaptive_design_steps = 0
            if needs_adaptive and adaptive_weights is None:
                source_name = str((config.get("adaptive_reward_design") or {}).get(
                    "source_pipeline", "no_se_fixed"))
                if source_name in models:
                    adaptive_source_model = models[source_name]
                else:
                    adaptive_source_model, _, _ = train_pipeline(
                        source_name, primary=False)
                adaptive_dataset, adaptive_manifest, _, adaptive_design_steps = (
                    _collect_adaptive_data(
                        cfg=cfg, camera=camera, model=adaptive_source_model,
                        source_pipeline=source_name, config=config,
                        config_hash=config_hash, results_dir=args.results_dir,
                        mode=args.mode, monitor=monitor,
                        episodes_override=args.rgat_data_episodes,
                        max_episodes_override=args.rgat_max_data_episodes))
                adaptive_design_episodes = int(adaptive_manifest["episodes"])
                adaptive_settings = dict(config.get("adaptive_reward_design") or {})
                reward_constraints = dict(config.get("adaptive_reward") or {})
                for source_key, target_key in (
                        ("baseline_weights", "baseline_weights"),
                        ("total_weight", "total_weight"),
                        ("logit_scale_kappa", "logit_scale_kappa"),
                        ("baseline_mixture_epsilon", "baseline_mixture_epsilon")):
                    if source_key in reward_constraints:
                        adaptive_settings[target_key] = reward_constraints[source_key]
                adaptive_settings["epochs"] = int(
                    args.rgat_epochs or adaptive_settings.get(
                        f"epochs_{args.mode}", 10 if args.mode == "quick" else 80))
                expected_architectures = {
                    get_pipeline(name).adaptive_reward_architecture
                    for name in args.pipelines
                    if get_pipeline(name).use_adaptive_reward_weights}
                if len(expected_architectures) != 1:
                    raise RuntimeError(
                        "structural ablations require separate runs/artifacts per architecture")
                adaptive_settings["architecture"] = next(iter(expected_architectures))
                monitor.stage("adaptive R-GAT training", "five constrained reward weights")
                _, adaptive_metadata = prepare_adaptive_reward_artifact(
                    args.adaptive_reward_design, adaptive_dataset,
                    dataset_manifest=adaptive_manifest,
                    config_hash=config_hash, settings=adaptive_settings,
                    seed=model_seed)
                adaptive_weights = FrozenAdaptiveRewardWeights(
                    args.adaptive_reward_design,
                    expected_config_hash=config_hash)
                manifest.update({
                    "adaptive_reward_design_id": adaptive_weights.design_id,
                    "adaptive_reward_design_sha256": adaptive_weights.sha256,
                    "adaptive_reward_dataset": adaptive_manifest,
                    "adaptive_reward_model": adaptive_metadata,
                })
                _write_json(manifest_path, manifest)
            elif needs_adaptive:
                data_manifest = adaptive_weights.metadata.get(
                    "dataset_manifest") or {}
                adaptive_design_episodes = int(data_manifest.get("episodes", 0))
                adaptive_design_steps = int(data_manifest.get("transitions", 0))

            # 이제 모든 동결 보상 설계가 준비됐다. PBRS/adaptive PPO에서는
            # 이 모델들이 optimizer에 포함되지 않으며 매 update 뒤 hash를 검사한다.
            for name in args.pipelines:
                spec = get_pipeline(name)
                if spec.use_direct_rgat_potential or spec.use_adaptive_reward_weights:
                    train_pipeline(name)

            training_records = [row for name in args.pipelines
                                for row in histories.get(name, [])]
            evaluation_rows = _read_csv(existing_eval)
            for row in evaluation_rows:
                row.setdefault("training_replicate", args.training_replicate)
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
                        _, metric = collect_episode_resilient(
                            environment, model, name, int(item["seed"]),
                            curriculum=float(seminar_fast.get(
                                "evaluation_curriculum", 1.0)),
                            potential=(adaptive_weights
                                       if get_pipeline(name).use_adaptive_reward_weights
                                       else potential
                                       if get_pipeline(name).use_direct_rgat_potential
                                       else None),
                            deterministic=True, gamma=float(ppo.get("gamma", .99)),
                            shaping_lambda=float(ppo.get("shaping_lambda", 1.0)),
                            reward_normalizer=runtime_reward_normalizer,
                            scenario=item["scenario"], monitor=monitor,
                            phase="evaluation")
                        metric.update({"method": name, "pipeline": name,
                                       "training_replicate": args.training_replicate,
                                       "scenario": item["scenario"]})
                        evaluation_rows.append(metric)
                        _write_csv(existing_eval, evaluation_rows)
                        monitor.evaluation_update(name, metric)

            reports = write_three_pipeline_outputs(
                evaluation_rows, training_records, args.results_dir,
                reward_design_episodes=design_episodes + source_training_episodes,
                reward_design_steps=design_steps + source_training_steps,
                behavior_cloning_episodes=(0 if demonstrations is None else
                    len(demonstrations.get("attempted_seeds", ()))),
                behavior_cloning_steps=(0 if demonstrations is None else
                    int(demonstrations.get("environment_steps", 0))),
                estimator_warmup_episodes=selected_warmup,
                estimator_warmup_steps=sum(
                    int(float(row.get("steps", 0)))
                    for name in args.pipelines
                    if get_pipeline(name).state_estimation_enabled
                    for row in histories.get(name, [])
                    if row.get("optimization_phase") == "perception_warmup"))
            reports["adaptive_reward_figures"] = write_adaptive_reward_figures(
                args.results_dir)
            reward_design_cost = design_episodes + source_training_episodes
            reward_design_step_cost = design_steps + source_training_steps
            warmup_step_cost = sum(
                int(float(row.get("steps", 0)))
                for name in args.pipelines
                if get_pipeline(name).state_estimation_enabled
                for row in histories.get(name, [])
                if row.get("optimization_phase") == "perception_warmup")
            total_design_episodes = reward_design_cost + adaptive_design_episodes
            total_design_steps = reward_design_step_cost + adaptive_design_steps
            cloning_episodes = (0 if demonstrations is None else
                                len(demonstrations.get("attempted_seeds", ())))
            cloning_steps = (0 if demonstrations is None else
                             int(demonstrations.get("environment_steps", 0)))
            manifest.update({
                "execution_status": "complete real Isaac/Pegasus/PX4 run",
                "N_PPO": train_count * len(args.pipelines),
                "N_estimator_warmup": selected_warmup,
                "estimator_warmup_environment_steps": warmup_step_cost,
                "N_reward_design": total_design_episodes,
                "N_behavior_cloning_demonstrations": cloning_episodes,
                "behavior_cloning_environment_steps": cloning_steps,
                "N_total": (train_count * len(args.pipelines)
                            + total_design_episodes + cloning_episodes),
                "N_total_including_estimator_warmup": (
                    train_count * len(args.pipelines) + total_design_episodes
                    + cloning_episodes + selected_warmup),
                "reward_design_environment_steps": total_design_steps,
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
        if rviz is not None:
            rviz.close()
        if dashboard is not None:
            dashboard.stop()
    print(f"Three-pipeline experiment complete: {args.results_dir}")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except KeyboardInterrupt:
        print("Pipeline interrupted by user; checkpoints and completed rows were preserved.")
        exit_code = 130
    raise SystemExit(exit_code)
