#!/usr/bin/env python3
"""Shared orchestration engine; the normal entry point exposes two pipelines."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from copy import deepcopy
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import threading
import time

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
from ontology_rgat.bridge import BridgeError
from ontology_rgat.cli import ensure_fastdds
from ontology_rgat.evaluation import (write_adaptive_reward_figures,
                                      write_presentation_results,
                                      write_three_pipeline_outputs,
                                      write_two_pipeline_outputs)
from ontology_rgat.perception import (RosGrayscaleSource,
                                      calibrate_keypoint_encoder,
                                      prepare_keypoint_encoder)
from ontology_rgat.pipelines import (assert_no_aruco_in_primary_system,
                                     available_pipeline_ids, get_pipeline,
                                     primary_pipeline_ids,
                                     validate_pipeline_configuration)
from ontology_rgat.ppo.behavior_cloning import (
    behavior_clone, encoded_demonstration_episode,
    load_encoded_demonstrations, merge_encoded_demonstrations,
    save_encoded_demonstrations)
from ontology_rgat.ppo.recurrent_train import (collect_episode_resilient,
                                               aggregate_deployment_validation,
                                               deployment_validation_key,
                                               train_live)
from ontology_rgat.reward_modes import RewardComponentNormalizer
from ontology_rgat.rgat import (
    FrozenAdaptiveRewardWeights, FrozenSemanticRGATPotential,
    adaptive_episode_records, build_adaptive_dataset, load_adaptive_dataset,
    load_semantic_dataset, prepare_adaptive_reward_artifact,
    merge_semantic_datasets, prepare_semantic_rgat_artifact,
    save_adaptive_dataset, save_semantic_dataset, semantic_episode_dataset,
    FrozenFOVRiskPredictor, build_fov_risk_dataset, horizon_steps,
    load_fov_risk_dataset, prepare_fov_risk_artifact,
    save_fov_risk_dataset)
from ontology_rgat.stack import ExternalStack
from ontology_rgat.viz.contracts import (algorithm_pipeline_contract,
                                          mdp_contract)
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


class _LockedMonitor:
    """Serialize RViz/dashboard calls made by concurrent pair workers."""

    def __init__(self, monitor, lock, *, method: str = "", pair_index: int = 0,
                 pair_count: int = 1):
        self._monitor = monitor
        self._lock = lock
        self._method = str(method)
        self._pair_index = int(pair_index)
        self._pair_count = int(pair_count)

    def stage(self, name: str, detail: str = "") -> None:
        with self._lock:
            if self._pair_count > 1:
                self._monitor.store.stage(
                    name, f"{self._pair_count} independent UAV/UGV pairs configured")
                self._monitor._update_pair(
                    self._method, activity=str(name), activity_detail=str(detail),
                    index=self._pair_index)
            else:
                self._monitor.stage(name, detail)

    def __getattr__(self, name):
        value = getattr(self._monitor, name)
        if not callable(value):
            return value

        def synchronized(*args, **kwargs):
            with self._lock:
                if name in {
                        "reset_episode", "step", "training_update",
                        "evaluation_update"}:
                    kwargs.setdefault("pair_index", self._pair_index)
                return value(*args, **kwargs)
        return synchronized


def _abort_parallel_workers(executor, futures, owned) -> None:
    """Stop shared infrastructure before joining interrupted pair workers.

    ``ThreadPoolExecutor.__exit__`` waits before the pipeline's outer cleanup.
    A worker whose UDP request is interrupted by Ctrl-C would therefore see the
    still-registered stack and start it again while the main thread was trying
    to exit. Unregister and stop first; the resulting bridge errors terminate
    all workers, after which their threads can be joined normally.
    """
    stack_module.current(None)
    if owned is not None:
        if hasattr(owned, "shutdown"):
            owned.shutdown()
        else:
            owned.stop()
    for future in futures:
        future.cancel()
    executor.shutdown(wait=True, cancel_futures=True)


def _pair_live_config(cfg, pair_index: int, pair_count: int):
    """Give one learner an exclusive UDP endpoint in the shared Isaac world."""
    if pair_count == 1:
        return cfg
    index = int(pair_index)
    # Entry convergence is governed by simulated PX4 time, while its safety
    # deadline is intentionally wall time. Three rendered landing cameras make
    # the shared stage slower than real time, so retain roughly the same amount
    # of simulated settling time without changing the measured episode horizon.
    entry_timeout_scale = 1.25 if int(pair_count) == 3 else 1.10
    return cfg.derive(**{
        "external.gateway_port": int(cfg.external.gateway_port) + 2 * index,
        "external.local_port": int(cfg.external.local_port) + 2 * index,
        "external.pair_index": index,
        "external.pair_count": int(pair_count),
        "external.entry_timeout": (
            float(cfg.external.entry_timeout) * entry_timeout_scale),
    })


def _calibration_system_for_pair(system, pair_index: int, pair_count: int):
    """Mirror the landing target actually rendered for one pair.

    In the primary keypoint benchmark every pair's deck carries the identical
    six-keypoint fiducial target, and each pair's camera only ever sees its own
    deck, so there is nothing to specialise: labels come from projecting the
    same known landmarks through that pair's own training-only pose.

    The branch below is the retained legacy ArUco path, where parallel scenes
    did have to replace the dictionary and offset marker IDs to prevent
    cross-pair detections.
    """
    if int(pair_count) == 1:
        return system
    if str((system.get("vision") or {}).get("mode", "")) == "keypoint_fiducial":
        return system
    resolved = deepcopy(system)
    parallel = dict(resolved.get("parallel") or {})
    vision = dict(resolved.get("vision") or {})
    vision["dictionary"] = str(parallel.get(
        "marker_dictionary", vision.get("dictionary", "DICT_4X4_100")))
    stride = int(parallel.get("marker_id_stride", 60))
    vision["board"] = [
        {**dict(marker), "id": int(marker["id"]) + stride * int(pair_index)}
        for marker in vision.get("board") or ()
    ]
    resolved["vision"] = vision
    return resolved


def _hold_after_complete(owned, monitor, *, sleep=time.sleep) -> None:
    """Keep the completed simulator and monitoring UI alive until Ctrl-C.

    This runs only after every checkpoint, evaluation row and report has been
    committed. If an owned flight process dies while the completed dashboard
    is displayed, rebuild the stack instead of silently leaving a stale UI.
    """
    monitor.stage("complete", "results saved · simulator monitoring active")
    print("Experiment complete; dashboard, RViz and Isaac/PX4 remain active. "
          "Press Ctrl-C for a controlled shutdown.")
    while True:
        if owned is not None and not owned.is_ready():
            print("WARNING: completed flight stack is no longer ready; restarting it.")
            owned.restart()
        sleep(2.0)


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


def _adverse_landing_teacher(environment, *, settings):
    """Generate real near-miss/unsafe-contact examples for reward design only."""
    stable = _privileged_velocity_teacher(environment, settings=settings)

    def transform(step, policy_action, semantic, rng):
        action = np.asarray(stable(step, policy_action, semantic, rng),
                            dtype=np.float64)
        # Follow the pad until the marker is large, then introduce a bounded
        # lateral/descent error.  This obtains informative terminal failures
        # without synthetic transitions or motor-level unsafe commands.
        if (semantic.apparent_target_scale > .30
                and semantic.visible_keypoint_fraction >= .5):
            direction = 1.0 if (step // 20) % 2 == 0 else -1.0
            action[0] = np.clip(action[0] + .34 * direction, -.90, .90)
            action[2] = min(float(action[2]), -.62)
            action[3] = .12 * direction
        return np.clip(action, -.90, .90)

    return transform


def _read_csv(path: Path):
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _checkpoint_candidates(results_dir: Path, method: str) -> list[dict]:
    """Return training-best/latest snapshots with stable identities."""
    model_dir = Path(results_dir) / "models" / str(method)
    candidates = []
    for kind, path in (
            ("training_best", model_dir / f"{method}.best.pt"),
            ("training_latest", model_dir / f"{method}.pt")):
        if not path.is_file():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        candidates.append({
            "kind": kind, "path": path, "sha256": _sha256_file(path),
            "episode": int(payload.get("episode", 0)),
        })
    if not candidates:
        raise FileNotFoundError(f"no checkpoint candidates for {method}")
    return candidates


def _load_reward_design_source_checkpoint(results_dir: Path, method: str, model,
                                           *, config_hash: str):
    """Load a compatible policy only as an empirical data-collection source.

    A new PPO training contract must invalidate publication checkpoints, but
    the old policy still supplies useful non-synthetic success/failure
    trajectories for the offline reward model. This path never registers the
    model as a completed comparison arm.
    """
    model_spec = getattr(model, "pipeline_spec", None)
    model_dir = Path(results_dir) / "models" / str(method)
    for path in (model_dir / f"{method}.best.pt", model_dir / f"{method}.pt"):
        if not path.is_file():
            continue
        payload = torch.load(path, map_location=model.device, weights_only=False)
        if (payload.get("format") !=
                "three-pipeline-recurrent-v3-scaled-estimator"):
            continue
        if (payload.get("method") != method
                or payload.get("config_hash") != config_hash):
            continue
        if (model_spec is not None
                and payload.get("pipeline_spec") != model_spec.to_manifest()):
            continue
        model.load_state_dict(payload["model"])
        model.eval()
        return path
    return None


def _checkpoint_validation_plan(method: str, scenarios, *, seed0: int) -> list[dict]:
    """Use one held-out deterministic seed per scenario for deployment selection."""
    return [{"method": str(method), "scenario": str(scenario),
             "seed": int(seed0) + index}
            for index, scenario in enumerate(scenarios)]


def _atomic_save_selected_checkpoint(path: Path, payload: dict, *, candidate,
                                     summary: dict, validation_seeds) -> None:
    selected = dict(payload)
    selected["training_selection_score"] = selected.get("selection_score")
    selected["selection_score"] = float(summary["mean_physical_score"])
    selected["selection_metric"] = dict(summary)
    selected["selection_method"] = "held_out_deterministic_multi_seed_v1"
    selected["selection_candidate_kind"] = str(candidate["kind"])
    selected["selection_candidate_sha256"] = str(candidate["sha256"])
    selected["selection_validation_seeds"] = [int(seed) for seed in validation_seeds]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(selected, temporary)
    os.replace(temporary, path)


def _crossover_evaluation_tasks(plan, pipelines, pair_count: int) -> list[list[dict]]:
    """Balance every method across the available physical route phases."""
    count = int(pair_count)
    if count < 1:
        raise ValueError("pair_count must be positive")
    methods = list(pipelines)
    by_method = {name: [dict(row) for row in plan if row["method"] == name]
                 for name in methods}
    tasks = [[] for _ in range(count)]
    rounds = max((len(rows) for rows in by_method.values()), default=0)
    for episode_index in range(rounds):
        for method_index, name in enumerate(methods):
            if episode_index >= len(by_method[name]):
                continue
            row = by_method[name][episode_index]
            pair_index = (method_index + episode_index) % count
            row["physical_pair_index"] = pair_index
            tasks[pair_index].append(row)
    return tasks


def _balanced_training_pair_assignment(pipelines, pair_count: int,
                                       replicate: int) -> tuple[dict, list]:
    """Counterbalance method-to-route-phase assignment across replicates."""
    methods = list(pipelines)
    count = int(pair_count)
    if count < 1 or (count > 1 and count != len(methods)):
        raise ValueError("parallel training needs one method per physical pair")
    if count == 1:
        return {name: 0 for name in methods}, methods[:1]
    offset = int(replicate) % count
    assignment = {
        name: (method_index + offset) % count
        for method_index, name in enumerate(methods)}
    pair_methods = [next(name for name, index in assignment.items()
                         if index == pair_index)
                    for pair_index in range(count)]
    return assignment, pair_methods


def _adaptive_reward_settings(config, *, robust: bool = False) -> dict:
    """Resolve artifact settings, including the explicit robust live profile."""
    settings = deepcopy(dict(config.get("adaptive_reward_design") or {}))
    reward = dict(config.get("adaptive_reward") or {})
    for source_key, target_key in (
            ("baseline_weights", "baseline_weights"),
            ("total_weight", "total_weight"),
            ("logit_scale_kappa", "logit_scale_kappa"),
            ("baseline_mixture_epsilon", "baseline_mixture_epsilon"),
            ("semantic_potential_shaping_lambda",
             "semantic_potential_shaping_lambda")):
        if source_key in reward:
            settings[target_key] = reward[source_key]
    if robust:
        # The small seminar artifact collapsed to nearly constant baseline
        # weights. This profile is explicit in the CLI/manifest and trades a
        # modest offline cost for observable state-dependent reward variation.
        # Keep the experiment's declared CV acceptance criterion authoritative:
        # raising it silently here made a valid, fully collected artifact abort
        # the entire live pipeline after the rollout cap was reached.
        settings["logit_scale_kappa"] = max(
            1.0, float(settings.get("logit_scale_kappa", 1.0)))
        settings["baseline_mixture_epsilon"] = min(
            .10, float(settings.get("baseline_mixture_epsilon", .10)))
        settings["semantic_potential_shaping_lambda"] = max(
            1.50, float(settings.get("semantic_potential_shaping_lambda", 1.50)))
        loss = dict(settings.get("loss") or {})
        loss.update({
            "baseline_prior": min(.002, float(loss.get("baseline_prior", .002))),
            "contextual_weight": max(1.50, float(loss.get("contextual_weight", 1.50))),
            "semantic_potential": max(1.0, float(loss.get(
                "semantic_potential", 1.0))),
            "observability_monotonic": max(.50, float(loss.get(
                "observability_monotonic", .50))),
        })
        settings["loss"] = loss
        quality = dict(settings.get("quality_gate") or {})
        quality.update({
            "enabled": True,
            "minimum_validation_episodes": max(
                4, int(quality.get("minimum_validation_episodes", 4))),
            "minimum_validation_accuracy": max(
                .50, float(quality.get("minimum_validation_accuracy", .50))),
            "minimum_validation_ranking_accuracy": max(
                .70, float(quality.get(
                    "minimum_validation_ranking_accuracy", .70))),
            "minimum_mean_weight_cv": max(
                .003, float(quality.get("minimum_mean_weight_cv", .003))),
            "minimum_potential_monotonic_compliance": max(
                .70, float(quality.get(
                    "minimum_potential_monotonic_compliance", .70))),
            "minimum_unsafe_failure_episodes": max(
                2, int(quality.get("minimum_unsafe_failure_episodes", 2))),
        })
        settings["quality_gate"] = quality
        settings["runtime_profile"] = "robust_live_v2"
    return settings


def _adaptive_artifact_quality_issues(metadata, settings, *,
                                      minimum_episodes: int = 0) -> list[str]:
    """Audit a cached artifact against today's requested quality contract."""
    model_metrics = dict(metadata.get("metrics") or {})
    dataset = dict(metadata.get("dataset_manifest") or {})
    strata = dict(dataset.get("outcome_strata") or {})
    quality = dict(settings.get("quality_gate") or {})
    issues = []
    episodes = int(dataset.get("episodes", 0))
    if episodes < int(minimum_episodes):
        issues.append(f"dataset episodes {episodes} < {int(minimum_episodes)}")
    validation_count = int(dataset.get(
        "validation_episodes", len(metadata.get("validation_episode_ids") or ())))
    required_validation = int(quality.get("minimum_validation_episodes", 2))
    if validation_count < required_validation:
        issues.append(
            f"validation episodes {validation_count} < {required_validation}")
    accuracy = model_metrics.get("validation_accuracy")
    required_accuracy = float(quality.get("minimum_validation_accuracy", .50))
    if accuracy is None or float(accuracy) < required_accuracy:
        issues.append(f"validation accuracy {accuracy} < {required_accuracy:.3f}")
    ranking_accuracy = model_metrics.get("validation_ranking_accuracy")
    required_ranking = float(quality.get(
        "minimum_validation_ranking_accuracy", .50))
    if ranking_accuracy is None or float(ranking_accuracy) < required_ranking:
        issues.append(
            f"validation ranking accuracy {ranking_accuracy} < "
            f"{required_ranking:.3f}")
    cv = float(model_metrics.get("mean_weight_coefficient_of_variation", 0.0))
    required_cv = float(quality.get("minimum_mean_weight_cv", .005))
    if cv < required_cv:
        issues.append(f"mean weight CV {cv:.6f} < {required_cv:.6f}")
    compliance = float(model_metrics.get(
        "potential_observability_monotonic_compliance", 0.0))
    required_compliance = float(quality.get(
        "minimum_potential_monotonic_compliance", .55))
    if compliance < required_compliance:
        issues.append(
            f"potential compliance {compliance:.3f} < {required_compliance:.3f}")
    unsafe = sum(int(strata.get(name, 0)) for name in (
        "unsafe_pad_contact", "collision", "excessive_drift"))
    required_unsafe = int(quality.get("minimum_unsafe_failure_episodes", 0))
    if unsafe < required_unsafe:
        issues.append(f"unsafe failure episodes {unsafe} < {required_unsafe}")
    return issues


def _archive_rejected_adaptive_artifact(path: Path, design_id: str = "unknown") -> None:
    """Preserve a rejected reward model and its provenance before replacement."""
    path = Path(path)
    if not path.is_file():
        return
    label = str(design_id or "unknown")[:16]
    targets = [path, path.with_suffix(".manifest.json"),
               path.parent / "adaptive_training_history.csv"]
    for source in targets:
        if not source.is_file():
            continue
        destination = source.with_name(
            f"{source.stem}.rejected-{label}{source.suffix}")
        sequence = 1
        while destination.exists():
            destination = source.with_name(
                f"{source.stem}.rejected-{label}-{sequence}{source.suffix}")
            sequence += 1
        os.replace(source, destination)


def _prepare_fast_demonstrations(*, cfg, camera, config, config_hash,
                                 keypoint_pretraining, results_dir, device,
                                 model_seed, monitor):
    """Collect successful teacher flights once and store compact actor inputs."""
    fast = dict(config.get("seminar_fast") or {})
    settings = dict(fast.get("behavior_cloning") or {})
    if not bool(settings.get("enabled", False)):
        return None
    required = max(1, int(settings.get("successful_episodes", 6)))
    maximum_flights = max(required, int(settings.get("max_attempts", 12)))
    maximum_infrastructure_skips = max(0, int(settings.get(
        "max_infrastructure_skips", maximum_flights)))
    maximum_seed_candidates = maximum_flights + maximum_infrastructure_skips
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
    flight_attempts = sum(
        str(row.get("status", "")) != "infrastructure_failure"
        for row in attempts)
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
                f"{flight_attempts}/{maximum_flights} flights and "
                f"{len(attempted_seeds) - flight_attempts} infrastructure skips.")
        except (OSError, ValueError, KeyError) as exc:
            print(f"Ignoring incompatible training-teacher demonstrations: {exc}")
            payload, dataset, attempts, attempted_seeds, environment_steps = (
                None, None, [], [], 0)
            flight_attempts = 0
    successes = (0 if dataset is None else
                 int(torch.unique(dataset["episode_id"]).numel()))
    if successes < required and flight_attempts < maximum_flights:
        torch.manual_seed(int(model_seed))
        teacher_model = _build_model(
            config, device, keypoint_pretraining, pipeline=source_pipeline)
        seed0 = int((config.get("seeds") or {}).get(
            "behavior_cloning_start", 90000))
        monitor.stage(
            "training-only teacher demonstrations",
            f"successful real Isaac/PX4 flights {successes}/{required}")
        original_entry_timeout = float(cfg.external.entry_timeout)
        original_reset_recoveries = int(cfg.external.reset_recoveries)
        original_episode_recoveries = int(cfg.external.get(
            "episode_recoveries", original_reset_recoveries))
        cfg.external.entry_timeout = min(original_entry_timeout, 45.0)
        cfg.external.reset_recoveries = 0
        cfg.external.episode_recoveries = 0
        with LiveShinEnvironment(
                cfg, camera, horizon_steps=int(cfg.sim.max_steps)) as environment:
            teacher = _privileged_velocity_teacher(
                environment, settings=settings)
            for attempt in range(maximum_seed_candidates):
                if flight_attempts >= maximum_flights:
                    break
                seed = seed0 + attempt
                if seed in attempted_seeds:
                    continue
                try:
                    rows, metric = collect_episode_resilient(
                        environment, teacher_model, source_pipeline, seed,
                        curriculum=float(settings.get("curriculum", 1.0)),
                        deterministic=True, scenario="training_random_walk",
                        monitor=monitor,
                        phase="training-only teacher demonstration",
                        action_transform=teacher)
                except BridgeError as exc:
                    # One seeded entry can fail PX4 preflight deterministically.
                    # It is infrastructure downtime, not an RL failure: record
                    # no transition/label, advance to the next seed, and give
                    # that seed a freshly owned stack.
                    attempted_seeds.append(seed)
                    attempts.append({
                        "method": "privileged_teacher",
                        "pipeline": "shared_warm_start",
                        "episode": len(attempted_seeds),
                        "seed": seed,
                        "status": "infrastructure_failure",
                        "accepted_for_cloning": 0.0,
                        "paper_success": 0.0,
                        "strict_success": 0.0,
                        "steps": 0,
                        "teacher": teacher_id,
                        "config_hash": config_hash,
                        "infrastructure_error": str(exc),
                    })
                    _write_csv(attempts_path, attempts)
                    print(
                        f"training teacher seed {seed} skipped after PX4/Isaac "
                        f"reset failure: {exc}")
                    environment.recover_infrastructure()
                    continue
                attempted_seeds.append(seed)
                flight_attempts += 1
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
                    f"training teacher flight {flight_attempts}/{maximum_flights} "
                    f"success={int(metric['paper_success'])} "
                    f"accepted={successes}/{required}")
                if successes >= required:
                    break
        cfg.external.entry_timeout = original_entry_timeout
        cfg.external.reset_recoveries = original_reset_recoveries
        cfg.external.episode_recoveries = original_episode_recoveries
        del teacher_model
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    if payload is None or int(payload["successful_episodes"]) < required:
        raise RuntimeError(
            "training teacher did not produce enough real successful landings "
            f"({successes}/{required}) after {flight_attempts}/{maximum_flights} "
            f"flights and {len(attempted_seeds) - flight_attempts} "
            "infrastructure skips")
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
                float(row.get("geometric_fov_loss_events", 0)) > 0 for row in rows),
            "episodes_with_reacquisition": sum(
                float(row.get("geometric_fov_reacquisition_events", 0)) > 0
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
                    "geometric_fov_loss_events": metric["geometric_fov_loss_events"],
                    "geometric_fov_reacquisition_events": metric[
                        "geometric_fov_reacquisition_events"],
                    "geometric_fov_reacquisition_rate": metric[
                        "geometric_fov_reacquisition_rate"],
                    "mean_geometric_fov_reacquisition_time_s": metric[
                        "mean_geometric_fov_reacquisition_time_s"],
                    "climb_during_geometric_fov_loss_fraction": metric["climb_during_geometric_fov_loss_fraction"],
                    "descent_during_low_keypoint_visibility_fraction": metric[
                        "descent_during_low_keypoint_visibility_fraction"],
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


def _fov_target_coverage(dataset) -> bool:
    """Both regimes present: some windows fully visible, some with loss.

    The target is a time fraction, so the old two-class check no longer
    applies. Coverage means the supervised targets are not all identical --
    a dataset of only zeros would train a constant predictor.
    """
    y = np.asarray(dataset["y"], dtype=np.float64)
    valid = np.asarray(dataset["valid"], dtype=bool)
    supervised = y[valid]
    if supervised.size == 0:
        return False
    return bool(np.any(supervised <= 0.0) and np.any(supervised > 0.0))


def _fov_dataset_progress(dataset) -> dict:
    """Counts the collection loop already has, in one place for telemetry."""
    y = np.asarray(dataset["y"], dtype=np.float64)
    valid = np.asarray(dataset["valid"], dtype=bool)
    supervised = y[valid]
    return {
        "supervised_samples": int(valid.sum()),
        "masked_samples": int((~valid).sum()),
        "target_mean": (float(np.mean(supervised)) if supervised.size else None),
        "covered": _fov_target_coverage(dataset),
    }


def _collect_fov_risk_data(*, cfg, camera, model, config, config_hash,
                           checkpoint_path, results_dir, mode, monitor,
                           episodes_override=None, max_episodes_override=None,
                           source_pipeline="shin_se_fixed"):
    """Collect same-domain trajectories and label future visibility offline."""
    design = dict(config.get("fov_risk_design") or {})
    risk = dict(config.get("fov_risk") or {})
    minimum = int(episodes_override if episodes_override is not None else
                  design.get(f"episodes_{mode}", 8 if mode == "quick" else 40))
    maximum = int(max_episodes_override if max_episodes_override is not None else
                  design.get(f"max_episodes_{mode}", max(3 * minimum, minimum)))
    if minimum < 2 or maximum < minimum:
        raise ValueError("FOV-risk collection requires 2+ episodes and a valid cap")
    seconds = float(risk.get("prediction_horizon_seconds", 1.0))
    control_hz = 1.0 / float(cfg.sim.dt)
    prediction_steps = horizon_steps(seconds, control_hz)
    dataset_path = Path(results_dir) / "rgat/fov_risk_rollouts.npz"
    if dataset_path.is_file() and dataset_path.with_suffix(".manifest.json").is_file():
        try:
            cached, cached_manifest = load_fov_risk_dataset(
                dataset_path, config_hash=config_hash)
            if (int(cached_manifest.get("episodes", 0)) >= minimum
                    and _fov_target_coverage(cached)):
                steps = int(cached_manifest.get(
                    "environment_steps", cached_manifest["samples"]))
                monitor.fov_dataset(
                    episodes=int(cached_manifest.get("episodes", 0)),
                    minimum=minimum, maximum=maximum,
                    loss_episodes=0, environment_steps=steps, cached=True,
                    **_fov_dataset_progress(cached))
                return cached, cached_manifest, dataset_path, steps
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            print(f"Ignoring incompatible FOV-risk rollout cache: {exc}")

    seed0 = int((config.get("seeds") or {}).get("fov_risk_dataset_start", 70000))
    episodes = []
    total_steps = 0
    loss_episodes = 0
    monitor.stage("FOV-risk data", "visual-only graph · future visibility labels")
    with LiveShinEnvironment(
            cfg, camera, horizon_steps=int(cfg.sim.max_steps)) as environment:
        for episode_id in range(1, maximum + 1):
            seed = seed0 + episode_id - 1
            rows, metric = collect_episode_resilient(
                environment, model, source_pipeline, seed, curriculum=1.0,
                deterministic=False, scenario="training_random_walk",
                monitor=monitor, phase="FOV-risk offline data",
                action_transform=_behavior_transform((episode_id - 1) % 3))
            episodes.append({
                "episode_id": episode_id,
                "seed": seed,
                "samples": ([{"graph_X": row["fov_graph_X"],
                              "geometric_in_fov": row["fov_graph_geometric_in_fov"]}
                             for row in rows]
                            + ([{"graph_X": rows[-1]["next_fov_graph_X"],
                                 "geometric_in_fov": rows[-1]["geometric_in_fov"]}]
                               if rows else [])),
            })
            total_steps += len(rows)
            loss_episodes += int(metric["geometric_fov_loss_episode_rate"])
            dataset = build_fov_risk_dataset(
                episodes, prediction_steps=prediction_steps)
            progress = _fov_dataset_progress(dataset)
            monitor.fov_dataset(
                episodes=episode_id, minimum=minimum, maximum=maximum,
                loss_episodes=loss_episodes, environment_steps=total_steps,
                **progress)
            if episode_id >= minimum and progress["covered"]:
                break
            print(f"FOV-risk data {episode_id}/{minimum} minimum "
                  f"(cap {maximum}): loss_episode={int(metric['geometric_fov_loss_episode_rate'])}")
    if not _fov_target_coverage(dataset):
        raise RuntimeError(
            f"FOV-risk data covers one target regime after {maximum} episodes; "
            "increase --rgat-max-data-episodes")
    manifest = save_fov_risk_dataset(
        dataset, dataset_path, config_hash=config_hash, seed=seed0,
        horizon_seconds=seconds, control_hz=control_hz)
    manifest.update({
        "source_pipeline": source_pipeline,
        "source_checkpoint_sha256": (
            _sha256_file(checkpoint_path) if checkpoint_path is not None
            and Path(checkpoint_path).is_file() else None),
        "environment_steps": total_steps,
        "episode_split_required": True,
    })
    # Persist the extended provenance without altering the checked NPZ digest.
    _write_json(dataset_path.with_suffix(".manifest.json"), manifest)
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
                           episodes_override=None, max_episodes_override=None,
                           minimum_unsafe_failures_override=None,
                           source_checkpoint_path=None,
                           parallel_contexts=None):
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
        "training_random_walk", "circle", "zigzag", "vertical_heave_boat"))
    fast_settings = dict(
        ((config.get("seminar_fast") or {}).get("behavior_cloning") or {}))
    deadline_teacher_enabled = bool(fast_settings.get("enabled", False))
    completed_episodes = set(int(row["episode_id"]) for row in records)

    minimum_successes = max(2, int(design.get("minimum_successful_episodes", 3)))
    minimum_failures = max(2, int(design.get("minimum_failed_episodes", 3)))
    minimum_risky = max(1, int(design.get("minimum_risky_failures", 2)))
    minimum_unsafe = max(0, int(
        minimum_unsafe_failures_override
        if minimum_unsafe_failures_override is not None else
        design.get("minimum_unsafe_failure_episodes", 0)))

    def dataset_quality(dataset):
        if dataset is None:
            return {"ready": False, "episodes": 0, "success": 0,
                    "failure": 0, "risky_failure": 0, "unsafe_failure": 0,
                    "validation_classes": []}
        episodes = np.unique(dataset["episode_id"])
        representative = [int(np.flatnonzero(dataset["episode_id"] == ep)[0])
                          for ep in episodes]
        success = np.asarray(dataset["success"], dtype=int)
        failure_type = np.asarray(dataset["failure_type"]).astype(str)
        risky = sum(
            not int(success[index]) and (
                failure_type[index] in {"collision", "unsafe_pad_contact"}
                or (float(dataset["touchdown_error"][index]) <= .70
                    and abs(float(dataset["touchdown_vertical_speed"][index])) <= 1.0))
            for index in representative)
        unsafe = sum(
            not int(success[index]) and failure_type[index] in {
                "collision", "unsafe_pad_contact", "excessive_drift"}
            for index in representative)
        validation = np.asarray(dataset["split"]).astype(str) == "validation"
        validation_classes = sorted(set(success[validation].tolist()))
        successes = sum(int(success[index]) for index in representative)
        failures = len(representative) - successes
        ready = (len(episodes) >= count and successes >= minimum_successes
                 and failures >= minimum_failures and risky >= minimum_risky
                 and unsafe >= minimum_unsafe
                 and validation_classes == [0, 1])
        return {"ready": bool(ready), "episodes": len(episodes),
                "success": successes, "failure": failures,
                "risky_failure": int(risky), "unsafe_failure": int(unsafe),
                "validation_classes": validation_classes}

    def requirements_met(dataset):
        return bool(dataset_quality(dataset)["ready"])

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
    collection_contexts = list(parallel_contexts or ({
        "cfg": cfg, "camera": camera, "monitor": monitor,
    },))
    if not collection_contexts:
        raise ValueError("adaptive reward data needs at least one live pair")
    for context in collection_contexts:
        if not all(key in context for key in ("cfg", "camera", "monitor")):
            raise ValueError("each adaptive collection context needs cfg/camera/monitor")

    behavior = {
        "name": str(design.get(
            "behavior_policy",
            "mixed_random_fixed_success_collision_drift_near_miss")),
        "source_pipeline": str(source_pipeline),
        "source_checkpoint": (None if source_checkpoint_path is None else
                              str(Path(source_checkpoint_path).resolve())),
        "source_checkpoint_sha256": (
            None if source_checkpoint_path is None else
            _sha256_file(Path(source_checkpoint_path))),
        "components": [
            "trained_fixed_policy",
            ("training_only_privileged_velocity_teacher"
             if deadline_teacher_enabled else "visual_servo_success_recovery"),
            "visual_servo_recovery", "moderate_noise_near_miss",
            "bounded_adverse_landing_teacher", "bounded_random_exploration",
            "empirical_collision_or_drift_failures"],
        "deadline_teacher": (
            str(fast_settings.get("teacher"))
            if deadline_teacher_enabled else None),
        "scenario_cycle": list(scenarios),
        "parallel_collection_pairs": len(collection_contexts),
        "synthetic_transitions_allowed": False,
    }
    pending = [(episode, seed0 + episode - 1)
               for episode in range(1, maximum + 1)
               if episode not in completed_episodes]
    if pending and not requirements_met(dataset):
        for context in collection_contexts:
            context["monitor"].stage(
                "adaptive reward data",
                f"real transition outcome mixture · {len(collection_contexts)} pairs")
        with ExitStack() as environment_stack:
            environments = [environment_stack.enter_context(
                LiveShinEnvironment(
                    context["cfg"], context["camera"],
                    horizon_steps=int(context["cfg"].sim.max_steps)))
                for context in collection_contexts]
            # Actor inference is read-only, but separate modules avoid any
            # accidental recurrent/module state sharing between worker threads.
            worker_models = [model] + [deepcopy(model)
                                       for _ in environments[1:]]
            for worker_model in worker_models:
                modules = getattr(worker_model, "modules", lambda: ())
                for module in modules():
                    flatten = getattr(module, "flatten_parameters", None)
                    if callable(flatten):
                        flatten()

            def collect_one(worker_index, item):
                episode, seed = item
                environment = environments[worker_index]
                local_monitor = collection_contexts[worker_index]["monitor"]
                scenario = scenarios[(episode - 1) % len(scenarios)]
                variant = (episode - 1) % (5 if deadline_teacher_enabled else 3)
                if deadline_teacher_enabled and variant == 0:
                    transform = _privileged_velocity_teacher(
                        environment, settings=fast_settings)
                elif deadline_teacher_enabled and variant == 3:
                    transform = _adverse_landing_teacher(
                        environment, settings=fast_settings)
                else:
                    transform = _behavior_transform(
                        {1: 0, 2: 1, 4: 2}.get(variant, variant))
                rows, metric = collect_episode_resilient(
                    environment, worker_models[worker_index], source_pipeline,
                    seed, curriculum=1.0, deterministic=False,
                    scenario=scenario, monitor=local_monitor,
                    phase="adaptive reward-design data",
                    action_transform=transform)
                return episode, seed, scenario, rows, metric

            cursor = 0
            while cursor < len(pending) and not requirements_met(dataset):
                batch = pending[cursor:cursor + len(environments)]
                cursor += len(batch)
                if len(batch) == 1:
                    completed_batch = [collect_one(0, batch[0])]
                else:
                    with ThreadPoolExecutor(
                            max_workers=len(batch),
                            thread_name_prefix="adaptive-data-pair") as executor:
                        futures = [executor.submit(collect_one, index, item)
                                   for index, item in enumerate(batch)]
                        completed_batch = [future.result() for future in futures]
                # Persist only on the main thread and in episode order. A crash
                # can therefore resume from a complete, deterministic prefix.
                for episode, seed, scenario, rows, metric in sorted(
                        completed_batch, key=lambda result: result[0]):
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
                            test_fraction=float(
                                design.get("test_fraction", 0.0)),
                            physical_scales=normalization.get("scales"),
                            normalization_quantile=float(
                                normalization.get("quantile", .99)),
                            exact_paper_raw=bool(
                                reward_cfg.get("exact_paper_raw", False)))
                        manifest = save_adaptive_dataset(
                            dataset, path, config_hash=config_hash,
                            source_behavior_policy=behavior)
                    quality = dataset_quality(dataset)
                    print(f"adaptive data episode {episode}/{count} minimum "
                          f"success={int(metric['paper_success'])} steps={len(rows)} "
                          f"strata=S{quality['success']}/F{quality['failure']}/"
                          f"R{quality['risky_failure']}/U{quality['unsafe_failure']}")
    if dataset is None:
        raise RuntimeError("adaptive reward-design dataset is unavailable")
    if not requirements_met(dataset):
        quality = dataset_quality(dataset)
        raise RuntimeError(
            "adaptive reward data failed its stratified quality contract at the hard "
            f"cap: {quality}; require at least S{minimum_successes}/F{minimum_failures}/"
            f"risky-F{minimum_risky}/unsafe-F{minimum_unsafe} and both validation classes")
    if manifest is None:
        manifest = save_adaptive_dataset(
            dataset, path, config_hash=config_hash,
            source_behavior_policy=behavior)
    steps = int(len(dataset["episode_id"]))
    return dataset, manifest, path, steps


def main(*, primary_only: bool = False):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--experiment", choices=(
        ("two_pipeline_fov_risk",) if primary_only else
        ("three_pipeline", "adaptive_reward_weight_comparison")),
                        default=None)
    parser.add_argument(
        "--pipelines", nargs="+",
        choices=(primary_pipeline_ids() if primary_only else available_pipeline_ids()))
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--config", type=Path,
                        default=ROOT / "config/experiments/two_pipeline_comparison.yaml"
                        if primary_only else
                        ROOT / "config/experiments/three_pipeline_comparison.yaml")
    parser.add_argument("--system-config", type=Path,
                        default=ROOT / "config/shin2026-system.yaml")
    parser.add_argument("--results-dir", type=Path)
    if primary_only:
        parser.set_defaults(reward_design=None, adaptive_reward_design=None,
                            robust_adaptive_reward=False)
    else:
        parser.add_argument("--reward-design", type=Path)
        parser.add_argument("--adaptive-reward-design", type=Path)
    parser.add_argument("--fov-risk-model", type=Path)
    parser.add_argument("--no-prepare-reward-design", action="store_true")
    parser.add_argument("--rgat-data-episodes", type=int)
    parser.add_argument(
        "--rgat-max-data-episodes", type=int,
        help="hard cap for automatic real rollout extension when one class is missing")
    parser.add_argument("--rgat-epochs", type=int)
    if not primary_only:
        parser.add_argument(
            "--robust-adaptive-reward", action="store_true",
            help="reject collapsed adaptive artifacts and use the robust live R-GAT profile")
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
    parser.add_argument(
        "--stay-open", action="store_true",
        help="after successful completion keep Isaac/PX4, RViz and dashboard active")
    parser.add_argument("--isaac-sim-path")
    parser.add_argument("--isaac-timeout", type=float)
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--no-rviz", action="store_true")
    parser.add_argument("--dashboard-port", type=int)
    parser.add_argument(
        "--parallel-pairs", type=int, default=(2 if primary_only else 3),
        help="number of isolated UAV/UGV pairs in the shared Isaac stage")
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
    if not 1 <= args.parallel_pairs <= 3:
        parser.error("--parallel-pairs must be between 1 and 3")

    config = load_experiment(args.config)
    seminar_fast = dict(config.get("seminar_fast") or {})
    if args.experiment is None:
        args.experiment = str(config.get("experiment", "three_pipeline"))
    validate_pipeline_configuration(config)
    configured = tuple(config.get("pipelines") or ())
    if args.pipelines is None:
        args.pipelines = list(configured)
    if primary_only and len(set(args.pipelines)) != len(args.pipelines):
        parser.error("a primary pipeline may be selected only once")
    if any(name not in configured for name in args.pipelines):
        parser.error("selected pipeline is absent from experiment configuration")
    for name in args.pipelines:
        get_pipeline(name)
    if any(name in primary_pipeline_ids() for name in args.pipelines):
        # Fail before the stack starts if the simulator profile still paints
        # an ArUco board or lets a detector-solved pose drive the policy.
        assert_no_aruco_in_primary_system(
            load_system_config(Path(args.system_config)))
    if args.parallel_pairs > 1 and args.parallel_pairs != len(args.pipelines):
        parser.error(
            "parallel mode requires exactly one selected pipeline per UAV/UGV pair")
    if args.results_dir is None:
        experiment_dir = ("two_pipeline_fov_risk" if
                          config.get("experiment") == "two_pipeline_fov_risk" else
                          "adaptive_reward_weight" if
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
    if args.fov_risk_model is None:
        args.fov_risk_model = args.results_dir / "rgat/fov_risk_model.pt"
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
    # Parallel placement, namespaces and ports change orchestration, not the
    # relative landing task or reward/actor contract. Keeping them outside the
    # scientific hash lets a completed single-pair checkpoint be evaluated in
    # the equivalent translated multi-pair world. They remain explicit in the
    # manifest, and the live camera encoder is still empirically revalidated.
    scientific_system = deepcopy(system)
    scientific_system.pop("parallel", None)
    config_hash = configuration_hash({
        "experiment": config, "system": scientific_system,
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
    needs_fov_risk = any(
        get_pipeline(name).fov_risk_reward_enabled for name in args.pipelines)
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
    adaptive_rejected_design_id = "unknown"
    adaptive_settings = _adaptive_reward_settings(
        config, robust=args.robust_adaptive_reward)
    adaptive_settings["epochs"] = int(
        args.rgat_epochs or adaptive_settings.get(
            f"epochs_{args.mode}", 10 if args.mode == "quick" else 80))
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
            quality_issues = _adaptive_artifact_quality_issues(
                adaptive_weights.metadata, adaptive_settings,
                minimum_episodes=int(args.rgat_data_episodes or
                    adaptive_settings.get(
                        f"episodes_{args.mode}", 8 if args.mode == "quick" else 40)))
            if quality_issues:
                raise ValueError("; ".join(quality_issues))
            print(f"Using frozen adaptive reward R-GAT {adaptive_weights.design_id}.")
        except (OSError, ValueError, KeyError) as exc:
            adaptive_error = str(exc)
            adaptive_rejected_design_id = getattr(
                adaptive_weights, "design_id", "unknown")
            adaptive_weights = None
            print(f"Existing adaptive reward model is not reusable: {exc}")
    if needs_adaptive and adaptive_weights is None and args.no_prepare_reward_design:
        parser.error(
            f"adaptive reward mode requires a valid frozen model: {adaptive_error}")
    fov_risk_model = None
    fov_risk_error = None
    if needs_fov_risk and args.fov_risk_model.is_file():
        try:
            fov_risk_model = FrozenFOVRiskPredictor(
                args.fov_risk_model, expected_config_hash=config_hash,
                device=args.device)
            print("Using frozen future-FOV-unavailability R-GAT "
                  f"{fov_risk_model.design_id}.")
        except (OSError, ValueError, KeyError) as exc:
            fov_risk_error = str(exc)
            print(f"Existing FOV-risk R-GAT is not reusable: {exc}")
    if needs_fov_risk and fov_risk_model is None and args.no_prepare_reward_design:
        parser.error(f"proposed pipeline requires a valid FOV-risk R-GAT: {fov_risk_error}")

    controlled_fields = {
        name: config.get(name) for name in
        ("camera", "estimator", "control", "ppo", "curriculum", "seeds")
    }
    landing_contract = dict(system.get("landing") or {})
    manifest = {
        "format": ("ontology_rgat.two_pipeline_fov_risk_experiment/1"
                   if primary_only else "ontology_rgat.legacy_multi_pipeline_experiment/1"),
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
        "reward_design_collection_contract": ({
            "target": "future FOV loss within configured horizon",
            "minimum_episodes": design_minimum,
            "maximum_episodes": design_maximum,
            "split": "whole episodes with no train/validation overlap",
            "loss": "binary cross entropy only",
            "synthetic_outcomes_allowed": False,
        } if needs_fov_risk else _reward_design_collection_contract(
            design_cfg, args.mode, design_minimum, design_maximum)),
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
            "pending held-out deterministic best-vs-latest validation"),
        "reward_design_id": getattr(potential, "design_id", None),
        "adaptive_reward_design_id": getattr(adaptive_weights, "design_id", None),
        "fov_risk_design_id": getattr(fov_risk_model, "design_id", None),
        "fov_risk": config.get("fov_risk"),
        "adaptive_reward": config.get("adaptive_reward"),
        "adaptive_reward_runtime_profile": adaptive_settings.get(
            "runtime_profile", "configured"),
        "seminar_fast": seminar_fast or None,
        "execution_status": "configured; real Isaac/Pegasus/PX4 results pending",
        "parallel_execution": {
            "pair_count": args.parallel_pairs,
            "one_pipeline_per_pair": args.parallel_pairs > 1,
            "shared_isaac_stage": args.parallel_pairs > 1,
            "shared_dds_agent": args.parallel_pairs > 1,
            "dependency_aware_training": args.parallel_pairs > 1,
            "reward_design_uses_dedicated_pair": args.parallel_pairs > 1,
            "independent_udp_ros_reset_and_optimizer": args.parallel_pairs > 1,
            "layout": system.get("parallel"),
        },
    }
    manifest_path = args.results_dir / "manifest.json"
    existing_eval = args.results_dir / "evaluation/per_episode.csv"
    if existing_eval.is_file() and manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("config_hash") != config_hash:
            parser.error("existing results use a different experiment configuration")
    _write_json(manifest_path, manifest)
    _write_csv(args.results_dir / "evaluation/paired_plan.csv", plan)
    presentation_lock = threading.RLock()

    def refresh_presentation_results():
        """결과 그림 실패가 비행/학습을 중단시키지 않게 별도로 갱신한다."""
        with presentation_lock:
            try:
                return write_presentation_results(args.results_dir)
            except Exception as exc:  # pragma: no cover - plotting backend dependent
                print(f"WARNING: presentation result refresh failed: {exc}")
                return None

    refresh_presentation_results()

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
    pair_cfgs = [_pair_live_config(cfg, index, args.parallel_pairs)
                 for index in range(args.parallel_pairs)]
    training_pair_for, pair_training_methods = _balanced_training_pair_assignment(
        args.pipelines, args.parallel_pairs, args.training_replicate)
    ppo = dict(config.get("ppo") or {})
    ppo["fov_risk_lambda"] = float(
        (config.get("fov_risk") or {}).get("lambda_fov", 0.1))
    ppo_runtime_overrides = {}
    if args.robust_adaptive_reward:
        ppo_runtime_overrides = {
            "learning_rate": min(float(ppo.get("learning_rate", 5e-5)), 5e-5),
            "minimum_learning_rate": min(
                float(ppo.get("minimum_learning_rate", 1e-5)), 1e-5),
            "epochs": min(int(ppo.get("epochs", 4)), 4),
            "target_kl": max(float(ppo.get("target_kl", .025)), .025),
            "learning_rate_recovery_factor": 1.10,
            "learning_rate_recovery_kl_fraction": .50,
        }
        ppo.update(ppo_runtime_overrides)
    training_contract_id = (
        "robust_ppo_anchor_lr_recovery_v1"
        if args.robust_adaptive_reward else None)
    manifest["ppo_runtime_overrides"] = ppo_runtime_overrides
    manifest["ppo_training_contract_id"] = training_contract_id
    manifest["parallel_execution"]["training_pair_assignment"] = {
        name: int(index) for name, index in training_pair_for.items()}
    manifest["parallel_execution"]["assignment_rule"] = (
        "cyclic Latin-square counterbalance by training_replicate; final "
        "evaluation crosses every method over every physical pair")
    _write_json(manifest_path, manifest)
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
        "max_geometric_fov_loss_fraction": float(curriculum_raw.get(
            "max_geometric_fov_loss_fraction", 0.50)),
    }

    rviz = (RvizPublisher.create(
        cfg, pair_methods=(args.pipelines if args.parallel_pairs > 1 else None))
        if not args.no_rviz else None)
    monitor = BenchmarkMonitor(STORE, rviz=rviz)
    monitor.configure(
        methods=args.pipelines, mode=args.mode, config_hash=config_hash,
        training_total=(train_count * len(args.pipelines) + selected_warmup),
        evaluation_total=len(plan),
        reward_design_id=getattr(
            fov_risk_model if needs_fov_risk else potential, "design_id", None),
        reward_design_sha256=getattr(
            fov_risk_model if needs_fov_risk else potential, "sha256", None),
        fov_risk_design_id=getattr(fov_risk_model, "design_id", None),
        algorithm_pipeline=algorithm_pipeline_contract(
            lambda_fov=float((config.get("fov_risk") or {}).get("lambda_fov", 0.1)),
            horizon_seconds=float((config.get("fov_risk") or {}).get(
                "prediction_horizon_seconds", 1.0)),
            control_hz=1.0 / float((config.get("control") or {}).get(
                "dt_seconds", 0.1)),
            hidden_dim=int((config.get("fov_risk_design") or {}).get(
                "hidden_dim", 24)),
            camera=(system.get("vision") or {}).get("camera"),
        ) if needs_fov_risk else None,
        mdp_contract=mdp_contract(
            control=config.get("control") or {},
            reward=config.get("reward") or {},
            lambda_fov=float((config.get("fov_risk") or {}).get("lambda_fov", 0.1)),
            horizon_seconds=float((config.get("fov_risk") or {}).get(
                "prediction_horizon_seconds", 1.0)),
            pad_speed_range=tuple(
                (system.get("pad") or {}).get("speed_range_m_s", ()) or ()) or None,
        ),
        pair_layout=[{
            "index": index,
            "method": name,
            "route_phase_fraction": float(
                ((system.get("parallel") or {}).get(
                    "route_phase_fractions", [0.0, 0.08, 0.16]))[index]),
            "px4_namespace": ("/fmu" if index == 0 else f"/px4_{index}/fmu"),
            "topic_root": (f"/landing_pair_{index}"
                           if args.parallel_pairs > 1 else ""),
            "camera_topic": (
                f"/landing_pair_{index}/uav/perception/landing_camera/annotated"
                if args.parallel_pairs > 1 else
                "/landing_uav0/perception/landing_camera/annotated"),
            "rviz_namespace": (
                f"{str(cfg.viz.rviz.namespace).rstrip('/')}/pair_{index}"
                if args.parallel_pairs > 1 else str(cfg.viz.rviz.namespace)),
            "gateway_port": int(pair_cfgs[index].external.gateway_port),
            "learner_port": int(pair_cfgs[index].external.local_port),
        } for index, name in enumerate(
            pair_training_methods if args.parallel_pairs > 1 else args.pipelines[:1])])
    # Dashboard graph inference is method-scoped. A single global model lets
    # fixed-reward workers overwrite the proposed arm and can even apply an
    # adaptive graph model to an incompatible baseline schema.
    for name in args.pipelines:
        spec = get_pipeline(name)
        reward_model = (adaptive_weights if spec.use_adaptive_reward_weights
                        else potential if spec.use_direct_rgat_potential else None)
        if reward_model is not None:
            monitor.set_potential(name, reward_model)
    dashboard = Dashboard(cfg, STORE).start()
    rviz_process, rviz_log = _start_rviz(
        cfg.viz.rviz.enabled and not args.no_rviz and not args.headless,
        parallel_pairs=args.parallel_pairs)
    owned = None
    training_executor = None
    training_futures = {}
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
                               else (600.0 if args.headless else 1200.0)),
                parallel_pairs=args.parallel_pairs)
            owned.start()
            stack_module.current(owned)
        with ExitStack() as camera_stack:
            cameras = [camera_stack.enter_context(RosGrayscaleSource(
                topic=("/landing_uav0/perception/landing_camera/image_raw"
                       if args.parallel_pairs == 1 else
                       f"/landing_pair_{index}/uav/perception/landing_camera/image_raw"),
                # Training-label-only: projects the known pad landmarks for
                # keypoint supervision. Never reaches the actor observation.
                truth_pose_topic=(
                    "/landing_uav0/perception/pad_relative_truth_pose"
                    if args.parallel_pairs == 1 else
                    f"/landing_pair_{index}/uav/perception/pad_relative_truth_pose"),
                node_name=f"shin2026_actor_camera_{index}"))
                for index in range(args.parallel_pairs)]
            camera = cameras[0]
            monitor.stage("keypoint validation", "live Isaac camera · held-out labels")
            calibration_system = _calibration_system_for_pair(
                system, pair_index=0, pair_count=args.parallel_pairs)
            keypoint_pretraining = calibrate_keypoint_encoder(
                args.results_dir / "models/shared/keypoint_encoder.pt",
                keypoint_pretraining, camera.labelled,
                system=calibration_system,
                experiment=config,
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
            monitor_lock = threading.RLock()
            gpu_update_lock = threading.RLock() if args.parallel_pairs > 1 else None
            worker_monitors = [
                _LockedMonitor(
                    monitor, monitor_lock,
                    method=(pair_training_methods[index]
                            if args.parallel_pairs > 1 else ""),
                    pair_index=index, pair_count=args.parallel_pairs)
                for index in range(args.parallel_pairs)]
            demonstrations = _prepare_fast_demonstrations(
                cfg=cfg, camera=camera, config=config,
                config_hash=config_hash,
                keypoint_pretraining=keypoint_pretraining,
                results_dir=args.results_dir, device=args.device,
                model_seed=model_seed,
                monitor=(worker_monitors[0]
                         if args.parallel_pairs > 1 else monitor))
            cloning_metrics = {}
            if demonstrations is not None:
                manifest["behavior_cloning_demonstrations"] = {
                    key: demonstrations[key] for key in (
                        "teacher", "successful_episodes", "attempted_seeds",
                        "environment_steps", "transitions")}
                _write_json(manifest_path, manifest)
            models = {}
            histories = {}

            def initialize_pipeline_model(name):
                torch.manual_seed(model_seed)
                model = _build_model(
                    config, args.device, keypoint_pretraining, pipeline=name)
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
                return model

            def train_pipeline(name, *, primary=True, pair_index=0, model=None):
                local_monitor = worker_monitors[pair_index]
                local_monitor.stage(
                    "parallel training" if args.parallel_pairs > 1 else "training",
                    f"pair {pair_index} · recurrent PPO · {name}")
                if model is None:
                    model = initialize_pipeline_model(name)
                target_dir = (args.results_dir / f"models/{name}" if primary else
                              args.results_dir / "models/reward_design_source")
                cloning = dict((seminar_fast.get("behavior_cloning") or {}))
                anchor = deepcopy(dict(cloning.get("ppo_anchor") or {}))
                if args.robust_adaptive_reward and anchor.get("enabled", False):
                    anchor["until_policy_episode"] = max(
                        int(anchor.get("until_policy_episode", 0)),
                        max(1, int(train_count) - 4))
                    anchor["minimum_learning_rate"] = min(
                        float(anchor.get("minimum_learning_rate", 5e-6)), 5e-6)
                warmup_count = (int(ppo["perception_warmup_episodes"])
                                if get_pipeline(name).state_estimation_enabled else 0)
                training_seeds = controlled_training_seeds(
                    training_seed0, train_count, warmup_episodes=warmup_count,
                    warmup_seed0=warmup_seed0)
                spec = get_pipeline(name)
                reward_design = (adaptive_weights
                                 if spec.use_adaptive_reward_weights else
                                 potential if spec.use_direct_rgat_potential else
                                 fov_risk_model if spec.fov_risk_reward_enabled else None)
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
                        pair_cfgs[pair_index], cameras[pair_index],
                        horizon_steps=int(pair_cfgs[pair_index].sim.max_steps)),
                    model, name,
                    training_seeds,
                    target_dir, config_hash=config_hash,
                    potential=reward_design,
                    ppo=ppo, curriculum_config=curriculum,
                    monitor=local_monitor if primary else None,
                    restart_incompatible=True,
                    demonstration_dataset=(
                        None if demonstrations is None else
                        demonstrations["dataset"]),
                    demonstration_anchor=(
                        {} if demonstrations is None else
                        anchor),
                    optimizer_lock=gpu_update_lock,
                    training_contract_id=training_contract_id)
                if primary:
                    for row in history:
                        row["training_replicate"] = args.training_replicate
                    with monitor_lock:
                        models[name] = model
                        histories[name] = history
                    _write_csv(args.results_dir / f"training/{name}.csv", history)
                    refresh_presentation_results()
                best_path = target_dir / f"{name}.best.pt"
                return (model, history, best_path if best_path.is_file() else
                        target_dir / f"{name}.pt")

            # 고정 보상 arm은 R-GAT artifact에 의존하지 않는다.
            # 따라서 병렬 실험에서는 이 두 PPO를 즉시 시작하고,
            # 남은 physical pair에서 reward-design rollout을 동시에 수집한다.
            independent_pipelines = [
                name for name in args.pipelines
                if not (get_pipeline(name).use_direct_rgat_potential
                        or get_pipeline(name).use_adaptive_reward_weights
                        or get_pipeline(name).fov_risk_reward_enabled)]
            if args.parallel_pairs > 1:
                # Build every trainable actor and frozen behavior source before
                # worker threads start. ``torch.manual_seed`` is process-global;
                # model construction during live PPO would otherwise perturb a
                # worker's exploration stream according to thread timing.
                source_names = []
                if needs_potential:
                    preferred = str((config.get("rgat_design") or {}).get(
                        "source_pipeline", "no_se"))
                    source_names.append(
                        preferred if preferred in available_pipeline_ids()
                        else "no_se")
                if needs_adaptive:
                    source_names.append(str(
                        (config.get("adaptive_reward_design") or {}).get(
                            "source_pipeline", "no_se_fixed")))
                if needs_fov_risk:
                    source_names.append(str(
                        (config.get("fov_risk_design") or {}).get(
                            "source_pipeline", "shin_se_fixed")))
                prepared_names = list(dict.fromkeys(
                    [*args.pipelines, *source_names]))
                prepared_parallel_models = {
                    name: initialize_pipeline_model(name)
                    for name in prepared_names}
                frozen_behavior_models = {
                    name: deepcopy(prepared_parallel_models[name])
                    for name in source_names}
                for source_model in frozen_behavior_models.values():
                    for module in source_model.modules():
                        flatten = getattr(module, "flatten_parameters", None)
                        if callable(flatten):
                            flatten()
                training_executor = ThreadPoolExecutor(
                    max_workers=args.parallel_pairs,
                    thread_name_prefix="landing-pair")
                training_futures = {
                    name: training_executor.submit(
                        train_pipeline, name,
                        pair_index=training_pair_for[name],
                        model=prepared_parallel_models[name])
                    for name in independent_pipelines}
                occupied_training_pairs = {
                    training_pair_for[name] for name in independent_pipelines}
                reward_design_pair_indices = [
                    index for index in range(args.parallel_pairs)
                    if index not in occupied_training_pairs]
                if not reward_design_pair_indices:
                    raise RuntimeError(
                        "parallel reward design needs one unoccupied physical pair")
                manifest["parallel_execution"].update({
                    "early_independent_pipelines": independent_pipelines,
                    "reward_design_pair_indices": reward_design_pair_indices,
                })
                _write_json(manifest_path, manifest)
                refresh_presentation_results()
            else:
                reward_design_pair_indices = [0]
                for name in args.pipelines:
                    spec = get_pipeline(name)
                    if not (spec.use_direct_rgat_potential
                            or spec.use_adaptive_reward_weights
                            or spec.fov_risk_reward_enabled):
                        train_pipeline(name)

            fov_design_episodes = 0
            fov_design_steps = 0
            if needs_fov_risk and fov_risk_model is None:
                source_name = str((config.get("fov_risk_design") or {}).get(
                    "source_pipeline", "shin_se_fixed"))
                source_checkpoint = None
                if source_name in models:
                    source_model = models[source_name]
                    source_dir = args.results_dir / "models" / source_name
                    source_checkpoint = (
                        source_dir / f"{source_name}.best.pt"
                        if (source_dir / f"{source_name}.best.pt").is_file()
                        else source_dir / f"{source_name}.pt")
                elif args.parallel_pairs > 1 and source_name in frozen_behavior_models:
                    source_model = frozen_behavior_models[source_name]
                    source_checkpoint = _load_reward_design_source_checkpoint(
                        args.results_dir, source_name, source_model,
                        config_hash=config_hash)
                    print(
                        f"Using an independently frozen {source_name} policy for "
                        "offline FOV-risk data while baseline PPO trains.")
                else:
                    source_model = initialize_pipeline_model(source_name)
                fov_dataset, fov_manifest, _, fov_design_steps = (
                    _collect_fov_risk_data(
                        cfg=pair_cfgs[reward_design_pair_indices[0]],
                        camera=cameras[reward_design_pair_indices[0]],
                        model=source_model, config=config,
                        config_hash=config_hash, checkpoint_path=source_checkpoint,
                        results_dir=args.results_dir, mode=args.mode,
                        monitor=worker_monitors[reward_design_pair_indices[0]],
                        episodes_override=args.rgat_data_episodes,
                        max_episodes_override=args.rgat_max_data_episodes,
                        source_pipeline=source_name))
                fov_design_episodes = int(fov_manifest["episodes"])
                settings = dict(config.get("fov_risk_design") or {})
                settings["epochs"] = int(
                    args.rgat_epochs or settings.get(
                        f"epochs_{args.mode}", 10 if args.mode == "quick" else 80))
                settings["device"] = args.device
                monitor.stage("R-GAT training",
                              "future FOV-unavailability scalar readout")

                def _fov_epoch(row, _total=int(settings["epochs"])):
                    monitor.fov_training(row)
                    if int(row["epoch"]) % 10 == 0 or int(row["epoch"]) == _total:
                        print(f"FOV readout epoch {int(row['epoch'])}/{_total} "
                              f"val={float(row['validation_loss']):.5f} "
                              f"best={float(row['best_validation_loss']):.5f}")
                    refresh_presentation_results()

                fov_risk_model, fov_metadata = prepare_fov_risk_artifact(
                    args.fov_risk_model, fov_dataset,
                    dataset_manifest=fov_manifest, config_hash=config_hash,
                    seed=model_seed, settings=settings, progress=_fov_epoch)
                monitor.fov_model(design_id=fov_risk_model.design_id,
                                  metadata=fov_metadata)
                manifest.update({
                    "fov_risk_design_id": fov_risk_model.design_id,
                    "fov_risk_design_sha256": fov_risk_model.sha256,
                    "fov_risk_dataset": fov_manifest,
                    "fov_risk_model": fov_metadata,
                })
                _write_json(manifest_path, manifest)
                STORE.set(reward_design_id=fov_risk_model.design_id,
                          reward_design_sha256=fov_risk_model.sha256)
            elif needs_fov_risk:
                fov_manifest = dict(fov_risk_model.metadata)
                fov_design_episodes = int((fov_manifest.get(
                    "dataset_manifest") or {}).get("episodes", 0))
                # A resumed run skips collection and training entirely; without
                # this the FOV panel would stay blank for the whole run even
                # though a validated readout is driving the proposed reward.
                monitor.fov_model(design_id=fov_risk_model.design_id,
                                  metadata=fov_manifest)

            if needs_potential and potential is None:
                preferred_source = str((config.get("rgat_design") or {}).get(
                    "source_pipeline", "no_se"))
                source_name = (preferred_source if preferred_source in available_pipeline_ids()
                               else "no_se")
                if source_name in models:
                    source_model = models[source_name]
                    best_source = (args.results_dir
                                   / f"models/{source_name}/{source_name}.best.pt")
                    source_checkpoint = (best_source if best_source.is_file() else
                        args.results_dir / f"models/{source_name}/{source_name}.pt")
                    source_training_episodes = 0
                elif source_name in training_futures:
                    # Reuse a compatible completed checkpoint when available;
                    # otherwise freeze this separately initialized/BC-warmed
                    # source while the comparison PPO continues independently.
                    # The reward-design rollout already mixes visual teachers,
                    # noise and adverse contacts, so waiting for all PPO episodes
                    # would add wall time without making the data more empirical.
                    source_model = frozen_behavior_models[source_name]
                    source_checkpoint = _load_reward_design_source_checkpoint(
                        args.results_dir, source_name, source_model,
                        config_hash=config_hash)
                    if source_checkpoint is None:
                        print(
                            f"Using a separately frozen BC-warmed {source_name} "
                            "policy for reward data while its PPO arm trains.")
                        source_training_episodes = 0
                    else:
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
                        cfg=pair_cfgs[reward_design_pair_indices[0]],
                        camera=cameras[reward_design_pair_indices[0]],
                        model=source_model, config=config,
                        config_hash=config_hash, checkpoint_path=source_checkpoint,
                        results_dir=args.results_dir, mode=args.mode,
                        monitor=worker_monitors[reward_design_pair_indices[0]],
                        episodes_override=args.rgat_data_episodes,
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
                for name in args.pipelines:
                    if get_pipeline(name).use_direct_rgat_potential:
                        monitor.set_potential(name, potential)
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
                for name in args.pipelines:
                    if get_pipeline(name).use_direct_rgat_potential:
                        monitor.set_potential(name, potential)
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
                _archive_rejected_adaptive_artifact(
                    args.adaptive_reward_design, adaptive_rejected_design_id)
                source_name = str((config.get("adaptive_reward_design") or {}).get(
                    "source_pipeline", "no_se_fixed"))
                adaptive_source_checkpoint = None
                if source_name in models:
                    adaptive_source_model = models[source_name]
                    source_dir = args.results_dir / "models" / source_name
                    adaptive_source_checkpoint = (
                        source_dir / f"{source_name}.best.pt"
                        if (source_dir / f"{source_name}.best.pt").is_file()
                        else source_dir / f"{source_name}.pt")
                elif source_name in args.pipelines:
                    # A legacy policy remains valid as a source of empirical
                    # reward-design outcomes even when the new PPO training
                    # contract requires all publication arms to be retrained.
                    # Do not register it in ``models``: the parallel stage
                    # below will independently rebuild the comparison arm.
                    adaptive_source_model = (
                        frozen_behavior_models[source_name]
                        if args.parallel_pairs > 1 else
                        initialize_pipeline_model(source_name))
                    adaptive_source_checkpoint = (
                        _load_reward_design_source_checkpoint(
                            args.results_dir, source_name,
                            adaptive_source_model, config_hash=config_hash))
                    if adaptive_source_checkpoint is None:
                        if source_name in training_futures:
                            print(
                                f"Using a separately frozen BC-warmed "
                                f"{source_name} policy for adaptive reward data "
                                "while its PPO arm trains.")
                        else:
                            adaptive_source_model, _, adaptive_source_checkpoint = (
                                train_pipeline(
                                    source_name, primary=True,
                                    pair_index=training_pair_for[source_name],
                                    model=adaptive_source_model))
                    else:
                        print(
                            "Using the prior compatible fixed policy only as "
                            "the empirical adaptive reward-data source: "
                            f"{adaptive_source_checkpoint}")
                else:
                    adaptive_source_model, _, adaptive_source_checkpoint = train_pipeline(
                        source_name, primary=False)
                adaptive_dataset, adaptive_manifest, _, adaptive_design_steps = (
                    _collect_adaptive_data(
                        cfg=cfg, camera=camera, model=adaptive_source_model,
                        source_pipeline=source_name, config=config,
                        config_hash=config_hash, results_dir=args.results_dir,
                        mode=args.mode, monitor=(
                            worker_monitors[0]
                            if args.parallel_pairs > 1 else monitor),
                        episodes_override=args.rgat_data_episodes,
                        max_episodes_override=args.rgat_max_data_episodes,
                        source_checkpoint_path=adaptive_source_checkpoint,
                        parallel_contexts=([{
                            "cfg": pair_cfgs[index],
                            "camera": cameras[index],
                            "monitor": worker_monitors[index],
                        } for index in reward_design_pair_indices]
                            if args.parallel_pairs > 1 else None),
                        minimum_unsafe_failures_override=(
                            (adaptive_settings.get("quality_gate") or {}).get(
                                "minimum_unsafe_failure_episodes"))))
                adaptive_design_episodes = int(adaptive_manifest["episodes"])
                expected_architectures = {
                    get_pipeline(name).adaptive_reward_architecture
                    for name in args.pipelines
                    if get_pipeline(name).use_adaptive_reward_weights}
                if len(expected_architectures) != 1:
                    raise RuntimeError(
                        "structural ablations require separate runs/artifacts per architecture")
                adaptive_settings["architecture"] = next(iter(expected_architectures))
                monitor.stage("adaptive R-GAT training", "five constrained reward weights")
                adaptive_minimum = int(
                    args.rgat_data_episodes or adaptive_settings.get(
                        f"episodes_{args.mode}", 8 if args.mode == "quick" else 40))
                adaptive_maximum = int(
                    args.rgat_max_data_episodes or adaptive_settings.get(
                        f"max_episodes_{args.mode}", max(
                            adaptive_minimum, 3 * adaptive_minimum)))
                while True:
                    try:
                        _, adaptive_metadata = prepare_adaptive_reward_artifact(
                            args.adaptive_reward_design, adaptive_dataset,
                            dataset_manifest=adaptive_manifest,
                            config_hash=config_hash, settings=adaptive_settings,
                            seed=model_seed)
                        break
                    except RuntimeError as exc:
                        if "quality gate rejected" not in str(exc):
                            raise
                        current_episodes = len(np.unique(
                            adaptive_dataset["episode_id"]))
                        if current_episodes >= adaptive_maximum:
                            raise RuntimeError(
                                f"{exc}; exhausted the {adaptive_maximum}-episode "
                                "real-rollout cap") from exc
                        next_minimum = min(
                            adaptive_maximum,
                            current_episodes + max(4, adaptive_minimum // 2))
                        print(
                            f"Adaptive R-GAT did not pass its quality gate after "
                            f"{current_episodes} episodes ({exc}). Extending the "
                            f"real dataset to at least {next_minimum} episodes.")
                        adaptive_dataset, adaptive_manifest, _, adaptive_design_steps = (
                            _collect_adaptive_data(
                                cfg=cfg, camera=camera,
                                model=adaptive_source_model,
                                source_pipeline=source_name, config=config,
                                config_hash=config_hash,
                                results_dir=args.results_dir,
                                mode=args.mode, monitor=(
                                    worker_monitors[0]
                                    if args.parallel_pairs > 1 else monitor),
                                episodes_override=next_minimum,
                                max_episodes_override=adaptive_maximum,
                                source_checkpoint_path=(
                                    adaptive_source_checkpoint),
                                parallel_contexts=([{
                                    "cfg": pair_cfgs[index],
                                    "camera": cameras[index],
                                    "monitor": worker_monitors[index],
                                } for index in reward_design_pair_indices]
                                    if args.parallel_pairs > 1 else None),
                                minimum_unsafe_failures_override=(
                                    (adaptive_settings.get("quality_gate") or {}).get(
                                        "minimum_unsafe_failure_episodes"))))
                        adaptive_design_episodes = int(
                            adaptive_manifest["episodes"])
                        monitor.stage(
                            "adaptive R-GAT retraining",
                            f"quality-gated real data · {adaptive_design_episodes} episodes")
                adaptive_weights = FrozenAdaptiveRewardWeights(
                    args.adaptive_reward_design,
                    expected_config_hash=config_hash)
                for name in args.pipelines:
                    if get_pipeline(name).use_adaptive_reward_weights:
                        monitor.set_potential(name, adaptive_weights)
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
            if args.parallel_pairs > 1:
                monitor.stage(
                    "parallel training preparation",
                    "R-GAT frozen · starting dependent PPO on its dedicated pair")
                dependent_pipelines = [
                    name for name in args.pipelines
                    if name not in training_futures]
                try:
                    for name in dependent_pipelines:
                        training_futures[name] = training_executor.submit(
                            train_pipeline, name,
                            pair_index=training_pair_for[name],
                            model=prepared_parallel_models[name])
                    for future in as_completed(list(training_futures.values())):
                        future.result()
                except BaseException:
                    _abort_parallel_workers(
                        training_executor, list(training_futures.values()), owned)
                    training_executor = None
                    raise
                else:
                    training_executor.shutdown(wait=True)
                    training_executor = None
            else:
                for name in args.pipelines:
                    spec = get_pipeline(name)
                    if (spec.use_direct_rgat_potential
                            or spec.use_adaptive_reward_weights
                            or spec.fov_risk_reward_enabled):
                        train_pipeline(name)

            training_records = [row for name in args.pipelines
                                for row in histories.get(name, [])]

            # A training-best checkpoint is based on one sampled rollout. It
            # can therefore encode a lucky exploration action that disappears
            # when evaluation deploys the actor mean. Compare training-best
            # and training-latest on disjoint deterministic seeds before any
            # reported evaluation and persist every selection flight.
            selection_path = (
                args.results_dir / "evaluation/checkpoint_selection.csv")
            selection_rows = _read_csv(selection_path)
            selection_lock = threading.RLock()
            selection_seed0 = int(seed_cfg.get(
                "checkpoint_validation_start",
                int(seed_cfg.get("evaluation_start", 5000)) - 1000))
            selection_scenarios = tuple(
                seminar_fast.get("evaluation_scenarios") or evaluation_cfg)
            selection_scenarios = selection_scenarios[:3]
            selected_checkpoints = {}

            def reward_design_for(name):
                spec = get_pipeline(name)
                return (adaptive_weights if spec.use_adaptive_reward_weights
                        else potential if spec.use_direct_rgat_potential
                        else fov_risk_model if spec.fov_risk_reward_enabled else None)

            def select_pipeline_checkpoint(index, name):
                model = models[name]
                candidates = _checkpoint_candidates(args.results_dir, name)
                validation_plan = _checkpoint_validation_plan(
                    name, selection_scenarios, seed0=selection_seed0)
                local_monitor = _LockedMonitor(
                    monitor, monitor_lock, method=name, pair_index=index,
                    pair_count=args.parallel_pairs)
                summaries = []
                with LiveShinEnvironment(
                        pair_cfgs[index], cameras[index],
                        horizon_steps=int(pair_cfgs[index].sim.max_steps)) as environment:
                    for candidate in candidates:
                        payload = torch.load(
                            candidate["path"], map_location=model.device,
                            weights_only=False)
                        model.load_state_dict(payload["model"])
                        model.eval()
                        candidate_metrics = []
                        for item in validation_plan:
                            key = (name, candidate["sha256"],
                                   item["scenario"], int(item["seed"]))
                            with selection_lock:
                                cached = next((row for row in selection_rows
                                    if (row.get("method"),
                                        row.get("checkpoint_candidate_sha256"),
                                        row.get("scenario"), int(row.get("seed", -1)))
                                    == key), None)
                            if cached is None:
                                local_monitor.stage(
                                    "deterministic checkpoint validation",
                                    f"pair {index} · {name} · {candidate['kind']}")
                                _, metric = collect_episode_resilient(
                                    environment, model, name, int(item["seed"]),
                                    curriculum=float(seminar_fast.get(
                                        "evaluation_curriculum", 1.0)),
                                    potential=reward_design_for(name),
                                    fov_risk_lambda=float(ppo.get(
                                        "fov_risk_lambda", 0.1)),
                                    deterministic=True,
                                    gamma=float(ppo.get("gamma", .99)),
                                    shaping_lambda=float(ppo.get(
                                        "shaping_lambda", 1.0)),
                                    reward_normalizer=runtime_reward_normalizer,
                                    scenario=item["scenario"], monitor=local_monitor,
                                    phase="checkpoint validation")
                                metric.update({
                                    "method": name, "pipeline": name,
                                    "scenario": item["scenario"],
                                    "training_replicate": args.training_replicate,
                                    "physical_pair_index": index,
                                    "checkpoint_candidate_kind": candidate["kind"],
                                    "checkpoint_candidate_episode": candidate["episode"],
                                    "checkpoint_candidate_sha256": candidate["sha256"],
                                    "checkpoint_validation_protocol": (
                                        "held_out_deterministic_multi_seed_v1"),
                                })
                                with selection_lock:
                                    selection_rows.append(metric)
                                    _write_csv(selection_path, selection_rows)
                                cached = metric
                            candidate_metrics.append(cached)
                        summary = aggregate_deployment_validation(candidate_metrics)
                        summaries.append((deployment_validation_key(summary),
                                          candidate["episode"], candidate,
                                          payload, summary))
                # Prefer the later snapshot only when every held-out safety and
                # physical metric ties exactly.
                _, _, candidate, payload, summary = max(
                    summaries, key=lambda item: (item[0], item[1]))
                selected_path = (args.results_dir / "models" / name
                                 / f"{name}.selected.pt")
                _atomic_save_selected_checkpoint(
                    selected_path, payload, candidate=candidate, summary=summary,
                    validation_seeds=[item["seed"] for item in validation_plan])
                model.load_state_dict(payload["model"])
                model.eval()
                model._selected_checkpoint_episode = int(candidate["episode"])
                model._selected_checkpoint_score = float(
                    summary["mean_physical_score"])
                model._selected_checkpoint_kind = str(candidate["kind"])
                model._selected_checkpoint_sha256 = str(candidate["sha256"])
                with selection_lock:
                    selected_checkpoints[name] = {
                        **candidate, "path": str(selected_path), "summary": summary}
                print(
                    f"Selected {name} {candidate['kind']} episode "
                    f"{candidate['episode']} by {int(summary['successes'])}/"
                    f"{int(summary['episodes'])} held-out deterministic landings.")

            if args.parallel_pairs > 1:
                executor = ThreadPoolExecutor(
                    max_workers=args.parallel_pairs,
                    thread_name_prefix="checkpoint-selection")
                futures = []
                try:
                    futures = [executor.submit(
                        select_pipeline_checkpoint, training_pair_for[name], name)
                        for name in args.pipelines]
                    for future in as_completed(futures):
                        future.result()
                except BaseException:
                    _abort_parallel_workers(executor, futures, owned)
                    raise
                else:
                    executor.shutdown(wait=True)
            else:
                for name in args.pipelines:
                    select_pipeline_checkpoint(0, name)
            manifest["checkpoint_selection_rule"] = (
                "best and latest training snapshots compared on three held-out "
                "deterministic scenario seeds; safe success count is primary")
            manifest["selected_checkpoints"] = selected_checkpoints
            _write_json(manifest_path, manifest)

            evaluation_rows = _read_csv(existing_eval)
            for row in evaluation_rows:
                row.setdefault("training_replicate", args.training_replicate)
            valid_evaluation_rows, superseded_rows = [], []
            for row in evaluation_rows:
                name = row.get("pipeline", row.get("method"))
                selected = selected_checkpoints.get(name)
                recorded_digest = str(row.get("selected_checkpoint_sha256", ""))
                recorded_episode = int(float(row.get(
                    "selected_checkpoint_episode", 0)))
                crossover_missing = (
                    args.parallel_pairs > 1 and not str(
                        row.get("physical_pair_index", "")).strip())
                mismatch = selected is None or crossover_missing or (
                    bool(recorded_digest)
                    and recorded_digest != selected["sha256"]) or (
                    not recorded_digest
                    and recorded_episode != int(selected["episode"]))
                if mismatch:
                    row["superseded_reason"] = (
                        "pre_crossover_evaluation" if crossover_missing
                        else "deployment_checkpoint_changed")
                    superseded_rows.append(row)
                    continue
                row["selected_checkpoint_kind"] = selected["kind"]
                row["selected_checkpoint_sha256"] = selected["sha256"]
                row["selected_checkpoint_score"] = selected[
                    "summary"]["mean_physical_score"]
                valid_evaluation_rows.append(row)
            if superseded_rows:
                superseded_path = (
                    args.results_dir / "evaluation/per_episode.superseded.csv")
                _write_csv(superseded_path,
                           _read_csv(superseded_path) + superseded_rows)
                print(f"Archived {len(superseded_rows)} evaluation rows collected "
                      "before deterministic selection/crossover balancing.")
            evaluation_rows = valid_evaluation_rows
            _write_csv(existing_eval, evaluation_rows)
            completed = {(row["pipeline"], row["scenario"], int(row["seed"]))
                         for row in evaluation_rows}
            monitor.restore_evaluation(evaluation_rows)
            evaluation_lock = threading.RLock()

            def evaluate_pair(index, tasks):
                new_rows = []
                with LiveShinEnvironment(
                        pair_cfgs[index], cameras[index],
                        horizon_steps=int(pair_cfgs[index].sim.max_steps)) as environment:
                    for item in tasks:
                        name = item["method"]
                        key = (name, item["scenario"], int(item["seed"]))
                        if key in completed:
                            continue
                        model = models[name]
                        local_monitor = _LockedMonitor(
                            monitor, monitor_lock, method=name, pair_index=index,
                            pair_count=args.parallel_pairs)
                        local_monitor.stage(
                            "parallel crossover evaluation"
                            if args.parallel_pairs > 1 else "paired evaluation",
                            f"pair {index} · {name}")
                        _, metric = collect_episode_resilient(
                            environment, model, name, int(item["seed"]),
                            curriculum=float(seminar_fast.get(
                                "evaluation_curriculum", 1.0)),
                            potential=reward_design_for(name),
                            fov_risk_lambda=float(ppo.get(
                                "fov_risk_lambda", 0.1)),
                            deterministic=True, gamma=float(ppo.get("gamma", .99)),
                            shaping_lambda=float(ppo.get("shaping_lambda", 1.0)),
                            reward_normalizer=runtime_reward_normalizer,
                            scenario=item["scenario"], monitor=local_monitor,
                            phase="evaluation")
                        metric.update({"method": name, "pipeline": name,
                                       "training_replicate": args.training_replicate,
                                       "scenario": item["scenario"],
                                       "physical_pair_index": index,
                                       "route_phase_fraction": float(
                                           ((system.get("parallel") or {}).get(
                                               "route_phase_fractions",
                                               [0.0, 0.08, 0.16]))[index])})
                        new_rows.append(metric)
                        local_monitor.evaluation_update(name, metric)
                        # Evaluation is real-time flight and can take long
                        # enough to be interrupted. Commit every completed
                        # pair/seed immediately instead of waiting for all
                        # worker plans to finish.
                        with evaluation_lock:
                            evaluation_rows.append(metric)
                            _write_csv(existing_eval, evaluation_rows)
                return new_rows

            crossover_tasks = _crossover_evaluation_tasks(
                plan, args.pipelines, args.parallel_pairs)
            _write_csv(
                args.results_dir / "evaluation/crossover_plan.csv",
                [row for rows in crossover_tasks for row in rows])
            if args.parallel_pairs > 1:
                executor = ThreadPoolExecutor(
                    max_workers=args.parallel_pairs,
                    thread_name_prefix="landing-eval")
                futures = []
                try:
                    futures = [executor.submit(evaluate_pair, index, tasks)
                               for index, tasks in enumerate(crossover_tasks)]
                    for future in as_completed(futures):
                        future.result()
                except BaseException:
                    _abort_parallel_workers(executor, futures, owned)
                    raise
                else:
                    executor.shutdown(wait=True)
            else:
                evaluate_pair(0, crossover_tasks[0])
            _write_csv(existing_eval, evaluation_rows)

            semantic_cost = {
                "episodes": design_episodes + source_training_episodes,
                "steps": design_steps + source_training_steps,
            }
            adaptive_cost = {
                "episodes": adaptive_design_episodes,
                "steps": adaptive_design_steps,
            }
            fov_cost = {"episodes": fov_design_episodes,
                        "steps": fov_design_steps}
            reward_design_costs = {}
            for pipeline_name in args.pipelines:
                pipeline_spec = get_pipeline(pipeline_name)
                episodes_cost = steps_cost = 0
                if pipeline_spec.use_direct_rgat_potential:
                    episodes_cost += semantic_cost["episodes"]
                    steps_cost += semantic_cost["steps"]
                if pipeline_spec.use_adaptive_reward_weights:
                    episodes_cost += adaptive_cost["episodes"]
                    steps_cost += adaptive_cost["steps"]
                if pipeline_spec.fov_risk_reward_enabled:
                    episodes_cost += fov_cost["episodes"]
                    steps_cost += fov_cost["steps"]
                if episodes_cost or steps_cost:
                    reward_design_costs[pipeline_name] = {
                        "episodes": episodes_cost, "steps": steps_cost}
            report_writer = (write_two_pipeline_outputs if primary_only
                             else write_three_pipeline_outputs)
            reports = report_writer(
                evaluation_rows, training_records, args.results_dir,
                reward_design_episodes=0, reward_design_steps=0,
                reward_design_costs=reward_design_costs,
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
            if needs_adaptive:
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
            total_design_episodes = (reward_design_cost + adaptive_design_episodes
                                     + fov_design_episodes)
            total_design_steps = (reward_design_step_cost + adaptive_design_steps
                                  + fov_design_steps)
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
                "reward_design_cost_by_pipeline": reward_design_costs,
                "reports": reports,
            })
            _write_json(manifest_path, manifest)
            presentation_summary = refresh_presentation_results()
            if presentation_summary is not None:
                manifest["reports"]["presentation"] = {
                    "status": presentation_summary["performance_status"],
                    "figures": presentation_summary["figures"],
                }
                _write_json(manifest_path, manifest)
            if args.stay_open:
                _hold_after_complete(owned, monitor)
    except Exception as exc:
        manifest.update({
            "execution_status": "failed; checkpoints and completed rows preserved",
            "failure": {
                "type": type(exc).__name__,
                "message": str(exc),
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            },
        })
        _write_json(manifest_path, manifest)
        refresh_presentation_results()
        raise
    finally:
        if training_executor is not None:
            _abort_parallel_workers(
                training_executor, list(training_futures.values()), owned)
            training_executor = None
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
    label = "Two-pipeline" if primary_only else "Legacy multi-pipeline"
    print(f"{label} experiment complete: {args.results_dir}")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except KeyboardInterrupt:
        print("Pipeline interrupted by user; checkpoints and completed rows were preserved.")
        exit_code = 130
    raise SystemExit(exit_code)
