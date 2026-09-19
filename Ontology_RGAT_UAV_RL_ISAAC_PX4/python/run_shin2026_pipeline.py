#!/usr/bin/env python3
"""One-command live Shin/OntoReward training, paired evaluation and reports."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "isaac_sim"))

from config_loader import load_config as load_system_config
from ontology_rgat import stack as stack_module
from ontology_rgat.benchmarks.experiment import (METHODS, configuration_hash,
                                                 episodes_per_method,
                                                 load_experiment, paired_seed_plan)
from ontology_rgat.benchmarks.live_env import LiveShinEnvironment
from ontology_rgat.bridge import BridgeError, EntryResetError, PX4Failsafe
from ontology_rgat.cli import ensure_fastdds
from ontology_rgat.config import default_config
from ontology_rgat.evaluation.shin2026 import write_benchmark_outputs
from ontology_rgat.perception import (RosGrayscaleSource,
                                      calibrate_keypoint_encoder,
                                      needs_empirical_calibration,
                                      prepare_keypoint_encoder)
from ontology_rgat.ppo.recurrent import PipelineActorCritic
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
    # The learner does not read Isaac's YAML directly. Copy the scoring/safety
    # contract plus presentation geometry from the authoritative merged live
    # profile so evaluation, deck rendering and the surveyed route agree.
    pad = simulator_config.get("pad") or {}
    benchmark = simulator_config.get("benchmark") or {}
    landing = simulator_config.get("landing") or {}
    cfg.criteria.xy = float(landing.get("success_xy_m", cfg.criteria.xy))
    cfg.criteria.vz = float(landing.get("success_vz_m_s", cfg.criteria.vz))
    cfg.criteria.tilt = math.radians(float(landing.get(
        "success_tilt_deg", math.degrees(cfg.criteria.tilt))))
    cfg.criteria.rate = math.radians(float(landing.get(
        "success_rate_deg_s", math.degrees(cfg.criteria.rate))))
    cfg.criteria.rel_speed_xy = float(landing.get(
        "success_rel_speed_xy_m_s", cfg.criteria.rel_speed_xy))
    cfg.sim.ground_z = float(landing.get("ground_z_m", cfg.sim.ground_z))
    cfg.sim.crash_tilt = math.radians(float(landing.get(
        "crash_tilt_deg", math.degrees(cfg.sim.crash_tilt))))
    cfg.sim.world_xy_limit = float(landing.get(
        "world_xy_limit_m", cfg.sim.world_xy_limit))
    cfg.external.entry_speed_tolerance = float(
        benchmark.get("entry_speed_tolerance_m_s",
                      cfg.external.entry_speed_tolerance))
    cfg.external.entry_settle = float(
        benchmark.get("entry_settle_s", cfg.external.entry_settle))
    cfg.external.entry_arm_grace = float(
        benchmark.get("entry_arm_grace_s", cfg.external.entry_arm_grace))
    # The budget the entry gate is judged against, in simulated PX4 seconds,
    # and the wall-clock hang guard that bounds a stopped simulator.
    cfg.external.entry_sim_budget = float(
        benchmark.get("entry_sim_budget_s", cfg.external.entry_sim_budget))
    cfg.external.entry_timeout = float(
        benchmark.get("entry_timeout_s", cfg.external.entry_timeout))
    cfg.external.entry_view_retries = int(
        benchmark.get("entry_view_retries", cfg.external.entry_view_retries))
    # The entry gate, the geometric FOV metric and the future-FOV-loss labels
    # all project through the camera Isaac actually renders with, so copy that
    # model rather than relying on the defaults.
    camera = dict((simulator_config.get("vision") or {}).get("camera") or {})
    landing_camera = dict(cfg.external.landing_camera)
    for key in ("resolution", "horizontal_fov_deg", "pitch_down_deg",
                "mount_translation_flu_m"):
        if key in camera:
            landing_camera[key] = camera[key]
    cfg.external.landing_camera = landing_camera
    if "entry_view_margin" in benchmark:
        cfg.external.entry_view_margin = float(benchmark["entry_view_margin"])
    cfg.viz.rviz.deck_size_m = list(pad.get("deck_size_m", (1.5, 1.5)))
    cfg.viz.rviz.deck_height_m = float(pad.get("deck_height_m", 0.0))
    # ``null`` clears an inherited key in this config family, and a profile
    # that drives an analytic scenario instead of a surveyed route has no
    # waypoints for RViz to draw.
    cfg.viz.rviz.route_waypoints_enu_m = list(
        pad.get("route_waypoints_enu_m") or ())
    return cfg


def _build_model(config, device, keypoint_pretraining=None, pipeline="shin_se"):
    estimator = config.get("estimator") or {}
    ppo = config.get("ppo") or {}
    torch_device = torch.device(device)
    pretraining_enabled = bool(
        (estimator.get("keypoint_pretraining") or {}).get("enabled", False))
    if pretraining_enabled and keypoint_pretraining is None:
        raise ValueError("enabled keypoint pretraining artifact was not prepared")
    model = PipelineActorCritic(
        image_embedding=int(estimator.get("image_embedding", 512)),
        lstm_hidden=int(estimator.get("lstm_hidden", 512)),
        latent_dim=int(estimator.get("latent_dimension", 256)),
        actor_hidden=int(ppo.get("hidden", 256)),
        critic_hidden=int(ppo.get("hidden", 256)),
        init_log_std=float(ppo.get("init_log_std", -1.5)),
        actor_output_gain=float(ppo.get("actor_output_gain", 0.01)),
        freeze_keypoint=pretraining_enabled,
        relative_state_scale=estimator.get(
            "relative_state_scale", (3.0, 3.0, 8.0, 3.0, 3.0, 2.0)),
        pipeline=pipeline,
    )
    if keypoint_pretraining is not None:
        model.encoder.load_state_dict(keypoint_pretraining["encoder"])
    return model.to(torch_device)


class KeypointCalibrationFlight:
    """Fly the pad-relative viewpoints the frozen encoder is certified on.

    The encoder used to be calibrated on whatever the camera saw while the
    vehicle sat where the stack left it after boot: 48 consecutive frames of
    one hover, whose randomly held-out quarter could only measure
    memorisation. It scored 10 px and 100 % on that split and then reported no
    landmark on 99.7 % of the in-frame steps of the run it was certified for.

    This takes the vehicle to each surveyed viewpoint under PX4's own position
    controller -- the same pad-frame ``goto`` the seeded entry pose is flown
    with -- so the held-out split is over poses the encoder never trained on.
    """

    def __init__(self, cfg, image_source, *, seed, settings=None):
        self.cfg = cfg
        self.image_source = image_source
        self.seed = int(seed)
        settings = dict(settings or {})
        self.travel_timeout_s = float(settings.get(
            "survey_travel_timeout_s", 45.0))
        self.tolerance_m = float(settings.get("survey_tolerance_m", 0.8))
        self.speed_tolerance_m_s = float(settings.get(
            "survey_speed_tolerance_m_s", 0.7))
        self.settle_s = float(settings.get("survey_settle_s", 0.7))
        # The survey is setup, not a measured episode, so it enters at the
        # gentlest seeded initial condition. What the calibration needs from
        # this flight is viewpoints, and a full-difficulty entry only makes
        # the one reset it depends on more likely to need a simulator rebuild.
        self.curriculum = float(settings.get("survey_curriculum", 0.0))
        self.environment = None

    def __enter__(self):
        return self

    def _ensure_airborne(self):
        """Arm and climb, but only once a viewpoint actually has to be flown.

        An accumulation that already covers the survey needs no flight at all,
        and opening one anyway would spend an entry gate -- the least reliable
        part of the stack -- on a calibration that has nothing to collect.
        """
        if self.environment is None:
            self.environment = LiveShinEnvironment(
                self.cfg, self.image_source,
                horizon_steps=int(self.cfg.sim.max_steps))
            # Arming, the offboard pre-stream and the climb are the reset's
            # job, and it owns simulator recovery if the stack has to be
            # rebuilt.
            self.environment.reset(self.seed, curriculum=self.curriculum)
        return self.environment

    def __exit__(self, *_):
        environment, self.environment = self.environment, None
        if environment is None:
            return
        try:
            environment.finish_episode()
        finally:
            environment.close()

    def __call__(self, viewpoint) -> bool:
        bridge = self._ensure_airborne().bridge
        target = np.asarray(viewpoint["position_pad_m"], dtype=float)
        # The hold must outlast the travel *and* the frames captured after
        # arrival: an expired goto makes the gateway command a landing.
        bridge.transact(
            "goto", {"position": target.tolist(),
                     "yaw": float(viewpoint["yaw_rad"]), "frame": "pad",
                     "hold_s": float(self.travel_timeout_s) + 60.0}, ("ack",))
        deadline = time.monotonic() + self.travel_timeout_s
        settled_since = None
        while time.monotonic() < deadline:
            try:
                state = bridge.get_state()
            except (EntryResetError, PX4Failsafe):
                # A latched contact, a disarm or an explicit failsafe is not a
                # transient: the owned-stack recovery has to rebuild it.
                raise
            except BridgeError:
                time.sleep(0.05)
                continue
            here, speed = bridge.entry_state(state)
            if (float(np.linalg.norm(here - target)) <= self.tolerance_m
                    and speed <= self.speed_tolerance_m_s):
                settled_since = settled_since or time.monotonic()
                if time.monotonic() - settled_since >= self.settle_s:
                    return True
            else:
                settled_since = None
            time.sleep(0.05)
        print(f"WARNING: keypoint survey could not reach viewpoint "
              f"{int(viewpoint['index'])} at "
              f"({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f}) m in "
              f"{self.travel_timeout_s:.0f} s; it is left out of the "
              "calibration set.")
        return False


def keypoint_calibration_seed(config) -> int:
    """The survey's own reset seed, disjoint from training and evaluation."""
    return int((config.get("seeds") or {}).get("keypoint_calibration", 31337))


def calibrate_keypoint_encoder_in_flight(
        path, artifact, camera, *, cfg, config, system, mode, device,
        datastore=None):
    """Calibrate against live Isaac, surveying poses when one is still owed.

    The flight is lazy: with an accumulation that already covers the survey no
    vehicle is armed at all.
    """
    if artifact is None or not needs_empirical_calibration(artifact):
        return calibrate_keypoint_encoder(
            path, artifact, camera.labelled, system=system,
            experiment=config, mode=mode, device=device, datastore=datastore)
    settings = (config.get("estimator") or {}).get("keypoint_pretraining")
    with KeypointCalibrationFlight(
            cfg, camera, seed=keypoint_calibration_seed(config),
            settings=settings) as flight:
        return calibrate_keypoint_encoder(
            path, artifact, camera.labelled, survey=flight, system=system,
            experiment=config, mode=mode, device=device, datastore=datastore)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _start_rviz(enabled: bool, parallel_pairs: int = 1
                ) -> tuple[subprocess.Popen | None, object | None]:
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
    command = [str(script)]
    if int(parallel_pairs) > 1:
        command.extend(["--parallel-pairs", str(int(parallel_pairs))])
    process = subprocess.Popen(
        command, cwd=str(ROOT), stdout=stream, stderr=subprocess.STDOUT)
    # Environment/Qt loader errors surface immediately. Do not let a broken
    # optional window abort a publication-scale flight run.
    time.sleep(2.0)
    if process.poll() is not None:
        stream.close()
        print(f"WARNING: RViz 2 exited immediately; see {log_path}.")
        return None, None
    layout = (f"{int(parallel_pairs)} landing cameras and isolated pair frames"
              if int(parallel_pairs) > 1 else "the landing camera and flight layout")
    print(f"RViz 2 opened with {layout}.")
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
        with LiveShinEnvironment(
                cfg, camera, horizon_steps=int(cfg.sim.max_steps)) as env:
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
                      f"safe_landing={int(metric['paper_success'])} | "
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
    parser.add_argument("--total-train-episodes", type=int,
                        help="divide this exact training budget across selected methods")
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
    if args.train_episodes is not None and args.total_train_episodes is not None:
        parser.error("use either --train-episodes or --total-train-episodes, not both")
    for option, value in (("--train-episodes", args.train_episodes),
                          ("--total-train-episodes", args.total_train_episodes),
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
    # Policies, empirical R-GAT data and frozen reward weights depend on both
    # the learning experiment and the resolved live simulator/hardware model.
    # Including the latter prevents a battery, camera or PX4 change from
    # silently resuming an incompatible checkpoint.
    resolved_system_config = load_system_config(args.system_config)
    config_hash = configuration_hash({
        "experiment": config,
        "system": resolved_system_config,
        "outcome_contract": (
            "safe_landing_contact_position_velocity_attitude_rate_v2"),
    })
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
    if args.total_train_episodes is not None:
        try:
            train_count = episodes_per_method(
                args.total_train_episodes, len(args.methods))
        except ValueError as exc:
            parser.error(str(exc))
    else:
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
    keypoint_pretraining = prepare_keypoint_encoder(
        args.results_dir / "models/shin2026_keypoint_encoder.pt",
        config_hash=config_hash, experiment=config, system=resolved_system_config,
        mode=args.mode, device=args.device)
    manifest = {
        "config": str(args.config.resolve()), "config_hash": config_hash,
        "system_config": str(args.system_config.resolve()), "mode": args.mode,
        "methods": args.methods, "training_episodes_per_method": train_count,
        "training_episodes_total": train_count * len(args.methods),
        "evaluation": scenarios, "paired_seeds": True,
        "landing_success_contract": {
            "version": "safe_landing_v2", "all_required": True,
            "pad_contact": True,
            "maximum_lateral_error_m": float(
                (resolved_system_config.get("landing") or {}).get(
                    "success_xy_m", 0.35)),
            "maximum_vertical_speed_m_s": float(
                (resolved_system_config.get("landing") or {}).get(
                    "success_vz_m_s", 0.55)),
            "maximum_relative_horizontal_speed_m_s": float(
                (resolved_system_config.get("landing") or {}).get(
                    "success_rel_speed_xy_m_s", 0.45)),
            "maximum_tilt_deg": float(
                (resolved_system_config.get("landing") or {}).get(
                    "success_tilt_deg", 10.0)),
            "maximum_angular_rate_deg_s": float(
                (resolved_system_config.get("landing") or {}).get(
                    "success_rate_deg_s", 45.0)),
            "unsafe_contact_is_failure": True,
        },
        "reward_design_id": getattr(potential, "design_id", None),
        "reward_design_sha256": getattr(potential, "sha256", None),
        "trajectory_note": "named evaluation trajectories are documented approximations",
        "table_ii_runtime_application": (
            "seeded PX4 gain spread, Isaac force/torque and initial-state "
            "perturbation, live-camera appearance randomization"),
        "keypoint_pretraining": (
            None if keypoint_pretraining is None else {
                "format": keypoint_pretraining["format"],
                "implementation": keypoint_pretraining["implementation"],
                "frozen_for_ppo": keypoint_pretraining["frozen_for_ppo"],
                "training_source": keypoint_pretraining["training_source"],
                "metrics": keypoint_pretraining["metrics"],
            }),
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
    control_config = dict(config.get("control") or {})
    cfg.benchmark_control = control_config
    cfg.sim.dt = float(control_config.get("dt_seconds", 0.1))
    cfg.sim.max_steps = int(control_config.get("horizon_steps", 300))
    cfg.sim.max_time = cfg.sim.dt * cfg.sim.max_steps
    cfg.external.control_hz = 1.0 / cfg.sim.dt
    cfg.viz.dashboard.enabled = not args.no_dashboard
    if args.dashboard_port is not None:
        cfg.viz.dashboard.port = int(args.dashboard_port)
    ppo_config = dict(config.get("ppo") or {})
    ppo_config["perception_warmup_episodes"] = int(ppo_config.get(
        f"perception_warmup_episodes_{args.mode}",
        2 if args.mode == "quick" else 32))
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
    # Headless is Isaac's window, not the operator's: RViz reads ROS topics
    # that are published in either mode.
    rviz_process, rviz_log = _start_rviz(
        cfg.viz.rviz.enabled and not args.no_rviz)
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
        with RosGrayscaleSource(
                # Training-label-only pose stream for keypoint supervision.
                truth_pose_topic=(
                    "/landing_uav0/perception/pad_relative_truth_pose")
        ) as camera:
            monitor.stage("keypoint validation",
                          "live Isaac camera · surveyed viewpoints")
            keypoint_pretraining = calibrate_keypoint_encoder_in_flight(
                args.results_dir / "models/shin2026_keypoint_encoder.pt",
                keypoint_pretraining, camera, cfg=cfg, config=config,
                system=resolved_system_config, mode=args.mode,
                device=args.device)
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
            manifest_path.write_text(
                json.dumps(manifest, indent=2), encoding="utf-8")
            models = {}
            training_by_method = {}
            curriculum_raw = dict(config.get("curriculum") or {})
            curriculum_levels = int(curriculum_raw.get("levels", 80))
            curriculum_interval = int(curriculum_raw.get(
                "update_every_episodes", 512))
            curriculum_config = {
                "levels": curriculum_levels,
                "episodes_per_update": curriculum_interval,
                "performance_gated": bool(curriculum_raw.get(
                    "performance_gated", False)),
                "assessment_window": int(curriculum_raw.get(
                    "assessment_window", 20)),
                "minimum_episodes_at_level": int(curriculum_raw.get(
                    "minimum_episodes_at_level", 20)),
                "success_rate_threshold": float(curriculum_raw.get(
                    "success_rate_threshold", 0.20)),
                "max_position_rmse_m": float(curriculum_raw.get(
                    "max_position_rmse_m", 2.0)),
                "max_geometric_fov_loss_fraction": float(curriculum_raw.get(
                    "max_geometric_fov_loss_fraction", 0.50)),
            }

            def train_requested(method):
                monitor.stage("training", f"recurrent PPO · {method}")
                # Identical initialization is part of the paired comparison.
                torch.manual_seed(model_seed)
                model = _build_model(config, args.device, keypoint_pretraining)
                method_potential = potential if method.startswith("ontoreward") else None
                history = train_live(
                    lambda: LiveShinEnvironment(
                        cfg, camera, horizon_steps=int(cfg.sim.max_steps)),
                    model, method, range(training_seed0, training_seed0 + train_count),
                    args.results_dir / "models", config_hash=config_hash,
                    potential=method_potential, ppo=ppo_config,
                    curriculum_config=curriculum_config, monitor=monitor,
                    restart_incompatible=True)
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
                    source_model = _build_model(
                        config, args.device, keypoint_pretraining)
                    source_dir = args.results_dir / "models/rgat_design_source"
                    train_live(
                        lambda: LiveShinEnvironment(
                            cfg, camera, horizon_steps=int(cfg.sim.max_steps)),
                        source_model, "shin2026",
                        range(training_seed0, training_seed0 + train_count),
                        source_dir, config_hash=config_hash, potential=None,
                        ppo=ppo_config, curriculum_config=curriculum_config,
                        monitor=None, restart_incompatible=True)
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
                with LiveShinEnvironment(
                        cfg, camera, horizon_steps=int(cfg.sim.max_steps)) as env:
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
    try:
        exit_code = main()
    except KeyboardInterrupt:
        # The main finally block has already stopped every owned process. Keep
        # an intentional Ctrl-C distinguishable from an experiment failure.
        print("Pipeline interrupted by user; checkpoint and completed rows were preserved.")
        exit_code = 130
    raise SystemExit(exit_code)
