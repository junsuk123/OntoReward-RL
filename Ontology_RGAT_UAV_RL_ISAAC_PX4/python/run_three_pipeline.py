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
import math
import os
from pathlib import Path
import subprocess
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
from ontology_rgat.controllers import (PLANAR_ACTION_DIM, PN_GUIDANCE_METHOD,
                                       PNGuidanceConfig, PNGuidanceController,
                                       PNGuidanceState)
from ontology_rgat.initialization import nadir_image_setpoint
from ontology_rgat.evaluation import (write_adaptive_reward_figures,
                                      write_presentation_results,
                                      write_three_pipeline_outputs,
                                      write_two_pipeline_outputs)
from ontology_rgat.perception import (RosGrayscaleSource,
                                      needs_empirical_calibration,
                                      prepare_keypoint_encoder)
from ontology_rgat.pipelines import (ablation_pipeline_ids,
                                     assert_no_aruco_in_primary_system,
                                     available_pipeline_ids, get_pipeline,
                                     primary_pipeline_ids,
                                     validate_pipeline_configuration)
from ontology_rgat.ppo.behavior_cloning import (
    behavior_clone, encoded_demonstration_episode,
    load_encoded_demonstrations, merge_encoded_demonstrations,
    save_encoded_demonstrations)
from ontology_rgat.ppo.recurrent_train import (_battery_reserve,
                                               collect_episode_resilient,
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
    FrozenFOVRiskPredictor, build_fov_risk_dataset, fov_risk_data_fingerprint,
    horizon_steps, load_fov_risk_dataset, prepare_fov_risk_artifact,
    save_fov_risk_dataset)
from ontology_rgat.datastore import KIND_FOV_RISK, open_datastore
from ontology_rgat.stack import ExternalStack
from ontology_rgat.viz.contracts import (algorithm_pipeline_contract,
                                          graph_state_pipeline_contract,
                                          mdp_contract)
from ontology_rgat.viz.dashboard import Dashboard
from ontology_rgat.viz.live import BenchmarkMonitor, STORE
from ontology_rgat.viz.rviz import RvizPublisher
from run_shin2026_pipeline import (_build_model, _live_config, _sha256_file,
                                   _start_rviz, _write_csv,
                                   calibrate_keypoint_encoder_in_flight)


def _json_safe(value):
    """Represent an unavailable statistic as null rather than as NaN.

    ``allow_nan=False`` below is deliberate: NaN is not JSON, and a manifest
    carrying it is unreadable by anything downstream. But a NaN here is usually
    not corruption -- it is a mean over zero samples, which is what every
    adaptive-R-GAT column is for an experiment whose arms do not use adaptive
    reward weights. That is honestly "not measured", so it is written as null.

    Without this the whole manifest write failed at the very end of a completed
    run (2026-09-22): every evaluation row, report and checkpoint was already
    committed, and the run still exited non-zero with execution_status left at
    "results pending" because a column nobody in the experiment uses summarised
    to NaN.
    """
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False),
                         encoding="utf-8")
    os.replace(temporary, path)


# The two halves a run is made of, and the reason they are named.
#
# Flying an episode and fitting a model are different resources: the first is
# bounded by one physical Isaac/PX4 stage, the second by a GPU. Until
# 2026-09-22 the run interleaved them -- the fixed-reward arms' PPO was
# submitted first and the FOV-risk collector was handed whatever pair was left
# over -- which made the collection single-pair on a four-pair stage, put the
# collector and a PPO replica on the same pair when the arithmetic did not
# work out, and meant the proposed arm's frozen readout was fitted while its
# comparison arm was already several hundred episodes into training.
#
# The stages below are strictly ordered instead. ``collect`` flies everything
# that produces a reusable dataset, on every pair at once, and stops. ``train``
# fits the frozen readouts, runs PPO for every arm and scores them, and flies
# nothing for collection -- a missing dataset is an error naming the collect
# command rather than a silent multi-hour flight. ``all`` is the two in one
# process, which is what a bare ``./run.sh`` still does.
# The deck every collection flies unless a profile names another one.
#
# ``straight_escape_burst_track``: the pad cruises at 0.50 m/s along a closed
# oval -- two 20 m straights joined by constant-speed 6 m semicircles -- the
# vehicle settles into following it, and then it doubles to 1.00 m/s and pulls
# out of the camera frame. It is the base scenario because it *produces* the
# event the whole method is about -- the pad leaving the FOV -- in every
# episode, rather than incidentally in some. A readout fitted on a deck that
# rarely leaves the frame has almost no positive target to learn from.
#
# The oval rather than the open straight run (``straight_escape_burst``, still
# available) because an open run is straight only *within* an episode: the deck
# is left where the episode ended and the next one draws a new heading, so the
# sequence walks away and has to be corrected by steering it back and by
# clamping it at the arena edge. The closed track needs neither -- measured
# over 300 episodes with a dash in every one, the deck stays inside 32 x 15 m
# and never reaches even the half-arena.
#
# The other decks are not deleted; they are simply not the default any more,
# and two_pipeline_comparison.yaml still declares all six. What overrides this,
# in order of precedence:
#
#   PPO training + every collection   training.scenarios
#   teacher demonstrations            behavior_cloning.scenarios
#   FOV-risk rollouts                 fov_risk_design.scenarios
#   semantic R-GAT rollouts           rgat_design.scenarios
#   adaptive reward rollouts          adaptive_reward_design.scenarios
#
# The per-design keys win over ``training.scenarios``; this constant applies
# only when a profile names no deck at all.
COLLECTION_BASE_SCENARIO = "straight_escape_burst_track"

STAGE_COLLECT = "collect"
STAGE_TRAIN = "train"
STAGE_ALL = "all"
STAGES = (STAGE_ALL, STAGE_COLLECT, STAGE_TRAIN)


class CollectionUnavailable(RuntimeError):
    """A training-stage run would have had to fly to fill a dataset."""


def _require_flight(allowed: bool, what: str, *, shortfall: str = "") -> None:
    """Refuse to collect when this stage is not the collecting one.

    Called at the point a collector has decided it must arm a vehicle. In
    ``--stage train`` that decision is a contract violation, not a task: the
    operator asked for training over data that was supposed to already exist,
    and flying it here would both take hours they did not ask for and fit the
    readout on data no collect run ever recorded.
    """
    if allowed:
        return
    detail = f" ({shortfall})" if shortfall else ""
    raise CollectionUnavailable(
        f"--stage train cannot continue: {what} is incomplete{detail}. "
        "Run the collection stage first (./run.sh --stage collect, same "
        "--config/--system-config), or use --stage all to collect and train "
        "in one process.")


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
                        "evaluation_update", "teacher_step",
                        "demonstration_progress"}:
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


# Mirrors isaac_sim/landing_world.py: one shared stage, bounded by how many
# pair offsets a profile defines rather than by a fixed guess.
MAX_PARALLEL_PAIRS = 8

# Marginal cost of one more UAV/UGV pair, measured on the shared stage rather
# than estimated: at four pairs Isaac held 4.7 GB RSS and 2.7 GiB of VRAM, the
# four gateways 305 MB between them, the four PX4 instances 33 MB, and the
# whole stack drew about 2.4 of 20 cores. The base figures are what one pair
# already costs before any of that scales.
PAIR_SYSTEM_RAM_GB = 0.45
BASE_SYSTEM_RAM_GB = 9.0          # Isaac, the learner and torch before pairs
PAIR_VRAM_GB = 0.35
BASE_VRAM_GB = 1.6
PAIR_CPU_CORES = 0.7

# Highest pair count whose throughput has actually been measured on the shared
# stage (4 pairs, 4.96x total). The resource model above happily allows more,
# but six pairs produced one pathological reading (0.363x) that was never
# explained, and an unmeasured recommendation is how several wrong conclusions
# were reached before this ceiling existed. Raise it by measuring, not by
# reasoning about headroom.
MEASURED_PAIR_CEILING = 4


def machine_resources() -> dict:
    """What this box has, for sizing the stage at start-up.

    Reported rather than assumed: the numbers go into the run's own log so a
    pair count can be explained after the fact.
    """
    resources = {"cpu_count": os.cpu_count() or 1,
                 "total_ram_gb": 0.0, "total_vram_gb": 0.0}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                resources["total_ram_gb"] = int(line.split()[1]) / (1024.0 ** 2)
                break
    except (OSError, ValueError, IndexError):
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
        totals = [float(value) for value in out.stdout.split() if value.strip()]
        if totals:
            # One shared Isaac stage renders on one device.
            resources["total_vram_gb"] = max(totals) / 1024.0
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return resources


def automatic_parallel_pairs(method_count, *, cpu_count, total_ram_gb,
                             total_vram_gb, offsets_available,
                             maximum=MAX_PARALLEL_PAIRS):
    """How many physical pairs this machine should run, as a multiple of arms.

    Every method must get the same number of pairs or the arms stop being
    budget-matched, so the answer is always a multiple of ``method_count``.

    More pairs is worth wanting: total simulated seconds per wall second
    measured 1.00x, 2.60x, 3.28x and 4.96x at one, two, three and four pairs,
    because the shared Isaac stage steps one scene however many vehicles stand
    in it. The bound is memory rather than speed, so the limits below are
    headroom limits and the caller still says what it actually wants.
    """
    methods = max(1, int(method_count))
    limits = {
        "profile pair offsets": int(offsets_available),
        "configured maximum": int(maximum),
        "measured throughput ceiling": MEASURED_PAIR_CEILING,
        "system RAM": int((float(total_ram_gb) - BASE_SYSTEM_RAM_GB)
                          / PAIR_SYSTEM_RAM_GB) if total_ram_gb else maximum,
        "GPU memory": int((float(total_vram_gb) - BASE_VRAM_GB) / PAIR_VRAM_GB)
                      if total_vram_gb else maximum,
        "CPU cores": int(float(cpu_count) / PAIR_CPU_CORES) if cpu_count else maximum,
    }
    allowed = min(limits.values())
    # Down to a whole number of rounds so both arms fly the same count.
    pairs = (allowed // methods) * methods
    if pairs < methods:
        # One round is the floor: fewer pairs than methods cannot give each
        # method a pair of its own, which parallel training requires.
        return methods, "minimum of one pair per method", limits
    binding = min(limits, key=limits.get)
    return pairs, binding, limits


def _pair_live_config(cfg, pair_index: int, pair_count: int):
    """Give one learner an exclusive UDP endpoint in the shared Isaac world."""
    if pair_count == 1:
        return cfg
    index = int(pair_index)
    # The entry budget itself is simulated PX4 time (external.entry_sim_budget)
    # and therefore already immune to how slowly the shared stage runs; this
    # scale applies only to the wall-clock hang guard, which has to stay
    # comfortably above that budget as more rendered cameras slow the stage.
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


def _planar_from_velocity_action(controller, velocity_action):
    """Convert a normalized velocity action into the shared planar action.

    Both analytic teachers are velocity laws written against the retired
    three-axis envelope, and their tests fix them in that form. The reduced
    envelope commands acceleration, has no lateral channel and has a tilt
    channel, so exactly one conversion stands between them -- here, so that the
    teacher, the control condition and the policy all reach PX4 through the
    same integrator and the same limits.

    The lateral component is dropped rather than folded in: there is no channel
    to give it to, and on the planar decks the entry is seeded on the deck's
    track so it is close to zero to begin with.
    """
    value = np.asarray(velocity_action, dtype=float).reshape(-1)
    if value.size < 3 or not np.isfinite(value).all():
        raise ValueError("a velocity action needs at least three finite values")
    limit = np.asarray(controller.velocity_limit_flu, dtype=float)
    return controller.action_for_velocity(
        float(value[0] * limit[0]), float(value[2] * limit[2]))


def _behavior_transform(variant: int, *, policy_blend: float = .35,
                        servo_gain: float = 1.0, noise_std: float | None = None,
                        tilt_blend: float = .25):
    """Estimator-free mixture over the planar channels.

    Three variants, so a collection pass covers approaches that converge, ones
    that wander and ones that clearly fail -- the dataset needs all three.
    Channels are ``[a_fwd, a_z, tilt]``: the longitudinal image error drives
    the first, the approach phase the second, and the third follows the first
    so the collected states include the airframe attitudes the graph's tilt
    channel is supposed to describe.
    """
    def transform(step, policy_action, semantic, rng):
        action = np.asarray(policy_action, dtype=np.float64).copy()
        if action.shape != (PLANAR_ACTION_DIM,):
            raise ValueError("the behaviour mixture takes the planar action")
        if variant == 0:
            # Direct image-plane servo on the column axis alone. The row axis
            # is the lateral one, and there is no lateral channel to give it
            # to; the envelope holds that degree of freedom at zero.
            column = float(semantic.centroid_xy[0])
            recovering = (semantic.visible_keypoint_fraction < 0.5
                          or semantic.visual_loss_risk > 0.0)
            correction = float(servo_gain) * (0.45 if recovering else 0.75)
            action[0] = np.clip(
                float(policy_blend) * action[0] + correction * column, -0.8, 0.8)
            if recovering:
                # Climb to widen the footprint and stop asking for attitude
                # while there is nothing to aim at.
                action[1] = 0.45
                action[2] *= min(float(tilt_blend), 0.15)
            else:
                if (semantic.image_alignment > 0.65
                        and semantic.keypoint_confidence > 0.01):
                    scale = semantic.apparent_target_scale
                    descent = (-0.55 if scale < 0.45
                               else -0.28 if scale < 0.75 else -0.10)
                    action[1] = min(0.25 * action[1], descent)
                else:
                    action[1] = max(0.25 * action[1], 0.0)
                # The policy's own tilt is damped wherever it is not being
                # overridden outright, so the collected attitudes stay inside
                # the range the graph's tilt channel is meant to describe.
                action[2] *= float(tilt_blend)
            action += rng.normal(
                0.0, 0.04 if noise_std is None else float(noise_std),
                PLANAR_ACTION_DIM)
        elif variant == 1:
            action += rng.normal(0.0, 0.18, PLANAR_ACTION_DIM)
        else:
            # Bounded exploration supplies clear failure-side graph states.
            action = (0.55 * action
                      + 0.45 * rng.uniform(-1.0, 1.0, PLANAR_ACTION_DIM))
        return np.clip(action, -0.9, 0.9)
    return transform


def _privileged_velocity_teacher_action(
        relative_state, body_velocity, semantic, velocity_limit,
        *, position_gain: float = .35, velocity_gain: float = .75,
        horizontal_speed_limit: float = .60, energy_urgency: float = 0.0,
        touchdown_speed_limit: float = .55, urgency_alignment_m: float = .25,
        visual_loss_climb_risk: float = .25,
        visual_loss_climb_fraction: float = 0.0,
        visual_loss_climb_source: str = "perception",
        geometric_in_fov: bool | None = None,
        descent_high_m_s: float = .35, descent_mid_m_s: float = .14,
        descent_flare_m_s: float = .04, approach_descent_m_s: float = 0.0,
        approach_descent_min_altitude_m: float = 2.5,
        approach_lateral_m: float = 1.5, climb_ceiling_m: float = 0.0,
        climb_lateral_m: float = .40, climb_altitude_m: float = .80,
        contact_relative_speed_m_s: float = .45,
        recovery_descent_m_s: float = 0.0, recovery_altitude_m: float = 0.0,
        noise_std: float = .01, rng=None, info: dict | None = None):
    """Return a stable training-only velocity label for deadline warm starts.

    ``relative_state`` is platform-in-UAV-body simulator truth and is never an
    actor input. The teacher reconstructs the deck's horizontal velocity from
    ``v_uav + v_relative`` and adds a damped position correction. A
    position-only image servo commands zero at image centre, lets a moving UGV
    escape, and repeatedly crosses the target.

    ``energy_urgency`` scales the descent with how little charge is left, the
    way ``expert.expert_action`` already does and for the reason its comment
    gives: without it every low-reserve episode is a negative example and
    nothing downstream has a successful one to learn from. The pack is drawn
    per episode and the draw is wide -- measured over 11 teacher flights on
    2026-09-19, 5 began with a reduced pack (0.42 to 0.85 of full, 2300 to
    4700 J against the 6000-7500 J a nominal descent actually spends) and this
    teacher landed none of them, three of those while *already inside* the
    0.35 m position criterion. It was aligned and simply could not get down
    before the pack ran out.

    ``semantic.battery_risk`` is ``1 - reserve`` read live, so the urgency
    ramps through *every* flight as the charge goes down rather than only on
    episodes dealt a small pack. That is the intent -- a vehicle with half a
    pack left has less reason to dawdle whatever it started with -- and it is
    visible in the flights: full-pack seed 90000 landed in 173 steps with this
    on against 258 with it off, from the same entry and to the same 0.13 m.

    Left at 0 this is exactly the previous controller, which is how it ships:
    turning it on changes the demonstrations, so it is a deliberate config
    edit rather than a silent default.
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
    # Recovery is triggered by *sustained blindness* -- fewer than two
    # landmarks reported for ``visual_loss_climb_risk`` x 2 s (0.5 s at the
    # default) -- and no longer by a partially visible pad. Measured
    # 2026-09-20: with the honest v4 encoder (it calls 15 % of out-of-frame
    # landmarks visible; the collapsed v5 encoder called 77 % of them visible)
    # the old ``visible_keypoint_fraction < 0.5`` trigger fired exactly where
    # a correct approach must pass. The camera is pitched 60 deg with a 32 deg
    # vertical half-angle, so directly above the deck -- the point this PD
    # drives to -- only 2 of 6 landmarks are in frame between 0.4 and 1.0 m
    # (4 at 1.2-1.5 m, 6 from 2 m). The teacher therefore climbed at ~1.1 m,
    # re-acquired at 1.2 m, descended, climbed again, and 40 of 40 flights
    # timed out or ran the pack down without a single touchdown, where the
    # same teacher behind the v5 encoder had landed 4 of 22. The v5 encoder
    # was not seeing the deck; it was hiding this rule.
    if str(visual_loss_climb_source) == "geometric":
        # Privileged teacher, privileged cue: the pipeline's one definition of
        # field-of-view loss (pad centre outside the camera frustum), which
        # no encoder calibration can move. The second v4 run still hovered at
        # 0.6-1.0 m over a deck it was aligned on to 12 cm (seed 90025:
        # 300 steps, no contact) because two in-frame landmarks reported by
        # an honest encoder flicker across the visibility threshold, and one
        # miss makes the semantic observation "blind".
        if geometric_in_fov is None:
            raise ValueError("geometric visual-loss source needs geometric_in_fov")
        visual_lost = not bool(geometric_in_fov)
    else:
        visual_lost = (float(semantic.visual_loss_risk) > float(visual_loss_climb_risk)
                       or float(semantic.visible_keypoint_fraction)
                       < float(visual_loss_climb_fraction))
    # The marker naturally fills and then leaves the downward camera at the
    # end of a correct flare. Treating that expected low-altitude
    # disappearance as a recovery event traps the vehicle centimetres above
    # the deck. Only climb on loss while still high or laterally displaced.
    # ``battery_risk`` is the estimator-free reading of how little is left, so
    # the urgency comes from what the student can also see rather than from
    # the pack itself.
    urgency = float(np.clip(
        float(energy_urgency) * float(semantic.battery_risk), 0.0, 1.0))
    # Vertical schedule. The defaults are the v4 ladder; the experiment sets
    # a faster one (2026-09-21) because the traced flights showed the ladder
    # itself was the binding constraint: from the 5 m entry it needs ~30 s of
    # *uninterrupted* alignment (11 s at 0.35, 3.6 s at 0.14, 15 s at 0.04
    # m/s) inside a 30 s episode, and alignment against the random-walk deck
    # is intermittent (seed 90004: aligned at step 54, timed out at 1.9 m).
    # ``approach_descent_m_s`` lets a high, roughly aligned vehicle come down
    # while it is still closing, which the old ladder forbade entirely.
    ceiling = float(climb_ceiling_m)
    # When a lost deck is worth climbing for. Both thresholds were fitted to
    # the random walk, where the vehicle sat almost over a deck that stayed
    # put, so a pad outside the frame meant it had been lost: climb whenever
    # lost above 0.80 m, or whenever more than 0.40 m off.
    #
    # Chasing a deck that drives holds 0.3-0.9 m of lateral error, and a
    # 60-degree down camera loses a pad at that offset as ordinary geometry.
    # The rule then fires continuously: of the first six route-deck flights
    # the three long ones spent 25%, 38% and 40% of their steps climbing and
    # ran the pack out between 0.13 m and 0.70 m above the deck, while the one
    # that landed climbed on none of its steps and spent 0.03% of its pack per
    # step against their 0.18-0.28%.
    #
    # Raising ``climb_altitude_m`` hands the band the flight actually lives in
    # back to ``climb_lateral_m``, which asks whether the deck is really away
    # rather than merely out of a downward camera. Above it, a lost deck still
    # gets the climb it was written for.
    if visual_lost and (altitude > float(climb_altitude_m)
                        or lateral_error > float(climb_lateral_m)):
        # Climb to re-acquire, but not forever. On the route decks the pad
        # leaves the frustum repeatedly while the vehicle is still chasing, and
        # an unbounded climb turns that into a runaway: two of the first eight
        # traced flights ended at 6.5 m and 12.0 m having spent 35% and 69% of
        # their steps climbing, with the deck 1-5 m away and the pack empty.
        # Above the ceiling the vehicle holds altitude and keeps chasing, which
        # is what actually brings the deck back into a 60-degree down camera.
        target_vz = 0.0 if 0.0 < ceiling <= altitude else 0.22
    elif (lateral_error > 0.50
            or relative_speed > float(contact_relative_speed_m_s)):
        # ``contact_relative_speed_m_s`` is the touchdown criterion itself
        # (``criteria.rel_speed_xy``), not a tuning constant: descending while
        # the deck is moving faster than that under the vehicle buys a contact
        # the landing gate scores as unsafe. The caller passes the live value
        # so the two cannot drift.
        if (altitude > float(approach_descent_min_altitude_m)
                and lateral_error < float(approach_lateral_m)
                and relative_speed < 0.70):
            target_vz = -abs(float(approach_descent_m_s))
        elif (float(recovery_descent_m_s) > 0.0 and not visual_lost
                and altitude > float(recovery_altitude_m)):
            # Give back the altitude a climb bought once the climb has paid
            # off. Otherwise only two branches can fire out here -- climb when
            # the deck is out of frame, hold when it is in -- so every loss
            # adds altitude and nothing removes it until the vehicle is inside
            # ``approach_lateral_m``, which on a deck that keeps driving may
            # not happen: measured 2026-09-22, a circle flight ratcheted 3.20
            # -> 5.09 m in eleven such cycles and parked at the climb ceiling
            # with the pack draining. The pad is in frame here, so this is
            # descending on a deck the vehicle can see, not blind.
            target_vz = -abs(float(recovery_descent_m_s))
        else:
            target_vz = 0.0
    elif altitude > 1.10:
        target_vz = -abs(float(descent_high_m_s))
    elif altitude > 0.60:
        target_vz = -abs(float(descent_mid_m_s))
    else:
        target_vz = -abs(float(descent_flare_m_s))
    # Only compress a descent that is already well aligned. Coming down faster
    # magnifies the bearing to a deck that is not directly below, and the deck
    # then leaves the frame: measured over the first four flights with this on,
    # geometric FOV loss ran 0.284 against 0.124 over the 24 flights before it,
    # and a seed that had landed twice at 0.12 m timed out at 0.74 m instead.
    # The ladder above already refuses to descend beyond 0.50 m, which is wide
    # enough to include that. The episodes this feature exists to save are the
    # ones that ran out of charge while *inside* the position criterion --
    # 0.084 m, 0.192 m, 0.222 m -- so the gate only has to clear those.
    if (target_vz < 0.0 and urgency > 0.0
            and lateral_error < float(urgency_alignment_m)):
        # Never ask for a touchdown speed the criteria would reject: a landing
        # that arrives too fast to pass the gate is not a demonstration.
        ceiling = .95 * float(touchdown_speed_limit)
        target_vz = -min(ceiling, abs(target_vz) * (1.0 + 2.0 * urgency))

    target_velocity = np.r_[target_xy, target_vz]
    if info is not None:
        info.update({
            "altitude_m": altitude, "lateral_error_m": lateral_error,
            "relative_speed_m_s": relative_speed, "target_vz_m_s": float(target_vz),
            "target_vxy_m_s": [float(v) for v in target_xy],
            "visual_lost": bool(visual_lost),
            "visible_keypoint_fraction": float(semantic.visible_keypoint_fraction),
            "visual_loss_risk": float(semantic.visual_loss_risk),
            "geometric_in_fov": (None if geometric_in_fov is None
                                 else bool(geometric_in_fov)),
            "battery_risk": float(semantic.battery_risk), "urgency": urgency})
    # This function stays a VELOCITY law and returns a normalized velocity
    # target, which is what its tests fix and what makes them readable. The
    # conversion to the shared planar envelope happens once, in the binding
    # below, so every analytic arm reaches PX4 through the same integrator
    # under the same limits as the policy.
    action = np.r_[target_velocity / limit, 0.0]
    if float(noise_std) > 0.0:
        generator = rng if rng is not None else np.random.default_rng()
        action += generator.normal(0.0, float(noise_std), 4)
    return np.clip(action, -0.90, 0.90)


class _PrivilegedVelocityTeacher:
    """Bind the training-only teacher to the environment's current live step.

    Optionally writes one JSON line per control step to ``trace_path``: the
    attempts CSV records only the final step of a flight, which cannot tell
    a teacher that never converged from one that converged and then hovered.
    ``seed`` is set by the caller before each flight.
    """

    def __init__(self, environment, *, settings, trace_path=None, monitor=None):
        self.environment = environment
        self.settings = dict(settings)
        self.trace_path = None if trace_path is None else Path(trace_path)
        self.monitor = monitor
        self.method = str(self.settings.get("source_pipeline", "no_se_fixed"))
        self.seed = None
        self._trace = None
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            self._trace = self.trace_path.open("a", encoding="utf-8")

    def close(self):
        if self._trace is not None:
            self._trace.close()
            self._trace = None

    def __call__(self, step, policy_action, semantic, rng):
        environment, settings = self.environment, self.settings
        current = environment.last_step
        if current is None:
            raise RuntimeError("privileged teacher requires a reset live environment")
        controller = environment.adapter.controller
        info = {}
        action = _privileged_velocity_teacher_action(
            current.critic.true_relative_state,
            current.actor.body_velocity,
            semantic,
            controller.velocity_limit_flu,
            position_gain=float(settings.get("position_gain", .35)),
            velocity_gain=float(settings.get("velocity_gain", .75)),
            # ``pd_`` first, as with the descent keys: both teachers read the
            # unprefixed clamp and mean different things by it. The servo is
            # measurably stable only at 0.60, while this teacher needs closing
            # authority above whatever the deck is doing -- on the route decks
            # the demonstrations now fly that is 0.48 m/s, which leaves 0.12 of
            # the shared value to close with.
            horizontal_speed_limit=float(settings.get(
                "pd_horizontal_speed_limit_m_s",
                settings.get("horizontal_speed_limit_m_s", .60))),
            energy_urgency=float(settings.get("energy_urgency", 0.0)),
            urgency_alignment_m=float(settings.get(
                "urgency_alignment_m", .25)),
            visual_loss_climb_risk=float(settings.get(
                "visual_loss_climb_risk", .25)),
            visual_loss_climb_fraction=float(settings.get(
                "visual_loss_climb_fraction", 0.0)),
            visual_loss_climb_source=str(settings.get(
                "visual_loss_climb_source", "perception")),
            geometric_in_fov=bool(current.geometric_pad_center_in_fov),
            descent_high_m_s=float(settings.get("pd_descent_high_m_s", .35)),
            descent_mid_m_s=float(settings.get("pd_descent_mid_m_s", .14)),
            descent_flare_m_s=float(settings.get("pd_descent_flare_m_s", .04)),
            approach_descent_m_s=float(settings.get("pd_approach_descent_m_s", 0.0)),
            approach_descent_min_altitude_m=float(settings.get(
                "pd_approach_min_altitude_m", 2.5)),
            approach_lateral_m=float(settings.get("pd_approach_lateral_m", 1.5)),
            climb_ceiling_m=float(settings.get("pd_climb_ceiling_m", 0.0)),
            climb_lateral_m=float(settings.get("pd_climb_lateral_m", .40)),
            climb_altitude_m=float(settings.get("pd_climb_altitude_m", .80)),
            # The landing gate's own value, so the teacher cannot hold against
            # a criterion the run no longer scores against.
            contact_relative_speed_m_s=float(
                environment.cfg.criteria.rel_speed_xy),
            recovery_descent_m_s=float(settings.get(
                "pd_recovery_descent_m_s", 0.0)),
            recovery_altitude_m=float(settings.get(
                "pd_recovery_altitude_m", 0.0)),
            noise_std=float(settings.get("noise_std", .01)),
            rng=rng, info=info)
        planar = _planar_from_velocity_action(controller, action)
        info["planar_action"] = [float(v) for v in planar]
        info["commanded_tilt_deg"] = float(np.degrees(
            float(planar[2]) * controller.max_longitudinal_tilt
            * controller.action_scale))
        _publish_teacher_step(self.monitor, method=self.method, step=step,
                              seed=self.seed, info=info)
        if self._trace is not None:
            state = current.state if isinstance(current.state, dict) else {}
            record = {
                "seed": self.seed, "step": int(step), "wall": time.time(),
                "px4_time_us": state.get("px4_time_us"),
                "battery_reserve": _battery_reserve(state) if state else None,
                # Both forms: the velocity the law asked for, and the planar
                # action it became. A trace with only one of them cannot tell a
                # teacher that asked for the wrong thing from one whose request
                # the envelope could not deliver.
                "velocity_action": [
                    float(v) for v in np.asarray(action).reshape(-1)],
                "action": [float(v) for v in planar],
                **info}
            self._trace.write(json.dumps(record, default=float) + "\n")
            self._trace.flush()
        return planar


def _publish_teacher_step(monitor, *, method: str, step, seed, info: dict) -> None:
    """Hand the teacher's per-step diagnostics to the live dashboard.

    Telemetry must never end a demonstration flight, so a monitor without the
    method, or one that fails, is reported and otherwise ignored.
    """
    publish = getattr(monitor, "teacher_step", None)
    if not callable(publish):
        return
    try:
        publish(method=str(method), step=int(step), seed=seed, **info)
    except Exception as exc:                  # pragma: no cover - defensive
        print(f"WARNING: teacher telemetry was not published: {exc}")


def _privileged_velocity_teacher(environment, *, settings, trace_path=None,
                                 monitor=None):
    """Bind the training-only teacher to the environment's current live step."""
    return _PrivilegedVelocityTeacher(environment, settings=settings,
                                      trace_path=trace_path, monitor=monitor)


VISUAL_SERVO_TEACHER = "image_based_visual_servo_v1"
PRIVILEGED_VELOCITY_TEACHER = "privileged_relative_state_velocity_pd_v4"
# The non-learned control condition of the CICS2026 comparison. Named here
# rather than imported inline so the id the config declares, the id the arm
# manifest carries and the id the dispatch below tests are one constant.
PN_GUIDANCE_CONTROLLER = PN_GUIDANCE_METHOD
ANALYTIC_CONTROLLERS = (PRIVILEGED_VELOCITY_TEACHER, VISUAL_SERVO_TEACHER,
                        PN_GUIDANCE_CONTROLLER)


def _visual_servo_teacher_action(
        semantic, previous, velocity_limit, *, setpoint, dt, integral,
        tan_half=(1.0, .625), committed_already=False, image_rate=None,
        position_gain=.55, integral_gain=.20, damping_gain=.35,
        integral_limit=3.0, horizontal_speed_limit=.60,
        reference_scale=.06, range_gain_bounds=(.35, 2.2),
        rate_filter_s=.30, integral_leak_s=0.0, anti_windup=True,
        alignment_tolerance=.30, cone_widening=1.2,
        rate_tolerance=.60, flare_scale=.20, approach_scale=.12,
        approach_descent=.50, descent_rate=.30, flare_descent=.25,
        descent_floor=.12, climb_rate=.22, noise_std=.01, rng=None):
    """Return a velocity label computed from the image plane alone.

    Unlike ``_privileged_velocity_teacher_action`` this reads nothing the
    student cannot: the confidence-weighted keypoint centroid, its apparent
    RMS scale, and the visibility scalars -- all of which come out of the same
    frozen encoder the actor sees. There is no relative position, no relative
    velocity and no deck twist anywhere in it, so a policy cloned from these
    labels is being taught a mapping it can actually reproduce in deployment.

    Three things make an image servo work here rather than the naive version,
    which "commands zero at image centre, lets a moving UGV escape, and
    repeatedly crosses the target":

    * The setpoint is the nadir projection, not the image centre. A 60-degree
      camera sees the pad centred while the vehicle hovers a height-dependent
      distance *behind* the deck, so centring is a station-keeping point, not
      a landing point. ``nadir_image_setpoint`` is the constant the vehicle
      sees when it is directly above -- constant because a projected direction
      does not move with altitude.

    * The loop is type 1. Commanded velocity carries an integral of the image
      error, so the steady state that cancels a constantly-driving deck is the
      integrator's output rather than a standing position error. A pure
      proportional image servo has to *lag* to generate any chase velocity at
      all, which is the escape the privileged teacher was written to avoid.

      Both clamps therefore have to clear the deck's own top speed with room
      to spare, or the loop is saturated before it can correct anything: at a
      0.60 m/s limit against a deck configured for 0.60 m/s the whole budget
      goes on matching it, and the lateral error simply stops converging.
      Four separate tunings all stalled at 0.32-0.37 m for exactly that
      reason before the limits were opened up.

    * Image error is an angle; metres of it depend on range. Apparent scale is
      the only range signal available without estimating relative state, and
      it is monotone in inverse range, so the proportional path is scaled by
      ``reference_scale / raw_scale``: wide corrections while the deck is far
      and small ones in the flare, from one gain.

    The error is carried in tangent-of-angle units rather than in raw frame
    fractions. The frame is not square -- the column axis spans
    ``tan(hfov/2)`` and the row axis ``tan(vfov/2)``, here 1.0 against 0.625 --
    so the same metre of offset reads 1.6x larger down one axis than the
    other. Left in frame fractions the servo pulls off the true bearing and
    the alignment test means a different distance fore/aft than laterally.
    Multiplying each axis by its own half-angle tangent makes both
    ``offset / range``, which is isotropic and is what the range gain then
    converts back into metres per second.

    Descent is gated on what the image says rather than on altitude, and the
    gate is that bearing: ``alignment_tolerance`` is a cone, so it admits about
    1.3 m of offset at 5 m of range and about 0.2 m at 1 m. The funnel tightens
    on the way down without anything having to measure altitude. The cone has
    to clear what the horizontal loop actually holds, or the vehicle hovers
    the horizon out: the first three flights of the 2026-09-19 run tracked to
    0.76 m at 5 m -- a bearing of 0.18 -- and a cone of that size left the
    descent shut for all 300 steps.

    The descent rates have to fit the horizon. An episode is 30 simulated
    seconds and Table I starts the vehicle between 2 m and 8 m up, so the
    schedule has to spend well under that on the way down and still leave the
    alignment pass its time. The touchdown criteria admit 0.55 m/s, which is
    what bounds the rates rather than caution: the first tuned flight arrived
    correctly positioned, slow, and level, and timed out having never touched
    the deck. The rates below are ceilings reached only when the vehicle is
    centred -- see ``margin`` for why a flat rate destabilises the servo. The
    marker legitimately fills and then leaves a downward camera at the end of a
    correct flare, so loss of visibility commands a climb only while the
    target still looks small; past ``flare_scale`` the vehicle is committed.

    Returns ``(action, integral, committed, image_rate)``; the caller carries
    the integral, the commit latch and the filtered image rate between steps.
    """
    limit = np.asarray(velocity_limit, dtype=np.float64).reshape(-1)
    target = np.asarray(setpoint, dtype=np.float64).reshape(-1)
    carried = np.asarray(integral, dtype=np.float64).reshape(-1)
    if limit.shape != (3,) or target.shape != (2,) or carried.shape != (2,):
        raise ValueError("visual servo teacher expects 3-D limits and 2-D image state")
    if (not np.isfinite(limit).all() or np.any(limit <= 0.0)
            or not np.isfinite(target).all() or not np.isfinite(carried).all()):
        raise ValueError("visual servo teacher inputs must be finite with positive limits")
    step = float(dt)
    speed_limit = float(horizontal_speed_limit)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("visual servo teacher needs a positive control period")
    if not np.isfinite(speed_limit) or speed_limit <= 0.0:
        raise ValueError("visual servo teacher speed limit must be positive")

    half_angle = np.asarray(tan_half, dtype=np.float64).reshape(-1)
    if half_angle.shape != (2,) or not np.isfinite(half_angle).all() or np.any(
            half_angle <= 0.0):
        raise ValueError("visual servo teacher needs positive half-angle tangents")
    centroid = np.asarray(semantic.centroid_xy, dtype=np.float64).reshape(-1)
    error = (centroid - target) * half_angle
    # Image column moves with the pad's fore/aft position and image row with
    # its lateral position, and +row means the pad is to the vehicle's right,
    # which is -y in the body FLU frame the action is expressed in. Both signs
    # are pinned by ``pad_image_position`` rather than asserted here.
    plane_error = np.array([error[0], -error[1]])
    if previous is None:
        measured_rate = np.zeros(2)
    else:
        previous_centroid = np.asarray(
            previous.centroid_xy, dtype=np.float64).reshape(-1)
        rate = (centroid - previous_centroid) * half_angle / step
        measured_rate = np.array([rate[0], -rate[1]])
    # Filter the image rate. This is the change that made the servo land.
    #
    # The damping term is the one path where image motion reaches the command
    # directly, and a one-step centroid difference at dt = 0.1 s multiplies it
    # by ten. The motion it differentiates is not only the deck's: a
    # multirotor delivers a velocity command by tilting, and the camera is
    # bolted to the airframe, so every correction swings the image by roughly
    # atan(a / g) -- about 10 degrees against a 45-degree half width for
    # 0.6 m/s over 0.35 s, which is several times the alignment cone. Raw, the
    # damping term reads that swing as deck motion and commands against it,
    # which tilts the airframe further: positive feedback through the
    # airframe, and the hunting and climb-aways this teacher was retired for.
    #
    # Offline, on the configured deck (tools/visual_servo_offline.py, 160
    # flights per variant): the raw derivative lands 1% with a late-flight
    # lateral swing of 2.44 m; the same gains with this filter at 0.30 s land
    # 94% at 0.89 m. Nothing else in the loop had to move -- raising the
    # authority to 1.00-1.60 m/s, the obvious reading of "the deck is as fast
    # as the clamp", measured worse (74% at 1.20) because more authority is
    # more tilt.
    #
    # The caller hands back what this returns; ``rate_filter_s`` = 0 is the
    # raw difference this used to use.
    filter_seconds = max(float(rate_filter_s), 0.0)
    if image_rate is None or previous is None or filter_seconds <= 0.0:
        plane_rate = measured_rate
    else:
        prior = np.asarray(image_rate, dtype=np.float64).reshape(-1)
        if prior.shape != (2,) or not np.isfinite(prior).all():
            raise ValueError("visual servo teacher needs a finite 2-D image rate")
        plane_rate = prior + (measured_rate - prior) * (
            step / (filter_seconds + step))

    scale = max(float(semantic.raw_scale), 1e-3)
    low, high = (float(bound) for bound in range_gain_bounds)
    if not 0.0 < low <= high:
        raise ValueError("visual servo range gain bounds must be positive and ordered")
    range_gain = float(np.clip(float(reference_scale) / scale, low, high))
    metric_error = range_gain * plane_error

    # A centroid nobody can see is not evidence about where the deck is, and
    # integrating it winds the chase velocity up while the vehicle is blind.
    trustworthy = (float(semantic.visible_keypoint_fraction) >= 0.5
                   and float(semantic.visual_loss_risk) <= 0.0)

    # The command is formed from the integral the *previous* step ended with,
    # so the saturation below is the one this step actually commands and the
    # integrator can be corrected against it.
    unsaturated = (float(position_gain) * metric_error
                   + float(integral_gain) * carried
                   - float(damping_gain) * plane_rate * range_gain)
    speed = float(np.linalg.norm(unsaturated))
    target_xy = (unsaturated * (speed_limit / speed)
                 if speed > speed_limit else unsaturated)
    # Conditional integration ("clamping"): hold the integrator wherever the
    # command is already saturated and this error would drive it further in.
    #
    # Integrating through saturation is what made this loop hunt -- at the old
    # 0.20 x 3.00 the integral alone could command the entire 0.60 m/s clamp,
    # and the offline flights measured it sitting on its limit in 94% of the
    # flights against a 0.60 m/s deck, none of which landed. Back-calculation
    # was tried first and is wrong here: against a deck that keeps driving,
    # a saturated command is often the integrator doing exactly its job --
    # cancelling the deck -- and unwinding it lost 20 points of landing rate
    # against this rule at the same authority. Holding it costs nothing when
    # the saturation is legitimate and stops the accumulation when it is not.
    pushing = float(np.dot(metric_error, unsaturated)) > 0.0
    windup = bool(anti_windup) and speed > speed_limit and pushing
    if trustworthy and not windup:
        carried = np.clip(carried + metric_error * step,
                          -float(integral_limit), float(integral_limit))
    # An optional slow leak, so an integral wound up against a deck that has
    # since changed direction need not be integrated back out. Off by default:
    # the deck the integral is cancelling is usually still there.
    leak = max(float(integral_leak_s), 0.0)
    if trustworthy and leak > 0.0:
        carried = carried * max(0.0, 1.0 - step / leak)

    alignment = float(np.linalg.norm(plane_error))
    # ``apparent_target_scale`` is the saturated [0, 1] reading the rest of the
    # pipeline defines its landing phases on (see _landing_phase); raw_scale is
    # the unsaturated value the range gain needs. Mixing the two up puts every
    # threshold an order of magnitude out: a target 5 m away reads raw_scale
    # 0.04 and apparent_target_scale 0.05.
    apparent = float(semantic.apparent_target_scale)
    # Only the commit decision: past this the marker is expected to fill and
    # leave the frame, so losing it stops commanding a climb.
    #
    # It has to latch. Apparent scale is measured *from the keypoints*, so the
    # moment the marker leaves the frame it collapses to zero -- and that is
    # precisely the moment the flare depends on it. Tested fresh each step the
    # vehicle un-commits exactly when it should be committed: the sampled
    # trajectory came down to 0.385 m, lost the deck, read apparent 0 and
    # climbed back to 0.498 m and away. Once committed, this episode stays
    # committed.
    committed = bool(committed_already) or apparent >= float(flare_scale)
    # A cone fixed in angle keeps shrinking in metres all the way down, and
    # near the deck it ends up demanding better than the touchdown criterion
    # itself: 0.30 of a 0.36 m range is 0.11 m against a 0.35 m position gate.
    # The descent then stalls centimetres short on any small disturbance,
    # which is how the first tuned flight met every landing criterion except
    # contact and still timed out. Past the flare the cone widens so it never
    # asks for more than the success test does.
    # A cone fixed in angle keeps shrinking in metres all the way down, and
    # near the deck it ends up demanding better alignment than the touchdown
    # criterion itself does -- 0.30 of a 0.5 m range is 0.15 m against a
    # 0.35 m position gate -- so the descent stalls short of a landing that
    # would have counted. Widening it with apparent scale, which goes as
    # 1 / range, holds the tolerance it implies roughly constant in metres
    # over the last stretch: about 1.3 m at 5 m, 0.39 m at 1 m and 0.20 m at
    # the flare, all inside what the criteria ask for.
    cone = float(alignment_tolerance) * (1.0 + float(cone_widening) * apparent)
    # Descending is not free: the bearing to a pad that is not directly below
    # grows as the range shrinks, so coming down while only roughly aligned
    # drives the deck out of frame and hands the vehicle to the climb branch.
    # Flown at a flat 0.50 m/s that is exactly what happened -- seed 90000
    # ended 8.6 m out and climbing, against 0.32 m and descending at 0.35 m/s
    # from the same entry.
    #
    # So the rate is proportional to how much of the cone is left rather than
    # switched on at its edge. A centred vehicle gets the full schedule; one
    # at a third of the cone gets about half of it; one at the edge coasts
    # level while the horizontal loop catches up. The 1.2 makes the ramp reach
    # full rate slightly inside the cone instead of only at dead centre.
    margin = float(np.clip(1.2 * (1.0 - alignment / max(cone, 1e-6)), 0.0, 1.0))
    if trustworthy:
        # Without a floor the last stretch is an asymptote. Below about half a
        # metre the camera -- mounted 0.16 m under the airframe -- is close
        # enough to the deck that the bearing grows faster than the cone
        # widens, so the ramp above tends to zero while the vehicle is still
        # above the pad and simply hangs there. Flown open loop that settled
        # at 0.37 m and stopped. A floor makes it creep instead of stall, at a
        # rate small enough not to be the destabilising descent above.
        margin = max(margin, float(descent_floor))
    # Hold the descent while the image is moving fast. This reads the same
    # filtered rate the damping term does, so a single noisy frame -- or one
    # frame of the vehicle's own tilt -- no longer chops the descent off.
    if float(np.linalg.norm(plane_rate)) > float(rate_tolerance):
        margin = 0.0
    if committed:
        # In the flare the camera is a couple of handspans above the deck --
        # the mount sits 0.16 m below the airframe -- so the bearing to
        # anything not exactly underneath blows up and the marker leaves the
        # frame on its own. Holding the vehicle there until it aligns to a
        # cone that geometry no longer supports is the trap the privileged
        # teacher documents as well. Past the commit the flare rate runs
        # regardless: the pad fills the frame, so the position gate is already
        # satisfied by a wide margin.
        margin = 1.0
    if apparent < float(approach_scale):
        scheduled = float(approach_descent)
    elif not committed:
        scheduled = float(descent_rate)
    else:
        scheduled = float(flare_descent)
    if not trustworthy and not committed:
        target_vz = float(climb_rate)
    else:
        target_vz = -scheduled * margin

    # A velocity law, like the privileged teacher: the binding converts it to
    # the shared planar envelope. See ``_planar_from_velocity_action``.
    action = np.r_[np.r_[target_xy, target_vz] / limit, 0.0]
    if float(noise_std) > 0.0:
        generator = rng if rng is not None else np.random.default_rng()
        action = action + generator.normal(0.0, float(noise_std), 4)
    return np.clip(action, -.90, .90), carried, committed, plane_rate


def _visual_servo_teacher(environment, *, settings, monitor=None):
    """Bind the estimator-free image servo to this environment's camera.

    The integral and the previous centroid are per-episode state, so the
    binding resets them whenever the environment hands back a step index of
    zero rather than carrying one episode's chase into the next.

    ``monitor`` receives the same per-step diagnostics the privileged teacher
    publishes. They are read from the live step's simulator truth for the
    operator's display only; nothing here enters the servo's action.
    """
    method = str(settings.get("source_pipeline", "no_se_fixed"))
    flight = {"seed": None}
    camera = dict(getattr(environment.cfg.external, "landing_camera", None) or {})
    setpoint = nadir_image_setpoint(
        float(camera.get("horizontal_fov_deg", 90.0)),
        float(camera.get("pitch_down_deg", 60.0)))
    width, height = (int(value) for value in camera.get("resolution", (512, 320)))
    tan_half_h = math.tan(math.radians(
        float(camera.get("horizontal_fov_deg", 90.0))) / 2.0)
    tan_half = (tan_half_h, tan_half_h * height / width)
    state = {"integral": np.zeros(2), "previous": None, "committed": False,
             "image_rate": np.zeros(2)}

    def transform(step, policy_action, semantic, rng):
        if int(step) == 0:
            state["integral"] = np.zeros(2)
            state["previous"] = None
            state["committed"] = False
            state["image_rate"] = np.zeros(2)
        controller = environment.adapter.controller
        action, integral, committed, image_rate = _visual_servo_teacher_action(
            semantic, state["previous"],
            controller.velocity_limit_flu,
            setpoint=setpoint, dt=float(environment.cfg.sim.dt),
            integral=state["integral"], tan_half=tan_half,
            committed_already=state["committed"],
            image_rate=state["image_rate"],
            rate_filter_s=float(settings.get("rate_filter_s", .30)),
            integral_leak_s=float(settings.get("integral_leak_s", 0.0)),
            anti_windup=bool(settings.get("anti_windup", True)),
            position_gain=float(settings.get("position_gain", .55)),
            integral_gain=float(settings.get("integral_gain", .20)),
            damping_gain=float(settings.get("damping_gain", .35)),
            integral_limit=float(settings.get("integral_limit", 3.0)),
            horizontal_speed_limit=float(settings.get(
                "horizontal_speed_limit_m_s", .60)),
            reference_scale=float(settings.get("reference_scale", .06)),
            alignment_tolerance=float(settings.get("alignment_tolerance", .30)),
            cone_widening=float(settings.get("cone_widening", 1.2)),
            rate_tolerance=float(settings.get("rate_tolerance", .60)),
            approach_descent=float(settings.get("approach_descent_m_s", .50)),
            descent_rate=float(settings.get("descent_rate_m_s", .30)),
            flare_descent=float(settings.get("flare_descent_m_s", .25)),
            flare_scale=float(settings.get("flare_scale", .20)),
            descent_floor=float(settings.get("descent_floor", .12)),
            approach_scale=float(settings.get("approach_scale", .12)),
            noise_std=float(settings.get("noise_std", .01)),
            rng=rng)
        state["integral"] = integral
        state["previous"] = semantic
        state["committed"] = committed
        state["image_rate"] = image_rate
        planar = _planar_from_velocity_action(controller, action)
        current = environment.last_step
        if monitor is not None and current is not None:
            relative = np.asarray(current.critic.true_relative_state,
                                  dtype=float).reshape(-1)
            # ``action`` is the planar triple now, so the commanded vertical
            # rate is the controller's integrated setpoint rather than a
            # channel of the action.
            _publish_teacher_step(monitor, method=method, step=step,
                                  seed=flight.get("seed"), info={
                "altitude_m": max(0.0, -float(relative[2])),
                "lateral_error_m": float(np.linalg.norm(relative[:2])),
                "relative_speed_m_s": float(np.linalg.norm(relative[3:5])),
                "target_vz_m_s": float(
                    action[2] * controller.velocity_limit_flu[2]),
                "commanded_tilt_deg": float(np.degrees(
                    float(planar[2]) * controller.max_longitudinal_tilt
                    * controller.action_scale)),
                "visual_lost": bool(
                    float(semantic.visible_keypoint_fraction) < 0.5
                    or float(semantic.visual_loss_risk) > 0.0),
                "geometric_in_fov": bool(current.geometric_pad_center_in_fov),
                "battery_risk": float(semantic.battery_risk)})
        return planar
    transform.flight = flight
    return transform


def _pn_guidance_controller(environment, *, settings, monitor=None):
    """Bind the sagittal-plane PN guidance law to this environment's camera.

    This is the non-learned control condition. Like the retired image servo it
    reads only the frozen encoder's output; unlike it, it is the reference law
    of the reduced 2-D study the new methodology comes from, so the three arms
    here are the same three arms as there.

    The bearing rate, the range rate and the flare latch are per-episode state,
    so the binding resets them whenever the environment hands back a step index
    of zero rather than carrying one flight's geometry into the next.
    """
    method = str(settings.get("source_pipeline", "no_se_fixed"))
    flight = {"seed": None}
    camera = dict(getattr(environment.cfg.external, "landing_camera", None) or {})
    horizontal_fov = float(camera.get("horizontal_fov_deg", 90.0))
    setpoint = nadir_image_setpoint(
        horizontal_fov, float(camera.get("pitch_down_deg", 60.0)))
    controller = environment.adapter.controller
    guidance = PNGuidanceController(
        PNGuidanceConfig(
            navigation_gain=float(settings.get("pn_navigation_gain", 3.0)),
            approach_speed_m_s=float(settings.get("pn_approach_speed_m_s", .45)),
            approach_gain=float(settings.get("pn_approach_gain", .60)),
            closing_gain=float(settings.get("pn_closing_gain", 1.20)),
            vertical_gain=float(settings.get("pn_vertical_gain", 2.00)),
            climb_reference_m_s=float(settings.get("pn_climb_reference_m_s", .35)),
            reference_scale=float(settings.get("reference_scale", .06)),
            flare_scale=float(settings.get("flare_scale", .20)),
            flare_descent_m_s=float(settings.get("flare_descent_m_s", .25)),
            descent_floor_m_s=float(settings.get("descent_floor", .10)),
            noise_std=float(settings.get("pn_noise_std", 0.0)),
        ),
        nadir_column=float(setpoint[0]),
        tan_half_horizontal=math.tan(math.radians(horizontal_fov) / 2.0),
        max_longitudinal_acceleration_m_s2=float(
            controller.max_acceleration[0] * controller.action_scale),
        max_vertical_acceleration_m_s2=float(
            controller.max_acceleration[1] * controller.action_scale),
        max_longitudinal_tilt_rad=float(
            controller.max_longitudinal_tilt * controller.action_scale),
        dt=float(environment.cfg.sim.dt))
    state = PNGuidanceState()

    def transform(step, policy_action, semantic, rng):
        if int(step) == 0:
            state.reset()
            # The envelope scale is a curriculum property and can change
            # between flights; rebuild the law's limits from it rather than
            # from whatever they were when the binding was made.
            live = environment.adapter.controller
            guidance.max_longitudinal_acceleration = float(
                live.max_acceleration[0] * live.action_scale)
            guidance.max_vertical_acceleration = float(
                live.max_acceleration[1] * live.action_scale)
            guidance.max_longitudinal_tilt = float(
                live.max_longitudinal_tilt * live.action_scale)
        current = environment.last_step
        body_velocity = (np.zeros(3) if current is None
                         else np.asarray(current.actor.body_velocity, dtype=float))
        action = guidance.action(semantic, body_velocity, state, rng=rng)
        if monitor is not None and current is not None:
            relative = np.asarray(current.critic.true_relative_state,
                                  dtype=float).reshape(-1)
            diagnostics = state.diagnostics
            _publish_teacher_step(monitor, method=method, step=step,
                                  seed=flight.get("seed"), info={
                "altitude_m": max(0.0, -float(relative[2])),
                "lateral_error_m": float(np.linalg.norm(relative[:2])),
                "relative_speed_m_s": float(np.linalg.norm(relative[3:5])),
                "pn_mode": str(diagnostics.get("mode", "track")),
                "pn_bearing_deg": float(np.degrees(
                    diagnostics.get("bearing_rad", 0.0))),
                "pn_bearing_rate_deg_s": float(np.degrees(
                    diagnostics.get("bearing_rate_rad_s", 0.0))),
                "pn_range_m": float(diagnostics.get("range_m", 0.0)),
                "pn_closing_speed_m_s": float(
                    diagnostics.get("closing_speed_m_s", 0.0)),
                "visual_lost": bool(
                    float(semantic.visible_keypoint_fraction) < 0.5
                    or float(semantic.visual_loss_risk) > 0.0),
                "geometric_in_fov": bool(current.geometric_pad_center_in_fov),
                "battery_risk": float(semantic.battery_risk)})
        return action
    transform.flight = flight
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
            # Planar channels: push the approach off along track, drop harder
            # than the criteria allow, and tilt with the push. The lateral
            # excursion the retired version used has no channel left.
            action[0] = np.clip(action[0] + .34 * direction, -.90, .90)
            action[1] = min(float(action[1]), -.62)
            action[2] = np.clip(float(action[2]) + .20 * direction, -.90, .90)
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


# Keys that name things for a reader and change nothing the experiment does.
# Excluded from the configuration hash so a label edit cannot invalidate a
# machine-day of training; see the note at the hash itself.
_DISPLAY_ONLY_CONFIG_KEYS = frozenset({"arm_labels"})


def _baseline_arms(config) -> list[dict]:
    """Non-learned control conditions flown beside the PPO arms.

    These are controllers, not policies: no checkpoint, no training curve, no
    reward. They exist because "does the ontology term help a PPO policy" does
    not answer "is any of this better than a controller you could have
    written", and the second question has to be answered on the same deck, the
    same seeds and the same landing criteria or it is not an answer.

    ``controller`` selects the implementation and is validated here rather than
    at flight time: a typo in an arm id should stop the run before it spends a
    machine-hour, not produce an arm that silently never flies.
    """
    arms = []
    for entry in (config.get("baseline_arms") or []):
        method = str(entry["method"])
        controller = str(entry.get("controller", method))
        if controller not in ANALYTIC_CONTROLLERS:
            raise ValueError(
                f"baseline arm {method!r} names an unknown controller "
                f"{controller!r}; expected one of {ANALYTIC_CONTROLLERS}")
        arms.append({"method": method, "label": str(entry.get("label", method)),
                     "controller": controller, "learned": False})
    ids = [arm["method"] for arm in arms]
    if len(set(ids)) != len(ids):
        raise ValueError(f"baseline arms must have distinct ids, got {ids}")
    return arms


def _arm_manifest(pipelines, baseline_arms, labels) -> list[dict]:
    """Registration order for the dashboard and the RViz HUD.

    Learned arms first, then the control conditions. ``learned`` is what the
    dashboard keys its learning curves off, so a non-learned arm has no
    training series rather than an empty chart.

    ``ontology`` marks the arm the ontology takes part in -- the proposed one.
    The dashboard used to decide that by comparing the method id against a
    literal ``shin_se_onto_rgat_recovery``, which silently mislabels every
    other ontology arm the repository declares and would have to be edited
    again for the next. The pipeline spec already knows, so it is published
    rather than guessed. ``ontology_role`` says *how* it takes part, because
    the current method puts the graph in the observation and the retired one
    put a scalar in the reward, and a reader of the manifest should not have to
    infer which from the arm's name.
    """
    def role(spec):
        if spec.graph_state_enabled:
            return "state_representation"
        if spec.fov_risk_reward_enabled:
            return "additive_reward_term"
        if spec.use_direct_rgat_potential:
            return "potential_based_shaping"
        if spec.use_adaptive_reward_weights:
            return "adaptive_reward_weights"
        return None

    return ([{"method": str(name), "label": str(labels.get(name, name)),
              "learned": True,
              "ontology": role(get_pipeline(name)) is not None,
              "ontology_role": role(get_pipeline(name)),
              "graph_state_representation":
                  get_pipeline(name).graph_state_representation}
             for name in pipelines]
            + [{**dict(arm), "ontology": bool(dict(arm).get("ontology", False)),
                "ontology_role": None, "graph_state_representation": None}
               for arm in baseline_arms])


def _plan_with_baseline_arms(plan, baseline_arms) -> list[dict]:
    """Fly every control condition on exactly the seeds the PPO arms fly.

    The comparison is paired, so the baseline rows are generated from the
    plan's own (scenario, seed) pairs rather than re-derived: any change to how
    evaluation seeds are drawn moves all arms together or not at all.
    """
    if not baseline_arms:
        return list(plan)
    keys, seen = [], set()
    for row in plan:
        key = (row["scenario"], int(row["seed"]))
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return list(plan) + [
        {"method": arm["method"], "scenario": scenario, "seed": seed}
        for arm in baseline_arms for scenario, seed in keys]


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


def _training_pair_replicas(pipelines, pair_count: int,
                            replicate: int) -> dict[str, list[int]]:
    """Physical pairs each method collects episodes on, in pair order.

    One pair per method is the sequential case this started as. Beyond that the
    pairs are divided evenly and a method flies all of its own at once, which
    is the only way to spend a machine that runs several UAV/UGV pairs faster
    in total than it runs one (measured: 1 pair 0.888x real time, 2 pairs
    1.315x in total). The block per method stays contiguous so a method's
    replicas keep the same route phases and world offsets across a resume.
    """
    methods = list(pipelines)
    count = int(pair_count)
    if count < 1:
        raise ValueError("parallel training needs at least one physical pair")
    if count == 1:
        return {name: [0] for name in methods}
    if count % len(methods):
        raise ValueError(
            f"{count} physical pairs cannot be divided evenly between "
            f"{len(methods)} method(s); parallel training gives every method "
            "the same number of pairs so the arms stay budget-matched")
    replicas = count // len(methods)
    offset = int(replicate) % len(methods)
    return {
        name: [((method_index + offset) % len(methods)) * replicas + replica
               for replica in range(replicas)]
        for method_index, name in enumerate(methods)}


def _reward_design_pair_indices(independent_pipelines, training_pair_replicas,
                                training_pair_for, pair_count: int) -> list[int]:
    """Physical pairs left free for reward-design collection beside early arms.

    A measured pipeline flies *every* replica pair assigned to it, not only the
    primary pair that stands for it elsewhere. Counting just the primary as
    occupied handed the FOV-risk collector a pair the baseline PPO arm was
    already flying. Two bridges then bound one local UDP port (SO_REUSEADDR
    admits the second bind) and the kernel delivered each gateway reply to one
    of them: the other saw only hello and state timeouts, and every timeout
    rebuilt the shared simulator for all four pairs -- 26 rebuilds in four
    hours on 2026-09-21, the arm dying each run before the comparison began.

    Since the stages were separated (2026-09-22) no arm is training while the
    reward design collects, so ``independent_pipelines`` is normally empty and
    this returns every pair -- which is the point: the collection gets the
    whole stage. It is still called rather than assumed, so that reinstating
    any early-training arm keeps the collector off its pairs automatically
    instead of silently re-creating the double booking.
    """
    occupied: set[int] = set()
    for name in independent_pipelines:
        occupied.update(int(index) for index in training_pair_replicas.get(
            name, [training_pair_for[name]]))
    free = [index for index in range(int(pair_count)) if index not in occupied]
    if not free:
        raise RuntimeError(
            "parallel reward design needs one unoccupied physical pair; pairs "
            f"{sorted(occupied)} are all flown by {list(independent_pipelines)}")
    return free


def _balanced_training_pair_assignment(pipelines, pair_count: int,
                                       replicate: int) -> tuple[dict, list]:
    """Counterbalance method-to-route-phase assignment across replicates.

    Returns each method's *primary* pair -- the one that stands for it outside
    the training loop, where a single pair is what is wanted: reward-design
    collection, checkpoint validation and per-pair monitors. The training loop
    itself uses :func:`_training_pair_replicas`.
    """
    methods = list(pipelines)
    count = int(pair_count)
    replicas = _training_pair_replicas(methods, count, replicate)
    assignment = {name: indices[0] for name, indices in replicas.items()}
    if count == 1:
        return assignment, methods[:1]
    pair_methods: list[str | None] = [None] * count
    for name, indices in replicas.items():
        for index in indices:
            pair_methods[index] = name
    return assignment, [name for name in pair_methods if name is not None]


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


def behavior_cloning_settings(config):
    """The behavior-cloning block, wherever the profile declares it.

    It began life under ``seminar_fast`` because only the deadline profile
    warmed a policy up. The primary experiment needs it for the same reason
    the seminar one did -- an unclonded policy spends its first episodes
    diverging, and on this stack a diverging episode ends tens of metres from
    a deck that is still driving -- so a top-level ``behavior_cloning`` block
    is read first and the old location kept working underneath it.
    """
    settings = dict(config.get("behavior_cloning") or {})
    if settings:
        return settings
    return dict((config.get("seminar_fast") or {}).get("behavior_cloning") or {})


# Settings a demonstration flight actually depends on. Anything else in the
# behavior_cloning block (epochs, learning rate, post_log_std, ...) shapes how
# the clone is *trained* from the flights and must not invalidate them.
_DEMONSTRATION_FLIGHT_KEYS = (
    "teacher", "curriculum", "position_gain", "velocity_gain",
    "horizontal_speed_limit_m_s", "energy_urgency", "urgency_alignment_m",
    "visual_loss_climb_risk", "visual_loss_climb_fraction",
    "visual_loss_climb_source", "pd_descent_high_m_s", "pd_descent_mid_m_s",
    "pd_descent_flare_m_s", "pd_approach_descent_m_s",
    "pd_approach_min_altitude_m", "pd_approach_lateral_m", "horizon_steps",
    "pd_climb_ceiling_m", "pd_climb_lateral_m", "pd_climb_altitude_m",
    "pd_recovery_descent_m_s", "pd_recovery_altitude_m",
    "pd_horizontal_speed_limit_m_s",
    "noise_std",
    "integral_gain", "damping_gain", "integral_limit", "reference_scale",
    "alignment_tolerance", "cone_widening", "rate_tolerance", "flare_scale",
    "approach_scale", "source_pipeline", "scenario", "scenarios",
    # The image-rate filter and the anti-windup rule change what the teacher
    # flies, not merely how long it is given to fly it, so demonstrations made
    # without them are a different set.
    "rate_filter_s", "integral_leak_s", "anti_windup")


# Wall-clock and retry budgets for the bridge and the shared stack. They
# decide when a flight is abandoned as infrastructure failure, never how a
# completed flight was flown, so a stored demonstration means the same thing
# under any of them. Keying on them re-flew the whole teacher set three times
# on 2026-09-21 while the budgets were being tuned.
_DEMONSTRATION_BUDGET_KEYS = frozenset({
    "gateway_timeout_s", "setup_timeout_s", "reset_recoveries",
    "entry_timeout_s"})


def demonstration_fingerprint(config, settings, *, cfg, system=None) -> str:
    """What a stored teacher flight means, for deciding reuse across runs.

    The teacher and its gains, the entry curriculum, the seed range, the
    control envelope and horizon, and the deck/battery/benchmark profile the
    flight was made against. Not the experiment's configuration hash: the
    2026-09-20 encoder redesign changed only perception settings, and keying
    the demonstrations on the whole hash threw away 26 teacher attempts that
    could have been re-embedded from their retained frames. Transport budgets
    (:data:`_DEMONSTRATION_BUDGET_KEYS`) are left out for the same reason.
    """
    from ontology_rgat.datastore import data_fingerprint

    system = dict(system or {})
    benchmark = system.get("benchmark")
    if isinstance(benchmark, dict):
        benchmark = {key: value for key, value in benchmark.items()
                     if key not in _DEMONSTRATION_BUDGET_KEYS}
    parts = {
        "flight_settings": {key: settings[key] for key in _DEMONSTRATION_FLIGHT_KEYS
                            if key in settings},
        "seed_start": int((config.get("seeds") or {}).get(
            "behavior_cloning_start", 90000)),
        "control": config.get("control"),
        "horizon_steps": int(cfg.sim.max_steps),
        "system": {"pad": system.get("pad"), "battery": system.get("battery"),
                   "benchmark": benchmark,
                   "domain_randomization": system.get("domain_randomization")},
        "camera": dict((dict(system.get("vision") or {})).get("camera") or {}),
    }
    return data_fingerprint(parts)


def _prepare_fast_demonstrations(*, cfg, camera, config, config_hash,
                                 keypoint_pretraining, results_dir, device,
                                 model_seed, monitor, system=None,
                                 parallel_contexts=None, flight_allowed=True):
    """Collect successful teacher flights once and store compact actor inputs.

    ``parallel_contexts`` gives the stage every idle UAV/UGV pair instead of
    the one it started on. The warm start runs before any PPO worker exists,
    so at that point every spawned pair is free, and a teacher flight is a
    flight like any other: the measured stage throughput is the same 4.96x of
    simulated seconds per wall second four pairs give the training loop.
    Nothing about the stored set depends on how many pairs flew it -- the
    seeds, their order and the per-episode RNG are unchanged, and the
    demonstration fingerprint deliberately does not mention the pair count, so
    a set collected on four pairs is reused by a one-pair run and back.
    """
    settings = behavior_cloning_settings(config)
    if not bool(settings.get("enabled", False)):
        return None
    collection_contexts = list(parallel_contexts or ({
        "cfg": cfg, "camera": camera, "monitor": monitor,
    },))
    if not collection_contexts:
        raise ValueError("teacher demonstrations need at least one live pair")
    for context in collection_contexts:
        if not all(key in context for key in ("cfg", "camera", "monitor")):
            raise ValueError(
                "each demonstration context needs cfg/camera/monitor")
    # A cap for the operator who wants the spare pairs left alone; the default
    # is every pair handed in.
    pair_limit = int(settings.get("parallel_pairs", 0) or 0)
    if pair_limit > 0:
        collection_contexts = collection_contexts[:pair_limit]
    # Physical pairs this stage flies on, for the dashboard's warm-start panel.
    collection_pair_indices = [
        int(context["cfg"].external.get("pair_index", position))
        for position, context in enumerate(collection_contexts)]
    required = max(1, int(settings.get("successful_episodes", 6)))
    maximum_flights = max(required, int(settings.get("max_attempts", 12)))
    maximum_infrastructure_skips = max(0, int(settings.get(
        "max_infrastructure_skips", maximum_flights)))
    maximum_seed_candidates = maximum_flights + maximum_infrastructure_skips
    source_pipeline = str(settings.get("source_pipeline", "no_se_fixed"))
    if get_pipeline(source_pipeline).state_estimation_enabled:
        raise ValueError("the behavior-teacher encoder source must be estimator-free")
    teacher_id = str(settings.get("teacher", PRIVILEGED_VELOCITY_TEACHER))
    if teacher_id not in (PRIVILEGED_VELOCITY_TEACHER, VISUAL_SERVO_TEACHER):
        raise ValueError(f"unknown behavior teacher: {teacher_id}")
    # Deck motion the teacher demonstrates against. ``scenarios`` flies a
    # rotation -- one deck per seed, so a given seed always flies the same one
    # however the flights are batched or resumed -- and ``scenario`` the single
    # deck this started as.
    #
    # The rotation exists because the student is cloned from these flights and
    # then trained on ``training.scenarios``. With the demonstrations on the
    # random walk and training on the six route decks, the clone had never
    # chased a deck that drives away: 2026-09-21 the baseline arm sat 9.3 m
    # behind, the pad was in frame for 47% of steps against 83% on the walk,
    # the estimator collapsed to a 8.7 m position RMSE and the health gate
    # stopped the run at episode 48.
    demonstration_scenarios = [
        str(name) for name in (settings.get("scenarios")
                               or [settings.get("scenario",
                                                COLLECTION_BASE_SCENARIO)])]
    if not demonstration_scenarios:
        raise ValueError("the behaviour-cloning teacher needs a deck to fly")
    demonstration_scenario = " · ".join(demonstration_scenarios)
    fingerprint = demonstration_fingerprint(config, settings, cfg=cfg, system=system)
    artifact_path = (Path(results_dir) / "models/shared"
                     / f"teacher_demonstrations_{fingerprint[:12]}.pt")
    attempts_path = (Path(results_dir) / "training"
                     / f"teacher_attempts_{fingerprint[:12]}.csv")
    # Sets written before provenance keying are named after the experiment
    # hash; one made under this exact configuration is still reusable.
    legacy_artifact = (Path(results_dir) / "models/shared"
                       / f"teacher_demonstrations_{config_hash[:12]}.pt")
    legacy_attempts = (Path(results_dir) / "training"
                       / f"teacher_attempts_{config_hash[:12]}.csv")
    if not artifact_path.is_file() and legacy_artifact.is_file():
        artifact_path = legacy_artifact
    if not attempts_path.is_file() and legacy_attempts.is_file():
        attempts_path = legacy_attempts
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
    teacher_model = None
    if artifact_path.is_file():
        # Built before loading, and outside the compatibility guard below, so
        # a set encoded by an earlier encoder can be re-embedded from its
        # retained frames instead of re-flown -- and so a model construction
        # error surfaces as what it is rather than as "incompatible
        # demonstrations" that reset the attempts ledger.
        torch.manual_seed(int(model_seed))
        teacher_model = _build_model(
            config, device, keypoint_pretraining, pipeline=source_pipeline)
        try:
            payload = load_encoded_demonstrations(
                artifact_path, config_hash=config_hash,
                encoder_sha256=encoder_sha, model=teacher_model,
                batch_size=int(settings.get("encoding_batch_size", 64)),
                demonstration_fingerprint=fingerprint)
            if (payload.get("re_embedded_from_encoder_sha256")
                    or payload.get("demonstration_fingerprint") != fingerprint):
                payload = save_encoded_demonstrations(
                    artifact_path, payload["dataset"], config_hash=config_hash,
                    encoder_sha256=encoder_sha,
                    attempted_seeds=payload.get("attempted_seeds", ()),
                    environment_steps=int(payload.get("environment_steps", 0)),
                    teacher=str(payload.get("teacher", teacher_id)),
                    demonstration_fingerprint=fingerprint)
                if payload.get("re_embedded_from_encoder_sha256"):
                    print(
                        "Re-embedded the stored training-teacher demonstrations "
                        f"({payload['transitions']} transitions) with the current "
                        "keypoint encoder from their retained frames; no teacher "
                        "flight is repeated.")
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

    def publish_progress() -> None:
        """Dashboard view of the warm start; never allowed to end the stage."""
        publish = getattr(monitor, "demonstration_progress", None)
        if not callable(publish):
            return
        try:
            publish(method=source_pipeline, required=int(required),
                    accepted=int(successes), flights=int(flight_attempts),
                    max_flights=int(maximum_flights),
                    skips=int(len(attempted_seeds) - flight_attempts),
                    teacher=teacher_id, scenario=demonstration_scenario,
                    fingerprint=fingerprint[:12], attempts=attempts,
                    collection_pairs=collection_pair_indices)
        except Exception as exc:              # pragma: no cover - defensive
            print(f"WARNING: demonstration progress was not published: {exc}")

    publish_progress()
    if successes < required and flight_attempts < maximum_flights:
        _require_flight(
            flight_allowed, "the shared teacher demonstration set",
            shortfall=f"{successes}/{required} successful flights stored")
        if teacher_model is None:
            torch.manual_seed(int(model_seed))
            teacher_model = _build_model(
                config, device, keypoint_pretraining, pipeline=source_pipeline)
        seed0 = int((config.get("seeds") or {}).get(
            "behavior_cloning_start", 90000))
        # Every pair that flies the warm start says so on its own panel; the
        # accepted/required progress stays on the stage monitor.
        stage_detail = (
            f"successful real Isaac/PX4 flights {successes}/{required} · "
            f"{len(collection_contexts)} pair"
            f"{'s' if len(collection_contexts) > 1 else ''}")
        for stage_monitor in dict.fromkeys(
                [monitor, *(context["monitor"] for context in collection_contexts)]):
            stage_monitor.stage(
                "training-only teacher demonstrations", stage_detail)
        # Every pair the stage flies gets the same shortened entry budget. The
        # contexts hold distinct Config objects (``_pair_live_config`` derives
        # by deep copy), so each has to be shortened and restored on its own.
        restore_budgets = []
        for context in collection_contexts:
            live = context["cfg"]
            restore_budgets.append((live, {
                "entry_timeout": float(live.external.entry_timeout),
                "entry_sim_budget": float(live.external.entry_sim_budget),
                "reset_recoveries": int(live.external.reset_recoveries),
                "episode_recoveries": int(live.external.get(
                    "episode_recoveries", live.external.reset_recoveries)),
            }))
            live.external.entry_timeout = min(
                float(live.external.entry_timeout), 120.0)
            # Fail a demonstration flight fast rather than spending the whole entry
            # budget on it: this pass only needs successful teacher episodes and
            # skips the rest. Shorten the budget the gate is judged on, not just
            # its hang guard, or the shortening does nothing.
            live.external.entry_sim_budget = min(
                float(live.external.entry_sim_budget), 25.0)
            live.external.reset_recoveries = 0
            live.external.episode_recoveries = 0
        # Demonstration flights may run longer than a PPO episode: the clone
        # learns from transitions, and a landing at 38 s is a landing.
        teacher_horizon = int(settings.get("horizon_steps", cfg.sim.max_steps))
        with ExitStack() as environment_stack:
            environments = [environment_stack.enter_context(
                LiveShinEnvironment(context["cfg"], context["camera"],
                                    horizon_steps=teacher_horizon))
                for context in collection_contexts]

            def build_teacher(index, environment):
                """One teacher per pair: the servo's integral, the previous
                centroid and the PD's trace file are all per-flight state."""
                local_monitor = collection_contexts[index]["monitor"]
                if teacher_id == PN_GUIDANCE_CONTROLLER:
                    return _pn_guidance_controller(
                        environment, settings=settings, monitor=local_monitor)
                if teacher_id == VISUAL_SERVO_TEACHER:
                    return _visual_servo_teacher(
                        environment, settings=settings, monitor=local_monitor)
                name = (f"teacher_trace_{fingerprint[:12]}.jsonl"
                        if len(collection_contexts) == 1 else
                        f"teacher_trace_{fingerprint[:12]}_pair{index}.jsonl")
                return _privileged_velocity_teacher(
                    environment, settings=settings,
                    trace_path=Path(results_dir) / "training" / name,
                    monitor=local_monitor)

            teachers = [build_teacher(index, environment)
                        for index, environment in enumerate(environments)]
            # Actor inference is read-only, but separate modules avoid any
            # accidental recurrent state sharing between worker threads. The
            # encoding below runs on the main thread against ``teacher_model``.
            worker_models = [teacher_model] + [deepcopy(teacher_model)
                                               for _ in environments[1:]]
            for worker_model in worker_models:
                for module in worker_model.modules():
                    flatten = getattr(module, "flatten_parameters", None)
                    if callable(flatten):
                        flatten()

            def scenario_for(seed) -> str:
                """The deck this seed flies, whatever batched it."""
                index = (int(seed) - seed0) % len(demonstration_scenarios)
                return demonstration_scenarios[index]

            def fly(worker_index, seed):
                """One demonstration flight on one pair, in its own thread."""
                teacher = teachers[worker_index]
                if hasattr(teacher, "seed"):
                    teacher.seed = int(seed)
                if isinstance(getattr(teacher, "flight", None), dict):
                    teacher.flight["seed"] = int(seed)
                try:
                    rows, metric = collect_episode_resilient(
                        environments[worker_index], worker_models[worker_index],
                        source_pipeline, seed,
                        curriculum=float(settings.get("curriculum", 1.0)),
                        deterministic=True, scenario=scenario_for(seed),
                        monitor=collection_contexts[worker_index]["monitor"],
                        phase="training-only teacher demonstration",
                        action_transform=teacher)
                except BridgeError as exc:
                    # One seeded entry can fail PX4 preflight deterministically.
                    # It is infrastructure downtime, not an RL failure: record
                    # no transition/label, advance to the next seed, and give
                    # that seed a freshly owned stack. The recovery itself is
                    # deferred to the main thread -- rebuilding the shared
                    # simulator under a sibling that is still flying would take
                    # its episode with it.
                    return worker_index, int(seed), None, None, exc
                return worker_index, int(seed), rows, metric, None

            already_flown = set(attempted_seeds)
            candidates = [seed0 + attempt
                          for attempt in range(maximum_seed_candidates)
                          if seed0 + attempt not in already_flown]
            cursor = 0
            while (cursor < len(candidates) and successes < required
                   and flight_attempts < maximum_flights):
                # Never launch more flights than the attempt budget still pays
                # for, and never more than there are pairs to fly them on.
                width = min(len(environments), len(candidates) - cursor,
                            maximum_flights - flight_attempts)
                batch = candidates[cursor:cursor + width]
                cursor += width
                if len(batch) == 1:
                    completed = [fly(0, batch[0])]
                else:
                    with ThreadPoolExecutor(
                            max_workers=len(batch),
                            thread_name_prefix="teacher-pair") as executor:
                        futures = [executor.submit(fly, index, seed)
                                   for index, seed in enumerate(batch)]
                        completed = [future.result() for future in futures]
                # Persist only on the main thread and in seed order, so the
                # stored set and its episode ids are the ones the same seeds
                # produce sequentially however many pairs flew them.
                degraded = []
                for worker_index, seed, rows, metric, failure in sorted(
                        completed, key=lambda result: result[1]):
                    if failure is not None:
                        attempted_seeds.append(seed)
                        attempts.append({
                            "method": "privileged_teacher",
                            "pipeline": "shared_warm_start",
                            "episode": len(attempted_seeds),
                            "seed": seed,
                            "scenario": scenario_for(seed),
                            "status": "infrastructure_failure",
                            "accepted_for_cloning": 0.0,
                            "paper_success": 0.0,
                            "strict_success": 0.0,
                            "steps": 0,
                            "teacher": teacher_id,
                            "config_hash": config_hash,
                            "physical_pair_index": int(worker_index),
                            "infrastructure_error": str(failure),
                        })
                        _write_csv(attempts_path, attempts)
                        print(
                            f"training teacher seed {seed} skipped after "
                            f"PX4/Isaac reset failure on pair {worker_index}: "
                            f"{failure}")
                        publish_progress()
                        degraded.append(worker_index)
                        continue
                    attempted_seeds.append(seed)
                    flight_attempts += 1
                    environment_steps += int(metric["steps"])
                    metric.update({
                        "method": ("visual_servo_teacher"
                                   if teacher_id == VISUAL_SERVO_TEACHER
                                   else "privileged_teacher"),
                        "pipeline": "shared_warm_start",
                        "episode": len(attempted_seeds),
                        "scenario": scenario_for(seed),
                        "accepted_for_cloning": float(metric["paper_success"]),
                        "teacher": teacher_id,
                        "config_hash": config_hash,
                        "physical_pair_index": int(worker_index),
                        "teacher_information": (
                            "estimator-free image-plane servo on the frozen keypoint "
                            "encoder's centroid and apparent scale; no relative state "
                            "of any kind enters the action labels"
                            if teacher_id == VISUAL_SERVO_TEACHER else
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
                            teacher=teacher_id,
                            demonstration_fingerprint=fingerprint)
                    print(
                        f"training teacher flight {flight_attempts}/{maximum_flights} "
                        f"pair {worker_index} seed {seed} "
                        f"success={int(metric['paper_success'])} "
                        f"accepted={successes}/{required}")
                    publish_progress()
                # A batch is joined before any rebuild, so the pair that failed
                # is reconnected while nothing else is flying. Sibling workers
                # that failed on the same degraded stack share the one restart:
                # LiveShinEnvironment.recover_infrastructure cycles by stack
                # generation, so the later callers only reconnect.
                for worker_index in degraded:
                    environments[worker_index].recover_infrastructure()
            for teacher in teachers:
                if hasattr(teacher, "close"):
                    teacher.close()
        for live, original in restore_budgets:
            live.external.entry_timeout = original["entry_timeout"]
            live.external.entry_sim_budget = original["entry_sim_budget"]
            live.external.reset_recoveries = original["reset_recoveries"]
            live.external.episode_recoveries = original["episode_recoveries"]
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
                           source_pipeline="no_se", flight_allowed=True,
                           scenarios=None):
    design = dict(config.get("rgat_design") or {})
    # One deck per episode, rotated by episode index, exactly as the FOV-risk
    # and adaptive collections do. This was a hardcoded ``training_random_walk``
    # until 2026-09-22 -- a deck that never leaves the vehicle's frame -- so the
    # semantic outcome model was fitted on trajectories the policy it shapes
    # never flies. The caller passes the run's decks; the profile can still
    # name its own under ``rgat_design.scenarios``.
    design_scenarios = tuple(
        str(name) for name in (design.get("scenarios") or scenarios
                               or (COLLECTION_BASE_SCENARIO,)))
    if not design_scenarios:
        raise ValueError("semantic R-GAT collection needs a deck to fly")
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
        "scenario_cycle": list(design_scenarios),
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
        _require_flight(
            flight_allowed, "the semantic R-GAT reward-design rollouts",
            shortfall=f"{len(completed)}/{count} episodes stored")
        monitor.stage("R-GAT data", "estimator-free semantic behavior mixture")
        with LiveShinEnvironment(
                cfg, camera, horizon_steps=int(cfg.sim.max_steps)) as environment:
            for episode, seed in pending:
                if requirements_met(manifest, episode_rows):
                    break
                rows, metric = collect_episode_resilient(
                    environment, model, source_pipeline, seed, curriculum=1.0,
                    deterministic=False, gamma=gamma,
                    scenario=design_scenarios[
                        (episode - 1) % len(design_scenarios)],
                    monitor=monitor,
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


def _fov_episode_payload(samples) -> dict:
    """One episode's collection, in the two arrays the labeller needs."""
    return {
        "graph_X": np.stack([np.asarray(sample["graph_X"], dtype=np.float32)
                             for sample in samples]),
        "geometric_in_fov": np.asarray(
            [bool(sample["geometric_in_fov"]) for sample in samples], dtype=bool),
    }


def _fov_episode_from_payload(payload, *, episode_id: int, seed: int) -> dict:
    graphs = np.asarray(payload["graph_X"], dtype=np.float32)
    visible = np.asarray(payload["geometric_in_fov"], dtype=bool)
    if graphs.shape[0] != visible.shape[0]:
        raise ValueError("stored FOV-risk episode has mismatched arrays")
    return {
        "episode_id": int(episode_id), "seed": int(seed),
        "samples": [{"graph_X": graphs[index],
                     "geometric_in_fov": bool(visible[index])}
                    for index in range(graphs.shape[0])],
    }


def peer_training_health(futures):
    """Raise a concurrently trained arm's failure now, not at the join.

    ``training_futures`` is submitted before the reward-design stages and
    joined only after them, so an arm that dies in its first minutes leaves
    its exception sitting unread inside the future while the run carries on
    collecting data for a comparison that can no longer be made. On the
    2026-09-18 run that was twelve hours and twenty-eight simulator rebuilds
    spent after the baseline PPO arm had already stopped -- and the reward
    design was flown by an untrained policy the whole time, because the
    checkpoint it was supposed to freeze never advanced past two episodes.

    The returned callable is cheap enough to run once per collected episode.
    An arm that finished *successfully* is not a failure; only a future that
    completed with an exception stops the run.

    Separating the stages (2026-09-22) removed the overlap this guards: no PPO
    is submitted until every dataset is collected, so during collection the
    future map is empty and this is a no-op. It is still wired in, because the
    check costs nothing and the failure it names costs twelve hours.
    """
    def check() -> None:
        for name, future in futures.items():
            if not future.done() or future.cancelled():
                continue
            error = future.exception()
            if error is not None:
                raise RuntimeError(
                    f"the {name} PPO arm stopped early, so nothing flown "
                    "after it can complete the paired comparison. Its own "
                    "failure is the cause below."
                ) from error
    return check


def _collect_fov_risk_data(*, cfg, camera, model, config, config_hash,
                           checkpoint_path, results_dir, mode, monitor,
                           episodes_override=None, max_episodes_override=None,
                           source_pipeline="shin_se_fixed", system=None,
                           datastore=None, keypoint_implementation=None,
                           validation_fraction=0.2,
                           scenarios=(COLLECTION_BASE_SCENARIO,),
                           peer_health=None, parallel_contexts=None,
                           flight_allowed=True):
    """Collect same-domain trajectories and label future visibility offline.

    With a ``datastore`` the episodes every previous run flew under the same
    data fingerprint are reused, and only the shortfall is flown. The labels
    are recomputed here from the stored visibility sequence rather than
    stored, so a change to the horizon re-derives them instead of reusing a
    target that no longer means what it says.

    ``parallel_contexts`` hands the collection every live pair instead of the
    one it started on, exactly as the adaptive and demonstration stages do.
    This used to be impossible: the collection ran beside the fixed-reward
    arms' PPO and the only free pair was whatever the training assignment left
    over. In the collection stage no PPO is running, so every pair is free,
    and a four-pair stage collects at four times the rate. Nothing about the
    resulting dataset depends on how many pairs flew it -- a seed decides its
    behaviour variant, its deck and its train/validation side, and episodes are
    appended in seed order however the batch completed.
    """
    design = dict(config.get("fov_risk_design") or {})
    risk = dict(config.get("fov_risk") or {})
    # ``fov_risk_design.scenarios`` overrides the run-wide deck list the caller
    # hands in (``training.scenarios``), which in turn overrides the base deck.
    # The result is part of the datastore fingerprint below, so this is also
    # what decides whether an accumulation may be extended.
    scenarios = tuple(str(name) for name in (design.get("scenarios")
                                             or scenarios
                                             or (COLLECTION_BASE_SCENARIO,)))
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
    fingerprint = None
    stored = []
    if datastore is not None:
        if system is None:
            raise ValueError(
                "the FOV-risk datastore needs the merged system profile to "
                "fingerprint what its episodes mean")
        fingerprint = fov_risk_data_fingerprint(
            system, prediction_steps=prediction_steps, control_hz=control_hz,
            scenarios=scenarios,
            keypoint_implementation=keypoint_implementation)
        stored = datastore.episodes(KIND_FOV_RISK, fingerprint)
        if stored:
            summary = datastore.summary(
                KIND_FOV_RISK, fingerprint,
                validation_fraction=validation_fraction)
            print(
                f"Reusing {summary['episodes']} accumulated FOV-risk episodes "
                f"({summary['samples']} samples, {summary['environment_steps']} "
                f"environment steps) from {len(summary['runs'])} run(s); "
                f"fingerprint {fingerprint[:12]}.")
    elif dataset_path.is_file() and dataset_path.with_suffix(".manifest.json").is_file():
        # Without an accumulation the run's own directory is the only cache.
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
                return cached, cached_manifest, dataset_path, steps, {}
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            print(f"Ignoring incompatible FOV-risk rollout cache: {exc}")

    seed0 = int((config.get("seeds") or {}).get("fov_risk_dataset_start", 70000))
    episodes = [_fov_episode_from_payload(record.payload(), episode_id=index,
                                          seed=record.seed)
                for index, record in enumerate(stored, start=1)]
    flown_steps = 0
    total_steps = sum(len(episode["samples"]) for episode in episodes)
    loss_episodes = 0
    checkpoint_sha = (
        _sha256_file(checkpoint_path) if checkpoint_path is not None
        and Path(checkpoint_path).is_file() else None)
    dataset = (build_fov_risk_dataset(episodes, prediction_steps=prediction_steps)
               if episodes else None)
    covered = bool(dataset is not None and _fov_target_coverage(dataset))
    collection_contexts = list(parallel_contexts or ({
        "cfg": cfg, "camera": camera, "monitor": monitor,
    },))
    if not collection_contexts:
        raise ValueError("FOV-risk data needs at least one live pair")
    for context in collection_contexts:
        if not all(key in context for key in ("cfg", "camera", "monitor")):
            raise ValueError("each FOV-risk collection context needs cfg/camera/monitor")
    if len(episodes) < minimum or not covered:
        _require_flight(
            flight_allowed, "the FOV-risk reward-design rollouts",
            shortfall=(f"{len(episodes)}/{minimum} episodes stored"
                       if len(episodes) < minimum else
                       "both future-visibility target regimes are not covered"))
        used_seeds = {int(episode["seed"]) for episode in episodes}
        for context in collection_contexts:
            context["monitor"].stage(
                "FOV-risk data",
                f"visual-only graph · future visibility labels · "
                f"{len(collection_contexts)} pair(s)")
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
                for module in getattr(worker_model, "modules", lambda: ())():
                    flatten = getattr(module, "flatten_parameters", None)
                    if callable(flatten):
                        flatten()

            def collect_one(worker_index, item):
                episode_id, seed, variant = item
                rows, metric = collect_episode_resilient(
                    environments[worker_index], worker_models[worker_index],
                    source_pipeline, seed, curriculum=1.0,
                    deterministic=False,
                    scenario=scenarios[(episode_id - 1) % len(scenarios)],
                    monitor=collection_contexts[worker_index]["monitor"],
                    phase="FOV-risk offline data",
                    action_transform=_behavior_transform(variant))
                return episode_id, seed, variant, rows, metric

            offset = 0
            while len(episodes) < maximum:
                # Before flying anything else, make sure there is still an arm
                # left to compare against: with --stage all this may still run
                # next to a PPO worker whose exception nothing reads until long
                # after. In the collection stage there is no peer and this is
                # a no-op.
                if peer_health is not None:
                    peer_health()
                # Allocate the whole batch's seeds up front, in order, so the
                # episode a seed becomes -- and therefore its behaviour
                # variant, its deck and its train/validation side -- does not
                # depend on which pair happened to land first.
                batch = []
                while (len(batch) < len(environments)
                       and len(episodes) + len(batch) < maximum):
                    while (seed0 + offset) in used_seeds:
                        offset += 1
                    seed = seed0 + offset
                    offset += 1
                    used_seeds.add(seed)
                    episode_id = len(episodes) + len(batch) + 1
                    # The behaviour cycle continues across runs so an
                    # accumulation keeps the three transforms balanced instead
                    # of re-collecting the same one every time.
                    batch.append((episode_id, seed, (episode_id - 1) % 3))
                if not batch:
                    break
                if len(batch) == 1:
                    completed_batch = [collect_one(0, batch[0])]
                else:
                    with ThreadPoolExecutor(
                            max_workers=len(batch),
                            thread_name_prefix="fov-risk-pair") as executor:
                        futures = [executor.submit(collect_one, index, item)
                                   for index, item in enumerate(batch)]
                        completed_batch = [future.result() for future in futures]
                # Persist only on this thread and in episode order, so a crash
                # resumes from a complete, deterministic prefix.
                stop = False
                for episode_id, seed, variant, rows, metric in sorted(
                        completed_batch, key=lambda result: result[0]):
                    samples = ([{"graph_X": row["fov_graph_X"],
                                 "geometric_in_fov": row["fov_graph_geometric_in_fov"]}
                                for row in rows]
                               + ([{"graph_X": rows[-1]["next_fov_graph_X"],
                                    "geometric_in_fov": rows[-1]["geometric_in_fov"]}]
                                  if rows else []))
                    episodes.append({"episode_id": episode_id, "seed": seed,
                                     "samples": samples})
                    flown_steps += len(rows)
                    total_steps += len(rows)
                    loss_episodes += int(metric["geometric_fov_loss_episode_rate"])
                    if datastore is not None:
                        datastore.store_episode(
                            KIND_FOV_RISK, fingerprint, seed=seed,
                            payload=_fov_episode_payload(samples),
                            identity={"seed": seed, "source_pipeline": source_pipeline,
                                      "source_checkpoint_sha256": checkpoint_sha,
                                      "behaviour_variant": variant},
                            provenance={
                                "source_pipeline": source_pipeline,
                                "source_checkpoint_sha256": checkpoint_sha,
                                "behaviour_variant": variant,
                                # Recorded, not fingerprinted: the features come
                                # from this encoder's weights, so a mixed-lineage
                                # accumulation stays auditable.
                                "keypoint_encoder_implementation": keypoint_implementation,
                                "scenario": scenarios[
                                    (episode_id - 1) % len(scenarios)],
                                "config_hash": str(config_hash),
                                "geometric_fov_loss_episode": int(
                                    metric["geometric_fov_loss_episode_rate"]),
                                "control_hz": float(control_hz),
                            },
                            samples=len(samples), environment_steps=len(rows))
                    dataset = build_fov_risk_dataset(
                        episodes, prediction_steps=prediction_steps)
                    progress = _fov_dataset_progress(dataset)
                    monitor.fov_dataset(
                        episodes=len(episodes), minimum=minimum, maximum=maximum,
                        loss_episodes=loss_episodes, environment_steps=total_steps,
                        **progress)
                    if len(episodes) >= minimum and progress["covered"]:
                        stop = True
                        continue
                    print(f"FOV-risk data {len(episodes)}/{minimum} minimum "
                          f"(cap {maximum}): loss_episode="
                          f"{int(metric['geometric_fov_loss_episode_rate'])}")
                if stop:
                    break
    else:
        monitor.fov_dataset(
            episodes=len(episodes), minimum=minimum, maximum=maximum,
            loss_episodes=0, environment_steps=total_steps, cached=True,
            **_fov_dataset_progress(dataset))
    if dataset is None or not _fov_target_coverage(dataset):
        raise RuntimeError(
            f"FOV-risk data covers one target regime after {maximum} episodes; "
            "increase --rgat-max-data-episodes")
    manifest = save_fov_risk_dataset(
        dataset, dataset_path, config_hash=config_hash, seed=seed0,
        horizon_seconds=seconds, control_hz=control_hz)
    reuse = {}
    if datastore is not None:
        # The split is the store's, frozen on each episode's seed. Recomputed
        # here over the episode ids this materialisation used.
        by_seed = {int(record.seed): record for record in
                   datastore.episodes(KIND_FOV_RISK, fingerprint,
                                      payloads=False)}
        consumed, validation_ids = [], []
        for episode in episodes:
            record = by_seed.get(int(episode["seed"]))
            if record is None:
                continue
            consumed.append(record)
            if record.split(validation_fraction) == "validation":
                validation_ids.append(int(episode["episode_id"]))
        reuse = {
            "fingerprint": fingerprint,
            "validation_episodes": validation_ids,
            "consumed": consumed,
            "validation_fraction": float(validation_fraction),
            "reused_episodes": len(stored),
            "flown_episodes": len(episodes) - len(stored),
            # One initial condition flown twice is not a duplicate sample, but
            # it does weigh that condition twice. The collection skips seeds
            # the store already holds, so this should equal the episode count;
            # it is reported rather than assumed.
            "distinct_seeds": len({int(record.seed) for record in consumed}),
            "encoder_lineages": sorted({
                str(record.provenance.get("keypoint_encoder_implementation"))
                for record in consumed}),
        }
    manifest.update({
        "source_pipeline": source_pipeline,
        "source_checkpoint_sha256": checkpoint_sha,
        "environment_steps": total_steps,
        "episode_split_required": True,
        "datastore_fingerprint": fingerprint,
        "reused_episodes": len(stored),
        "flown_episodes": len(episodes) - len(stored),
        "flown_environment_steps": flown_steps,
        "distinct_seeds": len({int(episode["seed"]) for episode in episodes}),
        "validation_episodes": reuse.get("validation_episodes"),
    })
    # Persist the extended provenance without altering the checked NPZ digest.
    _write_json(dataset_path.with_suffix(".manifest.json"), manifest)
    return dataset, manifest, dataset_path, flown_steps, reuse


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
                           parallel_contexts=None, flight_allowed=True):
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

    scenarios = tuple(design.get("scenarios") or (COLLECTION_BASE_SCENARIO,))
    fast_settings = behavior_cloning_settings(config)
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
        _require_flight(
            flight_allowed, "the adaptive reward-design rollouts",
            shortfall=(f"{dataset_quality(dataset)['episodes']}/{count} "
                       "episodes stored, or the stratified quality contract "
                       "is not met yet"))
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
        ("planar_ontology_graph_state", "three_arm_burst",
         "two_pipeline_fov_risk") if primary_only else
        ("three_pipeline", "adaptive_reward_weight_comparison")),
                        default=None)
    # The primary runner also accepts the ablations of the proposed arm and the
    # retired reward-side arm by name: they are legitimate runs of this
    # experiment, just not the default one.
    parser.add_argument(
        "--pipelines", nargs="+",
        choices=(tuple(dict.fromkeys(
            primary_pipeline_ids() + ablation_pipeline_ids()
            + ("shin_se_onto_rgat_recovery",)))
            if primary_only else available_pipeline_ids()))
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument(
        "--stage", choices=STAGES, default=STAGE_ALL,
        help="which half of the run to execute. 'collect' flies every "
             "reward-design and demonstration dataset on every pair and "
             "stops; 'train' fits the frozen readouts, runs PPO for every arm "
             "and scores them, and refuses to fly for collection; 'all' does "
             "both in one process, collection first")
    # The zero-argument contract: bare ``./run.sh`` runs the current headline
    # experiment. Since 2026-09-23 that is the planar three-arm comparison --
    # a reduced sagittal-plane control envelope, three constant-heading decks
    # at three speed profiles, PN guidance as the non-learned control
    # condition, and the ontology graph as the policy's STATE rather than as a
    # reward term. The designs it replaced are still declared and runnable:
    #   ./run.sh --config config/experiments/three_arm_burst_comparison.yaml
    #   ./run.sh --config config/experiments/two_pipeline_comparison.yaml
    parser.add_argument("--config", type=Path,
                        default=ROOT / "config/experiments/planar_three_arm_comparison.yaml"
                        if primary_only else
                        ROOT / "config/experiments/three_pipeline_comparison.yaml")
    # The planar profile is the minimal profile plus the two things the reduced
    # envelope depends on: constant-heading decks and an entry seeded on the
    # deck's track. The campus USD is still selectable and still the profile
    # for a photoreal run.
    parser.add_argument("--system-config", type=Path,
                        default=ROOT / "config/shin2026-planar-system.yaml")
    parser.add_argument("--results-dir", type=Path)
    # Collected flight data outlives one results directory. The accumulation
    # is reused only by a run whose data fingerprint matches; see
    # ontology_rgat.datastore for what may and may not be reused.
    parser.add_argument(
        "--datastore", type=Path,
        default=ROOT / "results/datastore/collected.sqlite3",
        help="SQLite accumulation of reusable offline collection")
    parser.add_argument(
        "--no-datastore", action="store_true",
        help="fly fresh offline data instead of reusing the accumulation")
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
        "--parallel-pairs", default="auto",
        help="number of isolated UAV/UGV pairs in the shared Isaac stage, or "
             "'auto' to size it from this machine and the profile")
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
    automatic_pairs = str(args.parallel_pairs).strip().lower() == "auto"
    if not automatic_pairs:
        try:
            args.parallel_pairs = int(args.parallel_pairs)
        except (TypeError, ValueError):
            parser.error("--parallel-pairs must be an integer or 'auto'")
        if not 1 <= args.parallel_pairs <= MAX_PARALLEL_PAIRS:
            parser.error(
                f"--parallel-pairs must be between 1 and {MAX_PARALLEL_PAIRS}")

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
    if (not automatic_pairs and args.parallel_pairs > 1
            and args.parallel_pairs % len(args.pipelines)):
        parser.error(
            f"--parallel-pairs {args.parallel_pairs} cannot be divided evenly "
            f"between {len(args.pipelines)} pipeline(s); every arm flies the "
            "same number of pairs so their budgets stay matched")
    if args.results_dir is None:
        experiment_dir = ("two_pipeline_fov_risk" if
                          config.get("experiment") == "two_pipeline_fov_risk" else
                          "adaptive_reward_weight" if
                          config.get("experiment") == "adaptive_reward_weight_comparison"
                          else "three_arm_burst" if
                          config.get("experiment") == "three_arm_burst"
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
    # Which half of the run this process is. ``collection_allowed`` gates every
    # place a vehicle would be armed to fill a dataset, and ``training_allowed``
    # every place a model is fitted or scored. Both are true for --stage all,
    # which is the bare ./run.sh contract: collect everything, then train.
    collection_allowed = args.stage in (STAGE_ALL, STAGE_COLLECT)
    training_allowed = args.stage in (STAGE_ALL, STAGE_TRAIN)

    system = load_system_config(args.system_config)
    if automatic_pairs:
        offsets = ((system.get("parallel") or {}).get("pair_offsets_enu_m")
                   or ())
        resources = machine_resources()
        args.parallel_pairs, binding, limits = automatic_parallel_pairs(
            len(args.pipelines), offsets_available=len(offsets), **resources)
        print(f"Sizing the stage from this machine: "
              f"{resources['cpu_count']} cores, "
              f"{resources['total_ram_gb']:.0f} GB RAM, "
              f"{resources['total_vram_gb']:.1f} GB VRAM, "
              f"{len(offsets)} pair offsets in the profile.")
        print(f"Running {args.parallel_pairs} UAV/UGV pair(s) "
              f"({args.parallel_pairs // len(args.pipelines)} per arm); "
              f"limited by {binding} "
              f"({', '.join(f'{k} {v}' for k, v in sorted(limits.items(), key=lambda kv: kv[1]))}).")
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
    # Display-only keys are excluded from the identity of a run. ``arm_labels``
    # names arms on a chart and has no scientific content, but the hash gates
    # checkpoint compatibility: leaving it in means fixing a typo in a legend
    # archives every checkpoint and restarts training from scratch. Measured on
    # 2026-09-22 -- adding arm_labels alone changed the hash. Anything that
    # changes what the experiment DOES still belongs in here.
    hashed_config = {key: value for key, value in config.items()
                     if key not in _DISPLAY_ONLY_CONFIG_KEYS}
    config_hash = configuration_hash({
        "experiment": hashed_config, "system": scientific_system,
        "training_replicate": args.training_replicate,
        "model_seed": model_seed, "training_seed_start": training_seed0,
        "outcome_contract": (
            "safe_landing_contact_position_velocity_attitude_rate_v2"),
    })
    configured_count = int(training_cfg.get(
        f"episodes_{args.mode}", 8 if args.mode == "quick" else 40960))
    # Deck motions the PPO episodes rotate through. The FOV-risk collection
    # below flies the same set, so the frozen readout is fitted on the deck
    # distribution the policies are actually trained and scored against.
    training_scenarios = tuple(training_cfg.get("scenarios")
                               or (COLLECTION_BASE_SCENARIO,))
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
        # Shrink the budget, but do not resurrect a deck the profile retired.
        # This mapping is scenario -> episode count, and a zero is a deliberate
        # statement that the deck is out of the experiment: three_arm_burst
        # zeroes five of the six named decks so every flight contains the
        # escape burst. Rebuilding it from the KEYS alone gave every retired
        # deck 2 flights back, so the quick pass -- the one run whose job is to
        # validate the configuration before a machine-day is spent on it --
        # silently exercised a scenario set the profile does not declare.
        evaluation_cfg = {
            name: (0 if not int(count)
                   else 4 if name == "training_random_walk" else 2)
            for name, count in evaluation_cfg.items()}
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
    # Control conditions are evaluation-only: they take no training pair, so
    # ``args.pipelines`` -- and with it _training_pair_replicas' divisibility
    # invariant -- is left alone and only the evaluation plan grows.
    baseline_arms = _baseline_arms(config)
    baseline_ids = {arm["method"] for arm in baseline_arms}
    baseline_by_id = {arm["method"]: arm for arm in baseline_arms}
    plan = _plan_with_baseline_arms(plan, baseline_arms)
    arm_manifest = _arm_manifest(args.pipelines, baseline_arms,
                                 dict(config.get("arm_labels") or {}))

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
    # The current method. It has no offline artifact, so nothing here gates a
    # collection stage; it only decides which algorithm diagram the manifest
    # and the dashboard carry.
    uses_graph_state = any(
        get_pipeline(name).graph_state_enabled for name in args.pipelines)
    graph_state_cfg = dict(config.get("graph_state") or {})
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
        # Every arm the run compares, learned or not, in display order. The
        # ``pipelines`` list above is the LEARNED arms only -- it is what
        # training iterates and what checkpoints are keyed on -- so a reader
        # that infers the arm set from it silently drops the control
        # conditions. That inference caused two separate defects on
        # 2026-09-22: a non-learned arm missing from every figure, and a
        # completed run reported as preliminary because the evaluation
        # completeness check demanded a checkpoint for an arm that has none by
        # construction. Persisting the manifest makes the distinction readable
        # instead of guessable.
        "arms": arm_manifest,
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
    # Physical pairs each method flies on. One per method is the sequential
    # case; more makes each method's episodes a vectorised batch. Measured on
    # this machine, four pairs move 4.96x simulated seconds per wall second in
    # total against 2.60x for two, so the pairs are worth spending.
    training_pair_replicas = _training_pair_replicas(
        args.pipelines, args.parallel_pairs, args.training_replicate)
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

    # One RViz view per *physical pair*: ``pair_training_methods`` has one
    # entry per pair (a method repeats when it flies several replicas), where
    # ``args.pipelines`` has one per arm and left pairs 2..N sharing pair 0/1's
    # topics and TF frames.
    rviz = (RvizPublisher.create(
        cfg, pair_methods=(pair_training_methods if args.parallel_pairs > 1
                           else None))
        if not args.no_rviz else None)
    monitor = BenchmarkMonitor(STORE, rviz=rviz)
    monitor.configure(
        methods=[arm["method"] for arm in arm_manifest], arms=arm_manifest,
        mode=args.mode, config_hash=config_hash,
        training_total=(train_count * len(args.pipelines) + selected_warmup),
        evaluation_total=len(plan),
        reward_design_id=getattr(
            fov_risk_model if needs_fov_risk else potential, "design_id", None),
        reward_design_sha256=getattr(
            fov_risk_model if needs_fov_risk else potential, "sha256", None),
        fov_risk_design_id=getattr(fov_risk_model, "design_id", None),
        algorithm_pipeline=(graph_state_pipeline_contract(
            control_hz=1.0 / float((config.get("control") or {}).get(
                "dt_seconds", 0.1)),
            hidden_dim=int(graph_state_cfg.get("hidden_dim", 32)),
            graph_dim=int(graph_state_cfg.get("graph_dim", 32)),
            representation=str(graph_state_cfg.get(
                "representation", "ontology_rgat")),
            camera=(system.get("vision") or {}).get("camera"),
        ) if uses_graph_state else algorithm_pipeline_contract(
            lambda_fov=float((config.get("fov_risk") or {}).get("lambda_fov", 0.1)),
            horizon_seconds=float((config.get("fov_risk") or {}).get(
                "prediction_horizon_seconds", 1.0)),
            control_hz=1.0 / float((config.get("control") or {}).get(
                "dt_seconds", 0.1)),
            hidden_dim=int((config.get("fov_risk_design") or {}).get(
                "hidden_dim", 24)),
            camera=(system.get("vision") or {}).get("camera"),
        ) if needs_fov_risk else None),
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
    datastore = open_datastore(None if args.no_datastore else args.datastore)
    if datastore is not None:
        datastore.record_run(
            experiment=str(args.experiment), results_dir=args.results_dir,
            config_hash=config_hash,
            notes=f"mode={args.mode} pipelines={','.join(args.pipelines)}")
        print(f"Accumulating reusable collection in {args.datastore}.")
    dashboard = Dashboard(cfg, STORE).start()
    # ``--headless`` is about Isaac Sim's own window, not about the operator's
    # view of the run. RViz 2 is a separate process reading ROS topics that are
    # published either way, so a headless flight still gets its live view; only
    # ``--no-rviz`` (or a missing DISPLAY) turns it off.
    # Pair groups are titled by the LEARNED arms: training divides the pairs
    # between those and the control conditions hold none, so the pair-order
    # titles follow args.pipelines rather than the full arm list.
    rviz_process, rviz_log = _start_rviz(
        cfg.viz.rviz.enabled and not args.no_rviz,
        parallel_pairs=args.parallel_pairs,
        arm_titles=[arm["label"] for arm in arm_manifest if arm["learned"]])
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
            monitor.stage("keypoint validation",
                          "live Isaac camera · surveyed viewpoints")
            # The encoder is an input to collection, not a product of it: the
            # semantic and FOV graphs are read out of its keypoints, so every
            # stored episode is labelled in its coordinates. A training stage
            # therefore expects the survey to have happened already -- flying
            # it here would mean the readout is fitted on frames from a lineage
            # no collect run ever recorded.
            if not collection_allowed and needs_empirical_calibration(
                    keypoint_pretraining):
                _require_flight(
                    False, "the keypoint encoder's live Isaac calibration",
                    shortfall="the saved encoder is not empirically validated")
            calibration_system = _calibration_system_for_pair(
                system, pair_index=0, pair_count=args.parallel_pairs)
            keypoint_pretraining = calibrate_keypoint_encoder_in_flight(
                args.results_dir / "models/shared/keypoint_encoder.pt",
                keypoint_pretraining, camera, cfg=pair_cfgs[0], config=config,
                system=calibration_system, mode=args.mode, device=args.device,
                datastore=datastore)
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
            # The warm start runs before the first PPO worker is submitted,
            # so every spawned pair is idle here and none of them is the one
            # a training arm or the reward-design collector is flying. The
            # teacher therefore collects on all of them at once.
            demonstration_contexts = [{
                "cfg": pair_cfgs[index], "camera": cameras[index],
                "monitor": worker_monitors[index],
            } for index in range(args.parallel_pairs)]
            demonstrations = _prepare_fast_demonstrations(
                cfg=cfg, camera=camera, config=config,
                config_hash=config_hash,
                keypoint_pretraining=keypoint_pretraining,
                results_dir=args.results_dir, device=args.device,
                system=system,
                model_seed=model_seed,
                monitor=(worker_monitors[0]
                         if args.parallel_pairs > 1 else monitor),
                parallel_contexts=demonstration_contexts,
                flight_allowed=collection_allowed)
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
                    cloning = behavior_cloning_settings(config)
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
                cloning = behavior_cloning_settings(config)
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
                # A measured pipeline flies every pair assigned to its method;
                # the reward-design source run stays on the single pair it was
                # handed, because nothing batches its collection.
                replica_pairs = (list(training_pair_replicas.get(name, [pair_index]))
                                 if primary else [pair_index])

                def live_environment(index):
                    return lambda: LiveShinEnvironment(
                        pair_cfgs[index], cameras[index],
                        horizon_steps=int(pair_cfgs[index].sim.max_steps))

                replica_factories = [live_environment(index)
                                     for index in replica_pairs]
                # Each replica reports to its own pair's panel. Sharing
                # ``local_monitor`` put two vehicles on one trajectory plot.
                replica_monitors = [worker_monitors[index]
                                    for index in replica_pairs]
                if len(replica_factories) > 1:
                    local_monitor.stage(
                        "parallel training" if args.parallel_pairs > 1 else "training",
                        f"pairs {', '.join(str(i) for i in replica_pairs)} · "
                        f"recurrent PPO · {name}")
                history = train_live(
                    replica_factories[0],
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
                    training_contract_id=training_contract_id,
                    scenarios=training_scenarios,
                    env_factories=replica_factories,
                    env_monitors=replica_monitors)
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

            # ================================================================
            # Stage 1 -- data collection.
            #
            # Every actor the run will ever need is built here, before a single
            # reward-design episode is flown and before any PPO exists.
            # ``torch.manual_seed`` is process-global, so constructing a model
            # while a worker is sampling would perturb that worker's
            # exploration stream according to thread timing; building them all
            # up front is what keeps a parallel run reproducible.
            #
            # None of them is trained here. The reward-design rollouts are
            # flown by the BC-warmed frozen source policy, or by a compatible
            # checkpoint an earlier run left on disk -- never by a policy this
            # process trains, because the collection stage runs no PPO by
            # construction. Every multi-pair run already worked this way (the
            # frozen source is what the parallel path flew while the
            # comparison arm trained); it is now the single rule, so a
            # one-pair and a four-pair run collect from the same behaviour
            # distribution and neither charges a source-policy PPO to the
            # reward-design budget.
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
            source_names = list(dict.fromkeys(source_names))
            prepared_names = list(dict.fromkeys([*args.pipelines, *source_names]))
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

            def reward_design_source(name):
                """The frozen policy a reward-design rollout is flown by.

                A compatible checkpoint from an earlier run is preferred --
                accumulated collection is the point of the datastore, and a
                trained source makes the rollouts more like the states the
                arms actually visit. Without one the BC-warmed initialization
                flies, which is a policy that already tracks the deck because
                the teacher demonstrations above are cloned into it.
                """
                model = frozen_behavior_models[name]
                checkpoint = _load_reward_design_source_checkpoint(
                    args.results_dir, name, model, config_hash=config_hash)
                if checkpoint is None:
                    print(f"Collecting reward-design data with the BC-warmed "
                          f"{name} policy; no compatible checkpoint on disk.")
                else:
                    print(f"Collecting reward-design data with the compatible "
                          f"{name} checkpoint {checkpoint}.")
                return model, checkpoint

            # Nothing else is flying, so the reward design collects on every
            # pair. This is what the stage split buys: the collection used to
            # be handed whatever pair the training assignment left over, which
            # on a four-pair stage was one pair, and when the arithmetic did
            # not work out was a pair a PPO replica was already flying.
            collection_pair_indices = _reward_design_pair_indices(
                list(training_futures), training_pair_replicas,
                training_pair_for, args.parallel_pairs)
            reward_design_pair_indices = collection_pair_indices
            collection_contexts = [{
                "cfg": pair_cfgs[index], "camera": cameras[index],
                "monitor": worker_monitors[index],
            } for index in collection_pair_indices]
            parallel_collection_contexts = (
                collection_contexts if args.parallel_pairs > 1 else None)
            collection_cfg = pair_cfgs[collection_pair_indices[0]]
            collection_camera = cameras[collection_pair_indices[0]]
            collection_monitor = worker_monitors[collection_pair_indices[0]]
            if args.parallel_pairs > 1:
                manifest["parallel_execution"].update({
                    "stage_order": "collect_then_train",
                    "training_pair_replicas": {
                        name: [int(index) for index in indices]
                        for name, indices in training_pair_replicas.items()},
                    "reward_design_pair_indices": reward_design_pair_indices,
                })
                _write_json(manifest_path, manifest)
                refresh_presentation_results()

            fov_design_episodes = 0
            fov_design_steps = 0
            fov_dataset = None
            fov_manifest = None
            fov_reuse = {}
            fov_validation_fraction = float(
                (config.get("fov_risk_design") or {}).get(
                    "validation_fraction", 0.2))
            if needs_fov_risk and fov_risk_model is None:
                fov_source_name = str((config.get("fov_risk_design") or {}).get(
                    "source_pipeline", "shin_se_fixed"))
                fov_source_model, fov_source_checkpoint = reward_design_source(
                    fov_source_name)
                fov_dataset, fov_manifest, _, fov_design_steps, fov_reuse = (
                    _collect_fov_risk_data(
                        cfg=collection_cfg, camera=collection_camera,
                        model=fov_source_model, config=config,
                        config_hash=config_hash,
                        checkpoint_path=fov_source_checkpoint,
                        results_dir=args.results_dir, mode=args.mode,
                        monitor=collection_monitor,
                        episodes_override=args.rgat_data_episodes,
                        max_episodes_override=args.rgat_max_data_episodes,
                        source_pipeline=fov_source_name, system=system,
                        datastore=datastore, scenarios=training_scenarios,
                        keypoint_implementation=(
                            None if keypoint_pretraining is None
                            else keypoint_pretraining["implementation"]),
                        validation_fraction=fov_validation_fraction,
                        parallel_contexts=parallel_collection_contexts,
                        flight_allowed=collection_allowed,
                        peer_health=peer_training_health(training_futures)))
                fov_design_episodes = int(fov_manifest["episodes"])
            elif needs_fov_risk:
                fov_manifest = dict(fov_risk_model.metadata)
                fov_design_episodes = int((fov_manifest.get(
                    "dataset_manifest") or {}).get("episodes", 0))
                # A resumed run skips collection and training entirely; without
                # this the FOV panel would stay blank for the whole run even
                # though a validated readout is driving the proposed reward.
                monitor.fov_model(design_id=fov_risk_model.design_id,
                                  metadata=fov_manifest)

            dataset = None
            dataset_manifest = None
            dataset_path = None
            if needs_potential and potential is None:
                preferred_source = str((config.get("rgat_design") or {}).get(
                    "source_pipeline", "no_se"))
                semantic_source_name = (
                    preferred_source if preferred_source in available_pipeline_ids()
                    else "no_se")
                semantic_source_model, semantic_source_checkpoint = (
                    reward_design_source(semantic_source_name))
                # The source policy is never trained by this run any more, so
                # it adds no episodes to the reward-design budget. Kept as
                # explicit zeros because the manifest and the cost report both
                # read them.
                source_training_episodes = 0
                source_training_steps = 0
                dataset, dataset_manifest, dataset_path, design_steps = (
                    _collect_semantic_data(
                        cfg=collection_cfg, camera=collection_camera,
                        model=semantic_source_model, config=config,
                        config_hash=config_hash,
                        checkpoint_path=semantic_source_checkpoint,
                        results_dir=args.results_dir, mode=args.mode,
                        monitor=collection_monitor,
                        episodes_override=args.rgat_data_episodes,
                        max_episodes_override=args.rgat_max_data_episodes,
                        source_pipeline=semantic_source_name,
                        source_training_episodes=source_training_episodes,
                        source_training_environment_steps=source_training_steps,
                        scenarios=training_scenarios,
                        flight_allowed=collection_allowed))
                design_episodes = int(dataset_manifest["episodes"])
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
            adaptive_dataset = None
            adaptive_manifest = None
            adaptive_source_name = None
            adaptive_source_model = None
            adaptive_source_checkpoint = None
            if needs_adaptive and adaptive_weights is None:
                _archive_rejected_adaptive_artifact(
                    args.adaptive_reward_design, adaptive_rejected_design_id)
                adaptive_source_name = str(
                    (config.get("adaptive_reward_design") or {}).get(
                        "source_pipeline", "no_se_fixed"))
                adaptive_source_model, adaptive_source_checkpoint = (
                    reward_design_source(adaptive_source_name))
                adaptive_dataset, adaptive_manifest, _, adaptive_design_steps = (
                    _collect_adaptive_data(
                        cfg=collection_cfg, camera=collection_camera,
                        model=adaptive_source_model,
                        source_pipeline=adaptive_source_name, config=config,
                        config_hash=config_hash, results_dir=args.results_dir,
                        mode=args.mode, monitor=collection_monitor,
                        episodes_override=args.rgat_data_episodes,
                        max_episodes_override=args.rgat_max_data_episodes,
                        source_checkpoint_path=adaptive_source_checkpoint,
                        parallel_contexts=parallel_collection_contexts,
                        flight_allowed=collection_allowed,
                        minimum_unsafe_failures_override=(
                            (adaptive_settings.get("quality_gate") or {}).get(
                                "minimum_unsafe_failure_episodes"))))
                adaptive_design_episodes = int(adaptive_manifest["episodes"])
            elif needs_adaptive:
                data_manifest = adaptive_weights.metadata.get(
                    "dataset_manifest") or {}
                adaptive_design_episodes = int(data_manifest.get("episodes", 0))
                adaptive_design_steps = int(data_manifest.get("transitions", 0))

            collection_summary = {
                "stage": args.stage,
                "collection_pairs": collection_pair_indices,
                # Counts only; the full demonstration provenance is already
                # under manifest["behavior_cloning_demonstrations"].
                "behavior_cloning_demonstrations": (
                    None if demonstrations is None else {
                        "successful_episodes": int(
                            demonstrations["successful_episodes"]),
                        "attempted_episodes": len(
                            demonstrations.get("attempted_seeds", ())),
                        "environment_steps": int(
                            demonstrations.get("environment_steps", 0))}),
                "fov_risk": ({"episodes": fov_design_episodes,
                              "flown_environment_steps": fov_design_steps}
                             if needs_fov_risk else None),
                "semantic_rgat": ({"episodes": design_episodes,
                                   "environment_steps": design_steps}
                                  if needs_potential else None),
                "adaptive_reward": ({"episodes": adaptive_design_episodes,
                                     "transitions": adaptive_design_steps}
                                    if needs_adaptive else None),
            }
            manifest["data_collection"] = collection_summary
            _write_json(manifest_path, manifest)
            monitor.collection_stage(
                stage=args.stage, pairs=collection_pair_indices,
                complete=True,
                datasets=[{
                    "name": name, "episodes": int(value["episodes"]),
                    **({"reused_episodes": int(fov_reuse["reused_episodes"]),
                        "flown_episodes": int(fov_reuse["flown_episodes"])}
                       if name == "FOV-risk" and fov_reuse.get("reused_episodes")
                       is not None else {}),
                } for name, value in (
                    ("FOV-risk", collection_summary["fov_risk"]),
                    ("semantic R-GAT", collection_summary["semantic_rgat"]),
                    ("adaptive reward", collection_summary["adaptive_reward"]))
                    if value is not None])
            if not training_allowed:
                manifest["execution_status"] = (
                    "data collection complete; no model was trained")
                _write_json(manifest_path, manifest)
                monitor.stage(
                    "collection complete",
                    "datasets persisted · run --stage train to fit and fly")
                print("Data collection stage complete. Collected: "
                      + ", ".join(
                          f"{name} {value['episodes']} episodes"
                          for name, value in (
                              ("FOV-risk", collection_summary["fov_risk"]),
                              ("semantic R-GAT", collection_summary["semantic_rgat"]),
                              ("adaptive reward", collection_summary["adaptive_reward"]))
                          if value is not None)
                      + f". Artifacts under {args.results_dir}"
                      + ("" if datastore is None else
                         f" and {args.datastore}") + ".")
                print("Next: rerun this command with --stage train "
                      "(same --config and --system-config).")
                refresh_presentation_results()
                if args.stay_open:
                    _hold_after_complete(owned, monitor)
                return 0

            # ================================================================
            # Stage 2 -- training.
            #
            # From here nothing is collected. The frozen readouts are fitted on
            # the datasets above, every arm's PPO starts against the same
            # finished reward design, and the arms are then scored. The one
            # exception is named where it happens: the adaptive artifact's
            # quality gate can ask for more real episodes, which under
            # --stage train is refused rather than flown.
            if needs_fov_risk and fov_risk_model is None:
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

                settings["validation_fraction"] = fov_validation_fraction
                fov_risk_model, fov_metadata = prepare_fov_risk_artifact(
                    args.fov_risk_model, fov_dataset,
                    dataset_manifest=fov_manifest, config_hash=config_hash,
                    seed=model_seed, settings=settings, progress=_fov_epoch,
                    validation_episodes=fov_reuse.get("validation_episodes"))
                if datastore is not None and fov_reuse.get("consumed"):
                    # A fresh FOVRiskModel every run, so the accumulation adds
                    # data rather than epochs. Written down so an artifact that
                    # was ever fine-tuned on one is distinguishable later.
                    datastore.record_consumption(
                        fov_risk_model.sha256, fov_reuse["consumed"],
                        validation_fraction=fov_reuse["validation_fraction"],
                        trained_from_scratch=True)
                    fov_metadata["datastore"] = {
                        key: fov_reuse[key] for key in (
                            "fingerprint", "reused_episodes", "flown_episodes",
                            "distinct_seeds", "validation_fraction",
                            "encoder_lineages")}
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

            if needs_potential and potential is None:
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

            if needs_adaptive and adaptive_weights is None:
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
                        # The only flight the training stage can ask for, and
                        # only because the gate is a property of the fitted
                        # model rather than of the raw episodes: whether more
                        # data is needed is not knowable until the fit is
                        # attempted. Under --stage train this refuses and names
                        # the collect command with the larger budget.
                        adaptive_dataset, adaptive_manifest, _, adaptive_design_steps = (
                            _collect_adaptive_data(
                                cfg=collection_cfg, camera=collection_camera,
                                model=adaptive_source_model,
                                source_pipeline=adaptive_source_name,
                                config=config, config_hash=config_hash,
                                results_dir=args.results_dir,
                                mode=args.mode, monitor=collection_monitor,
                                episodes_override=next_minimum,
                                max_episodes_override=adaptive_maximum,
                                source_checkpoint_path=(
                                    adaptive_source_checkpoint),
                                parallel_contexts=parallel_collection_contexts,
                                flight_allowed=collection_allowed,
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

            # 이제 모든 동결 보상 설계가 준비됐다. PBRS/adaptive PPO에서는
            # 이 모델들이 optimizer에 포함되지 않으며 매 update 뒤 hash를 검사한다.
            #
            # 모든 arm이 여기서 동시에 출발한다. 고정 보상 arm을 먼저 띄우고
            # 남은 pair에서 데이터를 모으던 예전 순서에서는, 제안 arm의 동결
            # readout이 비교 arm이 이미 수백 에피소드를 학습한 뒤에야 만들어졌다.
            # 이제 두 arm 모두 완성된 같은 보상 설계를 상대로 1 에피소드부터
            # 학습한다.
            if args.parallel_pairs > 1:
                monitor.stage(
                    "parallel training preparation",
                    "R-GAT frozen · starting every arm on its dedicated pairs")
                training_executor = ThreadPoolExecutor(
                    max_workers=args.parallel_pairs,
                    thread_name_prefix="landing-pair")
                try:
                    for name in args.pipelines:
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
                    train_pipeline(name, pair_index=training_pair_for[name],
                                   model=prepared_parallel_models[name])

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
                if name in baseline_ids:
                    # A control condition has no checkpoint, so the selection
                    # test below would find none and archive every one of its
                    # rows as "deployment_checkpoint_changed". Its flights are
                    # reproducible from the arm id and the seed alone, which is
                    # exactly what a checkpoint digest is standing in for.
                    valid_evaluation_rows.append(row)
                    continue
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
                    # One controller instance per pair: the servo's integral and
                    # previous centroid are per-flight state, exactly as in the
                    # demonstration stage. Built once here rather than per task
                    # so a pair's arm keeps its binding across its whole plan.
                    reference_pipeline = args.pipelines[0]
                    baseline_controllers, baseline_model = {}, None
                    if any(item["method"] in baseline_ids for item in tasks):
                        cloning = behavior_cloning_settings(config)
                        for arm_id in {item["method"] for item in tasks
                                       if item["method"] in baseline_ids}:
                            arm = baseline_by_id[arm_id]
                            if arm["controller"] == PN_GUIDANCE_CONTROLLER:
                                baseline_controllers[arm_id] = _pn_guidance_controller(
                                    environment, settings=cloning, monitor=None)
                            elif arm["controller"] == VISUAL_SERVO_TEACHER:
                                baseline_controllers[arm_id] = _visual_servo_teacher(
                                    environment, settings=cloning, monitor=None)
                            else:
                                baseline_controllers[arm_id] = _privileged_velocity_teacher(
                                    environment, settings=cloning,
                                    trace_path=None, monitor=None)
                        # The action comes from the controller; this model only
                        # carries the perception path, so the control condition
                        # sees the deck through the same encoder the policies
                        # do and a difference between arms is a difference in
                        # control rather than in observation.
                        torch.manual_seed(model_seed)
                        baseline_model = _build_model(
                            config, args.device, keypoint_pretraining,
                            pipeline=args.pipelines[0])
                        baseline_model.eval()
                    for item in tasks:
                        name = item["method"]
                        key = (name, item["scenario"], int(item["seed"]))
                        if key in completed:
                            continue
                        if name in baseline_ids:
                            # A control condition has no pipeline spec, so it
                            # flies under the REFERENCE arm's method for the
                            # observation and reward machinery and is relabelled
                            # below. That is also what makes the comparison
                            # fair: the servo sees the deck through the same
                            # encoder the policies do, and its row carries the
                            # same fields. Its return is meaningless -- nothing
                            # learns from it -- but every landing metric comes
                            # from the environment and is directly comparable.
                            model = baseline_model
                            transform = baseline_controllers[name]
                            collect_method = reference_pipeline
                            arm_potential = reward_design_for(reference_pipeline)
                            if hasattr(transform, "seed"):
                                transform.seed = int(item["seed"])
                            if isinstance(getattr(transform, "flight", None), dict):
                                transform.flight["seed"] = int(item["seed"])
                        else:
                            model = models[name]
                            transform = None
                            collect_method = name
                            arm_potential = reward_design_for(name)
                        local_monitor = _LockedMonitor(
                            monitor, monitor_lock, method=name, pair_index=index,
                            pair_count=args.parallel_pairs)
                        local_monitor.stage(
                            "parallel crossover evaluation"
                            if args.parallel_pairs > 1 else "paired evaluation",
                            f"pair {index} · {name}")
                        _, metric = collect_episode_resilient(
                            environment, model, collect_method, int(item["seed"]),
                            curriculum=float(seminar_fast.get(
                                "evaluation_curriculum", 1.0)),
                            potential=arm_potential,
                            fov_risk_lambda=float(ppo.get(
                                "fov_risk_lambda", 0.1)),
                            deterministic=True, gamma=float(ppo.get("gamma", .99)),
                            shaping_lambda=float(ppo.get("shaping_lambda", 1.0)),
                            reward_normalizer=runtime_reward_normalizer,
                            scenario=item["scenario"], monitor=local_monitor,
                            phase="evaluation", action_transform=transform)
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
                plan, [arm["method"] for arm in arm_manifest],
                args.parallel_pairs)
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
        if datastore is not None:
            datastore.close()
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
