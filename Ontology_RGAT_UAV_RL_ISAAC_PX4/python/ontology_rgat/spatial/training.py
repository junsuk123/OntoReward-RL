"""Shared PPO and signed checkpoints for spatial observations/actions.

The numerical PPO, masked pretraining, typed attention and relation guard are
the reference implementations, parameterised only at explicit 3D boundaries.
All three arms get the same task, measurements, supervisor and PPO budget.
"""
import copy
import hashlib
from dataclasses import asdict, replace
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn

from .core import SpatialConfig, FIELDS
from .environment import SpatialLandingEnv
from .auditing import terminal_hold_audit
from ..two_axis.models import POLICY_MODES, TwoAxisPPOAgent
from ..two_axis.models_v28 import ReferenceHead
from ..two_axis.training import PPOTrainer, PPOHyperparameters, selection_score
from ..two_axis.curriculum import CurriculumScheduler, replay_difficulty
from ..two_axis.learning import collect_rollout
from ..two_axis.pretraining import pretrain_causal_encoder
from ..two_axis.relational import guard_relational_candidate
from ..two_axis.artifacts import json_text
from ..landing.packet import head_geometry


def _ontology(config):
    """The task's typed ontology: the planar nine, plus 3D's own elements.

    The frozen ladder rungs keep the nine their checkpoints were trained on.
    """
    from ..landing.ontology import SPATIAL_EXTENSIONS, schema
    from .core import REFERENCE_SCHEMAS

    if config.schema in REFERENCE_SCHEMAS:
        return schema(SPATIAL_EXTENSIONS)
    return schema()


def _graph_planes(config):
    """One context plane per horizontal axis.

    The unified contract derives this from its axes. The frozen ladder rungs
    keep whatever shape their checkpoints were trained with: only v9/v10 ever
    built two planes.
    """
    from .core import REFERENCE_SCHEMAS

    if config.schema in REFERENCE_SCHEMAS:
        return head_geometry(config.axes, extras=True).graph_planes
    return 2 if config.reference_context else 1

VALIDATION_SEEDS = (2000, 2001)
TEST_SEEDS = (9000, 9001)
INITIALIZATION = "reference-v28-gaussian"


class CurriculumTrainingEnv:
    """Train-only outcome promotion and deterministic easy/bridge replay.

    Uses the reference scheduler. Replay successes never promote the active
    stage and cannot make a checkpoint nominal-eligible.
    """

    def __init__(self, cfg, scheduler, env_factory, angular_scales=(1.0, 1.0),
                 loss_timeout_start=None):
        self.env = env_factory(cfg)
        self.scheduler = scheduler
        self.angular_scales = angular_scales
        self.loss_timeout_start = loss_timeout_start

    @property
    def episode_history(self):
        return self.env.episode_history

    def reset(self, *, seed):
        self.role = self.scheduler.role_for(self.scheduler.episodes)
        difficulty = replay_difficulty(self.scheduler.difficulty, self.role)
        return self.env.reset(
            seed=seed,
            difficulty=difficulty,
            angular_curriculum_scales=self.angular_scales,
            loss_curriculum_start=self.loss_timeout_start,
        )

    def step(self, action):
        result = self.env.step(action)
        if result[2]:
            self.episode_history[-1]["role"] = self.role.value
            self.scheduler.record(
                role=self.role, landed=result[4]["status"] == "SUCCESS"
            )
        return result

    def close(self):
        self.env.close()


class SpatialAgent(nn.Module):
    tensors = TwoAxisPPOAgent.tensors
    act = TwoAxisPPOAgent.act
    parameter_count = TwoAxisPPOAgent.parameter_count

    def __init__(
        self, mode, config, seed=42, initial_log_std=-0.7, initialization=INITIALIZATION,
        state_dependent_log_std=False,
    ):
        super().__init__()
        if mode not in POLICY_MODES:
            raise ValueError("unknown spatial arm")
        self.mode, self.seed, self.device = mode, int(seed), torch.device("cpu")
        self.initialization = initialization
        # Architecture, so it belongs to the CHECKPOINT, not to config_sha256:
        # a new config field would change the hash even at its default and
        # orphan every checkpoint already on disk.
        self.state_dependent_log_std = bool(state_dependent_log_std)
        self.graph_config = config.ontology
        self.actor = ReferenceHead(
            mode,
            "actor",
            config.ontology,
            seed,
            initial_log_std,
            packet_dim=len(config.packet_fields),
            action_dim=3,
            descent_axis=2,
            initialization=initialization,
            minimum_log_std=-2.5 if config.direct_acceleration else -5.,
            graph_planes=_graph_planes(config),
            ontology=_ontology(config),
            state_dependent_log_std=self.state_dependent_log_std,
        )
        self.critic = ReferenceHead(
            mode,
            "critic",
            config.ontology,
            seed,
            packet_dim=len(config.packet_fields),
            action_dim=3,
            descent_axis=2,
            initialization=initialization,
            graph_planes=_graph_planes(config),
            ontology=_ontology(config),
        )
        self._generator = torch.Generator(device=self.device).manual_seed(seed + 2)


def _plain(value):
    """Coerce metadata to plain Python so the checkpoint stays weights-only.

    `load_agent` reads with ``weights_only=True``. A numpy scalar anywhere in
    the metadata makes that refuse the whole file, and callers pass numpy
    straight through: `clone_spatial_teacher` handed its holdout rates in as
    ``validation=``, which made every behaviour-cloned checkpoint unloadable by
    the pipeline -- the reason no clone had ever been fine-tuned.
    """
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    return value


def save_agent(path, agent, cfg, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "signature": cfg.signature,
        "mode": agent.mode,
        "seed": agent.seed,
        "initialization": agent.initialization,
        "state_dependent_log_std": bool(
            getattr(agent, "state_dependent_log_std", False)),
        "state_dict": agent.state_dict(),
        "config": asdict(cfg),
        "metadata": _plain({
            "initialization_source": getattr(agent, "_training_origin", None),
            **metadata,
        }),
    }
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_agent(path, cfg):
    # Checkpoints written before `_plain` carry numpy scalars in their
    # metadata. Allowlisting that one benign global keeps the weights-only
    # guarantee while letting those artifacts load; new ones never need it.
    torch.serialization.add_safe_globals(
        [np.core.multiarray.scalar, np.dtype]
        + [getattr(np.dtypes, name) for name in dir(np.dtypes)
           if name.endswith("DType")])
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("signature") != cfg.signature:
        raise ValueError("checkpoint is not compatible with the exact spatial contract")
    agent = SpatialAgent(
        payload["mode"],
        cfg,
        payload["seed"],
        initialization=payload.get("initialization", "torch-default"),
        state_dependent_log_std=payload.get("state_dependent_log_std", False),
    )
    agent.load_state_dict(payload["state_dict"], strict=True)
    return agent, payload["metadata"]


def evaluate(
    agent, cfg, *, seeds=VALIDATION_SEEDS, env_factory=SpatialLandingEnv, trace=None
):
    rows = []
    deltas = []
    for seed in seeds:
        env = env_factory(cfg)
        try:
            obs, _ = env.reset(seed=seed)
            total = 0.0
            for step in range(int(math.ceil(cfg.horizon / cfg.dt)) * 3 + 5):
                if agent.mode == "ppo_ontology_rgat":
                    with torch.no_grad():
                        p, g = agent.tensors(obs)
                        deltas.append(agent.actor.relational_delta(g)[0].abs().numpy())
                _, action, _, _ = agent.act(obs, deterministic=True)
                acted_on = obs
                obs, reward, done, _, info = env.step(action)
                total += reward
                if trace is not None:
                    # The packet the policy acted on travels with the action,
                    # so a flight can be compared with a local replay channel
                    # by channel instead of only through truth and outcomes.
                    trace(agent.mode, seed, step, action, info,
                          packet=np.asarray(acted_on.packet.values, dtype=float).tolist())
                if done:
                    break
            else:
                raise RuntimeError(
                    "episode did not terminate within bounded decision budget"
                )
            rows.append(
                {
                    "seed": seed,
                    "return": total,
                    "status": info["status"],
                    "steps": step + 1,
                    "elapsed_s": info["elapsed_s"],
                    "backend": env.backend.name,
                    "stop_confirmed": bool(env.episode_history and
                                           env.episode_history[-1].get("stop_confirmed")),
                    "terminal_hold_audit": terminal_hold_audit(info),
                }
            )
        finally:
            env.close()
    n = len(rows)
    rate = lambda status: sum(r["status"] == status for r in rows) / n
    landed, unsafe, abort, timeout = (
        rate("SUCCESS"),
        sum(r["status"] not in ("SUCCESS", "SAFE_ABORT", "TASK_TIMEOUT") for r in rows)
        / n,
        rate("SAFE_ABORT"),
        rate("TASK_TIMEOUT"),
    )
    mean_return = float(np.mean([r["return"] for r in rows]))
    return {
        "episodes": n,
        "landing_rate": landed,
        "unsafe_rate": unsafe,
        "safe_abort_rate": abort,
        "task_timeout_rate": timeout,
        "mean_return": mean_return,
        "selection_score": selection_score(
            [row["status"] for row in rows], [row["return"] for row in rows]
        ),
        "selection_score_model": "outcome_weighted_v28",
        "mean_abs_relation_residual": np.mean(deltas, axis=0).tolist()
        if deltas
        else [0.0, 0.0, 0.0],
        "rows": rows,
    }


def relational_health(agent, cfg, *, seeds=VALIDATION_SEEDS,
                      env_factory=SpatialLandingEnv, steps=64):
    """Did the proposed arm's relational path do anything at all?

    ``ppo_ontology_rgat`` differs from ``ppo_semantic_flat`` by exactly one
    term, ``relational_delta``. Its readout is zero-initialised and
    ``set_adaptation`` freezes it for the first
    ``ontology.adaptation_warmup_fraction`` of the planned budget, so an arm
    whose readout never leaves zero IS the flat baseline however it is
    labelled, and the arm comparison silently becomes two arms and a copy.

    That is not hypothetical. Measured 2026-10-05 by comparing per-iteration
    actor_loss and entropy to twelve decimals: the two arms were numerically
    identical for 8607/8607 iterations in results/spatial_long_nominal (the run
    ended before its 18000-iteration warmup) and for 1800/2000 in both
    2000-iteration runs. Every recorded three-arm spatial result predates this
    check and measures nothing about the proposed representation.

    Read ``active`` first. The magnitudes are diagnostics, not thresholds.
    """
    if agent.mode != "ppo_ontology_rgat":
        return {"applicable": False, "mode": agent.mode}
    observations = []
    for seed in seeds:
        env = env_factory(cfg)
        try:
            observation, _ = env.reset(seed=seed)
            for _ in range(steps):
                observations.append(observation)
                observation, _r, done, _t, _info = env.step(
                    agent.act(observation, deterministic=True)[1])
                if done:
                    break
        finally:
            env.close()
    from ..two_axis.models import observation_arrays
    from ..two_axis.models_v28 import relational_contribution

    packets, graphs = observation_arrays(observations)
    report = relational_contribution(
        agent,
        torch.as_tensor(packets, dtype=torch.float32),
        torch.as_tensor(graphs, dtype=torch.float32),
    )
    report["active"] = bool(report.get("actor_readout_norm", 0.0) > 0
                            and report.get("actor_delta_max_abs", 0.0) > 0)
    report["degenerate_to_semantic_flat"] = not report["active"]
    return report


def train_arm(
    mode,
    cfg,
    *,
    seed,
    hyper,
    output,
    activation_iterations=2,
    activation_decisions=128,
    env_factory=SpatialLandingEnv,
    training_backend="local-spatial",
    curriculum_enabled=True,
    initial_checkpoint=None,
    allow_v4_initialization=False,
    curriculum_angular_scales=(1.0, 1.0),
    episodes_per_iteration=0,
    curriculum_loss_timeout_start=None,
    state_dependent_log_std=False,
):
    if not 0 <= episodes_per_iteration <= 64:
        raise ValueError("episodes per iteration must be in [0,64]")
    # The relation activation phase below builds a FRESH PPOTrainer, whose
    # update counter restarts at zero, and PPOTrainer gates the actor on
    # `updates > value_warmup_iterations`. With the shipped defaults
    # (activation_iterations 2, value warmup 2) the actor is therefore never
    # updated: only the critic readout moves, the actor readout stays exactly
    # 0.0, every guard scale is rejected on residual norm, and the arm silently
    # remains numerically identical to ppo_semantic_flat. Measured in
    # results/spatial_fixed_nominal_20261005: actor_readout_norm 0.0 against
    # critic_readout_norm 0.1396. The failure is invisible in the artifacts --
    # it surfaces only as `accepted: false` with no reason attached.
    if (mode == "ppo_ontology_rgat" and activation_iterations
            and activation_iterations <= hyper.value_warmup_iterations):
        raise ValueError(
            f"relation activation would never update the actor: "
            f"activation_iterations={activation_iterations} must exceed "
            f"value_warmup_iterations={hyper.value_warmup_iterations}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob("checkpoint_*.pt")):
        raise ValueError(
            "choose fresh spatial output; existing checkpoints are never overwritten"
        )
    initialization_source = None
    if initial_checkpoint is None:
        agent = SpatialAgent(mode, cfg, seed, hyper.initial_log_std,
                             state_dependent_log_std=state_dependent_log_std)
        pretraining = pretrain_causal_encoder(
            agent, cfg, seed=seed, env_factory=env_factory, action_dimension=3
        )
    else:
        source = Path(initial_checkpoint).resolve()
        source_cfg = cfg
        if allow_v4_initialization:
            if cfg.schema != "spatial-causal-rgat/5":
                raise ValueError("explicit v4 initialization requires target v5")
            source_cfg = replace(cfg, schema="spatial-causal-rgat/4")
        agent, meta = load_agent(source, source_cfg)
        if agent.mode != mode or agent.seed != seed or not meta.get("eligible"):
            raise ValueError(
                "fine-tuning needs an eligible same-mode, same-seed checkpoint"
            )
        initialization_source = {
            "path": str(source),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "metadata": meta,
            "optimizer_reset": True,
            "mode": "PPO fine-tuning, not exact optimizer-state resume",
            "contract_migration": None if source_cfg == cfg else {
                "from": source_cfg.signature, "to": cfg.signature,
                "eligibility_transferred": False,
                "changes": ["1.5s estimated acceleration decay", "XY abort braking"]},
        }
        # Loaded encoder already has provenance. Do not overwrite it with
        # another masked pretraining pass or use actions as target labels.
        pretraining = {
            "enabled": False,
            "environment_steps": 0,
            "reason": "eligible PPO checkpoint initialization",
        }
    agent._training_origin = initialization_source
    trainer = PPOTrainer(agent, hyper)
    history = []
    started = time.monotonic()
    selected = None
    best_score = -math.inf
    completed = 0
    nominal_completed = 0
    environment_steps = 0
    scheduler = CurriculumScheduler(cfg.curriculum) if curriculum_enabled else None
    warmup = min(
        hyper.iterations - 1,
        int(math.ceil(cfg.ontology.adaptation_warmup_fraction * hyper.iterations)),
    )
    for iteration in range(1, hyper.iterations + 1):
        agent.actor.set_adaptation(cfg.ontology, iteration > warmup)
        agent.critic.set_adaptation(cfg.ontology, iteration > warmup)
        if hyper.final_log_std is not None:
            # Lower the floor with the ceiling, or the actor's own clamp
            # silently pins sigma at exp(minimum_log_std) and the schedule
            # has no effect below it -- the spatial actor floors at -2.5.
            ceiling = hyper.log_std_ceiling(iteration)
            agent.actor.maximum_log_std = ceiling
            agent.actor.minimum_log_std = min(agent.actor.minimum_log_std, ceiling)
        if scheduler is not None:
            scheduler.set_budget_progress(iteration / hyper.iterations)
        difficulty = scheduler.difficulty if scheduler is not None else 1.0
        env = (
            CurriculumTrainingEnv(
                cfg, scheduler, env_factory, curriculum_angular_scales,
                curriculum_loss_timeout_start
            )
            if curriculum_enabled
            else env_factory(cfg)
        )
        try:
            rollout = collect_rollout(
                agent,
                env,
                per_axis_evidence=hyper.intervention_masking == "per_axis",
                seed=100000 + seed * 10000 + iteration * 100,
                **(dict(episodes=episodes_per_iteration,
                        max_episode_decisions=2*math.ceil(cfg.horizon/cfg.sensor_dt)+10)
                   if episodes_per_iteration else dict(decisions=hyper.decisions_per_iteration)),
            )
            environment_steps += len(rollout)
            completed += sum(t.terminated for t in rollout)
            nominal_completed += sum(
                row["difficulty"] == 1.0 for row in env.episode_history
            )
            metrics = trainer.update(rollout, discount_time_constant_s=cfg.discount_tau)
        finally:
            env.close()
        metrics.update(
            iteration=iteration,
            completed_episodes=completed,
            difficulty=difficulty,
            environment_steps=environment_steps,
        )
        # An infrastructure failure during validation must not erase the PPO
        # update being diagnosed. This snapshot is deliberately INELIGIBLE;
        # deployment still requires completed validation and a final summary.
        save_agent(output / "checkpoint_last_update.pt", agent, cfg,
                   eligible=False, iteration=iteration,
                   completed_episodes=nominal_completed,
                   training_backend=training_backend,
                   phase="unvalidated_update_snapshot", optimizer_state_saved=False)
        if (
            iteration % hyper.evaluation_every == 0 or iteration == hyper.iterations
        ) and nominal_completed:
            val = evaluate(agent, cfg, env_factory=env_factory)
            metrics["validation"] = val
            margin = (
                cfg.ontology.selection_score_margin
                if mode == "ppo_ontology_rgat"
                else 0.0
            )
            if selected is None or val["selection_score"] > best_score + margin:
                best_score = val["selection_score"]
                selected = val
                save_agent(
                    output / "checkpoint_best.pt",
                    agent,
                    cfg,
                    eligible=True,
                    completed_episodes=nominal_completed,
                    validation=val,
                    training_backend=training_backend,
                )
        metrics["episode_outcomes"] = env.episode_history
        metrics["decision_dt_s"] = {
            "mean": float(np.mean([t.dt_s for t in rollout])),
            "max": float(max(t.dt_s for t in rollout)),
            "sum": float(sum(t.dt_s for t in rollout)),
        }
        history.append(metrics)
        # Preserve diagnostics during long runs, not just on successful exit.
        # These are not selectable checkpoints; nominal eligibility remains
        # enforced separately above.
        progress = {
            "mode": mode,
            "seed": seed,
            "signature": cfg.signature,
            "hyperparameters": asdict(hyper),
            "episodes_per_iteration": episodes_per_iteration,
            "curriculum_loss_timeout_start": curriculum_loss_timeout_start,
            "rollout_budget_unit": "complete_episodes" if episodes_per_iteration else "decisions",
            "initialization": agent.initialization,
            "initialization_source": initialization_source,
            "curriculum_angular_scales": list(curriculum_angular_scales),
            "training_backend": training_backend,
        "state_dependent_log_std": bool(state_dependent_log_std),
            "completed_nominal_episodes": nominal_completed,
            "wall_seconds": time.monotonic() - started,
            "history": history,
        }
        temporary = output / "progress.json.partial"
        temporary.write_text(json_text(progress, indent=2) + "\n")
        temporary.replace(output / "progress.json")
        print(
            f"[spatial train] {mode} seed={seed} iteration={iteration}/{hyper.iterations} "
            f"episodes={completed} difficulty={difficulty:.3f} "
            f'loss={metrics["actor_loss"]:.5f} wall={progress["wall_seconds"]:.1f}s',
            flush=True,
        )
    save_agent(
        output / "checkpoint_final.pt",
        agent,
        cfg,
        eligible=bool(nominal_completed),
        completed_episodes=nominal_completed,
        training_backend=training_backend,
    )
    selected_name = "checkpoint_best.pt" if selected is not None else None
    activation = None
    if mode == "ppo_ontology_rgat" and selected_name and activation_iterations:
        anchor, _ = load_agent(output / selected_name, cfg)
        candidate = copy.deepcopy(anchor)
        candidate.actor.set_adaptation(cfg.ontology, True)
        candidate.critic.set_adaptation(cfg.ontology, True)
        repair = PPOTrainer(candidate, hyper)
        for iteration in range(activation_iterations):
            env = env_factory(cfg)
            try:
                rollout = collect_rollout(
                    candidate,
                    env,
                    per_axis_evidence=hyper.intervention_masking == "per_axis",
                    seed=8000000 + seed * 1000 + iteration * 100,
                    decisions=activation_decisions,
                )
                repair.update(rollout, discount_time_constant_s=cfg.discount_tau)
            finally:
                env.close()
        calibrated, activation = guard_relational_candidate(
            anchor,
            candidate,
            evaluator=lambda model: evaluate(
                model, cfg, seeds=VALIDATION_SEEDS, env_factory=env_factory
            ),
        )
        activation["environment_steps"] = activation_iterations * activation_decisions
        if activation["accepted"]:
            selected_name = "checkpoint_relational.pt"
            save_agent(
                output / selected_name,
                calibrated,
                cfg,
                eligible=True,
                completed_episodes=nominal_completed,
                validation=activation["selected_info"],
                training_backend=training_backend,
            )
    ppo_steps = environment_steps
    activation_steps = 0 if activation is None else activation["environment_steps"]
    # Whether the arm that gets DEPLOYED still carries a live relational path.
    # Measured on the selected checkpoint, not on the live agent, because the
    # selection can fall back to a pre-activation checkpoint.
    health = {"applicable": mode == "ppo_ontology_rgat", "measured": False}
    if mode == "ppo_ontology_rgat" and selected_name:
        deployed, _meta = load_agent(output / selected_name, cfg)
        health = relational_health(deployed, cfg, env_factory=env_factory)
        health["measured"] = True
        health["adaptation_iterations"] = max(hyper.iterations - warmup, 0)
        health["adaptation_fraction_of_budget"] = (
            health["adaptation_iterations"] / hyper.iterations)
        state = ("ACTIVE" if health.get("active") else
                 "INERT -- this arm is numerically ppo_semantic_flat")
        print(f"[spatial train] {mode} seed={seed} relational path {state}; "
              f"trainable for {health['adaptation_iterations']}/"
              f"{hyper.iterations} iterations", flush=True)
    summary = {
        "mode": mode,
        "seed": seed,
        "signature": cfg.signature,
        "hyperparameters": asdict(hyper),
        "episodes_per_iteration": episodes_per_iteration,
        "curriculum_loss_timeout_start": curriculum_loss_timeout_start,
        "rollout_budget_unit": "complete_episodes" if episodes_per_iteration else "decisions",
        "initialization": agent.initialization,
        "initialization_source": initialization_source,
        "completed_nominal_episodes": nominal_completed,
        "selected_checkpoint": selected_name,
        "environment_steps": ppo_steps,
        "total_training_environment_steps": ppo_steps
        + pretraining["environment_steps"]
        + activation_steps,
        "training_backend": training_backend,
        "curriculum_enabled": curriculum_enabled,
        "curriculum_angular_scales": list(curriculum_angular_scales),
        "adaptation_starts_at_iteration": warmup + 1,
        "pretraining": pretraining,
        "relation_activation": activation,
        # Reading this is how you know the three-arm comparison had three arms.
        "relational_health": health,
        "history": history,
        "wall_seconds": time.monotonic() - started,
        "comparison_factors": [
            "state_representation",
            "causal_masked_pretraining",
            "staged_graph_adaptation",
            "guarded_relation_only_ppo",
            "validation_readout_calibration",
        ],
    }
    temporary = output / "summary.json.partial"
    temporary.write_text(json_text(summary, indent=2) + "\n")
    temporary.replace(output / "summary.json")
    return summary
