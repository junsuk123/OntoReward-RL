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
        self, mode, config, seed=42, initial_log_std=-0.7, initialization=INITIALIZATION
    ):
        super().__init__()
        if mode not in POLICY_MODES:
            raise ValueError("unknown spatial arm")
        self.mode, self.seed, self.device = mode, int(seed), torch.device("cpu")
        self.initialization = initialization
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
            graph_planes=2 if config.reference_context else 1,
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
            graph_planes=2 if config.reference_context else 1,
        )
        self._generator = torch.Generator(device=self.device).manual_seed(seed + 2)


def save_agent(path, agent, cfg, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "signature": cfg.signature,
        "mode": agent.mode,
        "seed": agent.seed,
        "initialization": agent.initialization,
        "state_dict": agent.state_dict(),
        "config": asdict(cfg),
        "metadata": {
            "initialization_source": getattr(agent, "_training_origin", None),
            **metadata,
        },
    }
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_agent(path, cfg):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("signature") != cfg.signature:
        raise ValueError("checkpoint is not compatible with the exact spatial contract")
    agent = SpatialAgent(
        payload["mode"],
        cfg,
        payload["seed"],
        initialization=payload.get("initialization", "torch-default"),
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
                obs, reward, done, _, info = env.step(action)
                total += reward
                if trace is not None:
                    trace(agent.mode, seed, step, action, info)
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
):
    if not 0 <= episodes_per_iteration <= 64:
        raise ValueError("episodes per iteration must be in [0,64]")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob("checkpoint_*.pt")):
        raise ValueError(
            "choose fresh spatial output; existing checkpoints are never overwritten"
        )
    initialization_source = None
    if initial_checkpoint is None:
        agent = SpatialAgent(mode, cfg, seed, hyper.initial_log_std)
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
