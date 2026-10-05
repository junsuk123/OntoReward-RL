"""PPO training driver shared bit-for-bit by all three representation arms.

Environment, reward, supervisor, rollout/optimizer machinery and evaluation
are shared. The v2.8 profile additionally declares graph-only masked pretraining,
staged adaptation and a selection margin. These are recorded comparison factors,
not hidden under a representation-only claim.

Two defects in the bounded `ppo_minibatch_update` helper made it unusable for a
real run, and both are fixed here rather than in that helper, which the
contract tests still exercise as a single-step numerical probe:

* it constructed fresh `Adam` optimisers on every call, so the moment estimates
  were discarded each iteration and the effective update was scaled SGD; and
* it took one full-batch gradient step, so the sampling ratio was exactly 1 on
  the only step taken and the PPO clip never engaged.

No behaviour cloning, no teacher, no PN action, no privileged observation. The
policy sees the causal packet or the context graph built from it, nothing else.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .config import ExperimentConfig, load_config
from .artifacts import json_text
from .contracts import experiment_signature
from .curriculum import (CurriculumScheduler, EpisodeRole,
                         checkpoint_is_eligible, landing_rates,
                         replay_difficulty)
from .environment import ObservationBundle, TwoAxisLandingEnv
from .learning import Transition, time_aware_gae
from .models import POLICY_MODES, TwoAxisActor, TwoAxisPPOAgent, observation_arrays
from .ontology_v28 import schema_hash


@dataclass(frozen=True)
class PPOHyperparameters:
    """One object, passed unchanged to every arm."""

    iterations: int = 2000
    decisions_per_iteration: int = 2048
    epochs: int = 4
    minibatch_size: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    clip_ratio: float = 0.2
    entropy_coefficient: float = 0.005
    gae_lambda: float = 0.95
    max_grad_norm: float = 1.0
    target_kl: float = 0.02
    initial_log_std: float = -0.7
    evaluation_every: int = 50
    evaluation_episodes: int = 12
    value_warmup_iterations: int = 0
    advantage_normalization: str = "minibatch"

    def __post_init__(self):
        if min(self.iterations, self.decisions_per_iteration, self.epochs,
               self.minibatch_size, self.evaluation_every, self.evaluation_episodes) <= 0:
            raise ValueError("PPO budgets must be positive")
        if self.decisions_per_iteration < 2 or self.minibatch_size < 2:
            raise ValueError("PPO needs at least two samples per minibatch")
        if self.value_warmup_iterations < 0:
            raise ValueError("value warmup cannot be negative")
        if self.advantage_normalization not in ("minibatch", "rollout"):
            raise ValueError("advantage normalization must be minibatch or rollout")


@dataclass
class IterationReport:
    iteration: int
    difficulty: float
    environment_steps: int
    mean_reward: float
    actor_loss: float
    critic_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    explained_variance: float
    episodes: int
    rates: dict[str, float] = field(default_factory=dict)
    validation: dict[str, float] | None = None


class EpisodeDrivenCollector:
    """Rollout collection that honours the curriculum and records outcomes.

    Unlike the bounded helper, each episode is opened at the difficulty the
    scheduler assigns and its terminal reason is reported back, so promotion
    depends on outcomes at the active difficulty and not on easy replays.
    """

    def __init__(self, env: TwoAxisLandingEnv, scheduler: CurriculumScheduler,
                 *, base_seed: int):
        self.env = env
        self.scheduler = scheduler
        self.base_seed = int(base_seed)
        self.episode_index = 0
        self._observation: ObservationBundle | None = None
        self._role: EpisodeRole | None = None
        self._difficulty = 0.0
        self.completed: list[tuple[EpisodeRole, float, str]] = []

    def _open_episode(self) -> ObservationBundle:
        role = self.scheduler.role_for(self.episode_index)
        difficulty = replay_difficulty(self.scheduler.difficulty, role)
        seed = self.base_seed + self.episode_index
        observation, _info = self.env.reset(seed=seed, difficulty=difficulty)
        self._role, self._difficulty = role, difficulty
        self.episode_index += 1
        return observation

    def collect(self, agent: TwoAxisPPOAgent, decisions: int) -> list[Transition]:
        if decisions <= 0:
            raise ValueError("rollout decisions must be positive")
        self.completed = []
        if self._observation is None:
            self._observation = self._open_episode()
        observation = self._observation
        transitions: list[Transition] = []
        while len(transitions) < decisions:
            raw, normalized, logp, value = agent.act(observation)
            next_observation, reward, terminated, truncated, info = self.env.step(
                normalized)
            transitions.append(Transition(
                observation=observation, raw_command=raw,
                normalized_command=normalized,
                requested_acceleration_m_s2=np.asarray(
                    info["requested_acceleration_m_s2"], dtype=float),
                applied_acceleration_m_s2=np.asarray(
                    info["applied_acceleration_m_s2"], dtype=float),
                old_log_probability=logp, value=value, reward=float(reward),
                reward_components=dict(info["reward_components"]),
                next_observation=next_observation, terminated=terminated,
                truncated=truncated, dt_s=float(info["dt_s"]),
                safety_flags={"intervened": info["safety_intervened"],
                              "reasons": info["safety_reasons"],
                              "abort_requested": info["abort_requested"]}))
            if terminated:
                assert self._role is not None
                self.completed.append(
                    (self._role, self._difficulty, str(info["status"])))
                self.scheduler.record(
                    role=self._role, landed=info["status"] == "SUCCESS")
                observation = self._open_episode()
            else:
                observation = next_observation
        self._observation = observation
        last = transitions[-1]
        if not last.terminated:
            # External rollout boundary, not a task terminal: bootstrap once so
            # the truncated episode is not treated as a failure.
            last.truncated = True
            _, _, _, last.next_value = agent.act(
                last.next_observation, deterministic=True)
        return transitions


class PPOTrainer:
    """Persistent-optimiser PPO over minibatches, identical for every arm."""

    def __init__(self, agent: TwoAxisPPOAgent, hyper: PPOHyperparameters):
        self.agent = agent
        self.hyper = hyper
        self.minibatch_generator = torch.Generator(device="cpu").manual_seed(agent.seed+3)
        self.actor_optimizer = torch.optim.Adam(
            agent.actor.parameters(), lr=hyper.actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            agent.critic.parameters(), lr=hyper.critic_lr)
        self.updates = 0

    def update(self, transitions: list[Transition], *,
               discount_time_constant_s: float) -> dict[str, float]:
        hyper = self.hyper
        self.updates += 1
        update_policy = self.updates > hyper.value_warmup_iterations
        advantages, returns = time_aware_gae(
            transitions, discount_time_constant_s=discount_time_constant_s,
            gae_lambda=hyper.gae_lambda)
        device = self.agent.device
        packet_np, graph_np = observation_arrays(
            [item.observation for item in transitions])
        packets = torch.as_tensor(packet_np, dtype=torch.float32, device=device)
        graphs = torch.as_tensor(graph_np, dtype=torch.float32, device=device)
        raw = torch.as_tensor(
            np.stack([item.raw_command for item in transitions]),
            dtype=torch.float32, device=device)
        old_logp = torch.as_tensor(
            [item.old_log_probability for item in transitions],
            dtype=torch.float32, device=device)
        advantage = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        if hyper.advantage_normalization == "rollout":
            # Upstream ppoTrain.m normalizes A once before shuffling epochs.
            advantage = (advantage-advantage.mean())/(advantage.std()+1e-8)
        target = torch.as_tensor(returns, dtype=torch.float32, device=device)
        values = torch.as_tensor([item.value for item in transitions],
                                 dtype=torch.float32, device=device)

        count = len(transitions)
        totals = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0,
                  "approx_kl": 0.0, "clip_fraction": 0.0}
        batches = 0
        generator = self.minibatch_generator
        stop = False
        for _epoch in range(hyper.epochs):
            order = torch.randperm(count, generator=generator).to(device)
            for start in range(0, count, hyper.minibatch_size):
                index = order[start:start + hyper.minibatch_size]
                if index.numel() < 2:
                    continue
                batch_advantage = advantage[index]
                if hyper.advantage_normalization == "minibatch":
                    batch_advantage = ((batch_advantage - batch_advantage.mean())
                                       / (batch_advantage.std() + 1e-8))
                mu, std = self.agent.actor(packets[index], graphs[index])
                logp = TwoAxisActor.raw_log_probability(raw[index], mu, std)
                ratio = torch.exp(logp - old_logp[index])
                clipped = ratio.clamp(1.0 - hyper.clip_ratio,
                                      1.0 + hyper.clip_ratio)
                objective = torch.minimum(ratio * batch_advantage,
                                          clipped * batch_advantage)
                entropy = torch.log(std * math.sqrt(2.0 * math.pi * math.e)).sum(-1).mean()
                actor_loss = -objective.mean() - hyper.entropy_coefficient * entropy
                if update_policy:
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.agent.actor.parameters(), hyper.max_grad_norm)
                    self.actor_optimizer.step()

                predicted = self.agent.critic(packets[index], graphs[index])
                critic_loss = 0.5 * ((predicted - target[index]) ** 2).mean()
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.agent.critic.parameters(), hyper.max_grad_norm)
                self.critic_optimizer.step()

                with torch.no_grad():
                    approx_kl = float(((ratio - 1.0) - (logp - old_logp[index])).mean())
                    totals["approx_kl"] += approx_kl
                    totals["clip_fraction"] += float(
                        ((ratio - 1.0).abs() > hyper.clip_ratio).float().mean())
                totals["actor_loss"] += float(actor_loss.detach())
                totals["critic_loss"] += float(critic_loss.detach())
                totals["entropy"] += float(entropy.detach())
                batches += 1
                if approx_kl > 1.5 * hyper.target_kl:
                    # Early stop on the trust region, the standard guard; it is
                    # applied identically to every arm.
                    stop = True
                    break
            if stop:
                break
        metrics = {key: value / max(batches, 1) for key, value in totals.items()}
        residual = float(((target - values) ** 2).mean())
        variance = float(target.var())
        metrics["explained_variance"] = (
            1.0 - residual / variance if variance > 1e-12 else 0.0)
        metrics["minibatches"] = float(batches)
        metrics["policy_updated"] = float(update_policy)
        if not all(np.isfinite(value) for value in metrics.values()):
            raise FloatingPointError(f"non-finite PPO metrics: {metrics}")
        return metrics


def evaluate_policy(agent: TwoAxisPPOAgent, config: ExperimentConfig, *,
                    seeds: list[int], difficulty: float = 1.0,
                    deterministic: bool = True) -> dict[str, Any]:
    """Nominal-difficulty evaluation; never an easier task than the contract."""
    if difficulty != 1.0:
        raise ValueError("validation/test must use nominal difficulty 1.0")
    if not seeds:
        raise ValueError("evaluation requires nonempty explicit seeds")
    env = TwoAxisLandingEnv(config, perturbations=True)
    unsafe = {"UNSAFE_CONTACT", "UNAUTHORIZED_CONTACT", "MISSED_PAD_CONTACT",
              "SAFETY_ENVELOPE_VIOLATION"}
    outcomes: list[str] = []
    durations: list[float] = []
    fov: list[float] = []
    intervention: list[float] = []
    errors: list[float] = []
    returns: list[float] = []
    relation_means: list[np.ndarray] = []
    for seed in seeds:
        observation, _ = env.reset(seed=int(seed), difficulty=difficulty)
        seen: list[float] = []
        intervened: list[float] = []
        episode_return = 0.0
        relation_sum = np.zeros(2)
        relation_count = 0
        while True:
            if hasattr(agent.actor, "relational_delta") and agent.mode == "ppo_ontology_rgat":
                with torch.no_grad():
                    _, graphs = agent.tensors(observation)
                    relation_sum += agent.actor.relational_delta(graphs)[0].abs().cpu().numpy()
                    relation_count += 1
            _raw, normalized, _logp, _value = agent.act(
                observation, deterministic=deterministic)
            observation, _reward, terminated, _truncated, info = env.step(normalized)
            episode_return += float(_reward)
            assert env.track is not None and env.state is not None
            seen.append(float(env.track.time_since_detection_s <= 1e-9))
            intervened.append(float(info["safety_intervened"]))
            if terminated:
                outcomes.append(str(info["status"]))
                durations.append(float(env.state.time_s))
                pad_x, _v, _a = env.scenario.state_at(env.state.time_s)
                errors.append(abs(pad_x - env.state.x_m))
                returns.append(episode_return)
                break
        fov.append(float(np.mean(seen)))
        intervention.append(float(np.mean(intervened)))
        relation_means.append(relation_sum / max(relation_count, 1))
    landed = [i for i, reason in enumerate(outcomes) if reason == "SUCCESS"]
    return {
        "episodes": float(len(outcomes)),
        "mean_return": float(np.mean(returns)),
        "selection_score": selection_score(outcomes, returns),
        # Mean of each episode's mean raw-policy residual, matching evaluateV2.
        "mean_abs_relation_residual": np.mean(relation_means, axis=0).tolist(),
        "landing_rate": float(np.mean([r == "SUCCESS" for r in outcomes])),
        "safe_abort_rate": float(np.mean([r == "SAFE_ABORT" for r in outcomes])),
        "task_timeout_rate": float(np.mean([r == "TASK_TIMEOUT" for r in outcomes])),
        "unsafe_rate": float(np.mean([r in unsafe for r in outcomes])),
        "fov_capture_rate": float(np.mean(fov)),
        "supervisor_intervention_rate": float(np.mean(intervention)),
        "mean_landing_time_s": (float(np.mean([durations[i] for i in landed]))
                                if landed else float("nan")),
        "mean_terminal_error_m": (float(np.mean([errors[i] for i in landed]))
                                  if landed else float("nan")),
    }


def selection_score(outcomes, returns):
    """Upstream selectionScoreV2: safety outcomes are not 1:1 with success."""
    counts = {reason: sum(o == reason for o in outcomes)/len(outcomes)
              for reason in ("SUCCESS", "SAFE_ABORT", "TASK_TIMEOUT")}
    unsafe = 1-sum(counts.values())
    return (1000*counts["SUCCESS"]-2500*unsafe-10*counts["TASK_TIMEOUT"]
            -100*counts["SAFE_ABORT"]+float(np.mean(returns)))


# Three disjoint held-out seed blocks, each used for exactly one purpose, so
# no number that appears in the headline claim was optimised against.
#
#   TUNING      hyperparameter search only
#   VALIDATION  checkpoint selection during training
#   TEST        the final held-out claim, scored once per run
#
# Training draws from none of them, and every split runs at difficulty 1.0.
TUNING_SEEDS = list(range(4000, 4024))
VALIDATION_SEEDS = list(range(2000, 2040))
TEST_SEEDS = list(range(3000, 3040))
TRAIN_SEED_BASE = 100_000


def train_arm(mode: str, *, seed: int, hyper: PPOHyperparameters,
              output_dir: Path, config: ExperimentConfig | None = None,
              progress_every: int = 10) -> dict[str, Any]:
    """Train one arm from a random initialisation. No demonstrations exist."""
    if mode not in POLICY_MODES:
        raise ValueError(f"unknown arm {mode}")
    config = config or load_config()
    output_dir = Path(output_dir)
    if (output_dir / "training_log.jsonl").exists() or (output_dir / "checkpoint_final.pt").exists():
        raise FileExistsError(f"run already exists; select a fresh output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2 ** 32))

    agent = TwoAxisPPOAgent(mode, seed=int(seed),
                            initial_log_std=hyper.initial_log_std,
                            graph_config=config.ontology)
    from .pretraining import pretrain_causal_encoder
    pretraining = pretrain_causal_encoder(agent, config, seed=int(seed))
    trainer = PPOTrainer(agent, hyper)
    env = TwoAxisLandingEnv(config, perturbations=True)
    scheduler = CurriculumScheduler(config.curriculum, difficulty=0.0)
    collector = EpisodeDrivenCollector(
        env, scheduler, base_seed=TRAIN_SEED_BASE + 10_000 * int(seed))
    signature = experiment_signature(config, graph_schema_hash=schema_hash(config))

    history: list[dict[str, Any]] = []
    all_episodes: list[tuple[EpisodeRole, float, str]] = []
    best: dict[str, Any] | None = None
    environment_steps = 0
    log_path = output_dir / "training_log.jsonl"
    with log_path.open("w", encoding="utf-8") as log:
        for iteration in range(1, hyper.iterations + 1):
            scheduler.set_budget_progress((iteration-1)/hyper.iterations)
            adapting = iteration > math.ceil(
                config.ontology.adaptation_warmup_fraction*hyper.iterations)
            for head in (agent.actor, agent.critic):
                if hasattr(head, "set_adaptation"):
                    head.set_adaptation(config.ontology, adapting)
            rollout = collector.collect(agent, hyper.decisions_per_iteration)
            environment_steps += len(rollout)
            all_episodes.extend(collector.completed)
            metrics = trainer.update(
                rollout,
                discount_time_constant_s=config.timing.discount_time_constant_s)
            rates = landing_rates(all_episodes[-400:])
            report = IterationReport(
                iteration=iteration, difficulty=scheduler.difficulty,
                environment_steps=environment_steps,
                mean_reward=float(np.mean([t.reward for t in rollout])),
                actor_loss=metrics["actor_loss"], critic_loss=metrics["critic_loss"],
                entropy=metrics["entropy"], approx_kl=metrics["approx_kl"],
                clip_fraction=metrics["clip_fraction"],
                explained_variance=metrics["explained_variance"],
                episodes=len(collector.completed), rates=rates)
            # Checkpoint selection: only a policy that has reached the full
            # nominal task is eligible, so an easy-curriculum policy can never
            # be chosen as the final model.
            if (iteration % hyper.evaluation_every == 0
                    or iteration == hyper.iterations):
                validation = evaluate_policy(
                    agent, config, seeds=VALIDATION_SEEDS[:hyper.evaluation_episodes],
                    difficulty=1.0)
                report.validation = validation
                nominal_seen = any(d >= 1.0 for _role,d,_reason in all_episodes)
                if checkpoint_is_eligible(scheduler.difficulty) and nominal_seen:
                    score = (validation["landing_rate"],
                             -validation["unsafe_rate"],
                             -validation["safe_abort_rate"])
                    if config.ontology.selection_score_model == "outcome_weighted_v28":
                        score = (validation["selection_score"],)
                    margin = (config.ontology.selection_score_margin
                        if mode == "ppo_ontology_rgat" and adapting else 0.0)
                    improved = best is None or (
                        score > tuple(best["score"]) and (margin == 0.0 or (
                            score[0] > best["score"][0] + margin)))
                    if improved:
                        best = {"iteration": iteration, "score": list(score),
                                "difficulty": scheduler.difficulty,
                                "validation": validation}
                        torch.save({
                            "state_dict": agent.state_dict(),
                            "mode": mode, "seed": int(seed),
                            "iteration": iteration,
                            "curriculum_difficulty": scheduler.difficulty,
                            "signature": asdict(signature),
                            "hyperparameters": asdict(hyper),
                            "graph_config": asdict(config.ontology),
                        }, output_dir / "checkpoint_best.pt")
            row = asdict(report)
            log.write(json_text(row) + "\n")
            log.flush()
            history.append(row)
            if iteration % progress_every == 0 or iteration == 1:
                print(f"[{mode} seed={seed}] it={iteration}/{hyper.iterations} "
                      f"d={scheduler.difficulty:.2f} r={report.mean_reward:+.4f} "
                      f"land={rates['overall']:.2f} nom={rates['nominal']:.2f} "
                      f"kl={metrics['approx_kl']:.4f} "
                      f"ev={metrics['explained_variance']:+.2f}", flush=True)
    torch.save({
        "state_dict": agent.state_dict(), "mode": mode, "seed": int(seed),
        "iteration": hyper.iterations,
        "curriculum_difficulty": scheduler.difficulty,
        "signature": asdict(signature), "hyperparameters": asdict(hyper),
        "graph_config": asdict(config.ontology),
    }, output_dir / "checkpoint_final.pt")
    activation = {"attempted": False, "changed": False, "environment_steps": 0,
                  "reason": "disabled, non-graph arm, or no nominal selected checkpoint"}
    if (config.ontology.relation_activation_enabled and best is not None
            and mode == "ppo_ontology_rgat"):
        from .relational import ensure_relational_path
        anchor, anchor_payload = load_agent(output_dir / "checkpoint_best.pt", config)
        selected, activation = ensure_relational_path(
            anchor, config, hyper, seed=int(seed),
            validation_seeds=VALIDATION_SEEDS[:hyper.evaluation_episodes])
        if activation["changed"]:
            # Preserve the pre-activation checkpoint and its training history.
            name = "checkpoint_relational.pt"
            validation = activation["guard"]["selected_info"]
            torch.save({**anchor_payload, "state_dict": selected.state_dict(),
                        "relation_activation": activation}, output_dir / name)
            best = {**best, "checkpoint": name, "anchor_validation": best["validation"],
                    "validation": validation, "score": [validation["selection_score"]],
                    "relation_scale": activation["guard"]["selected_scale"]}
    elapsed = time.perf_counter() - started
    summary = {
        "arm": mode, "seed": int(seed),
        "training_time_s": elapsed,
        "environment_steps": environment_steps,
        "final_difficulty": scheduler.difficulty,
        "reached_nominal": bool(scheduler.at_nominal),
        "promotions": scheduler.promotions,
        "episodes": len(all_episodes),
        "training_rates_overall": landing_rates(all_episodes),
        "training_rates_last_400": landing_rates(all_episodes[-400:]),
        "selected_checkpoint": best,
        "parameter_count": agent.parameter_count(),
        "signature": asdict(signature),
        "hyperparameters": asdict(hyper),
        "experiment": config.experiment,
        "config": config.canonical_dict,
        "pretraining": pretraining,
        "relation_activation": activation,
        "total_environment_steps_including_pretraining": environment_steps + pretraining["environment_steps"]
            + activation["environment_steps"],
        "validation_seeds": VALIDATION_SEEDS[:hyper.evaluation_episodes],
        "comparison_factors": ["state_representation"] + (
            ["staged_graph_adaptation", "graph_checkpoint_margin", "causal_masked_pretraining"]
            if config.ontology.adaptation_warmup_fraction > 0 else []) + (
            ["guarded_relation_only_ppo", "validation_readout_calibration"]
            if config.ontology.relation_activation_enabled else []),
    }
    (output_dir / "summary.json").write_text(
        json_text(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def load_agent(checkpoint_path: Path, config: ExperimentConfig
               ) -> tuple[TwoAxisPPOAgent, dict[str, Any]]:
    """Load a checkpoint and refuse one that was not trained on this contract."""
    payload = torch.load(Path(checkpoint_path), map_location="cpu",
                         weights_only=True)
    signature = experiment_signature(config, graph_schema_hash=schema_hash(config))
    signature.assert_compatible(payload["signature"])
    agent = TwoAxisPPOAgent(
        payload["mode"], seed=int(payload["seed"]),
        initial_log_std=float(payload["hyperparameters"].get(
            "initial_log_std", -0.7)), graph_config=config.ontology)
    agent.load_state_dict(payload["state_dict"])
    agent.eval()
    return agent, payload


def measure_inference_time_s(agent: TwoAxisPPOAgent,
                             config: ExperimentConfig, *,
                             decisions: int = 500) -> float:
    """Mean wall-clock seconds for one policy decision."""
    env = TwoAxisLandingEnv(config, perturbations=False)
    observation, _ = env.reset(seed=999_001, difficulty=1.0)
    for _ in range(20):  # warm up
        agent.act(observation, deterministic=True)
    started = time.perf_counter()
    for _ in range(decisions):
        agent.act(observation, deterministic=True)
    return (time.perf_counter() - started) / decisions


# Identical search space and identical budget for every arm. The baseline gets
# exactly the same number of trials, the same grid and the same scoring rule as
# the proposed model, so a win cannot come from one arm having been tuned
# harder. Scored on TUNING_SEEDS, which never appear in validation or test.
SEARCH_SPACE: tuple[dict[str, Any], ...] = (
    {"actor_lr": 3e-4, "critic_lr": 1e-3, "entropy_coefficient": 0.005},
    {"actor_lr": 3e-4, "critic_lr": 1e-3, "entropy_coefficient": 0.02},
    {"actor_lr": 1e-3, "critic_lr": 3e-3, "entropy_coefficient": 0.005},
    {"actor_lr": 1e-3, "critic_lr": 3e-3, "entropy_coefficient": 0.02},
    {"actor_lr": 1e-4, "critic_lr": 5e-4, "entropy_coefficient": 0.005},
    {"actor_lr": 1e-3, "critic_lr": 3e-3, "entropy_coefficient": 0.05},
)


def tuning_score(summary: dict[str, Any], probe: dict[str, float]) -> float:
    """Rank trials by progress towards the nominal task, not by raw reward.

    Reward alone rewards a policy that survives without landing. The ranking
    is: nominal-difficulty landing first, then how far the curriculum got,
    then the tuning-split landing rate, then a small unsafe-rate penalty.
    """
    return (3.0 * float(probe.get("landing_rate", 0.0))
            + 2.0 * float(summary["final_difficulty"])
            + 1.0 * float(summary["training_rates_last_400"]["nominal"])
            - 1.0 * float(probe.get("unsafe_rate", 0.0)))
