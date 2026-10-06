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

import copy
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
from .learning import Transition, time_aware_gae, executed_axis_mask
from .models import POLICY_MODES, TwoAxisActor, TwoAxisPPOAgent, observation_arrays, log_probability_axes
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
    #: "none" keeps the shipped objective: PPO regresses on every proposed
    #: action, including the ones the supervisor replaced before they reached
    #: the plant (38-44 % of 3D steps, 54 % of 2D, and 73-99 % on the rungs
    #: `tools/audit_exploration_reachability.py` measures). "per_axis" drops
    #: only the axes that were actually overridden. A per-STEP mask was tried
    #: and is the wrong shape: with a symmetric vertical hold it discards 92 %
    #: of steps and the actor stops moving.
    intervention_masking: str = "none"
    #: Linear ceiling on log_std from `initial_log_std` down to this over
    #: `log_std_anneal_fraction` of the run. None keeps exploration fixed,
    #: which is the shipped behaviour. Measured 2026-10-06: at the shipped
    #: sigma 0.333 a clone whose mean lands 95.8 % lands 0.0 % sampled, so
    #: every batch is success-free; annealing is how a success enters one.
    final_log_std: float | None = None
    log_std_anneal_fraction: float = 0.5
    #: The shipped early stop reads the trust region one minibatch LATE: a
    #: minibatch's `approx_kl` is measured from the ratio BEFORE its own step,
    #: so the first actor step of every iteration is unconstrained. Its size in
    #: action space is set by `actor_lr`, not by sigma, so once sigma is small
    #: one step moves the mean by several sigmas. Measured 2026-10-06 in
    #: results/full_pipeline_20261006_lowsigma_unconstrained (sigma pinned at
    #: 0.030 from iteration 5): per-iteration KL 0.5-3.4 against the 0.02
    #: target, and both graph arms collapsed from a 62.5 %-landing clone to
    #: 12/12 SAFE_ABORT at 118 steps within two iterations. True re-measures
    #: the KL on the minibatch AFTER each actor step, reverts a step above
    #: 1.5 * target_kl (parameters and optimiser moments) and halves the actor
    #: learning rate when nothing had been accepted yet that iteration (a
    #: rejection after accepted steps is the ordinary early stop); an
    #: iteration that completes under target_kl / 2 grows it 1.5x, never
    #: above `actor_lr`. Off keeps the shipped objective.
    enforce_target_kl: bool = False

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
        if self.intervention_masking not in ("none", "per_axis"):
            raise ValueError("intervention masking must be none or per_axis")
        if self.final_log_std is not None:
            if self.final_log_std > self.initial_log_std:
                raise ValueError("annealing must lower exploration, not raise it")
            if not 0.0 < self.log_std_anneal_fraction <= 1.0:
                raise ValueError("anneal fraction must be in (0,1]")

    def log_std_ceiling(self, iteration: int) -> float:
        """Upper clamp on log_std at `iteration` (1-based); 1.0 means inert."""
        if self.final_log_std is None:
            return 1.0
        span = max(1.0, self.log_std_anneal_fraction * self.iterations)
        progress = min(1.0, max(0.0, (iteration - 1) / span))
        return (self.initial_log_std
                + progress * (self.final_log_std - self.initial_log_std))


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
                 *, base_seed: int, per_axis_evidence: bool = False):
        self.env = env
        self.scheduler = scheduler
        self.base_seed = int(base_seed)
        self.per_axis_evidence = bool(per_axis_evidence)
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
            if self.per_axis_evidence:
                transitions[-1].old_log_probability_axes = log_probability_axes(
                    agent, transitions[-1].observation,
                    transitions[-1].raw_command)
                transitions[-1].executed_axes = executed_axis_mask(
                    transitions[-1].requested_acceleration_m_s2,
                    transitions[-1].applied_acceleration_m_s2)
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
        #: Live actor learning rate; only `enforce_target_kl` ever moves it.
        self.actor_lr = float(hyper.actor_lr)

    def _actor_snapshot(self):
        return ([p.detach().clone() for p in self.agent.actor.parameters()],
                copy.deepcopy(self.actor_optimizer.state_dict()))

    def _restore_actor(self, snapshot):
        parameters, optimizer_state = snapshot
        with torch.no_grad():
            for live, saved in zip(self.agent.actor.parameters(), parameters):
                live.copy_(saved)
        self.actor_optimizer.load_state_dict(optimizer_state)

    def _set_actor_lr(self, value: float):
        self.actor_lr = float(min(max(value, 1e-7), self.hyper.actor_lr))
        for group in self.actor_optimizer.param_groups:
            group["lr"] = self.actor_lr

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
        masking = hyper.intervention_masking
        executed = old_logp_axes = None
        if masking == "per_axis":
            missing = [item for item in transitions
                       if item.executed_axes is None
                       or item.old_log_probability_axes is None]
            if missing:
                raise ValueError(
                    "per-axis masking needs executed_axes and "
                    "old_log_probability_axes on every transition")
            executed = torch.as_tensor(
                np.stack([item.executed_axes for item in transitions]),
                dtype=torch.float32, device=device)
            old_logp_axes = torch.as_tensor(
                np.stack([item.old_log_probability_axes for item in transitions]),
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
        rejected = 0
        kl_after_total, kl_after_count = 0.0, 0

        def density(index, mu, std):
            """Log-density of the stored proposals under (mu, std), and the
            stored reference it is compared with -- masked identically."""
            if masking == "per_axis":
                mask = executed[index]
                return ((TwoAxisActor.raw_log_probability_axes(
                    raw[index], mu, std) * mask).sum(-1),
                        (old_logp_axes[index] * mask).sum(-1))
            return (TwoAxisActor.raw_log_probability(raw[index], mu, std),
                    old_logp[index])

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
                # Per-axis masking drops the SAME axes from numerator and
                # denominator, so the ratio stays a ratio of densities over the
                # axes the plant actually ran. A step whose every axis was
                # replaced contributes ratio 1 and therefore no actor gradient.
                logp, reference = density(index, mu, std)
                ratio = torch.exp(logp - reference)
                clipped = ratio.clamp(1.0 - hyper.clip_ratio,
                                      1.0 + hyper.clip_ratio)
                objective = torch.minimum(ratio * batch_advantage,
                                          clipped * batch_advantage)
                entropy = torch.log(std * math.sqrt(2.0 * math.pi * math.e)).sum(-1).mean()
                actor_loss = -objective.mean() - hyper.entropy_coefficient * entropy
                if update_policy:
                    snapshot = (self._actor_snapshot()
                                if hyper.enforce_target_kl else None)
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.agent.actor.parameters(), hyper.max_grad_norm)
                    self.actor_optimizer.step()
                    if hyper.enforce_target_kl:
                        # The trust region, measured where it applies: on the
                        # policy this step actually produced.
                        with torch.no_grad():
                            mu_after, std_after = self.agent.actor(
                                packets[index], graphs[index])
                            logp_after, _ = density(index, mu_after, std_after)
                            shift = logp_after - reference
                            kl_after = float(((shift.exp() - 1.0) - shift).mean())
                        if not math.isfinite(kl_after) or kl_after > 1.5 * hyper.target_kl:
                            self._restore_actor(snapshot)
                            if kl_after_count == 0:
                                # Nothing was accepted this iteration: the
                                # rate itself is too large for this sigma.
                                # A rejection AFTER accepted steps is the
                                # ordinary early stop -- the iteration made
                                # progress -- and must not shrink the rate,
                                # or it spirals to the floor and the actor
                                # freezes.
                                self._set_actor_lr(self.actor_lr * 0.5)
                            rejected += 1
                            stop = True
                            break
                        kl_after_total += kl_after
                        kl_after_count += 1

                predicted = self.agent.critic(packets[index], graphs[index])
                critic_loss = 0.5 * ((predicted - target[index]) ** 2).mean()
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.agent.critic.parameters(), hyper.max_grad_norm)
                self.critic_optimizer.step()

                with torch.no_grad():
                    approx_kl = float(((ratio - 1.0) - (logp - reference)).mean())
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
        if (hyper.enforce_target_kl and update_policy and not rejected
                and kl_after_count
                and kl_after_total / kl_after_count < 0.5 * hyper.target_kl):
            self._set_actor_lr(self.actor_lr * 1.5)
        metrics = {key: value / max(batches, 1) for key, value in totals.items()}
        metrics["kl_after_step"] = kl_after_total / max(kl_after_count, 1)
        metrics["accepted_steps"] = float(kl_after_count)
        metrics["rejected_steps"] = float(rejected)
        metrics["actor_lr"] = self.actor_lr
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
        env, scheduler, base_seed=TRAIN_SEED_BASE + 10_000 * int(seed),
        per_axis_evidence=hyper.intervention_masking == "per_axis")
    signature = experiment_signature(config, graph_schema_hash=schema_hash(config))

    history: list[dict[str, Any]] = []
    all_episodes: list[tuple[EpisodeRole, float, str]] = []
    best: dict[str, Any] | None = None
    environment_steps = 0
    log_path = output_dir / "training_log.jsonl"
    with log_path.open("w", encoding="utf-8") as log:
        for iteration in range(1, hyper.iterations + 1):
            if hyper.final_log_std is not None:
                # Same schedule in both dimensions; the floor moves with the
                # ceiling so the actor's own clamp cannot pin sigma above it.
                ceiling = hyper.log_std_ceiling(iteration)
                agent.actor.maximum_log_std = ceiling
                agent.actor.minimum_log_std = min(
                    agent.actor.minimum_log_std, ceiling)
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
