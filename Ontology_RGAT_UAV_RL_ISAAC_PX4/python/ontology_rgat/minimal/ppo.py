"""PPO fine-tuning of the three arms on the minimal contract.

The recipe is the one measured on the spatial route (AGENTS.md), ported:

* complete-episode batches (``episodes_per_iteration``, 12): one-episode
  batches centre every advantage inside its own episode and nothing learns;
* a FIXED low exploration sigma (the arm's ``initial_log_std``, 0.082): at
  sigma 0.333 the /5 clones land 58-71 % sampled, at <= 0.135 96-100 %, and
  PPO only sees sampled rollouts; log_std is not trained;
* the trust region enforced AFTER each actor step: re-measure the KL on the
  minibatch with the new weights, revert the step above 1.5 x target_kl, halve
  the rate only when nothing was accepted that iteration, grow it 1.5x after a
  clean iteration (``PPOHyperparameters.enforce_target_kl`` of two_axis);
* per-axis intervention masking: an axis the supervisor replaced contributes
  no actor gradient (``SafetyStatus.intervened``);
* critic-only warm-up: a behaviour clone's value head is untrained, and only
  the HEAD is updated during warm-up so the shared encoder -- and with it the
  actor -- does not move.

Rollouts run in spawned worker processes (a forked child of a process that ran
torch can hang on an inherited thread-pool lock; measured).
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from .arms import build_arm, load_arm


@dataclass(frozen=True)
class MinimalPPOConfig:
    iterations: int = 100
    episodes_per_iteration: int = 12
    epochs: int = 4
    minibatch_size: int = 256
    learning_rate: float = 1e-4
    clip_ratio: float = 0.2
    target_kl: float = 0.02
    gae_lambda: float = 0.95
    discount_tau_s: float = 70.0
    value_coefficient: float = 0.5
    max_grad_norm: float = 1.0
    critic_warmup_iterations: int = 5
    validation_every: int = 10
    validation_seeds: tuple = tuple(range(3000, 3012))
    train_seed_base: int = 100_000
    workers: int = 3
    #: Training rollouts only: None = nominal, "dr" = per-episode randomized
    #: stress (stress.RandomizedStressBackend). Validation is always nominal.
    train_scenario: str | None = None


# ---------------------------------------------------------------- rollouts
def _rollout_worker(args):
    """Fly whole episodes with a given arm state; return per-step records."""
    name, config, state_dict, seeds, sample_seed, scenario = args
    torch.set_num_threads(1)
    from .arms import FlatGraphArm, GraphArm, MlpConfig, VectorArm
    from .graph_policy import GraphPolicyConfig
    from .rollout import ArmController, make_env
    arm = (GraphArm(GraphPolicyConfig(**config)) if name == "ppo_ontology_rgat"
           else (FlatGraphArm if name == "ppo_semantic_flat" else VectorArm)(MlpConfig(**config)))
    arm.load_state_dict(state_dict)
    arm.eval()
    rng = np.random.default_rng(sample_seed)
    env = make_env(scenario)
    episodes = []
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        ctl = ArmController(arm)
        steps, done, info = [], False, {}
        while not done:
            ctl.inputs(obs)
            x = ctl.last_input
            with torch.no_grad():
                mean, log_std, value = arm(x)
            std = log_std.exp().numpy()
            action = mean.numpy() + rng.normal(size=3) * std
            logp_axes = (-0.5 * ((action - mean.numpy()) / std) ** 2
                         - np.log(std) - 0.5 * math.log(2 * math.pi))
            obs, reward, done, info = env.step(action)
            steps.append((x, action.astype(np.float32), logp_axes.astype(np.float32),
                          float(value), float(reward),
                          np.array([not i for i in info["intervened"]], np.float32)))
        episodes.append({"seed": int(seed), "status": info["status"], "steps": steps})
    return episodes


def collect(name, arm, seeds, pool, workers, sample_seed, scenario=None):
    state = {k: v.detach().cpu() for k, v in arm.state_dict().items()}
    chunks = [seeds[i::workers] for i in range(workers)]
    jobs = [(name, asdict(arm.config), state, chunk, sample_seed + k, scenario)
            for k, chunk in enumerate(chunks) if chunk]
    return [episode for part in pool.map(_rollout_worker, jobs) for episode in part]


def gae(rewards, values, gamma, lam):
    """Every episode ends in a scored terminal, so the last bootstrap is zero."""
    advantages = np.zeros(len(rewards), np.float32)
    last = 0.0
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(rewards) else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        last = delta + gamma * lam * last
        advantages[t] = last
    return advantages, advantages + np.asarray(values, np.float32)


# ------------------------------------------------------------------ update
def _head_parameters(arm):
    critic = arm.policy.critic if hasattr(arm, "policy") else arm.critic
    return list(critic.parameters())


def _log_std(arm):
    return arm.policy.log_std if hasattr(arm, "policy") else arm.log_std


class MinimalPPO:
    def __init__(self, name, arm, config: MinimalPPOConfig):
        self.name, self.arm, self.cfg = name, arm, config
        _log_std(arm).requires_grad_(False)          # exploration is fixed
        trainable = [p for p in arm.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable, lr=config.learning_rate)
        self.head_optimizer = torch.optim.Adam(_head_parameters(arm), lr=1e-3)
        self.lr = config.learning_rate
        self.iteration = 0

    def _set_lr(self, value):
        self.lr = float(min(max(value, 1e-7), self.cfg.learning_rate))
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr

    def update(self, episodes) -> dict:
        cfg = self.cfg
        self.iteration += 1
        gamma = math.exp(-0.1 / cfg.discount_tau_s)
        xs, acts, old, advs, rets, masks = [], [], [], [], [], []
        for ep in episodes:
            steps = ep["steps"]
            a, r = gae([s[4] for s in steps], [s[3] for s in steps], gamma, cfg.gae_lambda)
            for s, adv, ret in zip(steps, a, r):
                xs.append(s[0]); acts.append(s[1]); old.append(s[2])
                advs.append(adv); rets.append(ret); masks.append(s[5])
        x = torch.as_tensor(np.stack(xs))
        act = torch.as_tensor(np.stack(acts))
        old_axes = torch.as_tensor(np.stack(old))
        mask = torch.as_tensor(np.stack(masks))
        adv = torch.as_tensor(np.asarray(advs, np.float32))
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)        # rollout-level
        ret = torch.as_tensor(np.asarray(rets, np.float32))
        n = len(xs)
        warmup = self.iteration <= cfg.critic_warmup_iterations
        stats = dict(samples=n, accepted=0, rejected=0, kl=0.0, critic_loss=0.0,
                     masked_axes=float(1 - mask.mean()), warmup=warmup)
        gen = torch.Generator().manual_seed(self.iteration)
        stop = False
        for _ in range(cfg.epochs):
            order = torch.randperm(n, generator=gen)
            for start in range(0, n, cfg.minibatch_size):
                idx = order[start:start + cfg.minibatch_size]
                if len(idx) < 2:
                    continue
                mean, log_std, value = self.arm(x[idx])
                critic_loss = 0.5 * ((value - ret[idx]) ** 2).mean()
                if warmup:
                    self.head_optimizer.zero_grad(set_to_none=True)
                    critic_loss.backward()
                    self.head_optimizer.step()
                    stats["critic_loss"] += float(critic_loss)
                    continue
                std = log_std.exp()
                logp_axes = (-0.5 * ((act[idx] - mean) / std) ** 2 - log_std
                             - 0.5 * math.log(2 * math.pi))
                shift = ((logp_axes - old_axes[idx]) * mask[idx]).sum(-1)
                ratio = shift.exp()
                objective = torch.minimum(
                    ratio * adv[idx],
                    ratio.clamp(1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * adv[idx])
                loss = -objective.mean() + cfg.value_coefficient * critic_loss
                snapshot = ([p.detach().clone() for p in self.arm.parameters()],
                            copy.deepcopy(self.optimizer.state_dict()))
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.arm.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    mean2, log_std2, _ = self.arm(x[idx])
                    after = ((-0.5 * ((act[idx] - mean2) / log_std2.exp()) ** 2 - log_std2
                              - 0.5 * math.log(2 * math.pi) - old_axes[idx]) * mask[idx]).sum(-1)
                    kl = float(((after.exp() - 1) - after).mean())
                if not math.isfinite(kl) or kl > 1.5 * cfg.target_kl:
                    with torch.no_grad():
                        for p, saved in zip(self.arm.parameters(), snapshot[0]):
                            p.copy_(saved)
                    self.optimizer.load_state_dict(snapshot[1])
                    if stats["accepted"] == 0:
                        self._set_lr(self.lr * 0.5)
                    stats["rejected"] += 1
                    stop = True
                    break
                stats["accepted"] += 1
                stats["kl"] += kl
                stats["critic_loss"] += float(critic_loss)
            if stop:
                break
        if not warmup and stats["accepted"] and not stats["rejected"] \
                and stats["kl"] / stats["accepted"] < 0.5 * cfg.target_kl:
            self._set_lr(self.lr * 1.5)
        stats["kl"] /= max(stats["accepted"], 1)
        stats["lr"] = self.lr
        return stats


def train(name: str, init_checkpoint: str, out_dir, config: MinimalPPOConfig, *, seed: int):
    """Fine-tune one clone; keep the best validation checkpoint. Returns a summary."""
    import json
    import time
    from pathlib import Path
    from .arms import save_arm
    from .rollout import evaluate
    # One thread: nine runs share the machine, and torch's default (one per
    # core in EVERY process) put the load at 45 on 20 cores.
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    loaded_name, arm, _ = load_arm(init_checkpoint)
    if loaded_name != name:
        raise ValueError(f"{init_checkpoint} is {loaded_name}, not {name}")
    with torch.no_grad():
        _log_std(arm).fill_(float(arm.config.initial_log_std))
    trainer = MinimalPPO(name, arm, config)
    ctx = multiprocessing.get_context("spawn")
    history, best = [], None
    started = time.time()

    def validate(tag):
        arm.eval()
        result = evaluate(arm, list(config.validation_seeds))
        score = (result["landing"], -result["unsafe"])
        return result, score

    result, score = validate("init")
    save_arm(arm, name, out / "checkpoint_init.pt", iteration=0, validation=result)
    best = (score, 0, result)
    save_arm(arm, name, out / "checkpoint_best.pt", iteration=0, validation=result)
    with ProcessPoolExecutor(config.workers, mp_context=ctx) as pool:
        for it in range(1, config.iterations + 1):
            seeds = list(range(config.train_seed_base + seed * 10_000 + it * config.episodes_per_iteration,
                               config.train_seed_base + seed * 10_000 + (it + 1) * config.episodes_per_iteration))
            arm.eval()
            episodes = collect(name, arm, seeds, pool, config.workers,
                               sample_seed=seed * 7919 + it, scenario=config.train_scenario)
            arm.train()
            stats = trainer.update(episodes)
            statuses = [e["status"] for e in episodes]
            row = {"iteration": it, "success": statuses.count("SUCCESS"),
                   "unsafe": sum(s in ("UNSAFE_CONTACT", "MISSED_PAD_CONTACT",
                                       "SAFETY_ENVELOPE_VIOLATION", "UNAUTHORIZED_CONTACT")
                                 for s in statuses),
                   "episodes": len(statuses),
                   "mean_return": float(np.mean([sum(s[4] for s in e["steps"]) for e in episodes])),
                   **stats}
            if it % config.validation_every == 0 or it == config.iterations:
                result, score = validate(it)
                row["validation"] = result
                save_arm(arm, name, out / "checkpoint_last.pt", iteration=it, validation=result)
                if score >= best[0]:
                    best = (score, it, result)
                    save_arm(arm, name, out / "checkpoint_best.pt", iteration=it, validation=result)
            history.append(row)
            (out / "history.json").write_text(json.dumps(history, indent=1, default=float))
            print(json.dumps({k: row[k] for k in ("iteration", "success", "unsafe", "accepted",
                                                  "rejected", "kl", "lr")}
                             | ({"val_landing": row["validation"]["landing"]}
                                if "validation" in row else {}), default=float), flush=True)
    summary = {"arm": name, "seed": seed, "init": str(init_checkpoint),
               "config": asdict(config), "best_iteration": best[1],
               "best_validation": best[2], "elapsed_s": time.time() - started,
               "train_success_last10": float(np.mean([h["success"] / h["episodes"]
                                                      for h in history[-10:]]))}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary
