"""Custom continuous-action PPO with multi-episode rollout batching.

Port of the retired ``+training/trainPPO.m``. Episodes are accumulated into a
buffer of at least ``cfg.ppo.rollout_steps`` transitions before each update.
Updating from one short episode made the minibatch larger than the batch,
collapsing the inner loop to a single full-batch step per epoch and starving the
policy of gradient steps.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from ..config import Config
from .gae import compute_gae
from .networks import PPOAgent

__all__ = ["train_ppo", "PPOHistory"]


class PPOHistory(dict):
    """Per-episode training telemetry, one entry per list element."""

    @classmethod
    def empty(cls) -> "PPOHistory":
        return cls(episode=[], iteration=[], ret=[], success=[], length=[],
                   status=[], actor_loss=[], critic_loss=[], entropy=[],
                   policy_std=[])


def _collect_episode(agent: PPOAgent, reward_mode: str, potential, seed: int,
                     cfg: Config) -> dict[str, Any]:
    """One stochastic on-policy rollout with value and log-probability traces."""
    from ..env import LandingEnv

    env = LandingEnv.reset(seed, cfg)
    T = cfg.sim.max_steps
    O = np.zeros((T, cfg.rl.obs_dim))
    U = np.zeros((T, cfg.rl.act_dim))
    A = np.zeros((T, cfg.rl.act_dim))
    R = np.zeros(T); V = np.zeros(T); LP = np.zeros(T); D = np.zeros(T)
    status = "timeout"
    k = 0
    try:
        for k in range(T):
            cur = env.current()
            a, u, logp = agent.act(cur.obs, deterministic=False)
            value = agent.value(cur.obs)
            _, r, done, info = env.step(a, cur, reward_mode, potential)
            O[k], U[k], A[k] = cur.obs, u, a
            R[k], V[k], LP[k], D[k] = r, value, logp, float(done)
            status = info["status"]
            if done:
                break
        n = k + 1
        last_value = 0.0 if D[k] else agent.value(env.current().obs)
    finally:
        env.close()
    return {"O": O[:n], "U": U[:n], "A": A[:n], "R": R[:n], "V": V[:n],
            "LP": LP[:n], "D": D[:n], "last_value": last_value, "status": status}


def train_ppo(reward_mode: str, potential, cfg: Config, *,
              initial_agent: PPOAgent | None = None,
              initial_history: dict[str, Any] | None = None,
              on_episode: Callable[[PPOHistory], None] | None = None,
              on_update: Callable[[PPOHistory], None] | None = None,
              checkpoint: Callable[[PPOAgent, PPOHistory], None] | None = None,
              verbose: bool = True) -> tuple[PPOAgent, PPOHistory]:
    """Train one PPO arm and return the agent with its history."""
    device = torch.device(str(cfg.device.ppo))
    agent = (initial_agent.to(device) if initial_agent is not None
             else PPOAgent(cfg, device=device))
    actor_opt = torch.optim.Adam(agent.actor.parameters(), lr=float(cfg.ppo.actor_lr),
                                 betas=(0.9, 0.999), eps=1e-8)
    critic_opt = torch.optim.Adam(agent.critic.parameters(), lr=float(cfg.ppo.critic_lr),
                                  betas=(0.9, 0.999), eps=1e-8)
    optimizer_state = getattr(agent, "_optimizer_state", None)
    if optimizer_state:
        actor_opt.load_state_dict(optimizer_state["actor"])
        critic_opt.load_state_dict(optimizer_state["critic"])
        for optimizer in (actor_opt, critic_opt):
            for values in optimizer.state.values():
                for key, value in values.items():
                    if isinstance(value, torch.Tensor):
                        values[key] = value.to(device)
    history = PPOHistory.empty()
    for key, value in dict(initial_history or {}).items():
        if key in history and isinstance(value, list):
            history[key] = list(value)
    completed = len(history["episode"])
    total = completed + int(cfg.ppo.train_episodes)
    log_std_lo, log_std_hi = cfg.ppo.log_std_bounds

    episode = completed
    iteration = max(history["iteration"], default=0)
    while episode < total:
        iteration += 1
        first_of_iteration = episode + 1
        buffer: list[dict[str, Any]] = []
        buffered = 0
        while buffered < cfg.ppo.rollout_steps and episode < total:
            episode += 1
            tr = _collect_episode(agent, reward_mode, potential, 20000 + episode, cfg)
            adv, ret = compute_gae(tr["R"], tr["V"], tr["D"], tr["last_value"], cfg)
            tr["adv"], tr["ret"] = adv, ret
            buffer.append(tr)
            buffered += tr["R"].size
            history["episode"].append(episode)
            history["iteration"].append(iteration)
            history["ret"].append(float(tr["R"].sum()))
            history["success"].append(float(tr["status"] == "success"))
            history["length"].append(int(tr["R"].size))
            history["status"].append(tr["status"])
            # Filled in below once the update runs; kept aligned so every list
            # has one entry per episode and the dashboard can zip them.
            for key in ("actor_loss", "critic_loss", "entropy", "policy_std"):
                history[key].append(float("nan"))
            if on_episode is not None:
                on_episode(history)

        obs = torch.as_tensor(np.concatenate([b["O"] for b in buffer]),
                              dtype=torch.float32, device=device)
        u = torch.as_tensor(np.concatenate([b["U"] for b in buffer]),
                            dtype=torch.float32, device=device)
        act = torch.as_tensor(np.concatenate([b["A"] for b in buffer]),
                              dtype=torch.float32, device=device)
        old_logp = torch.as_tensor(np.concatenate([b["LP"] for b in buffer]),
                                   dtype=torch.float32, device=device)
        adv_np = np.concatenate([b["adv"] for b in buffer])
        # Standardize advantages once over the whole rollout, not per episode.
        adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)
        adv = torch.as_tensor(adv_np, dtype=torch.float32, device=device)
        ret = torch.as_tensor(np.concatenate([b["ret"] for b in buffer]),
                              dtype=torch.float32, device=device)

        n = adv.numel()
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        entropies: list[float] = []
        for _ in range(int(cfg.ppo.epochs)):
            order = torch.randperm(n, device=device)
            for start in range(0, n, int(cfg.ppo.minibatch)):
                idx = order[start:start + int(cfg.ppo.minibatch)]
                mu, std = agent.actor(obs[idx])
                logp = agent.actor.log_prob(u[idx], act[idx], mu, std)
                ratio = torch.exp(logp - old_logp[idx])
                clipped = ratio.clamp(1.0 - cfg.ppo.clip, 1.0 + cfg.ppo.clip)
                objective = torch.min(ratio * adv[idx], clipped * adv[idx])
                entropy = torch.log(std * np.sqrt(2.0 * np.pi * np.e)).sum(-1).mean()
                actor_loss = -objective.mean() - cfg.ppo.entropy_coef * entropy
                actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),
                                               float(cfg.ppo.grad_clip))
                actor_opt.step()
                with torch.no_grad():
                    agent.actor.log_std.clamp_(log_std_lo, log_std_hi)

                value = agent.critic(obs[idx])
                critic_loss = cfg.ppo.value_coef * ((value - ret[idx]) ** 2).mean()
                critic_opt.zero_grad(set_to_none=True)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.critic.parameters(),
                                               float(cfg.ppo.grad_clip))
                critic_opt.step()

                actor_losses.append(float(actor_loss.detach()))
                critic_losses.append(float(critic_loss.detach()))
                entropies.append(float(entropy.detach()))

        std_now = float(agent.actor.log_std.detach().exp().mean())
        span = slice(first_of_iteration - 1, episode)
        for key, value in (("actor_loss", float(np.mean(actor_losses))),
                           ("critic_loss", float(np.mean(critic_losses))),
                           ("entropy", float(np.mean(entropies))),
                           ("policy_std", std_now)):
            history[key][span] = [value] * (episode - first_of_iteration + 1)
        agent._optimizer_state = {
            "actor": actor_opt.state_dict(), "critic": critic_opt.state_dict()}
        if on_update is not None:
            on_update(history)
        if checkpoint is not None:
            checkpoint(agent, history)
        if verbose:
            window = slice(first_of_iteration - 1, episode)
            print(f"PPO {reward_mode:<8} iter {iteration:3d} | "
                  f"ep {first_of_iteration:4d}-{episode:4d}/{total} | "
                  f"buf {n:5d} ({len(actor_losses):3d} upd) | "
                  f"return {np.mean(history['ret'][window]):+8.2f} | "
                  f"succ {100 * np.mean(history['success'][window]):5.1f}% | "
                  f"steps {np.mean(history['length'][window]):5.1f} | "
                  f"La {np.mean(actor_losses):+.4f} Lc {np.mean(critic_losses):8.4f} | "
                  f"std {std_now:.3f}")
    return agent, history
