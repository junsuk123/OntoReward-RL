"""On-policy storage, time-aware GAE, and a bounded PPO update."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .environment import ObservationBundle, TwoAxisLandingEnv
from .models import TwoAxisActor, TwoAxisPPOAgent, observation_arrays
from .reward import discount_for_dt


@dataclass
class Transition:
    observation: ObservationBundle
    raw_command: np.ndarray
    normalized_command: np.ndarray
    requested_acceleration_m_s2: np.ndarray
    applied_acceleration_m_s2: np.ndarray
    old_log_probability: float
    value: float
    reward: float
    reward_components: dict[str, float]
    next_observation: ObservationBundle
    terminated: bool
    truncated: bool
    dt_s: float
    safety_flags: dict[str, Any]
    next_value: float = 0.0


def collect_rollout(agent: TwoAxisPPOAgent, env: TwoAxisLandingEnv, *,
                    seed: int, decisions: int) -> list[Transition]:
    """Collect sampled raw commands; never infer after a task terminal."""
    if decisions <= 0:
        raise ValueError("rollout decisions must be positive")
    observation, _ = env.reset(seed=seed)
    transitions: list[Transition] = []
    episode_seed = int(seed)
    while len(transitions) < decisions:
        raw, normalized, logp, value = agent.act(observation)
        next_observation, reward, terminated, truncated, info = env.step(normalized)
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
            if len(transitions) < decisions:
                episode_seed += 1
                observation, _ = env.reset(seed=episode_seed)
            continue
        observation = next_observation
    last = transitions[-1]
    if not last.terminated:
        last.truncated = True
        # External rollout boundary: bootstrap once from the final observation.
        _, _, _, last.next_value = agent.act(
            last.next_observation, deterministic=True)
    return transitions


def time_aware_gae(transitions: list[Transition], *,
                   discount_time_constant_s: float, gae_lambda: float = 0.95
                   ) -> tuple[np.ndarray, np.ndarray]:
    n = len(transitions)
    advantages = np.zeros(n, dtype=np.float64)
    gae = 0.0
    for i in range(n - 1, -1, -1):
        item = transitions[i]
        gamma = discount_for_dt(item.dt_s, discount_time_constant_s)
        if item.terminated:
            bootstrap, recurrence = 0.0, 0.0
        elif item.truncated:
            bootstrap, recurrence = item.next_value, 0.0
        else:
            bootstrap = transitions[i + 1].value if i + 1 < n else item.next_value
            recurrence = 1.0
        delta = item.reward + gamma * bootstrap - item.value
        gae = delta + gamma * gae_lambda * recurrence * gae
        advantages[i] = gae
    values = np.asarray([item.value for item in transitions], dtype=np.float64)
    return advantages, advantages + values


def ppo_minibatch_update(agent: TwoAxisPPOAgent,
                         transitions: list[Transition], *,
                         discount_time_constant_s: float,
                         actor_lr: float = 3e-4, critic_lr: float = 1e-3,
                         clip_ratio: float = 0.2, entropy_coefficient: float = 0.01
                         ) -> dict[str, float]:
    """One finite integration update; stored graphs are re-encoded here."""
    advantages, returns = time_aware_gae(
        transitions, discount_time_constant_s=discount_time_constant_s)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    packet_np, graph_np = observation_arrays(
        [item.observation for item in transitions])
    device = agent.device
    packets = torch.as_tensor(packet_np, dtype=torch.float32, device=device)
    graphs = torch.as_tensor(graph_np, dtype=torch.float32, device=device)
    raw = torch.as_tensor(np.stack([item.raw_command for item in transitions]),
                          dtype=torch.float32, device=device)
    old_logp = torch.as_tensor(
        [item.old_log_probability for item in transitions],
        dtype=torch.float32, device=device)
    adv = torch.as_tensor(advantages, dtype=torch.float32, device=device)
    ret = torch.as_tensor(returns, dtype=torch.float32, device=device)
    actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=actor_lr)
    critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr=critic_lr)

    mu, std = agent.actor(packets, graphs)
    logp = TwoAxisActor.raw_log_probability(raw, mu, std)
    ratio = torch.exp(logp - old_logp)
    objective = torch.minimum(ratio * adv,
                              ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * adv)
    entropy = torch.log(std * np.sqrt(2.0 * np.pi * np.e)).sum(-1).mean()
    actor_loss = -objective.mean() - entropy_coefficient * entropy
    actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    actor_grad = torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0)
    actor_optimizer.step()

    values = agent.critic(packets, graphs)
    critic_loss = 0.5 * ((values - ret) ** 2).mean()
    critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    critic_grad = torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1.0)
    critic_optimizer.step()
    metrics = {
        "actor_loss": float(actor_loss.detach()),
        "critic_loss": float(critic_loss.detach()),
        "entropy": float(entropy.detach()),
        "ratio_min": float(ratio.detach().min()),
        "ratio_max": float(ratio.detach().max()),
        "actor_grad_norm": float(actor_grad),
        "critic_grad_norm": float(critic_grad),
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise FloatingPointError(f"non-finite PPO minibatch metrics: {metrics}")
    return metrics
