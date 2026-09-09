"""Actor and critic for the continuous-action PPO agent.

Port of the retired ``+training/{initPPO,actorForward,criticForward,samplePolicy}``.
The architecture is unchanged: two ``tanh`` layers of ``cfg.ppo.hidden`` units,
a state-independent log-standard-deviation, and a ``tanh``-squashed Gaussian
with the change-of-variables correction in the log probability.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..config import Config

__all__ = ["Actor", "Critic", "PPOAgent", "save_agent", "load_agent"]

LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
# Matches the retired MATLAB expression exactly. It keeps the correction finite
# where the squashed action saturates, at the cost of a small bias there.
TANH_EPS = 1e-6


def _init_linear(layer: nn.Linear, generator: torch.Generator, scale: float) -> None:
    with torch.no_grad():
        layer.weight.normal_(0.0, 1.0, generator=generator).mul_(scale)
        if layer.bias is not None:
            layer.bias.zero_()


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int,
                 init_log_std: float, mu_scale: float = 1.5):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std)))
        self.mu_scale = float(mu_scale)

    def reset_parameters(self, generator: torch.Generator, scale: float = 0.10) -> None:
        for layer in (self.fc1, self.fc2, self.mu):
            _init_linear(layer, generator, scale)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.fc2(torch.tanh(self.fc1(obs))))
        mu = self.mu_scale * torch.tanh(self.mu(h))
        return mu, self.log_std.exp().expand_as(mu)

    @staticmethod
    def log_prob(u: torch.Tensor, a: torch.Tensor, mu: torch.Tensor,
                 std: torch.Tensor) -> torch.Tensor:
        """Log density of the squashed action, summed over the action dimensions."""
        gaussian = -0.5 * ((u - mu) / std) ** 2 - std.log() - LOG_SQRT_2PI
        return (gaussian - torch.log(1.0 - a ** 2 + TANH_EPS)).sum(-1)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, 1)

    def reset_parameters(self, generator: torch.Generator, scale: float = 0.10) -> None:
        for layer in (self.fc1, self.fc2, self.v):
            _init_linear(layer, generator, scale)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.fc2(torch.tanh(self.fc1(obs))))
        return self.v(h).squeeze(-1)


class PPOAgent(nn.Module):
    """Actor plus critic, and the sampling the control loop calls per step."""

    def __init__(self, cfg: Config, device: torch.device | str = "cpu"):
        super().__init__()
        self.obs_dim = int(cfg.rl.obs_dim)
        self.act_dim = int(cfg.rl.act_dim)
        self.actor = Actor(self.obs_dim, self.act_dim, cfg.ppo.hidden,
                           cfg.ppo.init_log_std, cfg.ppo.mu_scale)
        self.critic = Critic(self.obs_dim, cfg.ppo.hidden)
        generator = torch.Generator(device="cpu").manual_seed(cfg.seed + 202)
        self.actor.reset_parameters(generator)
        self.critic.reset_parameters(generator)
        self._rng = np.random.default_rng(cfg.seed + 909)
        self.to(device)

    @property
    def device(self) -> torch.device:
        return self.actor.log_std.device

    def seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False
            ) -> tuple[np.ndarray, np.ndarray, float]:
        """Sample one action. Returns ``(squashed action, pre-squash u, log prob)``."""
        t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        mu_t, std_t = self.actor(t)
        mu = mu_t.cpu().numpy().astype(np.float64)
        std = std_t.cpu().numpy().astype(np.float64)
        u = mu if deterministic else mu + std * self._rng.standard_normal(mu.shape)
        a = np.tanh(u)
        logp = float(np.sum(-0.5 * ((u - mu) / std) ** 2 - np.log(std) - LOG_SQRT_2PI
                            - np.log(1.0 - a ** 2 + TANH_EPS)))
        return a, u, logp

    @torch.no_grad()
    def value(self, obs: np.ndarray) -> float:
        t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        return float(self.critic(t).item())


def save_agent(agent: PPOAgent, cfg: Config, path: str | Path,
               history: dict[str, Any] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "ontology_rgat.ppo_agent/1",
        "state_dict": {k: v.detach().cpu() for k, v in agent.state_dict().items()},
        "schema": {"obs_dim": int(cfg.rl.obs_dim), "act_dim": int(cfg.rl.act_dim),
                   "hidden": int(cfg.ppo.hidden), "mu_scale": float(cfg.ppo.mu_scale)},
        "history": history or {},
    }, path)
    return path


def load_agent(path: str | Path, cfg: Config,
               device: torch.device | str = "cpu") -> tuple[PPOAgent, dict[str, Any]]:
    """Load an agent, refusing one trained against a different observation.

    The old fixed-pad policy has a 15-element observation; transferring it to
    the 20-element moving-pad one would silently feed it the wrong channels, so
    the dimension is checked before the weights are touched. On hardware that
    check is the difference between a flight and an accident.
    """
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    saved = blob.get("schema", {})
    for key, wanted in (("obs_dim", int(cfg.rl.obs_dim)),
                        ("act_dim", int(cfg.rl.act_dim)),
                        ("hidden", int(cfg.ppo.hidden))):
        if key in saved and int(saved[key]) != wanted:
            raise ValueError(
                f"PPO checkpoint {Path(path).name} has {key}={saved[key]} but this "
                f"configuration wants {wanted}. Retrain rather than transferring "
                "the policy.")
    agent = PPOAgent(cfg, device="cpu")
    agent.load_state_dict(blob["state_dict"])
    agent.to(device)
    return agent, blob.get("history", {})
