"""Custom continuous-action PPO with multi-episode rollout batching."""
from __future__ import annotations

from .gae import compute_gae
from .networks import PPOAgent, load_agent, save_agent
from .train import PPOHistory, train_ppo

from .recurrent import (ShinRecurrentActorCritic, StatefulShinPolicy,
                        recurrent_ppo_loss)
from .recurrent_train import (collect_episode, save_recurrent_checkpoint,
                              train_live, update_episode)

__all__ = [
    "PPOAgent", "PPOHistory", "compute_gae", "load_agent", "save_agent",
    "train_ppo", "ShinRecurrentActorCritic", "StatefulShinPolicy",
    "recurrent_ppo_loss", "collect_episode", "save_recurrent_checkpoint",
    "train_live", "update_episode",
]
