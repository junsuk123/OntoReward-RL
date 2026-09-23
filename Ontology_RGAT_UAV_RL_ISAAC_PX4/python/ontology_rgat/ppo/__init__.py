"""Custom continuous-action PPO with multi-episode rollout batching."""
from __future__ import annotations

from .gae import compute_gae
from .networks import PPOAgent, load_agent, save_agent
from .train import PPOHistory, train_ppo

from .recurrent import (PipelineActorCritic, ShinRecurrentActorCritic,
                        StatefulPipelinePolicy, StatefulShinPolicy,
                        recurrent_ppo_loss)
from .graph_state_encoder import (GRAPH_STATE_REPRESENTATIONS,
                                  GraphStateEncoder, graph_feature_tensor,
                                  graph_state_topology)
from .temporal_backbone import TemporalBackboneOutput, TemporalVisualBackbone
from .recurrent_train import (collect_episode, save_recurrent_checkpoint,
                              train_live, update_episode)

__all__ = [
    "PPOAgent", "PPOHistory", "compute_gae", "load_agent", "save_agent",
    "train_ppo", "PipelineActorCritic", "ShinRecurrentActorCritic",
    "StatefulPipelinePolicy", "StatefulShinPolicy",
    "TemporalBackboneOutput", "TemporalVisualBackbone",
    "recurrent_ppo_loss", "collect_episode", "save_recurrent_checkpoint",
    "train_live", "update_episode",
    "GRAPH_STATE_REPRESENTATIONS", "GraphStateEncoder", "graph_feature_tensor",
    "graph_state_topology",
]
