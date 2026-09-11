"""Reward-only plug-ins for the common Shin-compatible backbone."""

from .controlled_potential import FrozenControlledPotential
from .controlled_rgat import (episode_rollout_dataset, load_rollout_dataset,
                              merge_rollout_datasets,
                              prepare_controlled_rgat_artifact,
                              save_rollout_dataset)
from .ontoreward import OntoRewardPBRS
from .shin2026 import (ShinReward, ShinRewardConfig,
                       active_perception_approximation, active_perception_reward)
from .sparse import sparse_terminal_reward

__all__ = ["OntoRewardPBRS", "FrozenControlledPotential",
           "prepare_controlled_rgat_artifact", "episode_rollout_dataset",
           "load_rollout_dataset", "merge_rollout_datasets",
           "save_rollout_dataset", "ShinReward", "ShinRewardConfig",
           "active_perception_approximation", "active_perception_reward",
           "sparse_terminal_reward"]
