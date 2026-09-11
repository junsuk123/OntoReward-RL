"""Reward-only plug-ins for the common Shin-compatible backbone."""

from .controlled_potential import FrozenControlledPotential
from .ontoreward import OntoRewardPBRS
from .shin2026 import (ShinReward, ShinRewardConfig,
                       active_perception_approximation, active_perception_reward)
from .sparse import sparse_terminal_reward

__all__ = ["OntoRewardPBRS", "FrozenControlledPotential", "ShinReward", "ShinRewardConfig",
           "active_perception_approximation", "active_perception_reward",
           "sparse_terminal_reward"]
