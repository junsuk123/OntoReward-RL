"""Reward-only plug-ins for the common Shin-compatible backbone."""

from .controlled_potential import FrozenControlledPotential
from .controlled_rgat import (episode_rollout_dataset, load_rollout_dataset,
                              merge_rollout_datasets,
                              prepare_controlled_rgat_artifact,
                              save_rollout_dataset)
from .ontoreward import OntoRewardPBRS
from .contexts import (NoSERewardContext, OntologyRewardContext,
                       ShinSERewardContext, TerminalFlags)
from .shin2026 import (ShinReward, ShinRewardConfig,
                       active_perception_approximation, active_perception_reward)
from .sparse import sparse_terminal_reward
from .adaptive_weight import (
    BASELINE_REWARD_WEIGHTS, DEFAULT_COMPONENT_SCALES,
    REWARD_COMPONENT_NAMES, TOTAL_REWARD_WEIGHT, AdaptiveRewardConfig,
    AdaptiveWeightReward, FixedBaselineRewardWeights, RewardComponentNormalizer,
    constrained_adaptive_weights, optional_estimation_error_component,
    shin_reward_components)

__all__ = ["OntoRewardPBRS", "FrozenControlledPotential",
           "NoSERewardContext", "OntologyRewardContext",
           "ShinSERewardContext", "TerminalFlags",
           "prepare_controlled_rgat_artifact", "episode_rollout_dataset",
           "load_rollout_dataset", "merge_rollout_datasets",
           "save_rollout_dataset", "ShinReward", "ShinRewardConfig",
           "active_perception_approximation", "active_perception_reward",
           "sparse_terminal_reward", "BASELINE_REWARD_WEIGHTS",
           "DEFAULT_COMPONENT_SCALES", "REWARD_COMPONENT_NAMES",
           "TOTAL_REWARD_WEIGHT", "AdaptiveRewardConfig",
           "AdaptiveWeightReward", "RewardComponentNormalizer",
           "FixedBaselineRewardWeights",
           "constrained_adaptive_weights", "optional_estimation_error_component",
           "shin_reward_components"]
