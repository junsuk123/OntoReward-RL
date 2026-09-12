"""The relational graph attention potential.

See ``layers.py`` for what this takes from Busbridge et al. 2019 and the
reference release at https://github.com/babylonhealth/rgat, and NOTICE for the
attribution.
"""
from __future__ import annotations

from .layers import RGAT, RelationalGraphAttention
from .model import (PotentialHead, RGATEncoder, RGATPotential, build_potential,
                    load_potential, save_potential)
from .adaptive_model import (
    ADAPTIVE_GRAPH_INPUT_DIM, ADAPTIVE_GRAPH_VERSION, ADAPTIVE_MODEL_FORMAT,
    ADAPTIVE_NODE_NAMES, ADAPTIVE_REWARD_NODE_NAMES,
    AdaptiveRewardWeightHead, AdaptiveRewardWeightModel,
    FrozenAdaptiveRewardWeights, adaptive_model_digest,
    adaptive_reward_graph, build_adaptive_reward_model,
    empty_adaptive_reward_graph, freeze_adaptive_reward_model)
from .adaptive_dataset import (
    ADAPTIVE_DATASET_FORMAT, adaptive_dataset_digest, adaptive_episode_records,
    build_adaptive_dataset, load_adaptive_dataset, save_adaptive_dataset,
    validate_adaptive_dataset)
from .adaptive_train import (prepare_adaptive_reward_artifact,
                             select_ranking_pairs,
                             temporal_smoothness_pairs,
                             train_adaptive_reward_weights)
from .semantic_dataset import (
    FrozenSemanticRGATPotential, SEMANTIC_DATASET_FORMAT,
    SEMANTIC_MODEL_FORMAT, assert_no_privileged_semantic_fields,
    load_semantic_dataset, merge_semantic_datasets,
    prepare_semantic_rgat_artifact, save_semantic_dataset,
    semantic_episode_dataset, semantic_monotonic_counterfactuals,
    validate_semantic_dataset)
from .reward_design import (FixedRewardDesign, distill_reward_design,
                            load_reward_design, save_reward_design)
from .topology import Topology

__all__ = ["RGAT", "RelationalGraphAttention", "RGATEncoder", "PotentialHead",
           "RGATPotential", "Topology",
           "FixedRewardDesign", "build_potential", "load_potential", "save_potential",
           "FrozenSemanticRGATPotential", "SEMANTIC_DATASET_FORMAT",
           "SEMANTIC_MODEL_FORMAT", "assert_no_privileged_semantic_fields",
           "load_semantic_dataset", "merge_semantic_datasets",
           "prepare_semantic_rgat_artifact", "save_semantic_dataset",
           "semantic_episode_dataset", "semantic_monotonic_counterfactuals",
           "validate_semantic_dataset",
           "distill_reward_design", "load_reward_design", "save_reward_design",
           "ADAPTIVE_GRAPH_INPUT_DIM", "ADAPTIVE_GRAPH_VERSION",
           "ADAPTIVE_MODEL_FORMAT", "ADAPTIVE_NODE_NAMES",
           "ADAPTIVE_REWARD_NODE_NAMES", "AdaptiveRewardWeightHead",
           "AdaptiveRewardWeightModel", "FrozenAdaptiveRewardWeights",
           "adaptive_model_digest", "adaptive_reward_graph",
           "build_adaptive_reward_model", "empty_adaptive_reward_graph",
           "freeze_adaptive_reward_model", "ADAPTIVE_DATASET_FORMAT",
           "adaptive_dataset_digest", "build_adaptive_dataset",
           "adaptive_episode_records",
           "load_adaptive_dataset", "save_adaptive_dataset",
           "validate_adaptive_dataset", "prepare_adaptive_reward_artifact",
           "select_ranking_pairs", "temporal_smoothness_pairs",
           "train_adaptive_reward_weights"]
