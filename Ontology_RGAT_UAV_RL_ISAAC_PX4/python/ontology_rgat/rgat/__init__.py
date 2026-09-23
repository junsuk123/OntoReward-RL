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
from .fov_graph import (
    FOV_FEATURE_NAMES, FOV_GOAL_NODE, FOV_GRAPH_INPUT_DIM, FOV_GRAPH_VERSION,
    FOV_GRAPH_EDGES, FOV_NODE_NAMES, FOV_RELATION_NAMES,
    FOVSemanticObservation, build_fov_graph, derived_node_values,
    empty_fov_graph, fov_margin, fov_observation_from_visual_semantics,
    unreachable_input_nodes)
from .fov_risk_model import (FOV_RISK_MODEL_FORMAT, FOVRiskModel,
                             FrozenFOVRiskPredictor, save_fov_risk_model,
                             state_dict_digest)
from .fov_risk_dataset import (
    FOV_RISK_DATASET_FORMAT, build_fov_risk_dataset, dataset_digest,
    future_fov_unavailability_targets, horizon_steps, load_fov_risk_dataset,
    fov_risk_data_fingerprint, save_fov_risk_dataset, split_by_episode,
    split_by_episode_ids, validate_fov_risk_dataset)
from .fov_risk_train import (CONTRACT_RULE_ID, contract_loss,
                             prepare_fov_risk_artifact, regression_metrics,
                             train_fov_risk_model)
from .state_graph import (
    STATE_GOAL_NODE, STATE_GRAPH_EDGES, STATE_GRAPH_INPUT_DIM,
    STATE_GRAPH_VERSION, STATE_NODE_NAMES, STATE_RELATION_NAMES,
    STATE_RISK_NODES, StateGraphGeometry, StateGraphScales, build_state_graph,
    empty_state_graph, state_node_signs, state_node_values,
    unreachable_state_input_nodes)

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
           "train_adaptive_reward_weights",
           "FOV_FEATURE_NAMES", "FOV_GOAL_NODE", "FOV_GRAPH_INPUT_DIM",
           "FOV_GRAPH_VERSION", "FOV_NODE_NAMES", "FOV_RELATION_NAMES",
           "FOV_GRAPH_EDGES", "FOVSemanticObservation", "build_fov_graph",
           "derived_node_values", "empty_fov_graph", "fov_margin",
           "fov_observation_from_visual_semantics", "unreachable_input_nodes",
           "FOV_RISK_MODEL_FORMAT", "FOVRiskModel", "FrozenFOVRiskPredictor",
           "save_fov_risk_model", "state_dict_digest",
           "FOV_RISK_DATASET_FORMAT", "build_fov_risk_dataset",
           "dataset_digest", "future_fov_unavailability_targets", "horizon_steps",
           "load_fov_risk_dataset", "save_fov_risk_dataset",
           "split_by_episode", "split_by_episode_ids",
           "fov_risk_data_fingerprint", "validate_fov_risk_dataset",
           "CONTRACT_RULE_ID", "contract_loss", "regression_metrics",
           "prepare_fov_risk_artifact", "train_fov_risk_model",
           "STATE_GOAL_NODE", "STATE_GRAPH_EDGES", "STATE_GRAPH_INPUT_DIM",
           "STATE_GRAPH_VERSION", "STATE_NODE_NAMES", "STATE_RELATION_NAMES",
           "STATE_RISK_NODES", "StateGraphGeometry", "StateGraphScales",
           "build_state_graph", "empty_state_graph", "state_node_signs",
           "state_node_values", "unreachable_state_input_nodes"]
