"""The relational graph attention potential.

See ``layers.py`` for what this takes from Busbridge et al. 2019 and the
reference release at https://github.com/babylonhealth/rgat, and NOTICE for the
attribution.
"""
from __future__ import annotations

from .layers import RGAT, RelationalGraphAttention
from .model import RGATPotential, build_potential, load_potential, save_potential
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

__all__ = ["RGAT", "RelationalGraphAttention", "RGATPotential", "Topology",
           "FixedRewardDesign", "build_potential", "load_potential", "save_potential",
           "FrozenSemanticRGATPotential", "SEMANTIC_DATASET_FORMAT",
           "SEMANTIC_MODEL_FORMAT", "assert_no_privileged_semantic_fields",
           "load_semantic_dataset", "merge_semantic_datasets",
           "prepare_semantic_rgat_artifact", "save_semantic_dataset",
           "semantic_episode_dataset", "semantic_monotonic_counterfactuals",
           "validate_semantic_dataset",
           "distill_reward_design", "load_reward_design", "save_reward_design"]
