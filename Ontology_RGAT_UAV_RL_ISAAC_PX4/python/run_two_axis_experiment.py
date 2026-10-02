#!/usr/bin/env python3
"""Bounded smoke/manifests entry point for the primary two-axis experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ontology_rgat.two_axis.config import load_config
from ontology_rgat.two_axis.environment import TwoAxisLandingEnv
from ontology_rgat.two_axis.learning import (collect_rollout,
                                             ppo_minibatch_update)
from ontology_rgat.two_axis.models import (POLICY_MODES, TwoAxisPPOAgent,
                                           capacity_matched_mlp_hidden)
from ontology_rgat.two_axis.reward_audit import run_reward_audit
from ontology_rgat.two_axis.scenario import (build_seed_manifest,
                                             write_seed_manifest)


def smoke(seed: int, steps: int, *, update: bool) -> dict:
    config = load_config()
    results = {}
    for mode in POLICY_MODES:
        agent = TwoAxisPPOAgent(mode, seed=seed)
        env = TwoAxisLandingEnv(config, perturbations=True)
        rollout = collect_rollout(agent, env, seed=seed, decisions=steps)
        rewards = np.asarray([item.reward for item in rollout])
        normalized = np.stack([item.normalized_command for item in rollout])
        entry = {
            "decisions": len(rollout), "reward_finite": bool(np.isfinite(rewards).all()),
            "reward_min": float(rewards.min()), "reward_max": float(rewards.max()),
            "action_saturation_ratio": float(np.mean(np.abs(normalized) >= 1.0)),
            "parameter_count": agent.parameter_count(),
            "terminated_transitions": sum(item.terminated for item in rollout),
        }
        if update:
            entry["ppo_minibatch"] = ppo_minibatch_update(
                agent, rollout,
                discount_time_constant_s=config.timing.discount_time_constant_s)
        results[mode] = entry
    graph_parameters = results["ppo_ontology_rgat"]["parameter_count"]["total"]
    capacity_check = {}
    for mode in ("ppo_vector_canonical", "ppo_semantic_flat"):
        hidden, parameters = capacity_matched_mlp_hidden(mode, graph_parameters)
        capacity_check[mode] = {
            "hidden_dimension": hidden, "parameter_count": parameters,
            "difference_from_rgat": parameters - graph_parameters}
    return {"experiment": config.experiment, "seed": seed, "arms": results,
            "capacity_matched_mlp_check": capacity_check,
            "claim": "integration smoke only; no learned-performance conclusion"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ppo-minibatch", action="store_true",
                        help="run one bounded numerical PPO update")
    parser.add_argument("--manifest-dir", type=Path)
    parser.add_argument("--reward-audit", type=Path)
    args = parser.parse_args()
    config = load_config()
    output = {}
    if args.manifest_dir:
        args.manifest_dir.mkdir(parents=True, exist_ok=True)
        split_seeds = {
            "train": range(1000, 1016), "validation": range(2000, 2008),
            "test": range(3000, 3008), "stress": range(4000, 4004)}
        for split, seeds in split_seeds.items():
            manifest = build_seed_manifest(seeds, split, config.scenario,
                                           config.timing)
            path = args.manifest_dir / f"{split}_scenarios.json"
            write_seed_manifest(path, manifest)
            output[split] = str(path)
    if args.smoke:
        output["smoke"] = smoke(args.seed, args.steps, update=args.ppo_minibatch)
    if args.reward_audit:
        args.reward_audit.parent.mkdir(parents=True, exist_ok=True)
        args.reward_audit.write_text(
            json.dumps(run_reward_audit(config), indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        output["reward_audit"] = str(args.reward_audit)
    if not output:
        parser.error("select --smoke, --manifest-dir and/or --reward-audit")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
