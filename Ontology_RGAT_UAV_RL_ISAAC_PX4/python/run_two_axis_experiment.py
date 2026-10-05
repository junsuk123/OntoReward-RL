#!/usr/bin/env python3
"""Bounded smoke/manifests entry point for the primary two-axis experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ontology_rgat.two_axis.comparator import PNLandingController
from ontology_rgat.two_axis.config import load_config, DEFAULT_CONFIG_PATH
from ontology_rgat.two_axis.curriculum import stage_configs
from ontology_rgat.two_axis.diagnostics import (graph_utilisation,
                                                record_episode,
                                                summarize)
from ontology_rgat.two_axis.environment import TwoAxisLandingEnv
from ontology_rgat.two_axis.learning import (collect_rollout,
                                             ppo_minibatch_update)
from ontology_rgat.two_axis.models import (POLICY_MODES, TwoAxisPPOAgent,
                                           capacity_matched_mlp_hidden)
from ontology_rgat.two_axis.reward_audit import run_reward_audit
from ontology_rgat.two_axis.scenario import (build_seed_manifest,
                                             write_seed_manifest)


def smoke(seed: int, steps: int, *, update: bool, config=None) -> dict:
    config = config or load_config()
    results = {}
    for mode in POLICY_MODES:
        agent = TwoAxisPPOAgent(mode, seed=seed, graph_config=config.ontology)
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
        hidden, parameters = capacity_matched_mlp_hidden(mode, graph_parameters,
                                                        graph_config=config.ontology)
        capacity_check[mode] = {
            "hidden_dimension": hidden, "parameter_count": parameters,
            "difference_from_rgat": parameters - graph_parameters}
    return {"experiment": config.experiment, "seed": seed, "arms": results,
            "capacity_matched_mlp_check": capacity_check,
            "claim": "integration smoke only; no learned-performance conclusion"}


def pn_reference(seeds: range, *, difficulty: float = 1.0, config=None) -> dict:
    """Non-learned feasibility check. PN never teaches or supervises PPO."""
    config = config or load_config()
    env = TwoAxisLandingEnv(config, perturbations=True)
    controller = PNLandingController(config)
    records = []
    for seed in seeds:
        controller.reset()
        records.append(record_episode(
            env, lambda e: controller.act(e.state, e.track), seed=seed,
            difficulty=difficulty))
    report = summarize(records)
    report.update({
        "controller": "pn_guidance_two_axis_v1",
        "difficulty": difficulty,
        "seeds": [int(seed) for seed in seeds],
        "claim": "physical feasibility of the envelope; not a learned result",
    })
    return report


def graph_audit(seeds: range, config=None) -> dict:
    """Structural ontology audit over real rollouts (no trained policy needed)."""
    config = config or load_config()
    env = TwoAxisLandingEnv(config, perturbations=True)
    controller = PNLandingController(config)
    features = []
    for seed in seeds:
        controller.reset()
        observation, _ = env.reset(seed=seed, difficulty=1.0)
        while True:
            features.append(observation.graph.X.copy())
            observation, _r, terminated, _t, _i = env.step(
                controller.act(env.state, env.track))
            if terminated:
                break
    return graph_utilisation(features)


def curriculum_preview(config=None) -> dict:
    """Show that difficulty 1.0 is bit-for-bit the declared nominal contract."""
    config = config or load_config()
    rows = {}
    for difficulty in (0.0, 0.5, 1.0):
        stage = stage_configs(config, difficulty)
        rows[f"{difficulty:.1f}"] = {
            "initial_height_range_m": list(stage.scenario.initial_height_range_m),
            "T1_range_s": list(stage.scenario.T1_range_s),
            "touchdown_vertical_speed_m_s":
                stage.safety.touchdown_vertical_speed_m_s,
            "unsafe_contact_penalty":
                dict(stage.reward.terminal_bonus)["UNSAFE_CONTACT"],
        }
    nominal = stage_configs(config, 1.0)
    rows["nominal_identity"] = {
        "scenario": nominal.scenario == config.scenario,
        "safety": nominal.safety == config.safety,
        "reward": nominal.reward == config.reward,
    }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ppo-minibatch", action="store_true",
                        help="run one bounded numerical PPO update")
    parser.add_argument("--manifest-dir", type=Path)
    parser.add_argument("--reward-audit", type=Path)
    parser.add_argument("--pn-reference", action="store_true",
                        help="bounded non-learned feasibility check")
    parser.add_argument("--pn-seeds", type=int, nargs=2, default=(2000, 2020),
                        metavar=("START", "STOP"))
    parser.add_argument("--curriculum-preview", action="store_true")
    parser.add_argument("--graph-audit", action="store_true",
                        help="structural node/channel utilisation audit")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.threads <= 0:
        parser.error("--threads must be positive")
    import torch
    torch.set_num_threads(args.threads)
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
        output["smoke"] = smoke(args.seed, args.steps, update=args.ppo_minibatch, config=config)
    if args.pn_reference:
        output["pn_reference"] = pn_reference(range(*args.pn_seeds), config=config)
    if args.curriculum_preview:
        output["curriculum"] = curriculum_preview(config=config)
    if args.graph_audit:
        output["graph_audit"] = graph_audit(range(1000, 1008), config=config)
    if args.reward_audit:
        args.reward_audit.parent.mkdir(parents=True, exist_ok=True)
        args.reward_audit.write_text(
            json.dumps(run_reward_audit(config), indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        output["reward_audit"] = str(args.reward_audit)
    if not output:
        parser.error("select --smoke, --manifest-dir, --reward-audit, "
                     "--pn-reference, --curriculum-preview and/or "
                     "--graph-audit")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
