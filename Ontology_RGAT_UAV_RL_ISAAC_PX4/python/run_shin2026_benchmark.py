#!/usr/bin/env python3
"""Validate, plan, smoke-test, or summarize the paired Shin-2026 benchmark.

``--smoke-test`` is CPU-only and checks neural, data, reward, and controller
interfaces without claiming that an Isaac/PX4 flight took place.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ontology_rgat.benchmarks.experiment import (METHODS, configuration_hash,
                                                 load_experiment,
                                                 paired_seed_plan)
from ontology_rgat.benchmarks.shin2026 import ActorObservation
from ontology_rgat.controllers import VelocityYawRateController
from ontology_rgat.evaluation.shin2026 import write_benchmark_outputs
from ontology_rgat.ppo.recurrent import (ShinRecurrentActorCritic,
                                         recurrent_ppo_loss)
from ontology_rgat.reward_modes import (OntoRewardPBRS, ShinReward,
                                        ShinRewardConfig,
                                        active_perception_reward,
                                        sparse_terminal_reward)


ROOT = Path(__file__).resolve().parents[1]


def _read_records(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def smoke_test(seed: int, methods) -> dict:
    torch.manual_seed(seed)
    model = ShinRecurrentActorCritic(
        image_embedding=512, lstm_hidden=512, latent_dim=256,
        actor_hidden=64, critic_hidden=64)
    images = torch.zeros(2, 3, 1, 320, 512)
    proprio = torch.zeros(2, 3, 7)
    proprio[..., 3] = 1.0
    truth = torch.zeros(2, 3, 6)
    starts = torch.tensor([[True, False, True], [True, False, False]])
    output = model(images, proprio, true_relative_state=truth,
                   episode_start=starts)
    pre_squash = output.action_mean.detach()
    action = torch.tanh(pre_squash)
    batch = {
        "images": images, "proprioception": proprio,
        "true_relative_state": truth, "episode_start": starts,
        "pre_squash_action": pre_squash, "action": action,
        "old_log_prob": model.log_prob(pre_squash, action,
                                        output.action_mean.detach(),
                                        output.action_std.detach()),
        "advantage": torch.ones(2, 3), "return": torch.zeros(2, 3),
        "truth_valid": torch.ones(2, 3, dtype=torch.bool),
    }
    loss, telemetry = recurrent_ppo_loss(model, batch)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    observation = ActorObservation(np.zeros((320, 512), dtype=np.uint8),
                                   np.zeros(3), np.array([1, 0, 0, 0]))
    controller = VelocityYawRateController(dt=0.1)
    command = controller.command(np.array([1.0, -1.0, 0.5, 0.25])).as_array()
    reward, parts = ShinReward(ShinRewardConfig(active_enabled=True))(
        np.array([2.0, 0.0, -3.0, 0.0, 0.0, 0.0]),
        np.array([1.5, 0.0, -2.7, 0.0, 0.0, 0.0]), np.zeros(4),
        drone_vertical_velocity=-0.5,
        current_estimation_error=0.02, next_estimation_error=0.03)
    # Exercise every selectable reward through the same state/action contract.
    reward_contracts = {}
    for method in methods:
        if method == "sparse":
            value = sparse_terminal_reward(physical_contact=False, crash=False,
                                           excessive_drift=False, terminal=False)
        elif method in {"shin2026", "manual_no_active"}:
            value, _ = ShinReward(ShinRewardConfig(
                active_enabled=method == "shin2026"))(
                    np.ones(6), np.full(6, 0.9), np.zeros(4),
                    drone_vertical_velocity=-0.5,
                    next_estimation_loss=0.02)
        else:
            pbrs = OntoRewardPBRS(lambda state: -float(np.linalg.norm(state[:2])),
                                  gamma=0.99, ppo_gamma=0.99)
            value, _ = pbrs(np.ones(6), np.full(6, 0.9))
            if method == "ontoreward_plus_active":
                value += active_perception_reward(0.02, ShinRewardConfig())
        reward_contracts[method] = float(value)
    assert output.relative_state.shape == (2, 3, 6)
    assert output.action_mean.shape == (2, 3, 4)
    assert output.value.shape == (2, 3)
    assert observation.proprioception.shape == (7,)
    return {"status": "passed", "relative_state_shape": [2, 3, 6],
            "action_shape": [2, 3, 4], "critic_shape": [2, 3],
            "controller_first_command": command.tolist(), "reward": reward,
            "reward_parts": parts, "reward_contracts": reward_contracts,
            "optimizer_step": "passed",
            "loss_telemetry": {key: float(value) for key, value in telemetry.items()}}


def main() -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                        default=["shin2026", "ontoreward"])
    parser.add_argument("--reward", choices=METHODS,
                        help="single-method shorthand used by the stack script")
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--paired-seeds", action="store_true")
    parser.add_argument("--config", type=Path,
                        default=ROOT / "config/experiments/shin2026_ablation.yaml")
    parser.add_argument("--results-dir", type=Path,
                        default=ROOT / "results/shin2026")
    parser.add_argument("--input-results", type=Path,
                        help="real per-episode CSV/JSONL to summarize")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.reward:
        args.methods = [args.reward]

    config = load_experiment(args.config)
    scenarios = dict(config.get("evaluation") or {"training_random_walk": 10000})
    if args.mode == "quick":
        scenarios = {name: min(int(count), 4 if name == "training_random_walk" else 2)
                     for name, count in scenarios.items()}
    seed0 = int((config.get("seeds") or {}).get("evaluation_start", 5000))
    plan = paired_seed_plan(args.methods, scenarios, seed0)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": str(args.config.resolve()), "config_hash": configuration_hash(config),
        "mode": args.mode, "methods": args.methods, "paired_seeds": True,
        "scenarios": scenarios, "planned_runs": len(plan),
        "paper_doi": "10.1109/LRA.2026.3674011",
        "claim": "methodological-interface reproduction; see docs/SHIN2026_BASELINE.md",
        "execution_status": "planning/smoke/analysis only; live rollout trainer not wired",
    }
    (args.results_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    with (args.results_dir / "paired_plan.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("method", "scenario", "seed"))
        writer.writeheader()
        writer.writerows(plan)

    if args.input_results:
        result = write_benchmark_outputs(_read_records(args.input_results), args.results_dir)
        print(json.dumps({**manifest, "summaries": len(result["summary"])}, indent=2))
        return 0
    if args.smoke_test:
        result = smoke_test(seed0, args.methods)
        (args.results_dir / "smoke_test.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({**manifest, "smoke_test": result["status"]}, indent=2))
        return 0
    print(json.dumps(manifest, indent=2))
    if not args.plan_only:
        print("No episode records supplied. The paired plan was written; use "
              "--input-results to analyze collected Isaac/PX4 episodes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
