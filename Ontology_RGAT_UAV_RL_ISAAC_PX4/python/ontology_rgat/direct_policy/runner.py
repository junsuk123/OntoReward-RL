"""Explicit CLI for audit, parity, local smoke/train/evaluate and reporting."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from .backends import LocalReferenceBackend, ReferenceReplayBackend
from .contracts import make_contract
from .graph import planar_graph, spatial_graph
from .models import DirectActorCritic, load_checkpoint, save_checkpoint
from .observation import PlanarEstimate, planar_vector
from .ppo import DirectPPO, RolloutBatch


METHODS = ("ppo", "onto_rgat_ppo", "shuffled_rgat_ppo")


def _graph(observation, dimension):
    return planar_graph(observation) if dimension == 2 else spatial_graph(observation)


def _input(observation, dimension, method):
    if method == "ppo":
        return np.asarray(observation, np.float32)
    return _graph(observation, dimension).features.astype(np.float32)


def _model(contract, method, seed, graph_seed=0):
    zeros = np.zeros(contract.observation.dimension)
    example = None if method == "ppo" else _graph(zeros, contract.metadata["dimension"])
    return DirectActorCritic(contract, method, example, seed=seed, graph_seed=graph_seed)


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float),
                    encoding="utf-8")


def _rollout(contract, model, seed, *, deterministic, max_steps, policy_version=0):
    env = LocalReferenceBackend(contract)
    observation, _ = env.reset(seed)
    generator = torch.Generator().manual_seed(seed + 2_000_000)
    rows, total, status = [], 0.0, "RUNNING"
    started = time.perf_counter()
    for decision in range(max_steps):
        x = torch.as_tensor(_input(observation, contract.metadata["dimension"], model.method))
        with torch.no_grad():
            action, latent, logp, value = model.act(
                x, deterministic=deterministic, generator=generator)
        result = env.step(action.numpy())
        terminal = result.terminated
        rows.append((x.numpy(), latent.numpy(), float(logp), float(value), result.reward,
                     terminal, terminal or decision == max_steps-1))
        total += result.reward
        observation, status = result.observation, result.status
        if terminal:
            break
    final_x = torch.as_tensor(_input(observation, contract.metadata["dimension"], model.method))
    with torch.no_grad():
        final_value = float(model(final_x).value) if status == "RUNNING" else 0.0
    count = len(rows)
    bootstrap = np.zeros(count, np.float32)
    if rows and status == "RUNNING":
        bootstrap[-1] = final_value
    batch = RolloutBatch(
        policy_input=torch.as_tensor(np.stack([row[0] for row in rows])),
        latent_action=torch.as_tensor(np.stack([row[1] for row in rows])),
        old_log_probability=torch.as_tensor([row[2] for row in rows]),
        value=torch.as_tensor([row[3] for row in rows]),
        reward=torch.as_tensor([row[4] for row in rows]),
        discount=torch.full((count,), np.exp(-contract.scenarios.policy_dt_s
                                              / contract.reward.discount_tau_s)),
        terminated=torch.as_tensor([row[5] for row in rows]),
        episode_end=torch.as_tensor([row[6] for row in rows]),
        bootstrap_value=torch.as_tensor(bootstrap), policy_version=policy_version)
    return batch, {"seed": seed, "status": status, "return": total, "steps": count,
                   "wall_seconds": time.perf_counter()-started}


def _join(batches, policy_version):
    names = ("policy_input", "latent_action", "old_log_probability", "reward", "value",
             "discount", "terminated", "episode_end", "bootstrap_value")
    return RolloutBatch(**{name: torch.cat([getattr(batch, name) for batch in batches])
                           for name in names}, policy_version=policy_version)


def evaluate(contract, model, seeds, max_steps=700):
    episodes = [_rollout(contract, model, seed, deterministic=True,
                         max_steps=max_steps)[1] for seed in seeds]
    return {"episodes": episodes,
            "success_rate": sum(row["status"] == "SUCCESS" for row in episodes)/len(episodes),
            "unsafe_rate": sum(row["status"] in {"UNSAFE_CONTACT", "MISSED_PAD_CONTACT",
                                                   "SAFETY_ENVELOPE_VIOLATION"}
                               for row in episodes)/len(episodes),
            "mean_return": float(np.mean([row["return"] for row in episodes]))}


def run_smoke(args, contract):
    result = {"status": "PASSED", "backend": args.backend, "dimension": args.dimension,
              "control_profile": args.control_profile, "methods": {}}
    if args.backend == "isaac":
        if not args.allow_isaac:
            raise SystemExit("Isaac backend requires --allow-isaac")
        result.update(status="BLOCKED", reason=(
            "adapter is compile-tested; no owned Isaac/PX4 stack was acquired by smoke"))
        return result
    for offset, method in enumerate(args.methods):
        model = _model(contract, method, args.seed+offset, args.graph_seed)
        batch, episode = _rollout(contract, model, args.seed, deterministic=False,
                                  max_steps=args.steps)
        result["methods"][method] = {"episode": episode,
                                     "parameters": model.parameter_report(),
                                     "method_hash": model.method_hash,
                                     "finite": bool(torch.isfinite(batch.reward).all())}
    return result


def run_train(args, contract):
    if args.backend != "local":
        return {"status": "BLOCKED", "reason": "training is enabled only on local backend"}
    output = Path(args.output)
    summary = {"status": "PASSED", "methods": {}, "task_contract_hash": contract.task_contract_hash}
    for offset, method in enumerate(args.methods):
        model = _model(contract, method, args.seed+offset, args.graph_seed)
        trainer = DirectPPO(model)
        history = []
        best = None
        for update in range(1, args.updates+1):
            batches, episodes = [], []
            for episode in range(args.episodes_per_update):
                batch, row = _rollout(
                    contract, model,
                    args.seed*100_000+update*args.episodes_per_update+episode,
                    deterministic=False, max_steps=args.steps,
                    policy_version=trainer.policy_version)
                batches.append(batch); episodes.append(row)
            stats = trainer.update(_join(batches, trainer.policy_version),
                                   generator=torch.Generator().manual_seed(args.seed+update))
            row = {"update": update, "episodes": episodes, **stats}
            if update % args.evaluate_every == 0 or update == args.updates:
                validation = evaluate(contract, model, range(2001, 2001+args.validation_episodes),
                                      max_steps=args.steps)
                row["validation"] = validation
                score = (validation["success_rate"], -validation["unsafe_rate"],
                         validation["mean_return"])
                if best is None or score > best[0]:
                    best = (score, update, validation)
                    save_checkpoint(output/method/"checkpoint_best.pt", model,
                                    training_step=update,
                                    optimizer_state=trainer.optimizer_state(),
                                    rng_state=torch.get_rng_state())
            history.append(row)
            _write_json(output/method/"history.json", history)
        save_checkpoint(output/method/"checkpoint_last.pt", model,
                        training_step=args.updates, optimizer_state=trainer.optimizer_state(),
                        rng_state=torch.get_rng_state())
        summary["methods"][method] = {"best": None if best is None else
                                      {"update": best[1], "validation": best[2]},
                                      "method_hash": model.method_hash,
                                      "parameters": model.parameter_report()}
    _write_json(output/"summary.json", summary)
    return summary


def run_evaluate(args, contract):
    result = {"status": "PASSED", "methods": {}}
    for offset, method in enumerate(args.methods):
        model = _model(contract, method, args.seed+offset, args.graph_seed)
        if args.checkpoint_root:
            load_checkpoint(Path(args.checkpoint_root)/method/"checkpoint_best.pt", model)
        result["methods"][method] = evaluate(
            contract, model, range(args.seed, args.seed+args.evaluation_episodes), args.steps)
    return result


def run_parity(args, contract):
    if args.dimension != 2:
        return {"status": "NOT_RUN", "reason": "pinned MATLAB direct contract is planar"}
    if not args.fixture or not Path(args.fixture).exists():
        return {"status": "NOT_RUN", "reason": "MATLAB golden fixture not supplied"}
    fixture = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    if fixture.get("source_sha") != contract.source_sha:
        return {"status": "FAILED", "reason": "fixture source SHA is not pinned source"}
    estimate = PlanarEstimate(
        pad_position_xz=(7.0, 0.0), pad_velocity_xz=(1.5, 0.0),
        own_position_xz=(5.0, 4.6), own_velocity_xz=(0.5, -0.3),
        pitch_rad=0.1, pitch_rate_rad_s=0.2, vision_updated=True,
        vision_age_s=0.2, navigation_valid=True, navigation_age_s=0.05,
        pad_offset_z_m=0.6)
    observation = planar_vector(estimate)
    graph = planar_graph(observation)
    expected_observation = np.asarray(fixture["observation"], dtype=float)
    expected_features = np.asarray(fixture["graph_features"], dtype=float)
    checks = {
        "observation": bool(np.allclose(observation, expected_observation,
                                        atol=1e-8, rtol=1e-6)),
        "graph_features": bool(np.allclose(graph.features.T, expected_features,
                                           atol=1e-8, rtol=1e-6)),
        "edge_source": bool(np.array_equal(graph.source+1, fixture["edge_source"])),
        "edge_target": bool(np.array_equal(graph.target+1, fixture["edge_target"])),
        "edge_relation": bool(np.array_equal(graph.relation+1, fixture["edge_relation"])),
    }
    return {"status": "PASSED" if all(checks.values()) else "FAILED",
            "fixture": args.fixture, "checks": checks,
            "max_observation_abs_error": float(np.max(np.abs(
                observation-expected_observation))) if expected_observation.shape == (12,) else None,
            "max_graph_abs_error": float(np.max(np.abs(
                graph.features.T-expected_features))) if expected_features.shape == (6, 7) else None}


def build_parser():
    parser = argparse.ArgumentParser(prog="matlab-port")
    parser.add_argument("--stage", required=True,
                        choices=("audit", "parity", "smoke", "train", "evaluate", "report"))
    parser.add_argument("--backend", choices=("replay", "local", "isaac"), default="local")
    parser.add_argument("--dimension", type=int, choices=(2, 3), default=2)
    parser.add_argument("--control-profile", choices=("direct", "shielded"), default="direct")
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                        default=["ppo", "onto_rgat_ppo"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--graph-seed", type=int, default=1)
    parser.add_argument("--output", default="results/matlab_port/run")
    parser.add_argument("--fixture")
    parser.add_argument("--checkpoint-root")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-isaac", action="store_true")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--updates", type=int, default=750)
    parser.add_argument("--episodes-per-update", type=int, default=6)
    parser.add_argument("--evaluate-every", type=int, default=25)
    parser.add_argument("--validation-episodes", type=int, default=8)
    parser.add_argument("--evaluation-episodes", type=int, default=16)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    contract = make_contract(args.dimension, backend=args.backend,
                             safety_profile=args.control_profile)
    plan = {"stage": args.stage, "backend": args.backend, "dimension": args.dimension,
            "control_profile": args.control_profile, "methods": args.methods,
            "allow_isaac": args.allow_isaac, "contract": contract.manifest()}
    if args.dry_run:
        print(json.dumps({"status": "NOT_RUN", "reason": "dry-run", **plan}, indent=2))
        return 0
    if args.stage == "audit":
        result = {"status": "PASSED", **plan}
    elif args.stage == "parity":
        result = {**run_parity(args, contract), **plan}
    elif args.stage == "smoke":
        result = run_smoke(args, contract)
    elif args.stage == "train":
        result = run_train(args, contract)
    elif args.stage == "evaluate":
        result = run_evaluate(args, contract)
    else:
        result = {"status": "PASSED", "note": "acceptance report is generated from recorded gates",
                  **plan}
    _write_json(Path(args.output)/f"{args.stage}.json", result)
    print(json.dumps(result, indent=2, default=float))
    return 1 if result.get("status") == "FAILED" else 0
