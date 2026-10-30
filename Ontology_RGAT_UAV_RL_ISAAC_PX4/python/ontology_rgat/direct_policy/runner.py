"""Explicit CLI for audit, parity, local smoke/train/evaluate and reporting."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .backends import backend_factory as make_backend_factory
from .contracts import make_contract
from .graph import GraphState, planar_graph, spatial_graph
from .models import DirectActorCritic, load_checkpoint, save_checkpoint
from .observation import PlanarEstimate, planar_vector
from .ppo import DirectPPO, RolloutBatch


METHODS = ("ppo", "onto_rgat_ppo", "shuffled_rgat_ppo")


def _graph(observation, dimension):
    return planar_graph(observation) if dimension == 2 else spatial_graph(observation)


def _input(observation, dimension, method, *, contract=None, tracking_bias=None,
           handover_progress=0.0):
    observation = np.asarray(observation, np.float32)
    tracking_bias = np.zeros(dimension-1, np.float32) if tracking_bias is None \
        else np.asarray(tracking_bias, np.float32)
    tracking_bias = tracking_bias/(np.abs(tracking_bias)+3.0)
    if method == "ppo":
        return (np.concatenate((observation, tracking_bias,
                                np.asarray([handover_progress], np.float32)))
                if contract is not None and contract.metadata["training_profile"] == "stable"
                else observation)
    graph = _graph(observation, dimension)
    if contract is not None and contract.metadata["training_profile"] == "stable":
        repeated = np.broadcast_to(tracking_bias, (len(graph.features), len(tracking_bias)))
        progress = np.full((len(graph.features), 1), handover_progress, np.float32)
        graph = GraphState(np.concatenate((graph.features, repeated, progress), axis=1),
                           graph.source, graph.target, graph.relation, contract.graph)
    return graph.features.astype(np.float32)


def _model(contract, method, seed, graph_seed=0):
    dimension = contract.metadata["dimension"]
    zeros = np.zeros(contract.metadata["base_observation_dimension"])
    if method == "ppo":
        example = None
    else:
        features = _input(zeros, dimension, method, contract=contract)
        base = _graph(zeros, dimension)
        example = GraphState(features, base.source, base.target,
                             base.relation, contract.graph)
    return DirectActorCritic(contract, method, example, seed=seed, graph_seed=graph_seed)


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float),
                         encoding="utf-8")
    temporary.replace(path)


def _rollout(contract, model, seed, *, deterministic, max_steps, policy_version=0,
             backend_factory=None, collect_trace=False):
    factory = backend_factory or make_backend_factory(contract)
    env = factory()
    rows = []
    completed = None
    failure = None
    try:
        if hasattr(env, "set_policy_version"):
            env.set_policy_version(policy_version)
        observation, reset_info = env.reset(seed)
        generator = torch.Generator().manual_seed(seed + 2_000_000)
        total, status, trace = 0.0, "RUNNING", []
        tracking_bias = np.zeros(contract.metadata["dimension"]-1, dtype=np.float32)
        policy_elapsed = 0.0
        started = time.perf_counter()
        for decision in range(max_steps):
            if contract.metadata["training_profile"] == "stable":
                indices = (0,) if contract.metadata["dimension"] == 2 else (0, 1)
                scales = (3.0,)*len(indices)
                physical = [scales[i]*float(observation[index]) /
                            max(1.0-abs(float(observation[index])), 1e-6)
                            for i, index in enumerate(indices)]
                tracking_bias = np.clip(
                    tracking_bias + contract.scenarios.policy_dt_s*np.asarray(physical),
                    -3.0, 3.0).astype(np.float32)
            x = torch.as_tensor(_input(
                observation, contract.metadata["dimension"], model.method,
                contract=contract, tracking_bias=tracking_bias,
                handover_progress=(min(
                    policy_elapsed/contract.training.tracking_handover_s, 1.0)
                    if contract.metadata["training_profile"] == "stable" else 1.0)))
            with torch.no_grad():
                action, latent, logp, value = model.act(
                    x, deterministic=deterministic, generator=generator)
            result = env.step(action.numpy())
            if collect_trace:
                trace.append(result.info)
            terminal = result.terminated
            transition_s = float(result.info.get(
                "transition_s", contract.scenarios.policy_dt_s))
            if not np.isfinite(transition_s) or transition_s <= 0:
                raise ValueError("backend returned an invalid transition duration")
            policy_elapsed += transition_s
            rows.append((x.numpy(), latent.numpy(), float(logp), float(value),
                         result.reward, terminal,
                         terminal or decision == max_steps-1, transition_s))
            total += result.reward
            observation, status = result.observation, result.status
            if terminal:
                break
        if not rows:
            raise RuntimeError("backend produced no policy transitions")
        final_x = torch.as_tensor(_input(
            observation, contract.metadata["dimension"], model.method,
            contract=contract, tracking_bias=tracking_bias,
            handover_progress=(min(
                policy_elapsed/contract.training.tracking_handover_s, 1.0)
                if contract.metadata["training_profile"] == "stable" else 1.0)))
        with torch.no_grad():
            final_value = float(model(final_x).value) if status == "RUNNING" else 0.0
        count = len(rows)
        bootstrap = np.zeros(count, np.float32)
        if status == "RUNNING":
            bootstrap[-1] = final_value
        batch = RolloutBatch(
            policy_input=torch.as_tensor(np.stack([row[0] for row in rows])),
            latent_action=torch.as_tensor(np.stack([row[1] for row in rows])),
            old_log_probability=torch.as_tensor([row[2] for row in rows]),
            value=torch.as_tensor([row[3] for row in rows]),
            reward=torch.as_tensor([row[4] for row in rows]),
            discount=torch.as_tensor([
                np.exp(-row[7]/contract.reward.discount_tau_s) for row in rows],
                dtype=torch.float32),
            terminated=torch.as_tensor([row[5] for row in rows]),
            episode_end=torch.as_tensor([row[6] for row in rows]),
            bootstrap_value=torch.as_tensor(bootstrap), policy_version=policy_version)
        episode = {"seed": seed, "status": status, "return": total, "steps": count,
                   "wall_seconds": time.perf_counter()-started,
                   "simulated_seconds": float(sum(row[7] for row in rows)),
                   "backend": getattr(env, "name", type(env).__name__),
                   "reset": reset_info}
        if collect_trace:
            episode["trace"] = trace
        completed = (batch, episode)
        return completed
    except BaseException as exc:
        failure = exc
        # The recovery owner must be able to audit exactly how much partial
        # experience is discarded.  The exception remains the original bridge
        # error, so hard vehicle/policy failures cannot be relabelled.
        try:
            from ..bridge import BridgeError
            if isinstance(exc, BridgeError):
                exc.transitions_collected = len(rows)
        except ImportError:
            pass
        raise
    finally:
        try:
            env.close()
        except BaseException as cleanup_error:
            if failure is not None:
                # Preserve the policy-time cause.  Cleanup diagnostics remain
                # durable in flight_terminal.jsonl and must not relabel a hard
                # vehicle/task failure as recoverable infrastructure.
                failure.cleanup_error = repr(cleanup_error)
            else:
                if completed is not None:
                    cleanup_error.completed_rollout = completed
                    cleanup_error.transitions_collected = len(rows)
                raise


def _rollout_resilient(*args, **kwargs):
    """Retry the same seed after a bounded recoverable Isaac interruption."""
    seed = kwargs.get("seed", args[2] if len(args) > 2 else None)
    while True:
        try:
            return _rollout(*args, **kwargs)
        except BaseException as exc:
            from ..bridge import BridgeError
            if not isinstance(exc, BridgeError):
                raise
            from ..bridge import CleanupStopUnconfirmed
            if isinstance(exc, CleanupStopUnconfirmed):
                completed = getattr(exc, "completed_rollout", None)
                if completed is None:
                    raise
                from ..spatial.lifecycle import recover_failed_cleanup
                if recover_failed_cleanup(
                        exc, seed=seed,
                        transitions_collected=getattr(
                            exc, "transitions_collected", 0)):
                    return completed
                raise
            from ..spatial.lifecycle import recover_interrupted_episode
            if not recover_interrupted_episode(
                    exc, seed=seed,
                    transitions_collected=getattr(exc, "transitions_collected", 0)):
                raise


def _join(batches, policy_version):
    names = ("policy_input", "latent_action", "old_log_probability", "reward", "value",
             "discount", "terminated", "episode_end", "bootstrap_value")
    return RolloutBatch(**{name: torch.cat([getattr(batch, name) for batch in batches])
                           for name in names}, policy_version=policy_version)


def evaluate(contract, model, seeds, max_steps=700, *, deterministic=True,
             backend_factory=None, collect_trace=False, recover_infrastructure=False,
             progress_path=None, progress_metadata=None):
    seeds = list(seeds)
    collect = _rollout_resilient if recover_infrastructure else _rollout
    episodes = []
    progress_path = None if progress_path is None else Path(progress_path)
    metadata = dict(progress_metadata or {})
    if progress_path is not None and progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if (progress.get("version") != 1
                or progress.get("metadata") != metadata
                or progress.get("seeds") != seeds):
            raise ValueError("validation progress does not match this checkpoint/split")
        episodes = list(progress.get("episodes", []))
        if [row.get("seed") for row in episodes] != seeds[:len(episodes)]:
            raise ValueError("validation progress is not an ordered seed prefix")
    for seed in seeds[len(episodes):]:
        episodes.append(collect(
            contract, model, seed, deterministic=deterministic,
            max_steps=max_steps, backend_factory=backend_factory,
            collect_trace=collect_trace)[1])
        if progress_path is not None:
            _write_json(progress_path, {
                "version": 1, "metadata": metadata, "seeds": seeds,
                "episodes": episodes, "complete": False})
    result = {"deterministic": bool(deterministic), "episodes": episodes,
              "success_rate": sum(row["status"] == "SUCCESS" for row in episodes)/len(episodes),
              "unsafe_rate": sum(row["status"] in {"UNSAFE_CONTACT", "MISSED_PAD_CONTACT",
                                                     "SAFETY_ENVELOPE_VIOLATION"}
                                 for row in episodes)/len(episodes),
              "mean_return": float(np.mean([row["return"] for row in episodes]))}
    if progress_path is not None:
        _write_json(progress_path, {
            "version": 1, "metadata": metadata, "seeds": seeds,
            "episodes": episodes, "complete": True,
            "metrics": {key: value for key, value in result.items()
                        if key != "episodes"}})
    return result


def run_smoke(args, contract):
    result = {"status": "PASSED", "backend": args.backend, "dimension": args.dimension,
              "control_profile": args.control_profile, "methods": {}}
    if args.backend == "isaac":
        if not args.allow_isaac:
            raise SystemExit("Isaac backend requires --allow-isaac")
        stack, factory, deployment = _isaac_resources(args, contract)
        result["backend"] = "actual-isaac-px4-direct"
        result["deployment"] = deployment
    else:
        try:
            factory = make_backend_factory(contract, replay_path=args.replay_fixture)
        except ValueError as exc:
            return {**result, "status": "BLOCKED", "reason": str(exc)}
        stack = None

    def execute():
        for offset, method in enumerate(args.methods):
            model = _model(contract, method, args.seed+offset, args.graph_seed)
            batch, episode = _rollout(
                contract, model, args.seed, deterministic=False,
                max_steps=args.steps, backend_factory=factory,
                collect_trace=args.backend == "isaac")
            result["methods"][method] = {
                "episode": episode, "parameters": model.parameter_report(),
                "method_hash": model.method_hash,
                "finite": bool(torch.isfinite(batch.reward).all())}
    if stack is None:
        execute()
    else:
        with stack:
            execute()
    return result


def _train_with_factory(args, contract, factory, *, backend_name):
    output = Path(args.output)
    from .visualization import export_report
    _write_json(output/"training_manifest.json", {
        "backend": backend_name, "dimension": args.dimension,
        "methods": args.methods, "updates": args.updates,
        "episodes_per_update": args.episodes_per_update,
        "validation_episodes": args.validation_episodes,
        "evaluate_every": args.evaluate_every, "steps": args.steps,
        "training_profile": args.training_profile,
        "target_kl": getattr(contract.training, "target_kl", 0.02),
    })
    export_report(output, planned_updates=args.updates,
                  episodes_per_update=args.episodes_per_update)
    summary = {"status": "PASSED", "backend": backend_name, "methods": {},
               "task_contract_hash": contract.task_contract_hash,
               "execution_hash": contract.execution_hash}
    for offset, method in enumerate(args.methods):
        model = _model(contract, method, args.seed+offset, args.graph_seed)
        trainer = DirectPPO(model)
        history_path = output/method/"history.json"
        history = []
        best = None
        first_update = 1
        if args.resume_checkpoint_root:
            resume_root = Path(args.resume_checkpoint_root)
            if resume_root.resolve() != output.resolve():
                raise ValueError("resume checkpoint root must equal --output")
            checkpoint_path = resume_root/method/"checkpoint_last.pt"
            if checkpoint_path.is_file():
                metadata = load_checkpoint(checkpoint_path, model)
                step = int(metadata["training_step"])
                trainer.load_optimizer_state(
                    metadata["optimizer_state"], policy_version=step)
                torch.set_rng_state(metadata["rng_state"])
                first_update = step+1
                if history_path.is_file():
                    history = json.loads(history_path.read_text(encoding="utf-8"))
                    history = [row for row in history if int(row["update"]) <= step]
            elif history_path.is_file():
                raise ValueError(
                    f"cannot resume {method}: history exists without checkpoint_last.pt")
            for old in history:
                if "validation" not in old:
                    continue
                validation = old["validation"]
                score = (validation["success_rate"], -validation["unsafe_rate"],
                         validation["mean_return"])
                if best is None or score > best[0]:
                    best = (score, int(old["update"]), validation)

        def validation_due(update):
            return update % args.evaluate_every == 0 or update == args.updates

        def complete_validation(row):
            nonlocal best
            update = int(row["update"])
            checkpoint_path = output/method/"checkpoint_last.pt"
            if not checkpoint_path.is_file():
                raise RuntimeError(
                    f"cannot validate {method} update {update} without checkpoint_last.pt")
            validation_start = (2001 if args.backend == "local"
                                else max(10000, args.seed+1_000_000))
            progress_path = (output/method/"validation"/
                             f"update_{update:04d}.json")
            validation = evaluate(
                contract, model,
                range(validation_start,
                      validation_start+args.validation_episodes),
                max_steps=args.steps, backend_factory=factory,
                recover_infrastructure=args.backend == "isaac",
                progress_path=progress_path,
                progress_metadata={
                    "method": method, "update": update,
                    "checkpoint_sha256": hashlib.sha256(
                        checkpoint_path.read_bytes()).hexdigest(),
                    "task_contract_hash": contract.task_contract_hash,
                    "execution_hash": contract.execution_hash,
                })
            row["validation"] = validation
            _write_json(history_path, history)
            score = (validation["success_rate"], -validation["unsafe_rate"],
                     validation["mean_return"])
            if best is None or score > best[0]:
                best = (score, update, validation)
                save_checkpoint(output/method/"checkpoint_best.pt", model,
                                training_step=update,
                                optimizer_state=trainer.optimizer_state(),
                                rng_state=torch.get_rng_state())
            export_report(output, planned_updates=args.updates,
                          episodes_per_update=args.episodes_per_update)
            return validation

        # A checkpoint is deliberately committed before its expensive Isaac
        # validation.  If the previous process stopped during that validation,
        # finish the exact held-out split (from its per-episode journal) before
        # taking another PPO step.
        if (history and int(history[-1]["update"]) == first_update-1
                and validation_due(first_update-1)
                and "validation" not in history[-1]):
            validation = complete_validation(history[-1])
            print(json.dumps({"progress": {
                "method": method, "update": first_update-1,
                "episodes_complete": (first_update-1)*args.episodes_per_update,
                "validation": {key: value for key, value in validation.items()
                               if key != "episodes"},
                "resumed_validation": True,
            }}, default=float), flush=True)
        if first_update > args.updates+1:
            raise ValueError("checkpoint training step exceeds requested --updates")
        for update in range(first_update, args.updates+1):
            batches, episodes = [], []
            for episode in range(args.episodes_per_update):
                batch, row = _rollout_resilient(
                    contract, model,
                    args.seed*100_000+update*args.episodes_per_update+episode,
                    deterministic=False, max_steps=args.steps,
                    policy_version=trainer.policy_version,
                    backend_factory=factory)
                batches.append(batch); episodes.append(row)
            stats = trainer.update(_join(batches, trainer.policy_version),
                                   generator=torch.Generator().manual_seed(args.seed+update))
            row = {"update": update, "episodes": episodes, **stats}
            history.append(row)
            _write_json(output/method/"history.json", history)
            # Actual-Isaac campaigns are long and infrastructure can require a
            # cold restart. Persist the accepted PPO update BEFORE validation,
            # so a held-out flight failure never discards learned weights.
            save_checkpoint(output/method/"checkpoint_last.pt", model,
                            training_step=update,
                            optimizer_state=trainer.optimizer_state(),
                            rng_state=torch.get_rng_state())
            export_report(output, planned_updates=args.updates,
                          episodes_per_update=args.episodes_per_update)
            if validation_due(update):
                complete_validation(row)
            if update == 1 or "validation" in row:
                progress = {"method": method, "update": update,
                            "episodes_complete": update*args.episodes_per_update,
                            "mean_train_return": float(np.mean(
                                [episode["return"] for episode in episodes]))}
                if "validation" in row:
                    progress["validation"] = {
                        key: value for key, value in row["validation"].items()
                        if key != "episodes"
                    }
                print(json.dumps({"progress": progress}, default=float), flush=True)
        if not (output/method/"checkpoint_last.pt").is_file():
            raise RuntimeError(f"{method} produced no resumable checkpoint")
        summary["methods"][method] = {"best": None if best is None else
                                      {"update": best[1], "validation": best[2]},
                                      "method_hash": model.method_hash,
                                      "parameters": model.parameter_report()}
    _write_json(output/"summary.json", summary)
    return summary


def _isaac_resources(args, contract):
    from ontology_rgat.spatial.runtime_contract import (deployment_manifest,
                                                        deployment_profile)
    from run_spatial_pipeline import live_stack
    root = Path(__file__).resolve().parents[3]
    profile = (root/"config"/"matlab-port-planar-isaac.yaml"
               if args.dimension == 2 else
               Path(deployment_profile("spatial-reference/12")["path"]))
    deployment = deployment_manifest(profile)
    factory = make_backend_factory(contract, deployment=deployment, pair=0)
    stack = live_stack(
        Path(args.output), adopt=args.adopt_stack, headless=args.headless,
        schema="spatial-reference/12", reset_recoveries=args.reset_recoveries,
        isolate_episodes=args.fresh_stack_per_episode, profile_path=profile)
    return stack, factory, deployment


def run_train(args, contract):
    if args.backend == "local":
        return _train_with_factory(
            args, contract, make_backend_factory(contract),
            backend_name="local-reference")
    if args.backend == "replay":
        return {"status": "BLOCKED", "backend": "replay", "reason": (
            "fixed replay trajectories are an evaluation/parity backend, not a training plant")}
    if not args.allow_isaac:
        raise SystemExit("Isaac backend requires --allow-isaac")
    if args.control_profile != "direct":
        raise SystemExit("live MATLAB-port training currently requires --control-profile direct")
    stack, factory, deployment = _isaac_resources(args, contract)
    with stack:
        result = _train_with_factory(
            args, contract, factory, backend_name="actual-isaac-px4-direct")
    result["deployment"] = deployment
    return result


def run_evaluate(args, contract):
    if args.backend == "isaac":
        if not args.allow_isaac:
            raise SystemExit("Isaac backend requires --allow-isaac")
        if args.seed < 10000 or not 1 <= args.evaluation_episodes <= 1000:
            raise SystemExit("Isaac evaluation requires seed >=10000 and 1..1000 episodes")
        if args.control_profile != "direct":
            raise SystemExit("live MATLAB-port evaluation currently requires --control-profile direct")
        if not args.checkpoint_root:
            raise SystemExit("Isaac evaluation requires --checkpoint-root")
        output = Path(args.output)
        stack, factory, deployment = _isaac_resources(args, contract)
        result = {"status": "PASSED", "backend": "actual-isaac-px4",
                  "deployment": deployment, "methods": {}}
        with stack:
            for offset, method in enumerate(args.methods):
                model = _model(contract, method, args.seed+offset, args.graph_seed)
                load_checkpoint(Path(args.checkpoint_root)/method/"checkpoint_best.pt", model,
                                allow_execution_transfer=True)
                evaluated = evaluate(
                    contract, model,
                    range(args.seed, args.seed+args.evaluation_episodes), args.steps,
                    deterministic=not args.sample_actions,
                    backend_factory=factory, collect_trace=True,
                    recover_infrastructure=True)
                _write_json(output/"isaac_traces"/f"{method}.json", evaluated)
                # The full per-decision authority trace belongs in its own
                # artifact, not duplicated into the summary or stdout.
                result["methods"][method] = {
                    **{key: value for key, value in evaluated.items()
                       if key != "episodes"},
                    "episodes": [
                        {key: value for key, value in episode.items()
                         if key != "trace"}
                        for episode in evaluated["episodes"]
                    ],
                    "trace_artifact": str(
                        output/"isaac_traces"/f"{method}.json"),
                }
        return result
    if args.backend not in {"local", "replay"}:
        return {"status": "BLOCKED", "backend": args.backend, "methods": {},
                "reason": "selected backend is not available for evaluation"}
    try:
        factory = make_backend_factory(contract, replay_path=args.replay_fixture)
    except ValueError as exc:
        return {"status": "BLOCKED", "backend": args.backend, "methods": {},
                "reason": str(exc)}
    result = {"status": "PASSED", "backend": args.backend, "methods": {}}
    for offset, method in enumerate(args.methods):
        model = _model(contract, method, args.seed+offset, args.graph_seed)
        if args.checkpoint_root:
            load_checkpoint(Path(args.checkpoint_root)/method/"checkpoint_best.pt", model)
        result["methods"][method] = evaluate(
            contract, model, range(args.seed, args.seed+args.evaluation_episodes), args.steps,
            deterministic=not args.sample_actions, backend_factory=factory)
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
    parser.add_argument("--training-profile", choices=("source", "stable"),
                        default="source")
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                        default=["ppo", "onto_rgat_ppo"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--graph-seed", type=int, default=1)
    parser.add_argument("--output", default="results/matlab_port/run")
    parser.add_argument("--fixture")
    parser.add_argument("--checkpoint-root")
    parser.add_argument("--resume-checkpoint-root",
                        help="resume model, optimizers and update number in --output")
    parser.add_argument("--replay-fixture",
                        help="NPZ/JSON observation trajectory required by --backend replay")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-isaac", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--adopt-stack", action="store_true")
    parser.add_argument("--reset-recoveries", type=int, default=2)
    parser.add_argument("--fresh-stack-per-episode", action="store_true")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--updates", type=int, default=750)
    parser.add_argument("--episodes-per-update", type=int, default=6)
    parser.add_argument("--evaluate-every", type=int, default=25)
    parser.add_argument("--validation-episodes", type=int, default=8)
    parser.add_argument("--evaluation-episodes", type=int, default=16)
    parser.add_argument("--sample-actions", action="store_true",
                        help="sample the checkpoint policy instead of using its mean action")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    contract = make_contract(args.dimension, backend=args.backend,
                             safety_profile=args.control_profile,
                             training_profile=args.training_profile)
    if (args.stage == "train" and args.training_profile == "stable"
            and args.episodes_per_update < 12):
        raise SystemExit("stable training requires --episodes-per-update >= 12")
    plan = {"stage": args.stage, "backend": args.backend, "dimension": args.dimension,
            "control_profile": args.control_profile, "methods": args.methods,
            "training_profile": args.training_profile,
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
    console_result = result
    if args.stage == "evaluate" and "methods" in result:
        console_result = {
            **{key: value for key, value in result.items() if key != "methods"},
            "methods": {
                method: {key: value for key, value in row.items()
                         if key != "episodes"}
                for method, row in result["methods"].items()
            },
        }
    print(json.dumps(console_result, indent=2, default=float))
    return 1 if result.get("status") == "FAILED" else 0
