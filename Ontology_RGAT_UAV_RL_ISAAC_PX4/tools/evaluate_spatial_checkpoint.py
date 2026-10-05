#!/usr/bin/env python3
"""Validate one signed spatial checkpoint without claiming a complete matrix.

Default seeds are validation only. Held-out test is explicit and must not be
used for checkpoint selection. Isaac requires explicit adoption of a live stack.
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import torch
from ontology_rgat.spatial.core import SpatialConfig
from ontology_rgat.spatial.environment import SpatialLandingEnv, IsaacBackend
from ontology_rgat.spatial.training import load_agent, evaluate, VALIDATION_SEEDS
from ontology_rgat.spatial.runtime_contract import deployment_profile
from ontology_rgat.two_axis.config import GraphConfig, CurriculumConfig
from ontology_rgat.two_axis.artifacts import json_text
from run_spatial_pipeline import live_stack, request_shutdown


def evaluation_seeds(split, backend, start=None, episodes=2):
    """Keep selection validation fixed; explicit test ranges never tune a model."""
    if split == "validation":
        if start is not None or episodes != 2:
            raise ValueError("validation seeds are fixed; test ranges require --split test")
        return VALIDATION_SEEDS
    start = 12000 if start is None else start
    minimum = 10000 if backend == "isaac" else 9000
    # Training/pretraining/activation collectors occupy seed ranges >=100000.
    if not minimum <= start < 100000 or not 1 <= episodes <= 1000 or start + episodes > 100000:
        raise ValueError(f"test seeds must be in [{minimum},100000), episodes in [1,1000]")
    return tuple(range(start, start + episodes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["local", "isaac"], default="local")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--test-seed-start", type=int,
                        help="Explicit evaluation-only test range (default 12000)")
    parser.add_argument("--episodes", type=int, default=2,
                        help="Test episodes; validation always uses its fixed two seeds")
    parser.add_argument("--adopt-stack", action="store_true")
    parser.add_argument("--reset-recoveries", type=int, choices=range(6), default=0)
    args = parser.parse_args()
    try:
        seeds = evaluation_seeds(args.split, args.backend, args.test_seed_start, args.episodes)
    except ValueError as exc:
        parser.error(str(exc))
    torch.set_num_threads(1)
    config = dict(json.loads(args.plan.read_text())["config"])
    config["ontology"] = GraphConfig(**config["ontology"])
    config["curriculum"] = CurriculumConfig(**config["curriculum"])
    cfg = SpatialConfig(**config)
    # Training can atomically replace checkpoint_best while evaluation runs.
    # Freeze exactly the bytes evaluated, not just a path to a changing file.
    checkpoint_bytes = args.checkpoint.read_bytes()
    digest = hashlib.sha256(checkpoint_bytes).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output / "checkpoint_evaluated.pt"
    snapshot.write_bytes(checkpoint_bytes)
    agent, meta = load_agent(snapshot, cfg)
    if not meta.get("eligible"):
        raise ValueError("checkpoint has no completed nominal episode")
    status = dict(
        state="running", complete=False, checkpoint=str(args.checkpoint.resolve()),
        snapshot=str(snapshot.resolve()), sha256=digest, signature=cfg.signature,
        split=args.split, backend=args.backend, episode_seeds=list(seeds),
        checkpoint_selection=False, deployment=deployment_profile(cfg.schema),
        started_utc=datetime.now(timezone.utc).isoformat(),
        complete_three_arm_matrix=False, performance_parity_claim=False)
    def persist_status():
        (args.output / "status.json").write_text(json_text(status, indent=2) + "\n")
    persist_status()
    def execute():
        with (args.output / "trace.jsonl").open("x") as trace:

            def record(arm, seed, step, action, info):
                trace.write(
                    json_text(
                        dict(
                            mode=arm,
                            seed=seed,
                            step=step,
                            action=action.tolist(),
                            info=info,
                        )
                    )
                    + "\n"
                )
                trace.flush()
                if step % 25 == 0 or info["status"] != "RUNNING":
                    print(
                        arm,
                        seed,
                        step,
                        info["status"],
                        info["truth_relative_position"],
                        flush=True,
                    )

            kwargs = (
                {}
                if args.backend == "local"
                else dict(env_factory=lambda c: SpatialLandingEnv(c, IsaacBackend(c)))
            )
            metrics = evaluate(agent, cfg, seeds=seeds, trace=record, **kwargs)
        report = dict(
            checkpoint=str(args.checkpoint.resolve()),
            snapshot=str(snapshot.resolve()),
            sha256=digest,
            signature=cfg.signature,
            split=args.split,
            episode_seeds=list(seeds),
            checkpoint_selection=False,
            backend=args.backend,
            metrics=metrics,
            complete_three_arm_matrix=False,
            performance_parity_claim=False,
        )
        (args.output / "report.json").write_text(json_text(report, indent=2) + "\n")
        print(json_text(report), flush=True)
        return 0 if metrics["landing_rate"] > 0 and metrics["unsafe_rate"] == 0 else 2

    try:
        if args.backend == "isaac":
            preflight = evaluate(agent, cfg)
            (args.output / "preflight.json").write_text(
                json_text(preflight, indent=2) + "\n")
            if preflight["unsafe_rate"]:
                raise RuntimeError("nominal local preflight unsafe; Isaac refused")
            with live_stack(args.output, adopt=args.adopt_stack, schema=cfg.schema,
                            reset_recoveries=args.reset_recoveries):
                code = execute()
        else:
            code = execute()
        status.update(state="completed", complete=True, exit_code=code)
        return code
    except BaseException as exc:
        status.update(state="incomplete", complete=False,
                      error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        persist_status()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, request_shutdown)
    raise SystemExit(main())
