"""Fly the whole scenario plan, then learn from what it collected.

    ./run_scenarios.sh                      the full plan, then deploy
    ./run_scenarios.sh --dry-run            write the episode configs and stop
    ./run_scenarios.sh --scenario sensor_dropout
    ./run_scenarios.sh --episodes 2 --no-deploy

The order matters and is the whole point:

1. **Catch up first.** Anything collected by an earlier run but never trained on
   is ingested and learned before a single new episode starts, so the run begins
   from the best weights available.
2. **Fly each episode in isolation.** One scenario, one environment, its own
   simulator process, its own collection session, a complete shutdown, a
   cooldown, and only then the next.
3. **Catch up again.** The freshly collected episodes are now the unlearned
   data, so the same routine trains on them and the quality gate decides whether
   the new weights get deployed.
4. **Deploy.** The live stack starts on whatever model survived the gate.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, List

from simlab.config import load_config
from simlab.ml.continual import catch_up, load_perception_config
from simlab.scenarios.episode import (
    EpisodePlan,
    build_plan,
    run_episode,
    run_stamp,
    write_episode_config,
)
from simlab.utils.paths import PROJECT_ROOT, resolve_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the scenario plan end to end.")
    parser.add_argument("--config", default="configs/default.yaml", help="scene YAML with the plan")
    parser.add_argument("--perception", default="configs/perception.yaml")
    parser.add_argument("--scenario", action="append", default=[],
                        help="only fly these scenarios (repeatable)")
    parser.add_argument("--environment", action="append", default=[],
                        help="only fly these environments (repeatable)")
    parser.add_argument("--episodes", type=int, help="stop after this many episodes")
    parser.add_argument("--seed", type=int, help="base seed; makes the whole run reproducible")
    parser.add_argument("--episode-timeout", type=float, default=None,
                        help="seconds before an episode is force-stopped (default: duration + 420)")
    parser.add_argument("--no-bag", action="store_true", help="skip rosbag recording")
    parser.add_argument("--no-train-first", action="store_true",
                        help="do not learn outstanding data before flying")
    parser.add_argument("--no-train-after", action="store_true",
                        help="collect only; leave the new data for the next start")
    parser.add_argument("--no-deploy", action="store_true", help="stop after training")
    parser.add_argument("--dry-run", action="store_true",
                        help="write the episode configs and print the plan, fly nothing")
    parser.add_argument("--deploy-config", default="configs/default.yaml")
    return parser


def select(plans: List[EpisodePlan], args: argparse.Namespace) -> List[EpisodePlan]:
    chosen = [
        plan
        for plan in plans
        if (not args.scenario or plan.scenario in args.scenario)
        and (not args.environment or plan.environment in args.environment)
    ]
    return chosen[: args.episodes] if args.episodes else chosen


def describe_result(result: Dict[str, Any]) -> str:
    session = result.get("session") or {}
    summary = result.get("summary") or {}
    blackout = summary.get("blackout_seconds") or {}
    dark = sum(blackout.values())
    return (
        f"  {result['episode_id']}: status={result['status']} "
        f"images={session.get('images', 0)} labelled={session.get('labelled_objects', 0)} "
        f"occluded={session.get('occluded_objects', 0)} "
        f"min_separation={summary.get('minimum_separation_m', '-')}m "
        f"blackout={dark:.1f}s wall={result['wall_seconds']}s"
    )


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.chdir(PROJECT_ROOT)
    scene = load_config(args.config)
    perception = load_perception_config(args.perception)
    if not scene.scenarios.plan:
        print(f"[plan] {args.config} has no scenarios.plan entries; nothing to fly")
        return 2

    stamp = run_stamp()
    base_seed = (
        args.seed
        if args.seed is not None
        else scene.scenarios.seed
        if scene.scenarios.seed is not None
        else random.SystemRandom().randrange(1, 2**20)
    )
    plans = select(build_plan(scene, stamp, base_seed), args)
    if not plans:
        print("[plan] the filters matched no episodes")
        return 2

    print(f"[plan] run {stamp}, base seed {base_seed}, {len(plans)} episode(s):")
    for plan in plans:
        write_episode_config(plan, args.config)
        print(
            f"  {plan.index:02d} {plan.scenario:22s} {plan.environment:16s} "
            f"seed={plan.seed} {plan.duration_s:g}s -> {plan.config_path}"
        )
    if args.dry_run:
        return 0

    if not args.no_train_first:
        print("[start-up] learning anything collected earlier but never trained on")
        catch_up(perception, scene)

    run_dir = plans[0].run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    ledger = run_dir / "episodes.jsonl"
    results: List[Dict[str, Any]] = []
    sample_period = float(perception["auto_label"]["sample_period_s"])
    min_visibility = float(perception["auto_label"].get("label_min_visibility", 0.35))

    for position, plan in enumerate(plans):
        timeout = args.episode_timeout or (plan.duration_s + 420.0)
        print(
            f"\n[episode {position + 1}/{len(plans)}] {plan.scenario} in {plan.environment} "
            f"({plan.duration_s:g}s, seed {plan.seed})"
        )
        result = run_episode(
            plan,
            sample_period_s=sample_period,
            min_visibility=min_visibility,
            timeout_s=timeout,
            record_bag=not args.no_bag,
        )
        results.append(result)
        with ledger.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, sort_keys=True) + "\n")
        print(describe_result(result))
        if position + 1 < len(plans) and scene.scenarios.cooldown_s > 0:
            # Let the previous run's DDS participants expire before the next
            # simulator claims the same topic names.
            print(f"[episode] cooldown {scene.scenarios.cooldown_s:g}s")
            time.sleep(scene.scenarios.cooldown_s)

    print("\n[plan] finished:")
    for result in results:
        print(describe_result(result))
    (run_dir / "run_summary.json").write_text(
        json.dumps({"stamp": stamp, "base_seed": base_seed, "episodes": results}, indent=2),
        encoding="utf-8",
    )

    if not args.no_train_after:
        print("\n[training] learning the sessions this run collected")
        report = catch_up(perception, scene)
        print(json.dumps(report, indent=2))

    if args.no_deploy:
        return 0
    command = [
        "ros2", "launch", "launch/simlab.launch.py",
        f"config:={resolve_path(args.deploy_config)}",
        "rviz:=true", "pipeline:=true", "yolo:=true",
    ]
    print("\n[deploy] starting Isaac Sim + tracker + YOLO boxes + RViz2")
    sys.stdout.flush()
    os.execvp(command[0], command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
