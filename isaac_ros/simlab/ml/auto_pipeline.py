"""One-command collection -> SAM labelling -> YOLO training -> deployment.

A single ad-hoc episode: no scenario, the plain crossing shuttle. For the three
tracking-failure scenarios flown back to back, use
``simlab.scenarios.orchestrator`` (``./run_scenarios.sh``) instead -- both share
the same episode runner and the same catch-up training routine.
"""

from __future__ import annotations

import argparse
import json
import os
import random

from simlab.config import load_config
from simlab.ml.continual import catch_up, load_perception_config
from simlab.scenarios.episode import adhoc_plan, run_episode, run_stamp, write_episode_config
from simlab.utils.paths import PROJECT_ROOT, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/perception.yaml")
    parser.add_argument("--scene", default="configs/default.yaml")
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="train this run from the base weights while retaining the cumulative dataset",
    )
    parser.add_argument("--no-final", action="store_true", help="stop after collection/training")
    parser.add_argument("--no-bag", action="store_true", help="skip rosbag recording")
    parser.add_argument("--collection-seconds", type=int)
    args, launch_args = parser.parse_known_args()
    os.chdir(PROJECT_ROOT)

    perception = load_perception_config(args.config)
    scene = load_config(args.scene)
    seconds = args.collection_seconds or int(perception["auto_label"]["collection_seconds"])
    plan = adhoc_plan(
        scene, run_stamp(), float(seconds), random.SystemRandom().randrange(1, 2**20)
    )
    write_episode_config(plan, args.scene)
    print(f"[collect] {seconds}s episode -> {plan.raw_dir} (bag: {plan.bag_dir})")
    result = run_episode(
        plan,
        sample_period_s=float(perception["auto_label"]["sample_period_s"]),
        min_visibility=float(perception["auto_label"].get("label_min_visibility", 0.35)),
        timeout_s=seconds + 420.0,
        record_bag=not args.no_bag,
    )
    if result["status"] != "ok":
        print(f"[collect] episode ended with status {result['status']}")

    report = catch_up(perception, scene, force_base_model=args.force_retrain)
    print(json.dumps(report, indent=2))

    if args.no_final:
        return
    command = [
        "ros2", "launch", "launch/simlab.launch.py",
        f"config:={resolve_path(args.scene)}",
        "rviz:=true", "pipeline:=true", "yolo:=true", *launch_args,
    ]
    print("[deploy] starting Isaac Sim + tracker + YOLO boxes + RViz2")
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
