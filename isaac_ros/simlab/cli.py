"""Command line front end.

This module owns the one ordering rule that the rest of the package depends on:
``SimulationApp`` must exist before any ``omni``/``isaacsim`` module is
imported. Config parsing and argument handling therefore happen first, and
``simlab.sim.runner`` is imported only after Kit is up.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional, Sequence

from simlab.config import load_config
from simlab.config.loader import DEFAULT_CONFIG
from simlab.config.schema import SCENARIO_KEYS
from simlab.scenarios.environments import ENVIRONMENT_KEYS
from simlab.utils.logging import get_logger

log = get_logger("scene")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m simlab",
        description=(
            "Isaac Sim scene: quadrotors on randomized crossing shuttles, "
            "optionally flying one of the tracking-failure scenarios."
        ),
    )
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG), help="scene YAML (default: configs/default.yaml)"
    )
    parser.add_argument("--headless", action="store_true", help="run without the Kit GUI")
    parser.add_argument("--ugv", help="override ugv.model, e.g. nova_carter or jetbot")
    parser.add_argument("--controller", help="override ugv.controller.name")
    parser.add_argument("--people", type=int, help="override people.count")
    parser.add_argument("--allies", type=int, help="override drones.friendly.count")
    parser.add_argument("--enemies", type=int, help="override drones.enemy.count")
    parser.add_argument(
        "--seconds", type=float, help="override app.duration_s (0 = until closed)"
    )
    parser.add_argument(
        "--scenario",
        choices=SCENARIO_KEYS,
        help="fly one tracking-failure scenario; see ./run_scenarios.sh for the full plan",
    )
    parser.add_argument(
        "--environment", choices=ENVIRONMENT_KEYS, help="override world.environment"
    )
    parser.add_argument(
        "--seed", type=int, help="scenario and layout seed; makes a run reproducible"
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="override any config leaf, e.g. --set ugv.controller.params.linear_speed=0.8",
    )
    return parser


def _coerce(text: str) -> Any:
    """Parse a --set value as YAML so numbers, bools and lists come through typed."""
    import yaml

    return yaml.safe_load(text)


def _nest(path: str, value: Any) -> Dict[str, Any]:
    node: Any = value
    for key in reversed(path.split(".")):
        node = {key: node}
    return node


def collect_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Turn CLI flags into a config-shaped dict to merge over the YAML."""
    from simlab.config.loader import deep_merge

    overrides: Dict[str, Any] = {}
    if args.headless:
        overrides.setdefault("app", {})["headless"] = True
    if args.seconds is not None:
        overrides.setdefault("app", {})["duration_s"] = args.seconds
    if args.ugv:
        overrides.setdefault("ugv", {})["model"] = args.ugv
    if args.controller:
        overrides.setdefault("ugv", {}).setdefault("controller", {})["name"] = args.controller
    if args.people is not None:
        overrides.setdefault("people", {})["count"] = args.people
    if args.allies is not None:
        overrides.setdefault("drones", {}).setdefault("friendly", {})["count"] = args.allies
    if args.enemies is not None:
        overrides.setdefault("drones", {}).setdefault("enemy", {})["count"] = args.enemies
    if args.scenario:
        overrides.setdefault("scenarios", {}).update(
            {"enabled": True, "active": args.scenario}
        )
    if args.environment:
        overrides.setdefault("world", {})["environment"] = args.environment
    if args.seed is not None:
        overrides.setdefault("scenarios", {})["seed"] = args.seed
        overrides.setdefault("world", {})["environment_seed"] = args.seed

    for item in args.set:
        if "=" not in item:
            raise ValueError(f"--set expects PATH=VALUE, got {item!r}")
        path, _, raw = item.partition("=")
        overrides = deep_merge(overrides, _nest(path.strip(), _coerce(raw)))
    return overrides


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, collect_overrides(args))

    # Kit first -- nothing below simlab.sim may be imported before this call.
    from simlab.sim.app import (
        PEOPLE_EXTENSIONS,
        ROS2_EXTENSIONS,
        enable_extensions,
        fresh_stage,
        launch,
    )

    extensions = [] if cfg.drones.enabled else list(PEOPLE_EXTENSIONS)
    if cfg.ros2.enabled:
        extensions += [e for e in ROS2_EXTENSIONS if e not in extensions]

    app = launch(headless=cfg.app.headless)
    try:
        enable_extensions(app, extensions)
        fresh_stage(app)

        from simlab.sim.runner import SimulationRunner

        runner = SimulationRunner(app, cfg)
        runner.setup()
        runner.run()
    finally:
        app.close()
    return 0
