"""Shared command-line plumbing for the entry points."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .config import Config, default_config

__all__ = ["base_parser", "config_from_args", "ensure_fastdds", "bootstrap_path"]


def bootstrap_path() -> None:
    """Put ``python/`` on ``sys.path`` when a script is run directly."""
    here = Path(__file__).resolve().parents[1]
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


def ensure_fastdds() -> None:
    """Speak the RMW the XRCE-DDS agent speaks.

    Every consumer of ``/fmu/*`` must use Fast DDS, because that is what the
    agent speaks. A shell with a different default sees the topics but reads
    nothing from them, which looks like a dead simulator rather than a
    misconfiguration, so this is set rather than assumed.
    """
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    os.environ.pop("CYCLONEDDS_URI", None)


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mode", default="quick", choices=("quick", "full"),
                        help="quick is for smoke tests and iteration; full is the "
                             "paper-scale sweep")
    parser.add_argument("--target", default="sitl", choices=("sitl", "hardware"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rgat-device", default=None, choices=("auto", "cuda", "cpu"),
                        help="override cfg.device.rgat")
    parser.add_argument("--ppo-device", default=None,
                        help="override cfg.device.ppo (cpu or cuda)")
    parser.add_argument("--no-rviz", action="store_true",
                        help="do not publish the RViz 2 topics")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="do not serve the live training dashboard")
    parser.add_argument("--dashboard-port", type=int, default=None)
    parser.add_argument("--no-live-export", action="store_true",
                        help="do not write results/live PNG and CSV snapshots")
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    cfg = default_config(args.mode, args.target)
    if args.seed is not None:
        cfg.seed = int(args.seed)
    if args.rgat_device:
        cfg.device.rgat = args.rgat_device
    if args.ppo_device:
        cfg.device.ppo = args.ppo_device
    if args.no_rviz:
        cfg.viz.rviz.enabled = False
    if args.no_dashboard:
        cfg.viz.dashboard.enabled = False
    if args.dashboard_port is not None:
        cfg.viz.dashboard.port = int(args.dashboard_port)
    if args.no_live_export:
        cfg.viz.live_export = False
    return cfg
