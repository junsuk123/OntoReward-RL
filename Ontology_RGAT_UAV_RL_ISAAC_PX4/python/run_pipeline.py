#!/usr/bin/env python3
"""One command that runs the whole external experiment.

Starts the Micro XRCE-DDS agent, Isaac Sim + Pegasus + PX4 SITL and the
gateway, checks the link with one expert episode, runs dataset generation,
R-GAT training, both PPO runs, the paired evaluation and the figures, then
shuts down whatever it started. Processes that were already running are adopted
and left running.

``ISAACSIM_PATH`` must point at the Isaac Sim release directory -- the one with
``python.sh`` -- either in the environment or via ``--isaac-sim-path``.

Quick mode flies roughly 550 episodes and full mode roughly 8500, counting the
wind, deck-speed, GNSS-degradation and battery sweeps. Each is 20 s of flight
plus the climb to the entry pose, and PX4 runs in real time, so budget hours for
quick and days for full. If the simulator stops accepting arm commands mid-run,
the pipeline cycles it and continues (``cfg.external.reset_recoveries``).

See docs/OPERATIONS.md for the startup order this automates, and the README's
"Known limitations" before reading the numbers it produces.
"""
from __future__ import annotations

import sys
import os
import shutil
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ontology_rgat import stack as stack_module
from ontology_rgat.cli import base_parser, config_from_args, ensure_fastdds
from ontology_rgat.env import run_episode
from ontology_rgat.expert import PolicySpec
from ontology_rgat.pipeline import run_all
from ontology_rgat.stack import ExternalStack
from ontology_rgat.viz.dashboard import Dashboard
from ontology_rgat.viz.live import EpisodeMonitor
from ontology_rgat.viz.rviz import RvizPublisher


def main() -> int:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--isaac-sim-path", default=None,
                        help="Isaac Sim release directory; defaults to $ISAACSIM_PATH")
    parser.add_argument("--headless", action="store_true",
                        help="run Isaac Sim without a GUI window; GUI is the default")
    parser.add_argument("--use-running-stack", action="store_true",
                        help="attach to a stack you started yourself")
    parser.add_argument("--keep-stack", action="store_true",
                        help="leave the simulator up afterwards")
    parser.add_argument("--no-smoke-test", action="store_true",
                        help="skip the pre-flight episode")
    parser.add_argument("--smoke-test-only", action="store_true",
                        help="bring the stack up, fly one episode, tear it down")
    args = parser.parse_args()

    if args.target == "hardware":
        parser.error("This entry point arms and flies the vehicle unattended and is "
                     "SITL only. Use run_hardware_policy.py for a real vehicle.")

    ensure_fastdds()
    cfg = config_from_args(args)

    # Drop any stack left registered by an interrupted run.
    stack_module.current(None)
    owned = None
    rviz_process = None
    try:
        if args.use_running_stack:
            print("Using the stack that is already running.")
        else:
            owned = ExternalStack(cfg, isaac_sim_path=args.isaac_sim_path,
                                  headless=bool(args.headless))
            owned.start()
            # Let env.LandingEnv.reset cycle this simulator if PX4 stops arming.
            stack_module.current(owned)

        if cfg.viz.rviz.enabled and not args.headless and shutil.which("rviz2"):
            rviz_script = Path(__file__).resolve().parents[1] / "scripts" / "run_rviz.sh"
            if rviz_script.is_file() and os.environ.get("DISPLAY"):
                rviz_process = subprocess.Popen(
                    [str(rviz_script)], cwd=str(rviz_script.parent.parent),
                    stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                print("RViz 2 started with the landing ontology configuration.")

        if not args.no_smoke_test or args.smoke_test_only:
            # One expert episode costs about a minute and catches a broken link,
            # a mis-scaled action mapping or a refused arm before hours of training.
            print("\n=== 0) Pre-flight episode ===")
            smoke_dashboard = Dashboard(cfg).start()
            smoke_rviz = RvizPublisher.create(cfg)
            try:
                log = run_episode(
                    PolicySpec("expert"), "sparse", None, cfg.seed, cfg,
                    monitor=EpisodeMonitor(cfg, rviz=smoke_rviz, label="preflight"))
            finally:
                if smoke_rviz is not None:
                    smoke_rviz.close()
                if smoke_dashboard is not None:
                    smoke_dashboard.stop()
            for key, value in log.metrics.items():
                print(f"  {key:<24} {value}")
            if args.smoke_test_only:
                print("\nPre-flight only: the stack came up and one episode flew.")
                return 0

        run_all(cfg)
        return 0
    finally:
        if rviz_process is not None and rviz_process.poll() is None:
            rviz_process.terminate()
            try:
                rviz_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                rviz_process.kill()
        stack_module.current(None)
        if owned is not None:
            if args.keep_stack:
                print("Stack left running by request; stop it with "
                      "./scripts/stack_status.sh and kill the process group.")
            else:
                owned.stop()


if __name__ == "__main__":
    raise SystemExit(main())
