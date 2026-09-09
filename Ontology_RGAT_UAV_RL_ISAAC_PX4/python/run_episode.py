#!/usr/bin/env python3
"""Fly one episode against a running stack, with the live views attached.

    python3 python/run_episode.py --policy expert           # pre-flight check
    python3 python/run_episode.py --policy proposed         # the trained arm
    python3 python/run_episode.py --policy manual --seed 7

The trained policies are loaded from ``results/models/`` and are refused if
their observation or ontology dimensions do not match this configuration: an
old fixed-pad model must be retrained, never transferred.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from ontology_rgat.cli import base_parser, config_from_args, ensure_fastdds
from ontology_rgat.env import run_episode
from ontology_rgat.expert import PolicySpec
from ontology_rgat.ppo.networks import load_agent
from ontology_rgat.rgat.model import load_potential
from ontology_rgat.semantic import SemanticState, build_ontology_graph
from ontology_rgat.viz.dashboard import Dashboard
from ontology_rgat.viz.live import EpisodeMonitor
from ontology_rgat.viz.rviz import RvizPublisher

CHECKPOINTS = {"manual": "ppo_manual_external.pt",
               "proposed": "ppo_rgats_pbrs_external.pt"}


def main() -> int:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--policy", default="expert",
                        choices=("expert", "manual", "proposed"))
    parser.add_argument("--reward", default=None,
                        choices=("sparse", "manual", "proposed"),
                        help="defaults to sparse, or proposed for the proposed policy")
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()

    ensure_fastdds()
    cfg = config_from_args(args)
    models = Path(cfg.paths.models)

    potential = None
    template = build_ontology_graph(SemanticState(), cfg)
    potential_path = models / "rgat_model_external.pt"
    if potential_path.is_file():
        potential, _ = load_potential(potential_path, cfg, template)

    if args.policy == "expert":
        policy = PolicySpec("expert")
        reward = args.reward or "sparse"
    else:
        agent, _ = load_agent(models / CHECKPOINTS[args.policy], cfg)
        policy = PolicySpec("ppo", agent=agent, deterministic=True)
        reward = args.reward or ("proposed" if args.policy == "proposed" else "sparse")
    if reward == "proposed" and potential is None:
        parser.error(f"{potential_path} is missing; the proposed reward needs the "
                     "trained R-GAT potential.")

    dashboard = Dashboard(cfg).start()
    rviz = RvizPublisher.create(cfg, potential=potential)
    monitor = EpisodeMonitor(cfg, rviz=rviz, label=args.policy, potential=potential)
    try:
        for i in range(args.episodes):
            seed = cfg.seed + i
            monitor.reset(f"{args.policy} seed {seed}")
            log = run_episode(policy, reward, potential, seed, cfg, monitor=monitor)
            print(json.dumps({k: (float(v) if isinstance(v, (int, float, np.floating))
                                  else v)
                              for k, v in log.metrics.items()}, indent=2))
    finally:
        if rviz is not None:
            rviz.close()
        if dashboard is not None:
            dashboard.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
