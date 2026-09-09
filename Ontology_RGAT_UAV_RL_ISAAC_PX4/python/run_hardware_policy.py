#!/usr/bin/env python3
"""Hardware runtime. Never arms the vehicle.

The pilot or QGroundControl arms it; this script refuses to start otherwise,
stops when the pilot disarms, and yields to the configured PX4 offboard-loss
behaviour on exit. Read docs/HARDWARE_SAFETY.md first. First tests must be
performed without propellers.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from ontology_rgat.bridge import BridgeError, PX4Bridge
from ontology_rgat.cli import base_parser, config_from_args, ensure_fastdds
from ontology_rgat.env import LandingEnv
from ontology_rgat.expert import PolicySpec
from ontology_rgat.ppo.networks import load_agent
from ontology_rgat.rgat.model import load_potential
from ontology_rgat.semantic import SemanticState, build_ontology_graph
from ontology_rgat.viz.live import EpisodeMonitor
from ontology_rgat.viz.rviz import RvizPublisher


def _wait_hardware_battery(bridge: PX4Bridge, state: dict, timeout: float) -> dict:
    """Never fly an energy-aware hardware policy on the seeded SITL stand-in."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        battery = state.get("battery")
        if isinstance(battery, dict) and str(battery.get("source", "")).lower() == "px4":
            return state
        time.sleep(0.05)
        state = bridge.get_state()
    raise SystemExit("No live PX4 battery_status reached the gateway. The "
                     "energy-aware hardware policy refuses to substitute the "
                     "SITL battery model.")


def main() -> int:
    parser = base_parser(__doc__ or "")
    parser.set_defaults(target="hardware")
    args = parser.parse_args()
    if args.target != "hardware":
        parser.error("this entry point is for --target hardware")

    ensure_fastdds()
    cfg = config_from_args(args)
    models = Path(cfg.paths.models)
    agent, _ = load_agent(models / "ppo_rgats_pbrs_external.pt", cfg)
    policy = PolicySpec("ppo", agent=agent, deterministic=True)
    template = build_ontology_graph(SemanticState(), cfg)
    potential_path = models / "rgat_model_external.pt"
    potential = (load_potential(potential_path, cfg, template)[0]
                 if potential_path.is_file() else None)

    print("HARDWARE MODE: propellers must have been removed for initial tests.\n"
          "RC/QGroundControl takeover and the PX4 offboard-loss failsafe must be "
          "active.", file=sys.stderr)

    bridge = PX4Bridge(cfg)
    rviz = RvizPublisher.create(cfg, potential=potential)
    monitor = EpisodeMonitor(cfg, rviz=rviz, label="hardware", potential=potential)
    info: dict = {"status": "not started"}
    try:
        state = bridge.wait_valid_state()
        state = _wait_hardware_battery(bridge, state, cfg.external.estimator_warmup)
        if not state["armed"]:
            raise SystemExit("Vehicle is not armed. Arm only through the "
                             "pilot/QGroundControl path, then restart this script.")
        if state["marker_quality"] <= 0:
            print("WARNING: marker quality is zero; verify the landing perception "
                  "publisher.", file=sys.stderr)

        env = LandingEnv(bridge, cfg, seed=-1, state=state)
        bridge.enable_offboard()
        from ontology_rgat.env import EpisodeLog
        log = EpisodeLog()
        for k in range(1, cfg.sim.max_steps + 1):
            cur = env.current()
            action = policy.action(cur, env.x, cfg)
            x_before, diag_before, t_before = env.x.copy(), env.last_diag, env.t
            _, r, done, info = env.step(action, cur, "sparse", potential)
            log.t.append(t_before); log.x.append(x_before); log.r.append(r)
            log.a.append(np.asarray(action)); log.tilt.append(cur.sem.tilt)
            log.wind.append(diag_before["mean_wind_i"])
            log.aero_force.append(diag_before["aero_force_i"])
            log.aero_mag.append(diag_before["aero_force_mag"])
            log.pad_pos.append(diag_before["pad_position_i"])
            log.pad_vel.append(diag_before["pad_velocity_i"])
            log.pad_speed.append(diag_before["pad_speed"])
            log.marker_quality.append(diag_before["marker_quality"])
            log.battery_reserve.append(diag_before["battery"]["reserve"])
            log.battery_power_w.append(diag_before["battery"]["power_w"])
            log.hover_seconds_left.append(diag_before["battery"]["hover_seconds_remaining"])
            log.closing_speed.append(cur.sem.closing_speed)
            log.energy_margin.append(cur.sem.energy_margin)
            log.graph_x.append(cur.graph.X)
            log.phi.append(float("nan"))
            monitor(log, k, cur, info)
            if k % 10 == 0:
                print(f"t={env.t:5.2f} z={env.x[2]:6.2f} "
                      f"xy={float(np.linalg.norm(env.x[:2])):5.2f} "
                      f"marker={env.last_diag['marker_quality']:.2f} "
                      f"battery={100 * env.last_diag['battery']['state_of_charge']:.1f}% "
                      f"armed={info['armed']} nav={info['nav_state']}")
            if not info["armed"] or done:
                break
    finally:
        try:
            bridge.disable_offboard()
        except BridgeError:
            pass
        bridge.close()
        if rviz is not None:
            rviz.close()
    print(f"Hardware policy stopped: {info['status']}. "
          "Pilot/PX4 fallback now owns the vehicle.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
