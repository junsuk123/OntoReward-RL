#!/usr/bin/env python3
"""Measure the collective that actually holds altitude, and report hover_thrust.

The gateway converts a normalised collective to PX4 body thrust with
``thrust = hover_thrust * (1 + collective_span * a0)``. Every policy assumes
``a0 = 0`` hovers, so a wrong ``hover_thrust`` biases the whole vertical axis and
shows up as a vehicle that will not stop descending however hard the controller
pushes. Re-run this whenever the airframe, its mass, or the Isaac version
changes, and put the reported value in ``config/system.yaml``.

Requires the full SITL stack and a gateway started with --allow-arm.
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "ontology_rgat_px4"))

from ontology_rgat_px4.config import load_gateway_config
from ontology_rgat_px4.protocol import decode, encode


class Client:
    def __init__(self, cfg):
        self.cfg = cfg
        self.seq = 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(2.0)
        self.dst = (cfg.bind_host, cfg.gateway_port)

    def send(self, kind: str, **fields):
        self.seq += 1
        self.sock.sendto(encode({"v": self.cfg.protocol_version, "type": kind,
                                 "seq": self.seq, "time_ns": time.monotonic_ns(),
                                 **fields}), self.dst)
        deadline = time.time() + 2.0
        while time.time() < deadline:
            try:
                reply = decode(self.sock.recvfrom(32768)[0], self.cfg.protocol_version)
            except socket.timeout:
                return None
            if reply.get("ack_seq") == self.seq:
                return reply
        return None


def climb_to_entry(client, seed: int, timeout_s: float):
    """Fly to the seeded entry pose and hold it.

    The deck is held still (pad_scale=0) and the wind switched off: this
    measures what it costs to hover, so anything the vehicle has to chase would
    only add thrust that is not hover thrust.
    """
    ack = client.send("reset", seed=seed, wind_scale=0.0, pad_scale=0.0)
    if not ack or "detail" not in ack or not ack["detail"]:
        raise SystemExit("no reset acknowledgement; is Isaac running?")
    detail = ack["detail"]
    if "entry_offset_pad_m" not in detail:
        raise SystemExit(
            "reset acknowledgement carries no entry_offset_pad_m; restart Isaac "
            "with the current landing_world.py"
        )
    # An offset in the pad frame, which is also the frame the state comes back
    # in, so the convergence test below compares like with like.
    entry = detail["entry_offset_pad_m"]
    client.send("goto", position=entry, yaw=0.0, hold_s=timeout_s, frame="pad")
    deadline = time.monotonic() + timeout_s
    last_arm = 0.0
    while time.monotonic() < deadline:
        state = client.send("state")
        if state:
            if not state["armed"] and time.monotonic() - last_arm > 2.0:
                last_arm = time.monotonic()
                client.send("arm")
            offset = max(abs(a - b) for a, b in zip(state["position"], entry))
            if offset < 0.45 and abs(state["velocity"][2]) < 0.35:
                return entry
        time.sleep(0.1)
    raise SystemExit("vehicle never reached the entry pose")


def sample_hover_thrust(client, sample_s: float):
    """Read PX4's own body thrust while its position controller holds a hover.

    Sweeping the collective and watching the climb rate needs more altitude
    than the arena has, and it flies the vehicle into the ground when the
    configured hover thrust is badly wrong. PX4 already computes the thrust
    that holds this airframe, so read that instead.
    """
    start = time.monotonic()
    thrusts = []
    speeds = []
    while time.monotonic() - start < sample_s:
        state = client.send("state")
        if state:
            thrust = state.get("extra", {}).get("px4_thrust")
            if thrust is None:
                raise SystemExit(
                    "gateway does not report px4_thrust; rebuild PX4 with "
                    "patches/px4-v1.14-publish-land-detected.patch and restart the gateway"
                )
            thrusts.append(float(thrust))
            speeds.append(abs(state["velocity"][2]))
        time.sleep(0.05)
    if not thrusts:
        raise SystemExit("gateway stopped replying while sampling the hover")
    return sum(thrusts) / len(thrusts), max(speeds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "system.yaml"))
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--sample", type=float, default=4.0, help="seconds of hover samples")
    args = parser.parse_args()

    cfg = load_gateway_config(args.config)
    client = Client(cfg)
    client.send("hello")
    climb_to_entry(client, args.seed, 45.0)

    measured, worst_speed = sample_hover_thrust(client, args.sample)
    # The collective the policy would have to hold to reach that thrust under
    # the configured mapping; it must be ~0 for an unbiased vertical axis.
    bias = (measured / cfg.hover_thrust - 1.0) / cfg.collective_span
    print(f"PX4 hover thrust while holding position: {measured:.3f} "
          f"(sampled over {args.sample:.1f} s, worst |vz| {worst_speed:.2f} m/s)")
    print(f"config/system.yaml px4.hover_thrust: {cfg.hover_thrust:.3f} -> {measured:.3f}")
    print(f"collective needed to hover under the current config: {bias:+.3f}")
    if abs(bias) > 0.05:
        print("The configured hover_thrust is biased; update it and re-run the smoke test.")


if __name__ == "__main__":
    main()
