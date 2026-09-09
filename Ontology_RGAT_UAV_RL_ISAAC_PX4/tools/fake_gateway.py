#!/usr/bin/env python3
"""Protocol-only fake used to validate the MATLAB adapter without ROS/PX4."""

from __future__ import annotations

import argparse
import json
import math
import socket
import time


# A plausible seeded entry pose; the real one comes from Isaac's reset acknowledgement.
ENTRY_POSITION = [0.5, -0.2, 4.0]
ENTRY_YAW = 0.15


def state(seq: int, ack_seq: int, position: list[float], velocity: list[float]) -> dict:
    return {
        "v": 1, "type": "state", "seq": seq, "ack_seq": ack_seq,
        "time_ns": time.monotonic_ns(), "sample_time_ns": time.monotonic_ns(),
        "px4_time_us": time.monotonic_ns() // 1000,
        "frame": "ENU_FLU", "position": list(position),
        "velocity": list(velocity), "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "angular_velocity": [0.0, 0.0, 0.0], "acceleration": [0.0, 0.0, 0.0],
        "wind": [1.0, 0.2, 0.0], "aero_force": [0.1, 0.0, 0.0],
        "marker_quality": 0.8, "armed": True, "nav_state": 14,
        "landed": False, "estimator_valid": True, "source": "fake",
        # Mirrors config/system.yaml so the MATLAB adapter can verify that the
        # gateway scales actions the way the policy assumes.
        "extra": {"control_mapping": {"hover_thrust": 0.50, "collective_span": 0.85,
                                      "max_roll_pitch_rad": math.radians(28.0),
                                      "max_yaw_rate_rad_s": math.radians(90.0)}},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=14650)
    args = parser.parse_args()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    tx_seq = 0
    # The fake reaches the requested entry pose instantly so the MATLAB
    # protocol check stays deterministic and does not wait on a flight.
    position = [0.5, -0.2, 3.0]
    velocity = [0.0, 0.0, -0.2]
    while True:
        raw, peer = sock.recvfrom(32768)
        msg = json.loads(raw.decode("utf-8"))
        tx_seq += 1
        kind = msg["type"]
        if kind in {"hello", "state", "action"}:
            reply = state(tx_seq, msg["seq"], position, velocity)
        elif kind == "goto":
            position = list(msg["position"])
            velocity = [0.0, 0.0, 0.0]
            reply = {"v": 1, "type": "ack", "seq": tx_seq, "ack_seq": msg["seq"],
                     "time_ns": time.monotonic_ns(), "status": "goto_started",
                     "detail": {"position": position, "yaw": msg.get("yaw", 0.0)}}
        elif kind == "reset":
            position = [0.5, -0.2, 3.0]
            velocity = [0.0, 0.0, -0.2]
            reply = {"v": 1, "type": "ack", "seq": tx_seq, "ack_seq": msg["seq"],
                     "time_ns": time.monotonic_ns(), "status": "reset_complete",
                     "detail": {"v": 1, "seq": msg["seq"], "seed": msg.get("seed", 0),
                                "wind_scale": msg.get("wind_scale", 1.0),
                                "entry_position_enu_m": ENTRY_POSITION,
                                "entry_yaw_enu_rad": ENTRY_YAW,
                                "recovered_from_tipover": False}}
        elif kind in {"arm", "disarm", "enable_offboard", "disable_offboard"}:
            reply = {"v": 1, "type": "ack", "seq": tx_seq, "ack_seq": msg["seq"],
                     "time_ns": time.monotonic_ns(), "status": kind + "_complete"}
        else:
            reply = {"v": 1, "type": "error", "seq": tx_seq,
                     "time_ns": time.monotonic_ns(), "error": "unsupported"}
        sock.sendto(json.dumps(reply, separators=(",", ":")).encode("utf-8"), peer)


if __name__ == "__main__":
    main()
