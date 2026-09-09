#!/usr/bin/env python3
"""Protocol-only fake used to validate the MATLAB adapter without ROS/PX4.

It answers the current wire format: pad-relative position and velocity, the
deck's own twist, a depleting energy budget and a canyon GNSS fix. The deck
actually moves, the pack actually drains and the fix actually degrades, so the
adapter exercises the moving-target, energy and navigation paths rather than a
frozen best case.

It reaches the requested entry pose instantly, because the protocol check has to
be deterministic and must not wait on a flight.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import time


HOVER_THRUST = 0.580          # mirrors px4.hover_thrust in config/system.yaml
COLLECTIVE_SPAN = 0.85
HOVER_POWER_W = 185.8         # what battery.BatteryModel derives from that config
LANDING_RESERVE_S = 2.5
DECK_SPEED = 5.5              # m/s, a lorry moving with urban traffic
DECK_OMEGA = 0.09


def deck_state(t: float) -> tuple[list[float], list[float], float]:
    """A deck driving a gentle circle, with an exact analytic twist."""
    radius = DECK_SPEED / DECK_OMEGA
    position = [radius * math.sin(DECK_OMEGA * t), radius * (1 - math.cos(DECK_OMEGA * t)), 3.20]
    velocity = [DECK_SPEED * math.cos(DECK_OMEGA * t), DECK_SPEED * math.sin(DECK_OMEGA * t), 0.0]
    yaw = DECK_OMEGA * t
    return position, velocity, yaw


class Fake:
    def __init__(self) -> None:
        self.t0 = time.monotonic()
        self.position = [0.9, -0.4, 4.2]      # pad-relative
        self.velocity = [0.0, 0.0, -0.2]      # pad-relative
        self.hover_seconds = 22.0
        self.remaining_j = self.hover_seconds * HOVER_POWER_W
        self.initial_j = self.remaining_j
        self.power_w = 0.0
        self.last_burn = time.monotonic()

    @property
    def t(self) -> float:
        return time.monotonic() - self.t0

    def reset(self, hover_seconds: float) -> None:
        self.position = [0.9, -0.4, 4.2]
        self.velocity = [0.0, 0.0, -0.2]
        self.hover_seconds = hover_seconds
        self.initial_j = hover_seconds * HOVER_POWER_W
        self.remaining_j = self.initial_j
        self.last_burn = time.monotonic()

    def burn(self, collective: float) -> None:
        now = time.monotonic()
        dt = min(max(now - self.last_burn, 0.0), 0.1)
        self.last_burn = now
        thrust = max(HOVER_THRUST * (1.0 + COLLECTIVE_SPAN * collective), 0.0)
        self.power_w = HOVER_POWER_W * (thrust / HOVER_THRUST) ** 1.5
        self.remaining_j = max(0.0, self.remaining_j - self.power_w * dt)

    def gnss(self) -> dict:
        """A fix that cycles between canyon and intersection.

        The period is nothing physical; the point is that a client which
        assumes a constant fix quality is caught here rather than in Isaac.
        """
        openness = 0.5 + 0.5 * math.sin(0.35 * self.t)
        nlos = int(round(5 * (1.0 - openness)))
        sigma = 2.0 + 16.0 * (1.0 - openness)
        return {
            "enabled": True, "source": "fake", "valid": True, "fix_type": 3,
            "satellites_tracked": 11,
            "nlos_detected_fraction": 0.7 * nlos / 11.0,
            "cn0_mean_db": 44.0 - 5.0 * (1.0 - openness),
            "hdop": 1.0 + 0.9 * (1.0 - openness), "vdop": 1.8,
            "residual_rms_m": 0.5 * sigma, "sigma_xy_m": sigma,
            "quality": max(0.03, math.exp(-max(sigma - 1.5, 0.0) / 10.0)),
            "deck_quality": max(0.03, math.exp(-max(sigma - 1.0, 0.0) / 10.0)),
            "deck_sigma_xy_m": sigma * 0.9,
        }

    def battery(self) -> dict:
        hover_left = self.remaining_j / HOVER_POWER_W
        return {
            "enabled": True, "source": "model",
            "remaining_j": self.remaining_j, "initial_j": self.initial_j,
            "capacity_j": 139860.0,
            "energy_used_j": max(0.0, self.initial_j - self.remaining_j),
            "power_w": self.power_w, "hover_power_w": HOVER_POWER_W,
            "hover_seconds_remaining": hover_left,
            "reserve": min(max(hover_left / 20.0, 0.0), 1.0),
            "landing_reserve_s": LANDING_RESERVE_S,
            "state_of_charge": self.remaining_j / 139860.0,
            "voltage_v": 11.1,
            "depleted": self.remaining_j <= 0.0,
        }

    def state(self, seq: int, ack_seq: int) -> dict:
        pad_position, pad_velocity, pad_yaw = deck_state(self.t)
        return {
            "v": 1, "type": "state", "seq": seq, "ack_seq": ack_seq,
            "time_ns": time.monotonic_ns(), "sample_time_ns": time.monotonic_ns(),
            "px4_time_us": time.monotonic_ns() // 1000,
            "frame": "ENU_FLU", "position_frame": "pad",
            "position": list(self.position), "velocity": list(self.velocity),
            "world": {
                "position": [p + d for p, d in zip(self.position, pad_position)],
                "velocity": [v + d for v, d in zip(self.velocity, pad_velocity)],
            },
            "pad": {
                "valid": True, "source": "fake",
                "position": pad_position, "velocity": pad_velocity,
                "yaw": pad_yaw, "yaw_rate": DECK_OMEGA,
                "speed": math.hypot(pad_velocity[0], pad_velocity[1]),
                "sigma_xy_m": self.gnss()["deck_sigma_xy_m"],
            },
            "battery": self.battery(),
            "gnss": self.gnss(),
            # The simulator's own pad-relative geometry, for scoring only. The
            # fake makes it differ from the reported state by a metre so that a
            # client which grades on the sensor is caught here.
            "truth": {"valid": True,
                      "position": [self.position[0] + 1.0,
                                   self.position[1] - 0.7,
                                   self.position[2]],
                      "velocity": list(self.velocity)},
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "angular_velocity": [0.0, 0.0, 0.0], "acceleration": [0.0, 0.0, 0.0],
            "wind": [1.0, 0.2, 0.0], "aero_force": [0.1, 0.0, 0.0],
            "marker_quality": 0.8, "armed": True, "nav_state": 14,
            "landed": False, "estimator_valid": True, "source": "fake",
            # Mirrors config/system.yaml so the MATLAB adapter can verify that the
            # gateway scales actions the way the policy assumes.
            "extra": {"control_mapping": {"hover_thrust": HOVER_THRUST,
                                          "collective_span": COLLECTIVE_SPAN,
                                          "max_roll_pitch_rad": math.radians(28.0),
                                          "max_yaw_rate_rad_s": math.radians(90.0)},
                      "position_source": "uav_pose_in_pad",
                      "land_detector": "live",
                      "land_detector_authoritative": False,
                      "px4_thrust": HOVER_THRUST},
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=14650)
    args = parser.parse_args()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    fake = Fake()
    tx_seq = 0
    while True:
        raw, peer = sock.recvfrom(32768)
        msg = json.loads(raw.decode("utf-8"))
        tx_seq += 1
        kind = msg["type"]
        if kind == "action":
            fake.burn(float(msg["action"][0]))
            reply = fake.state(tx_seq, msg["seq"])
        elif kind in {"hello", "state"}:
            reply = fake.state(tx_seq, msg["seq"])
        elif kind == "goto":
            # Snapped to, so the check never waits on a climb. The request is a
            # pad-frame offset, and this fake already reports pad-relative
            # position, so the offset is the position.
            fake.position = list(msg["position"])
            fake.velocity = [0.0, 0.0, 0.0]
            reply = {"v": 1, "type": "ack", "seq": tx_seq, "ack_seq": msg["seq"],
                     "time_ns": time.monotonic_ns(), "status": "goto_started",
                     "detail": {"position": fake.position,
                                "frame": msg.get("frame", "world"),
                                "yaw": msg.get("yaw", 0.0)}}
        elif kind == "reset":
            hover_seconds = 18.0
            fake.reset(hover_seconds)
            pad_position, _, _ = deck_state(fake.t)
            offset = [0.5, -0.2, 4.0]
            reply = {"v": 1, "type": "ack", "seq": tx_seq, "ack_seq": msg["seq"],
                     "time_ns": time.monotonic_ns(), "status": "reset_complete",
                     "detail": {"v": 1, "seq": msg["seq"], "seed": msg.get("seed", 0),
                                "wind_scale": msg.get("wind_scale", 1.0),
                                "entry_offset_pad_m": offset,
                                "entry_position_enu_m":
                                    [o + p for o, p in zip(offset, pad_position)],
                                "entry_yaw_enu_rad": 0.15,
                                "battery_hover_seconds": hover_seconds,
                                "reseated_on_deck": False}}
        elif kind in {"arm", "disarm", "enable_offboard", "disable_offboard"}:
            reply = {"v": 1, "type": "ack", "seq": tx_seq, "ack_seq": msg["seq"],
                     "time_ns": time.monotonic_ns(), "status": kind + "_complete"}
        else:
            reply = {"v": 1, "type": "error", "seq": tx_seq,
                     "time_ns": time.monotonic_ns(), "error": "unsupported"}
        sock.sendto(json.dumps(reply, separators=(",", ":")).encode("utf-8"), peer)


if __name__ == "__main__":
    main()
