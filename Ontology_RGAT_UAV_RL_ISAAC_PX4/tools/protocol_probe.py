#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "ontology_rgat_px4"))

from ontology_rgat_px4.config import load_gateway_config
from ontology_rgat_px4.protocol import decode, encode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("hello", "state", "disarm"), default="state")
    parser.add_argument("--config", default=str(ROOT / "config" / "system.yaml"))
    args = parser.parse_args()
    cfg = load_gateway_config(args.config)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2.0)
    msg = {"v": cfg.protocol_version, "type": args.command, "seq": 1,
           "time_ns": time.monotonic_ns()}
    sock.sendto(encode(msg), (cfg.bind_host, cfg.gateway_port))
    raw, _ = sock.recvfrom(32768)
    print(json.dumps(decode(raw, cfg.protocol_version), indent=2))


if __name__ == "__main__":
    main()

