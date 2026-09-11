#!/usr/bin/env python3
"""Start the complete graphical Meta-Sejong UAV/UGV viewing demo."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "ontology_rgat_px4"))

from ontology_rgat.config import default_config  # noqa: E402
from ontology_rgat.stack import ExternalStack  # noqa: E402
from ontology_rgat_px4.config import load_gateway_config  # noqa: E402


class DemoLink:
    """Small sequenced client for the gateway's localhost control protocol."""

    def __init__(self, config_path: Path):
        cfg = load_gateway_config(config_path)
        self.version = int(cfg.protocol_version)
        self.address = (str(cfg.bind_host), int(cfg.gateway_port))
        self.sequence = 0
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.settimeout(0.25)

    def close(self) -> None:
        self.socket.close()

    def request(self, kind: str, **fields):
        self.sequence += 1
        seq = self.sequence
        message = {"v": self.version, "type": kind, "seq": seq,
                   "time_ns": time.time_ns(), **fields}
        self.socket.sendto(
            json.dumps(message, separators=(",", ":"), allow_nan=False).encode(),
            self.address)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            try:
                raw = self.socket.recv(32768)
            except socket.timeout:
                continue
            reply = json.loads(raw.decode())
            if reply.get("type") == "error":
                raise RuntimeError(reply.get("error", "gateway rejected request"))
            if int(reply.get("ack_seq", -1)) == seq:
                return reply
        raise TimeoutError(f"gateway did not answer {kind!r}")

    def state(self):
        return self.request("state")

    def goto_hover(self, height_m: float, hold_s: float = 120.0):
        return self.request(
            "goto", position=[0.0, 0.0, float(height_m)], frame="pad",
            yaw=0.0, hold_s=float(hold_s))


def _wait_until(link: DemoLink, predicate, timeout_s: float, description: str):
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        try:
            last = link.state()
            if predicate(last):
                return last
        except (OSError, TimeoutError, json.JSONDecodeError):
            pass
        time.sleep(1.0)
    detail = "no state" if last is None else json.dumps({
        "armed": last.get("armed"), "landed": last.get("landed"),
        "estimator_valid": last.get("estimator_valid"),
        "preflight": (last.get("extra") or {}).get("pre_flight_checks_pass")})
    raise TimeoutError(f"timed out waiting for {description}: {detail}")


def _start_rviz(log_dir: Path) -> tuple[subprocess.Popen, object]:
    log_dir.mkdir(parents=True, exist_ok=True)
    stream = (log_dir / "rviz.log").open("wb")
    process = subprocess.Popen(
        [str(ROOT / "scripts" / "run_rviz.sh")], cwd=ROOT,
        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    time.sleep(2.0)
    if process.poll() is not None:
        raise RuntimeError(f"RViz exited during startup; see {log_dir / 'rviz.log'}")
    return process, stream


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=float, default=2.5,
                        help="pad-relative PX4 hover height in metres")
    parser.add_argument("--isaac-sim-path", default=os.environ.get("ISAACSIM_PATH"))
    args = parser.parse_args()
    if not 0.5 <= args.height <= 8.0:
        parser.error("--height must be between 0.5 and 8.0 m")
    if not os.environ.get("DISPLAY"):
        parser.error("DISPLAY is not set; this graphical demo needs a desktop session")

    config_path = ROOT / "config" / "metasejong-demo.yaml"
    log_dir = Path("/tmp/ontology_rgat_metasejong_demo")
    cfg = default_config("quick", "sitl")
    stack = ExternalStack(
        cfg, isaac_sim_path=args.isaac_sim_path, headless=False,
        config_path=config_path, log_dir=log_dir)
    rviz = None
    rviz_log = None
    link = None
    try:
        stack.start()
        rviz, rviz_log = _start_rviz(log_dir)
        print("RViz 2 is open.", flush=True)
        link = DemoLink(config_path)
        link.request("hello")
        _wait_until(
            link,
            lambda s: bool(s.get("estimator_valid")) and bool(
                (s.get("extra") or {}).get("pre_flight_checks_pass")),
            120.0, "PX4 estimator and preflight readiness")
        print("PX4 is ready; commanding takeoff above the moving UGV.", flush=True)

        link.goto_hover(args.height)
        time.sleep(1.0)  # establish the offboard setpoint stream before arming
        arm_deadline = time.monotonic() + 60.0
        while time.monotonic() < arm_deadline:
            state = link.state()
            if rviz.poll() is not None:
                raise RuntimeError(
                    f"RViz exited unexpectedly; see {log_dir / 'rviz.log'}")
            if state.get("armed"):
                break
            link.request("arm")
            time.sleep(2.0)
        else:
            raise TimeoutError("PX4 did not arm within 60 seconds")

        state = _wait_until(
            link,
            lambda s: bool(s.get("armed")) and not bool(s.get("landed"))
            and float((s.get("truth") or {}).get("position", [0, 0, 0])[2])
            >= 0.6 * args.height,
            120.0, "automatic takeoff")
        print(
            f"TAKEOFF COMPLETE: altitude={state['truth']['position'][2]:.2f} m, "
            f"UGV speed={state['pad']['speed']:.2f} m/s. Ctrl-C lands and stops.",
            flush=True)

        refreshed = time.monotonic()
        while True:
            state = link.state()
            now = time.monotonic()
            if now - refreshed >= 45.0:
                link.goto_hover(args.height)
                refreshed = now
            print(
                f"hover={state['truth']['position'][2]:5.2f} m  "
                f"ugv={state['pad']['speed']:4.2f} m/s  "
                f"marker={state['marker_quality']:.2f}  "
                f"estimator={'OK' if state['estimator_valid'] else 'BAD'}",
                flush=True)
            time.sleep(5.0)
    except KeyboardInterrupt:
        print("\nStopping demo: requesting a PX4 landing...", flush=True)
        if link is not None:
            try:
                link.request("disarm")  # airborne requests are converted to LAND
                _wait_until(link, lambda s: bool(s.get("landed")), 30.0,
                            "safety landing")
            except Exception as exc:  # noqa: BLE001
                print(f"Landing confirmation unavailable: {exc}", file=sys.stderr)
        return 0
    finally:
        if link is not None:
            link.close()
        if rviz is not None and rviz.poll() is None:
            rviz.terminate()
            try:
                rviz.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                rviz.kill()
        if rviz_log is not None:
            rviz_log.close()
        stack.stop()


if __name__ == "__main__":
    raise SystemExit(main())
