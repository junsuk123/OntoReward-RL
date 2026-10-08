#!/usr/bin/env python3
"""Fly the minimal-observation contract in Isaac/PX4 and record the ROS chain.

Control path (the one that scores): ``MinimalLandingEnv(backend=IsaacBackend)``.
Own state is PX4 EKF2 through the existing gateway, the pad solve is Isaac's
capture-stamped ArUco PnP, and the supervised acceleration reaches PX4 through
that gateway, which also owns entry, OFFBOARD handover and confirmed cleanup.

With ``--control ros`` the ROS chain IS the control path: each decision is
the chain supervisor's setpoint (stamped with the observation it judged),
forwarded through the same gateway; the in-process supervisor only observes.

Observation path (optional, ``--ros-chain``): the six ROS nodes of
``ontology_rgat_landing`` run alongside on the live topics -- camera image ->
``aruco_pad_detector``, ``/fmu`` -> ``px4_localization_bridge``, ... -> R-GAT ->
supervisor -- and ``minimal_chain_recorder`` logs every message. The chain's
setpoint is NOT sent to PX4 (nothing subscribes to it), so it cannot fight the
gateway; it shows that every topic flows on the real stack and lets the
chain's observation be compared with the in-process one.

The stack is started by ``run_spatial_pipeline.live_stack``: single-writer
lock, refusal to take over a running Isaac, gateway sync, owned shutdown.

    python tools/minimal_isaac_flight.py --out results/minimal_contract_20261007/isaac \
        --controllers teacher results/.../bc/ppo_ontology_rgat__seed828.pt --episodes 2 --ros-chain
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
ASCII_WS = Path(os.environ.get("ASCII_ROS2_WS",
                               Path.home() / ".local/share/ontology_rgat_uav_rl/ros2_ws"))


def ros_shell(install: Path, command: str) -> list[str]:
    script = (
        "set +u; source /opt/ros/humble/setup.bash; "
        f"[ -f '{ASCII_WS}/install/local_setup.bash' ] && source '{ASCII_WS}/install/local_setup.bash'; "
        f"source '{install}/local_setup.bash'; "
        "export RMW_IMPLEMENTATION=rmw_fastrtps_cpp; unset CYCLONEDDS_URI; "
        f"export ONTOLOGY_RGAT_ROOT='{ROOT}'; exec {command}")
    return ["bash", "-c", script]


def start_chain(install: Path, out: Path, checkpoint: str | None):
    logs = out / "ros_chain"
    logs.mkdir(parents=True, exist_ok=True)
    args = "use_sim_time:=true sim_clock_from_isaac:=true"
    if checkpoint:
        args += f" checkpoint:={checkpoint}"
    launch = subprocess.Popen(
        ros_shell(install, f"ros2 launch ontology_rgat_landing minimal_pipeline.launch.py {args}"),
        stdout=open(logs / "launch.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
    recorder = subprocess.Popen(
        ros_shell(install, f"python3 '{ROOT}/tools/minimal_chain_recorder.py' '{logs}/chain.jsonl'"),
        stdout=open(logs / "recorder.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
    return [launch, recorder]


def stop_chain(processes) -> None:
    for proc in reversed(processes):
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGINT)
    deadline = time.monotonic() + 15
    for proc in processes:
        try:
            proc.wait(max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)


class ChainLink:
    """The flight process's window onto the ROS chain (``--control ros``).

    Subscribes to the chain supervisor's setpoint and status (published with
    the same decision stamp) and calls the per-episode reset services. Needs
    a ROS environment with ``ontology_rgat_interfaces`` sourced.
    """
    NODES = ("observation_assembler", "ontology_node", "safety_supervisor_node")

    def __init__(self, ns: str = "/landing_uav0"):
        import threading
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from std_srvs.srv import Trigger
        from ontology_rgat_interfaces.msg import SafetyStatus
        rclpy.init()
        self.rclpy = rclpy
        self.node = rclpy.create_node("minimal_flight_link")
        qos = QoSProfile(depth=20, history=HistoryPolicy.KEEP_LAST,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.lock = threading.Lock()
        self.decisions = {}
        self.node.create_subscription(SafetyStatus, f"{ns}/safety/status", self._on_status, qos)
        self.clients = [self.node.create_client(Trigger, f"{ns}/minimal/reset/{name}")
                        for name in self.NODES]
        self.Trigger = Trigger
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()

    def _on_status(self, msg):
        stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        with self.lock:
            self.decisions[stamp] = {
                "stamp": stamp, "mode": msg.mode, "pad_class": msg.pad_loss_class,
                "reason": msg.pad_lost_reason,
                "requested": [msg.requested_acceleration.x, msg.requested_acceleration.y,
                              msg.requested_acceleration.z],
                "applied": [msg.applied_acceleration.x, msg.applied_acceleration.y,
                            msg.applied_acceleration.z],
                "intervened": list(msg.intervened)}
            for old in sorted(self.decisions)[:-50]:
                del self.decisions[old]

    def reset(self, timeout_s: float = 10.0) -> None:
        deadline = time.monotonic() + timeout_s
        for client in self.clients:
            if not client.wait_for_service(timeout_sec=max(deadline - time.monotonic(), 0.1)):
                raise RuntimeError(f"chain reset service {client.srv_name} unavailable")
            future = client.call_async(self.Trigger.Request())
            while not future.done():
                if time.monotonic() > deadline:
                    raise RuntimeError(f"chain reset {client.srv_name} timed out")
                time.sleep(0.01)
        with self.lock:
            self.decisions.clear()

    def decision(self, after: float, until: float, wall_timeout_s: float):
        """The chain's decision stamped in (after, until], waiting up to the timeout."""
        deadline = time.monotonic() + wall_timeout_s
        while True:
            with self.lock:
                hits = [d for t, d in self.decisions.items() if after < t <= until + 1e-6]
            if hits:
                return max(hits, key=lambda d: d["stamp"])
            if time.monotonic() > deadline:
                return None
            time.sleep(0.005)

    def close(self) -> None:
        self.executor.shutdown()
        self.node.destroy_node()
        if self.rclpy.ok():
            self.rclpy.shutdown()


def make_controller(spec: str):
    from ontology_rgat.minimal.rollout import ArmController
    from ontology_rgat.minimal.teacher import MinimalTeacher, TeacherGains
    if spec == "teacher" or spec.startswith("teacher:"):
        gains = {}
        if ":" in spec:
            gains = json.loads(Path(spec.split(":", 1)[1]).read_text())["best"]["gains"]
        return "teacher", lambda: MinimalTeacher(TeacherGains(**gains))
    from ontology_rgat.minimal.arms import load_arm
    name, arm, _ = load_arm(spec)
    arm.eval()
    return f"{name}:{Path(spec).stem}", lambda: ArmController(arm)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--controllers", nargs="+", default=["teacher"])
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=12000)
    parser.add_argument("--ros-chain", action="store_true")
    parser.add_argument("--ros-install", type=Path, default=None,
                        help="colcon install base holding ontology_rgat_interfaces/_landing")
    parser.add_argument("--chain-checkpoint", default=None,
                        help="ppo_ontology_rgat checkpoint for the chain's R-GAT node")
    parser.add_argument("--control", choices=("inprocess", "ros"), default="inprocess",
                        help="ros: fly the ROS chain's supervised setpoint (needs --ros-chain "
                             "and a sourced ROS environment); inprocess: the Python controllers")
    parser.add_argument("--chain-timeout-s", type=float, default=1.0,
                        help="wall seconds to wait for the chain's decision each step")
    parser.add_argument("--reset-recoveries", type=int, default=2)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.ros_chain and args.ros_install is None:
        parser.error("--ros-chain needs --ros-install")
    if args.control == "ros" and not (args.ros_chain and args.chain_checkpoint):
        parser.error("--control ros needs --ros-chain and --chain-checkpoint")

    sys.path.insert(0, str(ROOT / "python"))
    from run_spatial_pipeline import live_stack
    from ontology_rgat.spatial.core import SpatialConfig
    from ontology_rgat.spatial.environment import IsaacBackend
    from ontology_rgat.minimal.local_env import MinimalLandingEnv
    from ontology_rgat.minimal.pad_loss import CLASS_NAMES, reason_names
    from ontology_rgat.minimal.supervisor import MODE_NAMES

    cfg = SpatialConfig()
    controllers = ([make_controller(spec) for spec in args.controllers]
                   if args.control == "inprocess"
                   else [(f"ros_chain:{Path(args.chain_checkpoint).stem}", None)])
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    results, chain = [], []
    with live_stack(args.out, headless=args.headless, schema=cfg.schema,
                    reset_recoveries=args.reset_recoveries):
        try:
            link = None
            if args.ros_chain:
                chain = start_chain(args.ros_install, args.out, args.chain_checkpoint)
            if args.control == "ros":
                link = ChainLink()
            env = MinimalLandingEnv(cfg, backend=IsaacBackend(cfg))
            for label, factory in controllers:
                for seed in seeds:
                    trace = (args.out / f"trace_{label.replace(':', '_')}_{seed}.jsonl").open("w")
                    obs, _ = env.reset(seed=seed)
                    controller = factory() if factory else None
                    if link is not None:
                        link.reset()
                    done, info, steps, fallbacks = False, {}, 0, 0
                    classes, modes = Counter(), Counter()
                    chain_classes = Counter()
                    while not done:
                        if link is None:
                            action, external = controller.act(obs), None
                        else:
                            # The chain decides on its own 10 Hz sim timer; take
                            # its decision for the step that ends at this
                            # observation. Missing it: zero request through the
                            # in-process supervisor, counted.
                            external = link.decision(obs.stamp - 0.1, obs.stamp,
                                                     args.chain_timeout_s)
                            if external is None:
                                fallbacks += 1
                                action = np.zeros(3)
                            else:
                                action = np.asarray(external["requested"])
                                chain_classes[CLASS_NAMES[external["pad_class"]]] += 1
                        obs, reward, done, info = env.step(action, external_decision=external)
                        steps += 1
                        classes[CLASS_NAMES[info["pad_class"]]] += 1
                        modes[MODE_NAMES[info["mode"]]] += 1
                        trace.write(json.dumps({
                            "t": info["elapsed_s"], "obs": obs.vector().tolist(),
                            "class": CLASS_NAMES[info["pad_class"]],
                            "reasons": reason_names(info["pad_reason"]),
                            "mode": MODE_NAMES[info["mode"]],
                            "requested": info["requested"], "applied": info["applied"],
                            "truth_pad_minus_body": info["truth_pad_minus_body"],
                            "status": info["status"]}) + "\n")
                    trace.close()
                    row = {"controller": label, "seed": seed, "status": info["status"],
                           "elapsed_s": info["elapsed_s"], "steps": steps,
                           "classes": dict(classes), "modes": dict(modes),
                           "chain_fallbacks": fallbacks if link else None,
                           "chain_classes": dict(chain_classes) if link else None,
                           "final_truth_pad_minus_body": info["truth_pad_minus_body"]}
                    results.append(row)
                    print(json.dumps(row), flush=True)
                    (args.out / "flights.json").write_text(json.dumps(results, indent=1))
            env.backend.close()
        finally:
            if link is not None:
                link.close()
            stop_chain(chain)
    summary = {}
    for label, _ in controllers:
        rows = [r for r in results if r["controller"] == label]
        summary[label] = dict(Counter(r["status"] for r in rows))
    (args.out / "summary.json").write_text(json.dumps(
        {"contract": "minimal-landing-obs/2 + minimal-landing-ontology/6 on Isaac/PX4",
         "seeds": seeds, "by_controller": summary, "flights": results}, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
