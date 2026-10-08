"""Shared helpers: repository import path, topic names, message conversion."""
from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np


def repo_root() -> Path:
    """The Ontology_RGAT checkout that holds ``python/ontology_rgat``.

    ``ONTOLOGY_RGAT_ROOT`` wins; otherwise walk up from this file, which works
    from the source tree and from ``ros2_ws/install`` alike because the
    workspace lives inside the repository.
    """
    env = os.environ.get("ONTOLOGY_RGAT_ROOT")
    if env:
        return Path(env).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "python" / "ontology_rgat").is_dir():
            return parent
    raise RuntimeError("cannot locate the repository; set ONTOLOGY_RGAT_ROOT")


def add_repo_paths() -> Path:
    root = repo_root()
    for sub in ("python", "isaac_sim"):
        path = str(root / sub)
        if path not in sys.path:
            sys.path.insert(0, path)
    return root


add_repo_paths()

from ontology_rgat.minimal.observation import LandingObservation  # noqa: E402


def topic(ns: str, relative: str) -> str:
    return f"{ns.rstrip('/')}/{relative.lstrip('/')}"


def stamp_to_float(stamp) -> float:
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


def float_to_stamp(value: float, stamp_type):
    stamp = stamp_type()
    sec = int(np.floor(value))
    stamp.sec, stamp.nanosec = sec, int(round((value - sec) * 1e9)) % 1_000_000_000
    return stamp


def observation_to_msg(obs: LandingObservation, msg_type, time_type, frame_id="map"):
    msg = msg_type()
    msg.header.stamp = float_to_stamp(obs.stamp, time_type)
    msg.header.frame_id = frame_id
    msg.schema_id = obs.schema_id
    msg.own_position.x, msg.own_position.y, msg.own_position.z = map(float, obs.own_position)
    msg.own_velocity.x, msg.own_velocity.y, msg.own_velocity.z = map(float, obs.own_velocity)
    msg.own_valid = bool(obs.own_valid)
    msg.own_age_s = float(obs.own_age_s)
    rel = msg.pad_relative_position
    rel.x, rel.y, rel.z = map(float, obs.pad_relative_position)
    msg.pad_detected = bool(obs.pad_detected)
    msg.pad_age_s = float(obs.pad_age_s)
    if obs.pad_capture_stamp is not None:
        msg.pad_capture_stamp = float_to_stamp(obs.pad_capture_stamp, time_type)
    msg.vector = [float(v) for v in obs.vector()]
    return msg


def msg_to_observation(msg) -> LandingObservation:
    capture = stamp_to_float(msg.pad_capture_stamp)
    return LandingObservation(
        stamp=stamp_to_float(msg.header.stamp),
        own_position=np.array([msg.own_position.x, msg.own_position.y, msg.own_position.z]),
        own_velocity=np.array([msg.own_velocity.x, msg.own_velocity.y, msg.own_velocity.z]),
        own_valid=bool(msg.own_valid), own_age_s=float(msg.own_age_s),
        pad_relative_position=np.array([msg.pad_relative_position.x,
                                        msg.pad_relative_position.y,
                                        msg.pad_relative_position.z]),
        pad_detected=bool(msg.pad_detected), pad_age_s=float(msg.pad_age_s),
        # A zero stamp means "never detected" on the wire.
        pad_capture_stamp=capture if capture > 0.0 else None,
        schema_id=msg.schema_id)


def add_reset_service(node, ns: str, callback) -> None:
    """``<ns>/minimal/reset/<node name>`` (std_srvs/Trigger): start of an episode.

    Called at policy handover. The assembler, ontology and supervisor carry
    per-episode state (held pad measurement, pad memory, TrackingBias
    integrals, handover timer); without a reset it leaks into the next
    episode.
    """
    from std_srvs.srv import Trigger

    def handle(_request, response):
        callback()
        response.success, response.message = True, f"{node.get_name()} reset"
        node.get_logger().info("episode reset")
        return response

    node.create_service(Trigger, topic(ns, f"minimal/reset/{node.get_name()}"), handle)


def reliable_latest_qos():
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    return QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                      reliability=ReliabilityPolicy.RELIABLE)


def run(node_factory) -> None:
    import rclpy
    rclpy.init()
    node = node_factory()
    from rclpy.executors import ExternalShutdownException
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # launch may deliver a second SIGINT while the node is being torn down
        try:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:
            pass
