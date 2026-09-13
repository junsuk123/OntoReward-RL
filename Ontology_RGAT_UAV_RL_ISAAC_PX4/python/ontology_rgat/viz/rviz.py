"""Real-time episode visualization through RViz 2.

This replaces the retired MATLAB ``viz.RealtimeMonitor`` figure. RViz is the
better tool for the job here and not merely a substitute: it already has the
simulator's own TF tree, the deck, the marker board and the camera, so the
learner only has to add what it alone knows -- where the policy thinks it is,
where it has been, and which ontology relations the potential is attending to.

Topics published under ``cfg.viz.rviz.namespace`` (default ``/landing_rl``):

===========================  ===================================================
``/uav_path``                ``nav_msgs/Path``, the trail in the **pad** frame
``/pad_path``                ``nav_msgs/Path``, the deck track relative to its
                             current pose
``/scene``                   ``visualization_msgs/MarkerArray``: deck, success
                             cylinder, wind/aero arrows, HUD and a persistent
                             colour-coded landing-outcome banner
``/ontology``                ``visualization_msgs/MarkerArray``: the 14 ontology
                             nodes coloured by activation and the 34 relation
                             edges scaled by the potential's attention
``/telemetry``               ``std_msgs/String``, the same JSON the dashboard
                             gets, for anything else that wants to subscribe
===========================  ===================================================

TF: ``map -> landing_pad -> uav_body``. The pad-relative trail lives under
``landing_pad``, so it stays put while the deck drives, which is what makes a
moving-target landing legible at all.

Everything degrades to a no-op if ``rclpy`` is missing or a node cannot be
created, because a training run must not fail for want of a viewer.
"""
from __future__ import annotations

import json
import math
from typing import Any, Sequence

import numpy as np

__all__ = ["RvizPublisher", "RvizPublisherGroup"]

# relation id -> RGB. Matches the ontology relation order:
# 0 degrades, 1 supports, 2 contributes, 3 self.
RELATION_COLORS = ((0.85, 0.25, 0.20), (0.15, 0.60, 0.35),
                   (0.20, 0.45, 0.80), (0.55, 0.55, 0.58))

# Column and row of each node in the graph overlay, in schema order. The layout
# follows the edge structure: raw channels on the left, the derived stability
# terms next, touchdown safety, then the goal.
GRAPH_LAYOUT = (
    (0, 3.5), (0, 2.5), (0, 1.5), (0, 0.5), (0, -0.5), (0, -1.5),
    (1, -1.5), (1, 1.0), (1, -0.2), (2, 0.4), (0, -2.5), (0, -3.5), (0, -4.5),
    (3, 0.4),
)


def _rgba(marker, rgb: Sequence[float], alpha: float = 1.0) -> None:
    marker.color.r, marker.color.g, marker.color.b = (float(c) for c in rgb)
    marker.color.a = float(alpha)


def _risk_color(value: float) -> tuple[float, float, float]:
    """Green at 0, red at 1: more of this node is worse."""
    v = float(np.clip(value, 0.0, 1.0))
    return (0.20 + 0.72 * v, 0.70 - 0.50 * v, 0.28 - 0.12 * v)


def _support_color(value: float) -> tuple[float, float, float]:
    """Red at 0, green at 1: more of this node is better."""
    return _risk_color(1.0 - value)


def _outcome_style(status: str):
    """Return (headline, colour) for a terminal episode status.

    ``None`` means the episode is still running.  An unconfirmed success is
    deliberately amber rather than green: the geometric criterion may have
    fired, but PX4 did not confirm that the vehicle reached a settled state.
    """
    status = str(status)
    if status == "running":
        return None
    if status == "success":
        return "LANDING SUCCESS", (0.10, 0.95, 0.28)
    if status.startswith("unconfirmed_"):
        return "RESULT UNCONFIRMED", (1.00, 0.62, 0.08)
    return "LANDING FAILED", (0.96, 0.12, 0.10)


class RvizPublisher:
    """Publishes one episode's live state for RViz 2.

    Construct with :meth:`create`, which returns ``None`` when ROS 2 is not
    importable so callers can treat "no viewer" as an ordinary condition.
    """

    def __init__(self, cfg, node, modules: dict[str, Any]):
        self.cfg = cfg
        self.opt = cfg.viz.rviz
        self.node = node
        self.m = modules
        ns = str(self.opt.namespace).rstrip("/")
        # RViz is an operator display, not an experiment data path. Keep only
        # the newest frame so a busy GPU/UI cannot accumulate seconds of old
        # paths and then reject them after the TF cache has moved on.
        qos = modules.get("viz_qos", 10)
        self.uav_path_pub = node.create_publisher(modules["Path"], ns + "/uav_path", qos)
        self.pad_path_pub = node.create_publisher(modules["Path"], ns + "/pad_path", qos)
        self.scene_pub = node.create_publisher(modules["MarkerArray"], ns + "/scene", qos)
        self.graph_pub = node.create_publisher(modules["MarkerArray"], ns + "/ontology", qos)
        # The simulator-side ROS node also observes telemetry and requests a
        # reliable endpoint. Keep this small JSON stream reliable; only the
        # replaceable visual frames above use best-effort/depth-one QoS.
        self.telemetry_pub = node.create_publisher(
            modules["String"], ns + "/telemetry", 10)
        self.tf = modules["TransformBroadcaster"](node)
        self.potential = None
        self._uav_trail: list[tuple[float, float, float]] = []
        self._pad_trail: list[tuple[float, float, float]] = []
        self._graph_every = 5
        self._counter = 0

    # ------------------------------------------------------------ lifecycle
    @classmethod
    def create(cls, cfg, potential=None, node_name: str = "ontology_rgat_viz",
               pair_methods: Sequence[str] | None = None):
        """Return a publisher, or ``None`` if ROS 2 is unavailable or refused."""
        methods = tuple(str(method) for method in (pair_methods or ()))
        if len(methods) > 1:
            publishers = {}
            for index, method in enumerate(methods):
                pair_cfg = cfg.derive(**{
                    "viz.rviz.namespace": f"{str(cfg.viz.rviz.namespace).rstrip('/')}/pair_{index}",
                    "viz.rviz.pad_frame": f"landing_pad_{index}",
                    "viz.rviz.body_frame": f"uav_body_{index}",
                })
                publisher = cls.create(
                    pair_cfg, potential=potential,
                    node_name=f"{node_name}_pair_{index}")
                if publisher is None:
                    for created in publishers.values():
                        created.close()
                    return None
                publishers[method] = publisher
            print(
                f"RViz 2 parallel view configured for {len(publishers)} pairs "
                f"under {cfg.viz.rviz.namespace}/pair_N.")
            return RvizPublisherGroup(publishers)
        if not cfg.viz.rviz.enabled:
            return None
        try:
            import rclpy
            from rclpy.node import Node
            from geometry_msgs.msg import Point, TransformStamped
            from nav_msgs.msg import Path
            from std_msgs.msg import String
            from visualization_msgs.msg import Marker, MarkerArray
            from tf2_ros import TransformBroadcaster
            from geometry_msgs.msg import PoseStamped
            from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                                   ReliabilityPolicy)
        except Exception as exc:                       # pragma: no cover - env dependent
            print(f"RViz 2 publishing disabled: ROS 2 is not importable ({exc}).")
            return None
        try:
            if not rclpy.ok():
                rclpy.init(args=None)
            node = Node(node_name)
        except Exception as exc:                       # pragma: no cover - env dependent
            print(f"RViz 2 publishing disabled: cannot create a node ({exc}).")
            return None
        viz_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE)
        publisher = cls(cfg, node, {
            "rclpy": rclpy, "Point": Point, "TransformStamped": TransformStamped,
            "Path": Path, "PoseStamped": PoseStamped, "String": String,
            "Marker": Marker, "MarkerArray": MarkerArray,
            "TransformBroadcaster": TransformBroadcaster, "viz_qos": viz_qos})
        publisher.potential = potential
        print(f"RViz 2 topics live under {cfg.viz.rviz.namespace}.")
        return publisher

    def close(self) -> None:
        try:
            self.node.destroy_node()
        except Exception:                              # pragma: no cover
            pass

    def clear_trails(self, method: str | None = None) -> None:
        self._uav_trail.clear()
        self._pad_trail.clear()
        self._delete_all(self.scene_pub)
        self._delete_all(self.graph_pub)

    # --------------------------------------------------------------- helpers
    def _stamp(self):
        return self.node.get_clock().now().to_msg()

    def _latest_stamp(self):
        """A zero ROS stamp asks RViz for the newest available transform.

        Paths are deliberately expressed relative to the *current* platform,
        not reconstructed as historical world poses. Exact wall timestamps
        only make a busy campus viewport reject replaceable visual messages
        after its ten-second TF cache advances. TF itself remains timestamped.
        """
        stamp = self._stamp()
        if hasattr(stamp, "sec") and hasattr(stamp, "nanosec"):
            stamp.sec = 0
            stamp.nanosec = 0
        return stamp

    def _delete_all(self, publisher) -> None:
        array = self.m["MarkerArray"]()
        marker = self.m["Marker"]()
        marker.action = self.m["Marker"].DELETEALL
        array.markers.append(marker)
        publisher.publish(array)

    def _marker(self, ns: str, mid: int, kind: int, frame: str):
        marker = self.m["Marker"]()
        marker.header.frame_id = frame
        marker.header.stamp = self._latest_stamp()
        marker.ns = ns
        marker.id = int(mid)
        marker.type = kind
        marker.action = self.m["Marker"].ADD
        marker.pose.orientation.w = 1.0
        return marker

    def _point(self, x: float, y: float, z: float):
        p = self.m["Point"]()
        p.x, p.y, p.z = float(x), float(y), float(z)
        return p

    # ------------------------------------------------------------------- TF
    def _broadcast_tf(self, pad_pos: np.ndarray, pad_yaw: float,
                      uav_pos: np.ndarray, uav_quat: np.ndarray) -> None:
        stamp = self._stamp()
        world, pad_frame, body = (self.opt.world_frame, self.opt.pad_frame,
                                  self.opt.body_frame)
        pad_tf = self.m["TransformStamped"]()
        pad_tf.header.stamp = stamp
        pad_tf.header.frame_id = world
        pad_tf.child_frame_id = pad_frame
        pad_tf.transform.translation.x = float(pad_pos[0])
        pad_tf.transform.translation.y = float(pad_pos[1])
        pad_tf.transform.translation.z = float(pad_pos[2])
        pad_tf.transform.rotation.z = math.sin(0.5 * float(pad_yaw))
        pad_tf.transform.rotation.w = math.cos(0.5 * float(pad_yaw))

        body_tf = self.m["TransformStamped"]()
        body_tf.header.stamp = stamp
        body_tf.header.frame_id = pad_frame
        body_tf.child_frame_id = body
        body_tf.transform.translation.x = float(uav_pos[0])
        body_tf.transform.translation.y = float(uav_pos[1])
        body_tf.transform.translation.z = float(uav_pos[2])
        w, x, y, z = (float(v) for v in uav_quat)
        body_tf.transform.rotation.w = w
        body_tf.transform.rotation.x = x
        body_tf.transform.rotation.y = y
        body_tf.transform.rotation.z = z
        self.tf.sendTransform([pad_tf, body_tf])

    def _publish_path(self, publisher, frame: str,
                      trail: Sequence[tuple[float, float, float]]) -> None:
        path = self.m["Path"]()
        path.header.frame_id = frame
        path.header.stamp = self._latest_stamp()
        for x, y, z in trail:
            pose = self.m["PoseStamped"]()
            pose.header = path.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = float(z)
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        publisher.publish(path)

    # ---------------------------------------------------------------- scene
    def _publish_scene(self, log, cur, info: dict[str, Any]) -> None:
        Marker = self.m["Marker"]
        pad_frame = self.opt.pad_frame
        array = self.m["MarkerArray"]()

        # The deck the vehicle is landing on, drawn where it actually is.
        deck = self._marker("scene", 0, Marker.CUBE, pad_frame)
        deck.scale.x, deck.scale.y, deck.scale.z = 1.30, 0.90, 0.04
        deck.pose.position.z = -0.02
        _rgba(deck, (0.20, 0.22, 0.26), 0.85)
        array.markers.append(deck)

        # The horizontal success tolerance, so "close enough" is visible rather
        # than something the reader has to take on trust from a metrics table.
        target = self._marker("scene", 1, Marker.CYLINDER, pad_frame)
        radius = 2.0 * float(self.cfg.criteria.xy)
        target.scale.x = target.scale.y = radius
        target.scale.z = 0.01
        outcome = _outcome_style(info["status"])
        target_rgb = outcome[1] if outcome is not None else (0.15, 0.75, 0.40)
        _rgba(target, target_rgb, 0.48 if outcome is not None else 0.35)
        array.markers.append(target)

        # Where the vehicle is now, and how far it still has to fall.
        pos = np.asarray(log.x[-1][0:3], dtype=float)
        drop = self._marker("scene", 2, Marker.LINE_LIST, pad_frame)
        drop.scale.x = 0.02
        drop.points = [self._point(*pos), self._point(pos[0], pos[1], 0.0)]
        _rgba(drop, (0.55, 0.60, 0.70), 0.7)
        array.markers.append(drop)

        # Wind and the aerodynamic resultant Isaac actually applied. Both are
        # scaled to be readable, so they show direction and relative size, not
        # an absolute length in metres.
        for index, (vector, rgb, gain) in enumerate((
                (np.asarray(log.wind[-1]), (0.30, 0.55, 0.90), 0.35),
                (np.asarray(log.aero_force[-1]), (0.90, 0.55, 0.15), 0.60))):
            arrow = self._marker("scene", 3 + index, Marker.ARROW, pad_frame)
            arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.035, 0.075, 0.10
            arrow.points = [self._point(*pos), self._point(*(pos + gain * vector))]
            _rgba(arrow, rgb, 0.9)
            array.markers.append(arrow)

        text = self._marker("scene", 5, Marker.TEXT_VIEW_FACING, pad_frame)
        text.pose.position.x, text.pose.position.y = pos[0], pos[1]
        text.pose.position.z = float(pos[2]) + 0.55
        text.scale.z = 0.16
        _rgba(text, (0.95, 0.95, 0.95), 0.95)
        phi = log.phi[-1]
        wind_risk = float(getattr(getattr(cur, "sem", None), "wind_risk", 0.0))
        text.text = (
            f"t={log.t[-1]:5.2f}s  {info['status']}\n"
            f"z={pos[2]:5.2f}m  xy={float(np.linalg.norm(pos[:2])):4.2f}m\n"
            f"deck={log.pad_speed[-1]:4.2f}  closing={log.closing_speed[-1]:4.2f} m/s\n"
            f"wind={np.linalg.norm(log.wind[-1]):4.2f} m/s  risk={wind_risk:4.2f}\n"
            f"marker={log.marker_quality[-1]:4.2f}  gnss={log.gnss_quality[-1]:4.2f}\n"
            f"hover left={log.hover_seconds_left[-1]:5.1f}s"
            + (f"\nPhi={phi:+.3f}" if np.isfinite(phi) else ""))
        array.markers.append(text)

        if outcome is not None:
            headline, colour = outcome
            truth = (info.get("diag", {}).get("ground_truth") or {})
            touchdown = np.asarray(
                truth.get("position", cur.meas.get("pos", pos)), dtype=float)
            if touchdown.shape != (3,) or not np.isfinite(touchdown).all():
                touchdown = np.asarray(cur.meas.get("pos", pos), dtype=float)
            xy_error = float(np.linalg.norm(touchdown[:2]))

            # A large fixed banner remains after the terminal sample, so the
            # result stays visible until clear_trails() starts the next episode.
            banner = self._marker("landing_outcome", 0, Marker.TEXT_VIEW_FACING,
                                  pad_frame)
            banner.pose.position.x = 0.0
            banner.pose.position.y = 0.0
            banner.pose.position.z = 2.8
            banner.scale.z = 0.46
            _rgba(banner, colour, 1.0)
            banner.text = (
                f"{headline}\n"
                f"{str(info['status']).replace('_', ' ').upper()}\n"
                f"TOUCHDOWN XY ERROR: {xy_error:.2f} m")
            array.markers.append(banner)

            # Mark the scored touchdown location and draw its miss distance to
            # the pad centre. The result is scored on simulator truth in SITL;
            # truth remains visualization/evaluation-only and never feeds the
            # controller, ontology or reward input.
            hit = self._marker("landing_outcome", 1, Marker.SPHERE, pad_frame)
            hit.pose.position.x = float(touchdown[0])
            hit.pose.position.y = float(touchdown[1])
            hit.pose.position.z = max(float(touchdown[2]), 0.12)
            hit.scale.x = hit.scale.y = hit.scale.z = 0.28
            _rgba(hit, colour, 1.0)
            array.markers.append(hit)

            miss = self._marker("landing_outcome", 2, Marker.LINE_LIST, pad_frame)
            miss.scale.x = 0.055
            miss.points = [self._point(0.0, 0.0, 0.10),
                           self._point(float(touchdown[0]), float(touchdown[1]), 0.10)]
            _rgba(miss, colour, 0.95)
            array.markers.append(miss)
        self.scene_pub.publish(array)

    # ------------------------------------------------------------- ontology
    def _publish_ontology(self, cur) -> None:
        """Draw the ontology with the potential's own attention on the edges.

        Attention is learned importance, not causal proof; the overlay is an
        interpretability aid and is labelled as one in the docs.
        """
        Marker = self.m["Marker"]
        graph = cur.graph
        values = cur.sem.node_values
        origin = np.asarray(self.opt.graph_origin_pad_m, dtype=float)
        scale = float(self.opt.graph_scale_m)
        positions = [origin + scale * np.array([2.0 * col, 0.0, row])
                     for col, row in GRAPH_LAYOUT]

        alpha = None
        if self.potential is not None:
            try:
                alpha = self.potential.explain(graph)["edge_alpha"]
            except Exception:                          # pragma: no cover - defensive
                alpha = None

        array = self.m["MarkerArray"]()
        from ..semantic import RISK_NODES, GOAL_NODE
        for i, (name, position) in enumerate(zip(graph.node_names, positions)):
            node = self._marker("ontology_node", i, Marker.SPHERE, self.opt.pad_frame)
            node.pose.position.x, node.pose.position.y, node.pose.position.z = (
                float(position[0]), float(position[1]), float(position[2]))
            size = 0.09 + 0.10 * float(np.clip(values[i], 0.0, 1.0))
            node.scale.x = node.scale.y = node.scale.z = size
            if i == GOAL_NODE:
                _rgba(node, (0.95, 0.78, 0.20), 0.95)
                node.scale.x = node.scale.y = node.scale.z = 0.20
            elif i in RISK_NODES:
                _rgba(node, _risk_color(values[i]), 0.95)
            else:
                _rgba(node, _support_color(values[i]), 0.95)
            array.markers.append(node)

            label = self._marker("ontology_label", i, Marker.TEXT_VIEW_FACING,
                                 self.opt.pad_frame)
            label.pose.position.x = float(position[0])
            label.pose.position.y = float(position[1])
            label.pose.position.z = float(position[2]) + 0.16
            label.scale.z = 0.085
            _rgba(label, (0.92, 0.92, 0.92), 0.9)
            label.text = (name if i == GOAL_NODE else f"{name} {values[i]:.2f}")
            array.markers.append(label)

        peak = float(np.max(alpha)) if alpha is not None and alpha.size else 1.0
        for e, (s, d, r) in enumerate(zip(graph.src, graph.dst, graph.rel)):
            if s == d:
                continue                                # the self-loops would be dots
            edge = self._marker("ontology_edge", e, Marker.ARROW, self.opt.pad_frame)
            a = float(alpha[e]) / max(peak, 1e-9) if alpha is not None else 0.35
            edge.scale.x = 0.006 + 0.030 * a
            edge.scale.y = 0.020 + 0.050 * a
            edge.scale.z = 0.05
            start, end = positions[int(s)], positions[int(d)]
            direction = end - start
            norm = float(np.linalg.norm(direction)) or 1.0
            trim = 0.11 * direction / norm
            edge.points = [self._point(*(start + trim)), self._point(*(end - trim))]
            _rgba(edge, RELATION_COLORS[int(r) % len(RELATION_COLORS)], 0.35 + 0.6 * a)
            array.markers.append(edge)
        self.graph_pub.publish(array)

    # ------------------------------------------------------------ per step
    def publish_step(self, log, cur, info: dict[str, Any]) -> None:
        """Publish one control step. Cheap enough for the 50 Hz loop."""
        diag = info["diag"]
        pad_pos = np.asarray(diag["pad_position_i"], dtype=float)
        pad_yaw = float(diag["pad"]["yaw"])
        x = np.asarray(log.x[-1], dtype=float)
        self._broadcast_tf(pad_pos, pad_yaw, x[0:3], x[6:10])

        limit = int(self.opt.trail_length)
        self._uav_trail.append(tuple(float(v) for v in x[0:3]))
        self._pad_trail.append(tuple(float(v) for v in pad_pos))
        del self._uav_trail[:-limit]
        del self._pad_trail[:-limit]
        self._publish_path(self.uav_path_pub, self.opt.pad_frame, self._uav_trail)
        # RViz's fixed frame is the moving landing_pad frame. Expressing both
        # the driven history and the surveyed road relative to the current pad
        # avoids asking its message filter for a transform at an already-past
        # wall timestamp, which caused the map-frame queue to overflow on the
        # heavy campus scene.
        relative_pad_trail = [tuple(np.asarray(point) - pad_pos)
                              for point in self._pad_trail]
        self._publish_path(
            self.pad_path_pub, self.opt.pad_frame, relative_pad_trail)
        self._publish_scene(log, cur, info)

        self._counter += 1
        if self.opt.publish_ontology_graph and self._counter % self._graph_every == 0:
            self._publish_ontology(cur)

        message = self.m["String"]()
        message.data = json.dumps({
            "t": float(log.t[-1]), "status": info["status"],
            "position_pad": [float(v) for v in x[0:3]],
            "velocity_pad": [float(v) for v in x[3:6]],
            "pad_position": [float(v) for v in pad_pos],
            "pad_speed": float(log.pad_speed[-1]),
            "closing_speed": float(log.closing_speed[-1]),
            "wind_enu": [float(v) for v in log.wind[-1]],
            "wind_speed": float(np.linalg.norm(log.wind[-1])),
            "wind_risk": float(cur.sem.wind_risk),
            "marker_quality": float(log.marker_quality[-1]),
            "battery_reserve": float(log.battery_reserve[-1]),
            "hover_seconds_left": float(log.hover_seconds_left[-1]),
            "reward": float(log.r[-1]),
            "phi": (float(log.phi[-1]) if np.isfinite(log.phi[-1]) else None),
            "node_values": [float(v) for v in cur.sem.node_values],
        }, allow_nan=False)
        self.telemetry_pub.publish(message)

    def publish_benchmark_step(self, *, state: dict[str, Any], method: str,
                               scenario: str, step: int, dt: float,
                               in_fov: bool, status: str) -> None:
        """Publish the recurrent Shin benchmark without its legacy log type.

        The controlled benchmark has a stricter actor boundary and therefore
        does not construct the ontology pipeline's ``EpisodeLog`` object.
        RViz still needs the physical flight, paths and operator status, all of
        which are already present in the gateway state returned after a step.
        """
        publish_rate = max(float(getattr(self.opt, "publish_rate_hz", 10.0)), 0.1)
        period = max(1, int(round(1.0 / max(float(dt) * publish_rate, 1e-9))))
        if int(step) % period and str(status) == "running":
            return
        pad = state.get("pad") if isinstance(state.get("pad"), dict) else {}
        truth = (state.get("truth")
                 if isinstance(state.get("truth"), dict) else {})
        pad_pos = np.asarray(pad.get("position", (0.0, 0.0, 0.0)), dtype=float)
        relative = np.asarray(
            truth.get("position") if truth.get("valid", False)
            else state.get("position", (0.0, 0.0, 0.0)), dtype=float)
        quaternion = np.asarray(state.get("quaternion_wxyz", (1.0, 0.0, 0.0, 0.0)),
                                dtype=float)
        if (pad_pos.shape != (3,) or relative.shape != (3,)
                or quaternion.shape != (4,) or not np.isfinite(relative).all()
                or not np.isfinite(pad_pos).all() or not np.isfinite(quaternion).all()):
            return

        # Gateway relative positions use gravity-aligned ENU axes translated
        # to the platform origin, so this display frame is translated but not
        # yaw-rotated. The platform's actual heading remains visible in its
        # Isaac odometry display.
        self._broadcast_tf(pad_pos, 0.0, relative, quaternion)
        limit = int(self.opt.trail_length)
        self._uav_trail.append(tuple(float(value) for value in relative))
        self._pad_trail.append(tuple(float(value) for value in pad_pos))
        del self._uav_trail[:-limit]
        del self._pad_trail[:-limit]
        self._publish_path(self.uav_path_pub, self.opt.pad_frame, self._uav_trail)
        # Use the current deck as the origin for its history.  This avoids an
        # exact-time map transform for delayed photoreal frames and shows the
        # road already travelled directly behind the vehicle.
        pad_relative_trail = [tuple(np.asarray(point) - pad_pos)
                              for point in self._pad_trail]
        self._publish_path(
            self.pad_path_pub, self.opt.pad_frame, pad_relative_trail)

        Marker = self.m["Marker"]
        array = self.m["MarkerArray"]()
        deck = self._marker("scene", 0, Marker.CUBE, self.opt.pad_frame)
        deck_size = tuple(getattr(self.opt, "deck_size_m", (1.5, 1.5)))
        deck.scale.x, deck.scale.y, deck.scale.z = (
            float(deck_size[0]), float(deck_size[1]), 0.04)
        deck.pose.position.z = -0.02
        _rgba(deck, (0.10, 0.32, 0.48), 0.82)
        array.markers.append(deck)

        tolerance = self._marker("scene", 1, Marker.CYLINDER, self.opt.pad_frame)
        tolerance.scale.x = tolerance.scale.y = 2.0 * float(self.cfg.criteria.xy)
        tolerance.scale.z = 0.012
        _rgba(tolerance, (0.18, 0.78, 0.42), 0.42)
        array.markers.append(tolerance)

        uav = self._marker("scene", 2, Marker.SPHERE, self.opt.pad_frame)
        uav.pose.position.x, uav.pose.position.y, uav.pose.position.z = (
            float(value) for value in relative)
        uav.scale.x, uav.scale.y, uav.scale.z = 0.34, 0.34, 0.14
        _rgba(uav, (0.16, 0.48, 0.95), 0.96)
        array.markers.append(uav)

        drop = self._marker("scene", 3, Marker.LINE_LIST, self.opt.pad_frame)
        drop.scale.x = 0.025
        drop.points = [self._point(*relative),
                       self._point(relative[0], relative[1], 0.0)]
        _rgba(drop, (0.64, 0.68, 0.75), 0.8)
        array.markers.append(drop)

        text = self._marker("scene", 4, Marker.TEXT_VIEW_FACING, self.opt.pad_frame)
        text.pose.position.x = float(relative[0])
        text.pose.position.y = float(relative[1])
        text.pose.position.z = float(relative[2]) + 0.55
        text.scale.z = 0.18
        colour = ((0.25, 0.95, 0.45) if status == "success"
                  else (0.98, 0.30, 0.22) if status == "failure"
                  else (0.94, 0.94, 0.94))
        _rgba(text, colour, 0.96)
        text.text = (f"{method} | {scenario}\n"
                     f"t={step * dt:5.1f}s  {status.upper()}\n"
                     f"relative xyz=({relative[0]:+.2f}, {relative[1]:+.2f}, "
                     f"{relative[2]:+.2f}) m  marker={'ON' if in_fov else 'LOST'}")
        array.markers.append(text)

        route_points = tuple(getattr(self.opt, "route_waypoints_enu_m", ()))
        if len(route_points) >= 2:
            route = self._marker("campus_road_route", 0, Marker.LINE_STRIP,
                                 self.opt.pad_frame)
            route.scale.x = 0.08
            route.points = [self._point(
                float(point[0]) - pad_pos[0],
                float(point[1]) - pad_pos[1],
                float(point[2]) + 0.08 - pad_pos[2])
                for point in route_points]
            _rgba(route, (0.95, 0.72, 0.12), 0.92)
            array.markers.append(route)
        self.scene_pub.publish(array)

        message = self.m["String"]()
        message.data = json.dumps({
            "method": str(method), "scenario": str(scenario),
            "step": int(step), "t": float(step * dt), "status": str(status),
            "position_pad": [float(value) for value in relative],
            "marker_visible": bool(in_fov),
        })
        self.telemetry_pub.publish(message)


class RvizPublisherGroup:
    """Route each concurrent method to its own RViz topics and TF frames.

    The three learners share one process and one Isaac stage, but visual state
    is not a shared control resource. Keeping an independent publisher and
    trail per method prevents one episode reset from erasing the other two and
    prevents three ``map -> landing_pad`` transforms from overwriting each
    other.
    """

    def __init__(self, publishers: dict[str, RvizPublisher]):
        if not publishers:
            raise ValueError("an RViz publisher group needs at least one pair")
        self.publishers = dict(publishers)

    @property
    def potential(self):
        return next(iter(self.publishers.values())).potential

    @potential.setter
    def potential(self, value) -> None:
        for publisher in self.publishers.values():
            publisher.potential = value

    def _for(self, method: str | None) -> RvizPublisher:
        if method in self.publishers:
            return self.publishers[str(method)]
        return next(iter(self.publishers.values()))

    def clear_trails(self, method: str | None = None) -> None:
        if method is None:
            for publisher in self.publishers.values():
                publisher.clear_trails()
            return
        self._for(method).clear_trails()

    def publish_benchmark_step(self, *, method: str, **kwargs) -> None:
        self._for(method).publish_benchmark_step(method=method, **kwargs)

    def publish_step(self, log, cur, info: dict[str, Any]) -> None:
        self._for(None).publish_step(log, cur, info)

    def close(self) -> None:
        for publisher in self.publishers.values():
            publisher.close()
