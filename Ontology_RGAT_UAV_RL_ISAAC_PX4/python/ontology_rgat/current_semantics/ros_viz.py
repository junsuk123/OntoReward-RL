"""ROS 2 view of a current-semantics flight for RViz2 and rqt (tools/wmon_demo.py).

Published under ``/wmon_demo`` (frame ``world``, ENU):

=================================  ==========================================================
TF ``world -> uav``                own navigation pose (the estimate the policy sees under /4)
TF ``world -> pad_estimate``       the pad track (ABG) the 21D vector is built from
TF ``world -> pad_truth``          the true pad placed at ``uav - truth(own - pad)``, so
                                   pad_estimate - pad_truth IS the relative-state error
``/wmon_demo/scene``               MarkerArray: pads, UAV, trail, action arrow, HUD banner
``/wmon_demo/ontology``            MarkerArray: the eight ontology nodes coloured by evidence
                                   weight (bond/1: c_i; other arms: registered validity),
                                   the semantic edges, LandingContext
``/wmon_demo/rel_pos_error``       Float64 [m]  |relative position estimate - truth|
``/wmon_demo/rel_vel_error``       Float64 [m/s] horizontal relative-velocity error
``/wmon_demo/vision_age``          Float64 [s]
``/wmon_demo/nav_valid``           Float64 0/1  drone_navigationValid
``/wmon_demo/nav_age``             Float64 [s]  GNSS fix age (observation /4), 0 under /3
``/wmon_demo/action``              Vector3Stamped [m/s^2] policy acceleration (world ENU)
``/wmon_demo/wmon_confidence``     Vector3Stamped mean WMON particle confidence (p, v, a)
``/wmon_demo/evidence``            Float64MultiArray the eight node evidence weights
``/wmon_demo/telemetry``           String, one JSON object per decision
=================================  ==========================================================

Everything is a no-op when rclpy is missing: a flight never fails for want of a viewer.
"""
from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

NAMESPACE = "/wmon_demo"
FRAME = "world"
NODE_LAYOUT = {  # ontology panel, 2 m beside the scene origin (x, y in the panel plane)
    "ObservationQuality": (0.0, 3.0), "RelativeGeometry": (0.0, 2.0),
    "MotionState": (0.0, 1.0), "AttitudeState": (0.0, 0.0),
    "ApproachSituation": (1.5, 2.5), "ViewGeometry": (1.5, 1.5),
    "ContactCondition": (1.5, 0.5), "LandingContext": (3.0, 1.5)}
NODE_SHORT = {"ObservationQuality": "Q", "RelativeGeometry": "G", "MotionState": "M",
              "AttitudeState": "A", "ApproachSituation": "P", "ViewGeometry": "O",
              "ContactCondition": "C", "LandingContext": "L"}
OUTCOME_COLOUR = {"SUCCESS": (0.15, 0.75, 0.30), "UNSAFE_CONTACT": (0.90, 0.20, 0.15),
                  "MISSED_PAD_CONTACT": (0.95, 0.55, 0.10), "TASK_TIMEOUT": (0.60, 0.60, 0.60),
                  "SAFETY_ENVELOPE_VIOLATION": (0.80, 0.10, 0.50)}


def _ramp(value: float) -> tuple[float, float, float]:
    """Red (0) -> green (1)."""
    v = float(np.clip(value, 0.0, 1.0))
    return (0.90 - 0.75 * v, 0.20 + 0.60 * v, 0.20)


class DemoPublisher:
    """Publishes one decision at a time; ``create`` returns a disabled stub without ROS."""

    def __init__(self, node, modules: Mapping[str, Any]):
        self.node, self.m = node, modules
        pub = lambda kind, name: node.create_publisher(modules[kind], f"{NAMESPACE}/{name}", 10)
        self.scene = pub("MarkerArray", "scene")
        self.ontology = pub("MarkerArray", "ontology")
        self.scalars = {name: pub("Float64", name) for name in
                        ("rel_pos_error", "rel_vel_error", "vision_age", "nav_valid", "nav_age")}
        self.action = pub("Vector3Stamped", "action")
        self.confidence = pub("Vector3Stamped", "wmon_confidence")
        self.evidence = pub("Float64MultiArray", "evidence")
        self.telemetry = pub("String", "telemetry")
        self.tf = modules["TransformBroadcaster"](node)
        self.trail: list[tuple[float, float, float]] = []
        self.enabled = True

    @classmethod
    def create(cls, node_name: str = "wmon_demo"):
        try:
            import rclpy
            from geometry_msgs.msg import TransformStamped, Vector3Stamped
            from std_msgs.msg import Float64, Float64MultiArray, String
            from tf2_ros import TransformBroadcaster
            from visualization_msgs.msg import Marker, MarkerArray
        except ImportError as exc:
            print(f"wmon-demo: ROS 2 not available ({exc}); flying without a viewer", flush=True)
            return _Disabled()
        if not rclpy.ok():
            rclpy.init()
        node = rclpy.create_node(node_name)
        return cls(node, {"MarkerArray": MarkerArray, "Marker": Marker, "Float64": Float64,
                          "Float64MultiArray": Float64MultiArray, "String": String,
                          "Vector3Stamped": Vector3Stamped, "TransformStamped": TransformStamped,
                          "TransformBroadcaster": TransformBroadcaster})

    def close(self) -> None:
        import rclpy
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    # ---------------------------------------------------------------- helpers

    def _stamp(self):
        return self.node.get_clock().now().to_msg()

    def _transform(self, child: str, xyz: Sequence[float], quat_wxyz=(1.0, 0.0, 0.0, 0.0)):
        t = self.m["TransformStamped"]()
        t.header.stamp, t.header.frame_id, t.child_frame_id = self._stamp(), FRAME, child
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = (
            float(v) for v in xyz)
        w, x, y, z = (float(v) for v in quat_wxyz)
        t.transform.rotation.w, t.transform.rotation.x = w, x
        t.transform.rotation.y, t.transform.rotation.z = y, z
        return t

    def _marker(self, ns: str, mid: int, kind: int, rgb, alpha: float = 1.0, frame: str = FRAME):
        mk = self.m["Marker"]()
        mk.header.frame_id, mk.header.stamp = frame, self._stamp()
        mk.ns, mk.id, mk.type, mk.action = ns, mid, kind, mk.ADD
        mk.pose.orientation.w = 1.0
        mk.color.r, mk.color.g, mk.color.b = (float(c) for c in rgb)
        mk.color.a = float(alpha)
        return mk

    def _point(self, xyz):
        from geometry_msgs.msg import Point
        p = Point()
        p.x, p.y, p.z = (float(v) for v in xyz)
        return p

    # ------------------------------------------------------------------ public

    def new_episode(self) -> None:
        self.trail.clear()
        array = self.m["MarkerArray"]()
        clear = self.m["Marker"]()
        clear.action = clear.DELETEALL
        array.markers.append(clear)
        self.scene.publish(array)

    def publish(self, s: Mapping[str, Any]) -> None:
        M = self.m["Marker"]
        uav, pad_est, pad_truth = (np.asarray(s[k], float) for k in
                                   ("uav", "pad_estimate", "pad_truth"))
        self.tf.sendTransform([self._transform("uav", uav, s["quat_wxyz"]),
                               self._transform("pad_estimate", pad_est),
                               self._transform("pad_truth", pad_truth)])
        self.trail.append(tuple(uav))
        array = self.m["MarkerArray"]()
        est = self._marker("pad", 0, M.CYLINDER, (0.20, 0.45, 0.90), 0.55)
        est.pose.position = self._point(pad_est)
        est.scale.x = est.scale.y = 1.0
        est.scale.z = 0.02
        truth = self._marker("pad", 1, M.CYLINDER, (0.15, 0.75, 0.30), 0.35)
        truth.pose.position = self._point(pad_truth)
        truth.scale.x = truth.scale.y = 1.05
        truth.scale.z = 0.01
        body = self._marker("uav", 0, M.SPHERE, (0.95, 0.85, 0.20))
        body.pose.position = self._point(uav)
        body.scale.x = body.scale.y = body.scale.z = 0.30
        trail = self._marker("uav", 1, M.LINE_STRIP, (0.95, 0.85, 0.20), 0.8)
        trail.scale.x = 0.03
        trail.points = [self._point(p) for p in self.trail[-600:]]
        action = self._marker("uav", 2, M.ARROW, (0.90, 0.30, 0.90))
        action.scale.x, action.scale.y, action.scale.z = 0.05, 0.10, 0.12
        action.points = [self._point(uav), self._point(uav + 0.5 * np.asarray(s["action"]))]
        colour = OUTCOME_COLOUR.get(s.get("status") or "", (1.0, 1.0, 1.0))
        hud = self._marker("hud", 0, M.TEXT_VIEW_FACING, colour)
        hud.pose.position = self._point(pad_truth + np.array([0.0, 0.0, 3.0]))
        hud.scale.z = 0.25
        hud.text = s["hud"]
        array.markers += [est, truth, body, trail, action, hud]
        self.scene.publish(array)
        self._publish_ontology(s, pad_truth)
        Float64 = self.m["Float64"]
        for name in self.scalars:
            msg = Float64()
            msg.data = float(s[name])
            self.scalars[name].publish(msg)
        for publisher, value in ((self.action, s["action"]),
                                 (self.confidence, s.get("wmon_confidence") or (0, 0, 0))):
            v = self.m["Vector3Stamped"]()
            v.header.stamp, v.header.frame_id = self._stamp(), FRAME
            v.vector.x, v.vector.y, v.vector.z = (float(x) for x in value)
            publisher.publish(v)
        ev = self.m["Float64MultiArray"]()
        ev.data = [float(x) for x in s["evidence"]]
        self.evidence.publish(ev)
        text = self.m["String"]()
        text.data = json.dumps(s["telemetry"], allow_nan=False, default=float)
        self.telemetry.publish(text)

    def _publish_ontology(self, s: Mapping[str, Any], anchor: np.ndarray) -> None:
        from .graph import SEMANTIC_EDGES
        M = self.m["Marker"]
        base = anchor + np.array([2.5, -1.5, 0.5])
        place = {n: base + np.array([0.0, x, y]) for n, (x, y) in NODE_LAYOUT.items()}
        array = self.m["MarkerArray"]()
        weights = dict(zip(NODE_LAYOUT, s["evidence"]))
        scale = max(1e-6, max(weights.values()))
        for i, node in enumerate(NODE_LAYOUT):
            w = weights[node] / scale
            ball = self._marker("nodes", i, M.SPHERE, _ramp(w) if node != "LandingContext"
                                else (0.55, 0.55, 0.95))
            ball.pose.position = self._point(place[node])
            ball.scale.x = ball.scale.y = ball.scale.z = 0.22 + 0.18 * w
            label = self._marker("labels", i, M.TEXT_VIEW_FACING, (1.0, 1.0, 1.0))
            label.pose.position = self._point(place[node] + np.array([0.0, 0.0, 0.25]))
            label.scale.z = 0.14
            label.text = f"{NODE_SHORT[node]} {weights[node]:.2f}"
            array.markers += [ball, label]
        edges = self._marker("edges", 0, M.LINE_LIST, (0.75, 0.75, 0.75), 0.6)
        edges.scale.x = 0.015
        for source, target, _ in SEMANTIC_EDGES:
            edges.points += [self._point(place[source]), self._point(place[target])]
        array.markers.append(edges)
        self.ontology.publish(array)


class _Disabled:
    enabled = False

    def new_episode(self) -> None:
        pass

    def publish(self, s: Mapping[str, Any]) -> None:
        pass

    def close(self) -> None:
        pass
