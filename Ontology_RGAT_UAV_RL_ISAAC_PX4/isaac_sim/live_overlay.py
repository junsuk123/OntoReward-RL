"""In-window 3D overlay for the Isaac Sim view.

Isaac already renders the vehicle, the rover and the marker board. What it does
not show is anything the *experiment* knows: where the vehicle has been, where
the deck has driven, how far the two still are apart, and whether the policy is
inside the horizontal success tolerance. This draws those, so the simulator
window is a usable live view of a landing rather than only a physics picture.

It uses the debug-draw extension, which draws lines and points and has no text
primitive. The textual heads-up display therefore lives in RViz 2 (see
``python/ontology_rgat/viz/rviz.py``), and this overlay stays geometric.

Everything degrades to a no-op if the extension is unavailable, because a
simulation run must not fail for want of a decoration.
"""
from __future__ import annotations

import json
import math
from typing import Any, Sequence

import numpy as np

__all__ = ["LiveOverlay"]

TRAIL_LIMIT = 900
# Status -> colour for the vehicle-to-deck line. Reserved for state, so these
# never double as a series palette.
STATUS_COLORS = {
    "running": (0.16, 0.47, 0.84, 0.85),
    "success": (0.11, 0.69, 0.48, 0.95),
    "unsafe_touchdown": (0.89, 0.29, 0.28, 0.95),
    "flight_failure": (0.89, 0.29, 0.28, 0.95),
    "battery_depleted": (0.93, 0.63, 0.13, 0.95),
    "timeout": (0.55, 0.55, 0.58, 0.85),
}
UAV_TRAIL_COLOR = (0.16, 0.47, 0.84, 0.55)
DECK_TRAIL_COLOR = (0.92, 0.41, 0.20, 0.55)
RING_COLOR = (0.11, 0.69, 0.48, 0.75)


def _acquire():
    """The debug-draw interface, under whichever name this Isaac release uses."""
    for module in ("isaacsim.util.debug_draw._debug_draw",
                   "omni.isaac.debug_draw._debug_draw"):
        try:
            imported = __import__(module, fromlist=["acquire_debug_draw_interface"])
            return imported.acquire_debug_draw_interface()
        except Exception:
            continue
    return None


class LiveOverlay:
    """Draws the episode into the Isaac viewport."""

    def __init__(self, node: Any, enabled: bool = True,
                 telemetry_topic: str = "/landing_rl/telemetry",
                 success_radius_m: float = 0.28):
        self.enabled = bool(enabled)
        self.success_radius = float(success_radius_m)
        self.draw = _acquire() if self.enabled else None
        self.status = "running"
        self._uav_trail: list[tuple[float, float, float]] = []
        self._deck_trail: list[tuple[float, float, float]] = []
        if self.draw is None:
            if self.enabled:
                print("Isaac overlay disabled: the debug-draw extension is not "
                      "available in this release.")
            self.enabled = False
            return
        # The learner publishes the episode status; subscribing is optional, so
        # a simulator running without a learner still draws the geometry.
        try:
            from std_msgs.msg import String
            node.create_subscription(String, telemetry_topic, self._on_telemetry, 10)
        except Exception as exc:                       # pragma: no cover
            print(f"Isaac overlay: telemetry subscription unavailable ({exc}); "
                  "drawing geometry only.")

    def _on_telemetry(self, message: Any) -> None:
        try:
            payload = json.loads(message.data)
        except (ValueError, AttributeError):
            return
        status = payload.get("status")
        if isinstance(status, str):
            self.status = status

    def reset(self) -> None:
        """Clear the trails at an episode boundary."""
        self._uav_trail.clear()
        self._deck_trail.clear()
        self.status = "running"
        if self.draw is not None:
            self.draw.clear_lines()
            self.draw.clear_points()

    def update(self, uav_position: Sequence[float], deck_position: Sequence[float]) -> None:
        """Redraw for one rendered frame. Both positions are world ENU."""
        if not self.enabled or self.draw is None:
            return
        uav = tuple(float(v) for v in np.asarray(uav_position, dtype=float).reshape(3))
        deck = tuple(float(v) for v in np.asarray(deck_position, dtype=float).reshape(3))
        self._uav_trail.append(uav)
        self._deck_trail.append(deck)
        del self._uav_trail[:-TRAIL_LIMIT]
        del self._deck_trail[:-TRAIL_LIMIT]

        starts: list[tuple[float, float, float]] = []
        ends: list[tuple[float, float, float]] = []
        colors: list[tuple[float, float, float, float]] = []
        widths: list[float] = []

        def segment(a, b, color, width):
            starts.append(a)
            ends.append(b)
            colors.append(color)
            widths.append(width)

        for trail, color in ((self._uav_trail, UAV_TRAIL_COLOR),
                             (self._deck_trail, DECK_TRAIL_COLOR)):
            for a, b in zip(trail, trail[1:]):
                segment(a, b, color, 2.0)

        # What the policy is actually being scored on: the vector still to be
        # closed, and the height still to be lost.
        status_color = STATUS_COLORS.get(self.status, STATUS_COLORS["running"])
        segment(uav, deck, status_color, 3.0)
        segment(uav, (uav[0], uav[1], deck[2]), (0.55, 0.58, 0.64, 0.6), 1.5)

        # The horizontal success tolerance, drawn on the deck where it applies.
        points = 48
        ring = [(deck[0] + self.success_radius * math.cos(2 * math.pi * i / points),
                 deck[1] + self.success_radius * math.sin(2 * math.pi * i / points),
                 deck[2] + 0.01) for i in range(points + 1)]
        for a, b in zip(ring, ring[1:]):
            segment(a, b, RING_COLOR, 2.0)

        self.draw.clear_lines()
        self.draw.draw_lines(starts, ends, colors, widths)
        self.draw.clear_points()
        self.draw.draw_points([uav, deck], [status_color, DECK_TRAIL_COLOR], [12.0, 12.0])
