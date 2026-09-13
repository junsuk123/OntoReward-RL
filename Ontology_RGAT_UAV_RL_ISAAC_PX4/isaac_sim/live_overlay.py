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
PAIR_COLORS = (
    (0.00, 0.45, 0.74, 0.88),
    (0.85, 0.33, 0.10, 0.88),
    (0.47, 0.67, 0.19, 0.88),
)


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
                 telemetry_topics: Sequence[str] | None = None,
                 success_radius_m: float = 0.28):
        self.enabled = bool(enabled)
        self.multi_pair = bool(telemetry_topics and len(telemetry_topics) > 1)
        self.success_radius = float(success_radius_m)
        self.draw = _acquire() if self.enabled else None
        self.status = "running"
        self._uav_trail: list[tuple[float, float, float]] = []
        self._deck_trail: list[tuple[float, float, float]] = []
        self._pair_uav_trails: dict[int, list[tuple[float, float, float]]] = {}
        self._pair_deck_trails: dict[int, list[tuple[float, float, float]]] = {}
        self._pair_status: dict[int, str] = {}
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
            topics = tuple(telemetry_topics or (telemetry_topic,))
            for index, topic in enumerate(topics):
                node.create_subscription(
                    String, str(topic),
                    lambda message, pair_index=index: self._on_telemetry(
                        message, pair_index), 10)
        except Exception as exc:                       # pragma: no cover
            print(f"Isaac overlay: telemetry subscription unavailable ({exc}); "
                  "drawing geometry only.")

    def _on_telemetry(self, message: Any, pair_index: int = 0) -> None:
        try:
            payload = json.loads(message.data)
        except (ValueError, AttributeError):
            return
        status = payload.get("status")
        if isinstance(status, str):
            self.status = status
            self._pair_status[int(pair_index)] = status

    def reset(self, pair_index: int | None = None) -> None:
        """Clear one pair's trails, or every trail in a single-pair run."""
        if pair_index is None:
            self._uav_trail.clear()
            self._deck_trail.clear()
            self._pair_uav_trails.clear()
            self._pair_deck_trails.clear()
            self._pair_status.clear()
        else:
            index = int(pair_index)
            self._pair_uav_trails.pop(index, None)
            self._pair_deck_trails.pop(index, None)
            self._pair_status.pop(index, None)
        self.status = "running"
        if self.draw is not None and pair_index is None:
            self.draw.clear_lines()
            self.draw.clear_points()

    def update(self, uav_position: Sequence[float], deck_position: Sequence[float]) -> None:
        """Redraw for one rendered frame. Both positions are world ENU."""
        self.update_many(((uav_position, deck_position),))

    def update_many(self, pairs: Sequence[tuple[Sequence[float], Sequence[float]]]
                    ) -> None:
        """Draw all UAV/UGV pairs in one pass without debug-draw erasure races."""
        if not self.enabled or self.draw is None:
            return
        current = []
        for index, (uav_position, deck_position) in enumerate(pairs):
            uav = tuple(float(v) for v in np.asarray(
                uav_position, dtype=float).reshape(3))
            deck = tuple(float(v) for v in np.asarray(
                deck_position, dtype=float).reshape(3))
            uav_trail = self._pair_uav_trails.setdefault(index, [])
            deck_trail = self._pair_deck_trails.setdefault(index, [])
            uav_trail.append(uav)
            deck_trail.append(deck)
            del uav_trail[:-TRAIL_LIMIT]
            del deck_trail[:-TRAIL_LIMIT]
            current.append((index, uav, deck, uav_trail, deck_trail))
        if len(current) == 1:
            self._uav_trail = current[0][3]
            self._deck_trail = current[0][4]

        starts: list[tuple[float, float, float]] = []
        ends: list[tuple[float, float, float]] = []
        colors: list[tuple[float, float, float, float]] = []
        widths: list[float] = []

        def segment(a, b, color, width):
            starts.append(a)
            ends.append(b)
            colors.append(color)
            widths.append(width)

        point_positions = []
        point_colors = []
        for index, uav, deck, uav_trail, deck_trail in current:
            pair_color = PAIR_COLORS[index % len(PAIR_COLORS)]
            deck_color = (*pair_color[:3], 0.48)
            for trail, color in ((uav_trail, pair_color),
                                 (deck_trail, deck_color)):
                for a, b in zip(trail, trail[1:]):
                    segment(a, b, color, 2.0)

            status = self._pair_status.get(
                index, "running" if self.multi_pair else self.status)
            status_color = (pair_color if status == "running" else
                            STATUS_COLORS.get(status, pair_color))
            segment(uav, deck, status_color, 3.0)
            segment(uav, (uav[0], uav[1], deck[2]),
                    (0.55, 0.58, 0.64, 0.6), 1.5)

            points = 48
            ring = [(deck[0] + self.success_radius * math.cos(2 * math.pi * i / points),
                     deck[1] + self.success_radius * math.sin(2 * math.pi * i / points),
                     deck[2] + 0.01) for i in range(points + 1)]
            ring_color = (*pair_color[:3], RING_COLOR[3])
            for a, b in zip(ring, ring[1:]):
                segment(a, b, ring_color, 2.0)
            point_positions.extend((uav, deck))
            point_colors.extend((status_color, deck_color))

        self.draw.clear_lines()
        self.draw.draw_lines(starts, ends, colors, widths)
        self.draw.clear_points()
        self.draw.draw_points(
            point_positions, point_colors, [12.0] * len(point_positions))
