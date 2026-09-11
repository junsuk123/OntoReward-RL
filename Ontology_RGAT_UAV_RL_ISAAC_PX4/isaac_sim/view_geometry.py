"""Pure geometry for keeping two moving subjects inside a spectator view."""

from __future__ import annotations

import math

import numpy as np


def street_offset_enu(offset_m, heading_rad: float) -> np.ndarray:
    """Convert ``[along, across, up]`` from a vehicle frame to world ENU."""
    offset = np.asarray(offset_m, dtype=float).reshape(3)
    along = np.array([math.cos(heading_rad), math.sin(heading_rad), 0.0])
    across = np.array([-along[1], along[0], 0.0])
    return offset[0] * along + offset[1] * across + np.array(
        [0.0, 0.0, offset[2]])


def paired_view_pose(uav_position, deck_position, offset_enu,
                     pair_span_m: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Return eye, midpoint target, and zoom for a two-subject camera.

    The base offset frames subjects up to ``pair_span_m`` apart. Beyond that,
    eye distance grows in the same ratio as subject separation, keeping their
    angular separation bounded instead of allowing either one to leave view.
    """
    uav = np.asarray(uav_position, dtype=float).reshape(3)
    deck = np.asarray(deck_position, dtype=float).reshape(3)
    offset = np.asarray(offset_enu, dtype=float).reshape(3)
    span = max(float(pair_span_m), 1e-6)
    separation = float(np.linalg.norm(uav - deck))
    zoom = max(1.0, separation / span)
    target = 0.5 * (uav + deck)
    return target + zoom * offset, target, zoom


def angular_separation_deg(eye, first, second) -> float:
    """Angle subtended by two subjects at the camera eye."""
    eye = np.asarray(eye, dtype=float).reshape(3)
    a = np.asarray(first, dtype=float).reshape(3) - eye
    b = np.asarray(second, dtype=float).reshape(3) - eye
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return 180.0
    cosine = float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))
