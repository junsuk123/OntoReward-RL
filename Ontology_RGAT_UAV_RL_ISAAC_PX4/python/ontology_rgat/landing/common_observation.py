"""The common observation O_t = {P_t, D_t, H_t}, dimension-generic.

One generator feeds every arm. It records what was MEASURED and nothing it
would have to infer:

``P_t``  landing-pad observation: for every marker registered on the pad, the
         four image corners in the detector's own order, normalized by the
         image size, and whether that marker was detected in this frame; plus
         the frame's capture time and whether a usable frame exists at all.
``D_t``  the drone's own fused navigation state: position, velocity, attitude
         as sin/cos pairs and body rate, with its time stamp.
``H_t``  the previous ``K`` (P, D) records and the last record in which any
         marker was detected, with their time stamps.

What is deliberately NOT here: pad position/velocity/acceleration estimates,
tracker uncertainty, predicted bearings or FOV margins, descent permission,
abort requests, "tracking too slow" -- every judgement belongs to the stage
after this one (the ontology, or whatever the baseline learns). Marker ids are
identification keys mapped to fixed slots, never numeric features, and corners
are never re-sorted, so a corner keeps its identity under rotation. Undetected
slots are zero with mask 0; a previous frame's corners are never left in the
current slot -- the past lives in ``H_t`` only, and ``H_t`` stores (P, D)
pairs, never a previous O (no history inside history).

The dimension is a parameter. 2D (``axes=('x',)``) is the x-z plane with
pitch only: D = [x, z, vx, vz, sin th, cos th, w_th] (7). 3D
(``axes=('x', 'y')``) is ENU with full attitude: D = [x, y, z, vx, vy, vz,
sin/cos of roll, pitch, yaw, p, q, r] (15). The image keeps both (u, v) in
either dimension: a planar task does not make the camera one-dimensional.

Times are stored absolute and exposed to a policy only as ages relative to the
decision time, so an old image paired with a fresh navigation state is visible
as such instead of hidden.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import numpy as np

CORNERS_PER_MARKER = 4
MARKER_FEATURES = 2 * CORNERS_PER_MARKER + 1  # eight coordinates and a mask


def normalize_corners_px(corners_px, image_size) -> np.ndarray:
    """Pixel corners -> [-1, 1] by image size: u~ = 2u/(W-1) - 1.

    Image-size normalization only: not metres, not camera rays. Distortion
    handling and resolution must be the same for every compared arm.
    """
    width, height = (float(v) for v in image_size)
    corners = np.asarray(corners_px, dtype=float)
    if corners.shape[-1] != 2:
        raise ValueError("corners must end in (u, v) pairs")
    scale = np.array([2.0 / (width - 1.0), 2.0 / (height - 1.0)])
    return corners * scale - 1.0


@dataclass(frozen=True)
class PadObservation:
    """Every registered marker of ONE pad, in fixed slots."""
    pad_id: str
    marker_ids: tuple[int, ...]
    corners: np.ndarray            # (M, 4, 2) normalized, zero where undetected
    detected_mask: np.ndarray      # (M,) 1.0 detected in this frame, else 0.0
    image_stamp: float | None      # capture time of the frame, None if none
    frame_valid: bool              # a usable detector result exists

    def __post_init__(self):
        m = len(self.marker_ids)
        if np.shape(self.corners) != (m, CORNERS_PER_MARKER, 2):
            raise ValueError("corners must be (M, 4, 2) for the registered markers")
        if np.shape(self.detected_mask) != (m,):
            raise ValueError("detected_mask must be (M,)")
        if not np.isfinite(self.corners).all():
            raise ValueError("corners must be finite; undetected slots are zero")
        if np.any(self.corners[np.asarray(self.detected_mask) == 0] != 0):
            raise ValueError("an undetected slot must not carry corner values")

    def features(self) -> np.ndarray:
        """(M, 9): eight corner coordinates and the detection mask per slot."""
        return np.concatenate(
            [self.corners.reshape(len(self.marker_ids), -1),
             np.asarray(self.detected_mask, dtype=float)[:, None]], axis=1)

    @property
    def any_detected(self) -> bool:
        return bool(self.frame_valid and np.any(self.detected_mask > 0))


@dataclass(frozen=True)
class DroneState:
    """The drone's own fused navigation output, in the common local frame."""
    position: np.ndarray           # (n+1,) horizontal axes then up
    velocity: np.ndarray           # (n+1,)
    attitude_sincos: np.ndarray    # 2D: (2,) pitch; 3D: (6,) roll, pitch, yaw
    body_rate: np.ndarray          # 2D: (1,) pitch rate; 3D: (3,) p, q, r
    navigation_stamp: float
    navigation_valid: bool = True

    def features(self) -> np.ndarray:
        return np.concatenate([self.position, self.velocity,
                               self.attitude_sincos, self.body_rate]).astype(float)


def drone_state_3d(position, velocity, quaternion_wxyz, body_rate, stamp,
                   valid=True) -> DroneState:
    """3D: ENU position/velocity, roll-pitch-yaw as sin/cos, body rates."""
    from scipy.spatial.transform import Rotation
    q = np.asarray(quaternion_wxyz, dtype=float)
    angles = Rotation.from_quat(q[[1, 2, 3, 0]]).as_euler("xyz")
    sincos = np.ravel(np.column_stack([np.sin(angles), np.cos(angles)]))
    return DroneState(np.asarray(position, float).copy(), np.asarray(velocity, float).copy(),
                      sincos, np.asarray(body_rate, float).copy(), float(stamp), bool(valid))


def drone_state_2d(x, z, vx, vz, pitch, pitch_rate, stamp, valid=True) -> DroneState:
    """2D: x-z plane, pitch only."""
    return DroneState(np.array([x, z], float), np.array([vx, vz], float),
                      np.array([math.sin(pitch), math.cos(pitch)]),
                      np.array([float(pitch_rate)]), float(stamp), bool(valid))


@dataclass(frozen=True)
class ObservationRecord:
    """F = (P, D): what was seen and where the drone was, nothing more."""
    pad: PadObservation
    drone: DroneState


@dataclass(frozen=True)
class ObservationHistory:
    previous: tuple[ObservationRecord | None, ...]   # newest first, length K
    last_seen: ObservationRecord | None


@dataclass(frozen=True)
class CommonObservation:
    pad: PadObservation
    drone: DroneState
    history: ObservationHistory

    def vector(self, decision_time: float | None = None) -> np.ndarray:
        """Flat representation for a vector policy; see ``layout``.

        Ages are decision time minus each record's own stamps, so a stale
        image next to a fresh navigation state stays visible.
        """
        now = self.drone.navigation_stamp if decision_time is None else float(decision_time)
        parts = [self._record_block(ObservationRecord(self.pad, self.drone), now, current=True)]
        for record in self.history.previous:
            parts.append(self._record_block(record, now))
        parts.append(self._record_block(self.history.last_seen, now))
        return np.concatenate(parts)

    def _record_block(self, record, now, current=False):
        width = self.pad.features().size + self.drone.features().size
        if record is None:
            return np.zeros(width + (3 if current else 4))
        image_age = (0.0 if record.pad.image_stamp is None
                     else now - record.pad.image_stamp)
        nav_age = now - record.drone.navigation_stamp
        head = ([float(record.pad.frame_valid), image_age, nav_age] if current
                else [1.0, float(record.pad.frame_valid), image_age, nav_age])
        return np.concatenate([head, record.pad.features().ravel(), record.drone.features()])

    def layout(self) -> dict:
        """Named slices of ``vector()``; every arm reads the same layout."""
        pad = self.pad.features().size
        drone = self.drone.features().size
        out, start = {}, 0

        def take(name, size):
            nonlocal start
            out[name] = slice(start, start + size)
            start += size
        take("current.meta(frame_valid, image_age, nav_age)", 3)
        take("current.pad", pad)
        take("current.drone", drone)
        for k in range(len(self.history.previous)):
            take(f"previous[{k}].meta(exists, frame_valid, image_age, nav_age)", 4)
            take(f"previous[{k}].pad", pad)
            take(f"previous[{k}].drone", drone)
        take("last_seen.meta(exists, frame_valid, image_age, nav_age)", 4)
        take("last_seen.pad", pad)
        take("last_seen.drone", drone)
        out["size"] = start
        return out


@dataclass
class CommonObservationBuilder:
    """Turns per-frame detections and navigation states into O_t.

    One instance per episode stream; ``reset`` at every episode boundary.
    """
    marker_ids: Sequence[int]
    image_size: tuple[int, int]
    history_length: int = 2
    pad_id: str = "landing_pad_0"
    max_image_age_s: float = 0.5
    _previous: deque = field(init=False, repr=False)
    _last_seen: ObservationRecord | None = field(init=False, default=None, repr=False)

    def __post_init__(self):
        ids = tuple(int(i) for i in self.marker_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("registered marker ids must be unique and non-empty")
        if self.history_length < 0:
            raise ValueError("history length must be non-negative")
        self.marker_ids = ids
        self._slot = {marker_id: k for k, marker_id in enumerate(ids)}
        self.reset()

    def reset(self) -> None:
        self._previous = deque(maxlen=self.history_length)
        self._last_seen = None

    def pad_observation(self, corners_px: Mapping[int, Sequence] | None,
                        image_stamp: float | None, decision_time: float) -> PadObservation:
        """Slot the detector's {id: 4x2 pixel corners}; unknown ids are dropped."""
        m = len(self.marker_ids)
        corners = np.zeros((m, CORNERS_PER_MARKER, 2))
        mask = np.zeros(m)
        frame_valid = (image_stamp is not None and corners_px is not None
                       and math.isfinite(float(image_stamp))
                       and 0.0 <= decision_time - float(image_stamp) <= self.max_image_age_s)
        if frame_valid:
            for marker_id, quad in corners_px.items():
                slot = self._slot.get(int(marker_id))
                quad = np.asarray(quad, dtype=float)
                if slot is None or quad.shape != (CORNERS_PER_MARKER, 2) or not np.isfinite(quad).all():
                    continue
                corners[slot] = normalize_corners_px(quad, self.image_size)
                mask[slot] = 1.0
        return PadObservation(self.pad_id, self.marker_ids, corners, mask,
                              None if image_stamp is None else float(image_stamp),
                              bool(frame_valid))

    def observe(self, pad: PadObservation, drone: DroneState) -> CommonObservation:
        """O_t from the current (P, D) and the stored past, then store (P, D)."""
        previous = tuple(self._previous) + (None,) * (self.history_length - len(self._previous))
        observation = CommonObservation(pad, drone, ObservationHistory(previous, self._last_seen))
        record = ObservationRecord(pad, drone)
        self._previous.appendleft(record)
        if pad.any_detected:
            self._last_seen = record
        return observation
