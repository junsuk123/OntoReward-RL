"""The minimal observation o_t and the assembler that builds it.

o_t = [own_x, own_y, own_z, own_vx, own_vy, own_vz,
       rel_x, rel_y, rel_z, own_valid, own_age_s, pad_detected, pad_age_s]

* ``own_*`` is the EKF state (IMU + GNSS fused), position AND velocity, ENU.
* ``rel`` is pad minus body in ENU axes, measured at the image CAPTURE time:
  the PnP solve gives the board origin in the body FLU frame, and the body
  attitude AT CAPTURE rotates it into ENU. Attitude is used for that rotation
  only; it never enters the vector.
* On a miss ``rel`` holds the last detected value and ``pad_age_s`` grows;
  zero would read as "directly above the pad". Never detected: rel = 0,
  age = cap.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import deque
from dataclasses import dataclass, field
import math

import numpy as np

from . import OBSERVATION_SCHEMA_ID
from .constants import DEFAULT_CONSTANTS, LandingConstants

VECTOR_FIELDS = ("own_x", "own_y", "own_z", "own_vx", "own_vy", "own_vz",
                 "rel_x", "rel_y", "rel_z", "own_valid", "own_age_s",
                 "pad_detected", "pad_age_s")
VECTOR_SIZE = len(VECTOR_FIELDS)


def quat_wxyz_to_matrix(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        raise ValueError("zero quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


@dataclass(frozen=True)
class OwnState:
    """One EKF solution, ENU position/velocity, ENU<-FLU attitude."""
    stamp: float
    position: np.ndarray
    velocity: np.ndarray
    quaternion_wxyz: np.ndarray
    valid: bool


@dataclass(frozen=True)
class PadDetectionSample:
    """One processed frame. ``position_body`` is the board origin in body FLU."""
    capture_stamp: float
    detected: bool
    position_body: np.ndarray


@dataclass(frozen=True)
class LandingObservation:
    stamp: float
    own_position: np.ndarray
    own_velocity: np.ndarray
    own_valid: bool
    own_age_s: float
    pad_relative_position: np.ndarray
    pad_detected: bool
    pad_age_s: float
    pad_capture_stamp: float | None    # None: never detected
    schema_id: str = OBSERVATION_SCHEMA_ID

    @property
    def ever_detected(self) -> bool:
        return self.pad_capture_stamp is not None

    def vector(self) -> np.ndarray:
        return np.concatenate([
            np.asarray(self.own_position, float), np.asarray(self.own_velocity, float),
            np.asarray(self.pad_relative_position, float),
            [float(self.own_valid), self.own_age_s, float(self.pad_detected),
             self.pad_age_s]]).astype(np.float32)

    @classmethod
    def from_vector(cls, stamp: float, vector, pad_capture_stamp=None,
                    *, constants: LandingConstants = DEFAULT_CONSTANTS):
        v = np.asarray(vector, dtype=float)
        if v.shape != (VECTOR_SIZE,):
            raise ValueError(f"observation vector must have {VECTOR_SIZE} entries")
        if pad_capture_stamp is None and v[12] < constants.pad_age_cap_s:
            pad_capture_stamp = stamp - v[12]
        return cls(stamp, v[0:3].copy(), v[3:6].copy(), bool(v[9] > 0.5),
                   float(v[10]), v[6:9].copy(), bool(v[11] > 0.5), float(v[12]),
                   pad_capture_stamp)


class OdometryBuffer:
    """Recent EKF solutions, queried at an image's capture time."""

    def __init__(self, horizon_s: float = 2.0):
        self.horizon_s = float(horizon_s)
        self._states: deque[OwnState] = deque()

    def push(self, state: OwnState) -> None:
        if self._states and state.stamp < self._states[-1].stamp:
            return  # out of order; the buffer is monotonic
        self._states.append(state)
        while self._states and state.stamp - self._states[0].stamp > self.horizon_s:
            self._states.popleft()

    @property
    def latest(self) -> OwnState | None:
        return self._states[-1] if self._states else None

    def at(self, stamp: float) -> OwnState | None:
        """Nearest solution to ``stamp`` (EKF runs ~100x faster than the camera)."""
        if not self._states:
            return None
        stamps = [s.stamp for s in self._states]
        i = bisect_left(stamps, stamp)
        candidates = [self._states[j] for j in (i - 1, i) if 0 <= j < len(stamps)]
        return min(candidates, key=lambda s: abs(s.stamp - stamp))


@dataclass
class ObservationAssembler:
    constants: LandingConstants = DEFAULT_CONSTANTS
    odometry: OdometryBuffer = field(default_factory=OdometryBuffer)
    _rel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    _capture: float | None = None
    _last_frame_stamp: float | None = None
    _last_frame_detected: bool = False

    def on_odometry(self, state: OwnState) -> None:
        self.odometry.push(state)

    def on_pad_detection(self, sample: PadDetectionSample) -> None:
        self._last_frame_stamp = sample.capture_stamp
        self._last_frame_detected = False
        if not sample.detected:
            return
        state = self.odometry.at(sample.capture_stamp)
        if state is None or not state.valid:
            return  # cannot rotate into ENU without an attitude at capture
        rotation = quat_wxyz_to_matrix(state.quaternion_wxyz)
        self._rel = rotation @ np.asarray(sample.position_body, dtype=float)
        self._capture = float(sample.capture_stamp)
        self._last_frame_detected = True

    def assemble(self, stamp: float) -> LandingObservation | None:
        own = self.odometry.latest
        if own is None:
            return None
        c = self.constants
        detected = (self._last_frame_detected and self._last_frame_stamp is not None
                    and stamp - self._last_frame_stamp <= c.frame_valid_age_s)
        if self._capture is None:
            age, rel = c.pad_age_cap_s, np.zeros(3)
        else:
            age, rel = min(max(stamp - self._capture, 0.0), c.pad_age_cap_s), self._rel.copy()
        return LandingObservation(
            stamp=float(stamp),
            own_position=np.asarray(own.position, float).copy(),
            own_velocity=np.asarray(own.velocity, float).copy(),
            own_valid=bool(own.valid),
            own_age_s=max(float(stamp - own.stamp), 0.0),
            pad_relative_position=rel, pad_detected=bool(detected),
            pad_age_s=float(age), pad_capture_stamp=self._capture)
