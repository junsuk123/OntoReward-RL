"""Pad memory and the ONE classifier of why the pad is not in the frame.

Without landing legs the camera ends 0.04 m BELOW the pad plane at contact, so
the frame covers only +-0.09 m of pad at a body height of 0.25 m and no tag
fits: losing the markers directly above the pad is what a correct landing
looks like. Losing them high up, after drifting off, or for longer than the
commit window is not. This module tells the two apart from the PREVIOUS pad
observations plus the EKF state since then, and nothing else.

Classes (``SafetyStatus.PAD_*``):

``VISIBLE``             detected in the current frame
``TRANSIENT_DROPOUT``   missed for <= ``transient_age_s``, any height
``TERMINAL_OCCLUSION``  missed above the pad: every condition below holds
``LOST``                any other miss; ``reason`` says why

Terminal conditions, judged at the LAST detection (t0) and now:
  (a) body height at t0 <= ``terminal_entry_height_m``       else HIGH_ALTITUDE
  (c) horizontal offset at t0 <= ``terminal_offset_m``       else DRIFT
  (d) at t0 not climbing, |v_rel_xy| <= touchdown xy speed   else FAST_APPROACH
  (e) dead-reckoned offset now <= ``terminal_offset_m``      else DRIFT
  (f) age <= ``terminal_window_s``                           else TIMEOUT
      and own_valid                                          else LOC_INVALID
  (g) the pad's velocity at t0 is KNOWN                      else NO_PAD_VELOCITY

The pad's velocity, not the relative one, is what is estimated: a least-
squares slope over the last 0.5 s of detections expressed in an own-
displacement frame (relative position + integral of EKF velocity). The pad
moves smoothly (<= 0.4 m/s^2); the vehicle does not, and a slope of the
RELATIVE position lagged a correcting vehicle's acceleration by up to 0.4 m/s
(seeds 3017, 3000 under oscillating gains: TERMINAL with the pad leaving at
0.6 m/s, MISSED_PAD_CONTACT). The relative velocity is then pad velocity
minus the CURRENT own velocity, so the vehicle's own manoeuvre enters at once.
Once TERMINAL it is LATCHED until (e) fails with hysteresis
(``terminal_release_offset_m``) or (f) fails, so the supervisor's mode does
not flicker. LOST is ABSORBING until the pad is detected again: TERMINAL can
only be entered at the start of a loss, never after a LOST verdict. EDGE_EXIT
is a diagnostic for a loss above the entry height whose predicted bearing left
the field of view.

Both (g) and the absorbing LOST were measured, not anticipated. On the local
plant with an oscillating teacher (kp 1.2, kd 2.2), seed 4110 had no valid pad
velocity, dead reckoning assumed the pad moved with the vehicle, and 1.7 s
after a LOST verdict the class flipped to TERMINAL with the pad 1.0 m away;
seed 4106 smoothed two-point differences (EMA 0.3) and missed a 0.6 m/s
relative velocity at loss. Both ended as MISSED_PAD_CONTACT.

The ontology and the supervisor each hold a ``PadMemory`` and call
``update`` with the same observations, so they cannot disagree.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .constants import DEFAULT_CONSTANTS, LandingConstants
from .observation import LandingObservation

VISIBLE, TRANSIENT_DROPOUT, TERMINAL_OCCLUSION, LOST = 0, 1, 2, 3
CLASS_NAMES = ("VISIBLE", "TRANSIENT_DROPOUT", "TERMINAL_OCCLUSION", "LOST")

EDGE_EXIT = 1
HIGH_ALTITUDE = 2
DRIFT = 4
TIMEOUT = 8
LOC_INVALID = 16
FAST_APPROACH = 32
NEVER_DETECTED = 64
NO_PAD_VELOCITY = 128
REASON_NAMES = {EDGE_EXIT: "EDGE_EXIT", HIGH_ALTITUDE: "HIGH_ALTITUDE",
                DRIFT: "DRIFT", TIMEOUT: "TIMEOUT", LOC_INVALID: "LOC_INVALID",
                FAST_APPROACH: "FAST_APPROACH", NEVER_DETECTED: "NEVER_DETECTED",
                NO_PAD_VELOCITY: "NO_PAD_VELOCITY"}


def reason_names(mask: int) -> tuple[str, ...]:
    return tuple(name for bit, name in REASON_NAMES.items() if mask & bit)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def bearing_fraction(rel: np.ndarray, constants: LandingConstants) -> float:
    """Off-axis angle of the pad over the narrow half-FOV (yaw unobserved)."""
    h_cam = constants.camera_height(-float(rel[2]))
    if h_cam <= 1e-6:
        return 1.0
    return math.atan2(float(np.hypot(rel[0], rel[1])), h_cam) / constants.min_half_fov_rad


@dataclass(frozen=True)
class PadLossAssessment:
    pad_class: int
    reason: int
    terminal_score: float          # T in [0, 1]
    lost_score: float              # L in [0, 1]
    relative_position: np.ndarray  # measured if visible, dead-reckoned otherwise
    relative_velocity: np.ndarray  # pad minus own, ENU (zero if not valid)
    relative_velocity_valid: bool
    pad_velocity: np.ndarray       # own EKF velocity + relative velocity
    last_height_m: float           # body height above the pad at t0
    last_offset_m: float
    last_bearing_fraction: float
    pad_freshness: float
    own_freshness: float

    @property
    def class_name(self) -> str:
        return CLASS_NAMES[self.pad_class]


@dataclass
class PadMemory:
    constants: LandingConstants = DEFAULT_CONSTANTS
    capture: float | None = None
    rel0: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v_rel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v_rel_valid: bool = False
    v_pad0: np.ndarray = field(default_factory=lambda: np.zeros(3))
    own_vz0: float = 0.0
    own_displacement: np.ndarray = field(default_factory=lambda: np.zeros(3))
    last_stamp: float | None = None
    latched_terminal: bool = False
    latched_lost: bool = False
    own_invalid_since: float | None = None
    recent: list = field(default_factory=list)   # (capture, pad in odometer frame)
    odometer: np.ndarray = field(default_factory=lambda: np.zeros(3))  # int v_own dt
    v_pad: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def update(self, obs: LandingObservation) -> PadLossAssessment:
        c = self.constants
        v_own = np.asarray(obs.own_velocity, float)
        if self.last_stamp is not None:
            self.odometer = self.odometer + v_own * max(obs.stamp - self.last_stamp, 0.0)
        new_detection = (obs.pad_capture_stamp is not None
                         and obs.pad_capture_stamp != self.capture)
        if new_detection:
            self._ingest(obs, v_own)
        elif self.last_stamp is not None and self.capture is not None:
            dt = max(obs.stamp - self.last_stamp, 0.0)
            self.own_displacement = self.own_displacement + v_own * dt
        self.last_stamp = obs.stamp
        if obs.own_valid:
            self.own_invalid_since = None
        elif self.own_invalid_since is None:
            self.own_invalid_since = obs.stamp
        return self._assess(obs)

    def own_invalid_duration(self, stamp: float) -> float:
        return 0.0 if self.own_invalid_since is None else stamp - self.own_invalid_since

    # ------------------------------------------------------------ internals
    def _ingest(self, obs: LandingObservation, v_own: np.ndarray) -> None:
        c = self.constants
        rel = np.asarray(obs.pad_relative_position, float)
        capture = float(obs.pad_capture_stamp)
        odometer_at_capture = self.odometer - v_own * max(obs.stamp - capture, 0.0)
        self.recent = [(t, r) for t, r in self.recent
                       if capture - t <= c.pad_velocity_window_s] + [(capture, rel + odometer_at_capture)]
        times = np.array([t for t, _ in self.recent])
        if (len(self.recent) >= c.pad_velocity_min_points
                and times[-1] - times[0] >= c.pad_velocity_min_span_s):
            centred = times - times.mean()
            points = np.array([r for _, r in self.recent])
            self.v_pad = (centred @ (points - points.mean(0))) / float(centred @ centred)
            self.v_rel = self.v_pad - v_own
            self.v_rel_valid = True
        else:
            self.v_pad, self.v_rel, self.v_rel_valid = np.zeros(3), np.zeros(3), False
        self.rel0, self.capture = rel, capture
        self.v_pad0 = self.v_pad.copy() if self.v_rel_valid else v_own.copy()
        self.own_vz0 = float(v_own[2])
        # The detection arrives after its capture: own motion since capture
        # already counts against the dead-reckoned offset.
        self.own_displacement = v_own * max(obs.stamp - capture, 0.0)
        self.latched_terminal = False
        self.latched_lost = False

    def _assess(self, obs: LandingObservation) -> PadLossAssessment:
        c = self.constants
        f_own = math.exp(-obs.own_age_s / c.own_freshness_tau_s) if obs.own_valid else 0.0
        if self.capture is None:
            return PadLossAssessment(
                LOST, NEVER_DETECTED, 0.0, 1.0, np.zeros(3), np.zeros(3), False,
                np.zeros(3), 0.0, 0.0, 1.0, 0.0, f_own)
        age = max(obs.stamp - self.capture, 0.0)
        f_pad = math.exp(-age / c.pad_freshness_tau_s)
        v_own = np.asarray(obs.own_velocity, float)
        h0 = -float(self.rel0[2])
        d0 = float(np.hypot(self.rel0[0], self.rel0[1]))
        b0 = bearing_fraction(self.rel0, c)
        v_rel_xy0 = float(np.hypot(self.v_rel[0], self.v_rel[1])) if self.v_rel_valid else 0.0
        if obs.pad_detected:
            rel = np.asarray(obs.pad_relative_position, float)
            self.latched_terminal = self.latched_lost = False
            v_rel_now = (self.v_pad - v_own) if self.v_rel_valid else np.zeros(3)
            return PadLossAssessment(
                VISIBLE, 0, 0.0, 0.0, rel, v_rel_now, self.v_rel_valid,
                self.v_pad.copy() if self.v_rel_valid else v_own.copy(),
                h0, d0, b0, f_pad, f_own)

        rel_pred = self.rel0 + self.v_pad0 * age - self.own_displacement
        d_pred = float(np.hypot(rel_pred[0], rel_pred[1]))
        v_rel_now = self.v_pad0 - v_own

        # Hard conditions -> class and reasons.
        reason = 0
        if h0 > c.terminal_entry_height_m:
            reason |= HIGH_ALTITUDE
            predicted = self.rel0 + self.v_rel * c.visibility_horizon_s
            if bearing_fraction(predicted, c) >= 1.0 or b0 >= 1.0:
                reason |= EDGE_EXIT
        if d0 > c.terminal_offset_m:
            reason |= DRIFT
        if self.own_vz0 > c.terminal_max_sink_up_m_s or v_rel_xy0 > c.touchdown_xy_speed_m_s:
            reason |= FAST_APPROACH
        if not self.v_rel_valid:
            reason |= NO_PAD_VELOCITY
        release = c.terminal_release_offset_m if self.latched_terminal else c.terminal_offset_m
        if d_pred > release:
            reason |= DRIFT
        if age > c.terminal_window_s:
            reason |= TIMEOUT
        if not obs.own_valid:
            reason |= LOC_INVALID

        if self.latched_lost:
            terminal = False
        elif self.latched_terminal:
            # Conditions at t0 cannot change while the pad is unseen and held
            # when the latch was set; only the live ones can end it.
            terminal = not (d_pred > release or age > c.terminal_window_s
                            or not obs.own_valid)
        else:
            terminal = reason == 0

        # Soft score: the same conditions as sigmoids, multiplied.
        terminal_score = (
            _sigmoid((c.terminal_entry_height_m - h0) / 0.03)
            * _sigmoid((c.terminal_offset_m - d0) / 0.03)
            * _sigmoid((c.terminal_max_sink_up_m_s - self.own_vz0) / 0.02)
            * _sigmoid((c.touchdown_xy_speed_m_s - v_rel_xy0) / 0.05)
            * _sigmoid((release - d_pred) / 0.03)
            * _sigmoid((c.terminal_window_s - age) / 0.2)
            * float(obs.own_valid) * float(self.v_rel_valid)
            * float(not self.latched_lost))
        lost_score = (1.0 - terminal_score) * (1.0 - f_pad)

        if age <= c.transient_age_s and not self.latched_terminal:
            pad_class = TRANSIENT_DROPOUT
            reason_out = 0
        elif terminal:
            pad_class, reason_out = TERMINAL_OCCLUSION, 0
            self.latched_terminal = True
        else:
            pad_class, reason_out = LOST, reason
            self.latched_terminal = False
            self.latched_lost = True
        return PadLossAssessment(
            pad_class, reason_out, float(terminal_score), float(lost_score),
            rel_pred, v_rel_now, self.v_rel_valid, self.v_pad0.copy(),
            h0, d0, b0, f_pad, f_own)
