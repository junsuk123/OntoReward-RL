"""WMON pad track: the WMON atoms (``spatial.wmon``) on the spatial ``Measurement`` stream.

A drop-in alternative to ``ReferenceSpatialTrack(capture_aligned=True)``: the
same ``update(m, *, level_ground) -> accepted`` call, the same freshness rule
for optical frames, and the same public fields (``position``, ``velocity``,
``acceleration``, ``std``, ``initialized``, ``last_t``, ``last_detection``,
``last_sample``) that ``current_semantics.backend.CurrentObserver`` reads.  It
is OPT-IN: nothing registered uses it, and the registered observation contract
(``spatial-current-observation/3``, ABG track) is unchanged.

Mapping of the MATLAB prototype onto the landing problem (per ENU axis):

* position particle  -- each fresh optical solve, pad = own pose at capture
  minus the camera's own-minus-pad vector.  It is intermittent exactly like the
  prototype's GNSS: FOV loss, the optical cadence and blackouts remove it.
* velocity particle  -- the least-squares slope of the last ``velocity_span``
  position particles in the wave window.  The UGV carries no velocity sensor,
  so this particle is derived, and absent until two frames exist.
* acceleration atom  -- the pad carries no accelerometer either; the atom is
  anchored by a persistence prior ``a_prev exp(-gap/tau)`` (the ABG track's
  1.5 s decay) with confidence 1 instead of a measured particle.
* wave field         -- the causal FFT window runs over the last ``window``
  FRAMES (position and velocity particles), not over 100 Hz samples, and is
  dissolved (restarted) after a frame gap longer than ``history_reset_gap_s``.
* bonds              -- the MATLAB p-v and v-a bonds between the previous and
  the new capture, weighted k(gap) = k (dt_ref/gap)^bond_dt_exponent: the
  MATLAB weight at gap == dt_ref (0.1 s, Isaac's ~10 Hz cadence), and weaker
  after a long blackout so the measurement takes over instead of the
  innovation gate that locked the ABG track out (test_cs3d_reacquire.py).
  Exponent 1 keeps the correction time constant of the MATLAB step; 2 (unit
  normalisation) was measurably worse on the local plant
  (tools/wmon_pad_track_eval.py, 2026-11-10).

Measured status (2026-11-10, open-loop replay, 12 held-out local /12 flights,
results/wmon_pad_track_20261110): the registered ABG track with re-acquisition
is better while the pad is visible (0.025 vs 0.038 m), in a blackout (0.043 vs
0.058 m) and in lost view (0.40 vs 1.36 m); WMON is marginally better only in
the 1.5 s after a blackout (0.019 vs 0.022 m).  SPSA calibration selected
epoch 1 (validation rose).  WMON's advantage needs MEASURED velocity/
acceleration particles to integrate through a position dropout; the pad has
none here.  Keep it a study option, not a replacement
(docs/current_semantics/WMON_PAD_TRACK_KO.md).

Atoms live at the capture time of the last accepted frame (capture-aligned,
like the reference) and are propagated to each sample with the decaying
acceleration in closed form.  There is no innovation gate: a frame that
disagrees with the wave window raises its high-band ratio and so lowers its own
weight.  ``std`` is sqrt(diag(H^-1)) of the last solve, a relative spread in
the units of the (unnormalised) weights -- not a calibrated SI sigma.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .wmon import MATLAB_TRAINED, WMONParams, fourier_mask, solve_atoms, wave_field


ESTIMATOR_ID = "spatial.wmon_track.WMONSpatialTrack/1"
FRESH_AGE_S = 0.5          # ReferenceSpatialTrack capture-aligned freshness rule
MAX_SAMPLE_GAP_S = 2.0     # the reference refuses larger clock gaps too


@dataclass(frozen=True)
class WMONTrackSettings:
    """Structure of the adapter; ``params`` are the seven learned scalars."""
    params: WMONParams = MATLAB_TRAINED
    window: int = 12
    n_modes: int = 3
    dt_ref_s: float = 0.1                 # MATLAB cfg.dt: k(gap) == k at this gap
    bond_dt_exponent: float = 1.0         # k(gap) = k (dt_ref/gap)**exponent
    velocity_span: int = 4
    history_reset_gap_s: float = FRESH_AGE_S
    acceleration_tau_s: float = 1.5
    initial_velocity: str = "own"         # "own" (the reference's init) or "zero"

    def __post_init__(self) -> None:
        fourier_mask(self.window, self.n_modes)  # validates window / n_modes
        if not 2 <= self.velocity_span <= self.window:
            raise ValueError("velocity_span must be in [2, window]")
        for name in ("dt_ref_s", "history_reset_gap_s", "acceleration_tau_s"):
            if not getattr(self, name) > 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.bond_dt_exponent <= 4.0:
            raise ValueError("bond_dt_exponent must be in [0, 4]")
        if self.initial_velocity not in ("own", "zero"):
            raise ValueError("initial_velocity must be 'own' or 'zero'")

    def to_dict(self) -> dict[str, Any]:
        return {"estimator": ESTIMATOR_ID, "params": self.params.to_dict(),
                "theta": self.params.to_theta().tolist(),
                **{name: getattr(self, name) for name in
                   ("window", "n_modes", "dt_ref_s", "bond_dt_exponent", "velocity_span",
                    "history_reset_gap_s",
                    "acceleration_tau_s", "initial_velocity")}}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WMONTrackSettings":
        data = dict(data)
        if data.pop("estimator", ESTIMATOR_ID) != ESTIMATOR_ID:
            raise ValueError(f"settings are not for {ESTIMATOR_ID}")
        theta = data.pop("theta", None)
        params = data.pop("params", None)
        if theta is not None:
            params = WMONParams.from_theta(theta)
        elif params is not None:
            params = WMONParams(**params)
        else:
            params = MATLAB_TRAINED
        return cls(params=params, **data)

    @classmethod
    def load(cls, path: str | Path) -> "WMONTrackSettings":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload.get("settings", payload))


class WMONSpatialTrack:
    """Capture-aligned WMON pad track; see the module docstring."""

    estimator_id = ESTIMATOR_ID

    def __init__(self, settings: WMONTrackSettings | None = None) -> None:
        self.settings = WMONTrackSettings() if settings is None else settings
        s = self.settings
        self._mask = fourier_mask(s.window, s.n_modes)
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.std = np.tile(np.array([2.0, 3.0, 2.0])[:, None], (1, 3))  # reference's fresh std
        self.initialized = False
        self.last_t: float | None = None
        self.last_detection = -math.inf
        self.last_sample = -1
        # atoms at the anchor (capture time of the last accepted frame): rows p, v, a
        self.anchor_t: float | None = None
        self.atoms = np.zeros((3, 3))
        self._frames: deque[tuple[float, np.ndarray, np.ndarray | None]] = deque(
            maxlen=s.window)            # (capture, position particle, velocity particle)
        # diagnostics of the last accepted frame (evaluator-only)
        self.particle_confidence = np.ones((3, 3))   # rows p, v, a particle; columns x, y, z
        self.hf_ratio = np.zeros((2, 3))             # rows p, v channel
        self.energy = np.zeros(5)
        self.dissolutions = 0
        self.frames_assimilated = 0

    # ----------------------------------------------------------- interface

    def update(self, m: Any, *, level_ground: bool) -> bool:
        now = float(m.time_s)
        if self.last_t is not None and now < self.last_t - 1e-8:
            raise ValueError("simulation clock reversed; reset the episode")
        dt = 0.0 if self.last_t is None else now - self.last_t
        if dt > MAX_SAMPLE_GAP_S:
            raise ValueError("sensor gap exceeds 2 simulated seconds")
        if self.last_t is not None and dt <= 1e-12:
            return False
        accepted = False
        capture = m.optical_time_s
        fresh = (m.optical_position is not None and m.sample_id != self.last_sample
                 and capture is not None and m.optical_own_position is not None
                 and capture > self.last_detection and 0 <= now - capture <= FRESH_AGE_S)
        if fresh:
            capture = float(capture)
            measured = (np.asarray(m.optical_own_position, dtype=np.float64)
                        - np.asarray(m.optical_position, dtype=np.float64))
            if not self.initialized:
                self._initialize(capture, measured, np.asarray(m.own_velocity, np.float64),
                                 level_ground)
            else:
                self._assimilate(capture, measured, level_ground)
            self.initialized = accepted = True
            self.last_sample = m.sample_id
        if accepted:
            self.last_detection = capture
        if self.initialized:
            self._propagate(now, level_ground)
        self.last_t = now
        return accepted

    def diagnostics(self) -> dict[str, Any]:
        """Evaluator-only state of the last accepted frame (never part of the 21D vector)."""
        return {"estimator": ESTIMATOR_ID,
                "particle_confidence": self.particle_confidence.tolist(),
                "hf_ratio": self.hf_ratio.tolist(),
                "energy": dict(zip(("position", "velocity", "acceleration",
                                    "position_velocity_bond", "velocity_acceleration_bond"),
                                   self.energy.tolist())),
                "frames_in_window": len(self._frames),
                "frames_assimilated": self.frames_assimilated,
                "dissolutions": self.dissolutions}

    # ------------------------------------------------------------ internals

    def _initialize(self, capture: float, measured: np.ndarray, own_velocity: np.ndarray,
                    level_ground: bool) -> None:
        self.atoms[0] = measured
        self.atoms[1] = own_velocity if self.settings.initial_velocity == "own" else 0.0
        self.atoms[2] = 0.0
        if level_ground:
            self.atoms[1:, 2] = 0.0
        self.anchor_t = capture
        self._frames.clear()
        self._frames.append((capture, measured, None))
        self.frames_assimilated = 1

    def _velocity_particle(self) -> np.ndarray | None:
        frames = list(self._frames)[-self.settings.velocity_span:]
        if len(frames) < 2:
            return None
        times = np.array([t for t, _, _ in frames])
        points = np.stack([p for _, p, _ in frames])
        spread = ((times - times.mean()) ** 2).sum()
        if spread <= 1e-12:
            return None
        return ((times - times.mean())[:, None] * (points - points.mean(0))).sum(0) / spread

    def _assimilate(self, capture: float, measured: np.ndarray, level_ground: bool) -> None:
        s, p = self.settings, self.settings.params
        gap = max(capture - float(self.anchor_t), 1e-3)
        if gap > s.history_reset_gap_s:  # stale wave field: the particles dissolve
            self._frames.clear()
            self.dissolutions += 1
        self._frames.append((capture, measured, None))
        velocity = self._velocity_particle()
        if velocity is not None and level_ground:
            velocity[2] = 0.0
        self._frames[-1] = (capture, measured, velocity)
        has_velocity = velocity is not None

        # L0: the frame-indexed wave window; a missing velocity particle is held
        # (MATLAB fillMissingPosition) or, before the first, the anchor velocity.
        held = self.atoms[1].copy()
        columns = []
        for _, position, v in self._frames:
            held = held if v is None else v
            columns.append(np.concatenate([position, held]))
        segment = np.stack(columns, axis=1)                         # (6, n)
        if segment.shape[1] < s.window:
            segment = np.concatenate(
                [np.repeat(segment[:, :1], s.window - segment.shape[1], axis=1), segment],
                axis=1)
        wave = wave_field(segment, self._mask, p.fft_mix, p.noise_sensitivity)

        # L1-L2: particles; L3-L5: atoms of every axis in one solve.
        decay = math.exp(-gap / s.acceleration_tau_s)
        prior_acceleration = self.atoms[2] * decay
        conf_p, conf_v = wave.confidence[:3], wave.confidence[3:]
        particles = np.stack([wave.sample[:3],
                              wave.sample[3:] if has_velocity else self.atoms[1],
                              prior_acceleration], axis=1)          # (3 axes, [p, v, a])
        bond = (s.dt_ref_s / gap) ** s.bond_dt_exponent
        w = np.empty((3, 5))
        w[:, 0] = p.w_position * conf_p
        w[:, 1] = p.w_velocity * conf_v * float(has_velocity)
        w[:, 2] = p.w_acceleration
        w[:, 3] = p.k_position_velocity * bond
        w[:, 4] = p.k_velocity_acceleration * bond
        atoms, energy, normal = solve_atoms(particles, self.atoms[0], self.atoms[1], w, gap)
        self.atoms = atoms.T.copy()
        if level_ground:
            self.atoms[1:, 2] = 0.0
        self.anchor_t = capture
        self.frames_assimilated += 1
        self.particle_confidence = np.stack([conf_p, conf_v * float(has_velocity),
                                             np.ones(3)])
        self.hf_ratio = np.stack([wave.hf_ratio[:3], wave.hf_ratio[3:]])
        self.energy = energy.sum(axis=0)
        self.std = np.sqrt(np.diagonal(np.linalg.inv(normal), axis1=-2, axis2=-1)).T.copy()

    def _propagate(self, now: float, level_ground: bool) -> None:
        """Atoms from the anchor to ``now`` under a(t) = a0 exp(-t/tau)."""
        lag = max(now - float(self.anchor_t), 0.0)
        tau = self.settings.acceleration_tau_s
        decay = math.exp(-lag / tau)
        p0, v0, a0 = self.atoms
        self.acceleration = a0 * decay
        self.velocity = v0 + a0 * tau * (1.0 - decay)
        self.position = p0 + v0 * lag + a0 * tau * (lag - tau * (1.0 - decay))
        if level_ground:
            self.velocity[2] = self.acceleration[2] = 0.0


def wmon_track_factory(settings: WMONTrackSettings | None = None):
    """A zero-argument factory for ``CurrentObserver(track_factory=...)``."""
    settings = WMONTrackSettings() if settings is None else settings

    def factory() -> WMONSpatialTrack:
        return WMONSpatialTrack(settings)

    factory.estimator_id = ESTIMATOR_ID  # type: ignore[attr-defined]
    factory.settings = settings          # type: ignore[attr-defined]
    return factory
