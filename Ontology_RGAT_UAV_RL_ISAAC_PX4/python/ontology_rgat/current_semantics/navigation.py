"""Live own navigation: GNSS outages + IMU between the flight and the observer (opt-in).

The registered observation (``spatial-current-observation/3``) takes the own
state from the flight as is and fixes ``drone_navigationValid = 1`` and
``drone_navigationAge = 0`` (backend.NAVIGATION_NOTE).  This module is the
opt-in revision ``spatial-current-observation/4``: the own state the observer
and the 21D vector see comes from a navigation estimator fed by degraded sensors,
and the two navigation channels are live.

Sensors (per ``Measurement`` sample, from the flight's own state as reference):

* GNSS position every ``gnss_period_s`` with N(0, gnss_sigma_m) noise, absent
  inside the episode's outage window (drawn from the episode seed);
* velocity every sample with N(0, velocity_sigma_m_s) (flow/VIO stand-in);
* acceleration every sample = finite difference of the reference velocity +
  N(0, imu_sigma_m_s2) (IMU stand-in: Measurement carries no specific force).

Estimators: ``hold`` (last GNSS fix, measured velocity), ``kalman``
(spatial.kinematic_kf constant acceleration), ``wmon`` (spatial.wmon.WMONStream).
The capture-time own position paired with each image is the estimator's own
history at the capture stamp, so the pad track sees the same navigation the
decision does.  Channels: drone_navigationValid := GNSS fix age < valid_age_s,
drone_navigationAge := decision time - last fix.  Same 21 channels, same order
and encodings as /3.  Truth, plant and verdict never change.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, replace
import math
from typing import Any

import numpy as np

from ..spatial.kinematic_kf import KalmanNoise, KinematicKalman
from ..spatial.wmon import MATLAB_TRAINED_THETA, WMONParams, WMONStream

LIVE_NAVIGATION_OBSERVATION_VERSION = "spatial-current-observation/4"
ESTIMATOR_KINDS = ("hold", "kalman", "wmon")
NAVIGATION_RULE = (
    "own position/velocity of the 21D vector and the capture-time own position of the pad "
    "track come from the navigation estimator; drone_navigationValid := GNSS fix age < "
    "valid_age_s; drone_navigationAge := decision time - last GNSS fix (inf before the first)")
_HISTORY_S = 2.0
_OUTAGE_STREAM = 0x6A55


@dataclass(frozen=True)
class NavigationDegradation:
    gnss_period_s: float = 0.1
    gnss_sigma_m: tuple[float, float, float] = (0.3, 0.3, 0.5)
    velocity_sigma_m_s: float = 0.05
    imu_sigma_m_s2: float = 0.2
    outage_probability: float = 1.0
    outage_start_min_s: float = 2.0
    outage_start_max_s: float = 6.0
    outage_duration_min_s: float = 1.0
    outage_duration_max_s: float = 4.0
    valid_age_s: float = 0.5

    def __post_init__(self) -> None:
        if len(self.gnss_sigma_m) != 3 or min(self.gnss_sigma_m) < 0:
            raise ValueError("gnss_sigma_m needs three non-negative values")
        for name in ("gnss_period_s", "valid_age_s"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive")
        for name in ("velocity_sigma_m_s", "imu_sigma_m_s2"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 0 <= self.outage_probability <= 1:
            raise ValueError("outage_probability must be in [0, 1]")
        if not (0 <= self.outage_start_min_s <= self.outage_start_max_s
                and 0 <= self.outage_duration_min_s <= self.outage_duration_max_s):
            raise ValueError("outage windows need 0 <= min <= max")

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["gnss_sigma_m"] = list(self.gnss_sigma_m)
        return out


@dataclass(frozen=True)
class NavigationEstimator:
    kind: str = "wmon"
    kalman: KalmanNoise | None = None
    wmon_theta: tuple[float, ...] = tuple(MATLAB_TRAINED_THETA)
    wmon_dt_ref: float | None = None
    wmon_bond_dt_exponent: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in ESTIMATOR_KINDS:
            raise ValueError(f"kind must be one of {ESTIMATOR_KINDS}")
        if self.kind == "kalman" and self.kalman is None:
            raise ValueError("a kalman estimator needs its KalmanNoise")

    def build(self) -> Any:
        if self.kind == "kalman":
            return _KalmanStep(KinematicKalman(self.kalman, 3))
        if self.kind == "wmon":
            return _WMONStep(WMONStream(WMONParams.from_theta(self.wmon_theta), 3,
                                        dt_ref=self.wmon_dt_ref,
                                        bond_dt_exponent=self.wmon_bond_dt_exponent))
        return _HoldStep()

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind}
        if self.kind == "kalman":
            out["kalman"] = self.kalman.to_dict()
        if self.kind == "wmon":
            out.update(wmon_theta=list(self.wmon_theta), wmon_dt_ref=self.wmon_dt_ref,
                       wmon_bond_dt_exponent=self.wmon_bond_dt_exponent)
        return out


_DEGRADATION_KEYS = tuple(NavigationDegradation.__dataclass_fields__)
_ESTIMATOR_KEYS = {"hold": ("kind",), "kalman": ("kind", "kalman"),
                   "wmon": ("kind", "wmon_theta", "wmon_dt_ref", "wmon_bond_dt_exponent")}
_KALMAN_KEYS = ("sigma_position", "sigma_velocity", "sigma_acceleration", "jerk_psd")


def _strict(data: Any, keys: tuple[str, ...], where: str) -> dict:
    if not isinstance(data, dict):
        raise ValueError(f"{where} must be a mapping")
    missing, extra = set(keys) - set(data), set(data) - set(keys)
    if missing or extra:
        raise ValueError(f"{where}: missing keys {sorted(missing)}, unknown keys {sorted(extra)}")
    return data


@dataclass(frozen=True)
class NavigationProfile:
    """The ``navigation`` section of a config: observation profile /4 for every stage.

    A config carrying it trains, evaluates and hashes (``contract_manifest``) under /4;
    every backend built from it (``make_local_backend``/``make_isaac_backend``) gets a
    fresh ``LiveNavigation`` of this profile.  A config without it is the registered /3.
    """
    degradation: NavigationDegradation
    estimator: NavigationEstimator

    def manifest_part(self) -> dict[str, Any]:
        return {"version": LIVE_NAVIGATION_OBSERVATION_VERSION, "rule": NAVIGATION_RULE,
                "degradation": self.degradation.to_dict(), "estimator": self.estimator.to_dict()}

    def build(self) -> "LiveNavigation":
        return LiveNavigation(self.degradation, self.estimator)

    @classmethod
    def from_mapping(cls, data: Any) -> "NavigationProfile":
        """Strict parse of the YAML section: every key required, none extra."""
        data = _strict(data, ("profile", "degradation", "estimator"), "navigation")
        if data["profile"] != LIVE_NAVIGATION_OBSERVATION_VERSION:
            raise ValueError(f"navigation.profile must be {LIVE_NAVIGATION_OBSERVATION_VERSION!r}")
        deg = dict(_strict(data["degradation"], _DEGRADATION_KEYS, "navigation.degradation"))
        deg["gnss_sigma_m"] = tuple(float(v) for v in deg["gnss_sigma_m"])
        degradation = NavigationDegradation(**{k: v if k == "gnss_sigma_m" else float(v)
                                               for k, v in deg.items()})
        est = data["estimator"]
        kind = est.get("kind") if isinstance(est, dict) else None
        if kind not in _ESTIMATOR_KEYS:
            raise ValueError(f"navigation.estimator.kind must be one of {ESTIMATOR_KINDS}")
        est = _strict(est, _ESTIMATOR_KEYS[kind], "navigation.estimator")
        if kind == "kalman":
            k = _strict(est["kalman"], _KALMAN_KEYS, "navigation.estimator.kalman")
            estimator = NavigationEstimator("kalman", kalman=KalmanNoise(
                **{key: float(k[key]) for key in _KALMAN_KEYS}))
        elif kind == "wmon":
            theta = est["wmon_theta"]
            if not isinstance(theta, list) or len(theta) != 7:
                raise ValueError("navigation.estimator.wmon_theta needs 7 numbers")
            estimator = NavigationEstimator(
                "wmon", wmon_theta=tuple(float(v) for v in theta),
                wmon_dt_ref=None if est["wmon_dt_ref"] is None else float(est["wmon_dt_ref"]),
                wmon_bond_dt_exponent=float(est["wmon_bond_dt_exponent"]))
        else:
            estimator = NavigationEstimator("hold")
        return cls(degradation, estimator)


class _HoldStep:
    def __init__(self) -> None:
        self.fix: np.ndarray | None = None

    def __call__(self, gnss, velocity, acceleration, dt):
        if gnss is not None:
            self.fix = gnss
        return self.fix.copy(), velocity.copy()


class _KalmanStep:
    def __init__(self, kf: KinematicKalman) -> None:
        self.kf = kf

    def __call__(self, gnss, velocity, acceleration, dt):
        x = self.kf.step(np.zeros(3) if gnss is None else gnss, velocity, acceleration,
                         gnss is not None, dt)
        return x[:, 0].copy(), x[:, 1].copy()


class _WMONStep:
    def __init__(self, stream: WMONStream) -> None:
        self.stream = stream

    def __call__(self, gnss, velocity, acceleration, dt):
        s = self.stream.step(np.zeros(3) if gnss is None else gnss, velocity, acceleration,
                             gnss is not None, dt)
        return s[:3].copy(), s[3:6].copy()


class LiveNavigation:
    """Per episode: degrade the flight's own state and replace it by an estimate."""

    def __init__(self, degradation: NavigationDegradation | None = None,
                 estimator: NavigationEstimator | None = None) -> None:
        self.degradation = NavigationDegradation() if degradation is None else degradation
        self.estimator = NavigationEstimator() if estimator is None else estimator
        self.reset(0)

    @property
    def profile(self) -> NavigationProfile:
        return NavigationProfile(self.degradation, self.estimator)

    def manifest_part(self) -> dict[str, Any]:
        """What ``contracts.observation_manifest(navigation=...)`` hashes."""
        return self.profile.manifest_part()

    def reset(self, seed: int) -> None:
        self.seed = int(seed)
        spec = self.degradation
        rng = np.random.default_rng([self.seed, _OUTAGE_STREAM])
        start = float(rng.uniform(spec.outage_start_min_s, spec.outage_start_max_s))
        duration = float(rng.uniform(spec.outage_duration_min_s, spec.outage_duration_max_s))
        self.outage = (start, duration) if rng.uniform() < spec.outage_probability else None
        self.noise = np.random.default_rng([self.seed, _OUTAGE_STREAM, 1])
        self.step = self.estimator.build()
        self.t0: float | None = None
        self.last_t: float | None = None
        self.last_reference_velocity: np.ndarray | None = None
        self.last_fix_t = -math.inf
        self.fixes = 0
        self.history: deque[tuple[float, np.ndarray]] = deque()
        self.position = self.velocity = None

    def in_outage(self, t: float) -> bool:
        if self.outage is None or self.t0 is None:
            return False
        start, duration = self.outage
        return 0.0 <= t - self.t0 - start < duration

    def _at(self, t: float) -> np.ndarray:
        times = [h[0] for h in self.history]
        if t <= times[0]:
            return self.history[0][1].copy()
        if t >= times[-1]:
            return self.history[-1][1].copy()
        j = int(np.searchsorted(times, t))
        (ta, pa), (tb, pb) = self.history[j - 1], self.history[j]
        w = (t - ta) / (tb - ta)
        return (1 - w) * pa + w * pb

    def __call__(self, m: Any) -> Any:
        spec = self.degradation
        t = float(m.time_s)
        p_ref = np.asarray(m.own_position, dtype=np.float64)
        v_ref = np.asarray(m.own_velocity, dtype=np.float64)
        first = self.t0 is None
        if first:
            self.t0 = t
        dt = 0.0 if self.last_t is None else t - self.last_t
        if not first and dt > 1e-12:
            acc = (v_ref - self.last_reference_velocity) / dt
            gnss = None
            due = t - self.last_fix_t >= spec.gnss_period_s - 1e-9
            if due and not self.in_outage(t):
                gnss = p_ref + self.noise.normal(0.0, 1.0, 3) * np.asarray(spec.gnss_sigma_m)
                self.last_fix_t, self.fixes = t, self.fixes + 1
            velocity = v_ref + self.noise.normal(0.0, spec.velocity_sigma_m_s, 3)
            acceleration = acc + self.noise.normal(0.0, spec.imu_sigma_m_s2, 3)
            self.position, self.velocity = self.step(gnss, velocity, acceleration, dt)
        elif first:
            # the first sample is a fix (handover needs a navigation solution)
            gnss = p_ref + self.noise.normal(0.0, 1.0, 3) * np.asarray(spec.gnss_sigma_m)
            velocity = v_ref + self.noise.normal(0.0, spec.velocity_sigma_m_s, 3)
            self.last_fix_t, self.fixes = t, 1
            self.position, self.velocity = self.step(gnss, velocity, np.zeros(3), 0.0)
        if first or dt > 1e-12:
            self.last_t, self.last_reference_velocity = t, v_ref.copy()
            self.history.append((t, self.position.copy()))
            while self.history and self.history[0][0] < t - _HISTORY_S:
                self.history.popleft()
        changes = {"own_position": self.position.copy(), "own_velocity": self.velocity.copy()}
        if m.optical_own_position is not None and m.optical_time_s is not None:
            changes["optical_own_position"] = self._at(float(m.optical_time_s))
        return replace(m, **changes)

    def status(self, now: float) -> tuple[bool, float]:
        age = float(now) - self.last_fix_t
        return age < self.degradation.valid_age_s, age

    def provenance(self, now: float) -> dict[str, Any]:
        valid, age = self.status(now)
        return {"version": LIVE_NAVIGATION_OBSERVATION_VERSION,
                "estimator": self.estimator.kind, "valid": valid,
                "gnss_age_s": age if math.isfinite(age) else None, "fixes": self.fixes,
                "outage_s": None if self.outage is None else list(self.outage),
                "in_outage": self.in_outage(float(now))}
