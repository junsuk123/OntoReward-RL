#!/usr/bin/env python3
"""GNSS in a street canyon: blocked satellites, reflected ones, and what the
receiver can and cannot tell about the difference.

Isaac is not imported here, so the model is testable without a simulator, and
it takes its obstruction geometry from ``urban_scene.UrbanLayout`` -- the same
boxes the stage is built from -- so an outage always has a building you can see
in the viewport.

What is modelled
----------------
``Blockage``
    A satellite whose line of sight crosses a facade is not received directly.

``NLOS reception``
    A blocked satellite is often still tracked, via a signal reflected off the
    facade across the street. The reflected path is *longer* than the direct
    one, so its pseudorange carries a strictly positive excess delay of roughly
    ``2 d cos(el)`` for a receiver ``d`` from the reflector. Positive-only bias
    is the thing that makes urban GNSS error a bias and not noise, and it is
    why the horizontal error points across the street.

``Diffuse multipath``
    Even a direct signal arrives with a reflected copy superimposed, worth a
    few metres of correlated, elevation-dependent error. Modelled as a
    first-order Gauss-Markov process per satellite, so it wanders instead of
    flickering.

``Geometry``
    Losing the satellites behind the buildings leaves the survivors strung out
    along the street, which is a poor geometry as well as a smaller one: DOP is
    computed from the direction cosines of the satellites actually used, so the
    cross-street component of the error inflates on its own.

``Carrier-to-noise ratio``
    A reflected signal arrives attenuated, so its C/N0 is several dB below what
    its elevation says it should be. This is the receiver's one direct handle
    on NLOS -- it is what C/N0-based NLOS detection and shadow matching are
    built on -- and without it a set of satellites that are *all* reflected off
    facades the same distance away would look like a perfectly consistent fix,
    because their biases agree with each other.

``What the receiver knows``
    The position error is computed by weighted least squares from the per
    satellite errors, exactly as a receiver's own solution would be, and the
    post-fit residuals are kept and used to inflate its own covariance. The
    integrity figure the policy sees is built from satellite count, DOP, that
    inflated covariance and the C/N0 anomaly -- every one an observable. The
    true error and the true NLOS count are kept apart, under ``truth``, because
    no receiver has either and a policy that read them would not be solving
    this problem.

What is not modelled
--------------------
Carrier phase, RTK, or any correction service; the constellation moves
negligibly over a fifteen-second episode and is held fixed within one; and PX4's
own EKF is not corrupted (Pegasus' GPS sensor is not part of this workspace).
The modelled error is applied to the estimate the *gateway* hands the policy,
which is documented in docs/ARCHITECTURE.md, "GNSS".
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# A fix needs four satellites: three coordinates plus the receiver clock.
MIN_SATELLITES = 4


def _stable_seed(name: str) -> int:
    """A per-receiver seed offset that is the same in every process.

    ``hash()`` on a string is salted per interpreter, so deriving a seed from it
    makes the noise a receiver draws depend on which process drew it. Every
    other draw in this workspace is reproducible from the episode seed, and
    this one has to be too.
    """
    return int(zlib.crc32(name.encode("utf-8")) % 100_000)


@dataclass(frozen=True)
class GnssConfig:
    enabled: bool
    seed: int
    n_satellites: int
    elevation_mask_deg: float
    # Fraction of blocked satellites that are still tracked through a
    # reflection instead of being lost. Measured values in dense urban areas
    # sit high: a receiver keeps far more NLOS signals than it drops.
    nlos_tracking_probability: float
    max_excess_path_m: float
    # Diffuse multipath, at zenith, before the elevation weighting.
    multipath_sigma_m: float
    multipath_tau_s: float
    thermal_sigma_m: float
    # Carrier-to-noise ratio, in dB-Hz: at the zenith, how far it falls by the
    # horizon, how noisy it is, and what a reflection costs. The detection
    # margin is how far below its elevation's expectation a satellite has to be
    # before the receiver calls it suspect.
    cn0_zenith_db: float
    cn0_elevation_db: float
    cn0_sigma_db: float
    nlos_cn0_penalty_db: float
    cn0_detection_margin_db: float
    # Doppler is far less sensitive to a reflected path than the pseudorange.
    velocity_error_scale: float
    # How much of the vertical GNSS error survives into the estimate. A
    # multirotor EKF holds altitude mostly on the barometer, so a canyon's
    # (large) vertical GNSS error is largely rejected; the horizontal channel
    # has nothing else to lean on, which is why the urban failure mode is
    # horizontal.
    vertical_blend: float
    # Reported-integrity shaping.
    hdop_scale: float
    sigma_floor_m: float
    sigma_scale_m: float
    # How much of the integrity a fully suspect signal set costs.
    cn0_weight: float
    # Dead reckoning while there are too few satellites for a fix.
    outage_drift_m_s: float

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "GnssConfig":
        gnss = (data.get("gnss") or {}) if isinstance(data, dict) else {}
        return cls(
            enabled=bool(gnss.get("enabled", True)),
            seed=int(gnss.get("seed", 17)),
            n_satellites=int(gnss.get("n_satellites", 12)),
            elevation_mask_deg=float(gnss.get("elevation_mask_deg", 7.0)),
            nlos_tracking_probability=float(gnss.get("nlos_tracking_probability", 0.60)),
            max_excess_path_m=float(gnss.get("max_excess_path_m", 60.0)),
            multipath_sigma_m=float(gnss.get("multipath_sigma_m", 0.5)),
            multipath_tau_s=float(gnss.get("multipath_tau_s", 6.0)),
            thermal_sigma_m=float(gnss.get("thermal_sigma_m", 0.6)),
            cn0_zenith_db=float(gnss.get("cn0_zenith_db", 48.0)),
            cn0_elevation_db=float(gnss.get("cn0_elevation_db", 12.0)),
            cn0_sigma_db=float(gnss.get("cn0_sigma_db", 1.5)),
            nlos_cn0_penalty_db=float(gnss.get("nlos_cn0_penalty_db", 9.0)),
            cn0_detection_margin_db=float(gnss.get("cn0_detection_margin_db", 4.0)),
            velocity_error_scale=float(gnss.get("velocity_error_scale", 0.06)),
            vertical_blend=float(gnss.get("vertical_blend", 0.15)),
            hdop_scale=float(gnss.get("hdop_scale", 2.5)),
            sigma_floor_m=float(gnss.get("sigma_floor_m", 1.5)),
            sigma_scale_m=float(gnss.get("sigma_scale_m", 10.0)),
            cn0_weight=float(gnss.get("cn0_weight", 0.85)),
            outage_drift_m_s=float(gnss.get("outage_drift_m_s", 0.8)),
        )


@dataclass
class GnssFix:
    """One receiver's fix, with what it knows kept apart from what is true.

    :meth:`observables` is everything a real receiver publishes. Everything
    else -- the true error, the true NLOS count, the true sky view -- is the
    simulator's, and reaches the wire only under a ``truth`` key so that no
    consumer can pick it up by accident.
    """
    valid: bool = False
    fix_type: int = 0                     # 0 none, 2 2D, 3 3D
    satellites_tracked: int = 0
    hdop: float = 99.9
    vdop: float = 99.9
    residual_rms_m: float = 0.0
    sigma_xy_m: float = 0.0
    cn0_mean_db: float = 0.0
    # What the receiver's own C/N0 test flags as probably reflected. This is a
    # detector, so it is neither the true count nor free of false alarms.
    nlos_detected_fraction: float = 0.0
    quality: float = 1.0
    # --- truth ---------------------------------------------------------
    satellites_los: int = 0
    satellites_nlos: int = 0
    nlos_fraction: float = 0.0
    sky_view: float = 1.0
    error_enu_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity_error_enu_m_s: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def observables(self) -> dict[str, Any]:
        """Exactly what a receiver publishes about itself."""
        return {
            "valid": bool(self.valid),
            "fix_type": int(self.fix_type),
            "satellites_tracked": int(self.satellites_tracked),
            "hdop": float(self.hdop),
            "vdop": float(self.vdop),
            "residual_rms_m": float(self.residual_rms_m),
            "sigma_xy_m": float(self.sigma_xy_m),
            "cn0_mean_db": float(self.cn0_mean_db),
            "nlos_detected_fraction": float(self.nlos_detected_fraction),
            "quality": float(self.quality),
        }

    def to_dict(self) -> dict[str, Any]:
        """Observables plus the simulator's truth, the latter clearly labelled.

        The truth crosses the simulator-to-gateway wire because the gateway is
        what injects the error into the estimate: Pegasus' GPS sensor is not in
        this workspace, so it cannot be applied upstream of PX4's EKF. The
        gateway forwards the observables to the policy and keeps the rest.
        """
        out = self.observables()
        out["truth"] = {
            "satellites_los": int(self.satellites_los),
            "satellites_nlos": int(self.satellites_nlos),
            "nlos_fraction": float(self.nlos_fraction),
            "sky_view": float(self.sky_view),
            "error_enu_m": [float(v) for v in self.error_enu_m],
            "velocity_error_enu_m_s": [float(v) for v in self.velocity_error_enu_m_s],
        }
        return out


def neutral_fix() -> dict[str, Any]:
    """What a link that models no GNSS at all must report.

    Every number is finite and the quality is one, so a consumer that has no
    GNSS information behaves like the open-sky control condition rather than
    reading an absent field as a total outage.
    """
    return GnssFix(valid=True, fix_type=3, satellites_tracked=MIN_SATELLITES + 4,
                   satellites_los=MIN_SATELLITES + 4, hdop=1.0, vdop=1.6,
                   sigma_xy_m=0.0, cn0_mean_db=45.0, quality=1.0).to_dict()


class Constellation:
    """Where the satellites are, for one episode.

    Elevations are drawn uniformly in ``sin(el)`` above the mask, which is the
    distribution that puts satellites uniformly over the visible hemisphere;
    drawing elevation uniformly instead would over-populate the zenith and
    quietly make every canyon look better than it is. The geometry is frozen
    for the episode: fifteen seconds moves a MEO satellite by well under a
    degree, so re-drawing it per step would be noise, not motion.
    """

    def __init__(self, cfg: GnssConfig):
        self.cfg = cfg
        self.azimuth = np.zeros(0)
        self.elevation = np.zeros(0)
        self.reflection = np.zeros(0)
        self.nlos_penalty = np.zeros(0)
        self.reset(cfg.seed)

    def reset(self, seed: int) -> None:
        cfg = self.cfg
        rng = np.random.default_rng(int(seed) + 5171)
        n = max(MIN_SATELLITES, int(cfg.n_satellites))
        mask = math.radians(cfg.elevation_mask_deg)
        self.azimuth = rng.uniform(0.0, 2.0 * math.pi, size=n)
        self.elevation = np.arcsin(rng.uniform(math.sin(mask), 1.0, size=n))
        # Whether a blocked signal still arrives via a reflection strong enough
        # to hold lock is a property of the facades, not of the receiver, so it
        # is drawn once for the constellation and shared. Two receivers a few
        # metres apart in the same street therefore lose and keep the same
        # satellites -- which is what makes their errors common-mode, and why
        # the drone knows where the lorry is far better than it knows where
        # either of them is.
        self.reflection = rng.uniform(size=n)
        # How much a given reflection costs in signal strength. Drawn per
        # satellite because it depends on the facade -- glass and metal reflect
        # far better than brick -- and drawn exponentially because most
        # reflections are weak and a few are nearly as strong as the direct
        # path. The strong ones are the dangerous ones: they carry the full
        # excess delay and the C/N0 test cannot see them.
        self.nlos_penalty = rng.exponential(size=n)

    @property
    def size(self) -> int:
        return int(self.azimuth.size)

    def line_of_sight_unit(self) -> np.ndarray:
        """Receiver-to-satellite unit vectors in ENU, one row per satellite."""
        ce = np.cos(self.elevation)
        return np.stack([ce * np.cos(self.azimuth), ce * np.sin(self.azimuth),
                         np.sin(self.elevation)], axis=1)


class GnssReceiver:
    """One receiver in the canyon: the drone's, or the truck's.

    The two share a :class:`Constellation`, so the satellites they lose are
    correlated and the errors they make are partly common-mode -- which is why
    the *relative* position of the drone and the deck stays better than either
    absolute position, the same way a moving-baseline setup behaves.
    """

    def __init__(self, cfg: GnssConfig, constellation: Constellation, name: str):
        self.cfg = cfg
        self.sky = constellation
        self.name = name
        self.rng = np.random.default_rng(_stable_seed(name))
        self.multipath = np.zeros(constellation.size)
        self.outage_drift = np.zeros(3)
        self.last = GnssFix(valid=True, quality=1.0)
        self.scale = 1.0

    def reset(self, seed: int, scale: float = 1.0) -> None:
        """Reseed for a new episode.

        SCALE multiplies the error mechanisms -- reflection tracking, excess
        delay and diffuse multipath -- so a sweep can ask how much GNSS
        degradation a policy survives without moving the buildings, and 0.0 is
        the open-sky control condition.
        """
        self.rng = np.random.default_rng(int(seed) + _stable_seed(self.name))
        self.multipath = self.cfg.multipath_sigma_m * self.rng.standard_normal(self.sky.size)
        self.outage_drift = np.zeros(3)
        self.scale = max(float(scale), 0.0)
        self.last = GnssFix(valid=True, quality=1.0)

    def update(self, position, layout, dt: float) -> GnssFix:
        """The fix at POSITION, given the buildings in LAYOUT."""
        cfg = self.cfg
        if not cfg.enabled or self.scale <= 0.0 or layout is None:
            self.last = GnssFix(valid=True, fix_type=3,
                                satellites_tracked=self.sky.size,
                                satellites_los=self.sky.size,
                                hdop=1.0, vdop=1.6,
                                cn0_mean_db=self.cfg.cn0_zenith_db - 4.0,
                                quality=1.0)
            return self.last

        self._step_multipath(dt)
        blocked, distance = layout.blocked_batch(position, self.sky.azimuth,
                                                 self.sky.elevation)
        # A blocked satellite is usually still tracked, through the facade
        # opposite. Which ones survive is redrawn slowly rather than every
        # step, so a signal does not flicker in and out at the update rate.
        tracked = ~blocked | (self.sky.reflection < cfg.nlos_tracking_probability)
        nlos = tracked & blocked
        used = np.flatnonzero(tracked)
        sky_view = float(np.mean(~blocked)) if blocked.size else 1.0

        if used.size < MIN_SATELLITES:
            return self._outage(dt, int(tracked.sum()), int(nlos.sum()), sky_view)

        unit = self.sky.line_of_sight_unit()[used]
        elevation = self.sky.elevation[used]
        # Excess path of a single reflection off a facade a perpendicular
        # DISTANCE away: the receiver and its mirror image are twice that
        # apart, and the extra length is foreshortened by the elevation.
        # Capped, because a path with several bounces is too weak to track.
        horizontal = distance[used] * np.cos(elevation)
        excess = np.zeros(used.size)
        is_nlos = nlos[used]
        excess[is_nlos] = np.minimum(
            2.0 * horizontal[is_nlos] * np.cos(elevation[is_nlos]),
            cfg.max_excess_path_m)
        # Range accuracy per satellite: thermal noise and diffuse multipath
        # both grow as the signal comes in flatter, which is the standard
        # 1/sin(el) weighting. The floor stops a satellite at the mask angle
        # from dominating the solution by sheer variance.
        sin_el = np.maximum(np.sin(elevation), 0.20)
        sigma = np.hypot(cfg.thermal_sigma_m, cfg.multipath_sigma_m) / sin_el
        error = (excess
                 + self.multipath[used] / sin_el
                 + cfg.thermal_sigma_m * self.rng.standard_normal(used.size) / sin_el)
        error *= self.scale

        # Weighted least squares, exactly as the receiver solves it: rows are
        # [-u, 1] because a pseudorange grows when the receiver moves away from
        # the satellite, and the fourth column is the receiver clock.
        design = np.concatenate([-unit, np.ones((used.size, 1))], axis=1)
        weight = 1.0 / sigma**2
        try:
            # Unweighted, so the reported DOP is the classical geometry figure
            # and not something in units of an assumed range accuracy.
            geometry = np.linalg.inv(design.T @ design)
            covariance = np.linalg.inv(design.T @ (weight[:, None] * design))
        except np.linalg.LinAlgError:                    # pragma: no cover - degenerate
            return self._outage(dt, int(tracked.sum()), int(nlos.sum()), sky_view)
        solution = covariance @ (design.T @ (weight * error))
        residual = error - design @ solution
        # The estimate keeps only a fraction of the vertical component: see
        # GnssConfig.vertical_blend.
        offset = np.asarray(solution[:3], dtype=float) * np.array(
            [1.0, 1.0, cfg.vertical_blend])

        hdop = float(math.sqrt(max(geometry[0, 0] + geometry[1, 1], 0.0)))
        vdop = float(math.sqrt(max(geometry[2, 2], 0.0)))
        residual_rms = float(math.sqrt(float(np.mean(residual ** 2))))
        cn0, suspect = self._carrier_to_noise(used, elevation, is_nlos)
        # The accuracy the receiver publishes: its own formal covariance,
        # rescaled by how badly the ranges disagree with each other. That
        # a-posteriori variance factor is the receiver's only handle on
        # multipath, and it still understates a bias the satellites share --
        # which is exactly the urban failure mode worth reproducing.
        variance_factor = (float(residual @ (weight * residual))
                           / max(used.size - 4, 1)) if used.size > 4 else 1.0
        inflation = math.sqrt(max(variance_factor, 1.0))
        sigma_xy = inflation * math.sqrt(max(covariance[0, 0] + covariance[1, 1], 0.0))

        self.outage_drift = np.zeros(3)
        self.last = GnssFix(
            valid=True, fix_type=3,
            satellites_tracked=int(used.size),
            satellites_los=int(used.size - is_nlos.sum()),
            satellites_nlos=int(is_nlos.sum()),
            nlos_fraction=float(is_nlos.mean()),
            hdop=hdop, vdop=vdop, residual_rms_m=residual_rms,
            sigma_xy_m=float(sigma_xy), sky_view=sky_view,
            cn0_mean_db=float(np.mean(cn0)),
            nlos_detected_fraction=float(np.mean(suspect)),
            quality=self._integrity(used.size, hdop, sigma_xy, float(np.mean(suspect))),
            error_enu_m=offset,
            velocity_error_enu_m_s=cfg.velocity_error_scale * offset,
        )
        return self.last

    # ------------------------------------------------------------ internals
    def _step_multipath(self, dt: float) -> None:
        """First-order Gauss-Markov: correlated in time, stationary in spread."""
        cfg = self.cfg
        tau = max(cfg.multipath_tau_s, 1e-3)
        beta = math.exp(-max(dt, 0.0) / tau)
        driving = cfg.multipath_sigma_m * math.sqrt(max(1.0 - beta * beta, 0.0))
        self.multipath = beta * self.multipath + driving * self.rng.standard_normal(
            self.multipath.size)

    def _outage(self, dt: float, tracked: int, nlos: int, sky_view: float) -> GnssFix:
        """Too few satellites for a fix: the estimate coasts and drifts."""
        cfg = self.cfg
        step = cfg.outage_drift_m_s * max(dt, 0.0) * self.rng.standard_normal(3)
        step[2] *= 0.5
        self.outage_drift = self.outage_drift + step
        error = np.asarray(self.last.error_enu_m, dtype=float) + self.outage_drift
        self.last = GnssFix(
            valid=False, fix_type=0, satellites_tracked=tracked,
            satellites_los=max(tracked - nlos, 0), satellites_nlos=nlos,
            nlos_fraction=float(nlos / tracked) if tracked else 1.0,
            hdop=99.9, vdop=99.9, residual_rms_m=self.last.residual_rms_m,
            sigma_xy_m=99.9, cn0_mean_db=0.0,
            nlos_detected_fraction=1.0 if tracked else 0.0,
            sky_view=sky_view, quality=0.0,
            error_enu_m=error,
            velocity_error_enu_m_s=np.zeros(3))
        return self.last

    def _carrier_to_noise(self, used: np.ndarray, elevation: np.ndarray,
                          is_nlos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Signal strength per satellite, and which ones look wrong for it.

        C/N0 falls off toward the horizon, and a reflection costs several dB on
        top of that. The receiver knows the elevation of every satellite it is
        tracking, so a signal well below what its elevation predicts is
        evidence of a reflected path -- the one piece of evidence that survives
        when every satellite is reflected off a facade the same distance away
        and their biases therefore agree with one another.
        """
        cfg = self.cfg
        expected = cfg.cn0_zenith_db - cfg.cn0_elevation_db * (1.0 - np.sin(elevation))
        penalty = cfg.nlos_cn0_penalty_db * self.sky.nlos_penalty[used]
        cn0 = expected - np.where(is_nlos, penalty, 0.0)
        cn0 = cn0 + cfg.cn0_sigma_db * self.rng.standard_normal(elevation.size)
        return cn0, cn0 < expected - cfg.cn0_detection_margin_db

    def _integrity(self, used: int, hdop: float, sigma_xy: float,
                   suspect_fraction: float) -> float:
        """The integrity figure the policy is allowed to see.

        Built only from observables: how many satellites are in the solution,
        how well they are spread, how far the receiver's own inflated
        covariance says its horizontal position could be out, and how many of
        its signals are too weak for the elevation they claim. A receiver that
        could see its true error would not need any of this, and neither would
        the ontology.
        """
        cfg = self.cfg
        redundancy = min(1.0, max(0.0, (used - MIN_SATELLITES) / 4.0))
        geometry = math.exp(-max(hdop - 1.0, 0.0) / max(cfg.hdop_scale, 1e-6))
        accuracy = math.exp(-max(sigma_xy - cfg.sigma_floor_m, 0.0)
                            / max(cfg.sigma_scale_m, 1e-6))
        signal = max(1.0 - cfg.cn0_weight * float(suspect_fraction), 0.0)
        return float(np.clip((0.25 + 0.75 * redundancy) * geometry * accuracy * signal,
                             0.0, 1.0))


class UrbanGnss:
    """Both receivers plus the constellation they share."""

    def __init__(self, cfg: GnssConfig, layout=None):
        self.cfg = cfg
        self.layout = layout
        self.constellation = Constellation(cfg)
        self.uav = GnssReceiver(cfg, self.constellation, "uav")
        self.deck = GnssReceiver(cfg, self.constellation, "deck")

    def reset(self, seed: int, scale: float = 1.0) -> None:
        self.constellation.reset(seed)
        self.uav.reset(seed, scale)
        self.deck.reset(seed + 1, scale)

    def update(self, uav_position, deck_position, dt: float) -> tuple[GnssFix, GnssFix]:
        return (self.uav.update(uav_position, self.layout, dt),
                self.deck.update(deck_position, self.layout, dt))
