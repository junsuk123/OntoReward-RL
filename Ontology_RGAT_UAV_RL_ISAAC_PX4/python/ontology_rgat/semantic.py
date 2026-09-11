"""Semantic features and the ontology graph they populate.

Port of the retired ``+semantic`` package. The feature encoding and the
scalings are unchanged from it; what has changed is the schema, which now
carries the three channels the external environment introduced -- the moving
deck, the energy budget and, since the experiment moved into a street canyon,
the integrity of the fix the pad-relative pose falls back on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import Config

__all__ = ["SemanticState", "OntologyGraph", "compute_features",
           "build_ontology_graph", "make_observation", "GOAL_NODE", "RISK_NODES",
           "N_NODES"]


# SafeLanding. It is node 14 because PadMotion, BatteryReserve and
# GnssIntegrity come before it; a model trained against a shorter schema is
# rejected on the dimension rather than silently attending over the wrong
# nodes.
GOAL_NODE = 13          # zero-based
# The nodes where more is worse. GnssIntegrity is an enabler like
# MarkerQuality, so it is deliberately not one of them.
RISK_NODES = (0, 1, 2, 3, 4, 10)

N_NODES = 14


@dataclass
class SemanticState:
    """The semantic channels one control step exposes."""
    position_error: float = 0.0
    vertical_speed: float = 0.0
    tilt: float = 0.0
    angular_rate: float = 0.0
    wind_risk: float = 0.0
    marker_quality: float = 0.0
    visual_stability: float = 0.0
    alignment: float = 0.0
    attitude_stability: float = 0.0
    touchdown_safety: float = 0.0
    pad_motion: float = 0.0
    battery_reserve: float = 1.0
    gnss_integrity: float = 1.0
    # Diagnostics, logged but not fed to the graph.
    mean_wind: np.ndarray = field(default_factory=lambda: np.zeros(3))
    wind_speed: float = 0.0
    wind_accel: float = 0.0
    wind_dir_change: float = 0.0
    r_drag: float = 0.0
    r_speed: float = 0.0
    r_acc: float = 0.0
    r_dir: float = 0.0
    closing_speed: float = 0.0
    pad_speed: float = 0.0
    gnss_sigma_xy: float = 0.0
    gnss_nlos_fraction: float = 0.0
    gnss_satellites: float = 0.0
    nav_confidence: float = 1.0
    energy_margin: float = 1.0
    energy_needed_s: float = 0.0
    hover_seconds_remaining: float = 0.0

    @property
    def node_values(self) -> np.ndarray:
        """The 14 node activations, in schema order."""
        return np.array([
            self.position_error, self.vertical_speed, self.tilt,
            self.angular_rate, self.wind_risk, self.marker_quality,
            self.visual_stability, self.alignment, self.attitude_stability,
            self.touchdown_safety, self.pad_motion, self.battery_reserve,
            self.gnss_integrity, 0.0,
        ])


@dataclass(frozen=True)
class OntologyGraph:
    """One graph sample: fixed topology, state-dependent node features."""
    X: np.ndarray                  # [in_dim x n_nodes]
    src: np.ndarray                # zero-based
    dst: np.ndarray                # zero-based
    rel: np.ndarray                # zero-based relation ids
    goal_node: int
    node_names: tuple[str, ...]
    relation_names: tuple[str, ...]

    def with_features(self, X: np.ndarray) -> "OntologyGraph":
        return OntologyGraph(X, self.src, self.dst, self.rel, self.goal_node,
                             self.node_names, self.relation_names)


def _pad_velocity(diag: dict[str, Any]) -> np.ndarray:
    v = np.asarray(diag.get("pad_velocity_i", (0.0, 0.0, 0.0)), dtype=float)
    if v.size < 2:
        return np.zeros(2)
    return v.reshape(-1)[:2]


def _gnss(diag: dict[str, Any]) -> dict[str, Any]:
    """The receiver's view, with the open-sky default.

    A link that models no GNSS has to read as open sky and not as an outage,
    the same way a link that reports no energy must not read as an empty pack.
    """
    base = {"enabled": False, "quality": 1.0, "deck_quality": 1.0,
            "sigma_xy_m": 0.0, "deck_sigma_xy_m": 0.0,
            "nlos_detected_fraction": 0.0, "cn0_mean_db": 45.0,
            "satellites_tracked": 12, "fix_type": 3, "valid": True}
    live = diag.get("gnss")
    if isinstance(live, dict):
        base.update(live)
    return base


def _battery(diag: dict[str, Any]) -> dict[str, Any]:
    """The battery view with a safe default.

    A link that reports no energy (the MAVLink fallback) must leave every
    energy channel neutral rather than read as an empty pack.
    """
    base = {"enabled": False, "hover_seconds_remaining": 0.0,
            "landing_reserve_s": 0.0, "reserve": 1.0}
    live = diag.get("battery")
    if isinstance(live, dict):
        base.update(live)
    return base


def compute_features(diag: dict[str, Any], meas: dict[str, Any],
                     prev: SemanticState | None, cfg: Config) -> SemanticState:
    """Semantic states following the landing-paper concepts.

    Three facts the in-process simulator did not have change what "landing
    safely" means, so they become semantic channels rather than being smuggled
    into the observation vector alone:

    ``PadMotion``
        The pad rides a ground vehicle, so the target moves and the horizontal
        velocity still to be cancelled is part of the state, not a nuisance.
    ``BatteryReserve``
        The pack starts each episode nearly empty, so whether there is energy
        left to finish a descent competes with doing it precisely.
    ``GnssIntegrity``
        The landing happens in a street canyon, where satellites are blocked by
        the facades and the ones that survive are often reflections carrying
        tens of metres of excess delay. When the markers are out of frame the
        pad-relative pose is a difference of two such fixes, so how much that
        pose can be trusted is part of the state rather than an assumption.
        The channel is built only from what a receiver actually publishes --
        satellite count, geometry and its own inflated covariance -- never from
        the true error, which no receiver has.

    Every position and velocity in ``meas`` is already pad-relative; the
    gateway expresses them that way (docs/ARCHITECTURE.md, "Coordinates").
    """
    pos = np.asarray(meas["pos"], dtype=float)
    vel = np.asarray(meas["vel"], dtype=float)
    rpy = np.asarray(meas["rpy"], dtype=float)
    omega = np.asarray(meas["omega"], dtype=float)

    pos_err = float(np.linalg.norm(pos[:2]))
    vz = float(vel[2])
    tilt = float(np.linalg.norm(rpy[:2]))
    rate = float(np.linalg.norm(omega))
    # Pad-relative, so this is the speed that must go to zero.
    closing_speed = float(np.linalg.norm(vel[:2]))

    # ---- wind risk ------------------------------------------------------
    f_cap = max(1.0, cfg.drone.max_total_thrust * max(np.cos(tilt), 0.1)
                - cfg.drone.mass * cfg.sim.g)
    observed_force = float(diag.get(
        "aero_force_observed_mag", diag.get("aero_force_mag", 0.0)))
    r_drag = min(1.0, observed_force / f_cap)
    mean_wind = np.asarray(diag.get("mean_wind_i", (0.0, 0.0, 0.0)), dtype=float)
    wind_speed = float(np.linalg.norm(mean_wind))
    r_speed = min(1.0, wind_speed / max(cfg.semantic.wind_speed_thr, 1e-6))
    if prev is None:
        a_wind = 0.0
        d_theta = 0.0
    else:
        a_wind = float(np.linalg.norm(mean_wind - prev.mean_wind) / cfg.sim.dt)
        n1 = float(np.linalg.norm(mean_wind))
        n0 = float(np.linalg.norm(prev.mean_wind))
        if n1 < 1e-8 or n0 < 1e-8:
            d_theta = 0.0
        else:
            c = float(np.clip(np.dot(mean_wind, prev.mean_wind) / (n1 * n0), -1.0, 1.0))
            d_theta = float(np.arccos(c))
    r_acc = min(1.0, a_wind / cfg.semantic.wind_accel_thr)
    r_dir = min(1.0, d_theta / cfg.semantic.wind_dir_thr)
    z = float(np.dot(cfg.semantic.wind_risk_w,
                     [r_speed, r_drag, r_acc, r_dir])) + cfg.semantic.wind_risk_b
    wind_risk = 1.0 / (1.0 + np.exp(-z))

    marker_quality = float(meas["marker_quality"])
    alignment = float(np.exp(-pos_err / cfg.semantic.align_scale))
    attitude_stability = float(np.exp(-tilt / cfg.semantic.att_tilt_scale
                                     - rate / cfg.semantic.att_rate_scale))
    visual_stability = float(np.clip(marker_quality, 0.0, 1.0))

    # ---- GNSS integrity ---------------------------------------------------
    # The pad-relative pose that the fallback path produces is a difference of
    # the drone's fix and the lorry's broadcast, so the weaker of the two
    # bounds it. No fix at all is zero whatever the two qualities say.
    gnss = _gnss(diag)
    gnss_integrity = float(np.clip(
        min(float(gnss["quality"]), float(gnss["deck_quality"])), 0.0, 1.0))
    if not bool(gnss.get("valid", True)):
        gnss_integrity = 0.0
    # Position knowledge comes from whichever source is working, so the two
    # combine as a noisy-OR rather than a product: good markers rescue a bad
    # fix, and a good fix carries the seconds after the pad leaves the frame.
    nav_confidence = 1.0 - (1.0 - visual_stability) * (1.0 - gnss_integrity)

    # ---- pad motion -----------------------------------------------------
    # Two independent contributions, both saturating in [0,1]: how hard the
    # deck is driving, and how far the vehicle is from being able to touch
    # down on it. A stationary deck with a matched vehicle gives exactly zero.
    rel_tol = max(cfg.criteria.rel_speed_xy, 1e-6)
    pad_speed = float(np.linalg.norm(_pad_velocity(diag)))
    pad_speed_ratio = min(1.0, pad_speed / max(cfg.semantic.pad_speed_scale, 1e-6))
    closing_ratio = min(1.0, closing_speed / rel_tol)
    pad_motion = min(1.0, 0.5 * pad_speed_ratio + 0.5 * closing_ratio)

    # ---- energy ---------------------------------------------------------
    # hover_seconds_remaining is the pack's own currency: how long it could
    # hold a hover. The margin is what is left after paying for the descent
    # still to fly plus the reserve the vehicle should touch down with,
    # expressed in units of the episode horizon so it is comparable with the
    # other normalized channels.
    batt = _battery(diag)
    if not batt["enabled"]:
        energy_margin = 1.0
        battery_reserve = 1.0
        energy_needed = 0.0
    else:
        energy_needed = (max(float(pos[2]), 0.0)
                         / max(cfg.battery.plan_descent_rate, 1e-6)
                         + float(batt["landing_reserve_s"]))
        energy_margin = ((float(batt["hover_seconds_remaining"]) - energy_needed)
                         / max(cfg.sim.max_time, 1e-6))
        energy_margin = float(np.clip(energy_margin, -3.0, 3.0))
        battery_reserve = float(np.clip(
            0.5 + 0.5 * energy_margin / max(cfg.semantic.energy_scale, 1e-6), 0.0, 1.0))

    # ---- touchdown safety ----------------------------------------------
    # The original product with the horizontal closing speed folded in:
    # arriving on the deck with velocity it does not share tips the airframe
    # over whether the deck is moving or not, so this applies to the static
    # control condition too. Visual stability has been replaced by the combined
    # navigation confidence, because in a canyon the question is not whether
    # the markers are visible but whether *anything* knows where the pad is.
    touchdown_safety = (alignment * attitude_stability * nav_confidence
                        * float(np.exp(-abs(vz) / cfg.semantic.vz_safe_scale))
                        * (1.0 - wind_risk)
                        * float(np.exp(-closing_speed / rel_tol))
                        * battery_reserve)

    return SemanticState(
        position_error=pos_err, vertical_speed=vz, tilt=tilt, angular_rate=rate,
        wind_risk=wind_risk, marker_quality=marker_quality,
        visual_stability=visual_stability, alignment=alignment,
        attitude_stability=attitude_stability, touchdown_safety=touchdown_safety,
        pad_motion=pad_motion, battery_reserve=battery_reserve,
        gnss_integrity=gnss_integrity,
        mean_wind=mean_wind, wind_speed=wind_speed,
        wind_accel=a_wind, wind_dir_change=d_theta,
        r_speed=r_speed, r_drag=r_drag, r_acc=r_acc, r_dir=r_dir,
        closing_speed=closing_speed, pad_speed=pad_speed,
        gnss_sigma_xy=float(max(gnss["sigma_xy_m"], gnss["deck_sigma_xy_m"])),
        gnss_nlos_fraction=float(gnss["nlos_detected_fraction"]),
        gnss_satellites=float(gnss["satellites_tracked"]),
        nav_confidence=float(nav_confidence),
        energy_margin=energy_margin, energy_needed_s=energy_needed,
        hover_seconds_remaining=float(batt["hover_seconds_remaining"]),
    )


def _edge_lists() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The fixed ontology edge list, zero-based.

    Relation ids: 0 degrades, 1 supports, 2 contributes, 3 self.
    The first block is the original schema with SafeLanding renumbered
    11 -> 14; the second is what the moving deck and the energy budget add;
    the third is what the street canyon adds.
    """
    # 1-based, transcribed from the retired semantic.buildOntologyGraph.
    src = [1, 5, 4, 3, 6, 7, 8, 9, 2, 5, 10, 7, 8, 9, 5]
    dst = [8, 9, 9, 9, 7, 8, 10, 10, 10, 10, 14, 14, 14, 14, 14]
    rel = [1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 3, 3, 3, 3, 1]
    # PadMotion degrades alignment, the pose it is solved from, the touchdown
    # itself, and the goal; BatteryReserve supports the touchdown and
    # contributes to the goal the way the other enablers do.
    src += [11, 11, 11, 11, 12, 12]
    dst += [8, 7, 10, 14, 10, 14]
    rel += [1, 1, 1, 1, 2, 3]
    # GnssIntegrity is the second source of the pad-relative pose, so it
    # supports exactly what MarkerQuality supports -- the alignment solved from
    # that pose and the touchdown flown on it -- and contributes to the goal.
    # It is the substitutability of these two that the relation weights have to
    # learn: with the markers in frame the fix hardly matters, and the moment
    # they leave it the fix is all there is.
    src += [13, 13, 13]
    dst += [8, 10, 14]
    rel += [2, 2, 3]
    n_nodes = 14
    for i in range(1, n_nodes + 1):
        src.append(i)
        dst.append(i)
        rel.append(4)
    return (np.asarray(src, dtype=np.int64) - 1,
            np.asarray(dst, dtype=np.int64) - 1,
            np.asarray(rel, dtype=np.int64) - 1)


_SRC, _DST, _REL = _edge_lists()


def node_features(values: np.ndarray, in_dim: int) -> np.ndarray:
    """``[in_dim x n_nodes]`` features: (value, 1-value, risk flag, bias, one-hot)."""
    values = np.asarray(values, dtype=float).reshape(-1)
    n = values.size
    X = np.zeros((in_dim, n))
    X[0, :] = values
    X[1, :] = 1.0 - values
    X[2, list(RISK_NODES)] = 1.0
    X[3, :] = 1.0
    X[4:4 + n, :] = np.eye(n)
    return X


def build_ontology_graph(sem: SemanticState, cfg: Config) -> OntologyGraph:
    """Fixed semantic schema plus state-dependent node features.

    Nodes retain domain meaning; R-GAT learns context-dependent relation
    weights. SafeLanding carries no value of its own, so no future label leaks
    into the potential's own input.
    """
    n = cfg.ontology.n_nodes
    if n != N_NODES:
        raise ValueError(
            f"The urban ontology has {N_NODES} nodes (PadMotion, BatteryReserve "
            "and GnssIntegrity before SafeLanding) but cfg.ontology.n_nodes is "
            f"{n}.")
    X = node_features(sem.node_values, cfg.ontology.in_dim)
    return OntologyGraph(X=X, src=_SRC, dst=_DST, rel=_REL, goal_node=GOAL_NODE,
                         node_names=tuple(cfg.ontology.node_names),
                         relation_names=tuple(cfg.ontology.relation_names))


def make_observation(meas: dict[str, Any], sem: SemanticState, cfg: Config) -> np.ndarray:
    """Normalized measured state plus semantic context, for PPO.

    Position and velocity are pad-relative. The deck's own velocity is given
    separately so the policy can feed it forward instead of having to infer a
    moving target from the error signal alone, and the two energy channels let
    it trade precision against reserve.

    The last three channels are the receiver's own account of itself: how much
    the fix can be trusted, how much of it is coming in off the facades, and
    how far out it says it could be. A policy that cannot see these has no way
    to tell a pad-relative pose worth descending on from one that is metres
    wrong, because in this environment the two look identical.
    """
    from .mathx import wrap_pi

    pos = np.asarray(meas["pos"], dtype=float)
    vel = np.asarray(meas["vel"], dtype=float)
    rpy = np.asarray(meas["rpy"], dtype=float)
    omega = np.asarray(meas["omega"], dtype=float)
    pad_vel = np.asarray(meas.get("pad_velocity", (0.0, 0.0, 0.0)), dtype=float).reshape(-1)
    pad_vel = pad_vel[:2] if pad_vel.size >= 2 else np.zeros(2)

    o = np.concatenate([
        pos[:2] / 4.0,
        [pos[2] / 6.0],
        vel / 3.0,
        rpy[:2] / cfg.rl.max_roll_pitch,
        [float(wrap_pi(rpy[2])) / np.pi],
        omega / np.deg2rad(180.0),
        [sem.wind_risk, sem.visual_stability, sem.alignment],
        pad_vel / max(cfg.semantic.pad_speed_scale, 1e-6),
        [sem.pad_motion, sem.battery_reserve, sem.energy_margin],
        [sem.gnss_integrity, sem.gnss_nlos_fraction,
         sem.gnss_sigma_xy / max(cfg.semantic.gnss_sigma_scale, 1e-6)],
    ])
    o = np.clip(o, -3.0, 3.0)
    if o.size != cfg.rl.obs_dim:
        raise ValueError(f"Observation dimension mismatch: built {o.size}, "
                         f"cfg.rl.obs_dim is {cfg.rl.obs_dim}.")
    return o
