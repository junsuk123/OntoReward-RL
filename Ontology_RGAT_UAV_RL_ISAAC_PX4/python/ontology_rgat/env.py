"""One flight-stack-in-the-loop episode against Isaac Sim and PX4.

Port of the retired ``+sim`` package. Nothing in here integrates a rigid body:
Isaac owns the physics, PX4 owns the estimator and the attitude loop, and this
module owns the episode -- the reset, the pad-relative state the algorithms
reason about, the termination test and the log.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .bridge import BridgeError, PX4Bridge
from .config import Config
from .mathx import quat_normalize, quat_to_euler_zyx
from .rewards import reward_for
from .semantic import (OntologyGraph, SemanticState, build_ontology_graph,
                       compute_features, make_observation)

__all__ = ["Current", "LandingEnv", "EpisodeLog", "run_episode",
           "state_to_model", "terminal_status", "truth_state"]


# --------------------------------------------------------------------- state
def _column(value: Any, n: int) -> np.ndarray:
    v = np.asarray(value, dtype=float).reshape(-1)
    return v if v.size == n else np.zeros(n)


def _pad_of(s: dict[str, Any]) -> dict[str, Any]:
    pad = s.get("pad") if isinstance(s.get("pad"), dict) else {}
    velocity = _column(pad.get("velocity", (0.0, 0.0, 0.0)), 3)
    return {
        "valid": bool(pad.get("valid", False)),
        "source": str(pad.get("source", "static")),
        "position": _column(pad.get("position", (0.0, 0.0, 0.0)), 3),
        "velocity": velocity,
        "yaw": float(pad.get("yaw", 0.0)),
        "yaw_rate": float(pad.get("yaw_rate", 0.0)),
        "speed": float(pad.get("speed", float(np.linalg.norm(velocity[:2])))),
    }


_BATTERY_FIELDS = {
    "enabled": (bool, False), "source": (str, "unavailable"),
    "remaining_j": (float, 0.0), "initial_j": (float, 0.0),
    "capacity_j": (float, 0.0), "energy_used_j": (float, 0.0),
    "power_w": (float, 0.0), "hover_power_w": (float, 0.0),
    "hover_seconds_remaining": (float, 0.0), "reserve": (float, 1.0),
    "landing_reserve_s": (float, 0.0), "state_of_charge": (float, 1.0),
    "voltage_v": (float, 0.0), "depleted": (bool, False),
}


_GNSS_FIELDS = {
    "enabled": (bool, False), "source": (str, "unavailable"),
    "valid": (bool, True), "fix_type": (int, 3),
    "satellites_tracked": (int, 12),
    "nlos_detected_fraction": (float, 0.0), "cn0_mean_db": (float, 45.0),
    "hdop": (float, 1.0), "vdop": (float, 1.6),
    "residual_rms_m": (float, 0.0), "sigma_xy_m": (float, 0.0),
    "quality": (float, 1.0),
    "deck_quality": (float, 1.0), "deck_sigma_xy_m": (float, 0.0),
}


def _gnss_of(s: dict[str, Any]) -> dict[str, Any]:
    """The receiver as the gateway reports it, with an open-sky default.

    A link that models no GNSS has to read as open sky, so a run against the
    protocol fake does not look like a total outage. Note what is *not* here:
    the true position error, the true NLOS count, the true sky view. None of
    them crosses this boundary, because the policy, the ontology and the reward
    would all be cheating if any of them did.
    """
    wire = s.get("gnss") if isinstance(s.get("gnss"), dict) else {}
    out: dict[str, Any] = {}
    for name, (cast, default) in _GNSS_FIELDS.items():
        value = wire.get(name, default)
        out[name] = cast(value) if value is not None else default
    return out


def _battery_of(s: dict[str, Any]) -> dict[str, Any]:
    """The pack as the gateway reports it, with a safe default for every field."""
    wire = s.get("battery") if isinstance(s.get("battery"), dict) else {}
    out: dict[str, Any] = {}
    for name, (cast, default) in _BATTERY_FIELDS.items():
        value = wire.get(name, default)
        out[name] = cast(value) if value is not None else default
    return out


def state_to_model(s: dict[str, Any], cfg: Config) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert one gateway ENU/FLU sample to the algorithms' state contract.

    ``x[0:3]`` and ``x[3:6]`` are **pad-relative**: the target rides a road
    vehicle, so the state the algorithms reason about is the state relative to
    the deck. The world pose PX4 estimates is carried in the diagnostics for
    telemetry only.

    In the canyon that pad-relative state is a *measurement*, and a fallible
    one: when the markers are out of frame it is the difference between two
    GNSS fixes, either of which can be metres out. The simulator's own answer
    comes alongside it in ``diag['ground_truth']`` and is used for one thing
    only -- deciding whether the landing succeeded. Grading on the sensor would
    score the receiver's mistake instead of the landing.
    """
    q = quat_normalize(s["quaternion_wxyz"])
    x = np.concatenate([_column(s["position"], 3), _column(s["velocity"], 3), q,
                        _column(s["angular_velocity"], 3)])
    pad = _pad_of(s)
    world = s.get("world") if isinstance(s.get("world"), dict) else {}
    aero = _column(s["aero_force"], 3)
    extra = s.get("extra") if isinstance(s.get("extra"), dict) else {}
    # Isaac publishes the exact applied force for validation plots. It must not
    # become a privileged policy input; real flights would need an estimator.
    # The measured anemometer still drives WindRisk through speed/change.
    aero_is_truth = extra.get("aero_force_source") == "simulator_truth"
    aero_for_control = np.zeros(3) if aero_is_truth else aero
    sensor = {
        "position": x[0:3].copy(), "velocity": x[3:6].copy(),
        "quaternion_wxyz": q.copy(), "angular_velocity": x[10:13].copy(),
        "acceleration": _column(s["acceleration"], 3),
        "marker_quality": float(s["marker_quality"]), "pad": pad,
        "battery": _battery_of(s), "gnss": _gnss_of(s),
    }
    truth = s.get("truth") if isinstance(s.get("truth"), dict) else {}
    truth_valid = bool(truth.get("valid", False))
    ground_truth = {
        "world_position": _column(world.get("position", s["position"]), 3),
        "world_velocity": _column(world.get("velocity", s["velocity"]), 3),
        # Pad-relative, and the only thing an episode may be scored on. Without
        # a truth block there is nothing better than the sensor, and the flag
        # says as much rather than letting a fallback look authoritative.
        "valid": truth_valid,
        "position": _column(truth.get("position", x[0:3]), 3),
        "velocity": _column(truth.get("velocity", x[3:6]), 3),
    }
    diag: dict[str, Any] = {
        "sensor": sensor, "ground_truth": ground_truth,
        "mean_wind_i": _column(s["wind"], 3),
        "aero_force_i": aero,
        "aero_force_mag": float(np.linalg.norm(aero)),
        "aero_force_observed_mag": float(np.linalg.norm(aero_for_control)),
        "aero_force_data_policy": "validation_only" if aero_is_truth else "measured",
        "acceleration_i": _column(s["acceleration"], 3),
        "marker_quality": float(s["marker_quality"]),
        "source": "PX4/Isaac",
        "pad": pad,
        "pad_position_i": pad["position"],
        "pad_velocity_i": pad["velocity"],
        "pad_speed": float(np.linalg.norm(pad["velocity"][:2])),
        "battery": _battery_of(s),
        "gnss": _gnss_of(s),
        "world_position_i": ground_truth["world_position"],
        "world_velocity_i": ground_truth["world_velocity"],
        "position_source": str((s.get("extra") or {}).get("position_source", "unknown")),
        "data_policy": "sensor_for_control_gt_for_validation_only",
    }
    if x.size != 13:
        raise ValueError("External state dimension mismatch.")
    return x, diag


def truth_state(x: np.ndarray, diag: dict[str, Any]) -> np.ndarray:
    """The state an episode is scored on: measured attitude, true geometry.

    Attitude and body rates come from the IMU and are not GNSS-derived, so they
    are the same in both views; position and velocity are replaced by the
    simulator's own pad-relative answer when it is available. A link that does
    not carry one falls back to the sensor, which is what the fixed-pad
    experiment always did.
    """
    truth = (diag.get("ground_truth") or {}) if isinstance(diag, dict) else {}
    if not truth.get("valid", False):
        return x
    out = np.asarray(x, dtype=float).copy()
    out[0:3] = _column(truth["position"], 3)
    out[3:6] = _column(truth["velocity"], 3)
    return out


def terminal_status(x: np.ndarray, cfg: Config, has_been_airborne: bool = True,
                    battery_depleted: bool = False) -> tuple[bool, str, float]:
    """Episode outcome for one external state sample.

    ``x`` must be the *truth* state (see :func:`truth_state`). Under GNSS
    degradation the pad-relative pose the policy flies on can be metres wrong,
    and an episode graded on it would report a landing the vehicle never made.

    ``has_been_airborne`` guards the ground test: with a real flight stack the
    vehicle can still be sitting on the pad when control is handed over, and
    that is a start condition, not a touchdown.

    The state is pad-relative, so the arena checks measure the distance to the
    deck: a rover that drives away faster than the drone follows it is a lost
    target, which is a mission failure and not a crash.

    ``viol`` carries a fifth ratio. Touching down on a deck the vehicle does not
    share a horizontal velocity with tips the airframe over, so closing speed is
    a landing criterion -- and it applies to the static control condition too,
    which is why these numbers are not comparable with older fixed-pad runs.
    """
    p, v = x[0:3], x[3:6]
    rpy = quat_to_euler_zyx(x[6:10])
    rate = float(np.linalg.norm(x[10:13]))
    tilt = float(np.linalg.norm(rpy[:2]))
    rel_speed_xy = float(np.linalg.norm(v[:2]))
    viol = float(max(
        np.linalg.norm(p[:2]) / cfg.criteria.xy,
        abs(v[2]) / cfg.criteria.vz,
        tilt / cfg.criteria.tilt,
        rate / cfg.criteria.rate,
        rel_speed_xy / cfg.criteria.rel_speed_xy))

    if has_been_airborne and p[2] <= cfg.sim.ground_z:
        return True, ("success" if viol <= 1.0 else "unsafe_touchdown"), viol
    if battery_depleted:
        # Out of energy in the air. Not a crash yet, but the mission is over and
        # the vehicle is about to become one, so it is its own failure mode
        # rather than a timeout.
        return True, "battery_depleted", viol
    if (float(np.linalg.norm(p[:2])) > cfg.sim.world_xy_limit
            or p[2] > cfg.sim.max_altitude or tilt > cfg.sim.crash_tilt):
        return True, "flight_failure", viol
    return False, "running", viol


def reconcile_touchdown(done: bool, status: str, viol: float,
                        x_truth: np.ndarray, state: dict[str, Any], diag: dict[str, Any],
                        has_been_airborne: bool) -> tuple[bool, str, bool]:
    """Require physical/authoritative contact without losing its first frame."""
    extra = state.get("extra") or {}
    authoritative = bool(extra.get("land_detector_authoritative", True))
    pad_valid = bool(diag["pad"].get("valid", False))
    confirmed = bool(state["landed"]) and authoritative and pad_valid

    if done and status in {"success", "unsafe_touchdown"} and not confirmed:
        # Geometry and the physics callback are sampled independently. Near
        # the roof, wait for the contact callback instead of declaring a miss.
        # Far below the roof is the road and is a genuine mislanding.
        if float(x_truth[2]) < -0.25 or not pad_valid:
            status = "ground_mislanding" if status == "success" else "unsafe_touchdown"
        else:
            done, status = False, "running"

    if not done and has_been_airborne and confirmed:
        done = True
        status = "success" if viol <= 1.0 else "unsafe_touchdown"
    return done, status, confirmed


# ------------------------------------------------------------------- current
@dataclass
class Current:
    """Everything one control step derives from a single state sample."""
    meas: dict[str, Any]
    sem: SemanticState
    graph: OntologyGraph
    obs: np.ndarray


# ----------------------------------------------------------------- the episode
class LandingEnv:
    """The live episode: a bridge, the current state, and the clock."""

    def __init__(self, bridge: PX4Bridge, cfg: Config, seed: int,
                 state: dict[str, Any]):
        self.bridge = bridge
        self.cfg = cfg
        self.seed = seed
        self.x, self.last_diag = state_to_model(state, cfg)
        self.t = 0.0
        self.step_index = 0
        self.prev_sem: SemanticState | None = None
        self.external_state = state
        self.has_been_airborne = not bool(state["landed"])
        self.done = False
        self.status = "running"

    # ---------------------------------------------------------------- reset
    @classmethod
    def reset(cls, seed: int, cfg: Config) -> "LandingEnv":
        """Reset and hand over once PX4 holds the entry pose.

        A long sweep outlives one PX4 SITL session: after hours the simulated
        battery status goes stale under lockstep and PX4 refuses to arm, which
        surfaces here as a reset that never reaches the entry pose. When the
        pipeline owns the simulator, cycle it and try again rather than throwing
        away the run.
        """
        from . import stack as stack_module

        attempts = 1 + max(0, int(cfg.external.reset_recoveries))
        last: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return cls._attempt_reset(seed, cfg)
            except Exception as exc:            # noqa: BLE001 - retried below
                last = exc
                owned = stack_module.current()
                if attempt == attempts or owned is None:
                    raise
                print(f"WARNING: reset failed ({exc}). Restarting the simulator "
                      f"and retrying ({attempt} of {attempts - 1}).")
                owned.restart()
        raise last                              # pragma: no cover - unreachable

    @classmethod
    def _attempt_reset(cls, seed: int, cfg: Config) -> "LandingEnv":
        bridge = PX4Bridge(cfg)
        try:
            state = bridge.reset(seed)
        except Exception:
            # Release the local UDP port so a retry can bind it again.
            bridge.close()
            raise
        return cls(bridge, cfg, seed, state)

    # -------------------------------------------------------------- stepping
    def current(self) -> Current:
        """The unchanged ontology/R-GAT input, built from PX4 telemetry."""
        rpy = quat_to_euler_zyx(self.x[6:10])
        diag = self.last_diag
        meas = {
            "pos": self.x[0:3], "vel": self.x[3:6], "rpy": rpy, "omega": self.x[10:13],
            "marker_quality": diag["marker_quality"],
            "marker_detected": diag["marker_quality"] > 0.1,
            "pad_velocity": diag["pad_velocity_i"],
            "pad_speed": diag["pad_speed"],
            "battery": diag["battery"],
        }
        sem = compute_features(diag, meas, self.prev_sem, self.cfg)
        graph = build_ontology_graph(sem, self.cfg)
        return Current(meas=meas, sem=sem, graph=graph,
                       obs=make_observation(meas, sem, self.cfg))

    def step(self, action: np.ndarray, cur: Current, reward_mode: str,
             potential) -> tuple[Current, float, bool, dict[str, Any]]:
        cfg = self.cfg
        s = self.bridge.step(action)
        x2, diag2 = state_to_model(s, cfg)

        # Advance the episode clock by the simulated time that actually elapsed.
        # The bridge paces to cfg.sim.dt but can only return a sample on a PX4
        # publication boundary, so assuming the nominal period leaves the log
        # short of the truth.
        self.t += self._elapsed_seconds(s)
        self.x, self.last_diag = x2, diag2
        self.step_index += 1
        self.prev_sem = cur.sem
        self.external_state = s
        self.has_been_airborne = self.has_been_airborne or not bool(s["landed"])
        nxt = self.current()

        # Scored on the simulator's own geometry, flown on the measured one.
        x_truth = truth_state(x2, diag2)
        done, status, viol = terminal_status(x_truth, cfg, self.has_been_airborne,
                                             diag2["battery"]["depleted"])
        done, status, _ = reconcile_touchdown(
            done, status, viol, x_truth, s, diag2, self.has_been_airborne)
        if not done and self.step_index >= cfg.sim.max_steps:
            done, status = True, "timeout"

        r, parts = reward_for(reward_mode, cur, nxt, action, status, viol,
                              potential, cfg)
        self.done, self.status = done, status
        info = {
            "status": status, "reward_parts": parts, "diag": diag2, "viol": viol,
            "armed": bool(s["armed"]), "landed": bool(s["landed"]),
            "nav_state": s.get("nav_state", 0), "pad_speed": diag2["pad_speed"],
            "touchdown_source": str((s.get("extra") or {}).get(
                "touchdown_source", "none")),
            "battery_reserve": diag2["battery"]["reserve"],
            "energy_used_j": diag2["battery"]["energy_used_j"],
            "gnss_quality": diag2["gnss"]["quality"],
            # How far the pose the policy is flying on actually is from the
            # truth. Diagnostic only, and the honest way to report what the
            # canyon cost: no consumer on the control path may read it.
            "position_error_m": float(np.linalg.norm((x_truth - x2)[0:3])),
        }
        if done and str(cfg.external.target).lower() == "sitl":
            try:
                confirmed_stop = self.bridge.stop_after_outcome()
                info_status = status if confirmed_stop else "unconfirmed_" + status
                status = info_status
            except BridgeError:
                status = "unconfirmed_" + status
            self.status = status
            info["status"] = status
            info["stop_confirmed"] = status.startswith(("success", "unsafe_touchdown",
                                                          "flight_failure", "battery_depleted",
                                                          "ground_mislanding", "timeout"))
        return nxt, r, done, info

    def _elapsed_seconds(self, s: dict[str, Any]) -> float:
        dt = self.cfg.sim.dt
        if "px4_time_us" not in s or "px4_time_us" not in self.external_state:
            return dt
        measured = (float(s["px4_time_us"])
                    - float(self.external_state["px4_time_us"])) * 1e-6
        # Ignore a clock reset or a stalled sample; the nominal period is the
        # sane fallback and the pacing loop already guards a stalled simulator.
        if math.isfinite(measured) and 0.0 < measured < 10.0 * dt:
            return measured
        return dt

    # -------------------------------------------------------------- teardown
    def close(self) -> None:
        """Disarm and release the bridge.

        Releasing matters as much as disarming: the next episode needs the local
        UDP port back.
        """
        try:
            self.bridge.disarm()
        except BridgeError:
            pass
        self.bridge.close()

    def __enter__(self) -> "LandingEnv":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------- log
@dataclass
class EpisodeLog:
    """Per-step traces plus the metrics one episode is summarised by."""
    t: list[float] = field(default_factory=list)
    x: list[np.ndarray] = field(default_factory=list)
    a: list[np.ndarray] = field(default_factory=list)
    r: list[float] = field(default_factory=list)
    wind: list[np.ndarray] = field(default_factory=list)
    aero_force: list[np.ndarray] = field(default_factory=list)
    aero_mag: list[float] = field(default_factory=list)
    tilt: list[float] = field(default_factory=list)
    phi: list[float] = field(default_factory=list)
    pad_pos: list[np.ndarray] = field(default_factory=list)
    pad_vel: list[np.ndarray] = field(default_factory=list)
    pad_speed: list[float] = field(default_factory=list)
    closing_speed: list[float] = field(default_factory=list)
    gnss_quality: list[float] = field(default_factory=list)
    gnss_sigma_xy: list[float] = field(default_factory=list)
    gnss_nlos_fraction: list[float] = field(default_factory=list)
    nav_confidence: list[float] = field(default_factory=list)
    estimate_error_m: list[float] = field(default_factory=list)
    battery_reserve: list[float] = field(default_factory=list)
    battery_power_w: list[float] = field(default_factory=list)
    hover_seconds_left: list[float] = field(default_factory=list)
    energy_margin: list[float] = field(default_factory=list)
    marker_quality: list[float] = field(default_factory=list)
    graph_x: list[np.ndarray] = field(default_factory=list)
    graph_template: OntologyGraph | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def as_arrays(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, list) and value:
                out[key] = np.asarray(value)
        return out


def run_episode(policy, reward_mode: str, potential, seed: int, cfg: Config,
                monitor: Callable[[EpisodeLog, int, Current, dict[str, Any]], None] | None = None
                ) -> EpisodeLog:
    """Fly one episode and log it.

    Alongside the original traces, the log carries the two external factors --
    what the deck was doing and what energy was left -- so a landing can be read
    against both.
    """
    env = LandingEnv.reset(seed, cfg)
    log = EpisodeLog()
    status = "timeout"
    energy_used_j = 0.0
    steps = 0
    try:
        for k in range(1, cfg.sim.max_steps + 1):
            cur = env.current()
            if log.graph_template is None:
                log.graph_template = cur.graph
            action = policy.action(cur, env.x, cfg)
            x_before, diag_before, t_before = env.x.copy(), env.last_diag, env.t
            nxt, r, done, info = env.step(action, cur, reward_mode, potential)

            log.t.append(t_before)
            log.x.append(x_before)
            log.a.append(np.asarray(action, dtype=float))
            log.r.append(r)
            log.wind.append(diag_before["mean_wind_i"])
            log.aero_force.append(diag_before["aero_force_i"])
            log.aero_mag.append(diag_before["aero_force_mag"])
            log.pad_pos.append(diag_before["pad_position_i"])
            log.pad_vel.append(diag_before["pad_velocity_i"])
            log.pad_speed.append(diag_before["pad_speed"])
            log.marker_quality.append(diag_before["marker_quality"])
            log.battery_reserve.append(diag_before["battery"]["reserve"])
            log.battery_power_w.append(diag_before["battery"]["power_w"])
            log.hover_seconds_left.append(diag_before["battery"]["hover_seconds_remaining"])
            log.closing_speed.append(cur.sem.closing_speed)
            log.gnss_quality.append(cur.sem.gnss_integrity)
            log.gnss_sigma_xy.append(cur.sem.gnss_sigma_xy)
            log.gnss_nlos_fraction.append(cur.sem.gnss_nlos_fraction)
            log.nav_confidence.append(cur.sem.nav_confidence)
            log.estimate_error_m.append(float(info["position_error_m"]))
            log.energy_margin.append(cur.sem.energy_margin)
            log.tilt.append(cur.sem.tilt)
            log.graph_x.append(cur.graph.X)
            log.phi.append(float(info["reward_parts"].get("phi0", np.nan))
                           if info["reward_parts"] else float("nan"))

            steps = k
            status = info["status"]
            energy_used_j = info["energy_used_j"]
            if monitor is not None and (k % cfg.sim.monitor_every == 0 or done):
                monitor(log, k, nxt, info)
            if done:
                break
    finally:
        env.close()

    # The final state as the simulator knows it: the touchdown numbers reported
    # here are the ones the episode was scored on, not the ones the receiver
    # believed.
    final = truth_state(env.x, env.last_diag)
    rpy = quat_to_euler_zyx(final[6:10])
    log.metrics = {
        "seed": int(seed), "status": status,
        "success": float(status == "success"),
        "unsafe": float(status in ("unsafe_touchdown", "flight_failure")),
        "timeout": float(status == "timeout"),
        "depleted": float(status == "battery_depleted"),
        "steps": int(steps), "return": float(np.sum(log.r)),
        "duration_s": float(env.t),
        "touchdown_xy": float(np.linalg.norm(final[0:2])),
        "touchdown_vz": float(abs(final[5])),
        "touchdown_rel_speed_xy": float(np.linalg.norm(final[3:5])),
        "max_tilt": float(np.max(log.tilt)) if log.tilt else 0.0,
        "max_aero_force": float(np.max(log.aero_mag)) if log.aero_mag else 0.0,
        "pad_speed_mean": float(np.mean(log.pad_speed)) if log.pad_speed else 0.0,
        "gnss_quality_mean": (float(np.mean(log.gnss_quality))
                              if log.gnss_quality else 1.0),
        "gnss_sigma_xy_mean": (float(np.mean(log.gnss_sigma_xy))
                               if log.gnss_sigma_xy else 0.0),
        "gnss_nlos_mean": (float(np.mean(log.gnss_nlos_fraction))
                           if log.gnss_nlos_fraction else 0.0),
        # What the canyon actually cost the estimate the policy flew on.
        "estimate_error_mean_m": (float(np.mean(log.estimate_error_m))
                                  if log.estimate_error_m else 0.0),
        "estimate_error_max_m": (float(np.max(log.estimate_error_m))
                                 if log.estimate_error_m else 0.0),
        "pad_speed_max": float(np.max(log.pad_speed)) if log.pad_speed else 0.0,
        "battery_reserve_final": float(env.last_diag["battery"]["reserve"]),
        "hover_seconds_left": float(env.last_diag["battery"]["hover_seconds_remaining"]),
        # Measurable, not NaN: the gateway prices the collective against a
        # momentum-theory hover model, so the energy an episode spent is known.
        "energy_j": float(energy_used_j),
        "final_tilt": float(np.linalg.norm(rpy[:2])),
        "backend": "Isaac/PX4",
    }
    return log
