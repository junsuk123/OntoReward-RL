"""G3 local coupling checks of spatial_current_semantics_v1 (guide section 10, G3).

Short deterministic flights of the spatial-reference/12 local plant through
``backend.CurrentSpatialBackend``, read through the new pipeline only -- the
observer's 21D vector, ``observation.decode`` and ``features.build_features``.
Truth is the yardstick reported next to each pipeline number, never the
signal under test.  The flying is done by small fixed probes on the decoded
current vector, registered under ``g3_local``: implementation checks, never
teachers and never training data.  Seeds are ``g3_pilot_seeds``.  This is the
local half of G3; the Isaac half needs a user-approved stack and is not run
here.  A failing check is reported with its numbers, never loosened.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import math
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from . import SCHEMA_ID
from .backend import (BACKENDS, CALIBRATION_GAPS, FLIGHT_SCHEMA, NAVIGATION_NOTE, ActionTrace,
                      make_local_backend)
from .contracts import (OBSERVATION_NAMES, AxisCouplingProbe, BlindZoneProbe,
                        CurrentSemanticsConfig, FovExitProbe, contract_manifest)
from .features import FEATURE_NAMES, FeatureFrame, build_feature_batch, build_features
from .geometry import CameraKinematics, camera_kinematics
from .observation import CurrentPhysical, decode
from ..spatial.core import REFERENCE_OPTICAL_IMAGE_PX, SpatialConfig, camera_bearings
from ..spatial.environment import LocalBackend, deployed_board, deployed_board_ids


CHECKS = ("axis_coupling", "fov_exit", "calibration_identity", "low_altitude_blind_zone",
          "quality_channels", "observation_purity")
# Exit axis -> (image edge crossed, unit drift of the pad relative to the vehicle in
# the pad plane).  At the local entry yaw 0 and board yaw 0, a pad drifting +x
# (body forward) leaves through the image right, +y (body left) through the top.
EXITS = MappingProxyType({"x": ("right", (1.0, 0.0)), "y": ("top", (0.0, 1.0))})
# The padded quad the local detector model requires inside the frame: 4/3 of the
# tag side, the ArUco quiet zone (LocalBackend._optical_quality/_marker_corners).
QUIET_ZONE_SCALE = 4.0 / 3.0
IDENTITY_TOLERANCE = 1e-9
_FI = MappingProxyType({name: index for index, name in enumerate(FEATURE_NAMES)})
_VISION_AGE = OBSERVATION_NAMES.index("ugv_visionAge")


# --------------------------------------------------------------- recording

@dataclass(frozen=True, eq=False)
class Decision:
    """One decision of a probe flight; ``trace`` is the action taken at it."""
    vector: np.ndarray
    frame: FeatureFrame
    physical: CurrentPhysical
    provenance: Mapping[str, Any]
    evaluator: Mapping[str, Any]
    trace: ActionTrace | None = None

    @property
    def time_s(self) -> float:
        return float(self.provenance["decision_stamp_s"])


@dataclass(frozen=True, eq=False)
class Episode:
    label: str
    seed: int
    decisions: tuple[Decision, ...]
    termination: Mapping[str, Any] | None


Policy = Callable[[CurrentPhysical], np.ndarray]


def fly(config: CurrentSemanticsConfig, seed: int, label: str, policy: Policy,
        max_steps: int, *, stop: Callable[[list[Decision]], bool] | None = None,
        backend: Any = None) -> Episode:
    """One nominal episode; the policy sees only the decoded current vector.

    Without ``backend`` it flies a fresh local plant and closes it.  A given
    backend (the Isaac half) is the caller's: an episode ended by ``stop`` is
    left active and the next ``reset`` cleans it up.
    """
    owned = backend is None
    backend = make_local_backend(config) if owned else backend
    decisions: list[Decision] = []
    termination = None
    try:
        vector, info = backend.reset(seed)
        while True:
            decisions.append(Decision(vector, build_features(vector, config), decode(vector),
                                      info["observation_provenance"], info["evaluator_only"]))
            if (termination is not None or len(decisions) > max_steps
                    or (stop is not None and stop(decisions))):
                break
            step = backend.step(policy(decisions[-1].physical))
            decisions[-1] = replace(decisions[-1], trace=step.trace)
            vector, info = step.observation, step.info
            if step.done:
                termination = info["termination"]
    finally:
        if owned:
            backend.close()
    return Episode(label, int(seed), tuple(decisions), termination)


def _truth_camera(decision: Decision, config: CurrentSemanticsConfig) -> CameraKinematics:
    """The pad centre's true camera geometry.  Truth carries roll/pitch only; yaw is
    the own measurement's, which the local plant reports exactly."""
    ev = decision.evaluator
    r, v = ev["relative_position_own_minus_pad_m"], ev["relative_velocity_own_minus_pad_m_s"]
    roll, pitch = ev["roll_pitch_rad"]
    p, q, yaw_rate = ev["angular_rate_rad_s"]
    physical = CurrentPhysical(
        ex=-r[0], ey=-r[1], h=r[2], vrx=-v[0], vry=-v[1], vgx=0.0, vgy=0.0, vz=v[2],
        roll=roll, pitch=pitch, yaw=decision.physical.yaw, p=p, q=q, r=yaw_rate,
        vision_updated=True, vision_age_s=0.0, nav_valid=True, nav_age_s=0.0,
        estimate_valid=True)
    return camera_kinematics(physical, config.calibration)


def _overshoot_m(camera: CameraKinematics, edge: str, config: CurrentSemanticsConfig) -> float:
    """Pad-plane distance of the pad centre beyond ``edge`` (> 0 outside); NaN
    without a projection."""
    if not camera.front_valid:
        return math.nan
    cal = config.calibration
    span, focal = ((cal.width - 1, cal.fx) if edge in ("left", "right")
                   else (cal.height - 1, cal.fy))
    return -camera.margin(edge) * span * camera.p_c[2] / (2.0 * focal)


def _at(times: Sequence[float], values: Sequence[float], t: float) -> float | None:
    """Linear interpolation between the two decisions bracketing ``t``."""
    times = np.asarray(times, dtype=np.float64)
    i = int(np.searchsorted(times, t))
    if i < len(times) and times[i] == t:
        return float(values[i]) if math.isfinite(values[i]) else None
    if i == 0 or i >= len(times):
        return None
    t0, t1, v0, v1 = times[i - 1], times[i], values[i - 1], values[i]
    if not (math.isfinite(v0) and math.isfinite(v1)):
        return None
    return float(v0 + (v1 - v0) * (t - t0) / (t1 - t0))


def _last_update(episode: Episode) -> int | None:
    flags = [d.physical.vision_updated for d in episode.decisions]
    return max((k for k, flag in enumerate(flags) if flag), default=None)


def _stats(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {"min": float(array.min()), "median": float(np.median(array)),
            "max": float(array.max()), "count": int(array.size)}


def _seeds(config: CurrentSemanticsConfig, seeds: Sequence[int] | None) -> tuple[int, ...]:
    chosen = tuple(int(seed) for seed in (config.g3.pilot_seeds if seeds is None else seeds))
    foreign = sorted(set(chosen) - set(config.g3.pilot_seeds))
    if not chosen or foreign:
        raise ValueError(f"G3 runs only registered g3_pilot_seeds; refused {foreign or chosen}")
    return chosen


# ----------------------------------------------------------------- probes

def _schedule(command: Any, probe: AxisCouplingProbe) -> Policy:
    """``command`` for ``push_steps`` decisions, then zero net acceleration (open loop)."""
    command = np.asarray(command, dtype=np.float64)
    calls = [0]

    def policy(physical: CurrentPhysical) -> np.ndarray:
        calls[0] += 1
        return command.copy() if calls[0] <= probe.push_steps else np.zeros(3)
    return policy


def _exit_policy(probe: FovExitProbe, axis: str) -> Policy:
    """Hold the first estimated height, hold the pad centred on the cross axis and
    drive the estimated pad-relative velocity to ``exit_speed`` along ``axis``;
    hover until an estimate exists."""
    drift = np.array(EXITS[axis][1])
    target, cross = drift * probe.exit_speed_m_s, 1.0 - np.abs(drift)
    height: list[float] = []

    def policy(physical: CurrentPhysical) -> np.ndarray:
        if not physical.estimate_valid:
            return np.zeros(3)
        if not height:
            height.append(physical.h)
        lateral = (probe.velocity_gain_per_s * (np.array([physical.vrx, physical.vry]) - target)
                   + probe.cross_position_gain_per_s2 * cross
                   * np.array([physical.ex, physical.ey]))
        vertical = (probe.height_gain_per_s2 * (height[0] - physical.h)
                    - probe.velocity_gain_per_s * physical.vz)
        return np.clip(np.r_[lateral, vertical], -probe.acceleration_limit_m_s2,
                       probe.acceleration_limit_m_s2)
    return policy


def _descent_policy(probe: BlindZoneProbe, dt: float) -> Policy:
    """Lateral PD on the estimated pad offset and a height ramp at ``sink_speed``;
    after the last frame it flies the track's prediction."""
    reference: list[float] = []

    def policy(physical: CurrentPhysical) -> np.ndarray:
        if not physical.estimate_valid:
            return np.zeros(3)
        reference.append((reference[-1] if reference else physical.h) - probe.sink_speed_m_s * dt)
        lateral = (probe.position_gain_per_s2 * np.array([physical.ex, physical.ey])
                   + probe.velocity_gain_per_s * np.array([physical.vrx, physical.vry]))
        vertical = (probe.height_gain_per_s2 * (reference[-1] - physical.h)
                    + probe.vertical_velocity_gain_per_s * (-probe.sink_speed_m_s - physical.vz))
        return np.clip(np.r_[lateral, vertical], -probe.acceleration_limit_m_s2,
                       probe.acceleration_limit_m_s2)
    return policy


# ------------------------------------------------------- 1. axis coupling

def _paired(run: Episode, base: Episode, k: int) -> dict[str, float]:
    a, b = run.decisions[k], base.decisions[k]
    if not (a.frame.camera.front_valid and b.frame.camera.front_valid):
        raise ValueError("pad centre not projectable at a paired decision")
    feature = lambda d, name: float(d.frame.flat[_FI[name]])
    ta, tb = a.evaluator["roll_pitch_rad"], b.evaluator["roll_pitch_rad"]
    return {"d_sin_pitch": feature(a, "drone_sinTheta") - feature(b, "drone_sinTheta"),
            "d_sin_roll": feature(a, "drone_sinPhi") - feature(b, "drone_sinPhi"),
            "d_xi": a.frame.camera.xi - b.frame.camera.xi,
            "d_eta": a.frame.camera.eta - b.frame.camera.eta,
            "d_view_xi": feature(a, "view_xi") - feature(b, "view_xi"),
            "d_view_eta": feature(a, "view_eta") - feature(b, "view_eta"),
            "truth_d_roll_rad": ta[0] - tb[0], "truth_d_pitch_rad": ta[1] - tb[1],
            "yaw_rad": a.physical.yaw}


def _small(cross: float, main: float, ratio: float) -> bool:
    return abs(cross) <= ratio * abs(main)


def check_axis_coupling(config: CurrentSemanticsConfig, seeds: Sequence[int] | None = None
                        ) -> dict[str, Any]:
    """x-only, y-only and diagonal pushes at ONE magnitude, against a hover of the same seed.

    Isaac convention: +ax -> sin(pitch) > 0, +ay -> sin(roll) < 0.  With
    R_CB = diag(1, -1, -1) (image right = body forward, image down = body
    right) the tilt first swings the nadir view (+ax: xi up, +ay: eta down);
    after the coast the offset is translation (+x: xi down, +y: eta up).
    """
    probe = config.g3.local.axis_coupling
    ratio, residual_max = probe.cross_axis_ratio_max, probe.superposition_residual_max
    a = probe.acceleration_m_s2
    commands = {"x": (a, 0.0, 0.0), "y": (0.0, a, 0.0),
                "diagonal": (a / math.sqrt(2.0), a / math.sqrt(2.0), 0.0)}
    push, end = probe.push_steps, probe.push_steps + probe.coast_steps
    rows, failures = [], []
    for seed in _seeds(config, seeds):
        base = fly(config, seed, "hover", _schedule(np.zeros(3), probe), end)
        runs = {name: fly(config, seed, f"push_{name}", _schedule(cmd, probe), end)
                for name, cmd in commands.items()}
        if any(len(ep.decisions) != end + 1 for ep in (base, *runs.values())):
            failures.append({"seed": seed, "reason": "an episode ended before the coast did"})
            continue
        try:
            at_push = {name: _paired(run, base, push) for name, run in runs.items()}
            at_end = {name: _paired(run, base, end) for name, run in runs.items()}
        except ValueError as exc:
            failures.append({"seed": seed, "reason": str(exc)})
            continue
        x, y, d = (at_push[name] for name in ("x", "y", "diagonal"))
        xe, ye, de = (at_end[name] for name in ("x", "y", "diagonal"))
        attitude = {name: np.array([v["d_sin_pitch"], v["d_sin_roll"]])
                    for name, v in at_push.items()}
        image = {name: np.array([v["d_xi"], v["d_eta"]]) for name, v in at_end.items()}
        superposed = lambda parts: (parts["x"] + parts["y"]) / math.sqrt(2.0)
        attitude_residual = float(np.linalg.norm(attitude["diagonal"] - superposed(attitude))
                                  / np.linalg.norm(attitude["diagonal"]))
        image_residual = float(np.linalg.norm(image["diagonal"] - superposed(image))
                               / np.linalg.norm(image["diagonal"]))
        tilt_ratio = float(np.linalg.norm(attitude["diagonal"])
                           / np.mean([np.linalg.norm(attitude[n]) for n in ("x", "y")]))
        conditions = {
            "x_pitch_positive": x["d_sin_pitch"] > 0,
            "x_roll_small": _small(x["d_sin_roll"], x["d_sin_pitch"], ratio),
            "x_tilt_moves_xi_up": x["d_xi"] > 0 and _small(x["d_eta"], x["d_xi"], ratio),
            "x_translation_moves_xi_down": (xe["d_xi"] < 0
                                            and _small(xe["d_eta"], xe["d_xi"], ratio)),
            "y_roll_negative": y["d_sin_roll"] < 0,
            "y_pitch_small": _small(y["d_sin_pitch"], y["d_sin_roll"], ratio),
            "y_tilt_moves_eta_down": y["d_eta"] < 0 and _small(y["d_xi"], y["d_eta"], ratio),
            "y_translation_moves_eta_up": (ye["d_eta"] > 0
                                           and _small(ye["d_xi"], ye["d_eta"], ratio)),
            "diagonal_signs": (d["d_sin_pitch"] > 0 and d["d_sin_roll"] < 0
                               and de["d_xi"] < 0 and de["d_eta"] > 0),
            "diagonal_superposes": max(attitude_residual, image_residual) <= residual_max,
            "diagonal_same_tilt": abs(tilt_ratio - 1.0) <= residual_max,
        }
        constrained = any(t.trace is not None and t.trace.adapter_constrained
                          for ep in runs.values() for t in ep.decisions)
        frames = {name: ep.decisions[-1].provenance["frames_accepted"]
                  for name, ep in (("hover", base), *runs.items())}
        rows.append({"seed": seed, "passed": all(conditions.values()), "conditions": conditions,
                     "push_end": at_push, "coast_end": at_end,
                     "superposition_residual": {"attitude": attitude_residual,
                                                "image": image_residual},
                     "diagonal_tilt_over_axis_tilt": tilt_ratio,
                     "adapter_constrained": constrained, "frames_accepted": frames,
                     "paired_frames_equal": len(set(frames.values())) == 1})
    passed = not failures and all(row["passed"] for row in rows)
    return {"passed": passed, "probe": _plain(probe), "failures": failures, "seeds": rows,
            "summary": {
                "x_d_sin_pitch": _stats([r["push_end"]["x"]["d_sin_pitch"] for r in rows]),
                "y_d_sin_roll": _stats([r["push_end"]["y"]["d_sin_roll"] for r in rows]),
                "x_coast_d_xi": _stats([r["coast_end"]["x"]["d_xi"] for r in rows]),
                "y_coast_d_eta": _stats([r["coast_end"]["y"]["d_eta"] for r in rows]),
                "worst_superposition_residual": max(
                    (max(r["superposition_residual"].values()) for r in rows), default=None),
                "yaw_abs_max_rad": max((abs(v["yaw_rad"]) for r in rows
                                        for v in r["push_end"].values()), default=None)},
            "note": ("paired against a zero-command run of the same seed, which cancels the "
                     "seeded disturbance, pad motion and sensor draws; same magnitude on every "
                     "direction, never the axis maximum on both")}


# ----------------------------------------------------------- 2. FOV exit

def board_trailing_extent_m(direction: Sequence[float], board: np.ndarray) -> float:
    """Pinhole prediction: the board centre's distance beyond the edge when the last
    padded tag quad leaves the frame, for a board drifting along ``direction``."""
    d = np.asarray(direction, dtype=np.float64)
    return float(max(-(row[:2] @ d) - 0.5 * QUIET_ZONE_SCALE * row[2] for row in board))


def fov_exit_episodes(config: CurrentSemanticsConfig, seeds: Sequence[int] | None = None,
                      *, backend: Any = None) -> list[Episode]:
    probe = config.g3.local.fov_exit

    def stop_after_loss(edge: str) -> Callable[[list[Decision]], bool]:
        def stop(decisions: list[Decision]) -> bool:
            crossed = any(d.frame.camera.front_valid and d.frame.camera.margin(edge) < 0
                          for d in decisions)
            tail = decisions[-probe.steps_after_loss:]
            return (crossed and len(tail) == probe.steps_after_loss
                    and not any(d.physical.vision_updated for d in tail))
        return stop
    return [fly(config, seed, f"fov_exit_{axis}", _exit_policy(probe, axis), probe.max_steps,
                stop=stop_after_loss(EXITS[axis][0]), backend=backend)
            for seed in _seeds(config, seeds) for axis in EXITS]


def _exit_row(episode: Episode, config: CurrentSemanticsConfig,
              board: np.ndarray) -> dict[str, Any]:
    probe = config.g3.local.fov_exit
    tolerance = config.g3.local.quality_channels.age_tolerance_s
    axis = episode.label.rsplit("_", 1)[-1]
    edge, direction = EXITS[axis]
    D = episode.decisions
    times = [d.time_s for d in D]
    truth = [_truth_camera(d, config) for d in D]
    estimated = [_overshoot_m(d.frame.camera, edge, config) for d in D]
    true = [_overshoot_m(c, edge, config) for c in truth]
    k_exit, k_exit_truth = _final_exit(estimated), _final_exit(true)
    k_last = _last_update(episode)
    predicted = board_trailing_extent_m(direction, board)
    row: dict[str, Any] = {"seed": episode.seed, "axis": axis, "edge": edge,
                           "decisions": len(D), "termination": episode.termination,
                           "k_centre_exit_estimate": k_exit, "k_centre_exit_truth": k_exit_truth,
                           "k_last_update": k_last, "pinhole_board_prediction_m": predicted}
    if k_exit is None or k_last is None:
        return {**row, "passed": False, "reason": "no estimated centre exit or no detection"}
    capture = float(D[k_last].provenance["capture_stamp_s"])
    # Where detection stops is a sensor property: measure it at the sensor's last
    # detected capture, before the cadence filter thins what the track accepts.
    sensor = D[-1].provenance.get("sensor_capture_stamp_s")
    sensor = capture if sensor is None else float(sensor)
    after = D[k_last + 1:]
    overshoot = _at(times, estimated, sensor)
    others = [name for name in ("left", "right", "top", "bottom") if name != edge]
    inside = lambda camera: camera.front_valid and min(camera.margin(n) for n in others) > 0
    window = slice(max(min(k_exit - 1, k_last), 0), max(k_exit, k_last) + 1)
    ages = [d.physical.vision_age_s for d in after]
    age_errors = [abs(d.physical.vision_age_s - (d.time_s - capture)) for d in after]
    moved = [float(np.hypot(b.physical.ex - a.physical.ex, b.physical.ey - a.physical.ey))
             for a, b in zip(after, after[1:])]
    negative_front = sum(d.frame.camera.front_valid and d.frame.camera.margin(edge) < 0
                         for d in after)
    conditions = {
        # The overshoot is measured against ONE edge; a final exit through a
        # corner would be measured against the wrong one: failed precondition.
        "single_edge_exit": all(inside(d.frame.camera) for d in D[window]),
        "detection_stopped": len(after) >= probe.steps_after_loss,
        "stop_near_centre_exit": (overshoot is not None and -probe.overshoot_tolerance_m
                                  <= overshoot <= predicted + probe.overshoot_tolerance_m),
        "age_is_decision_minus_last_capture": bool(age_errors) and max(age_errors) <= tolerance,
        "estimate_keeps_predicting": (all(d.physical.estimate_valid for d in after)
                                      and bool(moved) and min(moved) > 0.0),
    }
    truth_overshoot = _at(times, true, sensor)
    return {**row, "passed": all(conditions.values()), "conditions": conditions,
            "last_capture_s": capture, "last_sensor_capture_s": sensor,
            "centre_exit_to_last_capture_s": sensor - _crossing_time(times, estimated),
            "overshoot_at_last_capture_m": overshoot,
            "overshoot_at_last_accepted_capture_m": _at(times, estimated, capture),
            "overshoot_at_last_capture_truth_m": truth_overshoot,
            "overshoot_minus_pinhole_m": None if overshoot is None else overshoot - predicted,
            "decisions_after_loss": len(after),
            "vision_age_after_loss_s": ages[:probe.steps_after_loss],
            "front_valid_with_negative_margin": int(negative_front),
            "truth_single_edge_exit": all(inside(c) for c in truth[window]),
            "estimate_error_at_exit_m": abs(estimated[k_exit] - true[k_exit])}


def _final_exit(overshoot: Sequence[float]) -> int | None:
    """First decision of the last run of decisions with the centre beyond the edge:
    an earlier excursion that came back is not the exit."""
    start = None
    for k, value in enumerate(overshoot):
        start = (start if start is not None else k) if value > 0 else None
    return start


def _crossing_time(times: Sequence[float], overshoot: Sequence[float]) -> float:
    k = _final_exit(overshoot)
    if k == 0 or not math.isfinite(overshoot[k - 1]):
        return float(times[k])
    v0, v1 = overshoot[k - 1], overshoot[k]
    return float(times[k - 1] + (times[k] - times[k - 1]) * (-v0) / (v1 - v0))


def check_fov_exit(config: CurrentSemanticsConfig,
                   episodes: Sequence[Episode]) -> dict[str, Any]:
    """Detection must stop at the frame edge, and the observation must say so.

    The stop is measured as the pad centre's estimated distance beyond the
    edge at the sensor's last detected capture (the cadence filter only thins
    what the track accepts; age is checked against the last ACCEPTED capture).  It must lie between the centre's own
    exit and the deployed board's pinhole prediction (the last padded tag quad
    leaving), each widened by ``overshoot_tolerance_m``.  The difference to
    the pinhole prediction is the tag-vs-centre record of guide 7.1.
    """
    board = deployed_board(SpatialConfig(schema=FLIGHT_SCHEMA))
    rows = [_exit_row(ep, config, board) for ep in episodes]
    gaps = [r["overshoot_minus_pinhole_m"] for r in rows
            if r.get("overshoot_minus_pinhole_m") is not None]
    return {"passed": bool(rows) and all(r["passed"] for r in rows),
            "probe": _plain(config.g3.local.fov_exit), "episodes": rows,
            "summary": {
                "overshoot_at_last_capture_m": _stats(
                    [r["overshoot_at_last_capture_m"] for r in rows
                     if r.get("overshoot_at_last_capture_m") is not None]),
                "overshoot_minus_pinhole_m": _stats(gaps),
                "front_valid_with_negative_margin": sum(
                    r.get("front_valid_with_negative_margin", 0) for r in rows)},
            "note": ("the local detector drops a tag by angular sums (CALIBRATION_GAPS "
                     "edge_visibility_rule), so its stop precedes the pinhole prediction; "
                     "Isaac's edge behaviour is not recorded")}


# ------------------------------------------------- 3. calibration identity

class _ZeroNoise:
    """Corner-noise stream of zeros, so the plant's projection can be compared exactly."""

    @staticmethod
    def normal(loc: float, scale: float, size: Any) -> np.ndarray:
        return np.zeros(size)


def _physical(relative_pad_minus_own: Sequence[float],
              angles: Sequence[float]) -> CurrentPhysical:
    ex, ey, ez = relative_pad_minus_own
    roll, pitch, yaw = angles
    return CurrentPhysical(ex=ex, ey=ey, h=-ez, vrx=0.0, vry=0.0, vgx=0.0, vgy=0.0, vz=0.0,
                           roll=roll, pitch=pitch, yaw=yaw, p=0.0, q=0.0, r=0.0,
                           vision_updated=True, vision_age_s=0.0, nav_valid=True,
                           nav_age_s=0.0, estimate_valid=True)


def check_calibration_identity(config: CurrentSemanticsConfig, *, poses: int = 64
                               ) -> dict[str, Any]:
    """The local plant's two hard-coded cameras equal ``CameraCalibration.nominal_v12``.

    Constants are compared directly; the mount and R_CB, which the plant
    inlines, are compared by behaviour: ``spatial.core.camera_bearings``
    (detection and bearings) and ``LocalBackend._marker_corners`` (pixel
    corners, called with a zero-noise stream) against ``geometry.camera_kinematics``
    over deterministic poses drawn from the first pilot seed.
    """
    cal = config.calibration
    cfg = SpatialConfig(schema=FLIGHT_SCHEMA)
    width, height = REFERENCE_OPTICAL_IMAGE_PX
    local = {"width": width, "height": height,
             "fx": (width / 2.0) / math.tan(cfg.fov[0] / 2.0),
             "fy": (height / 2.0) / math.tan(cfg.fov[1] / 2.0),
             "cx": width / 2.0, "cy": height / 2.0,
             "hfov_rad": cfg.fov[0], "vfov_rad": cfg.fov[1]}
    nominal = {"width": cal.width, "height": cal.height, "fx": cal.fx, "fy": cal.fy,
               "cx": cal.cx, "cy": cal.cy, "hfov_rad": cal.hfov_rad, "vfov_rad": cal.vfov_rad}
    constants = {key: abs(float(local[key]) - float(nominal[key])) for key in nominal}

    rng = np.random.default_rng(config.g3.pilot_seeds[0])
    depth_error = tangent_error = 0.0
    for _ in range(poses):
        own_minus_pad = np.r_[rng.uniform(-1.0, 1.0, 2), rng.uniform(0.5, 3.0)]
        angles = np.r_[rng.uniform(-0.3, 0.3, 2), rng.uniform(-math.pi, math.pi)]
        q = Rotation.from_euler("xyz", angles).as_quat()[[3, 0, 1, 2]]
        bearing, depth = camera_bearings(own_minus_pad, q, cfg)
        camera = camera_kinematics(_physical(-own_minus_pad, angles), cal)
        x, y, z = camera.p_c
        depth_error = max(depth_error, abs(depth - z))
        tangent_error = max(tangent_error,
                            float(np.max(np.abs(np.tan(bearing) - [x / z, y / z]))))
    level = camera_bearings(np.array([0.0, 0.0, 2.0]), np.array([1.0, 0.0, 0.0, 0.0]), cfg)[1]

    board, ids = deployed_board(cfg), deployed_board_ids(cfg)
    plant = SimpleNamespace(cfg=cfg, board=board, corner_rng=_ZeroNoise())
    corner_error, corners_compared = 0.0, 0
    for _ in range(16):
        own_minus_pad = np.r_[rng.uniform(-0.3, 0.3, 2), rng.uniform(1.0, 2.5)]
        angles = np.r_[rng.uniform(-0.1, 0.1, 2), rng.uniform(-math.pi, math.pi)]
        q = Rotation.from_euler("xyz", angles).as_quat()[[3, 0, 1, 2]]
        projected = LocalBackend._marker_corners(plant, own_minus_pad, q, True)
        for k, (tx, ty, side) in enumerate(board):
            marker = int(ids[k])
            if marker not in projected:
                continue
            half = side / 2.0
            square = ((tx - half, ty + half), (tx + half, ty + half),
                      (tx + half, ty - half), (tx - half, ty - half))
            for (cx, cy), pixel in zip(square, projected[marker]):
                camera = camera_kinematics(
                    _physical(np.r_[cx, cy, 0.0] - own_minus_pad, angles), cal)
                corner_error = max(corner_error, float(np.max(np.abs(
                    np.array([camera.u, camera.v]) - pixel))))
                corners_compared += 1
    behaviour = {"bearing_depth_m": depth_error, "bearing_tangent": tangent_error,
                 "marker_corner_px": corner_error,
                 "level_mount_m": abs((2.0 - level) + cal.t_bc_flu[2])}
    passed = (max(constants.values()) <= IDENTITY_TOLERANCE
              and max(behaviour.values()) <= IDENTITY_TOLERANCE and corners_compared > 0)
    return {"passed": passed, "local": local, "nominal": nominal,
            "constant_abs_error": constants, "behaviour_max_abs_error": behaviour,
            "poses": poses, "corners_compared": corners_compared,
            "calibration_status": cal.status, "tolerance": IDENTITY_TOLERANCE,
            "edge_conventions": ("xi/eta margins reach 0 at u = W-1, v = H-1 (guide 7.1); "
                                 "camera_bearings visibility ends at |X/Z| = W/(2fx) (u = W) and "
                                 "_marker_corners at u <= W-1: sub-pixel, recorded not changed"),
            "note": ("identity with the NOMINAL K; Isaac swaps in renderer intrinsics at runtime, "
                     "so the Isaac half of G3 must re-check them on the flown stack")}


# ------------------------------------------- 4. low-altitude blind zone

def blind_zone_episodes(config: CurrentSemanticsConfig, seeds: Sequence[int] | None = None,
                        *, backend: Any = None) -> list[Episode]:
    probe = config.g3.local.low_altitude_blind_zone
    dt = SpatialConfig(schema=FLIGHT_SCHEMA).dt
    return [fly(config, seed, "descent", _descent_policy(probe, dt), probe.max_steps,
                backend=backend)
            for seed in _seeds(config, seeds)]


def check_low_altitude_blind_zone(config: CurrentSemanticsConfig,
                                  episodes: Sequence[Episode]) -> dict[str, Any]:
    """Body height at the last accepted frame and at contact, against Isaac's record.

    Report-only (``passed`` None) once measured: the Isaac numbers are a
    calibration gap, not a pass mark.  ``passed`` is False only when a probe
    descent did not reach contact with a frame accepted on the way.
    """
    gap = CALIBRATION_GAPS["touchdown_body_height"]
    isaac_low, isaac_high = gap.numbers["isaac_first_contact_body_height_m"]
    mount_z = config.calibration.t_bc_flu[2]
    rows = []
    for episode in episodes:
        D = episode.decisions
        k_last = _last_update(episode)
        contact = (episode.termination or {}).get("reason") == "truth_contact"
        if not contact or k_last is None:
            rows.append({"seed": episode.seed, "measured": False,
                         "termination": episode.termination, "k_last_update": k_last})
            continue
        times = [d.time_s for d in D]
        heights = [d.evaluator["relative_position_own_minus_pad_m"][2] for d in D]
        offsets = [float(np.hypot(*d.evaluator["relative_position_own_minus_pad_m"][:2]))
                   for d in D]
        capture = float(D[k_last].provenance["capture_stamp_s"])
        at_capture = _at(times, heights, capture)
        end = D[-1]
        rows.append({
            "seed": episode.seed, "measured": True, "termination": episode.termination,
            "last_capture_s": capture, "contact_s": end.time_s,
            "blind_interval_s": end.time_s - capture,
            "body_height_at_last_frame_m": at_capture,
            "estimated_height_at_last_update_m": D[k_last].physical.h,
            "lateral_offset_at_last_frame_m": _at(times, offsets, capture),
            "body_height_at_contact_m": heights[-1],
            "estimated_height_at_contact_m": end.physical.h,
            "lateral_offset_at_contact_m": offsets[-1],
            "contact_vertical_speed_m_s": end.evaluator["relative_velocity_own_minus_pad_m_s"][2],
            "camera_height_above_pad_at_contact_m": heights[-1] + mount_z,
            "contact_height_minus_isaac_m": (heights[-1] - isaac_high, heights[-1] - isaac_low)})
    measured = [r for r in rows if r["measured"]]
    return {"passed": None if rows and len(measured) == len(rows) else False,
            "probe": _plain(config.g3.local.low_altitude_blind_zone), "episodes": rows,
            "summary": {
                "body_height_at_last_frame_m": _stats([r["body_height_at_last_frame_m"]
                                                       for r in measured]),
                "body_height_at_contact_m": _stats([r["body_height_at_contact_m"]
                                                    for r in measured]),
                "blind_interval_s": _stats([r["blind_interval_s"] for r in measured]),
                "lateral_offset_at_last_frame_m": _stats([r["lateral_offset_at_last_frame_m"]
                                                          for r in measured])},
            "isaac": gap.to_dict(),
            "note": ("local contact is declared at local_plant.contact_height_m, the median "
                     "of Isaac's 78 legless first contacts; verify the blind interval in Isaac")}


# ---------------------------------------------------- 5. quality channels

def check_quality_channels(config: CurrentSemanticsConfig,
                           episodes: Sequence[Episode]) -> dict[str, Any]:
    """vision_updated/age must be the track's acceptances and capture stamps; nav is reported.

    At every decision: the updated flag equals "the observer accepted a frame
    since the previous decision" and "the last accepted capture changed";
    with an estimate, age == decision - capture of the last accepted frame and
    capture <= receive <= decision; without one, the age channel is exactly
    the sentinel 1.0.
    """
    tolerance = config.g3.local.quality_channels.age_tolerance_s
    counts: Counter = Counter()
    frames: Counter = Counter()
    age_error = 0.0
    ages_on_update, transports, in_view = [], [], []
    for episode in episodes:
        previous = None
        for k, d in enumerate(episode.decisions):
            ph, pv = d.physical, d.provenance
            counts["decisions"] += 1
            counts["navigation_constant"] += ph.nav_valid and ph.nav_age_s == 0.0
            counts["flag_mismatch"] += ph.vision_updated != (pv["frames_since_decision"] > 0)
            counts["estimate_mismatch"] += ph.estimate_valid != pv["estimate_initialized"]
            capture = pv["capture_stamp_s"]
            if not ph.estimate_valid:
                counts["uninitialised"] += 1
                counts["sentinel_mismatch"] += d.vector[_VISION_AGE] != 1.0
                previous = capture
                continue
            age_error = max(age_error, abs(ph.vision_age_s - (d.time_s - capture)))
            counts["stamp_order_violation"] += not (capture <= pv["receive_stamp_s"] <= d.time_s)
            changed = capture != previous
            counts["capture_mismatch"] += ph.vision_updated != changed
            frames[pv["frames_since_decision"]] += 1
            if ph.vision_updated:
                ages_on_update.append(ph.vision_age_s)
                transports.append(pv["receive_stamp_s"] - capture)
            truth = _truth_camera(d, config)
            if truth.front_valid and min(truth.margins) > 0:
                in_view.append(ph.vision_updated)
            previous = capture
    rate = CALIBRATION_GAPS["optical_update_rate"]
    passed = (counts["decisions"] > 0 and age_error <= tolerance
              and not any(counts[key] for key in ("flag_mismatch", "estimate_mismatch",
                                                  "sentinel_mismatch", "stamp_order_violation",
                                                  "capture_mismatch")))
    return {"passed": passed, "counts": dict(counts), "max_age_error_s": age_error,
            "age_tolerance_s": tolerance,
            "frames_per_decision": {str(key): frames[key] for key in sorted(frames)},
            "updated_fraction_with_centre_in_view": (float(np.mean(in_view)) if in_view
                                                     else None),
            "age_at_updated_decision_s": _stats(ages_on_update),
            "receive_minus_capture_s": _stats(transports),
            "navigation": {"constant": counts["navigation_constant"] == counts["decisions"],
                           "note": NAVIGATION_NOTE},
            "isaac": rate.to_dict(),
            "note": ("the local plant delivers its own 20 Hz (thinning was measured and "
                     "rejected, CALIBRATION_GAPS optical_update_rate); compare this fraction "
                     "with the Isaac half at the observer")}


# ------------------------------------------------- 6. observation purity

def check_observation_purity(config: CurrentSemanticsConfig,
                             episodes: Sequence[Episode]) -> dict[str, Any]:
    """Features at a decision are a function of that decision's 21D vector alone:
    rebuilt in reverse order, interleaved with other decisions and as one batch,
    every frame is byte-identical to the one built during the flight."""
    decisions = [d for episode in episodes for d in episode.decisions]
    mismatches = 0
    for i, d in enumerate(reversed(decisions)):
        build_features(decisions[i].vector, config)
        again = build_features(d.vector.copy(), config)
        mismatches += not (again.flat.tobytes() == d.frame.flat.tobytes()
                           and again.nodes.tobytes() == d.frame.nodes.tobytes()
                           and d.frame.raw.tobytes() == np.asarray(d.vector).tobytes())
    batch_equal = False
    if decisions:
        flat, nodes = build_feature_batch(np.stack([d.vector for d in decisions]), config)
        recorded = lambda field: np.stack([getattr(d.frame, field) for d in decisions]).tobytes()
        batch_equal = flat.tobytes() == recorded("flat") and nodes.tobytes() == recorded("nodes")
    return {"passed": bool(decisions) and mismatches == 0 and batch_equal,
            "decisions": len(decisions), "rebuild_mismatches": mismatches,
            "batch_equal": batch_equal,
            "note": "over every decision of the FOV-exit and descent flights"}


# ------------------------------------------- Isaac half: non-paired checks

def check_axis_signs(config: CurrentSemanticsConfig,
                     episodes: Sequence[Episode]) -> dict[str, Any]:
    """Attitude follows the commanded acceleration with the Isaac signs.

    A real flight is not repeatable, so the paired local protocol cannot run;
    instead, over every decision, the adapter-limited command at one decision
    is correlated with the attitude the 21D vector reports at the next:
    sin(pitch) must rise with +a_x and sin(roll) fall with +a_y, each with
    |corr| >= ``g3_isaac.axis_sign_min_abs_correlation``.
    """
    ax, ay, sin_pitch, sin_roll = [], [], [], []
    for episode in episodes:
        D = episode.decisions
        for a, b in zip(D, D[1:]):
            if a.trace is None:
                continue
            limited = a.trace.adapter_limited_accel
            ax.append(limited[0]); ay.append(limited[1])
            sin_pitch.append(float(b.frame.flat[_FI["drone_sinTheta"]]))
            sin_roll.append(float(b.frame.flat[_FI["drone_sinPhi"]]))

    def corr(x: list[float], y: list[float]) -> float | None:
        if len(x) < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
            return None
        return float(np.corrcoef(x, y)[0, 1])
    pitch, roll = corr(ax, sin_pitch), corr(ay, sin_roll)
    floor = config.g3.isaac.axis_sign_min_abs_correlation
    passed = pitch is not None and roll is not None and pitch >= floor and roll <= -floor
    return {"passed": passed, "decisions": len(ax),
            "corr_sin_pitch_vs_ax": pitch, "corr_sin_roll_vs_ay": roll, "floor": floor,
            "note": "Isaac convention: +a_x tilts the nose down (sin pitch > 0), +a_y rolls left"}


def check_frame_rate(config: CurrentSemanticsConfig,
                     episodes: Sequence[Episode]) -> dict[str, Any]:
    """Report-only: the optical cadence AT THE OBSERVER (frames per decision and
    the spacing of accepted captures), the number decision-level traces cannot
    give and the local plant's optical_cadence waits for."""
    frames: Counter = Counter()
    spacing: list[float] = []
    for episode in episodes:
        last = None
        for d in episode.decisions:
            pv = d.provenance
            frames[pv["frames_since_decision"]] += 1
            capture = pv["capture_stamp_s"]
            if pv["frame_used"] and capture is not None:
                if last is not None:
                    spacing.append(round(float(capture) - last, 4))
                last = float(capture)
    total = sum(frames.values())
    return {"passed": None,
            "frames_per_decision": {str(key): frames[key] for key in sorted(frames)},
            "mean_frames_per_decision": (sum(k * v for k, v in frames.items()) / total
                                         if total else None),
            "accepted_capture_spacing_s": _stats(spacing),
            "local_reference": CALIBRATION_GAPS["optical_update_rate"].to_dict(),
            "note": ("set local_plant.optical_cadence from this measurement, then re-run the "
                     "local half; thinning without it locked the reference track out")}


def run_isaac_checks(config: CurrentSemanticsConfig, backend: Any, *,
                     seeds: Sequence[int] | None = None) -> dict[str, Any]:
    """The Isaac half of G3 through an open ``CurrentSpatialBackend`` (the caller
    owns the stack and closes the backend).  Runs the FOV-exit and descent
    probe flights on ``g3_isaac.seeds`` and every check that does not need a
    repeatable flight; the paired axis protocol is replaced by axis_signs.
    Any backend works, so the logic is testable on the local plant.
    """
    chosen = tuple(config.g3.isaac.seeds if seeds is None else seeds)
    foreign = sorted(set(chosen) - set(config.g3.isaac.seeds))
    if not chosen or foreign:
        raise ValueError(f"the Isaac half runs only g3_isaac.seeds; refused {foreign or chosen}")
    exits = fov_exit_episodes(config, chosen, backend=backend)
    descents = blind_zone_episodes(config, chosen, backend=backend)
    flights = [*exits, *descents]
    checks = {
        "axis_signs": check_axis_signs(config, flights),
        "fov_exit": check_fov_exit(config, exits),
        "low_altitude_blind_zone": check_low_altitude_blind_zone(config, descents),
        "quality_channels": check_quality_channels(config, flights),
        "observation_purity": check_observation_purity(config, flights),
        "frame_rate": check_frame_rate(config, flights),
    }
    terminations = [{"label": e.label, "seed": e.seed, "termination": e.termination}
                    for e in flights]
    cleanup_ok = all(t["termination"] is None or t["termination"]["cleanup_confirmed"]
                     for t in terminations)
    checks["termination_and_cleanup"] = {
        "passed": cleanup_ok, "episodes": terminations,
        "note": "a terminal verdict and its cleanup are separate events"}
    manifest = contract_manifest(config)
    return _plain({
        "schema_id": SCHEMA_ID, "stage": "G3 Isaac coupling checks",
        "backend": getattr(backend, "name", None),
        "profile_sha256": getattr(getattr(backend, "flight", None), "profile_sha256", None),
        "scientific_use": False, "seeds": list(chosen),
        "hashes": {key: manifest[key] for key in ("g3_hash", "algorithm_hash", "task_hash",
                                                  "calibration_hash", "action_hash")},
        "isaac_checks_passed": all(check["passed"] is not False for check in checks.values()),
        "checks": checks,
    })


# ------------------------------------------------------------------ report

def _plain(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {name: _plain(getattr(value, name)) for name in value.__dataclass_fields__}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run_local_checks(config: CurrentSemanticsConfig, *,
                     seeds: Sequence[int] | None = None) -> dict[str, Any]:
    """All six local G3 checks as a JSON-able report ``{check: {passed, ...}}``.

    ``passed`` None marks a report-only check.  ``local_checks_passed`` is
    not G3: the Isaac half is not run here.
    """
    chosen = _seeds(config, seeds)
    exits = fov_exit_episodes(config, chosen)
    descents = blind_zone_episodes(config, chosen)
    checks = {
        "axis_coupling": check_axis_coupling(config, chosen),
        "fov_exit": check_fov_exit(config, exits),
        "calibration_identity": check_calibration_identity(config),
        "low_altitude_blind_zone": check_low_altitude_blind_zone(config, descents),
        "quality_channels": check_quality_channels(config, [*exits, *descents]),
        "observation_purity": check_observation_purity(config, [*exits, *descents]),
    }
    manifest = contract_manifest(config)
    return _plain({
        "schema_id": SCHEMA_ID, "stage": "G3 local coupling checks",
        "backend": "spatial_local_v12",
        "backend_status": BACKENDS["spatial_local_v12"].status,
        "difficulty": 1.0, "scientific_use": False,
        "isaac_half": "not run: requires a user-approved owned Isaac/PX4 stack",
        "seeds": list(chosen),
        "hashes": {key: manifest[key] for key in ("g3_hash", "algorithm_hash",
                                                  "calibration_hash", "action_hash")},
        "local_checks_passed": all(checks[name]["passed"] is not False for name in CHECKS),
        "checks": checks,
        "calibration_gaps": {name: gap.to_dict() for name, gap in CALIBRATION_GAPS.items()},
    })
