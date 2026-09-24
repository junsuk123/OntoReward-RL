#!/usr/bin/env python3
"""Fly the PN guidance law offline, to check it before it costs a stack boot.

The first PN demonstration stage, 2026-09-23, spent an Isaac boot, four PX4
instances, a keypoint fine-tune and four flights to discover that the law flew
backwards. This closes the loop on a model instead: the projection
(``pad_image_position``), the envelope (``PlanarLongitudinalController``) and
the law (``PNGuidanceController``) are the repository's own, and only the
airframe and the encoder are modelled.

    python3 tools/pn_guidance_offline.py                    # the three decks
    python3 tools/pn_guidance_offline.py --seeds 64         # tighter intervals
    python3 tools/pn_guidance_offline.py --reference-scale 0.06   # the old one

What is modelled, and why each part is here:

* **The encoder's range signal.** ``raw_scale = k / range``, with ``k = 0.65``
  measured from 256 in-flight frames in which the pad was fully visible between
  4.3 m and 14 m (sd 0.07). The same fit read as an altitude signal scatters
  four times as widely, which is the measurement that says this is a range
  signal and the shipped 0.06 was an altitude constant eleven times out.
* **Airframe tilt.** The camera is bolted to the airframe, so a longitudinal
  acceleration swings the image by about ``atan(a / g)``. Without it a law can
  look stable in the model and hunt in flight.
* **Velocity tracking lag**, first order, 0.35 s: the envelope emits a velocity
  setpoint and PX4 takes time to reach it.
* **Blindness.** Out of frame, the encoder holds its last centroid and scale
  and reports zero apparent scale and full loss risk -- what
  ``semantic_observation`` actually does -- so the search branch is exercised.
* **Visibility, measured rather than assumed.** 2026-09-24, over 2818 frames of
  flight telemetry: with the pad centre geometrically in frame the encoder
  reports 0.85-1.00 of its landmarks at EVERY altitude, below one metre
  included. It does not lose landmarks on approach. What it loses is the frame
  -- at 0.0-1.0 m the pad centre was inside it in 38 of 210 samples, at
  1.0-1.5 m in 111 of 190, above 3 m in all 292 -- so visibility here is a
  geometric test and nothing else. The ladder this tool first inherited from
  the image-servo harness (six landmarks above 2 m, four at 1.2-1.5 m, two
  below 1.0 m) predates the U-Net v4 encoder, and believing it is what made
  this model forty times too optimistic: 73% against a stack that flew 1 in 60.
* **Battery**, because it is what actually ends these flights: 37 of those 60
  attempts died on it. The step budget is drawn per flight from the depletion
  steps they recorded, so a law that takes 40 s to arrive fails here for the
  same reason it failed there.
* **Deck**: ``pad_motion.segmented_cruise_speeds``, the run's own profile.

What it cannot do: prove a landing. Contact, the entry gate, PX4's controller,
the real encoder's failure modes and the domain randomisation are all outside
it. It measures loop properties -- does a correction point at the pad, does the
approach converge, does the flare commit where it should -- which is what
decides whether a stack boot is worth spending.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "isaac_sim"))

from config_loader import load_config                              # noqa: E402
import pad_motion                                                  # noqa: E402

from ontology_rgat.benchmarks.experiment import load_experiment     # noqa: E402
from ontology_rgat.controllers import (PlanarLongitudinalController,  # noqa: E402
                                       PNGuidanceConfig,
                                       PNGuidanceController,
                                       PNGuidanceState)
from ontology_rgat.initialization import (nadir_image_setpoint,     # noqa: E402
                                          pad_image_position)
from ontology_rgat.perception.semantic_observation import (         # noqa: E402
    SemanticObservation)

SYSTEM = load_config(ROOT / "config/shin2026-planar-system.yaml")
EXPERIMENT = load_experiment(
    ROOT / "config/experiments/planar_three_arm_comparison.yaml")
CAMERA_CFG = dict((SYSTEM.get("vision") or {}).get("camera") or {})
LANDING = dict(SYSTEM.get("landing") or {})
CAMERA = {
    "horizontal_fov_deg": float(CAMERA_CFG.get("horizontal_fov_deg", 90.0)),
    "pitch_down_deg": float(CAMERA_CFG.get("pitch_down_deg", 60.0)),
    "image_size": tuple(CAMERA_CFG.get("resolution", (512, 320))),
    "mount_translation_flu_m": tuple(
        CAMERA_CFG.get("mount_translation_flu_m", (0.0, 0.0, -0.16))),
}
SETPOINT = nadir_image_setpoint(CAMERA["horizontal_fov_deg"],
                                CAMERA["pitch_down_deg"])
TAN_HALF = math.tan(math.radians(CAMERA["horizontal_fov_deg"]) / 2.0)
LEVEL = (1.0, 0.0, 0.0, 0.0)

# raw_scale * range, measured in flight. See the module docstring.
MEASURED_RANGE_SCALE = 0.65
APPARENT_DIVISOR = 0.75          # semantic_observation's own normalisation


def tilt_quaternion(longitudinal_acceleration: float):
    """The attitude a multirotor holds to command that acceleration."""
    pitch = math.atan2(float(longitudinal_acceleration), 9.81)
    cp, sp = math.cos(pitch), math.sin(pitch)
    rotation = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    w = math.sqrt(max(1.0 + float(np.trace(rotation)), 1e-12)) / 2.0
    return (w, (rotation[2, 1] - rotation[1, 2]) / (4 * w),
            (rotation[0, 2] - rotation[2, 0]) / (4 * w),
            (rotation[1, 0] - rotation[0, 1]) / (4 * w))


# Battery step budgets, as the 60 real attempts of 2026-09-24 recorded them
# (min 113, median 328, max 571). Drawn per flight rather than averaged: the
# initial reserve is randomised between 0.31 and 1.00, so a short budget is a
# case the teacher has to survive, not an outlier to smooth away.
BATTERY_STEPS = (113, 198, 236, 322, 328, 404, 442, 458, 465, 495, 571)


def visible_fraction(altitude: float, in_frame: bool) -> float:
    """What the encoder reports, measured -- see the module docstring.

    Geometry decides whether there is anything to report; the altitude only
    trims the fraction once the pad begins to overflow the frame.
    """
    if not in_frame:
        return 0.0
    return 0.85 if altitude < 1.0 else 0.97


def observe(relative, previous, rng, *, noise, attitude, range_scale, dt):
    """One encoder frame, including what it reports when it sees nothing."""
    column_row = pad_image_position(relative, attitude, **CAMERA)
    in_frame = (column_row is not None and abs(column_row[0]) <= 1.0
                and abs(column_row[1]) <= 1.0)
    altitude = float(relative[2])
    fraction = visible_fraction(altitude, in_frame)
    if fraction > 0.0:
        centroid = np.asarray(column_row, dtype=float)
        if noise > 0.0:
            centroid = centroid + rng.normal(0.0, noise, 2)
        centroid = np.clip(centroid, -1.0, 1.0)
        raw = range_scale / max(float(np.linalg.norm(relative)), 0.05)
        apparent = min(1.0, raw / APPARENT_DIVISOR)
        loss_duration = 0.0
        loss_risk = 0.0
    else:
        # Blind: hold the last image location and scale, report the loss.
        centroid = (np.zeros(2) if previous is None
                    else np.asarray(previous.centroid_xy, dtype=float))
        raw = 0.0 if previous is None else float(previous.raw_scale)
        apparent = 0.0
        loss_duration = dt + (0.0 if previous is None
                              else previous.visual_loss_duration_s)
        loss_risk = 1.0
    return SemanticObservation(
        keypoint_confidence=0.9, visible_keypoint_fraction=fraction,
        image_alignment=0.5, apparent_target_scale=apparent,
        image_plane_motion_safety=0.9, scale_rate_safety=0.9,
        visibility_memory=1.0 if fraction > 0 else 0.0, reacquisition_trend=0.5,
        vertical_motion_safety=0.9, attitude_stability=0.9, battery_risk=0.1,
        visual_loss_risk=loss_risk,
        centroid_xy=(float(centroid[0]), float(centroid[1])),
        raw_scale=raw, visual_loss_duration_s=loss_duration)


def build_guidance(controller, *, settings, range_scale):
    """The binding ``_pn_guidance_controller`` makes, without the simulator."""
    return PNGuidanceController(
        PNGuidanceConfig(
            navigation_gain=float(settings.get("pn_navigation_gain", 3.0)),
            approach_speed_m_s=float(settings.get("pn_approach_speed_m_s", .45)),
            approach_gain=float(settings.get("pn_approach_gain", .60)),
            closing_gain=float(settings.get("pn_closing_gain", 1.20)),
            vertical_gain=float(settings.get("pn_vertical_gain", 2.00)),
            climb_reference_m_s=float(settings.get("pn_climb_reference_m_s", .35)),
            reference_scale=float(range_scale),
            flare_range_m=float(settings.get("pn_flare_range_m", 1.20)),
            alignment_tolerance=float(
                settings.get("pn_alignment_tolerance", .80)),
            flare_descent_m_s=float(settings.get("pn_flare_descent_m_s", .25)),
            descent_floor_m_s=float(settings.get("pn_descent_floor_m_s", .10)),
            noise_std=0.0),
        nadir_column=float(SETPOINT[0]), tan_half_horizontal=TAN_HALF,
        max_longitudinal_acceleration_m_s2=float(
            controller.max_acceleration[0] * controller.action_scale),
        max_vertical_acceleration_m_s2=float(
            controller.max_acceleration[1] * controller.action_scale),
        max_longitudinal_tilt_rad=float(
            controller.max_longitudinal_tilt * controller.action_scale),
        dt=float(controller.dt))


def fly(*, scenario, seed, settings, control, curriculum, range_scale,
        model_scale=MEASURED_RANGE_SCALE, steps=600, tau=.35, noise=.02,
        tilt=True):
    """One demonstration flight against one deck. Returns its trace."""
    rng = np.random.default_rng(int(seed))
    dt = float(control.get("dt_seconds", .1))
    controller = PlanarLongitudinalController.from_mapping(control, dt=dt)
    controller.set_curriculum(float(curriculum))
    # ``range_scale`` is what the LAW believes; ``model_scale`` is what the
    # encoder actually does. Holding them apart is the whole point of the
    # --reference-scale sweep: passing one value to both cancels the error out.
    guidance = build_guidance(controller, settings=settings,
                              range_scale=range_scale)
    state = PNGuidanceState()

    altitude = float(rng.uniform(3.0, 5.0))
    # Entry: the camera-centred hover point, plus the planar along-track draw.
    stand_off = altitude / math.tan(math.radians(CAMERA["pitch_down_deg"]))
    relative = np.array([-(stand_off + float(rng.uniform(-1.5, 1.5))), 0.0,
                         altitude])
    times = np.arange(steps + 1) * dt
    deck = pad_motion.segmented_cruise_speeds(times, scenario)

    velocity = np.zeros(2)          # actual (longitudinal, vertical), m/s
    battery_steps = int(rng.choice(BATTERY_STEPS))
    previous, longitudinal_acceleration, trace = None, 0.0, []
    for step in range(min(steps, battery_steps)):
        attitude = (tilt_quaternion(longitudinal_acceleration) if tilt
                    else LEVEL)
        semantic = observe(relative, previous, rng, noise=noise,
                           attitude=attitude, range_scale=model_scale, dt=dt)
        previous = semantic
        body_velocity = np.array([velocity[0], 0.0, velocity[1]])
        action = guidance.action(semantic, body_velocity, state, rng=rng)
        command = controller.command(action)
        longitudinal_acceleration = command.longitudinal_acceleration_m_s2
        setpoint = np.array([command.velocity_body_heading_m_s[0],
                             command.velocity_body_heading_m_s[2]])
        velocity = velocity + (setpoint - velocity) * (dt / tau)
        relative[0] += (velocity[0] - float(deck[step])) * dt
        relative[2] = max(relative[2] + velocity[1] * dt, 0.0)
        trace.append({"step": step, "t": float(times[step]),
                      "along_m": float(relative[0]),
                      "altitude_m": float(relative[2]),
                      "mode": str(state.diagnostics.get("mode", "track")),
                      "range_m": float(state.diagnostics.get("range_m", 0.0)),
                      "closing_m_s": float(velocity[0] - float(deck[step])),
                      "vertical_m_s": float(velocity[1]),
                      "visible": float(semantic.visible_keypoint_fraction)})
        if relative[2] <= float(LANDING.get("ground_z_m", 0.08)):
            break
        if abs(relative[0]) > float(LANDING.get("world_xy_limit_m", 200.0)):
            break
    return trace


def summarise(trace) -> dict:
    """The run's own landing gate, applied to the last step of the flight.

    A flight still airborne when its trace ends ran out of battery, which is a
    failure however close it had come.
    """
    last = trace[-1]
    touched = last["altitude_m"] <= float(LANDING.get("ground_z_m", 0.08))
    landed = bool(
        touched
        and abs(last["along_m"]) <= float(LANDING.get("success_xy_m", .35))
        and abs(last["vertical_m_s"]) <= float(LANDING.get("success_vz_m_s", .55))
        and abs(last["closing_m_s"])
        <= float(LANDING.get("success_rel_speed_xy_m_s", .45)))
    return {
        "landed": landed, "touched": touched,
        "along_m": abs(last["along_m"]), "t_s": last["t"],
        "closing_m_s": abs(last["closing_m_s"]),
        "vertical_m_s": abs(last["vertical_m_s"]),
        "committed": any(row["mode"] == "flare" for row in trace),
        "searched": any(row["mode"] == "search" for row in trace),
        "blind_fraction": float(np.mean([row["visible"] <= 0.0
                                         for row in trace])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=24)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--reference-scale", type=float,
                        default=MEASURED_RANGE_SCALE,
                        help="what the LAW believes; the model always uses the "
                             "measured %.2f" % MEASURED_RANGE_SCALE)
    parser.add_argument("--trace", type=str, default=None,
                        help="print one flight's trace for this deck")
    args = parser.parse_args()

    settings = dict(EXPERIMENT.get("behavior_cloning") or {})
    control = dict(EXPERIMENT.get("control") or {})
    curriculum = float(settings.get("curriculum", 1.0))
    decks = [str(name) for name in (settings.get("scenarios") or [])]

    if args.trace:
        trace = fly(scenario=args.trace, seed=90000, settings=settings,
                    control=control, curriculum=curriculum,
                    range_scale=MEASURED_RANGE_SCALE, steps=args.steps)
        print(f"{'t':>5} {'along':>8} {'alt':>7} {'range':>7} {'close':>7} "
              f"{'vz':>7} {'vis':>5}  mode")
        for row in trace[::max(1, len(trace) // 30)] + [trace[-1]]:
            print(f"{row['t']:5.1f} {row['along_m']:8.2f} {row['altitude_m']:7.2f} "
                  f"{row['range_m']:7.2f} {row['closing_m_s']:7.2f} "
                  f"{row['vertical_m_s']:7.2f} {row['visible']:5.2f}  "
                  f"{row['mode']}")
        return 0

    believed = args.reference_scale
    print(f"law believes raw_scale * range = {believed:.2f}; "
          f"the modelled encoder gives {MEASURED_RANGE_SCALE:.2f}")
    print(f"curriculum {curriculum:.2f}, {args.seeds} seeds per deck, "
          f"{args.steps} steps, landing gate "
          f"{LANDING.get('success_xy_m')} m / "
          f"{LANDING.get('success_rel_speed_xy_m_s')} m/s\n")
    print(f"{'deck':>26} {'landed':>8} {'touched':>8} {'|along| med':>12} "
          f"{'t med':>7} {'blind':>7}")
    overall = []
    for deck in decks:
        rows = [summarise(fly(scenario=deck, seed=90000 + index,
                              settings=settings, control=control,
                              curriculum=curriculum, range_scale=believed,
                              steps=args.steps))
                for index in range(args.seeds)]
        overall.extend(rows)
        landed = sum(row["landed"] for row in rows)
        touched = sum(row["touched"] for row in rows)
        print(f"{deck:>26} {landed:4d}/{len(rows):<3d} {touched:4d}/{len(rows):<3d} "
              f"{statistics.median(row['along_m'] for row in rows):12.2f} "
              f"{statistics.median(row['t_s'] for row in rows):7.1f} "
              f"{statistics.mean(row['blind_fraction'] for row in rows):7.2f}")
    landed = sum(row["landed"] for row in overall)
    print(f"\ntotal {landed}/{len(overall)} "
          f"({100.0 * landed / max(1, len(overall)):.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
