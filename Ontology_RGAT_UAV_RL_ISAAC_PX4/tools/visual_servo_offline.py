#!/usr/bin/env python3
"""Fly the image-servo teacher offline, to tune it without the simulator.

A teacher tuning costs a run of real Isaac/PX4 flights, and the servo spent
seven of them landing nothing. This closes the loop on a model instead: the
projection (``pad_image_position``) and the control law
(``_visual_servo_teacher_action``) are the repository's own, and only the
vehicle and the encoder are modelled.

    python3 tools/visual_servo_offline.py                 # the ablation table
    python3 tools/visual_servo_offline.py --seeds 64      # tighter intervals

What is modelled, and why each part is here:

* **Airframe tilt.** A multirotor delivers a velocity command by tilting, and
  the camera is bolted to the airframe, so a correction swings the image by
  about ``atan(a / g)``. Without this the model landed every flight under the
  gains that had just landed none in flight; with it, the shipped gains land
  1% -- which is the record this tool exists to explain.
* **Velocity tracking lag**, first order, 0.35 s by default.
* **Encoder**: centroid from the true projection plus gaussian noise, an
  apparent scale going as 1 / range, and the documented visible-landmark
  fractions (6 above 2 m, 4 at 1.2-1.5 m, 2 below 1.0 m).
* **Deck**: the configured cruise speed, optionally with the escape burst's
  1.0 m/s dash.

What it cannot do: prove a landing. Contact, the entry gate, PX4's own
controller, the real encoder's failure modes and the deck's random walk are
all outside it. It measures loop properties -- hunting, wind-up, whether a
correction fights itself -- which is what a tuning decision needs and what a
30-flight simulator run answers far more slowly.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from ontology_rgat.initialization import (nadir_image_setpoint,   # noqa: E402
                                          pad_image_position)
from ontology_rgat.perception.semantic_observation import (       # noqa: E402
    SemanticObservation)
from run_three_pipeline import _visual_servo_teacher_action       # noqa: E402

LEVEL = (1.0, 0.0, 0.0, 0.0)
CAMERA = {"horizontal_fov_deg": 90.0, "pitch_down_deg": 60.0,
          "image_size": (512, 320), "mount_translation_flu_m": (0.10, 0.0, -0.16)}
SETPOINT = nadir_image_setpoint(CAMERA["horizontal_fov_deg"],
                                CAMERA["pitch_down_deg"])
TAN_HALF_H = math.tan(math.radians(CAMERA["horizontal_fov_deg"]) / 2.0)
TAN_HALF = (TAN_HALF_H, TAN_HALF_H * 320 / 512)
LIMIT = np.array([2.0, 2.0, 1.0])       # controller.max_velocity * action_scale

# behavior_cloning as the experiment configures it, with the two knobs this
# tool exists to compare left explicit.
SHIPPED = dict(
    position_gain=.55, integral_gain=.20, damping_gain=.35, integral_limit=3.0,
    horizontal_speed_limit=.60, reference_scale=.06, alignment_tolerance=.30,
    cone_widening=1.2, rate_tolerance=.60, approach_descent=.50,
    descent_rate=.30, flare_descent=.25, descent_floor=.12, approach_scale=.12,
    flare_scale=.20, rate_filter_s=0.0, integral_leak_s=0.0, anti_windup=False,
)


def tilt_quaternion(acceleration):
    """The attitude a multirotor holds to command that acceleration."""
    pitch = math.atan2(float(acceleration[0]), 9.81)
    roll = -math.atan2(float(acceleration[1]), 9.81)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    rotation = np.array([[cp, sp * sr, sp * cr],
                         [0.0, cr, -sr],
                         [-sp, cp * sr, cp * cr]])
    w = math.sqrt(max(1.0 + float(np.trace(rotation)), 1e-12)) / 2.0
    return (w, (rotation[2, 1] - rotation[1, 2]) / (4 * w),
            (rotation[0, 2] - rotation[2, 0]) / (4 * w),
            (rotation[1, 0] - rotation[0, 1]) / (4 * w))


def visible_fraction(altitude: float, in_frame: bool) -> float:
    """Landmarks in frame, from the geometry the encoder work measured."""
    if not in_frame:
        return 0.0
    if altitude >= 2.0:
        return 1.0
    if altitude >= 1.2:
        return 4.0 / 6.0
    if altitude >= 1.0:
        return 3.0 / 6.0
    return 2.0 / 6.0


def observe(relative, rng, *, noise, attitude=LEVEL):
    column_row = pad_image_position(relative, attitude, **CAMERA)
    in_frame = (column_row is not None and abs(column_row[0]) <= 1.0
                and abs(column_row[1]) <= 1.0)
    centroid = (np.asarray(column_row, dtype=float) if column_row is not None
                else np.array([1.0, 0.0]))
    if noise > 0.0:
        centroid = centroid + rng.normal(0.0, noise, 2)
    # The encoder cannot report outside its own frame; a pad past the edge is
    # carried by visual_loss_risk, as in flight.
    centroid = np.clip(centroid, -1.0, 1.0)
    raw = 0.20 / max(float(np.linalg.norm(relative)), 0.05)
    return SemanticObservation(
        keypoint_confidence=0.9,
        visible_keypoint_fraction=visible_fraction(float(relative[2]), in_frame),
        image_alignment=0.5, apparent_target_scale=min(1.0, 1.25 * raw),
        image_plane_motion_safety=0.9, scale_rate_safety=0.9,
        visibility_memory=1.0, reacquisition_trend=0.5,
        vertical_motion_safety=0.9, attitude_stability=0.9, battery_risk=0.1,
        visual_loss_risk=0.0 if in_frame else 1.0,
        centroid_xy=(float(centroid[0]), float(centroid[1])), raw_scale=raw)


def burst_speed(step, base, *, start=80, length=60, peak=1.0):
    """The escape burst: a dash at ``peak`` that ramps in and eases off."""
    if step < start or step >= start + length:
        return base
    ramp = min(1.0, (step - start) / 10.0, (start + length - step) / 10.0)
    return base + (peak - base) * ramp


def fly(*, gains, seed=0, steps=600, dt=0.1, entry=(-1.2, .6, 5.), tilt=True,
        deck_speed=.366, deck_heading=.35, tau=.35, noise=.02, burst=False):
    """One demonstration flight. Returns the per-step trace."""
    rng = np.random.default_rng(seed)
    relative = np.array(entry, dtype=float)       # UAV in the pad's frame
    velocity, acceleration = np.zeros(3), np.zeros(3)
    heading = np.array([math.cos(deck_heading), math.sin(deck_heading), 0.0])
    deck = deck_speed * heading
    integral, image_rate = np.zeros(2), np.zeros(2)
    previous, committed, trace = None, False, []
    for step in range(steps):
        if burst:
            deck = burst_speed(step, deck_speed) * heading
        semantic = observe(relative, rng, noise=noise,
                           attitude=tilt_quaternion(acceleration) if tilt else LEVEL)
        action, integral, committed, image_rate = _visual_servo_teacher_action(
            semantic, previous, LIMIT, setpoint=SETPOINT, dt=dt,
            integral=integral, tan_half=TAN_HALF, committed_already=committed,
            image_rate=image_rate, noise_std=0.0, rng=rng, **gains)
        previous = semantic
        command = np.asarray(action[:3], dtype=float) * LIMIT
        acceleration = (command - velocity) / tau
        velocity = velocity + acceleration * dt
        relative = relative + (velocity - deck) * dt
        relative[2] = max(float(relative[2]), 0.0)
        trace.append({
            "step": step, "lateral": float(np.linalg.norm(relative[:2])),
            "altitude": float(relative[2]),
            "integral": float(np.linalg.norm(integral))})
        if relative[2] <= 0.02:
            break
    return trace


def summarise(trace, *, integral_limit):
    lateral = np.array([row["lateral"] for row in trace])
    altitude = np.array([row["altitude"] for row in trace])
    late = lateral[len(lateral) // 2:]
    return {
        "landed": bool(altitude[-1] <= 0.05 and lateral[-1] <= 0.35),
        "swing": float(late.max() - late.min()) if late.size else float("nan"),
        "pinned": bool(max(row["integral"] for row in trace)
                       >= 0.95 * float(integral_limit)),
    }


def score(gains, *, seeds, noise, tau, burst, deck_speed=.366, tilt=True):
    """Landing rate, median late-flight swing and how often the integral pins."""
    landed, swings, pinned = 0, [], 0
    for seed in range(seeds):
        rng = np.random.default_rng(1000 + seed)
        row = summarise(fly(
            gains=gains, seed=seed, tilt=tilt, noise=noise, tau=tau,
            burst=burst, deck_speed=deck_speed,
            deck_heading=float(rng.uniform(0, 2 * math.pi)),
            entry=(-1.0 - .6 * rng.random(), 1.2 * (rng.random() - .5),
                   3.0 + 3.0 * rng.random())),
            integral_limit=gains["integral_limit"])
        landed += row["landed"]
        swings.append(row["swing"])
        pinned += row["pinned"]
    return landed / seeds, float(np.median(swings)), pinned / seeds


VARIANTS = {
    "shipped 2026-09-19 (raw d/dt)": dict(SHIPPED),
    "+ rate_filter_s 0.30": dict(SHIPPED, rate_filter_s=.30),
    "+ filter + anti_windup": dict(SHIPPED, rate_filter_s=.30, anti_windup=True),
    "+ filter + AW + 1.20 m/s": dict(SHIPPED, rate_filter_s=.30, anti_windup=True,
                                     horizontal_speed_limit=1.20,
                                     integral_gain=.32),
    "+ filter + AW + leak 12 s": dict(SHIPPED, rate_filter_s=.30, anti_windup=True,
                                      integral_leak_s=12.0),
}

CONDITIONS = (("quiet", .02, .35, False), ("dash", .02, .35, True),
              ("noisy", .04, .35, True), ("laggy", .02, .60, True),
              ("both", .04, .60, True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=32,
                        help="flights per condition (default 32)")
    parser.add_argument("--no-tilt", action="store_true",
                        help="level camera: shows why the tilt has to be modelled")
    arguments = parser.parse_args()
    names = "  ".join(f"{name:>10}" for name, *_ in CONDITIONS)
    print(f"{'variant':<32} {names}   {'mean':>5}")
    for label, gains in VARIANTS.items():
        cells = [score(gains, seeds=arguments.seeds, noise=noise, tau=tau,
                       burst=burst, tilt=not arguments.no_tilt)
                 for _name, noise, tau, burst in CONDITIONS]
        print(f"{label:<32} "
              + " ".join(f"{landed:>5.0%}({pinned:>3.0%})"
                         for landed, _swing, pinned in cells)
              + f"   {np.mean([c[0] for c in cells]):>5.0%}"
              + f"  swing~{np.median([c[1] for c in cells]):.2f} m")
    print(f"\n{arguments.seeds} flights per cell; cell = landed(integral pinned "
          "to its clamp).\nquiet = cruise only; the rest carry the escape "
          "burst's 1.0 m/s dash.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
