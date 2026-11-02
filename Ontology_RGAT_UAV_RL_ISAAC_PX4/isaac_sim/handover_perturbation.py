"""The seeded policy-handover velocity/rate perturbation, applied so the IMU can see it.

The domain-randomization handover kick (``initial_velocity_m_s``, U(-1, 1) m/s per
body axis) used to be written as ONE velocity step. The VN-100 model differentiates
successive velocities and clips at its +-16 g range, so in one 4 ms physics tick it
can report at most 0.63 m/s per axis, and PX4 SITL's int16 FIFO turned a value
pinned at +16 g (FRD) into -16 g (sensor_profiles.saturate_16bit). The EKF own
velocity then started up to 1.6 m/s wrong, on 358 of 700 recorded current-semantics
Isaac episodes with the WRONG SIGN, and the policy acted on it. Landed: kick
measurable 104/110, clipped only 207/232, inverted 61/358 (112 of the 115 handover
tilts). The local plant applies the same kick exactly and lands 90-94 % of each.

The step is now spread over the fewest physics ticks that keep each tick's
acceleration inside a fraction of the accelerometer range. At most a few ticks
(<= 16 ms at 250 Hz) against a 0.1 s decision, so the kick the policy meets is
unchanged; only the part PX4 could not measure is gone.

Pure numpy: no Isaac import, so the arithmetic is unit tested.
"""

from __future__ import annotations

import math

import numpy as np

# Share of the accelerometer range one tick of the kick may use. The rest is
# headroom for gravity on the vertical axis, rotor thrust and sensor noise.
RANGE_FRACTION = 0.5


def perturbation_ticks(body_velocity_m_s, physics_dt: float, accel_range_m_s2: float,
                       range_fraction: float = RANGE_FRACTION) -> int:
    """Fewest ticks that keep every body axis of the step inside the budget."""
    if not (math.isfinite(physics_dt) and physics_dt > 0.0):
        raise ValueError("physics_dt must be positive and finite")
    if not (math.isfinite(accel_range_m_s2) and accel_range_m_s2 > 0.0):
        raise ValueError("accelerometer range must be positive and finite")
    if not 0.0 < range_fraction <= 1.0:
        raise ValueError("range_fraction must lie in (0, 1]")
    per_tick = range_fraction * accel_range_m_s2 * physics_dt
    largest = float(np.max(np.abs(np.asarray(body_velocity_m_s, dtype=float))))
    return max(1, int(math.ceil(largest / per_tick - 1e-12)))


def perturbation_increments(world_velocity_m_s, world_rate_rad_s, ticks: int):
    """Equal per-tick increments; their sum is exactly the requested step."""
    if ticks < 1:
        raise ValueError("ticks must be >= 1")
    velocity = np.asarray(world_velocity_m_s, dtype=float) / ticks
    rate = np.asarray(world_rate_rad_s, dtype=float) / ticks
    return [(velocity.copy(), rate.copy()) for _ in range(ticks)]
