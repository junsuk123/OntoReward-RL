"""The handover kick must reach PX4 through an IMU that can measure it.

Isaac wrote the seeded handover velocity step in ONE 4 ms physics tick. The VN-100
model (isaac_sim/vn100_imu.py) differentiates successive velocities and clipped at
+-16 g, i.e. at most 0.63 m/s per axis per tick, and PX4 SITL's int16 FIFO turned
a value pinned at +16 g (FRD) into -16 g. Over the 700 recorded current-semantics
Isaac episodes: kick measurable 104/110 landed, clipped only 207/232, inverted
61/358 with 112 of the 115 handover tilts (local plant on the same cases: 90-94 %).
That model predicts PX4's first-decision velocity change on all 600
isaac-development episodes to a median 0.03 m/s per axis with every sign right.
This replays the IMU arithmetic on the old and the new application.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "isaac_sim"))
sys.path.insert(0, str(ROOT / "python"))

from config_loader import load_config  # noqa: E402
from handover_perturbation import perturbation_increments, perturbation_ticks  # noqa: E402
from sensor_profiles import saturate_16bit, vn100_pegasus_config  # noqa: E402

from ontology_rgat.benchmarks.randomization import sample_domain_randomization  # noqa: E402

G = np.array([0.0, 0.0, -9.80665])  # ENU, as vn100_imu.GRAVITY_VECTOR is used there
# Every seed flown on Isaac by this route (isaac-eval/1-/5, isaac-development/1)
# and the two spatial-reference flight seeds.
FLOWN = list(range(994001, 994051)) + list(range(994101, 994131)) + [12000, 12001]


@pytest.fixture(scope="module")
def imu():
    config = load_config(ROOT / "config" / "spatial-isaac-system-v12.yaml")
    dt = float(config["isaac"]["physics_dt"])
    return dt, float(vn100_pegasus_config(config, dt)["accelerometer"]["measurement_range"])


def reported_step(increments, dt, accel_range):
    """Velocity change PX4 integrates from the clipped specific force, level vehicle."""
    reported = np.zeros(3)
    for step in increments:
        specific_force = np.clip(step / dt - G, -accel_range, accel_range)
        reported += (specific_force + G) * dt
    return reported


def test_one_tick_step_was_clipped_for_most_flown_kicks(imu):
    dt, accel_range = imu
    lost = [np.linalg.norm(k - reported_step([k], dt, accel_range))
            for k in (np.asarray(sample_domain_randomization(s).initial_velocity_m_s)
                      for s in FLOWN)]
    assert accel_range * dt == pytest.approx(0.628, abs=0.001)
    assert np.mean(np.asarray(lost) > 1e-9) > 0.7
    assert max(lost) > 0.5


@pytest.mark.parametrize("seed", FLOWN + list(range(5000, 5300)))
def test_spread_step_is_measured_exactly(imu, seed):
    dt, accel_range = imu
    sample = sample_domain_randomization(seed)
    kick = np.asarray(sample.initial_velocity_m_s, dtype=float)
    ticks = perturbation_ticks(kick, dt, accel_range)
    steps = perturbation_increments(kick, sample.initial_angular_rate_rad_s, ticks)
    np.testing.assert_allclose(sum(v for v, _ in steps), kick, atol=1e-12)
    np.testing.assert_allclose(sum(r for _, r in steps),
                               sample.initial_angular_rate_rad_s, atol=1e-12)
    np.testing.assert_allclose(reported_step([v for v, _ in steps], dt, accel_range),
                               kick, atol=1e-12)
    # Instantaneous at the 0.1 s decision scale: U(-1, 1) needs at most 4 ticks.
    assert 1 <= ticks <= 4 and ticks * dt <= 0.016 + 1e-12


def test_within_range_kick_still_takes_one_tick(imu):
    dt, accel_range = imu
    assert perturbation_ticks([0.2, -0.1, 0.05], dt, accel_range) == 1
    assert perturbation_ticks([0.0, 0.0, 0.0], dt, accel_range) == 1


def px4_fifo_counts(values, full_scale):
    """SimulatorMavlink.cpp: int16 FIFO sample = float(value) / (full_scale / 32768)."""
    scale = np.float32(np.float32(full_scale) / np.float32(32768.0))
    return (np.asarray(values, dtype=np.float32) / scale).astype(np.int32).astype(np.int16)


@pytest.mark.parametrize("full_scale", [16 * 9.80665, np.radians(2000.0)])
def test_saturated_sensor_never_reaches_px4_with_the_wrong_sign(full_scale):
    over = np.array([full_scale, 1.5 * full_scale, -full_scale, -1.5 * full_scale])
    # The old model: np.clip at +full scale wrapped to the most negative count.
    wrapped = px4_fifo_counts(np.clip(over, -full_scale, full_scale), full_scale)
    assert wrapped[0] == wrapped[1] == -32768
    counts = px4_fifo_counts(saturate_16bit(over, full_scale), full_scale)
    np.testing.assert_array_equal(np.sign(counts), np.sign(over))
    assert counts.max() == 32767 and counts.min() == -32768


def test_rejects_meaningless_arguments():
    with pytest.raises(ValueError):
        perturbation_ticks([1, 0, 0], 0.0, 156.9)
    with pytest.raises(ValueError):
        perturbation_ticks([1, 0, 0], 0.004, -1.0)
    with pytest.raises(ValueError):
        perturbation_increments([1, 0, 0], [0, 0, 0], 0)
