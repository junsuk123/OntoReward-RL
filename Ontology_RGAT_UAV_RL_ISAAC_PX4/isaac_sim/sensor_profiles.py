"""Validated hardware profiles used by the Isaac/Pegasus sensor adapters.

The values in this module are expressed in SI units.  Keeping the conversion
outside ``landing_world`` makes it possible to unit-test what is actually sent
to Pegasus without importing Isaac Sim.
"""

from __future__ import annotations

import math
from typing import Any


VN100_MODEL = "vectornav_vn100"
ZED2I_MONO_MODEL = "stereolabs_zed2i_mono"


def vn100_pegasus_config(data: dict[str, Any], physics_dt: float) -> dict[str, Any]:
    """Return a Pegasus IMU configuration for a VectorNav VN-100.

    VN-100 can output IMU data at 800 Hz, but a simulated sensor cannot create
    independent samples faster than the physics state it observes.  Therefore
    ``simulation_update_rate_hz`` is capped at the Isaac physics frequency;
    ``hardware_output_rate_hz`` remains available as device metadata.
    """
    imu = (data.get("imu") or {}) if isinstance(data, dict) else {}
    model = str(imu.get("model", VN100_MODEL)).lower()
    if model != VN100_MODEL:
        raise ValueError(f"unsupported imu.model {model!r}; expected {VN100_MODEL!r}")
    if not math.isfinite(physics_dt) or physics_dt <= 0.0:
        raise ValueError("isaac.physics_dt must be positive and finite")

    physics_rate = 1.0 / float(physics_dt)
    requested_rate = float(imu.get("simulation_update_rate_hz", physics_rate))
    if requested_rate <= 0.0:
        raise ValueError("imu.simulation_update_rate_hz must be positive")
    update_rate = min(requested_rate, physics_rate)

    gyro = imu.get("gyroscope") or {}
    accel = imu.get("accelerometer") or {}
    gyro_range = float(gyro.get("measurement_range_rad_s", math.radians(2000.0)))
    accel_range = float(accel.get("measurement_range_m_s2", 16.0 * 9.80665))
    if min(gyro_range, accel_range) <= 0.0:
        raise ValueError("VN-100 measurement ranges must be positive")

    return {
        "update_rate": update_rate,
        "seed": int(imu.get("seed", 101)),
        "gyroscope": {
            "noise_density": float(gyro.get(
                "noise_density_rad_s_sqrt_hz", math.radians(0.0035))),
            # Bias stability is represented as a slowly varying Gauss-Markov
            # bias. No unlisted angle-random-walk value is invented.
            "random_walk": float(gyro.get("random_walk_rad_s_sqrt_s", 0.0)),
            "bias_correlation_time": float(gyro.get("bias_correlation_time_s", 1.0e9)),
            "turn_on_bias_sigma": float(gyro.get(
                "bias_stability_rad_s", math.radians(5.0) / 3600.0)),
            "measurement_range": gyro_range,
        },
        "accelerometer": {
            "noise_density": float(accel.get(
                "noise_density_m_s2_sqrt_hz", 0.00014 * 9.80665)),
            "random_walk": float(accel.get("random_walk_m_s2_sqrt_s", 0.0)),
            "bias_correlation_time": float(accel.get("bias_correlation_time_s", 1.0e9)),
            "turn_on_bias_sigma": float(accel.get(
                "bias_stability_m_s2", 0.00004 * 9.80665)),
            "measurement_range": accel_range,
        },
    }


def validate_zed2i_mono(camera: dict[str, Any]) -> None:
    """Validate the selected ZED 2i single-eye operating point."""
    model = str(camera.get("model", ZED2I_MONO_MODEL)).lower()
    if model != ZED2I_MONO_MODEL:
        raise ValueError(
            f"unsupported vision.camera.model {model!r}; expected {ZED2I_MONO_MODEL!r}")
    eye = str(camera.get("eye", "left")).lower()
    if eye not in ("left", "right"):
        raise ValueError("ZED 2i mono eye must be 'left' or 'right'")
    resolution = tuple(int(v) for v in camera.get("resolution", ()))
    rate = float(camera.get("rate_hz", 0.0))
    supported = {
        (2208, 1242): 15.0,
        (1920, 1080): 30.0,
        (1280, 720): 60.0,
        (672, 376): 100.0,
    }
    if resolution not in supported or rate > supported[resolution] + 1e-9:
        raise ValueError(
            "unsupported ZED 2i per-eye mode; use 2208x1242@15, "
            "1920x1080@30, 1280x720@60, or 672x376@100")
    fov = float(camera.get("horizontal_fov_deg", 0.0))
    if not 1.0 < fov <= 110.0:
        raise ValueError("ZED 2i 2.1 mm horizontal FOV must be in (1, 110] degrees")
