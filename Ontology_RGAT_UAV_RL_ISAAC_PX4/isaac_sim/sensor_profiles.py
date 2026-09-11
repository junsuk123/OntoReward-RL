"""Validated hardware profiles used by the Isaac/Pegasus sensor adapters.

The values in this module are expressed in SI units.  Keeping the conversion
outside ``landing_world`` makes it possible to unit-test what is actually sent
to Pegasus without importing Isaac Sim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


VN100_MODEL = "vectornav_vn100"
ZED2I_MONO_MODEL = "stereolabs_zed2i_mono"
SHIN2026_CAMERA_MODEL = "shin2026_grayscale"


@dataclass(frozen=True)
class IsaacRuntimeProfile:
    """Render settings selected without importing Isaac Sim.

    Physics remains at ``isaac.physics_dt`` in every profile.  Only rendered
    frames are decimated in a GUI run; PX4 HIL IMU/GPS callbacks continue to
    run on physics time.
    """

    rendering_dt: float
    startup_rendering_dt: float
    startup_max_sim_s: float
    viewport_resolution: tuple[int, int]
    camera_rate_hz: int


def isaac_runtime_profile(data: dict[str, Any], headless: bool) -> IsaacRuntimeProfile:
    """Resolve the GUI/headless render budget and validate its timing."""
    isaac = data.get("isaac") or {}
    camera = (data.get("vision") or {}).get("camera") or {}
    physics_dt = float(isaac.get("physics_dt", 0.0))
    sensor_rendering_dt = float(isaac.get("rendering_dt", 0.0))
    rendering_dt = (sensor_rendering_dt if headless else
                    float(isaac.get("gui_rendering_dt", sensor_rendering_dt)))
    startup_rendering_dt = float(
        isaac.get("startup_rendering_dt", rendering_dt))
    startup_max_sim_s = float(isaac.get("startup_max_sim_s", 0.0))
    viewport = tuple(int(v) for v in isaac.get(
        "gui_viewport_resolution", (1280, 720)))
    nominal_camera_rate = float(camera.get("rate_hz", 0.0))

    values = (physics_dt, sensor_rendering_dt, rendering_dt,
              startup_rendering_dt, nominal_camera_rate)
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("Isaac physics, rendering and camera rates must be positive and finite")
    if rendering_dt + 1e-12 < physics_dt:
        raise ValueError("Isaac rendering period cannot be shorter than physics_dt")
    if startup_rendering_dt + 1e-12 < rendering_dt:
        raise ValueError("isaac.startup_rendering_dt cannot be shorter than the run-time rendering period")
    if not math.isfinite(startup_max_sim_s) or startup_max_sim_s < 0.0:
        raise ValueError("isaac.startup_max_sim_s must be finite and non-negative")
    if len(viewport) != 2 or min(viewport) < 320:
        raise ValueError("isaac.gui_viewport_resolution must contain two values of at least 320 pixels")

    # A camera cannot produce more independent frames than the stage renders.
    # Use an integer frequency because Isaac's Camera API requires one.
    render_rate = max(1, int(round(1.0 / rendering_dt)))
    camera_rate = max(1, min(int(round(nominal_camera_rate)), render_rate))
    return IsaacRuntimeProfile(
        rendering_dt=rendering_dt,
        startup_rendering_dt=startup_rendering_dt,
        startup_max_sim_s=startup_max_sim_s,
        viewport_resolution=(viewport[0], viewport[1]),
        camera_rate_hz=camera_rate,
    )


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


def validate_camera_profile(camera: dict[str, Any]) -> None:
    """Validate either the retained ZED profile or the benchmark profile."""
    model = str(camera.get("model", ZED2I_MONO_MODEL)).lower()
    if model == ZED2I_MONO_MODEL:
        validate_zed2i_mono(camera)
        return
    if model != SHIN2026_CAMERA_MODEL:
        raise ValueError(f"unsupported vision.camera.model {model!r}")
    if tuple(int(v) for v in camera.get("resolution", ())) != (512, 320):
        raise ValueError("Shin-2026 camera resolution must be 512x320")
    if abs(float(camera.get("horizontal_fov_deg", 0.0)) - 90.0) > 1e-9:
        raise ValueError("Shin-2026 camera horizontal FOV must be 90 degrees")
    if abs(float(camera.get("pitch_down_deg", 0.0)) - 60.0) > 1e-9:
        raise ValueError("Shin-2026 camera pitch must be 60 degrees downward")
    if float(camera.get("rate_hz", 0.0)) <= 0.0:
        raise ValueError("Shin-2026 camera rate must be positive")
