"""Regression tests for the requested flight/ground hardware profiles."""

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "isaac_sim"))

from config_loader import load_config  # noqa: E402
from gnss import GnssConfig, GnssFix, hil_gps_measurement  # noqa: E402
from sensor_profiles import validate_zed2i_mono, vn100_pegasus_config  # noqa: E402


@pytest.fixture(scope="module")
def config():
    return load_config(ROOT / "config" / "metasejong-pipeline.yaml")


def test_zed_f9p_multi_constellation_rtk_profile(config):
    receiver = GnssConfig.from_mapping(config)
    assert receiver.model == "ublox_zed_f9p_05b"
    assert receiver.constellations == ("GPS", "GLONASS", "Galileo", "BeiDou")
    assert receiver.update_rate_hz == pytest.approx(5.0)
    assert receiver.rtk_convergence_s == pytest.approx(10.0)
    assert receiver.rtk_horizontal_accuracy_m == pytest.approx(0.01)
    assert receiver.rtk_vertical_accuracy_m == pytest.approx(0.01)

    fixed = GnssFix(valid=True, fix_type=6, satellites_tracked=18,
                    hdop=1.0, vdop=1.6, sigma_xy_m=0.01)
    sample = hil_gps_measurement(
        fixed, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 37.5, 127.0, 20.0, receiver)
    assert sample["fix_type"] == 6
    assert sample["eph"] == 1
    assert sample["epv"] == 1


def test_vn100_values_are_converted_to_pegasus_si(config):
    profile = vn100_pegasus_config(config, config["isaac"]["physics_dt"])
    assert config["imu"]["hardware_output_rate_hz"] == pytest.approx(800.0)
    assert config["imu"]["attitude_output_rate_hz"] == pytest.approx(400.0)
    assert profile["update_rate"] == pytest.approx(250.0)
    assert profile["gyroscope"]["measurement_range"] == pytest.approx(math.radians(2000.0))
    assert profile["gyroscope"]["noise_density"] == pytest.approx(math.radians(0.0035))
    assert profile["accelerometer"]["measurement_range"] == pytest.approx(16.0 * 9.80665)
    assert profile["accelerometer"]["noise_density"] == pytest.approx(0.00014 * 9.80665)


def test_zed2i_left_mono_mode_is_a_supported_per_eye_mode(config):
    camera = config["vision"]["camera"]
    validate_zed2i_mono(camera)
    assert camera["eye"] == "left"
    assert camera["resolution"] == [1280, 720]
    assert camera["rate_hz"] == pytest.approx(60.0)
    assert camera["horizontal_fov_deg"] == pytest.approx(110.0)
    assert config["isaac"]["rendering_dt"] == pytest.approx(1.0 / 60.0)


def test_ranger_asset_was_generated_from_the_pinned_official_source(config):
    pad = config["pad"]
    assert pad["vehicle_model"] == "agilex_ranger_mini_v3"
    assert pad["vehicle_mass_kg"] == pytest.approx(75.0)
    assert pad["vehicle_payload_kg"] == pytest.approx(120.0)
    assert pad["vehicle_max_speed_m_s"] == pytest.approx(2.0)
    assert (ROOT / pad["vehicle_visual_usd"]).is_file()
    assert (ROOT / "assets/ranger_mini_v3/source/urdf/ranger_mini.xacro").is_file()
