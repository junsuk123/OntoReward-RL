from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "isaac_sim"))

from config_loader import load_config  # noqa: E402
from wind_sensor import WindSensor  # noqa: E402


def test_metasejong_research_pipeline_enables_wind_sensor_and_moving_pad():
    cfg = load_config(ROOT / "config" / "metasejong-pipeline.yaml")
    assert cfg["wind"]["enabled"] is True
    assert cfg["wind"]["sensor"]["enabled"] is True
    assert cfg["pad"]["motion"] == "waypoints"
    assert cfg["pad"]["preview_motion"] is False


def test_sensor_is_seeded_noisy_and_separate_from_truth():
    cfg = {"seed": 9, "noise_std_m_s": [0.1, 0.1, 0.1],
           "bias_std_m_s": [0.05, 0.05, 0.05], "time_constant_s": 0.0}
    a, b = WindSensor(cfg), WindSensor(cfg)
    a.reset(42, 0.0)
    b.reset(42, 0.0)
    truth = np.array([3.0, -1.0, 0.4])
    ma, mb = a.measure(truth, 0.1), b.measure(truth, 0.1)
    np.testing.assert_allclose(ma, mb)
    assert not np.allclose(ma, truth)


def test_sensor_response_has_lag():
    sensor = WindSensor({"noise_std_m_s": [0, 0, 0],
                         "bias_std_m_s": [0, 0, 0], "time_constant_s": 1.0})
    sensor.reset(1, 0.0)
    np.testing.assert_allclose(sensor.measure(np.zeros(3), 0.0), np.zeros(3))
    measured = sensor.measure(np.array([10.0, 0.0, 0.0]), 0.1)
    assert 0.0 < measured[0] < 10.0


def test_bad_sensor_configuration_is_rejected():
    with pytest.raises(ValueError, match="noise_std"):
        WindSensor({"noise_std_m_s": [-1.0, 0.0, 0.0]})
