import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "isaac_sim"))

from config_loader import load_config
from metasejong_scene import MetaSejongConfig
from pad_motion import PadMotionConfig, PadTrajectory


def test_overlay_deep_merges_system_configuration():
    config = load_config(ROOT / "config" / "metasejong-demo.yaml")

    assert config["metasejong"]["scenario"] == "demo"
    assert config["metasejong"]["hide_vegetation"] is True
    assert config["pad"]["motion"] == "waypoints"
    assert config["pad"]["carrier"] == "ugv"
    assert config["pad"]["preview_motion"] is True
    assert config["pad"]["preview_speed_scale"] == pytest.approx(1.0)
    assert config["battery"]["enabled"] is False
    assert len(config["pad"]["route_waypoints_enu_m"]) == 44
    assert config["pad"]["arena_radius_m"] == 0.0
    assert config["wind"]["enabled"] is False
    assert config["isaac"]["physics_dt"] == pytest.approx(0.004)
    assert config["system"]["state_timeout_s"] == pytest.approx(1.5)
    assert config["network"]["gateway_port"] == 14650

    trajectory = PadTrajectory(PadMotionConfig.from_mapping(config))
    position, velocity = trajectory.pose(0.0)
    assert position.tolist() == pytest.approx([-65.0, 131.0, 16.896])
    assert velocity.tolist() == [0.0, 0.0, 0.0]

    # Once the idle preview releases the deck, the waypoint shuttle visibly
    # accelerates away from its first stop without leaving the audited route.
    trajectory.reset(
        seed=49, sim_time=0.0,
        speed_scale=config["pad"]["preview_speed_scale"])
    preview_position, preview_velocity = trajectory.pose(20.0)
    assert preview_position.tolist() != pytest.approx([-65.0, 131.0, 16.896])
    assert sum(v * v for v in preview_velocity[:2]) > 0.01


def test_pipeline_overlay_restores_episode_control_and_battery():
    config = load_config(ROOT / "config" / "metasejong-pipeline.yaml")

    assert config["metasejong"]["enabled"] is True
    assert config["pad"]["carrier"] == "ugv"
    assert config["pad"]["preview_motion"] is False
    assert config["battery"]["enabled"] is True


def test_alias_and_asset_path_are_resolved_from_workspace(tmp_path):
    config = MetaSejongConfig.from_mapping(
        {
            "metasejong": {
                "enabled": True,
                "scenario": "jipyhyeon",
                "asset_root": "assets",
            }
        },
        tmp_path,
    )
    assert config.scenario == "jiphyeon"
    assert config.hide_vegetation is False
    assert config.usd_path == tmp_path / "assets/playground/S4/SejongUniv_S4.usd"
    with pytest.raises(FileNotFoundError, match="import_metasejong_map"):
        config.validate_assets()


def test_unknown_scenario_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="metasejong.scenario"):
        MetaSejongConfig.from_mapping(
            {"metasejong": {"scenario": "not-a-campus"}}, tmp_path
        )
