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
    assert config["pad"]["vehicle_model"] == "agilex_ranger_mini_v3"
    assert config["pad"]["vehicle_dimensions_m"] == pytest.approx([0.720, 0.500, 0.345])
    assert config["pad"]["preview_motion"] is True
    assert config["pad"]["preview_speed_scale"] == pytest.approx(1.0)
    assert config["pad"]["vehicle_max_speed_m_s"] == pytest.approx(1.0)
    assert config["pad"]["speed_range_m_s"] == pytest.approx([0.25, 0.60])
    assert config["battery"]["enabled"] is False
    assert len(config["pad"]["route_waypoints_enu_m"]) == 44
    assert config["pad"]["arena_radius_m"] == 0.0
    assert config["wind"]["enabled"] is False
    assert config["isaac"]["physics_dt"] == pytest.approx(0.004)
    assert config["system"]["state_timeout_s"] == pytest.approx(1.5)
    assert config["network"]["gateway_port"] == 14650

    trajectory = PadTrajectory(PadMotionConfig.from_mapping(config))
    position, velocity = trajectory.pose(0.0)
    assert position.tolist() == pytest.approx([-65.0, 131.0, 16.566])
    assert velocity.tolist() == [0.0, 0.0, 0.0]

    # Once the idle preview releases the deck, the waypoint shuttle visibly
    # accelerates away from its first stop without leaving the audited route.
    trajectory.reset(
        seed=49, sim_time=0.0,
        speed_scale=config["pad"]["preview_speed_scale"])
    preview_position, preview_velocity = trajectory.pose(20.0)
    assert preview_position.tolist() != pytest.approx([-65.0, 131.0, 16.566])
    assert sum(v * v for v in preview_velocity[:2]) > 0.01


def test_pipeline_overlay_restores_episode_control_and_battery():
    config = load_config(ROOT / "config" / "metasejong-pipeline.yaml")

    assert config["metasejong"]["enabled"] is True
    assert config["pad"]["carrier"] == "ugv"
    assert config["pad"]["preview_motion"] is False
    assert config["battery"]["enabled"] is True


def test_shin_profile_uses_campus_plaza_and_fitted_platform():
    config = load_config(ROOT / "config" / "shin2026-system.yaml")
    pad = PadMotionConfig.from_mapping(config)

    assert config["metasejong"]["enabled"] is True
    assert config["metasejong"]["scenario"] == "gwanggaeto"
    assert config["metasejong"]["hide_vegetation"] is True
    assert pad.carrier == "ugv"
    assert pad.deck_size_m == pytest.approx((1.5, 1.5))
    assert pad.deck_height_m == pytest.approx(0.42)
    assert pad.vehicle_max_speed_m_s == pytest.approx(1.0)
    assert (pad.speed_min_m_s, pad.speed_max_m_s) == pytest.approx((0.25, 0.60))
    assert pad.mode == "waypoints"
    assert pad.waypoint_loop is True
    assert pad.route_start == "continue"
    assert pad.arena_radius_m == 0.0
    assert len(pad.route_waypoints_enu_m) == 37
    assert pad.route_waypoints_enu_m[0] == pytest.approx(
        (-41.765, -75.569, 21.360))
    assert pad.route_waypoints_enu_m[-1] == pytest.approx(
        pad.route_waypoints_enu_m[0])
    assert (ROOT / pad.vehicle_visual_usd).is_file()
    assert config["isaac"]["start_airborne"] is True
    # At or above Table II's per-axis initial-velocity randomization, so the
    # handover gate is never stricter than the paper's own initial condition.
    assert config["benchmark"]["entry_speed_tolerance_m_s"] == pytest.approx(1.0)
    assert config["benchmark"]["entry_settle_s"] == pytest.approx(1.0)
    assert config["isaac"]["viewport_follow"]["focus"] == "pair"
    operator_view = config["parallel"]["operator_view"]
    assert operator_view["focus"] == "group"
    assert operator_view["viewport_resolution"] == [640, 360]
    assert operator_view["pair_span_m"] == pytest.approx(22.0)
    assert config["px4"]["sitl_parameters"] == {
        "COM_RC_IN_MODE": 4,
        "COM_RCL_EXCEPT": 4,
        "COM_OF_LOSS_T": 30.0,
        "COM_OBL_RC_ACT": 5,
        # Strictly above landing.crash_tilt_deg, so the learner -- not PX4 --
        # decides when a tilt has become a crash.
        "FD_FAIL_R": 85,
        "FD_FAIL_P": 85,
        # Forwarding a tip-over instead of raising means the run keeps flying
        # the same PX4, which a latched flight termination would make unarmable.
        "CBRK_FLIGHTTERM": 121212,
        "MPC_THR_HOVER": 0.58,
        "MPC_USE_HTE": 0,
        "MPC_XY_VEL_MAX": 2.0,
        "MPC_Z_VEL_MAX_UP": 1.0,
        "MPC_Z_VEL_MAX_DN": 1.0,
        "MPC_ACC_HOR_MAX": 1.5,
        "MPC_ACC_UP_MAX": 1.0,
        "MPC_ACC_DOWN_MAX": 1.0,
        "SIM_BAT_MIN_PCT": 100.0,
    }
    assert config["battery"]["enabled"] is True
    assert config["battery"]["capacity_mah"] == 3500
    assert config["battery"]["nominal_voltage_v"] == pytest.approx(11.1)
    assert config["battery"]["episode_hover_seconds_range"] == pytest.approx(
        [9.0, 55.0])

    half_length, half_width = (0.5 * value for value in pad.deck_size_m)
    # The deployed policy is keypoint-based, so the deck carries six fixed
    # landmarks rather than a bit-coded tag board.
    from keypoint_geometry import pad_landmarks

    assert config["vision"]["mode"] == "keypoint_fiducial"
    assert not config["vision"].get("dictionary")
    assert not config["vision"].get("board")
    landing_pad = config["vision"]["landing_pad"]
    assert landing_pad["layout"] == "hexagonal"
    landmarks = pad_landmarks(float(landing_pad["landmark_radius_m"]))
    assert landmarks.shape == (6, 3)
    half_landmark = 0.5 * float(landing_pad["landmark_diameter_m"])
    for x, y, _ in landmarks:
        assert abs(float(x)) + half_landmark <= half_length + 1e-9
        assert abs(float(y)) + half_landmark <= half_width + 1e-9

    # Landmarks may not touch: overlapping bullseyes would make two of the six
    # keypoints ambiguous exactly where the encoder has to separate them.
    for index, first in enumerate(landmarks):
        for second in landmarks[index + 1:]:
            separation = float(
                ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5)
            assert separation > 2.0 * half_landmark


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
