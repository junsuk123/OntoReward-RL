"""The common observation O_t = {P_t, D_t, H_t} and its 3D instantiation.

P_t: per-marker image corners in fixed id slots plus a detection mask.
D_t: the drone's own fused navigation state. H_t: previous (P, D) records and
the last record in which anything was seen. No estimate, no judgement.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from ontology_rgat.landing.common_observation import (
    CommonObservationBuilder, drone_state_2d, drone_state_3d, normalize_corners_px)
from ontology_rgat.spatial.core import Measurement, SpatialConfig
from ontology_rgat.spatial.environment import SpatialLandingEnv, deployed_board_ids

ROOT = Path(__file__).resolve().parents[1]
QUAD = np.array([[100.0, 100.0], [140.0, 100.0], [140.0, 140.0], [100.0, 140.0]])


def _drone(t):
    return drone_state_3d([1.0, 2.0, 3.0], [0.1, 0.0, -0.2], [1.0, 0, 0, 0], [0, 0, 0.1], t)


def test_corners_are_normalized_by_image_size_only():
    out = normalize_corners_px([[0.0, 0.0], [639.0, 479.0], [319.5, 239.5]], (640, 480))
    assert np.allclose(out, [[-1, -1], [1, 1], [0, 0]])


def test_ids_are_slots_not_features_and_unknown_ids_are_dropped():
    builder = CommonObservationBuilder([10, 11, 12], (640, 480))
    a = builder.pad_observation({12: QUAD, 10: QUAD + 5}, 1.0, 1.05)
    b = builder.pad_observation({10: QUAD + 5, 12: QUAD, 99: QUAD}, 1.0, 1.05)
    assert np.array_equal(a.features(), b.features())
    assert list(a.detected_mask) == [1.0, 0.0, 1.0]
    assert np.all(a.corners[1] == 0)                     # undetected slot is empty
    assert a.features().shape == (3, 9)
    # corner order is the detector's, never re-sorted
    assert np.allclose(a.corners[2], normalize_corners_px(QUAD, (640, 480)))


def test_a_stale_or_missing_frame_is_invalid_and_carries_no_corners():
    builder = CommonObservationBuilder([10], (640, 480), max_image_age_s=0.5)
    stale = builder.pad_observation({10: QUAD}, 1.0, 1.6)
    none = builder.pad_observation(None, None, 1.6)
    for pad in (stale, none):
        assert not pad.frame_valid and pad.detected_mask.sum() == 0 and not pad.corners.any()


def test_history_holds_previous_records_and_the_last_detection_only():
    builder = CommonObservationBuilder([10, 11], (640, 480), history_length=2)
    seen = builder.observe(builder.pad_observation({10: QUAD}, 0.0, 0.1), _drone(0.1))
    assert seen.history.previous == (None, None) and seen.history.last_seen is None
    miss1 = builder.observe(builder.pad_observation({}, 0.1, 0.2), _drone(0.2))
    miss2 = builder.observe(builder.pad_observation({}, 0.2, 0.3), _drone(0.3))
    miss3 = builder.observe(builder.pad_observation({}, 0.3, 0.4), _drone(0.4))
    assert miss1.history.previous[0].drone.navigation_stamp == 0.1      # newest first
    assert [r.drone.navigation_stamp for r in miss3.history.previous] == [0.3, 0.2]
    # the detection left the K-window but is still the last-seen record,
    # together with the drone state at that moment
    assert miss3.history.last_seen.pad.detected_mask[0] == 1.0
    assert miss3.history.last_seen.drone.navigation_stamp == 0.1
    # records are (P, D): no history inside history
    assert not hasattr(miss3.history.previous[0], "history")
    builder.reset()
    fresh = builder.observe(builder.pad_observation({}, 1.0, 1.1), _drone(1.1))
    assert fresh.history.previous == (None, None) and fresh.history.last_seen is None


def test_vector_layout_and_dimensions_in_both_dimensions():
    builder = CommonObservationBuilder(list(range(5)), (640, 480), history_length=2)
    obs = builder.observe(builder.pad_observation({0: QUAD}, 0.95, 1.0), _drone(1.0))
    m, d = 5, 15
    assert obs.drone.features().size == d
    assert obs.layout()["size"] == obs.vector().size == (3 + 9 * m + d) + 2 * (4 + 9 * m + d) + (4 + 9 * m + d)
    assert obs.vector()[1] == pytest.approx(0.05)            # image age, not a stamp
    two_d = drone_state_2d(1.0, 2.0, 0.1, -0.2, 0.3, 0.05, 1.0)
    assert two_d.features().size == 7
    assert np.allclose(two_d.features()[4:6], [math.sin(0.3), math.cos(0.3)])


def test_measurement_carries_wire_corners_and_refuses_future_frames():
    state = {"extra": {"spatial_clock": {"valid": True, "sim_time_s": 10.0},
                       "optical_measurement": {"valid": False},
                       "marker_corners": {"capture_time_s": 9.9, "image_size": [640, 480],
                                          "markers": {"10": QUAD.tolist(), "bad": None}}},
             "estimator_valid": True, "world": {"position": [0, 0, 2], "velocity": [0, 0, 0]},
             "quaternion_wxyz": [1, 0, 0, 0], "angular_velocity": [0, 0, 0]}
    try:
        m = Measurement.from_wire(state)
    except (ValueError, TypeError):
        state["extra"]["marker_corners"]["markers"].pop("bad")
        m = Measurement.from_wire(state)
    assert m.marker_capture_time_s == pytest.approx(9.9) and list(m.marker_corners_px) == [10]
    state["extra"]["marker_corners"]["capture_time_s"] = 10.5
    assert Measurement.from_wire(state).marker_corners_px is None


def test_the_local_detector_sees_corners_exactly_when_it_sees_the_pad():
    cfg = SpatialConfig()
    env = SpatialLandingEnv(cfg)
    obs, _ = env.reset(seed=4100)
    assert obs.common is not None
    assert obs.common.pad.marker_ids == deployed_board_ids(cfg) and len(obs.common.pad.marker_ids) == 17
    agree = 0
    for _ in range(30):
        obs, _r, done, _t, info = env.step(np.array([0.05, 0.0, -0.2]))
        optical = env.estimator.own.optical_position is not None
        assert optical == (info["common_markers_detected"] > 0)
        agree += 1
        if done:
            break
    assert agree > 10


def test_local_corners_follow_the_isaac_detectors_conventions():
    """With noise off, the Isaac detector's own PnP on the local corners
    recovers the true pose: same corner order, axes and camera mount."""
    pytest.importorskip("cv2")
    sys.path.insert(0, str(ROOT / "isaac_sim"))
    from marker_vision import MarkerBoard, MarkerPoseEstimator, intrinsics_from_fov
    import ontology_rgat.spatial.environment as environment
    from ontology_rgat.spatial.runtime_contract import deployment_profile
    from scipy.spatial.transform import Rotation
    cfg = SpatialConfig()
    board = MarkerBoard.from_config(
        deployment_profile(cfg.schema)["resolved_scientific_configuration"]["vision"]["board"])
    estimator = MarkerPoseEstimator(board, intrinsics_from_fov(640, 480, 90.0), [0, 0, -0.16])
    original = environment.LOCAL_CORNER_NOISE_PX
    environment.LOCAL_CORNER_NOISE_PX = 0.0
    try:
        env = SpatialLandingEnv(cfg)
        env.reset(seed=4101)
        backend, solved = env.backend, 0
        for _ in range(15):
            env.step(np.array([0.1, -0.05, -0.15]))
            r = backend.position - backend.pad
            q = Rotation.from_euler("xyz", backend.angles).as_quat()[[3, 0, 1, 2]]
            corners = backend._marker_corners(r, q, True)
            if not corners:
                continue
            object_points = np.concatenate([board.object_points(i) for i in corners])
            image_points = np.concatenate([corners[i] for i in corners])
            pose = estimator._solve(object_points, image_points)
            observation = estimator._observation(*pose, 1.0, tuple(corners))
            assert np.allclose(observation.position_pad_enu, r, atol=1e-4)
            solved += 1
        assert solved > 5
    finally:
        environment.LOCAL_CORNER_NOISE_PX = original


def test_rungs_without_a_deployed_board_have_no_common_observation():
    from dataclasses import replace
    env = SpatialLandingEnv(replace(SpatialConfig(), schema="spatial-reference/4"))
    obs, _ = env.reset(seed=4100)
    assert obs.common is None
