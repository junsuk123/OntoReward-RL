"""Round-trip the landing-pad board through a synthetic downward camera."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "isaac_sim"))

from marker_vision import (  # noqa: E402
    R_BODY_FROM_OPTICAL,
    BoardMarker,
    MarkerBoard,
    MarkerPoseEstimator,
    generate_marker_png,
    intrinsics_from_fov,
    texture_side_ratio,
)


DICTIONARY = "DICT_4X4_50"
WIDTH, HEIGHT = 800, 600
FOV_DEG = 90.0
MOUNT = (0.0, 0.0, -0.05)

# One small tag at the pad centre for touchdown, four large ones around it for
# the approach; see the module docstring for why a single tag cannot do both.
BOARD = MarkerBoard([
    BoardMarker(10, 0.10, (0.0, 0.0)),
    BoardMarker(11, 0.45, (-0.45, +0.45)),
    BoardMarker(12, 0.45, (+0.45, +0.45)),
    BoardMarker(13, 0.45, (+0.45, -0.45)),
    BoardMarker(14, 0.45, (-0.45, -0.45)),
])


def render_board_view(body_position, yaw_rad=0.0):
    """Draw what the downward camera at BODY_POSITION sees of the pad."""
    camera_matrix = intrinsics_from_fov(WIDTH, HEIGHT, FOV_DEG)
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    r_pad_from_body = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    r_pad_from_optical = r_pad_from_body @ R_BODY_FROM_OPTICAL
    camera_in_pad = np.asarray(body_position, float) + r_pad_from_body @ np.asarray(MOUNT)
    r_optical_from_pad = r_pad_from_optical.T
    tvec = -r_optical_from_pad @ camera_in_pad
    rvec, _ = cv2.Rodrigues(r_optical_from_pad)

    image = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
    span, margin = 400, 100          # the margin is the printed quiet zone
    lo, hi = margin, margin + span - 1
    src = np.array([[lo, lo], [hi, lo], [hi, hi], [lo, hi]], dtype=np.float32)
    for marker in BOARD.markers.values():
        corners, _ = cv2.projectPoints(
            marker.object_points(), rvec, tvec, camera_matrix, np.zeros(5))
        corners = corners.reshape(4, 2).astype(np.float32)
        if not np.isfinite(corners).all():
            continue
        tag = cv2.aruco.generateImageMarker(
            cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY)),
            marker.marker_id, span)
        padded = cv2.copyMakeBorder(tag, margin, margin, margin, margin,
                                    cv2.BORDER_CONSTANT, value=255)
        warped = cv2.warpPerspective(
            padded, cv2.getPerspectiveTransform(src, corners), (WIDTH, HEIGHT),
            borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        image = np.minimum(image, warped)   # tags are dark on white
    return image, camera_matrix


def estimator():
    return MarkerPoseEstimator(
        BOARD, intrinsics_from_fov(WIDTH, HEIGHT, FOV_DEG), MOUNT, dictionary=DICTIONARY)


@pytest.mark.parametrize("body_position", [
    (0.0, 0.0, 4.6),      # episode entry altitude
    (0.4, -0.3, 3.0),
    (-0.2, 0.15, 1.5),
    (0.05, 0.05, 0.6),
    (0.0, 0.0, 0.25),     # about to touch down
])
def test_pose_round_trip_across_the_approach(body_position):
    image, _ = render_board_view(body_position)
    obs = estimator().detect(image)
    assert obs.detected, f"nothing seen at {body_position}"
    np.testing.assert_allclose(obs.position_pad_enu, body_position, atol=0.05)


def test_touchdown_altitude_still_sees_the_centre_tag():
    image, _ = render_board_view((0.0, 0.0, 0.15))
    obs = estimator().detect(image)
    assert obs.detected
    assert 10 in obs.marker_ids
    np.testing.assert_allclose(obs.position_pad_enu, (0.0, 0.0, 0.15), atol=0.03)


def test_entry_altitude_uses_the_large_tags():
    image, _ = render_board_view((0.0, 0.0, 4.6))
    obs = estimator().detect(image)
    assert obs.detected
    assert set(obs.marker_ids) >= {11, 12, 13, 14}


def test_yaw_is_recovered():
    yaw = 0.6
    image, _ = render_board_view((0.2, 0.1, 2.0), yaw_rad=yaw)
    obs = estimator().detect(image)
    assert obs.detected
    recovered = 2.0 * np.arctan2(obs.quaternion_pad_flu_wxyz[3], obs.quaternion_pad_flu_wxyz[0])
    assert recovered == pytest.approx(yaw, abs=0.05)


def test_quality_falls_with_altitude():
    low = estimator().detect(render_board_view((0.0, 0.0, 1.0))[0]).quality
    high = estimator().detect(render_board_view((0.0, 0.0, 4.6))[0]).quality
    assert 0.0 < high < low <= 1.0


def test_empty_view_reports_nothing():
    obs = estimator().detect(np.full((HEIGHT, WIDTH), 255, dtype=np.uint8))
    assert not obs.detected and obs.quality == 0.0


def test_unknown_ids_are_ignored():
    image, camera_matrix = render_board_view((0.0, 0.0, 2.0))
    other = MarkerPoseEstimator(
        MarkerBoard([BoardMarker(31, 0.45, (0.0, 0.0))]), camera_matrix, MOUNT,
        dictionary=DICTIONARY)
    assert not other.detect(image).detected


def test_texture_carries_a_quiet_zone(tmp_path):
    path = generate_marker_png(tmp_path / "marker.png", DICTIONARY, 10, pixels=240)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert image.shape[0] == image.shape[1] > 240
    assert image.shape[0] / 240 == pytest.approx(texture_side_ratio(DICTIONARY), rel=0.02)
    assert image[0, 0] == 255
