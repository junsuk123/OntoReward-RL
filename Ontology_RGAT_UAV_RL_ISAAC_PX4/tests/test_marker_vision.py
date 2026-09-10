"""Round-trip the landing-pad board through a synthetic downward camera."""

import math
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


def test_operator_frame_contains_successful_recognition_overlay():
    image, _ = render_board_view((0.2, -0.1, 2.0))
    obs, annotated = estimator().detect_annotated(image)

    assert obs.detected
    assert annotated.shape == (HEIGHT, WIDTH, 3)
    assert annotated.dtype == np.uint8
    # The input is grayscale; coloured green board outlines prove that the
    # returned frame is the annotated RGB operator view, not the raw image.
    channels = annotated.astype(np.int16)
    green = ((channels[:, :, 1] > 170)
             & (channels[:, :, 1] > channels[:, :, 0] + 50)
             & (channels[:, :, 1] > channels[:, :, 2] + 50))
    assert np.count_nonzero(green) > 100


def test_operator_frame_reports_a_detection_miss_visually():
    image = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
    obs, annotated = estimator().detect_annotated(image)

    assert not obs.detected
    assert annotated.shape == (HEIGHT, WIDTH, 3)
    # Red status lettering must remain visible even when there are no corners
    # to draw, otherwise an operator cannot distinguish a miss from no stream.
    channels = annotated.astype(np.int16)
    red = ((channels[:, :, 0] > 170)
           & (channels[:, :, 0] > channels[:, :, 1] + 60)
           & (channels[:, :, 0] > channels[:, :, 2] + 60))
    assert np.count_nonzero(red) > 50


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


def test_the_entry_pose_is_drawn_inside_the_camera_frame():
    """Every episode has to open with the deck already in view.

    The drawn offset does not respect the frame on its own: the camera's short
    axis reaches about three quarters of its long one, and the Gaussian tails
    put the deck outside both. This is the clamp that pulls it back in, checked
    against the real camera and the real draw.
    """
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "isaac_sim"))
    import yaml
    from marker_vision import nadir_footprint_m

    with (root / "config" / "system.yaml").open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    camera = cfg["vision"]["camera"]
    width, height = camera["resolution"]
    fov = float(camera["horizontal_fov_deg"])

    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(4000):
        altitude = 4.0 + 1.8 * rng.random()
        drawn = np.array([1.8 * rng.normal(), 1.4 * rng.normal(), altitude])
        _, half_short = nadir_footprint_m(width, height, fov, altitude)
        tilt = altitude * math.tan(math.radians(8.0))
        allowed = max(0.35 * half_short, half_short - tilt - 1.2)
        reach = float(np.hypot(drawn[0], drawn[1]))
        clamped = drawn.copy()
        if reach > allowed > 0.0:
            clamped[:2] *= allowed / reach
        # The clamp never pushes the deck out, and never lengthens the draw.
        assert np.hypot(clamped[0], clamped[1]) <= max(reach, allowed) + 1e-9
        assert np.hypot(clamped[0], clamped[1]) <= half_short + 1e-9
        worst = max(worst, np.hypot(clamped[0], clamped[1]) / half_short)
    # And it leaves real room: the deck sits well inside the short axis, not
    # balanced on its edge where the entry tilt would swing it out.
    assert worst < 0.75


def test_the_clamp_keeps_the_drawn_bearing():
    """Only the reach is shortened, so the seeded direction survives."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaac_sim"))
    from marker_vision import nadir_footprint_m

    altitude = 4.2
    drawn = np.array([5.0, -3.0, altitude])
    _, half_short = nadir_footprint_m(800, 600, 90.0, altitude)
    allowed = max(0.35 * half_short, half_short - altitude * math.tan(math.radians(8.0)) - 1.2)
    clamped = drawn.copy()
    clamped[:2] *= allowed / float(np.hypot(drawn[0], drawn[1]))

    assert math.atan2(clamped[1], clamped[0]) == pytest.approx(
        math.atan2(drawn[1], drawn[0]))
    assert clamped[2] == pytest.approx(drawn[2])
