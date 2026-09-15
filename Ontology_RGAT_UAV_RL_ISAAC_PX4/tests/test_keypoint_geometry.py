"""Geometric pad-centre FOV and six-keypoint projection.

The whole point of this module is that *geometric* field of view and *neural*
perception quality are two different variables. The tests below keep them
separable by construction: geometry is decided by the camera pose alone, and
nothing a detector or an encoder reports can move it.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "isaac_sim"))

from config_loader import load_config  # noqa: E402
from keypoint_geometry import (  # noqa: E402
    KEYPOINT_LAYOUT_ID, PAD_LANDMARK_COUNT, PAD_LANDMARK_RADIUS_M, CameraModel,
    body_from_optical, geometric_pad_center_in_fov, pad_center, pad_landmarks,
    project_landing_pad)
from landing_pad_visual import (  # noqa: E402
    LANDING_PAD_VISUAL_VERSION, LandingPadVisual, landing_pad_texture)

from ontology_rgat.initialization import (  # noqa: E402
    camera_centered_hover_offset, pad_view_margin)
from ontology_rgat.perception import pad_geometry  # noqa: E402
from ontology_rgat.perception.semantic_observation import (  # noqa: E402
    semantic_observation)

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])
SHIN_CAMERA = CameraModel()


def _yaw_quaternion(yaw_rad: float) -> np.ndarray:
    return np.array([math.cos(yaw_rad / 2.0), 0.0, 0.0, math.sin(yaw_rad / 2.0)])


# --------------------------------------------------------------- landmarks

def test_six_landmarks_form_one_shared_hexagon_on_the_deck():
    landmarks = pad_landmarks()

    assert landmarks.shape == (PAD_LANDMARK_COUNT, 3)
    assert np.allclose(landmarks[:, 2], 0.0)
    np.testing.assert_allclose(
        np.linalg.norm(landmarks[:, :2], axis=1), PAD_LANDMARK_RADIUS_M)
    # Distinct, evenly spaced and off the pad's own axes, so the layout is
    # unambiguous under the deck's 90-degree symmetry.
    bearings = np.sort(np.degrees(np.arctan2(landmarks[:, 1], landmarks[:, 0])) % 360.0)
    np.testing.assert_allclose(bearings, [30.0, 90.0, 150.0, 210.0, 270.0, 330.0])
    np.testing.assert_allclose(pad_center(), np.zeros(3))


def test_the_painted_target_and_the_supervised_labels_share_one_definition():
    """A drifted texture would silently teach the encoder the wrong points."""
    system = load_config(ROOT / "config" / "shin2026-system.yaml")
    visual = LandingPadVisual(
        system["vision"], ROOT, "/World/landing_pad",
        deck_size_m=system["pad"]["deck_size_m"])

    np.testing.assert_allclose(
        visual.landmarks_pad_m, pad_landmarks(visual.landmark_radius_m))
    assert visual.landmark_count == PAD_LANDMARK_COUNT
    assert visual.layout_id == KEYPOINT_LAYOUT_ID
    # The learner package resolves the very same module, not a copy.
    assert pad_geometry.KEYPOINT_LAYOUT_ID == KEYPOINT_LAYOUT_ID
    np.testing.assert_allclose(pad_geometry.pad_landmarks(), pad_landmarks())
    assert pad_geometry.LANDING_PAD_VISUAL_VERSION == LANDING_PAD_VISUAL_VERSION


def test_the_fiducial_texture_is_high_contrast_and_carries_every_landmark():
    deck = (1.5, 1.5)
    image = landing_pad_texture(deck, pixels=1024)

    assert image.shape == (1024, 1024)
    assert image.min() == 0 and image.max() == 255
    # Each landmark centre is the dark core of its bullseye, and its immediate
    # surroundings are the bright ring: that is what a keypoint detector locks
    # onto, at every altitude.
    for x, y, _ in pad_landmarks():
        column = int(round((float(x) + deck[0] / 2.0) / deck[0] * 1023))
        row = int(round((deck[1] / 2.0 - float(y)) / deck[1] * 1023))
        assert image[row, column] == 0
        patch = image[row - 40:row + 41, column - 40:column + 41]
        assert patch.max() == 255 and patch.min() == 0


def test_the_texture_refuses_landmarks_that_would_fall_off_the_deck():
    with pytest.raises(ValueError, match="physical landing deck"):
        landing_pad_texture((1.5, 1.5), pixels=256, landmark_radius_m=0.74)
    with pytest.raises(ValueError, match="landmark diameter"):
        landing_pad_texture((1.5, 1.5), pixels=256, landmark_diameter_m=1.6)


# ------------------------------------------------------- geometric FOV truth

def test_geometric_fov_is_positive_depth_plus_normalized_bounds():
    centred = camera_centered_hover_offset(4.5)
    projection = project_landing_pad(centred, IDENTITY, camera=SHIN_CAMERA)

    assert projection.geometric_pad_center_in_fov
    assert projection.pad_center_depth_m > 0.0
    np.testing.assert_allclose(projection.pad_center_normalized, [0.0, 0.0], atol=1e-9)
    assert np.all(np.abs(projection.pad_center_normalized) <= 1.0)

    # Ahead of and just above the deck the pad is behind the image plane.
    behind = project_landing_pad([5.0, 0.0, 1.0], IDENTITY, camera=SHIN_CAMERA)
    assert not behind.geometric_pad_center_in_fov
    assert behind.pad_center_depth_m <= 0.0


def test_geometric_fov_agrees_with_the_entry_gate_margin():
    """One definition, used by the gate, the metrics and the R-GAT labels."""
    rng = np.random.default_rng(7)
    checked_inside = checked_outside = 0
    for _ in range(400):
        altitude = float(rng.uniform(0.5, 9.0))
        offset = camera_centered_hover_offset(altitude) + np.r_[
            rng.uniform(-4.0, 4.0, 2), 0.0]
        quaternion = _yaw_quaternion(float(rng.uniform(-math.pi, math.pi)))
        margin = pad_view_margin(offset, quaternion)
        inside = geometric_pad_center_in_fov(
            offset, quaternion, camera=SHIN_CAMERA)
        assert inside == (margin <= 1.0)
        checked_inside += int(inside)
        checked_outside += int(not inside)
    assert checked_inside > 20 and checked_outside > 20


def test_geometric_fov_is_invariant_to_a_shared_yaw_of_camera_and_pad():
    """Why the training-only pose may be published in the deck-local frame."""
    offset = camera_centered_hover_offset(4.0) + np.array([0.6, -0.4, 0.0])
    base = project_landing_pad(offset, IDENTITY, camera=SHIN_CAMERA)
    for yaw in (0.3, 1.1, -2.4):
        rotation = np.array([[math.cos(yaw), -math.sin(yaw), 0.0],
                             [math.sin(yaw), math.cos(yaw), 0.0],
                             [0.0, 0.0, 1.0]])
        rotated = project_landing_pad(
            rotation @ offset, _yaw_quaternion(yaw), camera=SHIN_CAMERA)
        assert rotated.geometric_pad_center_in_fov == base.geometric_pad_center_in_fov
        np.testing.assert_allclose(
            rotated.pad_center_normalized, base.pad_center_normalized, atol=1e-9)


def test_the_optical_axis_matches_the_configured_down_pitch():
    view = body_from_optical(60.0)[:, 2]
    np.testing.assert_allclose(
        view, [math.cos(math.radians(60.0)), 0.0, -math.sin(math.radians(60.0))],
        atol=1e-12)
    assert CameraModel().tan_half_vertical < CameraModel().tan_half_horizontal


def test_the_six_keypoint_target_is_projectable_across_the_approach_range():
    """Table-I entry altitudes down to touchdown all keep landmarks in frame."""
    system = load_config(ROOT / "config" / "shin2026-system.yaml")
    camera = CameraModel.from_mapping(system["vision"]["camera"])
    conditions = system["benchmark"]["initial_conditions"]
    low, high = (float(v) for v in conditions["relative_altitude_m"])

    for altitude in np.linspace(0.35, high, 24):
        offset = camera_centered_hover_offset(
            float(altitude), camera.pitch_down_deg,
            np.asarray(camera.mount_translation_flu_m))
        projection = project_landing_pad(offset, IDENTITY, camera=camera)
        assert projection.geometric_pad_center_in_fov, altitude
        # At and above the entry altitudes the whole hexagon is in frame; on
        # the way down it leaves gradually, which is what the encoder's
        # visibility head is trained for.
        if altitude >= low:
            assert projection.visible_keypoint_fraction == 1.0, altitude
        else:
            assert projection.visible_keypoint_fraction > 0.0, altitude


# ------------------------------- geometry and perception vary independently

def _keypoint_semantics(*, peaked: bool, visible_points: int):
    points = pad_landmarks()[:, :2] / (2.0 * PAD_LANDMARK_RADIUS_M)
    heatmaps = np.zeros((6, 4, 5))
    if peaked:
        heatmaps[:, 1, 2] = 20.0
    visibility = np.zeros(6)
    visibility[:visible_points] = 1.0
    return semantic_observation(
        points, heatmaps, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        keypoint_visibility=visibility)


def test_poor_keypoint_confidence_cannot_make_the_pad_geometrically_lost():
    """Pad centre inside the frame; the encoder sees almost nothing."""
    centred = camera_centered_hover_offset(5.0)
    assert geometric_pad_center_in_fov(centred, IDENTITY, camera=SHIN_CAMERA)

    degraded = _keypoint_semantics(peaked=False, visible_points=1)
    assert degraded.keypoint_confidence < 0.01
    assert degraded.visible_keypoint_fraction < 0.5
    # Perception degradation, recorded as such -- geometry is untouched.
    assert geometric_pad_center_in_fov(centred, IDENTITY, camera=SHIN_CAMERA)


def test_total_keypoint_failure_is_not_a_geometric_fov_loss():
    centred = camera_centered_hover_offset(3.0)
    blind = _keypoint_semantics(peaked=False, visible_points=0)

    assert blind.visible_keypoint_fraction == 0.0
    assert blind.apparent_target_scale == 0.0
    assert blind.visual_loss_risk > 0.0
    assert geometric_pad_center_in_fov(centred, IDENTITY, camera=SHIN_CAMERA)


def test_leaving_the_frustum_is_a_geometric_fov_loss_despite_confident_keypoints():
    outside = np.array([6.0, 0.0, 4.5])
    assert not geometric_pad_center_in_fov(outside, IDENTITY, camera=SHIN_CAMERA)

    confident = _keypoint_semantics(peaked=True, visible_points=6)
    assert confident.visible_keypoint_fraction == 1.0
    assert confident.keypoint_confidence > 0.5
    # A stale, still-confident estimate does not keep the pad in frame.
    assert not geometric_pad_center_in_fov(outside, IDENTITY, camera=SHIN_CAMERA)
