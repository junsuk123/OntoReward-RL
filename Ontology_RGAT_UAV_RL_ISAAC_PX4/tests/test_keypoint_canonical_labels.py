"""Canonical image-plane landmark order, renderer v2 and exact label warps.

Background (2026-09-20): the shipped six-keypoint encoder predicted the pad
centre six times.  The hexagonal target is 60-degree symmetric and its
identity pips are one to three pixels at approach altitude, so a per-index
loss on pad-frame landmark identities has the centroid as its optimum.  These
tests pin the fix: labels are ordered in the image plane, the renderer looks
like Isaac, augmentation moves the labels with the pixels, stored calibration
frames survive an encoder change, and a collapsed encoder is refused.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "isaac_sim"))

from config_loader import load_config  # noqa: E402
from keypoint_geometry import project_landing_pad  # noqa: E402
from ontology_rgat.datastore import KIND_KEYPOINT_CALIBRATION, CollectedDataStore  # noqa: E402
from ontology_rgat.perception.keypoint_encoder import ShinKeypointEncoder  # noqa: E402
from ontology_rgat.perception.keypoint_pretrain import (  # noqa: E402
    AugmentationSettings, EMPIRICAL_CALIBRATION_FORMAT, LABEL_CONVENTION,
    _empirical_metrics, _pad_texture, _quat_wxyz_from_euler, _quat_wxyz_from_yaw,
    _render_landing_target, _rotation_z, _texture_pyramid, augment_labelled_frame,
    calibrate_keypoint_encoder, calibration_viewpoints, camera_model,
    canonical_landmark_shift, canonicalize_landmarks,
    keypoint_calibration_fingerprints, landing_pad_settings,
    needs_empirical_calibration, prepare_keypoint_encoder,
    synthetic_keypoint_dataset)
from ontology_rgat.initialization import camera_centered_hover_offset  # noqa: E402

SYSTEM = load_config(ROOT / "config/shin2026-system.yaml")
CAMERA = camera_model(SYSTEM)
PAD = landing_pad_settings(SYSTEM)


def _random_pose(rng):
    """A camera pose above the deck that keeps the pad roughly in view."""
    altitude = float(rng.uniform(0.5, 8.0))
    yaw = float(rng.uniform(-math.pi, math.pi))
    tilt = rng.uniform(-0.25, 0.25, 2)
    centred = camera_centered_hover_offset(altitude, CAMERA.pitch_down_deg,
                                           CAMERA.mount_translation_flu_m)
    jitter = np.array([rng.uniform(-0.3, 0.3) * altitude,
                       rng.uniform(-0.25, 0.25) * altitude, 0.0])
    position = _rotation_z(yaw) @ (centred + jitter)
    return position, _quat_wxyz_from_euler(yaw, float(tilt[0]), float(tilt[1])), yaw


def _image_angles(pixels, center):
    return np.mod(np.arctan2(-(pixels[:, 1] - center[1]), pixels[:, 0] - center[0]),
                  2.0 * math.pi)


# ------------------------------------------------------------- label order

def test_canonical_index_zero_is_the_smallest_ccw_angle_about_the_centre():
    rng = np.random.default_rng(7)
    checked = 0
    for _ in range(400):
        position, quaternion, _ = _random_pose(rng)
        projection = project_landing_pad(position, quaternion, camera=CAMERA,
                                         landmark_radius_m=PAD["landmark_radius_m"])
        if not (projection.keypoint_depth_m > 0.0).all():
            continue
        pixels, visible, shift = canonicalize_landmarks(
            projection.keypoint_pixels, projection.keypoint_visible,
            projection.pad_center_pixels)
        angles = _image_angles(pixels, projection.pad_center_pixels)
        assert int(np.argmin(angles)) == 0
        # The pad's cyclic order is kept: a canonical label set is a roll.
        np.testing.assert_allclose(
            pixels, np.roll(projection.keypoint_pixels, -shift, axis=0))
        np.testing.assert_array_equal(
            visible, np.roll(projection.keypoint_visible, -shift, axis=0))
        # Consecutive vertices step the same way round for every pose.
        assert np.all(np.diff(np.unwrap(angles)) > 0.0)
        checked += 1
    assert checked > 300


def test_canonical_labels_depend_on_the_image_and_not_on_pad_landmark_identity():
    """Turning the whole scene by the hexagon's own 60 degrees renders the same
    image; pad-frame identities shift by one, canonical labels do not."""
    rng = np.random.default_rng(11)
    for _ in range(50):
        position, quaternion, yaw = _random_pose(rng)
        turned_position = _rotation_z(math.pi / 3.0) @ position
        w, x, y, z = quaternion
        turned = _quat_wxyz_from_euler(yaw + math.pi / 3.0,
                                       *_pitch_roll(quaternion))
        a = project_landing_pad(position, quaternion, camera=CAMERA,
                                landmark_radius_m=PAD["landmark_radius_m"])
        b = project_landing_pad(turned_position, turned, camera=CAMERA,
                                landmark_radius_m=PAD["landmark_radius_m"])
        if not ((a.keypoint_depth_m > 0).all() and (b.keypoint_depth_m > 0).all()):
            continue
        # Same pixels as a set (the pad is symmetric) ...
        assert np.allclose(np.sort(a.keypoint_pixels, axis=0),
                           np.sort(b.keypoint_pixels, axis=0), atol=1e-6)
        # ... pad-frame identities differ ...
        assert not np.allclose(a.keypoint_pixels, b.keypoint_pixels, atol=1e-3)
        # ... canonical labels agree element by element.
        ca, _, _ = canonicalize_landmarks(a.keypoint_pixels, a.keypoint_visible,
                                          a.pad_center_pixels)
        cb, _, _ = canonicalize_landmarks(b.keypoint_pixels, b.keypoint_visible,
                                          b.pad_center_pixels)
        np.testing.assert_allclose(ca, cb, atol=1e-6)


def _pitch_roll(quaternion):
    w, x, y, z = (float(v) for v in quaternion)
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    return pitch, roll


def test_a_non_finite_projection_is_left_in_pad_order():
    pixels = np.full((6, 2), np.nan)
    assert canonical_landmark_shift(pixels, np.zeros(2)) == 0


# ---------------------------------------------------------------- renderer

def test_synthetic_renderer_v2_matches_isaac_photometry_and_is_canonical():
    dataset = synthetic_keypoint_dataset(SYSTEM, samples=32, seed=5)
    grid = ShinKeypointEncoder.heatmap_shape(320, 512)
    assert dataset["images"].shape == (32, 320, 512)
    assert dataset["heatmaps"].shape == (32, 6, *grid)
    assert dataset["pixels"].shape == (32, 6, 2) and dataset["center"].shape == (32, 2)
    assert dataset["poses"].shape == (32, 5)
    means = dataset["images"].reshape(32, -1).mean(axis=1)
    # Isaac frames average 20-52/255; most synthetic frames must be that dark,
    # with a minority of bright scenes for robustness.
    assert np.mean(means < 90.0) >= 0.6
    assert means.min() > 3.0 and means.max() < 235.0
    assert 0.3 < float(dataset["visible"].mean()) < 0.95
    visible = dataset["visible"] > 0.5
    np.testing.assert_allclose(dataset["heatmaps"][visible].sum(axis=(1, 2)), 1.0,
                               atol=1e-5)
    for i in range(32):
        if np.isfinite(dataset["pixels"][i]).all():
            assert canonical_landmark_shift(dataset["pixels"][i], dataset["center"][i]) == 0
    # The pose head's heading target is the hexagon-symmetric 6*yaw.
    np.testing.assert_allclose(np.hypot(dataset["poses"][:, 3], dataset["poses"][:, 4]),
                               1.0, atol=1e-5)


def test_synthetic_rendering_does_not_depend_on_the_worker_count():
    inline = synthetic_keypoint_dataset(SYSTEM, samples=6, seed=9, workers=1)
    again = synthetic_keypoint_dataset(SYSTEM, samples=6, seed=9, workers=1)
    np.testing.assert_array_equal(inline["images"], again["images"])
    np.testing.assert_allclose(inline["coordinates"], again["coordinates"])


def test_unknown_rendering_settings_are_rejected():
    with pytest.raises(ValueError, match="rendering setting"):
        synthetic_keypoint_dataset(SYSTEM, samples=4, seed=1,
                                   rendering={"pad_whiteness": 1})


# ------------------------------------------------------------ augmentation

def test_augmentation_moves_the_labels_with_the_pixels():
    """Paint a bright square at every landmark; after the warp the transformed
    labels must still sit on bright pixels."""
    rng = np.random.default_rng(3)
    dataset = synthetic_keypoint_dataset(SYSTEM, samples=8, seed=21)
    settings = AugmentationSettings(geometric_probability=1.0, photometric_probability=0.0,
                                    scale_range=(0.7, 1.4), rotation_deg=40.0)
    tested = 0
    for i in range(8):
        pixels, visible = dataset["pixels"][i], dataset["visible"][i] > 0.5
        if visible.sum() < 3:
            continue
        canvas = np.zeros((320, 512), dtype=np.uint8)
        for point in pixels[visible]:
            x, y = int(round(point[0])), int(round(point[1]))
            canvas[max(0, y - 3):y + 4, max(0, x - 3):x + 4] = 255
        image, new_pixels, new_center, new_visible = augment_labelled_frame(
            canvas, pixels, dataset["center"][i], visible, rng, settings, CAMERA)
        assert image.shape == (320, 512) and image.dtype == np.uint8
        for point, seen in zip(new_pixels, new_visible):
            if not seen:
                continue
            x, y = int(round(point[0])), int(round(point[1]))
            assert 0 <= x < 512 and 0 <= y < 320
            assert image[max(0, y - 2):y + 3, max(0, x - 2):x + 3].max() > 120
            tested += 1
        # Labels come back canonical for the new image, and never mirrored.
        assert canonical_landmark_shift(new_pixels, new_center) == 0
        assert np.all(np.diff(np.unwrap(_image_angles(new_pixels, new_center))) > 0.0)
    assert tested >= 12


def test_augmentation_hides_landmarks_pushed_out_of_frame():
    rng = np.random.default_rng(5)
    dataset = synthetic_keypoint_dataset(SYSTEM, samples=4, seed=33)
    settings = AugmentationSettings(geometric_probability=1.0, photometric_probability=0.0,
                                    scale_range=(3.0, 3.5), rotation_deg=0.0,
                                    recenter_probability=1.0, recenter_margin=0.45)
    i = int(np.argmax((dataset["visible"] > 0.5).sum(axis=1)))
    _, pixels, _, visible = augment_labelled_frame(
        dataset["images"][i], dataset["pixels"][i], dataset["center"][i],
        dataset["visible"][i] > 0.5, rng, settings, CAMERA)
    inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] <= 511)
              & (pixels[:, 1] >= 0) & (pixels[:, 1] <= 319))
    assert not visible[~inside].any()


# -------------------------------------------------------- encoder contract

def test_encoder_outputs_stride_8_heatmaps_and_ignores_global_exposure():
    torch.manual_seed(0)
    encoder = ShinKeypointEncoder(embedding_dim=32, keypoints=6).eval()
    image = torch.rand(2, 1, 320, 512)
    output = encoder(image)
    assert output.heatmaps.shape == (2, 6, 40, 64)
    assert ShinKeypointEncoder.heatmap_shape(320, 512) == (40, 64)
    dimmer = encoder(image * 0.4 + 0.05)
    torch.testing.assert_close(output.keypoints, dimmer.keypoints, atol=1e-3, rtol=0)
    torch.testing.assert_close(output.visibility, dimmer.visibility, atol=1e-3, rtol=0)
    assert "canonical" in ShinKeypointEncoder.implementation


def test_empirical_metrics_expose_a_centre_collapse():
    class Collapsed(torch.nn.Module):
        def forward(self, images):
            batch = images.shape[0]
            keypoints = torch.zeros(batch, 6, 2)
            return type("Out", (), {
                "keypoints": keypoints, "visibility": torch.ones(batch, 6) * 0.9,
                "heatmaps": torch.zeros(batch, 6, 40, 64),
                "embedding": torch.zeros(batch, 8)})()

    dataset = synthetic_keypoint_dataset(SYSTEM, samples=6, seed=2)
    metrics = _empirical_metrics(Collapsed(), dataset, np.arange(6), "cpu")
    assert metrics["spread_ratio"] < 0.05
    assert metrics["visibility_recall"] == pytest.approx(1.0)


# ---------------------------------------------------------- calibration

def _surveyed_camera(system, *, seed=3):
    model = camera_model(system)
    settings = landing_pad_settings(system)
    pyramid = _texture_pyramid(_pad_texture(settings))
    rng = np.random.default_rng(seed)
    flown = {"viewpoint": None}

    def survey(viewpoint):
        flown["viewpoint"] = viewpoint
        return True

    def labelled():
        viewpoint = flown["viewpoint"]
        position = (np.asarray(viewpoint["position_pad_m"], dtype=float)
                    + rng.normal(0.0, 0.02, 3))
        quaternion = _quat_wxyz_from_yaw(float(viewpoint["yaw_rad"]))
        image = np.clip(rng.normal(40.0, 6.0, (model.height, model.width)),
                        0.0, 255.0)
        image = _render_landing_target(
            image, pyramid, settings["deck_size_m"], position, quaternion, model)
        return image.astype(np.uint8), np.concatenate((position, quaternion))

    return survey, labelled


def _experiment(**overrides):
    keypoint = {
        "enabled": True, "samples_quick": 24, "epochs_quick": 1,
        "batch_size": 8, "learning_rate": 1e-3, "seed": 17,
        "empirical_samples_quick": 24, "empirical_steps": 3,
        "empirical_learning_rate": 1e-3, "empirical_replay_samples": 8,
        "empirical_eval_interval": 3, "render_workers": 1,
        "minimum_holdout_visibility_recall": 0.0,
        "minimum_holdout_spread_ratio": 0.0,
    }
    keypoint.update(overrides)
    return {"estimator": {"image_embedding": 32, "keypoint_pretraining": keypoint}}


def test_calibration_records_the_label_convention_and_augmentation(tmp_path):
    experiment = _experiment()
    path = tmp_path / "encoder.pt"
    artifact = prepare_keypoint_encoder(path, config_hash="cfg", experiment=experiment,
                                        system=SYSTEM, mode="quick", device="cpu")
    assert artifact["label_convention"] == LABEL_CONVENTION
    assert artifact["heatmap_stride"] == ShinKeypointEncoder.heatmap_stride
    survey, labelled = _surveyed_camera(SYSTEM)
    calibrated = calibrate_keypoint_encoder(
        path, artifact, labelled, survey=survey, system=SYSTEM,
        experiment=experiment, mode="quick", device="cpu")
    empirical = calibrated["empirical_calibration"]
    assert empirical["format"] == EMPIRICAL_CALIBRATION_FORMAT
    assert empirical["label_convention"] == LABEL_CONVENTION
    assert empirical["steps"] == 3 and empirical["replay_samples"] == 8
    assert "scale_range" in empirical["augmentation"]
    assert "spread_ratio" in empirical["after"]
    assert not needs_empirical_calibration(calibrated)
    stored = np.load(calibrated["empirical_dataset"])
    assert stored["pixels"].shape[1:] == (6, 2) and stored["center"].shape[1:] == (2,)


def test_a_centre_collapsed_encoder_is_refused(tmp_path):
    experiment = _experiment(minimum_holdout_spread_ratio=1.01)
    path = tmp_path / "encoder.pt"
    artifact = prepare_keypoint_encoder(path, config_hash="cfg", experiment=experiment,
                                        system=SYSTEM, mode="quick", device="cpu")
    survey, labelled = _surveyed_camera(SYSTEM)
    with pytest.raises(RuntimeError, match="collapsed onto the pad centre"):
        calibrate_keypoint_encoder(
            path, artifact, labelled, survey=survey, system=SYSTEM,
            experiment=experiment, mode="quick", device="cpu")


def test_calibration_frames_stored_under_the_superseded_encoder_key_are_reused(tmp_path):
    """A stored viewpoint is an image and a pose; the labels are re-projected.
    A new encoder therefore reads the old survey instead of flying it again."""
    import sqlite3

    experiment = _experiment()
    survey, labelled = _surveyed_camera(SYSTEM)
    store = CollectedDataStore(tmp_path / "collected.sqlite3", run_id="first")
    first_path = tmp_path / "first.pt"
    first = calibrate_keypoint_encoder(
        first_path, prepare_keypoint_encoder(
            first_path, config_hash="cfg", experiment=experiment, system=SYSTEM,
            mode="quick", device="cpu"),
        labelled, survey=survey, system=SYSTEM, experiment=experiment,
        mode="quick", device="cpu", datastore=store)
    current, legacy = keypoint_calibration_fingerprints(
        SYSTEM, experiment["estimator"]["keypoint_pretraining"])[:2]
    assert first["empirical_calibration"]["datastore_fingerprint"] == current
    viewpoints = first["empirical_calibration"]["viewpoints"]
    store.close()

    # Re-key every stored viewpoint to the fingerprint the v5 encoder used.
    connection = sqlite3.connect(tmp_path / "collected.sqlite3")
    connection.execute("UPDATE episode SET fingerprint = ? WHERE kind = ?",
                       (legacy, KIND_KEYPOINT_CALIBRATION))
    connection.commit()
    connection.close()

    store = CollectedDataStore(tmp_path / "collected.sqlite3", run_id="second")
    flown = []

    def refuse_to_fly(viewpoint):
        flown.append(int(viewpoint["index"]))
        return True

    second_path = tmp_path / "second.pt"
    second = calibrate_keypoint_encoder(
        second_path, prepare_keypoint_encoder(
            second_path, config_hash="cfg", experiment=experiment, system=SYSTEM,
            mode="quick", device="cpu"),
        labelled, survey=refuse_to_fly, system=SYSTEM, experiment=experiment,
        mode="quick", device="cpu", datastore=store)
    assert flown == []
    reused = second["empirical_calibration"]
    assert reused["reused_viewpoints"] == viewpoints
    assert legacy in reused["datastore_fingerprints_read"]
    assert reused["split_source"] == "frozen per-viewpoint datastore split"
    store.close()


def test_the_live_datastore_fingerprint_of_the_v5_survey_is_still_readable():
    """The 24 viewpoints surveyed on 2026-09-17 live under this key."""
    minimal = load_config(ROOT / "config/shin2026-minimal-system.yaml")
    fingerprints = keypoint_calibration_fingerprints(minimal, {})
    assert fingerprints[0] != fingerprints[1]
    assert fingerprints[1].startswith("8f3bf48eabbebb7f")
    assert len(calibration_viewpoints(minimal, {})) >= 12
