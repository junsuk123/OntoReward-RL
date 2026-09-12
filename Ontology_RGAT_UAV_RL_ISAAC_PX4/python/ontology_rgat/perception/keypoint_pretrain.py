"""Synthetic pretraining plus empirical Isaac validation/fine-tuning.

The paper's PACMAN weights are not public.  This module does not pretend to be
PACMAN: it renders the repository's deployed multi-scale ArUco board under the
Shin camera geometry, supervises six fixed pad landmarks, and trains the same
descriptor embedding later consumed by the LSTM.  The resulting encoder is
frozen before PPO so the policy cannot erase its geometric representation.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..initialization import camera_centered_hover_offset
from .keypoint_encoder import ShinKeypointEncoder


PRETRAIN_FORMAT = "shin2026-hybrid-keypoint-pretrain-v3-visibility"


def _balanced_visibility_loss(prediction, target):
    """Give visible and absent keypoint decisions equal optimizer weight."""
    element = F.binary_cross_entropy(prediction, target, reduction="none")
    positive = target >= 0.5
    negative = ~positive
    parts = []
    if torch.any(positive):
        parts.append(element[positive].mean())
    if torch.any(negative):
        parts.append(element[negative].mean())
    return torch.stack(parts).mean()


def _rotation_z(yaw: float) -> np.ndarray:
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _body_from_optical(pitch_down_deg: float) -> np.ndarray:
    angle = math.radians(float(pitch_down_deg) - 90.0)
    c, s = math.cos(angle), math.sin(angle)
    pitch = np.array([[c, 0.0, s], [0.0, 1.0, 0.0],
                      [-s, 0.0, c]])
    nadir = np.diag([1.0, -1.0, -1.0])
    return pitch @ nadir


def _project(points_pad: np.ndarray, camera_pad: np.ndarray,
             pad_from_optical: np.ndarray, focal: float,
             width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    optical = (pad_from_optical.T
               @ (np.asarray(points_pad, dtype=float) - camera_pad).T).T
    depth = optical[:, 2]
    safe = np.where(depth > 1e-6, depth, 1.0)
    pixels = np.column_stack((
        focal * optical[:, 0] / safe + width / 2.0,
        focal * optical[:, 1] / safe + height / 2.0,
    ))
    return pixels, depth


def _square(center, side: float) -> np.ndarray:
    half = float(side) / 2.0
    cx, cy = (float(v) for v in center)
    return np.array([[cx - half, cy + half, 0.0],
                     [cx + half, cy + half, 0.0],
                     [cx + half, cy - half, 0.0],
                     [cx - half, cy - half, 0.0]])


def _hexagonal_landmarks(radius_m: float = 0.52) -> np.ndarray:
    angles = np.arange(6, dtype=float) * (math.pi / 3.0) + math.pi / 6.0
    return np.column_stack((radius_m * np.cos(angles),
                            radius_m * np.sin(angles), np.zeros(6)))


def _marker_textures(
        dictionary_name: str,
        markers: list[dict]) -> tuple[Any, dict[int, np.ndarray], float]:
    import cv2

    if not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(f"unknown ArUco dictionary: {dictionary_name}")
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, dictionary_name))
    marker_pixels = 72
    margin = max(4, int(round(marker_pixels / (dictionary.markerSize + 2))))
    textures = {}
    for marker in markers:
        tag = cv2.aruco.generateImageMarker(
            dictionary, int(marker["id"]), marker_pixels)
        textures[int(marker["id"])] = cv2.copyMakeBorder(
            tag, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=255)
    return dictionary, textures, (marker_pixels + 2.0 * margin) / marker_pixels


def synthetic_keypoint_dataset(system: Mapping[str, Any], *, samples: int,
                               seed: int) -> dict[str, np.ndarray]:
    """Render deployed-board views and exact six-landmark supervision."""
    import cv2

    vision = dict(system.get("vision") or {})
    camera = dict(vision.get("camera") or {})
    markers = [dict(value) for value in vision.get("board") or ()]
    if not markers:
        raise ValueError("synthetic keypoint pretraining requires vision.board")
    width, height = (int(v) for v in camera.get("resolution", (512, 320)))
    if (width, height) != (512, 320):
        raise ValueError("Shin keypoint pretraining requires a 512x320 camera")
    fov = float(camera.get("horizontal_fov_deg", 90.0))
    pitch_down = float(camera.get("pitch_down_deg", 60.0))
    mount = np.asarray(camera.get(
        "mount_translation_flu_m", (0.0, 0.0, -0.16)), dtype=float)
    focal = (width / 2.0) / math.tan(math.radians(fov) / 2.0)
    _, textures, quiet_ratio = _marker_textures(
        str(vision.get("dictionary", "DICT_4X4_100")), markers)
    rng = np.random.default_rng(int(seed))
    landmarks = _hexagonal_landmarks()
    feature_h, feature_w = height // 16, width // 16

    images = np.empty((int(samples), height, width), dtype=np.uint8)
    heatmaps = np.zeros((int(samples), 6, feature_h, feature_w), dtype=np.float32)
    coordinates = np.zeros((int(samples), 6, 2), dtype=np.float32)
    visible = np.zeros((int(samples), 6), dtype=np.float32)
    poses = np.empty((int(samples), 5), dtype=np.float32)
    grid_y, grid_x = np.mgrid[0:feature_h, 0:feature_w]
    src_cache = {}

    for index in range(int(samples)):
        altitude = float(rng.uniform(0.45, 8.0))
        yaw = float(rng.uniform(-math.pi, math.pi))
        rotation = _rotation_z(yaw)
        centred = camera_centered_hover_offset(altitude, pitch_down, mount)
        # Image-plane jitter produces centred, edge and partially visible pads.
        jitter_body = np.array([
            rng.uniform(-0.34, 0.34) * altitude,
            rng.uniform(-0.25, 0.25) * altitude, 0.0])
        body_pad = rotation @ (centred + jitter_body)
        camera_pad = body_pad + rotation @ mount
        pad_from_optical = rotation @ _body_from_optical(pitch_down)

        base = float(rng.uniform(45.0, 145.0))
        gx, gy = rng.uniform(-35.0, 35.0, 2)
        xx = np.linspace(-1.0, 1.0, width)[None, :]
        yy = np.linspace(-1.0, 1.0, height)[:, None]
        image = base + gx * xx + gy * yy
        image = image + rng.normal(0.0, rng.uniform(2.0, 10.0), (height, width))
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)

        pad_side = np.asarray((system.get("pad") or {}).get(
            "deck_size_m", (1.5, 1.5)), dtype=float)
        pad_corners = np.array([
            [-pad_side[0] / 2, +pad_side[1] / 2, 0.0],
            [+pad_side[0] / 2, +pad_side[1] / 2, 0.0],
            [+pad_side[0] / 2, -pad_side[1] / 2, 0.0],
            [-pad_side[0] / 2, -pad_side[1] / 2, 0.0],
        ])
        pad_pixels, pad_depth = _project(
            pad_corners, camera_pad, pad_from_optical, focal, width, height)
        if np.all(pad_depth > 0.0):
            cv2.fillConvexPoly(image, np.rint(pad_pixels).astype(np.int32),
                               int(rng.uniform(175, 235)))

        for marker in markers:
            marker_id = int(marker["id"])
            outer_side = float(marker["side_m"]) * quiet_ratio
            outer = _square(marker.get("center_xy_m", (0.0, 0.0)), outer_side)
            corners, depth = _project(
                outer, camera_pad, pad_from_optical, focal, width, height)
            if np.any(depth <= 0.0) or not np.isfinite(corners).all():
                continue
            texture = textures[marker_id]
            if marker_id not in src_cache:
                th, tw = texture.shape
                src_cache[marker_id] = np.array(
                    [[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]],
                    dtype=np.float32)
            transform = cv2.getPerspectiveTransform(
                src_cache[marker_id], corners.astype(np.float32))
            warped = cv2.warpPerspective(
                texture, transform, (width, height),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                borderValue=255)
            mask = cv2.warpPerspective(
                np.full(texture.shape, 255, dtype=np.uint8), transform,
                (width, height), flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            image[mask > 0] = warped[mask > 0]

        if rng.random() < 0.55:
            # PACMAN's purpose is partial visibility; keep landmark labels so
            # descriptors learn to infer geometry from the remaining patches.
            ow = int(rng.uniform(0.08, 0.30) * width)
            oh = int(rng.uniform(0.08, 0.30) * height)
            ox = int(rng.integers(0, max(1, width - ow)))
            oy = int(rng.integers(0, max(1, height - oh)))
            image[oy:oy + oh, ox:ox + ow] = int(rng.uniform(30, 170))
        if rng.random() < 0.6:
            image = cv2.GaussianBlur(image, (3, 3), rng.uniform(0.25, 1.1))
        image = np.clip(
            image.astype(np.float32) * rng.uniform(0.65, 1.25)
            + rng.normal(0.0, rng.uniform(0.0, 4.0), image.shape),
            0.0, 255.0).astype(np.uint8)

        keypoint_pixels, keypoint_depth = _project(
            landmarks, camera_pad, pad_from_optical, focal, width, height)
        keypoint_visible = ((keypoint_depth > 0.0)
                            & (keypoint_pixels[:, 0] >= 0.0)
                            & (keypoint_pixels[:, 0] <= width - 1.0)
                            & (keypoint_pixels[:, 1] >= 0.0)
                            & (keypoint_pixels[:, 1] <= height - 1.0))
        if rng.random() < 0.20:
            # Fully negative target-absent frames teach an explicit visibility
            # head. A heatmap soft-argmax alone always returns six coordinates
            # and cannot distinguish a blank/textured background from a pad.
            image = np.clip(
                rng.uniform(25.0, 225.0)
                + rng.normal(0.0, rng.uniform(3.0, 18.0), image.shape),
                0.0, 255.0).astype(np.uint8)
            keypoint_visible[:] = False
            heatmaps[index] = 0.0
        coordinates[index, :, 0] = 2.0 * keypoint_pixels[:, 0] / (width - 1.0) - 1.0
        coordinates[index, :, 1] = 2.0 * keypoint_pixels[:, 1] / (height - 1.0) - 1.0
        visible[index] = keypoint_visible.astype(np.float32)
        for point in np.flatnonzero(keypoint_visible):
            px = keypoint_pixels[point, 0] / (width - 1.0) * (feature_w - 1.0)
            py = keypoint_pixels[point, 1] / (height - 1.0) * (feature_h - 1.0)
            gaussian = np.exp(-((grid_x - px) ** 2 + (grid_y - py) ** 2)
                              / (2.0 * 0.85 ** 2))
            heatmaps[index, point] = gaussian / max(float(gaussian.sum()), 1e-9)

        relative_platform_body = rotation.T @ -body_pad
        poses[index, :3] = np.clip(
            relative_platform_body / np.array([8.0, 8.0, 8.0]), -1.0, 1.0)
        poses[index, 3:] = (math.sin(yaw), math.cos(yaw))
        images[index] = image

    if int(np.count_nonzero(visible)) < int(samples) * 2:
        raise RuntimeError("synthetic keypoint generator produced too few visible landmarks")
    return {"images": images, "heatmaps": heatmaps,
            "coordinates": coordinates, "visible": visible, "poses": poses}


def empirical_keypoint_dataset(frames, system: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Label real rendered frames from detected board-plane correspondences.

    The image is never annotated and simulator pose truth is not injected into
    the actor.  Known metric marker corners establish a pad-plane homography;
    that homography labels the same six hexagonal landmarks used in synthetic
    pretraining. Frames without a valid deployed-board solve are rejected.
    """
    import cv2

    vision = dict(system.get("vision") or {})
    camera = dict(vision.get("camera") or {})
    width, height = (int(v) for v in camera.get("resolution", (512, 320)))
    board = {int(entry["id"]): dict(entry)
             for entry in vision.get("board") or ()}
    if not board:
        raise ValueError("empirical keypoint calibration requires vision.board")
    dictionary_name = str(vision.get("dictionary", "DICT_4X4_100"))
    if not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(f"unknown ArUco dictionary: {dictionary_name}")
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name)))
    landmarks = _hexagonal_landmarks()[:, :2].astype(np.float32)
    feature_h, feature_w = height // 16, width // 16
    grid_y, grid_x = np.mgrid[0:feature_h, 0:feature_w]
    labelled = []
    for value in frames:
        image = np.asarray(value, dtype=np.uint8)
        if image.shape != (height, width):
            continue
        corners, ids, _ = detector.detectMarkers(image)
        if ids is None:
            continue
        object_xy, image_xy = [], []
        for quad, marker_id in zip(corners, ids.flatten()):
            entry = board.get(int(marker_id))
            if entry is None:
                continue
            object_xy.append(_square(
                entry.get("center_xy_m", (0.0, 0.0)),
                float(entry["side_m"]))[:, :2])
            image_xy.append(np.asarray(quad, dtype=np.float32).reshape(4, 2))
        if not object_xy:
            continue
        homography, inliers = cv2.findHomography(
            np.concatenate(object_xy).astype(np.float32),
            np.concatenate(image_xy).astype(np.float32), cv2.RANSAC, 3.0)
        if homography is None or inliers is None or int(inliers.sum()) < 4:
            continue
        pixels = cv2.perspectiveTransform(
            landmarks[None], homography).reshape(6, 2)
        if not np.isfinite(pixels).all():
            continue
        visible = ((pixels[:, 0] >= 0.0) & (pixels[:, 0] <= width - 1.0)
                   & (pixels[:, 1] >= 0.0) & (pixels[:, 1] <= height - 1.0))
        heatmaps = np.zeros((6, feature_h, feature_w), dtype=np.float32)
        for point in np.flatnonzero(visible):
            px = pixels[point, 0] / (width - 1.0) * (feature_w - 1.0)
            py = pixels[point, 1] / (height - 1.0) * (feature_h - 1.0)
            gaussian = np.exp(-((grid_x - px) ** 2 + (grid_y - py) ** 2)
                              / (2.0 * 0.85 ** 2))
            heatmaps[point] = gaussian / max(float(gaussian.sum()), 1e-9)
        coordinates = np.column_stack((
            2.0 * pixels[:, 0] / (width - 1.0) - 1.0,
            2.0 * pixels[:, 1] / (height - 1.0) - 1.0)).astype(np.float32)
        labelled.append((image.copy(), heatmaps, coordinates,
                         visible.astype(np.float32)))
    if not labelled:
        return {
            "images": np.empty((0, height, width), dtype=np.uint8),
            "heatmaps": np.empty((0, 6, feature_h, feature_w), dtype=np.float32),
            "coordinates": np.empty((0, 6, 2), dtype=np.float32),
            "visible": np.empty((0, 6), dtype=np.float32),
        }
    return {name: np.stack([item[index] for item in labelled])
            for index, name in enumerate(
                ("images", "heatmaps", "coordinates", "visible"))}


@torch.no_grad()
def _empirical_metrics(encoder, dataset, indices, device) -> dict:
    images = torch.as_tensor(dataset["images"][indices, None],
                             dtype=torch.float32, device=device) / 255.0
    target = torch.as_tensor(dataset["coordinates"][indices],
                             dtype=torch.float32, device=device)
    visible = torch.as_tensor(dataset["visible"][indices],
                              dtype=torch.bool, device=device)
    output = encoder(images)
    predicted = output.keypoints
    absent = images.mean(dim=(-1, -2), keepdim=True).expand_as(images)
    absent_visibility = encoder(absent).visibility
    false_positive_rate = float((absent_visibility >= 0.5).float().mean().cpu())
    scale = torch.tensor([511.0 / 2.0, 319.0 / 2.0], device=device)
    pixel_error = torch.linalg.vector_norm((predicted - target) * scale, dim=-1)
    selected = pixel_error[visible]
    if selected.numel() == 0:
        return {"coordinate_rmse_px": float("inf"), "pck_20px": 0.0,
                "visibility_accuracy": 0.0,
                "absent_false_positive_rate": false_positive_rate}
    return {
        "coordinate_rmse_px": float(torch.sqrt(selected.square().mean()).cpu()),
        "pck_20px": float((selected <= 20.0).float().mean().cpu()),
        "visibility_accuracy": float(
            ((output.visibility >= 0.5) == visible).float().mean().cpu()),
        "absent_false_positive_rate": false_positive_rate,
    }


def calibrate_keypoint_encoder(
        path: str | Path, artifact: dict, camera_source,
        *, system: Mapping[str, Any], experiment: Mapping[str, Any],
        mode: str, device: str | torch.device) -> dict:
    """Validate and optionally fine-tune a frozen encoder on live Isaac RGB output."""
    if artifact is None:
        return None
    empirical = artifact.get("empirical_calibration") or {}
    if empirical.get("validated"):
        print(f"Using empirically validated Isaac keypoint encoder from {path}.")
        return artifact
    estimator = dict(experiment.get("estimator") or {})
    settings = dict(estimator.get("keypoint_pretraining") or {})
    requested = int(settings.get(
        f"empirical_samples_{mode}", settings.get("empirical_samples", 48)))
    minimum = max(8, int(settings.get("minimum_empirical_samples", 8)))
    frames = []
    dataset = empirical_keypoint_dataset(frames, system)
    for _ in range(max(requested * 5, minimum)):
        frames.append(camera_source())
        if len(frames) >= minimum:
            dataset = empirical_keypoint_dataset(frames, system)
            if len(dataset["images"]) >= requested:
                break
    count = len(dataset["images"])
    if count < minimum:
        raise RuntimeError(
            f"Isaac keypoint calibration found only {count} labelled frames "
            f"(minimum {minimum}); refusing a synthetic-only frozen encoder")
    if count > requested:
        dataset = {name: value[:requested] for name, value in dataset.items()}
        count = requested
    seed = int(settings.get("seed", 41026)) + 73
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    order = rng.permutation(count)
    validation_count = max(4, int(round(count * float(
        settings.get("empirical_validation_fraction", .25)))))
    validation_count = min(validation_count, count - 4)
    validation, training = order[:validation_count], order[validation_count:]
    torch_device = torch.device(device)
    encoder = ShinKeypointEncoder(
        int(estimator.get("image_embedding", 512)), keypoints=6).to(torch_device)
    encoder.load_state_dict(artifact["encoder"])
    before = _empirical_metrics(encoder, dataset, validation, torch_device)
    best = {name: value.detach().cpu().clone()
            for name, value in encoder.state_dict().items()}
    best_metrics = before
    optimizer = torch.optim.Adam(
        encoder.parameters(), lr=float(settings.get(
            "empirical_learning_rate", 1e-4)))
    batch_size = int(settings.get("batch_size", 16))
    epochs = int(settings.get("empirical_epochs", 2))
    for _ in range(max(1, epochs)):
        for start in range(0, len(training), batch_size):
            indices = training[start:start + batch_size]
            images = torch.as_tensor(
                dataset["images"][indices, None], dtype=torch.float32,
                device=torch_device) / 255.0
            target_heatmaps = torch.as_tensor(
                dataset["heatmaps"][indices], dtype=torch.float32,
                device=torch_device)
            target_coordinates = torch.as_tensor(
                dataset["coordinates"][indices], dtype=torch.float32,
                device=torch_device)
            mask = torch.as_tensor(
                dataset["visible"][indices], dtype=torch.float32,
                device=torch_device)
            output = encoder(images)
            denominator = mask.sum().clamp_min(1.0)
            heatmap_loss = (-(target_heatmaps.flatten(2)
                              * F.log_softmax(output.heatmaps.flatten(2), -1))
                            .sum(-1) * mask).sum() / denominator
            coordinate_loss = (((output.keypoints - target_coordinates)
                                .square().sum(-1) * mask).sum() / denominator)
            visibility_loss = _balanced_visibility_loss(output.visibility, mask)
            # Retain the synthetic target-absent decision boundary while the
            # shared convolutional trunk adapts to the live Isaac renderer.
            negative = images.mean(dim=(-1, -2), keepdim=True).expand_as(images)
            negative = torch.clamp(
                negative + 0.08 * torch.randn_like(negative), 0.0, 1.0)
            negative_visibility = encoder(negative).visibility
            absent_loss = _balanced_visibility_loss(
                negative_visibility, torch.zeros_like(negative_visibility))
            loss = (heatmap_loss + 3.0 * coordinate_loss
                    + 3.0 * visibility_loss + absent_loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
            optimizer.step()
        candidate = _empirical_metrics(
            encoder, dataset, validation, torch_device)
        candidate_score = (candidate["coordinate_rmse_px"]
                           + 50.0 * candidate["absent_false_positive_rate"])
        best_score = (best_metrics["coordinate_rmse_px"]
                      + 50.0 * best_metrics["absent_false_positive_rate"])
        if candidate_score < best_score:
            best_metrics = candidate
            best = {name: value.detach().cpu().clone()
                    for name, value in encoder.state_dict().items()}
    artifact = dict(artifact)
    artifact["encoder"] = best
    artifact["training_source"] = (
        "synthetic board projections plus labelled Isaac camera frames")
    artifact["empirical_calibration"] = {
        "validated": True, "label_source": "ArUco board-plane homography",
        "samples": count, "training_samples": len(training),
        "validation_samples": len(validation), "before": before,
        "after": best_metrics,
        "selection_score": "coordinate_rmse_px + 50*absent_false_positive_rate",
        "fine_tune_selected": bool(
            best_metrics["coordinate_rmse_px"]
            + 50.0 * best_metrics["absent_false_positive_rate"]
            < before["coordinate_rmse_px"]
            + 50.0 * before["absent_false_positive_rate"]),
    }
    dataset_path = Path(path).with_name("keypoint_isaac_calibration.npz")
    np.savez_compressed(dataset_path, **dataset)
    artifact["empirical_dataset"] = str(dataset_path.resolve())
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    torch.save(artifact, temporary)
    os.replace(temporary, path)
    print(
        "Isaac keypoint validation complete: "
        f"{before['coordinate_rmse_px']:.1f}px -> "
        f"{best_metrics['coordinate_rmse_px']:.1f}px, "
        f"PCK@20 {before['pck_20px']:.1%} -> {best_metrics['pck_20px']:.1%}, "
        f"visibility accuracy {best_metrics['visibility_accuracy']:.1%}, "
        "target-absent false positives "
        f"{best_metrics['absent_false_positive_rate']:.1%}.")
    del encoder, optimizer
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()
    return artifact


def _train_encoder(system: Mapping[str, Any], settings: Mapping[str, Any],
                   *, mode: str, device: torch.device) -> tuple[dict, dict]:
    samples = int(settings.get(
        f"samples_{mode}", settings.get("samples", 512)))
    epochs = int(settings.get(
        f"epochs_{mode}", settings.get("epochs", 6)))
    batch_size = int(settings.get("batch_size", 16))
    seed = int(settings.get("seed", 41026))
    if samples < 4 or epochs < 1 or batch_size < 1:
        raise ValueError("keypoint pretraining needs >=4 samples and positive epochs/batch")
    dataset = synthetic_keypoint_dataset(system, samples=samples, seed=seed)
    torch.manual_seed(seed)
    encoder = ShinKeypointEncoder(
        int(settings.get("image_embedding", 512)), keypoints=6).to(device)
    pose_head = nn.Linear(int(settings.get("image_embedding", 512)), 5).to(device)
    optimizer = torch.optim.Adam(
        [*encoder.parameters(), *pose_head.parameters()],
        lr=float(settings.get("learning_rate", 1e-3)))
    rng = np.random.default_rng(seed + 1)
    last = {}
    for _ in range(epochs):
        order = rng.permutation(samples)
        totals = {"loss": 0.0, "heatmap": 0.0, "coordinate": 0.0,
                  "visibility": 0.0, "pose": 0.0, "batches": 0}
        for start in range(0, samples, batch_size):
            indices = order[start:start + batch_size]
            images = torch.as_tensor(
                dataset["images"][indices, None], dtype=torch.float32,
                device=device) / 255.0
            target_heatmaps = torch.as_tensor(
                dataset["heatmaps"][indices], dtype=torch.float32, device=device)
            target_coordinates = torch.as_tensor(
                dataset["coordinates"][indices], dtype=torch.float32, device=device)
            mask = torch.as_tensor(
                dataset["visible"][indices], dtype=torch.float32, device=device)
            pose = torch.as_tensor(
                dataset["poses"][indices], dtype=torch.float32, device=device)
            output = encoder(images)
            log_prob = F.log_softmax(output.heatmaps.flatten(2), dim=-1)
            heatmap_per_point = -(target_heatmaps.flatten(2) * log_prob).sum(-1)
            denominator = mask.sum().clamp_min(1.0)
            heatmap_loss = (heatmap_per_point * mask).sum() / denominator
            coordinate_loss = (((output.keypoints - target_coordinates).square().sum(-1)
                                * mask).sum() / denominator)
            visibility_loss = _balanced_visibility_loss(output.visibility, mask)
            pose_mask = (mask.sum(-1) > 0.0).float()
            pose_error = (pose_head(output.embedding) - pose).square().mean(-1)
            pose_loss = (pose_error * pose_mask).sum() / pose_mask.sum().clamp_min(1.0)
            loss = (heatmap_loss + 3.0 * coordinate_loss
                    + 3.0 * visibility_loss + 2.0 * pose_loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [*encoder.parameters(), *pose_head.parameters()], 5.0)
            optimizer.step()
            for name, value in (("loss", loss), ("heatmap", heatmap_loss),
                                ("coordinate", coordinate_loss),
                                ("visibility", visibility_loss),
                                ("pose", pose_loss)):
                totals[name] += float(value.detach())
            totals["batches"] += 1
        last = {name: value / totals["batches"]
                for name, value in totals.items() if name != "batches"}
    state = {name: value.detach().cpu() for name, value in encoder.state_dict().items()}
    metrics = {**last, "samples": samples, "epochs": epochs,
               "visible_fraction": float(dataset["visible"].mean())}
    # Isaac Sim starts immediately after this phase and shares the GPU. Keep
    # only the CPU artifact so the temporary pose head and optimizer do not
    # reserve an otherwise invisible CUDA block during simulator startup.
    del encoder, pose_head, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return state, metrics


def prepare_keypoint_encoder(
        path: str | Path, *, config_hash: str, experiment: Mapping[str, Any],
        system: Mapping[str, Any], mode: str, device: str | torch.device) -> dict | None:
    """Load or deterministically train the frozen encoder artifact."""
    estimator = dict(experiment.get("estimator") or {})
    settings = dict(estimator.get("keypoint_pretraining") or {})
    if not bool(settings.get("enabled", False)):
        return None
    settings["image_embedding"] = int(estimator.get("image_embedding", 512))
    path = Path(path)
    if path.is_file():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if (saved.get("format") == PRETRAIN_FORMAT
                and saved.get("config_hash") == str(config_hash)
                and saved.get("mode") == str(mode)
                and saved.get("implementation") == ShinKeypointEncoder.implementation):
            print(f"Using frozen synthetic six-keypoint encoder from {path}.")
            return saved
    print("Pretraining six-keypoint descriptor encoder on synthetic deployed-board views...")
    state, metrics = _train_encoder(
        system, settings, mode=str(mode), device=torch.device(device))
    payload = {
        "format": PRETRAIN_FORMAT,
        "config_hash": str(config_hash),
        "mode": str(mode),
        "implementation": ShinKeypointEncoder.implementation,
        "frozen_for_ppo": True,
        "landmark_layout": "six vertices of a 0.52 m pad-centred hexagon",
        "training_source": (
            "synthetic projections and target-absent negatives of configured "
            "deployed marker board"),
        "empirical_calibration": None,
        "metrics": metrics,
        "encoder": state,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    print(f"Frozen keypoint pretraining complete: loss={metrics['loss']:.4f}, "
          f"coordinate={metrics['coordinate']:.4f}, artifact={path}")
    return payload
