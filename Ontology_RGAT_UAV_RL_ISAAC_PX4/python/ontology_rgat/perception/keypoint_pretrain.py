"""Synthetic pretraining plus empirical Isaac validation/fine-tuning.

The paper's PACMAN weights are not public.  This module does not pretend to be
PACMAN: it renders the repository's deployed six-keypoint fiducial landing
target under the Shin camera geometry, supervises the six pad landmarks that
:mod:`keypoint_geometry` defines, and trains the same descriptor embedding
later consumed by the LSTM.  The resulting encoder is frozen before PPO so the
policy cannot erase its geometric representation.

No ArUco dictionary, marker id or ``cv2.aruco`` call takes part in this path.
Both the synthetic and the empirical labels are *projections of known pad
geometry*, which makes them ``training-label-only`` simulator truth; they are
never concatenated into the actor observation.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..initialization import camera_centered_hover_offset
from .keypoint_encoder import ShinKeypointEncoder
from .pad_geometry import (KEYPOINT_LAYOUT_ID, LANDING_PAD_VISUAL_VERSION,
                           PAD_LANDMARK_COUNT, PAD_LANDMARK_RADIUS_M,
                           CameraModel, camera_pose_in_pad,
                           landing_pad_texture, project_landing_pad)


PRETRAIN_FORMAT = "shin2026-six-keypoint-fiducial-pretrain-v4"
# Training-label-only pose payload published by the simulator alongside each
# empirical frame: pad-relative UAV position (3) and ENU/FLU attitude (4).
EMPIRICAL_POSE_LENGTH = 7
_HEATMAP_SIGMA = 0.85


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


def _quat_wxyz_from_yaw(yaw: float) -> np.ndarray:
    half = float(yaw) / 2.0
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)])


def camera_model(system: Mapping[str, Any]) -> CameraModel:
    """The rendered landing camera exactly as the simulator configures it."""
    camera = dict((dict(system.get("vision") or {})).get("camera") or {})
    model = CameraModel.from_mapping(camera)
    if (model.width, model.height) != (512, 320):
        raise ValueError("Shin keypoint pretraining requires a 512x320 camera")
    return model


def landing_pad_settings(system: Mapping[str, Any]) -> dict:
    """Deck size and landmark layout shared by the texture and the labels."""
    vision = dict(system.get("vision") or {})
    mode = str(vision.get("mode", "keypoint_fiducial"))
    if mode != "keypoint_fiducial":
        raise ValueError(
            "the primary keypoint pipeline requires vision.mode "
            f"'keypoint_fiducial', not {mode!r}")
    if vision.get("dictionary") or vision.get("board"):
        raise ValueError(
            "vision.dictionary/vision.board are ArUco settings and must be "
            "null in the primary keypoint experiment")
    pad = dict(vision.get("landing_pad") or {})
    deck = tuple(float(v) for v in (system.get("pad") or {}).get(
        "deck_size_m", (1.5, 1.5)))
    return {
        "deck_size_m": deck,
        "landmark_radius_m": float(pad.get("landmark_radius_m",
                                           PAD_LANDMARK_RADIUS_M)),
        "landmark_diameter_m": float(pad.get("landmark_diameter_m", 0.22)),
        "texture_pixels": int(pad.get("texture_pixels", 1024)),
        "layout": str(pad.get("layout", "hexagonal")),
    }


def _pad_texture(settings: Mapping[str, Any]) -> np.ndarray:
    """Rasterize the same fiducial target Isaac paints on the deck."""
    return landing_pad_texture(
        settings["deck_size_m"], pixels=int(settings["texture_pixels"]),
        landmark_radius_m=float(settings["landmark_radius_m"]),
        landmark_diameter_m=float(settings["landmark_diameter_m"]))


def _texture_pyramid(texture: np.ndarray) -> list:
    """Half-resolution levels, so a distant deck is filtered, not aliased."""
    import cv2

    levels = [texture]
    while min(levels[-1].shape) > 64:
        levels.append(cv2.pyrDown(levels[-1]))
    return levels


def _render_landing_target(image: np.ndarray, pyramid, deck_size_m,
                           position_pad, quaternion, model: CameraModel):
    """Paint the deck by intersecting each camera ray with the pad plane.

    Warping the four deck corners is only valid while all four are in front of
    the camera.  At touchdown altitude the near corners of a 1.5 m deck pass
    behind a 60-degree pitched camera, and a corner-warp then silently paints
    nothing at all -- producing frames with no visible target but with valid
    landmark labels, which is exactly the wrong thing to train on.  Ray-plane
    intersection is correct at every altitude and needs no special case.
    """
    import cv2

    origin, pad_from_optical = camera_pose_in_pad(position_pad, quaternion, model)
    half_x, half_y = (float(v) / 2.0 for v in deck_size_m)
    focal = model.focal_px
    columns, rows = np.meshgrid(np.arange(model.width, dtype=np.float64),
                                np.arange(model.height, dtype=np.float64))
    rays_optical = np.stack((
        (columns - model.width / 2.0) / focal,
        (rows - model.height / 2.0) / focal,
        np.ones_like(columns)), axis=-1)
    rays_pad = rays_optical @ pad_from_optical.T
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = -float(origin[2]) / rays_pad[..., 2]
    x = float(origin[0]) + distance * rays_pad[..., 0]
    y = float(origin[1]) + distance * rays_pad[..., 1]
    inside = (np.isfinite(distance) & (distance > 0.0)
              & (np.abs(x) <= half_x) & (np.abs(y) <= half_y))
    if not inside.any():
        return image

    # Pick the pyramid level whose texel spacing matches the projected deck, so
    # a pad a few pixels wide is averaged rather than point-sampled.
    span = max(int(inside.sum(axis=1).max()), int(inside.sum(axis=0).max()), 1)
    level = pyramid[0]
    for candidate in pyramid:
        level = candidate
        if candidate.shape[1] <= 2 * span:
            break
    height, width = level.shape
    # Texture row 0 is pad north, column 0 is pad west, matching the USD quad.
    map_x = np.where(inside, (x + half_x) / (2.0 * half_x) * (width - 1), 0.0)
    map_y = np.where(inside, (half_y - y) / (2.0 * half_y) * (height - 1), 0.0)
    sampled = cv2.remap(level, map_x.astype(np.float32), map_y.astype(np.float32),
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
    image[inside] = sampled[inside]
    return image


def _heatmap_targets(pixels: np.ndarray, visible: np.ndarray, model: CameraModel,
                     feature_h: int, feature_w: int) -> np.ndarray:
    grid_y, grid_x = np.mgrid[0:feature_h, 0:feature_w]
    heatmaps = np.zeros((PAD_LANDMARK_COUNT, feature_h, feature_w),
                        dtype=np.float32)
    for point in np.flatnonzero(visible):
        px = pixels[point, 0] / (model.width - 1.0) * (feature_w - 1.0)
        py = pixels[point, 1] / (model.height - 1.0) * (feature_h - 1.0)
        gaussian = np.exp(-((grid_x - px) ** 2 + (grid_y - py) ** 2)
                          / (2.0 * _HEATMAP_SIGMA ** 2))
        heatmaps[point] = gaussian / max(float(gaussian.sum()), 1e-9)
    return heatmaps


def synthetic_keypoint_dataset(system: Mapping[str, Any], *, samples: int,
                               seed: int) -> dict[str, np.ndarray]:
    """Render deployed-target views and exact six-landmark supervision."""
    import cv2

    model = camera_model(system)
    settings = landing_pad_settings(system)
    width, height = model.width, model.height
    mount = np.asarray(model.mount_translation_flu_m, dtype=float)
    pitch_down = model.pitch_down_deg
    pyramid = _texture_pyramid(_pad_texture(settings))
    deck_size = settings["deck_size_m"]
    landmark_radius = float(settings["landmark_radius_m"])

    rng = np.random.default_rng(int(seed))
    feature_h, feature_w = height // 16, width // 16

    images = np.empty((int(samples), height, width), dtype=np.uint8)
    heatmaps = np.zeros((int(samples), PAD_LANDMARK_COUNT, feature_h, feature_w),
                        dtype=np.float32)
    coordinates = np.zeros((int(samples), PAD_LANDMARK_COUNT, 2), dtype=np.float32)
    visible = np.zeros((int(samples), PAD_LANDMARK_COUNT), dtype=np.float32)
    poses = np.empty((int(samples), 5), dtype=np.float32)
    # The training-label-only pose each frame was rendered from, in the exact
    # ``[x, y, z, qw, qx, qy, qz]`` layout the simulator publishes, so the
    # empirical labeller can be round-tripped against the synthetic one.
    pad_relative_pose = np.empty(
        (int(samples), EMPIRICAL_POSE_LENGTH), dtype=np.float32)

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
        quaternion = _quat_wxyz_from_yaw(yaw)

        base = float(rng.uniform(45.0, 145.0))
        gx, gy = rng.uniform(-35.0, 35.0, 2)
        xx = np.linspace(-1.0, 1.0, width)[None, :]
        yy = np.linspace(-1.0, 1.0, height)[:, None]
        image = base + gx * xx + gy * yy
        image = image + rng.normal(0.0, rng.uniform(2.0, 10.0), (height, width))
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)

        image = _render_landing_target(
            image, pyramid, deck_size, body_pad, quaternion, model)

        if rng.random() < 0.55:
            # Partial visibility is the point of a keypoint encoder; keep the
            # landmark labels so descriptors learn to infer the rest.
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

        projection = project_landing_pad(
            body_pad, quaternion, camera=model,
            landmark_radius_m=landmark_radius)
        keypoint_visible = projection.keypoint_visible.copy()
        if rng.random() < 0.20:
            # Fully negative target-absent frames teach an explicit visibility
            # head. A heatmap soft-argmax alone always returns six coordinates
            # and cannot distinguish a blank/textured background from a pad.
            image = np.clip(
                rng.uniform(25.0, 225.0)
                + rng.normal(0.0, rng.uniform(3.0, 18.0), image.shape),
                0.0, 255.0).astype(np.uint8)
            keypoint_visible[:] = False
        coordinates[index] = projection.keypoint_normalized
        visible[index] = keypoint_visible.astype(np.float32)
        heatmaps[index] = _heatmap_targets(
            projection.keypoint_pixels, keypoint_visible, model,
            feature_h, feature_w)

        relative_platform_body = rotation.T @ -body_pad
        poses[index, :3] = np.clip(
            relative_platform_body / np.array([8.0, 8.0, 8.0]), -1.0, 1.0)
        poses[index, 3:] = (math.sin(yaw), math.cos(yaw))
        pad_relative_pose[index, :3] = body_pad
        pad_relative_pose[index, 3:] = quaternion
        images[index] = image

    if int(np.count_nonzero(visible)) < int(samples) * 2:
        raise RuntimeError("synthetic keypoint generator produced too few visible landmarks")
    return {"images": images, "heatmaps": heatmaps,
            "coordinates": coordinates, "visible": visible, "poses": poses,
            "pad_relative_pose": pad_relative_pose}


def empirical_keypoint_dataset(samples: Sequence, system: Mapping[str, Any],
                               ) -> dict[str, np.ndarray]:
    """Label real rendered frames by projecting the known pad landmarks.

    Each sample pairs one rendered grayscale frame with the simulator's
    ``training-label-only`` pose payload for that frame:
    ``[x, y, z, qw, qx, qy, qz]``, the UAV body origin in the gravity-aligned
    pad frame and its ENU/FLU attitude.  The labels are therefore produced by
    the same projection that defines the synthetic targets -- no detector, no
    homography and no marker id is involved.  Frames whose pose payload is
    missing, malformed or places every landmark outside the frame are dropped.
    """
    model = camera_model(system)
    settings = landing_pad_settings(system)
    width, height = model.width, model.height
    feature_h, feature_w = height // 16, width // 16
    landmark_radius = float(settings["landmark_radius_m"])
    labelled = []
    for sample in samples:
        try:
            image, pose = sample
        except (TypeError, ValueError):
            continue
        image = np.asarray(image, dtype=np.uint8)
        pose = np.asarray(pose, dtype=float).reshape(-1)
        if image.shape != (height, width):
            continue
        if pose.shape != (EMPIRICAL_POSE_LENGTH,) or not np.isfinite(pose).all():
            continue
        if float(np.linalg.norm(pose[3:])) < 1e-8:
            continue
        projection = project_landing_pad(
            pose[:3], pose[3:], camera=model,
            landmark_radius_m=landmark_radius)
        if not np.any(projection.keypoint_visible):
            continue
        labelled.append((
            image.copy(),
            _heatmap_targets(projection.keypoint_pixels,
                             projection.keypoint_visible, model,
                             feature_h, feature_w),
            projection.keypoint_normalized.astype(np.float32),
            projection.keypoint_visible.astype(np.float32)))
    if not labelled:
        return {
            "images": np.empty((0, height, width), dtype=np.uint8),
            "heatmaps": np.empty((0, PAD_LANDMARK_COUNT, feature_h, feature_w),
                                 dtype=np.float32),
            "coordinates": np.empty((0, PAD_LANDMARK_COUNT, 2), dtype=np.float32),
            "visible": np.empty((0, PAD_LANDMARK_COUNT), dtype=np.float32),
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
        path: str | Path, artifact: dict, labelled_source,
        *, system: Mapping[str, Any], experiment: Mapping[str, Any],
        mode: str, device: str | torch.device) -> dict:
    """Validate and optionally fine-tune a frozen encoder on live Isaac output.

    ``labelled_source()`` must return ``(image, pose)`` where ``pose`` is the
    simulator's training-label-only pad-relative UAV pose for that exact
    frame.  Labels are projected from the known pad landmarks; there is no
    detector in this path, so a frame whose target is unreadable still gets
    correct supervision instead of being silently discarded.
    """
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
    samples = []
    dataset = empirical_keypoint_dataset(samples, system)
    for _ in range(max(requested * 5, minimum)):
        samples.append(labelled_source())
        if len(samples) >= minimum:
            dataset = empirical_keypoint_dataset(samples, system)
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
        "synthetic six-keypoint fiducial projections plus geometry-labelled "
        "Isaac camera frames")
    artifact["empirical_calibration"] = {
        "validated": True,
        "label_source": "simulator pad-landmark projection (training-only)",
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
    bootstrap_value = settings.get("bootstrap_artifact")
    if bootstrap_value:
        bootstrap = Path(str(bootstrap_value)).expanduser()
        if not bootstrap.is_absolute():
            # Collapse ``..`` before testing existence: the new artifact's
            # parent may not have been created yet on the first fast run.
            bootstrap = (path.parent / bootstrap).resolve()
        if bootstrap.is_file():
            saved = torch.load(bootstrap, map_location="cpu", weights_only=False)
            compatible = (saved.get("format") == PRETRAIN_FORMAT
                          and saved.get("mode") == str(mode)
                          and saved.get("implementation")
                          == ShinKeypointEncoder.implementation)
            if compatible:
                probe = ShinKeypointEncoder(
                    settings["image_embedding"], keypoints=6)
                probe.load_state_dict(saved["encoder"])
                copied = dict(saved)
                copied.update({
                    "config_hash": str(config_hash),
                    "bootstrap_source": str(bootstrap.resolve()),
                    "training_source": (
                        f"compatible encoder bootstrapped from {bootstrap}"),
                })
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(path.suffix + ".tmp")
                torch.save(copied, temporary)
                os.replace(temporary, path)
                print(f"Bootstrapped compatible six-keypoint encoder from "
                      f"{bootstrap} into {path}.")
                return copied
    print("Pretraining six-keypoint descriptor encoder on synthetic "
          "deployed fiducial-target views...")
    state, metrics = _train_encoder(
        system, settings, mode=str(mode), device=torch.device(device))
    payload = {
        "format": PRETRAIN_FORMAT,
        "config_hash": str(config_hash),
        "mode": str(mode),
        "implementation": ShinKeypointEncoder.implementation,
        "frozen_for_ppo": True,
        "landmark_layout": KEYPOINT_LAYOUT_ID,
        "landing_pad_visual": LANDING_PAD_VISUAL_VERSION,
        "training_source": (
            "synthetic projections and target-absent negatives of the "
            "configured six-keypoint fiducial landing target"),
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
