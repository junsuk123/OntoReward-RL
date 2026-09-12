"""Deterministic synthetic pretraining for the six-keypoint image encoder.

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


PRETRAIN_FORMAT = "shin2026-synthetic-keypoint-pretrain-v1"


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
                  "pose": 0.0, "batches": 0}
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
            pose_loss = F.mse_loss(pose_head(output.embedding), pose)
            loss = heatmap_loss + 3.0 * coordinate_loss + 2.0 * pose_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [*encoder.parameters(), *pose_head.parameters()], 5.0)
            optimizer.step()
            for name, value in (("loss", loss), ("heatmap", heatmap_loss),
                                ("coordinate", coordinate_loss), ("pose", pose_loss)):
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
        "training_source": "synthetic projections of configured deployed marker board",
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
