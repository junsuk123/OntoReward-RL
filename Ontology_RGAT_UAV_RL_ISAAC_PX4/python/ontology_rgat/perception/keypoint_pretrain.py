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

Label convention (v6, 2026-09-20)
---------------------------------
The six landmarks sit on a regular hexagon.  Seen from the air the layout is
60-degree symmetric and the only cue to *which* vertex is pad-frame landmark 0
is the pip count painted next to each landmark: 0.8 px at 6 m, 1.2 px at 4 m,
3 px at 1.6 m.  Landmark identity is therefore not observable over most of the
approach, and a per-index loss whose target could be any of six cyclic
assignments has the pad centre as its optimum.  That is exactly what the
previous encoder learned: measured on 48 geometry-labelled Isaac frames its six
predictions had a spread of 12.9 px around a true spread of 44.5 px, i.e. it
reported the centroid six times (PCK@20 15 %).

Labels are now *canonical in the image plane*: index 0 is the projected vertex
with the smallest counter-clockwise angle from the +x image axis about the
projected pad centre, and the pad's cyclic order is kept (the projected order
around the centre is the same for every camera pose above the deck).  The
target is then a deterministic function of the image and the six channels
mean "the k-th vertex counter-clockwise", which is all the downstream consumers
-- centroid, apparent scale, pooled descriptors -- ever needed.

Synthetic frames (renderer v2) imitate what Isaac actually shows the camera: a
grey target on a dark, cluttered scene (building edges, poles, shadows, the
rover body and its mast next to the deck) rather than a white pad on a flat
gradient, and the empirical fine-tune warps and re-exposes the surveyed frames
with the labels transformed exactly, interleaved with synthetic replay so the
touchdown-scale views the survey never reaches are not forgotten.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..datastore import KIND_KEYPOINT_CALIBRATION, data_fingerprint
from ..initialization import (camera_centered_hover_offset,
                              yaw_aligned_hover_offset)
from .keypoint_encoder import KeypointEncoderOutput, ShinKeypointEncoder
from .pad_geometry import (KEYPOINT_LAYOUT_ID, LANDING_PAD_VISUAL_VERSION,
                           PAD_LANDMARK_COUNT, PAD_LANDMARK_RADIUS_M,
                           CameraModel, camera_pose_in_pad,
                           landing_pad_texture, project_landing_pad)


# v6: canonical image-plane landmark order, renderer v2, stride-8 heatmaps.
# The bump invalidates every v5 artifact: a v5 encoder was supervised on
# unobservable landmark identities and predicts the pad centre six times.
PRETRAIN_FORMAT = "shin2026-six-keypoint-fiducial-pretrain-v6-canonical"
# Stamped into ``empirical_calibration`` so an artifact calibrated under an
# older procedure is recalibrated instead of silently reused.
EMPIRICAL_CALIBRATION_FORMAT = "isaac-pose-surveyed-holdout-augmented-v2"
LABEL_CONVENTION = ("image-plane canonical cyclic order: index 0 is the projected "
                    "vertex with the smallest counter-clockwise angle from +x about "
                    "the projected pad centre; pad cyclic order kept")
# Training-label-only pose payload published by the simulator alongside each
# empirical frame: pad-relative UAV position (3) and ENU/FLU attitude (4).
EMPIRICAL_POSE_LENGTH = 7
# Gaussian target width in heatmap cells (8 px at stride 8).
_HEATMAP_SIGMA_CELLS = 1.0
# The survey's default pad-relative viewpoints. Altitudes span the Table-I
# approach band and the lateral offsets are fractions of altitude, which is
# how the synthetic generator jitters the pad across the image plane.
_DEFAULT_SURVEY_ALTITUDES_M = (1.6, 2.6, 4.0, 6.0)
# Two rings: the inner one keeps all six landmarks in frame, the outer one
# pushes the pad to the edge so the encoder is also certified on the partially
# visible views the policy spends its approach in.
_DEFAULT_SURVEY_LATERAL_FRACTIONS = (
    (0.0, 0.0), (0.30, 0.14), (-0.30, 0.14), (-0.30, -0.14), (0.30, -0.14),
    (0.70, 0.0), (-0.70, 0.0), (0.0, 0.34), (0.0, -0.34))
_DEFAULT_SURVEY_YAWS_DEG = (0.0, 35.0, -35.0)
# A viewpoint the pad is barely inside is a frame the labeller would drop, so
# the geometry is checked before the vehicle is ever sent there.
_SURVEY_MIN_VISIBLE_LANDMARKS = 4
# Pad-frame goto limits from the gateway protocol, kept as a margin here so a
# mis-parameterised survey is rejected on this side rather than by the gateway.
_SURVEY_MAX_RADIUS_M = 9.5
_SURVEY_MAX_ALTITUDE_M = 24.0
# Calibration frames stored by the superseded procedures. Their pixels and
# poses are exactly what the current labeller consumes, so they are read back
# under these fingerprints rather than re-flown (see
# :func:`keypoint_calibration_fingerprints`).
_LEGACY_CALIBRATION_FINGERPRINT_KEYS = (
    {"pretrain_format": "shin2026-six-keypoint-fiducial-pretrain-v5",
     "encoder_implementation":
         "hybrid-isaac-validated-six-keypoint-descriptor-v3-visibility"},
)


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


def _quat_wxyz_from_euler(yaw: float, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """ZYX Euler angles to a ``[w, x, y, z]`` quaternion."""
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def _yaw_from_quat_wxyz(quaternion) -> float:
    w, x, y, z = (float(v) for v in np.asarray(quaternion, dtype=float).reshape(4))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


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


_PYRAMID_CACHE: dict[tuple, list] = {}


def _cached_pyramid(settings: Mapping[str, Any]) -> list:
    key = (tuple(settings["deck_size_m"]), float(settings["landmark_radius_m"]),
           float(settings["landmark_diameter_m"]), int(settings["texture_pixels"]))
    pyramid = _PYRAMID_CACHE.get(key)
    if pyramid is None:
        pyramid = _texture_pyramid(_pad_texture(settings))
        _PYRAMID_CACHE[key] = pyramid
    return pyramid


def _plane_hits(position_pad, quaternion, model: CameraModel):
    """For every pixel: the pad-plane hit ``(x, y)`` and whether the ray hits.

    Ray-plane intersection is correct at every altitude: warping the deck's
    four corners is only valid while all four are in front of the camera, and
    at touchdown altitude the near corners of a 1.5 m deck pass behind a
    60-degree pitched camera.
    """
    origin, pad_from_optical = camera_pose_in_pad(position_pad, quaternion, model)
    focal = model.focal_px
    columns, rows = np.meshgrid(np.arange(model.width, dtype=np.float64),
                                np.arange(model.height, dtype=np.float64))
    rays = np.stack(((columns - model.width / 2.0) / focal,
                     (rows - model.height / 2.0) / focal,
                     np.ones_like(columns)), axis=-1) @ pad_from_optical.T
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = -float(origin[2]) / rays[..., 2]
    x = float(origin[0]) + distance * rays[..., 0]
    y = float(origin[1]) + distance * rays[..., 1]
    hit = np.isfinite(distance) & (distance > 0.0)
    return x, y, hit


def _sample_texture(x, y, inside, pyramid, deck_size_m) -> np.ndarray:
    """Texture values (0-255 float) at the plane hits, pyramid level matched."""
    import cv2

    half_x, half_y = (float(v) / 2.0 for v in deck_size_m)
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
    return cv2.remap(level, map_x.astype(np.float32), map_y.astype(np.float32),
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE).astype(np.float32)


def _render_landing_target(image: np.ndarray, pyramid, deck_size_m,
                           position_pad, quaternion, model: CameraModel):
    """Paint the deck (texture values as-is) into ``image`` by ray casting."""
    x, y, hit = _plane_hits(position_pad, quaternion, model)
    half_x, half_y = (float(v) / 2.0 for v in deck_size_m)
    inside = hit & (np.abs(x) <= half_x) & (np.abs(y) <= half_y)
    if not inside.any():
        return image
    sampled = _sample_texture(x, y, inside, pyramid, deck_size_m)
    image[inside] = sampled[inside]
    return image


# --------------------------------------------------------------------------
# canonical labels and heatmap targets
# --------------------------------------------------------------------------
def canonical_landmark_shift(pixels, center_pixels, in_front=None) -> int:
    """Roll amount ``s`` such that ``np.roll(labels, -s, 0)[0]`` is the vertex
    with the smallest counter-clockwise image-plane angle from +x about the
    projected pad centre (rows point down, so they are negated).

    Every projected vertex takes part, in frame or not, so the choice depends
    on the pad's image-plane pose and not on what happens to be visible.
    ``in_front`` (optional, one flag per vertex) excludes vertices behind the
    camera plane: :func:`project_pad_points` gives those a placeholder pixel
    that has no image meaning, and at touchdown altitude the rear vertices of
    a 1.5 m deck do pass behind a 60-degree pitched camera.  A projection
    with a non-finite vertex or centre, or no vertex in front, is left in pad
    order.
    """
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    center = np.asarray(center_pixels, dtype=np.float64).reshape(2)
    if not np.isfinite(pixels).all() or not np.isfinite(center).all():
        return 0
    angles = np.mod(np.arctan2(-(pixels[:, 1] - center[1]),
                               pixels[:, 0] - center[0]), 2.0 * math.pi)
    if in_front is not None:
        mask = np.asarray(in_front, dtype=bool).reshape(-1)
        if not mask.any():
            return 0
        angles = np.where(mask, angles, np.inf)
    return int(np.argmin(angles))


def canonicalize_landmarks(pixels, visible, center_pixels, in_front=None):
    """Return ``(pixels, visible, shift)`` rolled to the canonical order."""
    shift = canonical_landmark_shift(pixels, center_pixels, in_front)
    return (np.roll(np.asarray(pixels, dtype=np.float64), -shift, axis=0),
            np.roll(np.asarray(visible, dtype=bool), -shift, axis=0), shift)


def _normalized(pixels: np.ndarray, model: CameraModel) -> np.ndarray:
    pixels = np.asarray(pixels, dtype=np.float64)
    return np.column_stack((2.0 * pixels[:, 0] / (model.width - 1.0) - 1.0,
                            2.0 * pixels[:, 1] / (model.height - 1.0) - 1.0))


def _in_frame(pixels: np.ndarray, model: CameraModel) -> np.ndarray:
    pixels = np.asarray(pixels, dtype=np.float64)
    return (np.isfinite(pixels).all(axis=1)
            & (pixels[:, 0] >= 0.0) & (pixels[:, 0] <= model.width - 1.0)
            & (pixels[:, 1] >= 0.0) & (pixels[:, 1] <= model.height - 1.0))


def _heatmap_grid(model: CameraModel) -> tuple[int, int]:
    return ShinKeypointEncoder.heatmap_shape(model.height, model.width)


def _heatmap_targets(pixels: np.ndarray, visible: np.ndarray, model: CameraModel,
                     feature_h: int, feature_w: int,
                     sigma_cells: float = _HEATMAP_SIGMA_CELLS) -> np.ndarray:
    grid_y, grid_x = np.mgrid[0:feature_h, 0:feature_w]
    heatmaps = np.zeros((PAD_LANDMARK_COUNT, feature_h, feature_w),
                        dtype=np.float32)
    for point in np.flatnonzero(visible):
        px = pixels[point, 0] / (model.width - 1.0) * (feature_w - 1.0)
        py = pixels[point, 1] / (model.height - 1.0) * (feature_h - 1.0)
        gaussian = np.exp(-((grid_x - px) ** 2 + (grid_y - py) ** 2)
                          / (2.0 * float(sigma_cells) ** 2))
        heatmaps[point] = gaussian / max(float(gaussian.sum()), 1e-9)
    return heatmaps


def _canonical_projection(position_pad, quaternion, model: CameraModel,
                          landmark_radius_m: float):
    """Project the pad and return canonical ``(pixels, visible, center, shift)``."""
    projection = project_landing_pad(
        position_pad, quaternion, camera=model, landmark_radius_m=landmark_radius_m)
    center = np.asarray(projection.pad_center_pixels, dtype=float)
    if float(projection.pad_center_depth_m) <= 1e-9:
        # Camera at or below deck level: no image-plane order exists.
        return (np.asarray(projection.keypoint_pixels, dtype=float),
                np.asarray(projection.keypoint_visible, dtype=bool), center, 0)
    pixels, visible, shift = canonicalize_landmarks(
        projection.keypoint_pixels, projection.keypoint_visible, center,
        in_front=np.asarray(projection.keypoint_depth_m) > 0.0)
    return pixels, visible, center, shift


# --------------------------------------------------------------------------
# synthetic renderer v2
# --------------------------------------------------------------------------
@dataclass
class RenderSettings:
    """Domain randomisation matched to the Isaac calibration frames.

    Measured on the 48 surveyed frames: per-image mean 20-52/255, pad white
    ~100-126, pad black ~20-40, background 15-60 with hard-edged facades, thin
    poles and wires, soft shadows, a dark rover body touching the deck and a
    dark disc (the mast) next to one landmark.
    """

    altitude_range: tuple = (0.4, 8.0)
    touchdown_fraction: float = 0.2          # extra weight on 0.3-1.6 m
    tilt_sigma_deg: float = 6.0
    tilt_max_deg: float = 16.0
    jitter_fraction: tuple = (0.34, 0.25)    # of altitude, body x / y
    edge_fraction: float = 0.15              # push the pad partly out of frame
    negative_fraction: float = 0.15
    pad_white: tuple = (75.0, 145.0)
    pad_black: tuple = (8.0, 45.0)
    pad_min_contrast: float = 40.0
    pad_shading: float = 0.2
    background_base: tuple = (12.0, 65.0)
    bright_scene_probability: float = 0.2
    rover_body_probability: float = 0.8
    mast_probability: float = 0.8
    mast_radius_m: tuple = (0.06, 0.16)
    quads_max: int = 3
    lines_max: int = 6
    blobs_max: int = 3
    stripes_max: int = 2
    text_probability: float = 0.5
    noise_sigma: tuple = (0.5, 5.0)
    blur_probability: float = 0.7
    blur_sigma: tuple = (0.2, 1.3)
    gain_range: tuple = (0.8, 1.2)
    gamma_range: tuple = (0.85, 1.2)
    cutout_probability: float = 0.3

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "RenderSettings":
        values = {}
        names = {f.name for f in fields(cls)}
        for key, value in dict(mapping or {}).items():
            if key not in names:
                raise ValueError(f"unknown synthetic rendering setting {key!r}")
            values[key] = tuple(value) if isinstance(value, (list, tuple)) else value
        return cls(**values)


def _background(rng: np.random.Generator, model: CameraModel,
                cfg: RenderSettings) -> np.ndarray:
    import cv2

    H, W = model.height, model.width
    bright = rng.random() < cfg.bright_scene_probability
    base = rng.uniform(*cfg.background_base) + (rng.uniform(40, 110) if bright else 0.0)
    xx = np.linspace(-1.0, 1.0, W)[None, :]
    yy = np.linspace(-1.0, 1.0, H)[:, None]
    image = (base + rng.uniform(-25, 25) * xx + rng.uniform(-25, 25) * yy).astype(np.float32)
    # Large flat regions with hard edges: facades, road, grass.
    for _ in range(rng.integers(0, cfg.quads_max + 1)):
        points = np.array([[rng.uniform(-0.3, 1.3) * W, rng.uniform(-0.3, 1.3) * H]
                           for _ in range(4)], dtype=np.int32)
        hull = cv2.convexHull(points)
        level = rng.uniform(5, 110) + (rng.uniform(0, 80) if bright else 0.0)
        layer = image.copy()
        cv2.fillConvexPoly(layer, hull, float(level))
        alpha = rng.uniform(0.6, 1.0)
        image = (1.0 - alpha) * image + alpha * layer
        if rng.random() < 0.7:
            edge = rng.uniform(0, 30) if rng.random() < 0.5 else rng.uniform(90, 170)
            cv2.polylines(image, [hull], True, float(edge), int(rng.integers(1, 4)),
                          cv2.LINE_AA)
    # Thin straight structures: poles, wires, kerbs.
    for _ in range(rng.integers(0, cfg.lines_max + 1)):
        p0 = (int(rng.uniform(-0.2, 1.2) * W), int(rng.uniform(-0.2, 1.2) * H))
        p1 = (int(rng.uniform(-0.2, 1.2) * W), int(rng.uniform(-0.2, 1.2) * H))
        level = rng.uniform(0, 40) if rng.random() < 0.6 else rng.uniform(90, 170)
        cv2.line(image, p0, p1, float(level), int(rng.integers(1, 6)), cv2.LINE_AA)
    # Wide bright stripes: road markings, pavement.
    for _ in range(rng.integers(0, cfg.stripes_max + 1)):
        p0 = (int(rng.uniform(0, W)), int(rng.uniform(0, H)))
        p1 = (int(rng.uniform(0, W)), int(rng.uniform(0, H)))
        cv2.line(image, p0, p1, float(rng.uniform(70, 125)),
                 int(rng.integers(6, 22)), cv2.LINE_AA)
    # Text-like clutter on facades.
    if rng.random() < cfg.text_probability:
        x = int(rng.uniform(0, W * 0.8))
        y = int(rng.uniform(0, H * 0.9))
        level = rng.uniform(60, 140)
        glyph_h = int(rng.integers(6, 26))
        for _ in range(rng.integers(3, 9)):
            glyph_w = int(rng.integers(4, 18))
            cv2.rectangle(image, (x, y), (min(W - 1, x + glyph_w), min(H - 1, y + glyph_h)),
                          float(level), -1)
            x += glyph_w + int(rng.integers(2, 8))
            if x >= W:
                break
    # Soft dark blobs: tree shadows.
    if cfg.blobs_max:
        shadow = np.zeros_like(image)
        for _ in range(rng.integers(0, cfg.blobs_max + 1)):
            centre = (int(rng.uniform(0, W)), int(rng.uniform(0, H)))
            axes = (int(rng.uniform(10, 90)), int(rng.uniform(10, 60)))
            cv2.ellipse(shadow, centre, axes, float(rng.uniform(0, 180)), 0, 360,
                        float(rng.uniform(10, 40)), -1)
        if shadow.any():
            image = image - cv2.GaussianBlur(shadow, (0, 0), rng.uniform(3, 12))
    image = image * (1.0 + rng.normal(0.0, 0.04, image.shape)).astype(np.float32)
    return np.clip(image, 0.0, 255.0).astype(np.float32)


def render_synthetic_frame(rng: np.random.Generator, model: CameraModel,
                           pad: Mapping[str, Any], cfg: RenderSettings):
    """One synthetic frame.

    Returns ``(image uint8, pixels (6,2) canonical, visible (6,) canonical,
    center (2,), pose7, body_relative (3,))`` where ``pose7`` is the exact
    ``[x, y, z, qw, qx, qy, qz]`` layout the simulator publishes and
    ``body_relative`` is the pad relative to the vehicle in its yaw frame.
    """
    import cv2

    deck = tuple(float(v) for v in pad["deck_size_m"])
    landmark_radius = float(pad["landmark_radius_m"])
    mount = np.asarray(model.mount_translation_flu_m, dtype=float)
    image = _background(rng, model, cfg)
    negative = rng.random() < cfg.negative_fraction

    if rng.random() < cfg.touchdown_fraction:
        altitude = float(rng.uniform(0.3, 1.6))
    else:
        altitude = float(rng.uniform(*cfg.altitude_range))
    yaw = float(rng.uniform(-math.pi, math.pi))
    tilt = np.clip(rng.normal(0.0, math.radians(cfg.tilt_sigma_deg), 2),
                   -math.radians(cfg.tilt_max_deg), math.radians(cfg.tilt_max_deg))
    quaternion = _quat_wxyz_from_euler(yaw, float(tilt[0]), float(tilt[1]))
    rotation = _rotation_z(yaw)
    centred = camera_centered_hover_offset(altitude, model.pitch_down_deg, mount)
    jx, jy = cfg.jitter_fraction
    if rng.random() < cfg.edge_fraction:
        jitter = np.array([
            rng.uniform(-1, 1) * rng.uniform(jx, 2.2 * jx) * altitude,
            rng.uniform(-1, 1) * rng.uniform(jy, 2.2 * jy) * altitude, 0.0])
    else:
        jitter = np.array([rng.uniform(-jx, jx) * altitude,
                           rng.uniform(-jy, jy) * altitude, 0.0])
    body_pad = centred + jitter
    position = rotation @ body_pad

    x, y, hit = _plane_hits(position, quaternion, model)
    half_x, half_y = deck[0] / 2.0, deck[1] / 2.0
    if not negative:
        inside = hit & (np.abs(x) <= half_x) & (np.abs(y) <= half_y)
        if inside.any():
            texture = _sample_texture(x, y, inside, _cached_pyramid(pad), deck)
            white = rng.uniform(*cfg.pad_white)
            black = rng.uniform(*cfg.pad_black)
            if white - black < cfg.pad_min_contrast:
                white = black + cfg.pad_min_contrast + rng.uniform(0, 30)
            shade = 1.0 + cfg.pad_shading * (rng.uniform(-1, 1) * x / half_x
                                             + rng.uniform(-1, 1) * y / half_y) / 2.0
            painted = (black + texture / 255.0 * (white - black)) * shade
            image[inside] = painted[inside]
    # Rover body: a dark region touching the deck on one side, in the plane.
    if rng.random() < cfg.rover_body_probability:
        side = int(rng.integers(0, 4))
        length = rng.uniform(0.3, 1.2)
        width = rng.uniform(0.7, 1.15)
        if side == 0:
            body = (x > half_x) & (x <= half_x + length) & (np.abs(y) <= half_y * width)
        elif side == 1:
            body = (x < -half_x) & (x >= -half_x - length) & (np.abs(y) <= half_y * width)
        elif side == 2:
            body = (y > half_y) & (y <= half_y + length) & (np.abs(x) <= half_x * width)
        else:
            body = (y < -half_y) & (y >= -half_y - length) & (np.abs(x) <= half_x * width)
        body &= hit
        if negative and rng.random() < 0.5:
            # A dark deck-shaped quad with no target is a useful negative.
            body |= hit & (np.abs(x) <= half_x) & (np.abs(y) <= half_y)
        image[body] = rng.uniform(3, 30)
    # Mast: a dark disc next to one landmark, on the pad plane.
    if rng.random() < cfg.mast_probability and not negative:
        k = int(rng.integers(0, PAD_LANDMARK_COUNT))
        angle = math.pi / 6.0 + k * math.pi / 3.0 + rng.uniform(-0.3, 0.3)
        radius = landmark_radius * rng.uniform(0.75, 1.15)
        mx, my = radius * math.cos(angle), radius * math.sin(angle)
        disc_radius = rng.uniform(*cfg.mast_radius_m)
        disc = hit & ((x - mx) ** 2 + (y - my) ** 2 <= disc_radius ** 2)
        image[disc] = rng.uniform(0, 20)

    pixels, visible, center, _shift = _canonical_projection(
        position, quaternion, model, landmark_radius)
    if negative:
        visible = np.zeros(PAD_LANDMARK_COUNT, dtype=bool)

    if rng.random() < cfg.cutout_probability:
        # Partial occlusion; labels are kept so the descriptors learn to infer
        # the rest, exactly as the empirical visibility label (in frame) does.
        cw = int(rng.uniform(0.06, 0.3) * model.width)
        ch = int(rng.uniform(0.06, 0.3) * model.height)
        cx = int(rng.integers(0, model.width - cw))
        cy = int(rng.integers(0, model.height - ch))
        image[cy:cy + ch, cx:cx + cw] = rng.uniform(5, 150)
    if rng.random() < cfg.blur_probability:
        image = cv2.GaussianBlur(image, (0, 0), rng.uniform(*cfg.blur_sigma))
    image = image * rng.uniform(*cfg.gain_range)
    image = 255.0 * (np.clip(image, 0.0, 255.0) / 255.0) ** rng.uniform(*cfg.gamma_range)
    image = image + rng.normal(0.0, rng.uniform(*cfg.noise_sigma), image.shape)
    image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    pose = np.r_[position, quaternion].astype(np.float32)
    body_relative = rotation.T @ -position
    return image, pixels, visible, center, pose, body_relative


def _render_chunk(seeds, system: Mapping[str, Any], rendering: Mapping[str, Any]):
    """Worker entry point: render one frame per seed (spawn-safe)."""
    import cv2

    cv2.setNumThreads(1)
    model = camera_model(system)
    pad = landing_pad_settings(system)
    cfg = RenderSettings.from_mapping(rendering)
    return [render_synthetic_frame(np.random.default_rng(int(seed)), model, pad, cfg)
            for seed in seeds]


def _render_frames(seeds, system, rendering: Mapping[str, Any], workers: int | None):
    """Render inline unless ``workers`` > 1 was asked for explicitly.

    Inline rendering costs ~25 ms a frame (4096 frames in under two minutes),
    which is cheap next to the flight it precedes.  A spawn pool is faster
    but re-imports the caller's ``__main__`` in every worker; started from a
    stdin script it hung forever on 2026-09-20 instead of raising, and inside
    the flight runner it would re-import the whole learner.  So the pool is
    opt-in (``render_workers`` in the keypoint settings), never the default.
    """
    count = len(seeds)
    if workers is None or count < 256 or int(workers) <= 1:
        return _render_chunk(seeds, system, rendering)
    try:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing

        chunks = [seeds[i::int(workers)] for i in range(int(workers))]
        with ProcessPoolExecutor(
                max_workers=int(workers),
                mp_context=multiprocessing.get_context("spawn")) as pool:
            rendered = list(pool.map(_render_chunk, chunks,
                                     [dict(system)] * len(chunks),
                                     [dict(rendering)] * len(chunks)))
        # Restore the seed order so the result does not depend on the worker count.
        ordered = [None] * count
        for offset, chunk in enumerate(rendered):
            for index, frame in enumerate(chunk):
                ordered[offset + index * int(workers)] = frame
        return ordered
    except Exception as exc:  # pragma: no cover - depends on the host
        print(f"Parallel synthetic rendering unavailable ({exc}); rendering inline.")
        return _render_chunk(seeds, system, rendering)


def synthetic_keypoint_dataset(system: Mapping[str, Any], *, samples: int,
                               seed: int, workers: int | None = None,
                               rendering: Mapping[str, Any] | None = None,
                               ) -> dict[str, np.ndarray]:
    """Render deployed-target views and exact six-landmark supervision.

    Labels follow :data:`LABEL_CONVENTION`.  ``poses`` is the training-only
    auxiliary target of the synthetic pose head: the pad relative to the
    vehicle in its yaw frame, normalised by 8 m, plus ``sin 6*yaw``,
    ``cos 6*yaw`` -- the heading modulo the hexagon's own symmetry, which is
    what the image actually determines.
    """
    model = camera_model(system)
    feature_h, feature_w = _heatmap_grid(model)
    rendering = dict(rendering or {})
    RenderSettings.from_mapping(rendering)  # validate early
    seeds = np.random.default_rng(int(seed)).integers(0, 2**31 - 1, int(samples))
    frames = _render_frames([int(s) for s in seeds], system, rendering, workers)

    images = np.stack([frame[0] for frame in frames])
    pixels = np.stack([frame[1] for frame in frames]).astype(np.float32)
    visible = np.stack([frame[2] for frame in frames]).astype(np.float32)
    center = np.stack([frame[3] for frame in frames]).astype(np.float32)
    pad_relative_pose = np.stack([frame[4] for frame in frames]).astype(np.float32)
    body_relative = np.stack([frame[5] for frame in frames])
    heatmaps = np.stack([
        _heatmap_targets(pixels[i], visible[i] > 0.5, model, feature_h, feature_w)
        for i in range(len(frames))])
    coordinates = np.stack([_normalized(pixels[i], model) for i in range(len(frames))]
                           ).astype(np.float32)
    yaw = np.array([_yaw_from_quat_wxyz(pose[3:]) for pose in pad_relative_pose])
    poses = np.column_stack((
        np.clip(body_relative / 8.0, -1.0, 1.0),
        np.sin(6.0 * yaw), np.cos(6.0 * yaw))).astype(np.float32)

    if int(np.count_nonzero(visible)) < int(samples) * 2:
        raise RuntimeError("synthetic keypoint generator produced too few visible landmarks")
    return {"images": images, "heatmaps": heatmaps,
            "coordinates": coordinates, "visible": visible, "poses": poses,
            "pad_relative_pose": pad_relative_pose,
            "pixels": pixels, "center": center}


def empirical_keypoint_dataset(samples: Sequence, system: Mapping[str, Any],
                               *, rejections: dict[str, int] | None = None,
                               ) -> dict[str, np.ndarray]:
    """Label real rendered frames by projecting the known pad landmarks.

    Each sample pairs one rendered grayscale frame with the simulator's
    ``training-label-only`` pose payload for that frame:
    ``[x, y, z, qw, qx, qy, qz]``, the UAV body origin in the gravity-aligned
    pad frame and its ENU/FLU attitude.  The labels are therefore produced by
    the same projection that defines the synthetic targets -- no detector, no
    homography and no marker id is involved -- and ordered by the same
    :data:`LABEL_CONVENTION`.  Frames whose pose payload is missing, malformed
    or places every landmark outside the frame are dropped.

    ``rejections`` is filled, when given, with why each sample was dropped.
    Five separate conditions silently discard a frame here, and "0 labelled
    frames" on its own cannot tell an unpublished camera from a camera that
    works while the truth-pose stream does not.

    A sample may carry a third element, the index of the surveyed viewpoint it
    was captured at.  It is returned as ``viewpoint`` so the held-out split is
    taken over *poses* rather than over frames: 48 frames of one hover are 48
    copies of one measurement, and splitting them at random produced a 100 %
    validation score for an encoder that could not see the pad from anywhere
    else.  Frames from callers that do not survey are tagged ``-1``.
    """
    model = camera_model(system)
    settings = landing_pad_settings(system)
    width, height = model.width, model.height
    feature_h, feature_w = _heatmap_grid(model)
    landmark_radius = float(settings["landmark_radius_m"])
    counts = {"unpaired": 0, "no_image": 0, "no_pose": 0,
              "degenerate_attitude": 0, "no_landmark_in_frame": 0}
    labelled = []
    for sample in samples:
        try:
            image, pose, *surveyed = sample
        except (TypeError, ValueError):
            counts["unpaired"] += 1
            continue
        try:
            viewpoint = int(surveyed[0]) if surveyed else -1
        except (TypeError, ValueError):
            counts["unpaired"] += 1
            continue
        if image is None:
            counts["no_image"] += 1
            continue
        image = np.asarray(image, dtype=np.uint8)
        if image.shape != (height, width):
            counts["no_image"] += 1
            continue
        if pose is None:
            counts["no_pose"] += 1
            continue
        pose = np.asarray(pose, dtype=float).reshape(-1)
        if pose.shape != (EMPIRICAL_POSE_LENGTH,) or not np.isfinite(pose).all():
            counts["no_pose"] += 1
            continue
        if float(np.linalg.norm(pose[3:])) < 1e-8:
            counts["degenerate_attitude"] += 1
            continue
        pixels, visible, center, _shift = _canonical_projection(
            pose[:3], pose[3:], model, landmark_radius)
        if not np.any(visible):
            counts["no_landmark_in_frame"] += 1
            continue
        labelled.append((
            image.copy(),
            _heatmap_targets(pixels, visible, model, feature_h, feature_w),
            _normalized(pixels, model).astype(np.float32),
            visible.astype(np.float32),
            pose.astype(np.float32),
            viewpoint,
            pixels.astype(np.float32),
            center.astype(np.float32)))
    if rejections is not None:
        rejections.clear()
        rejections.update(counts)
    if not labelled:
        return {
            "images": np.empty((0, height, width), dtype=np.uint8),
            "heatmaps": np.empty((0, PAD_LANDMARK_COUNT, feature_h, feature_w),
                                 dtype=np.float32),
            "coordinates": np.empty((0, PAD_LANDMARK_COUNT, 2), dtype=np.float32),
            "visible": np.empty((0, PAD_LANDMARK_COUNT), dtype=np.float32),
            "pad_relative_pose": np.empty((0, EMPIRICAL_POSE_LENGTH),
                                          dtype=np.float32),
            "viewpoint": np.empty((0,), dtype=np.int64),
            "pixels": np.empty((0, PAD_LANDMARK_COUNT, 2), dtype=np.float32),
            "center": np.empty((0, 2), dtype=np.float32),
        }
    names = ("images", "heatmaps", "coordinates", "visible", "pad_relative_pose",
             "viewpoint", "pixels", "center")
    dataset = {name: np.stack([item[index] for item in labelled])
               for index, name in enumerate(names)}
    dataset["viewpoint"] = np.asarray(dataset["viewpoint"], dtype=np.int64)
    return dataset


# --------------------------------------------------------------------------
# empirical augmentation (labels transformed exactly)
# --------------------------------------------------------------------------
@dataclass
class AugmentationSettings:
    """Similarity warps and re-exposure of the surveyed Isaac frames.

    Two frames of one hover are one measurement; 24 hovers are 24.  A warp that
    moves the labels with the pixels turns each into a family of views at
    other scales, roll angles and image positions, which is what lets the
    fine-tune generalise to poses it never flew.  Mirroring is deliberately
    absent: it would turn the hexagon's chirality into a target that never
    exists.
    """

    scale_range: tuple = (0.6, 1.6)          # log-uniform
    rotation_deg: float = 30.0
    recenter_probability: float = 0.7        # move the pad centre anywhere in frame
    recenter_margin: float = 0.08
    jitter_fraction: float = 0.06            # otherwise: small translation
    gain_range: tuple = (0.7, 1.35)
    bias_range: tuple = (-15.0, 15.0)
    gamma_range: tuple = (0.8, 1.25)
    noise_sigma_max: float = 5.0
    blur_probability: float = 0.4
    blur_sigma_max: float = 1.0
    cutout_max: int = 2
    cutout_fraction: tuple = (0.05, 0.25)
    geometric_probability: float = 0.9
    photometric_probability: float = 0.9

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "AugmentationSettings":
        values = {}
        names = {f.name for f in fields(cls)}
        for key, value in dict(mapping or {}).items():
            if key not in names:
                raise ValueError(f"unknown empirical augmentation setting {key!r}")
            values[key] = tuple(value) if isinstance(value, (list, tuple)) else value
        return cls(**values)


def augment_labelled_frame(image, pixels, center, visible, rng: np.random.Generator,
                           cfg: AugmentationSettings, model: CameraModel):
    """Return ``(image, pixels, center, visible)`` after a random similarity
    warp and photometric perturbation.

    Landmarks outside the source frame stay invisible (there is no pixel
    content for them) and landmarks the warp pushes out of frame become
    invisible.  The returned labels are in canonical order for the new image.
    """
    import cv2

    width, height = model.width, model.height
    img = np.asarray(image).astype(np.float32)
    pts = np.asarray(pixels, dtype=np.float64).copy()
    ctr = np.asarray(center, dtype=np.float64).copy()
    vis = np.asarray(visible, dtype=bool).copy()
    finite_centre = bool(np.isfinite(ctr).all())
    if rng.random() < cfg.geometric_probability:
        scale = math.exp(rng.uniform(math.log(cfg.scale_range[0]),
                                     math.log(cfg.scale_range[1])))
        theta = math.radians(rng.uniform(-cfg.rotation_deg, cfg.rotation_deg))
        c, s = math.cos(theta), math.sin(theta)
        A = np.array([[scale * c, -scale * s], [scale * s, scale * c]])
        pivot = ctr if finite_centre else np.array([width / 2.0, height / 2.0])
        if finite_centre and rng.random() < cfg.recenter_probability:
            m = cfg.recenter_margin
            target = np.array([rng.uniform(m * width, (1.0 - m) * width),
                               rng.uniform(m * height, (1.0 - m) * height)])
        else:
            target = pivot + np.array([
                rng.uniform(-1, 1) * cfg.jitter_fraction * width,
                rng.uniform(-1, 1) * cfg.jitter_fraction * height])
        t = target - A @ pivot
        M = np.hstack((A, t[:, None]))
        fill = float(np.clip(img.mean() + rng.normal(0.0, 8.0), 0.0, 255.0))
        img = cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
        pts = pts @ A.T + t
        ctr = A @ ctr + t
        vis = vis & _in_frame(pts, model)
    if rng.random() < cfg.photometric_probability:
        img = np.clip(img * rng.uniform(*cfg.gain_range) + rng.uniform(*cfg.bias_range),
                      0.0, 255.0)
        img = 255.0 * (img / 255.0) ** rng.uniform(*cfg.gamma_range)
        if rng.random() < cfg.blur_probability:
            img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.2, cfg.blur_sigma_max))
        sigma = rng.uniform(0.0, cfg.noise_sigma_max)
        if sigma > 0.0:
            img = img + rng.normal(0.0, sigma, img.shape)
        for _ in range(rng.integers(0, cfg.cutout_max + 1)):
            cw = int(rng.uniform(*cfg.cutout_fraction) * width)
            ch = int(rng.uniform(*cfg.cutout_fraction) * height)
            cx = int(rng.integers(0, max(1, width - cw)))
            cy = int(rng.integers(0, max(1, height - ch)))
            img[cy:cy + ch, cx:cx + cw] = rng.uniform(5.0, 120.0)
    img = np.clip(img, 0.0, 255.0).astype(np.uint8)
    pts, vis, _shift = canonicalize_landmarks(pts, vis, ctr)
    return img, pts, ctr, vis


# --------------------------------------------------------------------------
# metrics and gates
# --------------------------------------------------------------------------
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
    # Spread of the six predictions around their centroid against the labels'
    # spread: an encoder that reports the pad centre six times has a ratio
    # near zero whatever its coordinate error looks like.
    ratios = []
    for frame in range(images.shape[0]):
        mask = visible[frame]
        if int(mask.sum()) < 2:
            continue
        pred_points = predicted[frame][mask] * scale
        true_points = target[frame][mask] * scale
        pred_spread = torch.sqrt(((pred_points - pred_points.mean(0)) ** 2).sum(-1).mean())
        true_spread = torch.sqrt(((true_points - true_points.mean(0)) ** 2).sum(-1).mean())
        ratios.append(float((pred_spread / true_spread.clamp_min(1e-6)).cpu()))
    # NaN, not 0, when no frame has two visible landmarks: the collapse gate
    # below must not fire on a split that cannot measure spread at all.
    spread_ratio = float(np.mean(ratios)) if ratios else float("nan")
    if selected.numel() == 0:
        return {"coordinate_rmse_px": float("inf"), "pck_20px": 0.0,
                "visibility_accuracy": 0.0, "visibility_recall": 0.0,
                "absent_false_positive_rate": false_positive_rate,
                "spread_ratio": spread_ratio}
    return {
        "coordinate_rmse_px": float(torch.sqrt(selected.square().mean()).cpu()),
        "pck_20px": float((selected <= 20.0).float().mean().cpu()),
        "visibility_accuracy": float(
            ((output.visibility >= 0.5) == visible).float().mean().cpu()),
        # Accuracy alone cannot separate "sees the pad" from "calls everything
        # absent" on a mostly-visible split. Recall is the quantity the flight
        # depends on: a landmark the encoder does not report is a landmark the
        # semantic observation counts as blind.
        "visibility_recall": float(
            (output.visibility >= 0.5)[visible].float().mean().cpu()),
        "absent_false_positive_rate": false_positive_rate,
        "spread_ratio": spread_ratio,
    }


def calibration_score(metrics: Mapping[str, Any]) -> float:
    """Rank a candidate encoder on the held-out viewpoints.

    Coordinate error alone selected the encoder this replaces: it scored 10 px
    on the frames it had memorised while reporting no landmark at all in
    flight. Blindness on unseen poses therefore has to cost as much as a gross
    localisation error, or the pose-held-out split changes nothing.
    """
    return (float(metrics["coordinate_rmse_px"])
            + 50.0 * float(metrics["absent_false_positive_rate"])
            + 50.0 * (1.0 - float(metrics.get("visibility_recall", 0.0))))


def calibration_viewpoints(system: Mapping[str, Any],
                           settings: Mapping[str, Any]) -> list[dict]:
    """The pad-relative poses the empirical calibration is sampled at.

    Every viewpoint is a camera-centred hover offset (the same construction the
    seeded entry pose uses) displaced across the image plane by a fraction of
    its own altitude and turned by a yaw, which is exactly how the synthetic
    generator jitters the pad.  Poses whose pad would fall outside the goto
    envelope, or which do not put at least
    ``_SURVEY_MIN_VISIBLE_LANDMARKS`` landmarks in frame, are dropped here
    rather than flown and then discarded by the labeller.
    """
    model = camera_model(system)
    pad = landing_pad_settings(system)
    mount = np.asarray(model.mount_translation_flu_m, dtype=float)
    pitch_down = float(model.pitch_down_deg)
    landmark_radius = float(pad["landmark_radius_m"])
    altitudes = tuple(float(value) for value in settings.get(
        "survey_altitudes_m", _DEFAULT_SURVEY_ALTITUDES_M))
    laterals = tuple(tuple(float(axis) for axis in pair) for pair in settings.get(
        "survey_lateral_fractions", _DEFAULT_SURVEY_LATERAL_FRACTIONS))
    yaws = tuple(math.radians(float(value)) for value in settings.get(
        "survey_yaws_deg", _DEFAULT_SURVEY_YAWS_DEG))
    if not altitudes or not laterals or not yaws:
        raise ValueError("the keypoint calibration survey needs at least one "
                         "altitude, lateral offset and yaw")
    if any(len(pair) != 2 for pair in laterals):
        raise ValueError("survey_lateral_fractions holds (x, y) pairs")
    viewpoints: list[dict] = []
    for altitude in altitudes:
        centred = camera_centered_hover_offset(altitude, pitch_down, mount)
        for lateral in laterals:
            yaw = yaws[len(viewpoints) % len(yaws)]
            body = centred + np.array(
                [lateral[0] * altitude, lateral[1] * altitude, 0.0])
            position = yaw_aligned_hover_offset(body, yaw)
            if (math.hypot(position[0], position[1]) > _SURVEY_MAX_RADIUS_M
                    or not 0.0 < position[2] <= _SURVEY_MAX_ALTITUDE_M):
                continue
            projection = project_landing_pad(
                position, _quat_wxyz_from_yaw(yaw), camera=model,
                landmark_radius_m=landmark_radius)
            visible = int(np.count_nonzero(projection.keypoint_visible))
            if visible < _SURVEY_MIN_VISIBLE_LANDMARKS:
                continue
            viewpoints.append({
                "index": len(viewpoints),
                "position_pad_m": tuple(float(axis) for axis in position),
                "yaw_rad": float(yaw), "altitude_m": float(altitude),
                "landmarks_in_frame": visible})
    if not viewpoints:
        raise ValueError(
            "no configured calibration viewpoint places the landing pad in "
            "the camera frame; check vision.camera and survey_altitudes_m")
    return viewpoints


def _calibration_fingerprint_parts(system: Mapping[str, Any],
                                   settings: Mapping[str, Any]) -> dict:
    model = camera_model(system)
    pad = landing_pad_settings(system)
    return {
        "calibration_format": "isaac-pose-surveyed-holdout-v1",
        "landmark_layout": KEYPOINT_LAYOUT_ID,
        "landing_pad_visual": LANDING_PAD_VISUAL_VERSION,
        "camera": {"width": model.width, "height": model.height,
                   "focal_px": round(float(model.focal_px), 6),
                   "pitch_down_deg": round(float(model.pitch_down_deg), 6),
                   "mount_translation_flu_m": [
                       round(float(axis), 6)
                       for axis in np.asarray(model.mount_translation_flu_m,
                                              dtype=float).reshape(-1)]},
        "landing_pad": {key: pad[key] for key in sorted(pad)},
        "survey": {
            "altitudes_m": list(settings.get(
                "survey_altitudes_m", _DEFAULT_SURVEY_ALTITUDES_M)),
            "lateral_fractions": [list(pair) for pair in settings.get(
                "survey_lateral_fractions", _DEFAULT_SURVEY_LATERAL_FRACTIONS)],
            "yaws_deg": list(settings.get(
                "survey_yaws_deg", _DEFAULT_SURVEY_YAWS_DEG)),
        },
    }


def keypoint_calibration_fingerprint(system: Mapping[str, Any],
                                     settings: Mapping[str, Any]) -> str:
    """What a stored calibration frame means, for deciding reuse across runs.

    The camera it was rendered by, the target painted on the deck and the
    survey that decides where the vehicle stands.  Not the experiment's
    configuration hash: an edit to the PPO budget does not change a pixel of a
    calibration frame.  And -- since v6 -- not the encoder architecture or the
    label format either: a stored viewpoint is an image and a pose, and every
    label is re-projected from that pose when the frames are read back, so a
    new encoder or a new label convention consumes the same frames.  Keying
    the frames on the encoder made every perception fix cost a fresh survey
    flight for nothing.
    """
    return data_fingerprint(_calibration_fingerprint_parts(system, settings))


def keypoint_calibration_fingerprints(system: Mapping[str, Any],
                                      settings: Mapping[str, Any]) -> list[str]:
    """Current fingerprint first, then the superseded ones whose frames are
    still exactly what the labeller consumes.  New viewpoints are stored under
    the first; all of them are read."""
    parts = _calibration_fingerprint_parts(system, settings)
    current = data_fingerprint(parts)
    legacy = []
    for extra in _LEGACY_CALIBRATION_FINGERPRINT_KEYS:
        legacy.append(data_fingerprint({**parts, **extra}))
    return [current, *legacy]


def _even_subset(items: Sequence, count: int) -> list:
    """``count`` entries spread evenly across ``items``, endpoints kept."""
    if count >= len(items):
        return list(items)
    positions = np.linspace(0.0, len(items) - 1.0, int(count)).round().astype(int)
    return [items[int(position)] for position in dict.fromkeys(positions.tolist())]


def _survey_frames(labelled_source, survey, viewpoints: Sequence[Mapping],
                   *, frames_per_viewpoint: int, frame_stride: int,
                   already_stored=(), store=None):
    """Fly each viewpoint and keep a few decorrelated frames from it.

    Consecutive 30 Hz frames of one hover differ by less than a grey level, so
    the stride is what makes ``frames_per_viewpoint`` more than one sample.

    A viewpoint the accumulation already holds is not flown again: re-flying
    it would spend simulator minutes to add a second, near-identical copy of a
    pose the fit is already anchored on, which is the weighting this store
    exists to avoid.
    """
    samples, visited, unreachable = [], [], []
    stride = max(1, int(frame_stride))
    skip = {int(index) for index in already_stored}
    for viewpoint in viewpoints:
        index = int(viewpoint["index"])
        if index in skip:
            continue
        if not survey(viewpoint):
            unreachable.append(index)
            continue
        visited.append(index)
        captured = []
        for poll in range(int(frames_per_viewpoint) * stride):
            image, pose = labelled_source()
            if poll % stride:
                continue
            captured.append((image, pose))
            samples.append((image, pose, index))
        if store is not None:
            store(viewpoint, captured)
    return samples, visited, unreachable


def _viewpoint_groups(dataset: Mapping[str, np.ndarray]) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for index, viewpoint in enumerate(np.asarray(dataset["viewpoint"]).tolist()):
        groups.setdefault(int(viewpoint), []).append(int(index))
    return groups


def _pose_span(dataset: Mapping[str, np.ndarray]) -> dict:
    """How far the accepted frames actually spread, in metres."""
    poses = np.asarray(dataset["pad_relative_pose"], dtype=float)
    altitude = poses[:, 2]
    lateral = poses[:, :2]
    if len(lateral) > 1:
        distances = np.linalg.norm(
            lateral[:, None, :] - lateral[None, :, :], axis=-1)
        lateral_span = float(distances.max())
    else:
        lateral_span = 0.0
    return {"altitude_span_m": float(altitude.max() - altitude.min()),
            "altitude_min_m": float(altitude.min()),
            "altitude_max_m": float(altitude.max()),
            "lateral_span_m": lateral_span}


def _reject_degenerate_calibration(span: Mapping[str, float],
                                   groups: Mapping[int, Sequence[int]],
                                   settings: Mapping[str, Any]) -> None:
    """Refuse to certify an encoder on frames that are all the same view.

    The previous procedure polled 48 consecutive frames of a single hover and
    split them at random.  Its held-out score was a memorisation test, the
    artifact was written with ``validated: true``, and the encoder then
    reported no landmark on 99.7 % of the in-frame steps of the run it was
    certified for.  Measure the spread of what was actually collected, and
    stop here rather than after four hundred blind episodes.
    """
    minimum_viewpoints = int(settings.get("minimum_calibration_viewpoints", 6))
    minimum_altitude = float(settings.get(
        "minimum_calibration_altitude_span_m", 1.5))
    minimum_lateral = float(settings.get(
        "minimum_calibration_lateral_span_m", 1.0))
    reasons = []
    if len(groups) < minimum_viewpoints:
        reasons.append(f"{len(groups)} distinct viewpoint(s) "
                       f"(minimum {minimum_viewpoints})")
    if span["altitude_span_m"] < minimum_altitude:
        reasons.append(
            f"altitude spread {span['altitude_span_m']:.2f} m "
            f"(minimum {minimum_altitude:.2f} m)")
    if span["lateral_span_m"] < minimum_lateral:
        reasons.append(
            f"lateral spread {span['lateral_span_m']:.2f} m "
            f"(minimum {minimum_lateral:.2f} m)")
    if reasons:
        raise RuntimeError(
            "Isaac keypoint calibration collected a degenerate frame set: "
            + "; ".join(reasons)
            + ". A held-out split of one pose measures memorisation, not "
            "perception, so no encoder is certified from it. Check that the "
            "calibration survey reached its viewpoints.")


def _held_out_viewpoints(groups: Mapping[int, Sequence[int]],
                         settings: Mapping[str, Any], rng,
                         frozen=None) -> tuple:
    """Hold out whole viewpoints, never frames of a viewpoint that trains.

    ``frozen`` is the accumulating store's own split, decided once per
    viewpoint and never redrawn. Without it the draw is per-run, which is
    correct for a one-off calibration but would let a viewpoint held out by
    one run train the next one -- and the held-out score that certifies the
    encoder would then be measured on a pose its lineage had already fitted.
    """
    identifiers = sorted(groups)
    fraction = float(settings.get("empirical_validation_fraction", .25))
    count = max(2, int(round(len(identifiers) * fraction)))
    count = min(count, len(identifiers) - 3)
    if count < 2:
        raise RuntimeError(
            f"Isaac keypoint calibration reached {len(identifiers)} viewpoints, "
            "too few to hold out two of them and still train on three.")
    if frozen is not None:
        held_out = sorted(set(frozen) & set(identifiers))
        if len(held_out) < 2 or len(identifiers) - len(held_out) < 3:
            raise RuntimeError(
                "the accumulated calibration split holds out "
                f"{len(held_out)} of {len(identifiers)} viewpoints, which "
                "cannot both validate on two and train on three. Survey more "
                "viewpoints rather than redrawing the split.")
    else:
        order = rng.permutation(len(identifiers))
        held_out = sorted(identifiers[int(position)] for position in order[:count])
    chosen = set(held_out)
    validation = np.asarray(
        sorted(index for key in held_out for index in groups[key]), dtype=int)
    training = np.asarray(
        sorted(index for key in identifiers if key not in chosen
               for index in groups[key]), dtype=int)
    return training, validation, held_out


def needs_empirical_calibration(artifact: Mapping[str, Any] | None) -> bool:
    """Whether this artifact still has to be surveyed against live Isaac.

    Also true for an artifact certified under a superseded procedure, so an
    old ``validated`` flag cannot carry a blind encoder into a new run.
    """
    if artifact is None:
        return False
    empirical = artifact.get("empirical_calibration") or {}
    return not (bool(empirical.get("validated"))
                and str(empirical.get("format")) == EMPIRICAL_CALIBRATION_FORMAT)


def _calibration_diagnosis(attempted: int, rejections: Mapping[str, int]) -> str:
    """Name the condition that discarded the frames, not just the count.

    Each of these points at a different part of the stack, and guessing
    between them costs an hour of a run that has already started.
    """
    if not attempted:
        return "The camera source was never polled."
    counts = {name: int(value) for name, value in rejections.items() if value}
    if not counts:
        return (f"{attempted} frames were polled and none was rejected, so the "
                "labelled set was simply never filled.")
    order = ["no_image", "no_pose", "degenerate_attitude",
             "no_landmark_in_frame", "unpaired"]
    dominant = max(order, key=lambda name: counts.get(name, 0))
    detail = ", ".join(f"{name}={counts[name]}" for name in order
                       if counts.get(name))
    remedy = {
        "no_image": ("the rendered camera topic published nothing usable -- "
                     "check that Isaac is stepping and the pair's image topic "
                     "is alive"),
        "no_pose": ("frames arrived but the training-only pad-relative truth "
                    "pose never matched their timestamps -- check that the "
                    "pair's truth-pose topic is publishing and that PX4's "
                    "timesync is not resetting"),
        "degenerate_attitude": "the truth pose carried a zero-norm quaternion",
        "no_landmark_in_frame": ("poses arrived but the pad projected entirely "
                                 "outside the frame -- the vehicle is not "
                                 "looking at the deck"),
        "unpaired": "the camera source did not return (image, pose) pairs",
    }[dominant]
    return f"Of {attempted} polled frames: {detail}. Most likely {remedy}."


# --------------------------------------------------------------------------
# training helpers shared by pretraining and calibration
# --------------------------------------------------------------------------
def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    import contextlib
    return contextlib.nullcontext()


def _float_output(output: KeypointEncoderOutput) -> KeypointEncoderOutput:
    return KeypointEncoderOutput(
        output.embedding.float(), output.keypoints.float(),
        output.heatmaps.float(), output.visibility.float())


def _keypoint_losses(output: KeypointEncoderOutput, coordinates, mask, heatmaps):
    denominator = mask.sum().clamp_min(1.0)
    heatmap_loss = (-(heatmaps.flatten(2)
                      * F.log_softmax(output.heatmaps.flatten(2), -1))
                    .sum(-1) * mask).sum() / denominator
    coordinate_loss = (((output.keypoints - coordinates).square().sum(-1) * mask)
                       .sum() / denominator)
    visibility_loss = _balanced_visibility_loss(output.visibility, mask)
    return heatmap_loss, coordinate_loss, visibility_loss


def _absent_loss(encoder, images, device):
    """A flat frame at the image mean must report no landmark."""
    negative = images.mean(dim=(-1, -2), keepdim=True).expand_as(images)
    negative = torch.clamp(negative + 0.08 * torch.randn_like(negative), 0.0, 1.0)
    with _autocast(device):
        visibility = encoder(negative).visibility
    visibility = visibility.float()
    return _balanced_visibility_loss(visibility, torch.zeros_like(visibility))


def _cosine_learning_rate(optimizer, base_lr: float, step: int, total: int,
                          warmup: int) -> None:
    if step < warmup:
        lr = base_lr * (step + 1) / max(1, warmup)
    else:
        progress = (step - warmup) / max(1, total - warmup)
        lr = 0.5 * base_lr * (1.0 + math.cos(math.pi * min(1.0, progress)))
    for group in optimizer.param_groups:
        group["lr"] = lr


def _batch_tensors(dataset: Mapping[str, np.ndarray], indices, device):
    images = torch.as_tensor(dataset["images"][indices, None],
                             dtype=torch.float32, device=device) / 255.0
    heatmaps = torch.as_tensor(dataset["heatmaps"][indices], dtype=torch.float32,
                               device=device)
    coordinates = torch.as_tensor(dataset["coordinates"][indices],
                                  dtype=torch.float32, device=device)
    mask = torch.as_tensor(dataset["visible"][indices], dtype=torch.float32,
                           device=device)
    return images, heatmaps, coordinates, mask


def _augmented_batch(dataset: Mapping[str, np.ndarray], indices, rng,
                     augmentation: AugmentationSettings, model: CameraModel, device):
    feature_h, feature_w = _heatmap_grid(model)
    images, heatmaps, coordinates, masks = [], [], [], []
    for index in indices:
        image, pixels, center, visible = augment_labelled_frame(
            dataset["images"][index], dataset["pixels"][index],
            dataset["center"][index], dataset["visible"][index] > 0.5, rng,
            augmentation, model)
        images.append(image)
        heatmaps.append(_heatmap_targets(pixels, visible, model, feature_h, feature_w))
        coordinates.append(_normalized(pixels, model).astype(np.float32))
        masks.append(visible.astype(np.float32))
    return (torch.as_tensor(np.stack(images)[:, None], dtype=torch.float32,
                            device=device) / 255.0,
            torch.as_tensor(np.stack(heatmaps), dtype=torch.float32, device=device),
            torch.as_tensor(np.stack(coordinates), dtype=torch.float32, device=device),
            torch.as_tensor(np.stack(masks), dtype=torch.float32, device=device))


# --------------------------------------------------------------------------
# empirical calibration
# --------------------------------------------------------------------------
def calibrate_keypoint_encoder(
        path: str | Path, artifact: dict, labelled_source,
        *, system: Mapping[str, Any], experiment: Mapping[str, Any],
        mode: str, device: str | torch.device, survey=None,
        datastore=None) -> dict:
    """Survey live Isaac output across poses, then fine-tune and certify.

    ``labelled_source()`` must return ``(image, pose)`` where ``pose`` is the
    simulator's training-label-only pad-relative UAV pose for that exact
    frame.  Labels are projected from the known pad landmarks; there is no
    detector in this path, so a frame whose target is unreadable still gets
    correct supervision instead of being silently discarded.

    ``survey(viewpoint)`` flies the vehicle to one entry of
    :func:`calibration_viewpoints` and returns whether it arrived.  Without it
    the frames all come from wherever the vehicle happens to be sitting, which
    the degeneracy gate below then refuses to certify.

    The fine-tune warps and re-exposes the training frames (labels transformed
    exactly) and interleaves synthetic replay so the synthetic trunk is
    adapted to the renderer without forgetting the touchdown-scale views the
    survey never reaches.  Two gates certify the result on held-out
    viewpoints: landmark recall, and the spread of the six predictions
    relative to the labels' spread, which is what catches an encoder that
    reports the pad centre six times.
    """
    if artifact is None:
        return None
    if not needs_empirical_calibration(artifact):
        print(f"Using empirically validated Isaac keypoint encoder from {path}.")
        return artifact
    estimator = dict(experiment.get("estimator") or {})
    settings = dict(estimator.get("keypoint_pretraining") or {})
    requested = int(settings.get(
        f"empirical_samples_{mode}", settings.get("empirical_samples", 48)))
    minimum = max(8, int(settings.get("minimum_empirical_samples", 8)))
    samples = []
    rejections: dict[str, int] = {}
    visited: list[int] = []
    unreachable: list[int] = []
    planned: list[dict] = []
    frames_per_viewpoint = 0
    fingerprint = None
    fingerprints: list[str] = []
    stored_records = []
    if datastore is not None:
        fingerprints = keypoint_calibration_fingerprints(system, settings)
        fingerprint = fingerprints[0]
        for candidate in fingerprints:
            stored_records.extend(datastore.episodes(
                KIND_KEYPOINT_CALIBRATION, candidate))
        for record in stored_records:
            payload = record.payload()
            images = np.asarray(payload["images"], dtype=np.uint8)
            poses = np.asarray(payload["poses"], dtype=float)
            index = int(record.provenance.get("viewpoint", record.seed))
            for frame in range(images.shape[0]):
                samples.append((images[frame], poses[frame], index))
        if stored_records:
            legacy = sum(record.fingerprint != fingerprint for record in stored_records)
            print(f"Reusing {len(stored_records)} accumulated calibration "
                  f"viewpoints ({sum(record.samples for record in stored_records)} "
                  f"frames) from {len({record.run_id for record in stored_records})} "
                  f"run(s); fingerprint {fingerprint[:12]}"
                  + (f", {legacy} of them stored under a superseded encoder key "
                     "and relabelled from their poses." if legacy else "."))
    if survey is not None:
        planned = calibration_viewpoints(system, settings)
        frame_stride = int(settings.get("survey_frame_stride", 4))
        # Two frames per pose, then as many poses as the frame budget allows:
        # a third frame of the same hover adds far less than a further
        # viewpoint does.
        frames_per_viewpoint = max(
            1, int(settings.get("survey_frames_per_viewpoint", 2)))
        planned = _even_subset(
            planned, max(1, requested // frames_per_viewpoint))
        held = {int(record.provenance.get("viewpoint", record.seed))
                for record in stored_records}

        def _store_viewpoint(viewpoint, captured):
            if datastore is None or not captured:
                return
            index = int(viewpoint["index"])
            datastore.store_episode(
                KIND_KEYPOINT_CALIBRATION, fingerprint, seed=index,
                payload={
                    "images": np.stack([
                        np.asarray(image, dtype=np.uint8)
                        for image, _pose in captured]),
                    "poses": np.stack([
                        np.asarray(pose, dtype=np.float32)
                        if pose is not None else
                        np.full(EMPIRICAL_POSE_LENGTH, np.nan, dtype=np.float32)
                        for _image, pose in captured]),
                },
                identity={"viewpoint": index},
                provenance={
                    "viewpoint": index,
                    "position_pad_m": [float(axis)
                                       for axis in viewpoint["position_pad_m"]],
                    "yaw_rad": float(viewpoint["yaw_rad"]),
                    "altitude_m": float(viewpoint["altitude_m"]),
                    "landmarks_in_frame": int(viewpoint["landmarks_in_frame"]),
                    "mode": str(mode),
                },
                samples=len(captured), environment_steps=0,
                split_key=f"viewpoint:{index}")

        surveyed, visited, unreachable = _survey_frames(
            labelled_source, survey, planned,
            frames_per_viewpoint=frames_per_viewpoint,
            frame_stride=frame_stride, already_stored=held,
            store=_store_viewpoint)
        samples.extend(surveyed)
        dataset = empirical_keypoint_dataset(
            samples, system, rejections=rejections)
    elif samples:
        # The accumulation already covers the survey, so nothing is flown at
        # all: no entry gate, no simulator minutes, and no second copy of a
        # viewpoint the fit is already anchored on.
        dataset = empirical_keypoint_dataset(samples, system, rejections=rejections)
    else:
        dataset = empirical_keypoint_dataset(samples, system, rejections=rejections)
        for _ in range(max(requested * 5, minimum)):
            samples.append(labelled_source())
            if len(samples) >= minimum:
                dataset = empirical_keypoint_dataset(
                    samples, system, rejections=rejections)
                if len(dataset["images"]) >= requested:
                    break
    count = len(dataset["images"])
    if count < minimum:
        detail = ""
        if survey is not None:
            detail = (f" The survey reached {len(visited)} of {len(planned)} "
                      f"viewpoints and could not reach {len(unreachable)}.")
        raise RuntimeError(
            f"Isaac keypoint calibration found only {count} labelled frames "
            f"(minimum {minimum}); refusing a synthetic-only frozen encoder. "
            + _calibration_diagnosis(len(samples), rejections) + detail)
    groups = _viewpoint_groups(dataset)
    span = _pose_span(dataset)
    _reject_degenerate_calibration(span, groups, settings)
    seed = int(settings.get("seed", 41026)) + 73
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    frozen_split = None
    if datastore is not None:
        fraction = float(settings.get("empirical_validation_fraction", .25))
        frozen_split = set()
        for candidate in fingerprints:
            frozen_split.update(
                int(record.provenance.get("viewpoint", record.seed))
                for record in datastore.episodes(
                    KIND_KEYPOINT_CALIBRATION, candidate, payloads=False)
                if record.split(fraction) == "validation")
    training, validation, held_out = _held_out_viewpoints(
        groups, settings, rng, frozen=frozen_split)
    torch_device = torch.device(device)
    model = camera_model(system)
    encoder = ShinKeypointEncoder(
        int(estimator.get("image_embedding", 512)), keypoints=6).to(torch_device)
    # Always restart from the synthetic weights. Fine-tuning on top of a
    # previous calibration compounds whatever that calibration overfitted to.
    encoder.load_state_dict(artifact.get("synthetic_encoder") or artifact["encoder"])
    before = _empirical_metrics(encoder, dataset, validation, torch_device)
    best = {name: value.detach().cpu().clone()
            for name, value in encoder.state_dict().items()}
    best_metrics = before

    augmentation = AugmentationSettings.from_mapping(
        settings.get("empirical_augmentation"))
    real_batch = max(1, int(settings.get("empirical_real_batch", 8)))
    replay_batch = max(0, int(settings.get("empirical_replay_batch", 8)))
    replay_samples = int(settings.get("empirical_replay_samples", 1024))
    replay = None
    if replay_batch > 0 and replay_samples > 0:
        replay = synthetic_keypoint_dataset(
            system, samples=replay_samples, seed=seed + 11,
            workers=settings.get("render_workers"),
            rendering=settings.get("synthetic_rendering"))
    else:
        replay_batch = 0
    if settings.get("empirical_steps") is not None:
        steps = max(1, int(settings["empirical_steps"]))
    else:
        epochs = max(1, int(settings.get("empirical_epochs", 60)))
        steps = epochs * max(1, math.ceil(len(training) / real_batch))
    learning_rate = float(settings.get("empirical_learning_rate", 3e-4))
    eval_interval = max(1, int(settings.get("empirical_eval_interval", 50)))
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=learning_rate,
                                  weight_decay=1e-4)
    encoder.train()
    for step in range(steps):
        real_indices = training[rng.integers(0, len(training), real_batch)]
        images, heatmaps, coordinates, mask = _augmented_batch(
            dataset, real_indices, rng, augmentation, model, torch_device)
        if replay_batch:
            replay_indices = np.sort(rng.integers(0, len(replay["images"]), replay_batch))
            r_images, r_heatmaps, r_coordinates, r_mask = _batch_tensors(
                replay, replay_indices, torch_device)
            images = torch.cat((images, r_images))
            heatmaps = torch.cat((heatmaps, r_heatmaps))
            coordinates = torch.cat((coordinates, r_coordinates))
            mask = torch.cat((mask, r_mask))
        with _autocast(torch_device):
            raw = encoder(images)
        output = _float_output(raw)
        heatmap_loss, coordinate_loss, visibility_loss = _keypoint_losses(
            output, coordinates, mask, heatmaps)
        # Retain the target-absent decision boundary while the trunk adapts
        # to the live Isaac renderer.
        absent_loss = _absent_loss(encoder, images[:real_batch], torch_device)
        loss = (heatmap_loss + 3.0 * coordinate_loss
                + 3.0 * visibility_loss + absent_loss)
        _cosine_learning_rate(optimizer, learning_rate, step, steps,
                              warmup=min(30, steps // 10))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
        optimizer.step()
        if (step + 1) % eval_interval == 0 or step + 1 == steps:
            encoder.eval()
            candidate = _empirical_metrics(
                encoder, dataset, validation, torch_device)
            encoder.train()
            if calibration_score(candidate) < calibration_score(best_metrics):
                best_metrics = candidate
                best = {name: value.detach().cpu().clone()
                        for name, value in encoder.state_dict().items()}
    minimum_recall = float(settings.get(
        "minimum_holdout_visibility_recall", 0.50))
    if float(best_metrics["visibility_recall"]) < minimum_recall:
        raise RuntimeError(
            "Isaac keypoint calibration produced an encoder that reports "
            f"{best_metrics['visibility_recall']:.1%} of the labelled "
            f"landmarks on the {len(held_out)} held-out viewpoints "
            f"(minimum {minimum_recall:.1%}). It would fly blind: the "
            "semantic observation counts an unreported landmark as no "
            "landmark. Collect more viewpoints or raise empirical_steps "
            "rather than starting PPO on it.")
    minimum_spread = float(settings.get("minimum_holdout_spread_ratio", 0.50))
    spread_ratio = float(best_metrics.get("spread_ratio", float("nan")))
    if math.isfinite(spread_ratio) and spread_ratio < minimum_spread:
        raise RuntimeError(
            "Isaac keypoint calibration produced an encoder whose six "
            "predictions have collapsed onto the pad centre: their spread on "
            f"the {len(held_out)} held-out viewpoints is "
            f"{best_metrics['spread_ratio']:.2f} of the labels' spread "
            f"(minimum {minimum_spread:.2f}). Such an encoder reports the "
            "same point six times and the apparent target scale is zero, so "
            "no policy can learn from it. Check the label convention and the "
            "synthetic pretraining rather than starting PPO on it.")
    artifact = dict(artifact)
    artifact["encoder"] = best
    artifact["training_source"] = (
        "synthetic six-keypoint fiducial projections (renderer v2) plus "
        "geometry-labelled Isaac camera frames surveyed across pad-relative "
        "viewpoints, warped and re-exposed with exact label transforms")
    artifact["label_convention"] = LABEL_CONVENTION
    artifact["empirical_calibration"] = {
        "validated": True,
        "format": EMPIRICAL_CALIBRATION_FORMAT,
        "label_source": "simulator pad-landmark projection (training-only)",
        "label_convention": LABEL_CONVENTION,
        "samples": count,
        "training_samples": int(len(training)),
        "validation_samples": int(len(validation)),
        "viewpoints": len(groups),
        "planned_viewpoints": len(planned),
        "unreachable_viewpoints": unreachable,
        "frames_per_viewpoint": frames_per_viewpoint,
        "held_out_viewpoints": held_out,
        "pose_span": span,
        "datastore_fingerprint": fingerprint,
        "datastore_fingerprints_read": fingerprints,
        "reused_viewpoints": len(stored_records),
        "split_source": ("frozen per-viewpoint datastore split"
                         if frozen_split is not None else "seeded per-run draw"),
        "steps": int(steps),
        "learning_rate": learning_rate,
        "real_batch": real_batch,
        "replay_batch": replay_batch,
        "replay_samples": int(replay_samples if replay_batch else 0),
        "augmentation": asdict(augmentation),
        "before": before, "after": best_metrics,
        "selection_score": ("coordinate_rmse_px + 50*absent_false_positive_rate"
                            " + 50*(1 - visibility_recall), on held-out "
                            "viewpoints"),
        "fine_tune_selected": bool(
            calibration_score(best_metrics) < calibration_score(before)),
    }
    dataset_path = Path(path).with_name("keypoint_isaac_calibration.npz")
    np.savez_compressed(dataset_path, **dataset)
    artifact["empirical_dataset"] = str(dataset_path.resolve())
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    torch.save(artifact, temporary)
    os.replace(temporary, path)
    print(
        "Isaac keypoint validation complete: "
        f"{count} frames from {len(groups)} viewpoints "
        f"({span['altitude_min_m']:.1f}-{span['altitude_max_m']:.1f} m "
        f"altitude, {span['lateral_span_m']:.1f} m lateral spread), "
        f"{len(held_out)} viewpoints held out, {steps} fine-tune steps; "
        f"{before['coordinate_rmse_px']:.1f}px -> "
        f"{best_metrics['coordinate_rmse_px']:.1f}px, "
        f"PCK@20 {before['pck_20px']:.1%} -> {best_metrics['pck_20px']:.1%}, "
        f"landmark recall {before['visibility_recall']:.1%} -> "
        f"{best_metrics['visibility_recall']:.1%}, "
        f"spread ratio {best_metrics['spread_ratio']:.2f}, "
        "target-absent false positives "
        f"{best_metrics['absent_false_positive_rate']:.1%}.")
    del encoder, optimizer, replay
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()
    return artifact


# --------------------------------------------------------------------------
# synthetic pretraining
# --------------------------------------------------------------------------
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
    dataset = synthetic_keypoint_dataset(
        system, samples=samples, seed=seed,
        workers=settings.get("render_workers"),
        rendering=settings.get("synthetic_rendering"))
    torch.manual_seed(seed)
    encoder = ShinKeypointEncoder(
        int(settings.get("image_embedding", 512)), keypoints=6).to(device)
    pose_head = nn.Linear(int(settings.get("image_embedding", 512)), 5).to(device)
    parameters = [*encoder.parameters(), *pose_head.parameters()]
    learning_rate = float(settings.get("learning_rate", 1e-3))
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
    rng = np.random.default_rng(seed + 1)
    total_steps = epochs * math.ceil(samples / batch_size)
    step = 0
    last = {}
    encoder.train()
    for _ in range(epochs):
        order = rng.permutation(samples)
        totals = {"loss": 0.0, "heatmap": 0.0, "coordinate": 0.0,
                  "visibility": 0.0, "pose": 0.0, "batches": 0}
        for start in range(0, samples, batch_size):
            indices = np.sort(order[start:start + batch_size])
            images, target_heatmaps, target_coordinates, mask = _batch_tensors(
                dataset, indices, device)
            pose = torch.as_tensor(
                dataset["poses"][indices], dtype=torch.float32, device=device)
            with _autocast(device):
                raw = encoder(images)
            output = _float_output(raw)
            heatmap_loss, coordinate_loss, visibility_loss = _keypoint_losses(
                output, target_coordinates, mask, target_heatmaps)
            pose_mask = (mask.sum(-1) > 0.0).float()
            pose_error = (pose_head(output.embedding) - pose).square().mean(-1)
            pose_loss = (pose_error * pose_mask).sum() / pose_mask.sum().clamp_min(1.0)
            loss = (heatmap_loss + 3.0 * coordinate_loss
                    + 3.0 * visibility_loss + 2.0 * pose_loss)
            _cosine_learning_rate(optimizer, learning_rate, step, total_steps,
                                  warmup=min(50, total_steps // 10))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            step += 1
            for name, value in (("loss", loss), ("heatmap", heatmap_loss),
                                ("coordinate", coordinate_loss),
                                ("visibility", visibility_loss),
                                ("pose", pose_loss)):
                totals[name] += float(value.detach())
            totals["batches"] += 1
        last = {name: value / totals["batches"]
                for name, value in totals.items() if name != "batches"}
    encoder.eval()
    state = {name: value.detach().cpu() for name, value in encoder.state_dict().items()}
    metrics = {**last, "samples": samples, "epochs": epochs,
               "visible_fraction": float(dataset["visible"].mean())}
    # Isaac Sim starts immediately after this phase and shares the GPU. Keep
    # only the CPU artifact so the temporary pose head and optimizer do not
    # reserve an otherwise invisible CUDA block during simulator startup.
    del encoder, pose_head, optimizer, dataset
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return state, metrics


def encoder_settings_fingerprint(experiment: Mapping[str, Any],
                                 system: Mapping[str, Any]) -> str:
    """What decides the trained encoder's weights: the estimator block (the
    pretraining, calibration and augmentation settings live there), the camera
    and the target painted on the deck -- not the PPO budget or the teacher
    gains, which share the experiment's config hash but change no weight.
    The 2026-09-20 restarts each spent 12 minutes re-training an identical
    encoder because a behavior-cloning key had changed."""
    return data_fingerprint({
        "pretrain_format": PRETRAIN_FORMAT,
        "implementation": ShinKeypointEncoder.implementation,
        "estimator": dict(experiment.get("estimator") or {}),
        "camera": dict((dict(system.get("vision") or {})).get("camera") or {}),
        "landing_pad": landing_pad_settings(system),
    })


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
    settings_fingerprint = encoder_settings_fingerprint(experiment, system)
    if path.is_file():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        same_lineage = (saved.get("format") == PRETRAIN_FORMAT
                        and saved.get("mode") == str(mode)
                        and saved.get("implementation") == ShinKeypointEncoder.implementation)
        if same_lineage and saved.get("config_hash") == str(config_hash):
            print(f"Using frozen synthetic six-keypoint encoder from {path}.")
            return saved
        if (same_lineage
                and saved.get("encoder_settings_fingerprint") == settings_fingerprint):
            # Same encoder settings, camera and target under a different
            # experiment hash: the weights would come out identical, so keep
            # them (and their empirical calibration) and only re-stamp the hash.
            saved = dict(saved)
            saved["config_hash"] = str(config_hash)
            saved.setdefault("config_hash_history", []).append(str(config_hash))
            temporary = path.with_suffix(path.suffix + ".tmp")
            torch.save(saved, temporary)
            os.replace(temporary, path)
            print(f"Using frozen six-keypoint encoder from {path} (identical "
                  f"encoder settings under a new experiment hash).")
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
          "deployed fiducial-target views (renderer v2, canonical labels)...")
    state, metrics = _train_encoder(
        system, settings, mode=str(mode), device=torch.device(device))
    payload = {
        "format": PRETRAIN_FORMAT,
        "config_hash": str(config_hash),
        "encoder_settings_fingerprint": settings_fingerprint,
        "mode": str(mode),
        "implementation": ShinKeypointEncoder.implementation,
        "frozen_for_ppo": True,
        "landmark_layout": KEYPOINT_LAYOUT_ID,
        "landing_pad_visual": LANDING_PAD_VISUAL_VERSION,
        "label_convention": LABEL_CONVENTION,
        "heatmap_stride": int(ShinKeypointEncoder.heatmap_stride),
        "training_source": (
            "synthetic projections (renderer v2: Isaac-like photometry and "
            "clutter) and target-absent negatives of the configured "
            "six-keypoint fiducial landing target"),
        "empirical_calibration": None,
        "metrics": metrics,
        "encoder": state,
        # Kept so a recalibration restarts from the synthetic weights instead
        # of fine-tuning on top of an earlier calibration's overfit.
        "synthetic_encoder": state,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    print(f"Frozen keypoint pretraining complete: loss={metrics['loss']:.4f}, "
          f"coordinate={metrics['coordinate']:.4f}, artifact={path}")
    return payload
