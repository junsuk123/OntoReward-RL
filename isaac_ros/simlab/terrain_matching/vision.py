"""Procedural terrain, handcrafted visual features, and sparse optical flow."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .config import FlowConfig, ImageConfig
from .schema import TerrainVisualObservation


class TerrainSceneGenerator:
    def __init__(self, size: int, seed: int):
        self.size, self.seed = size, seed

    def generate(self, terrain: str, variant: int = 0) -> np.ndarray:
        rng = np.random.default_rng(self.seed + variant * 101 + sum(map(ord, terrain)))
        n = self.size
        image = np.zeros((n, n, 3), np.uint8)
        if terrain == "road":
            image[:] = (65, 65, 65)
            noise = rng.normal(0, 9, image.shape[:2]).astype(np.int16)
            image = np.clip(image.astype(np.int16) + noise[..., None], 0, 255).astype(np.uint8)
            cv2.line(image, (n // 2, 0), (n // 2, n), (230, 220, 70), max(3, n // 80))
            for y in range(-n, n * 2, n // 6):
                cv2.line(image, (n // 4, y), (n // 4, y + n // 12), (230, 230, 230), 4)
        elif terrain == "grass":
            image[:] = (38, 105, 48)
            for _ in range(n * 5):
                x, y = rng.integers(0, n, 2)
                color = (int(rng.integers(20, 70)), int(rng.integers(75, 150)), 35)
                cv2.line(image, (x, y), (x + int(rng.integers(-3, 4)), y - int(rng.integers(2, 8))), color, 1)
        elif terrain == "urban_building":
            image[:] = (105, 105, 110)
            block = max(28, n // 8)
            for y in range(8, n, block):
                for x in range(8, n, block):
                    shade = int(rng.integers(65, 190))
                    cv2.rectangle(image, (x, y), (min(n - 1, x + block - 8), min(n - 1, y + block - 8)), (shade, shade, shade), -1)
                    cv2.circle(image, (min(n - 1, x + block // 3), min(n - 1, y + block // 3)), 4, (30, 30, 30), -1)
        elif terrain == "water":
            image[:] = (150, 95, 35)
            for y in range(0, n, 11):
                offset = int(6 * math.sin(y / 17 + variant))
                cv2.line(image, (max(0, offset), y), (n - 1, y), (190, 135, 60), 2)
            image = cv2.GaussianBlur(image, (9, 9), 2.5)
        else:
            raise ValueError(f"unsupported terrain {terrain!r}")
        return image

    @staticmethod
    def perturb(image: np.ndarray, condition: str, rng: np.random.Generator) -> np.ndarray:
        h, w = image.shape[:2]
        if condition == "normal":
            matrix = np.float32([[1, 0, 5], [0, 1, 3]])
            return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        if condition in {"bright", "dark"}:
            alpha, beta = ((1.25, 22) if condition == "bright" else (0.60, -12))
            moved = cv2.warpAffine(image, np.float32([[1, 0, 4], [0, 1, 2]]), (w, h), borderMode=cv2.BORDER_REFLECT)
            return cv2.convertScaleAbs(moved, alpha=alpha, beta=beta)
        if condition == "blur":
            kernel = np.zeros((13, 13), np.float32); kernel[6, :] = 1 / 13
            return cv2.filter2D(image, -1, kernel)
        if condition == "yaw":
            angle = 30.0 + float(rng.normal(0, 1))
            return cv2.warpAffine(image, cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1), (w, h), borderMode=cv2.BORDER_REFLECT)
        raise ValueError(f"unsupported condition {condition!r}")


class VisualFeatureExtractor:
    def __init__(self, cfg: ImageConfig):
        self.cfg = cfg
        self.orb = cv2.ORB_create(nfeatures=cfg.orb_features)

    @staticmethod
    def _entropy(gray: np.ndarray) -> float:
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        p = hist[hist > 0] / gray.size
        return float(-(p * np.log2(p)).sum())

    def extract(self, frame_id: str, image: np.ndarray, timestamp: float, terrain: str | None = None) -> tuple[TerrainVisualObservation, list[cv2.KeyPoint], np.ndarray | None]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        edges = cv2.Canny(gray, self.cfg.canny_low, self.cfg.canny_high)
        corners = cv2.goodFeaturesToTrack(gray, 1000, 0.01, 5)
        cells = np.zeros((self.cfg.grid_size, self.cfg.grid_size), dtype=float)
        for keypoint in keypoints:
            x = min(self.cfg.grid_size - 1, int(keypoint.pt[0] * self.cfg.grid_size / gray.shape[1]))
            y = min(self.cfg.grid_size - 1, int(keypoint.pt[1] * self.cfg.grid_size / gray.shape[0]))
            cells[y, x] += 1
        occupancy = float(np.count_nonzero(cells) / cells.size)
        if cells.sum():
            p = cells.ravel() / cells.sum(); p = p[p > 0]
            uniformity = float(-(p * np.log(p)).sum() / np.log(cells.size))
        else:
            uniformity = 0.0
        # Similar descriptors at different locations are direct evidence of aliasing.
        repetitive = 0.0
        if descriptors is not None and len(descriptors) > 2:
            pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(descriptors, descriptors, k=3)
            ambiguous = [m[1].distance < 45 for m in pairs if len(m) >= 3]
            repetitive = float(np.mean(ambiguous)) if ambiguous else 0.0
        obs = TerrainVisualObservation(
            frame_id=frame_id, timestamp_s=timestamp,
            texture_entropy=self._entropy(gray), texture_variance=float(gray.var()),
            edge_density=float(np.mean(edges > 0)),
            corner_density=0.0 if corners is None else min(1.0, len(corners) / (gray.size / 1000)),
            feature_count=len(keypoints), feature_distribution_uniformity=min(1.0, uniformity * occupancy),
            repetitiveness_score=min(1.0, repetitive), brightness=float(gray.mean()),
            blur_score=float(cv2.Laplacian(gray, cv2.CV_64F).var()), terrain_type_if_available=terrain,
        )
        obs.normalized = {
            "texture_entropy": min(1.0, obs.texture_entropy / 8.0),
            "corner_density": obs.corner_density,
            "feature_count": min(1.0, obs.feature_count / max(1, self.cfg.orb_features)),
            "feature_distribution_uniformity": obs.feature_distribution_uniformity,
            "repetitiveness_score": obs.repetitiveness_score,
            "brightness_quality": max(0.0, 1.0 - abs(obs.brightness - 127.5) / 127.5),
            "blur_quality": min(1.0, obs.blur_score / 500.0),
        }
        obs.validate()
        return obs, keypoints, descriptors


class OpticalFlowAnalyzer:
    def __init__(self, cfg: FlowConfig): self.cfg = cfg

    def update(self, previous: np.ndarray, current: np.ndarray, obs: TerrainVisualObservation) -> None:
        old = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY); new = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        points = cv2.goodFeaturesToTrack(old, self.cfg.max_corners, self.cfg.quality_level, self.cfg.min_distance)
        if points is None:
            return
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(old, new, points, None)
        if nxt is None or status is None:
            return
        valid = status.ravel().astype(bool); p0, p1 = points[valid, 0], nxt[valid, 0]
        obs.flow_valid_ratio = float(len(p1) / len(points))
        if len(p1) < 4:
            return
        vectors = p1 - p0; magnitude = np.linalg.norm(vectors, axis=1)
        obs.flow_mean, obs.flow_std = float(magnitude.mean()), float(magnitude.std())
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        obs.flow_direction_consistency = float(np.hypot(np.sin(angles).mean(), np.cos(angles).mean()))
        _, mask = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=self.cfg.ransac_threshold_px)
        obs.flow_inlier_ratio = 0.0 if mask is None else float(mask.mean())
        obs.normalized.update({"flow_valid_ratio": obs.flow_valid_ratio, "flow_inlier_ratio": obs.flow_inlier_ratio, "flow_direction_consistency": obs.flow_direction_consistency})
        obs.validate()
