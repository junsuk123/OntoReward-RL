"""Public keyframe retrieval and geometric verification with no truth access."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import MatchingConfig


@dataclass(frozen=True)
class ReferenceKeyframe:
    """Only algorithm-visible fields. Pose is held by GroundTruthLedger."""
    keyframe_id: str
    image: np.ndarray
    keypoints: tuple[cv2.KeyPoint, ...]
    descriptors: np.ndarray | None
    visual_signature: dict[str, float]


class ReferenceKeyframeDatabase:
    def __init__(self): self._items: list[ReferenceKeyframe] = []
    def add(self, keyframe: ReferenceKeyframe) -> None: self._items.append(keyframe)
    def all(self) -> tuple[ReferenceKeyframe, ...]: return tuple(self._items)


@dataclass
class MatchResult:
    selected_reference_keyframe: str | None = None
    raw_match_count: int = 0
    geometric_inlier_count: int = 0
    matching_inlier_ratio: float = 0.0
    reprojection_rmse: float = float("nan")
    rejected: bool = False


class FeatureMatcher:
    def __init__(self, cfg: MatchingConfig):
        self.cfg = cfg; self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def _verify(self, query_kp, query_desc, ref: ReferenceKeyframe, ratio: float) -> MatchResult:
        result = MatchResult()
        if query_desc is None or ref.descriptors is None or len(query_desc) < 2 or len(ref.descriptors) < 2: return result
        pairs = self.matcher.knnMatch(query_desc, ref.descriptors, k=2)
        good = [a for a, b in pairs if a.distance < ratio * b.distance]
        result.raw_match_count = len(good)
        if len(good) < self.cfg.min_matches: return result
        source = np.float32([query_kp[m.queryIdx].pt for m in good])
        target = np.float32([ref.keypoints[m.trainIdx].pt for m in good])
        model, mask = cv2.findHomography(source, target, cv2.RANSAC, self.cfg.ransac_threshold_px)
        if model is None or mask is None: return result
        result.selected_reference_keyframe = ref.keyframe_id
        keep = mask.ravel().astype(bool); result.geometric_inlier_count = int(keep.sum())
        result.matching_inlier_ratio = float(keep.mean())
        if keep.any():
            projected = cv2.perspectiveTransform(source[keep, None, :], model)[:, 0]
            result.reprojection_rmse = float(np.sqrt(np.mean(np.sum((projected-target[keep])**2, axis=1))))
        return result

    def localize(self, query_kp, query_desc, database: ReferenceKeyframeDatabase, matchability: float | None = None, guided: bool = False) -> MatchResult:
        if guided and matchability is not None and matchability < self.cfg.low_matchability_reject:
            return MatchResult(rejected=True)
        ratio = self.cfg.ratio_test
        if guided and matchability is not None:
            ratio = max(.65, min(.88, ratio + .08 * (matchability - .5)))
        candidates = [self._verify(query_kp, query_desc, ref, ratio) for ref in database.all()]
        if not candidates: return MatchResult()
        return max(candidates, key=lambda item: (item.geometric_inlier_count, item.matching_inlier_ratio, item.raw_match_count))


class GroundTruthLedger:
    """Evaluation-only pose/terrain store, never passed to FeatureMatcher."""
    def __init__(self): self._reference_xy: dict[str, tuple[float, float]] = {}; self._query_xy: dict[str, tuple[float, float]] = {}; self._query_ref: dict[str, str] = {}
    def reference(self, keyframe_id: str, xy: tuple[float, float]) -> None: self._reference_xy[keyframe_id] = xy
    def query(self, frame_id: str, xy: tuple[float, float], correct_reference: str) -> None: self._query_xy[frame_id] = xy; self._query_ref[frame_id] = correct_reference
    def evaluate(self, frame_id: str, result: MatchResult, tolerance_m: float) -> dict[str, float | bool | str | None]:
        selected = result.selected_reference_keyframe
        correct = selected == self._query_ref[frame_id]
        error = float("nan") if selected is None else float(np.linalg.norm(np.asarray(self._query_xy[frame_id])-np.asarray(self._reference_xy[selected])))
        rotation_error = float("nan") if selected is None else (0.0 if correct else 180.0)
        return {"ground_truth_reference": self._query_ref[frame_id], "is_correct_place_match": bool(correct and error <= tolerance_m), "translation_error": error, "rotation_error": rotation_error}
