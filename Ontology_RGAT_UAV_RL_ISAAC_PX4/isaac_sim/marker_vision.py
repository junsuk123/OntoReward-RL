#!/usr/bin/env python3
"""ArUco landing pad: texture generation and camera-based pose recovery.

Isaac is not imported here, so the geometry is testable without a simulator.

Frames
------
``pad``      ENU at the pad centre: +X east, +Y north, +Z up.
``body``     FLU on the vehicle: +X forward, +Y left, +Z up.
``optical``  OpenCV/ROS camera: +X right, +Y down, +Z along the view direction.

The camera looks straight down, so optical +Z is body -Z. Taking optical +X
along body +X leaves optical +Y along body -Y, which is the 180 degree roll in
``R_BODY_FROM_OPTICAL``; everything else follows from it.

Why a board and not one marker
------------------------------
One marker cannot cover a landing. A tag large enough to resolve from the
entry altitude overflows the field of view before touchdown, and a tag small
enough to stay in view at touchdown is a handful of pixels from altitude. The
pad therefore carries markers at two scales with known positions, and the pose
is solved from whichever ones are currently visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


R_BODY_FROM_OPTICAL = np.array([
    [1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
])


def aruco_dictionary(name: str):
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"unknown ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def _dictionary_cells(name: str) -> int:
    """Cells across a marker including its one-cell black border."""
    return aruco_dictionary(name).markerSize + 2


def texture_side_ratio(dictionary: str, border_cells: float = 1.0) -> float:
    """How much wider the padded texture is than the marker itself.

    The pad quad carries the padded image, so the quad must be scaled by this
    or the metric marker side the pose assumes will be wrong.
    """
    cells = _dictionary_cells(dictionary)
    return (cells + 2.0 * border_cells) / cells


def generate_marker_png(path: str | Path, dictionary: str, marker_id: int,
                        pixels: int = 800, border_cells: float = 1.0) -> Path:
    """Write a marker image padded with the white quiet zone detection needs."""
    image = cv2.aruco.generateImageMarker(aruco_dictionary(dictionary), int(marker_id), int(pixels))
    margin = int(round(pixels / _dictionary_cells(dictionary) * border_cells))
    padded = cv2.copyMakeBorder(image, margin, margin, margin, margin,
                                cv2.BORDER_CONSTANT, value=255)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), padded):
        raise RuntimeError(f"could not write marker texture to {path}")
    return path


@dataclass(frozen=True)
class BoardMarker:
    """One marker lying flat on the pad."""
    marker_id: int
    side_m: float
    center_xy_m: tuple[float, float] = (0.0, 0.0)

    def object_points(self) -> np.ndarray:
        """Corners in the pad frame, in the order the detector returns them.

        OpenCV lists corners clockwise from the marker image's top-left. The
        marker lies flat with its image "up" along pad +Y, so image columns run
        towards pad +X and image rows towards pad -Y.
        """
        half = self.side_m / 2.0
        cx, cy = self.center_xy_m
        return np.array([
            [cx - half, cy + half, 0.0],
            [cx + half, cy + half, 0.0],
            [cx + half, cy - half, 0.0],
            [cx - half, cy - half, 0.0],
        ], dtype=np.float64)


class MarkerBoard:
    """The markers painted on the pad, keyed by id."""

    def __init__(self, markers: Iterable[BoardMarker]):
        self.markers = {int(m.marker_id): m for m in markers}
        if not self.markers:
            raise ValueError("a landing pad needs at least one marker")

    @classmethod
    def from_config(cls, entries: Sequence[dict]) -> "MarkerBoard":
        return cls(BoardMarker(int(e["id"]), float(e["side_m"]),
                               tuple(float(v) for v in e.get("center_xy_m", (0.0, 0.0))))
                   for e in entries)

    def __contains__(self, marker_id: int) -> bool:
        return int(marker_id) in self.markers

    def __len__(self) -> int:
        return len(self.markers)

    def object_points(self, marker_id: int) -> np.ndarray:
        return self.markers[int(marker_id)].object_points()


def intrinsics_from_fov(width: int, height: int, horizontal_fov_deg: float) -> np.ndarray:
    """Pinhole matrix for square pixels and a centred principal point."""
    if not 0.0 < horizontal_fov_deg < 180.0:
        raise ValueError("horizontal_fov_deg must be in (0, 180)")
    fx = (width / 2.0) / np.tan(np.radians(horizontal_fov_deg) / 2.0)
    return np.array([[fx, 0.0, width / 2.0],
                     [0.0, fx, height / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def matrix_to_quat_wxyz(r: np.ndarray) -> np.ndarray:
    trace = float(np.trace(r))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = np.array([0.25 * s, (r[2, 1] - r[1, 2]) / s,
                      (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(r)))
        if i == 0:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            q = np.array([(r[2, 1] - r[1, 2]) / s, 0.25 * s,
                          (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s])
        elif i == 1:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            q = np.array([(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s,
                          0.25 * s, (r[1, 2] + r[2, 1]) / s])
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            q = np.array([(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s,
                          (r[1, 2] + r[2, 1]) / s, 0.25 * s])
    q = q / np.linalg.norm(q)
    return q if q[0] >= 0.0 else -q


@dataclass(frozen=True)
class MarkerObservation:
    """One detection, in the terms the rest of the workspace speaks."""
    detected: bool
    position_pad_enu: np.ndarray          # vehicle body origin in the pad frame
    quaternion_pad_flu_wxyz: np.ndarray
    quality: float                        # 0 unseen, 1 for large crisp tags
    reprojection_px: float
    marker_pixels: float                  # side of the largest tag, in pixels
    marker_ids: tuple[int, ...] = ()

    @classmethod
    def missed(cls) -> "MarkerObservation":
        return cls(False, np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]),
                   0.0, float("nan"), 0.0, ())


def _detector_parameters() -> "cv2.aruco.DetectorParameters":
    parameters = cv2.aruco.DetectorParameters()
    # Integer corners cost real accuracy on a small, distant tag, which is
    # exactly the regime the entry altitude puts the pad in.
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return parameters


class MarkerPoseEstimator:
    """Recover the vehicle pose in the pad frame from a downward camera frame."""

    def __init__(self, board: MarkerBoard, camera_matrix: np.ndarray,
                 mount_translation_body: Sequence[float],
                 distortion: np.ndarray | None = None,
                 dictionary: str = "DICT_4X4_50",
                 quality_reprojection_px: float = 3.0,
                 quality_full_scale_px: float = 120.0,
                 body_from_optical: np.ndarray | None = None):
        self.board = board
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        self.distortion = (np.zeros(5) if distortion is None
                           else np.asarray(distortion, dtype=np.float64))
        self.mount_translation_body = np.asarray(mount_translation_body, dtype=np.float64)
        # Measured from the stage where possible: a simulator's camera-axis
        # convention is not something to assume, and getting it wrong rotates
        # every pose the policy flies on.
        self.body_from_optical = (R_BODY_FROM_OPTICAL if body_from_optical is None
                                  else np.asarray(body_from_optical, dtype=np.float64))
        self.quality_reprojection_px = float(quality_reprojection_px)
        self.quality_full_scale_px = float(quality_full_scale_px)
        self.detector = cv2.aruco.ArucoDetector(
            aruco_dictionary(dictionary), _detector_parameters())

    def detect(self, image: np.ndarray) -> MarkerObservation:
        if image is None or image.size == 0:
            return MarkerObservation.missed()
        gray = image if image.ndim == 2 else cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return MarkerObservation.missed()

        object_points, image_points, seen, sides = [], [], [], []
        for quad, marker_id in zip(corners, ids.flatten()):
            marker_id = int(marker_id)
            if marker_id not in self.board:
                continue
            quad = quad.reshape(4, 2).astype(np.float64)
            object_points.append(self.board.object_points(marker_id))
            image_points.append(quad)
            seen.append(marker_id)
            sides.append(np.sqrt(abs(cv2.contourArea(quad.astype(np.float32)))))
        if not seen:
            return MarkerObservation.missed()

        object_points = np.concatenate(object_points, axis=0)
        image_points = np.concatenate(image_points, axis=0)
        pose = self._solve(object_points, image_points)
        if pose is None:
            return MarkerObservation.missed()
        rvec, tvec, reprojection = pose
        return self._observation(rvec, tvec, reprojection, max(sides), tuple(sorted(seen)))

    def _solve(self, object_points, image_points):
        """Pick the pose supported by the image that also has the camera above the pad.

        Coplanar points are two-fold ambiguous, and a level downward camera
        over a flat pad sits exactly on that degeneracy: the mirrored pose
        reprojects just as well but places the vehicle underground. OpenCV's
        planar solver does not reliably return the branch we want, so score
        every candidate here and reject the ones below the pad.
        """
        flags = [cv2.SOLVEPNP_ITERATIVE]
        if len(object_points) == 4:
            flags.insert(0, cv2.SOLVEPNP_IPPE_SQUARE)
        candidates = []
        for flag in flags:
            try:
                count, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                    object_points, image_points, self.camera_matrix,
                    self.distortion, flags=flag)
            except cv2.error:
                continue
            candidates.extend((rvecs[i], tvecs[i]) for i in range(count))

        best = None
        for rvec, tvec in candidates:
            rotation, _ = cv2.Rodrigues(rvec)
            if float((-rotation.T @ tvec).reshape(3)[2]) <= 0.0:
                continue
            error = self._reprojection_error(object_points, image_points, rvec, tvec)
            if best is None or error < best[0]:
                best = (error, rvec, tvec)
        if best is None:
            return None
        return best[1], best[2], best[0]

    def _reprojection_error(self, object_points, image_points, rvec, tvec) -> float:
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, self.camera_matrix, self.distortion)
        return float(np.mean(np.linalg.norm(
            projected.reshape(-1, 2) - image_points, axis=1)))

    def _observation(self, rvec, tvec, reprojection, side_px, seen) -> MarkerObservation:
        r_optical_from_pad, _ = cv2.Rodrigues(rvec)
        r_pad_from_optical = r_optical_from_pad.T
        camera_in_pad = (-r_pad_from_optical @ tvec).reshape(3)
        r_pad_from_body = r_pad_from_optical @ self.body_from_optical.T
        body_in_pad = camera_in_pad - r_pad_from_body @ self.mount_translation_body
        return MarkerObservation(
            detected=True,
            position_pad_enu=body_in_pad,
            quaternion_pad_flu_wxyz=matrix_to_quat_wxyz(r_pad_from_body),
            quality=self._quality(reprojection, side_px, len(seen)),
            reprojection_px=reprojection,
            marker_pixels=float(side_px),
            marker_ids=seen,
        )

    def _quality(self, reprojection_px: float, side_px: float, count: int) -> float:
        """Confidence the ontology can consume: crisp, large and corroborated."""
        sharpness = 1.0 - min(1.0, reprojection_px / max(self.quality_reprojection_px, 1e-6))
        scale = min(1.0, side_px / max(self.quality_full_scale_px, 1e-6))
        # A second visible tag braces the pose; beyond that it adds little.
        corroboration = 1.0 if count > 1 else 0.85
        return float(np.clip(sharpness * scale * corroboration, 0.0, 1.0))
