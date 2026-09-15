"""Single-source access to the simulator's six-keypoint pad definition.

``isaac_sim/keypoint_geometry.py`` is the one definition of the landing-pad
landmarks, of their projection into the rendered camera, and of geometric
pad-centre field-of-view retention; ``isaac_sim/landing_pad_visual.py`` is the
one definition of the fiducial target painted on the deck.  The Isaac stage
builder imports both directly.  The learner package re-exports the very same
modules here so the painted target, the supervised labels and the geometric
FOV ground truth can never drift into three different hexagons.

Everything re-exported from ``keypoint_geometry`` is simulator truth and
carries that module's provenance contract: training-label-only,
initialization-only or evaluation-only.  None of it may reach the PPO actor or
the online R-GAT graph.
"""
from __future__ import annotations

from pathlib import Path
import sys


_ISAAC_DIR = Path(__file__).resolve().parents[3] / "isaac_sim"
if not (_ISAAC_DIR / "keypoint_geometry.py").is_file():
    raise ImportError(
        f"the canonical pad keypoint geometry is missing at {_ISAAC_DIR}")
# Appended rather than prepended: this must never shadow an installed package,
# only make the simulator-side geometry modules resolvable by name.
if str(_ISAAC_DIR) not in sys.path:
    sys.path.append(str(_ISAAC_DIR))

from keypoint_geometry import (  # noqa: E402
    DEFAULT_CAMERA, KEYPOINT_LAYOUT_ID, PAD_LANDMARK_COUNT,
    PAD_LANDMARK_RADIUS_M, CameraModel, PadKeypointProjection,
    body_from_optical, camera_pose_in_pad, focal_length_px,
    geometric_pad_center_in_fov, nadir_footprint_m, pad_center, pad_landmarks,
    project_landing_pad, project_pad_points)
from landing_pad_visual import (  # noqa: E402
    DEFAULT_LANDMARK_DIAMETER_M, LANDING_PAD_VISUAL_VERSION,
    landing_pad_texture)

__all__ = [
    "CameraModel", "DEFAULT_CAMERA", "DEFAULT_LANDMARK_DIAMETER_M",
    "KEYPOINT_LAYOUT_ID", "LANDING_PAD_VISUAL_VERSION", "PAD_LANDMARK_COUNT",
    "PAD_LANDMARK_RADIUS_M", "PadKeypointProjection", "body_from_optical",
    "camera_pose_in_pad", "focal_length_px", "geometric_pad_center_in_fov",
    "landing_pad_texture", "nadir_footprint_m", "pad_center", "pad_landmarks",
    "project_landing_pad", "project_pad_points",
]
