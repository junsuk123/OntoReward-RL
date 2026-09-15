#!/usr/bin/env python3
"""The six-keypoint fiducial landing target painted on the deck.

This replaces the multi-scale ArUco board on the primary benchmark path.  The
deployed policy perceives the pad with a learned six-keypoint encoder, so the
pad must carry six distinctive, geometrically fixed landmarks rather than
bit-coded tags that a detector decodes into a pose.

No ArUco dictionary, marker id or ``cv2.aruco`` call appears anywhere in this
module.  Only ``cv2``'s plain drawing primitives are used to rasterize the
texture.

Design (a documented PACMAN-compatible approximation)
-----------------------------------------------------
The PACMAN asset and its landmark table are not public, so this is an
independent target built to the same *interface*: six fixed coplanar
landmarks, learnable from partial views, resolvable at both approach and
touchdown scale.

* Six bullseye landmarks sit on the vertices of the pad-centred hexagon
  defined once in :mod:`keypoint_geometry` -- the exact points the keypoint
  supervision labels.
* Each landmark carries ``index + 1`` solid pips on a small circle around it,
  so the six are mutually distinguishable without any bit decoding.
* Concentric graduation rings and radial spokes aligned with the landmark
  bearings keep the image informative at touchdown altitude, when the hexagon
  itself has already grown past the frame.

The geometry is imported from :mod:`keypoint_geometry` rather than restated,
so the painted target and the supervised labels cannot drift apart.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from keypoint_geometry import (KEYPOINT_LAYOUT_ID, PAD_LANDMARK_COUNT,
                               PAD_LANDMARK_RADIUS_M, pad_landmarks)


LANDING_PAD_VISUAL_VERSION = "six-keypoint-fiducial/1"
DEFAULT_TEXTURE_PIXELS = 2048
DEFAULT_LANDMARK_DIAMETER_M = 0.22

_WHITE = 255
_BLACK = 0


def _to_pixels(xy, deck_size_m, pixels: int) -> tuple[int, int]:
    """Pad-frame metres to texture pixels.

    ``st`` (0, 0) is the texture's bottom-left, which the spawned quad maps to
    the pad's south-west corner, so image rows run towards pad -Y.
    """
    half_x, half_y = (float(v) / 2.0 for v in deck_size_m)
    u = (float(xy[0]) + half_x) / (2.0 * half_x)
    v = (half_y - float(xy[1])) / (2.0 * half_y)
    return int(round(u * (pixels - 1))), int(round(v * (pixels - 1)))


def _scale(deck_size_m, pixels: int) -> float:
    """Texture pixels per metre along the pad's X axis."""
    return (pixels - 1) / float(deck_size_m[0])


def landing_pad_texture(deck_size_m=(1.5, 1.5), *,
                        pixels: int = DEFAULT_TEXTURE_PIXELS,
                        landmark_radius_m: float = PAD_LANDMARK_RADIUS_M,
                        landmark_diameter_m: float = DEFAULT_LANDMARK_DIAMETER_M,
                        ) -> np.ndarray:
    """Rasterize the fiducial target as a grayscale ``pixels x pixels`` image."""
    size = int(pixels)
    if size < 64:
        raise ValueError("landing-pad texture needs at least 64 pixels")
    deck = tuple(float(v) for v in deck_size_m)
    if len(deck) != 2 or not all(math.isfinite(v) and v > 0.0 for v in deck):
        raise ValueError("deck size must be two positive finite metres")
    landmark_radius = float(landmark_radius_m)
    landmark_diameter = float(landmark_diameter_m)
    if not 0.0 < landmark_diameter < 2.0 * landmark_radius:
        raise ValueError("landmark diameter must fit between pad centre and vertex")
    if landmark_radius + landmark_diameter / 2.0 > min(deck) / 2.0 + 1e-9:
        raise ValueError("landmarks must lie on the physical landing deck")

    per_metre = _scale(deck, size)
    image = np.full((size, size), _WHITE, dtype=np.uint8)

    # Deck border: a wide dark frame is the first thing resolvable on approach.
    border = max(2, int(round(0.055 * per_metre)))
    cv2.rectangle(image, (0, 0), (size - 1, size - 1), _BLACK, border * 2)

    centre_px = _to_pixels((0.0, 0.0), deck, size)

    # Graduation rings every 0.10 m keep touchdown-altitude frames textured.
    ring = max(1, int(round(0.012 * per_metre)))
    radius_m = 0.10
    while radius_m <= landmark_radius - landmark_diameter / 2.0 - 0.02:
        cv2.circle(image, centre_px, int(round(radius_m * per_metre)),
                   _BLACK, ring, cv2.LINE_AA)
        radius_m += 0.10

    # Radial spokes on the landmark bearings tie close-range structure to the
    # same hexagon the labels use.
    spoke = max(1, int(round(0.010 * per_metre)))
    for landmark in pad_landmarks(landmark_radius):
        angle = math.atan2(float(landmark[1]), float(landmark[0]))
        inner = 0.12 * landmark_radius
        outer = landmark_radius - landmark_diameter / 2.0 - 0.015
        if outer <= inner:
            continue
        cv2.line(image,
                 _to_pixels((inner * math.cos(angle), inner * math.sin(angle)),
                            deck, size),
                 _to_pixels((outer * math.cos(angle), outer * math.sin(angle)),
                            deck, size),
                 _BLACK, spoke, cv2.LINE_AA)

    # Centre bullseye: the touchdown-scale anchor.
    for scale, colour in ((0.075, _BLACK), (0.048, _WHITE), (0.024, _BLACK)):
        cv2.circle(image, centre_px, int(round(scale * per_metre)),
                   colour, -1, cv2.LINE_AA)

    # Six bullseye landmarks with an identity pip count.
    outer_px = int(round(landmark_diameter / 2.0 * per_metre))
    for index, landmark in enumerate(pad_landmarks(landmark_radius)):
        anchor = _to_pixels(landmark[:2], deck, size)
        cv2.circle(image, anchor, outer_px + max(2, outer_px // 8),
                   _WHITE, -1, cv2.LINE_AA)
        for fraction, colour in ((1.0, _BLACK), (0.62, _WHITE), (0.28, _BLACK)):
            cv2.circle(image, anchor, max(1, int(round(outer_px * fraction))),
                       colour, -1, cv2.LINE_AA)
        pip_orbit = landmark_diameter / 2.0 * 1.34
        pip_radius = max(1, int(round(landmark_diameter * 0.085 * per_metre)))
        bearing = math.atan2(float(landmark[1]), float(landmark[0]))
        for pip in range(index + 1):
            angle = bearing + math.pi + (pip - index / 2.0) * math.radians(26.0)
            offset = (float(landmark[0]) + pip_orbit * math.cos(angle),
                      float(landmark[1]) + pip_orbit * math.sin(angle))
            cv2.circle(image, _to_pixels(offset, deck, size), pip_radius,
                       _BLACK, -1, cv2.LINE_AA)
    return image


def generate_landing_pad_png(path: str | Path, deck_size_m=(1.5, 1.5),
                             **kwargs) -> Path:
    """Write the fiducial texture so Isaac can bind it as an OmniPBR map."""
    image = landing_pad_texture(deck_size_m, **kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"could not write landing-pad texture to {path}")
    return path


class LandingPadVisual:
    """The fiducial target as one textured quad parented to the deck prim.

    The quad is a child of the deck, so it rides the carrier without any
    per-frame bookkeeping: moving the parent moves the target.
    """

    def __init__(self, config: dict, workspace: Path,
                 parent_path: str = "/World/landing_pad",
                 deck_size_m=(1.5, 1.5)):
        pad = dict(config.get("landing_pad") or {})
        self.layout = str(pad.get("layout", "hexagonal"))
        if self.layout != "hexagonal":
            raise ValueError(
                "the six-keypoint landing target only defines the hexagonal layout")
        self.landmark_radius_m = float(
            pad.get("landmark_radius_m", PAD_LANDMARK_RADIUS_M))
        self.landmark_diameter_m = float(
            pad.get("landmark_diameter_m", DEFAULT_LANDMARK_DIAMETER_M))
        self.texture_pixels = int(pad.get("texture_pixels", DEFAULT_TEXTURE_PIXELS))
        self.deck_size_m = tuple(float(v) for v in deck_size_m)
        self.texture_dir = Path(workspace) / "assets" / "landing_pad"
        self.parent_path = parent_path
        self.landmark_count = PAD_LANDMARK_COUNT
        self.layout_id = KEYPOINT_LAYOUT_ID

    @property
    def landmarks_pad_m(self) -> np.ndarray:
        """The exact points the keypoint supervision labels."""
        return pad_landmarks(self.landmark_radius_m)

    def texture_path(self) -> Path:
        name = (f"six_keypoint_{self.landmark_radius_m:.3f}"
                f"_{self.landmark_diameter_m:.3f}_{self.texture_pixels}.png")
        return self.texture_dir / name

    @staticmethod
    def _omni_pbr():
        """Isaac renamed this module; accept either spelling.

        ``landing_world`` imports ``isaacsim.core.api.materials`` directly, so
        that name is the one this build is known to have; the legacy alias is
        kept for older installations.
        """
        try:
            from isaacsim.core.api.materials import OmniPBR
        except ImportError:  # pragma: no cover - depends on the Isaac build
            from omni.isaac.core.materials import OmniPBR
        return OmniPBR

    def spawn(self, world) -> None:
        # Imported lazily so this module stays testable outside Isaac.
        from pxr import UsdGeom, UsdShade

        OmniPBR = self._omni_pbr()

        stage = world.stage
        UsdGeom.Xform.Define(stage, self.parent_path)
        texture = generate_landing_pad_png(
            self.texture_path(), self.deck_size_m,
            pixels=self.texture_pixels,
            landmark_radius_m=self.landmark_radius_m,
            landmark_diameter_m=self.landmark_diameter_m)
        material = OmniPBR(
            prim_path=f"{self.parent_path}/material_six_keypoint",
            name="landing_pad_six_keypoint",
            texture_path=str(texture),
            texture_scale=np.array([1.0, 1.0]),
            texture_translate=np.array([0.0, 0.0]))
        # OmniPBR enables world-space UV projection in its constructor, which
        # ignores the quad's own UVs and crops the target.
        material.set_project_uvw(False)
        path = f"{self.parent_path}/six_keypoint_target"
        self._quad(stage, path, self.deck_size_m)
        UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).Bind(
            UsdShade.Material(stage.GetPrimAtPath(material.prim_path)))

    @staticmethod
    def _quad(stage, path: str, deck_size_m):
        from pxr import Gf, Sdf, UsdGeom

        mesh = UsdGeom.Mesh.Define(stage, path)
        half_x, half_y = (float(v) / 2.0 for v in deck_size_m)
        corners = [(-half_x, -half_y), (half_x, -half_y),
                   (half_x, half_y), (-half_x, half_y)]
        mesh.CreatePointsAttr([Gf.Vec3f(x, y, 0.0) for x, y in corners])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateNormalsAttr([Gf.Vec3f(0.0, 0.0, 1.0)] * 4)
        mesh.CreateExtentAttr([Gf.Vec3f(-half_x, -half_y, 0.0),
                               Gf.Vec3f(half_x, half_y, 0.0)])
        # st (0, 0) is the texture's bottom-left, so pad south-west; pad north
        # therefore lands on the texture's top row, matching ``_to_pixels``.
        primvars = UsdGeom.PrimvarsAPI(mesh)
        st = primvars.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                    UsdGeom.Tokens.varying)
        st.Set([Gf.Vec2f(0.0, 0.0), Gf.Vec2f(1.0, 0.0),
                Gf.Vec2f(1.0, 1.0), Gf.Vec2f(0.0, 1.0)])
        # Lifted a hair so the target does not z-fight with the deck.
        UsdGeom.XformCommonAPI(mesh).SetTranslate(Gf.Vec3d(0.0, 0.0, 0.002))
        return mesh
