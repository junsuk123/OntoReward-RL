"""Calibrated parallel front pair and nadir satellite-imaging camera."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import omni.usd
from pxr import Gf, Sdf, UsdGeom

from simlab.config.schema import CameraOpticsConfig, CamerasConfig
from simlab.utils.logging import get_logger

log = get_logger("cameras")
Vec3 = Tuple[float, float, float]


def _normalize(vector: Vec3) -> Vec3:
    length = math.sqrt(sum(component * component for component in vector))
    if length < 1e-12:
        raise ValueError("camera view vector cannot be zero")
    return tuple(component / length for component in vector)  # type: ignore[return-value]


@dataclass(frozen=True)
class CameraDescriptor:
    sensor_id: str
    prim_path: str
    role: str
    resolution: Tuple[int, int]


class CameraRig:
    """Three camera prims with stable sensor IDs and calibration metadata."""

    def __init__(self, descriptors: Dict[str, CameraDescriptor]) -> None:
        self.descriptors = descriptors

    @classmethod
    def spawn(cls, cfg: CamerasConfig) -> "CameraRig":
        stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage, cfg.root_prim)

        direction = _normalize(cfg.front.view_direction)
        baseline = _normalize(cfg.front.baseline_direction)
        near_position = cfg.front.near_position
        far_position = tuple(
            near_position[index] + baseline[index] * cfg.front.separation_m
            for index in range(3)
        )
        descriptors: Dict[str, CameraDescriptor] = {}
        camera_specs = (
            (
                "front_near",
                f"{cfg.root_prim}/FrontNear",
                near_position,
                direction,
                cfg.front.optics,
                "front_parallel",
                cfg.front.image_rotation_clockwise_deg,
            ),
            (
                "front_far",
                f"{cfg.root_prim}/FrontFar",
                far_position,
                direction,
                cfg.front.optics,
                "front_parallel",
                cfg.front.image_rotation_clockwise_deg,
            ),
            (
                "satellite_nadir",
                f"{cfg.root_prim}/SatelliteNadir",
                cfg.satellite.position,
                _normalize(
                    tuple(
                        cfg.satellite.target[index] - cfg.satellite.position[index]
                        for index in range(3)
                    )
                ),
                cfg.satellite.optics,
                "satellite",
                0.0,
            ),
        )
        for sensor_id, path, position, optical_axis, optics, group, clockwise_deg in camera_specs:
            cls._create_camera(
                stage,
                path,
                position,
                optical_axis,
                optics,
                sensor_id=sensor_id,
                calibration_group=group,
                front_separation=cfg.front.separation_m if group == "front_parallel" else 0.0,
                image_rotation_clockwise_deg=clockwise_deg,
            )
            descriptors[sensor_id] = CameraDescriptor(
                sensor_id=sensor_id,
                prim_path=path,
                role=sensor_id,
                resolution=optics.resolution,
            )

        log(
            f"created parallel front cameras, y-baseline={cfg.front.separation_m:g}m; "
            f"satellite camera at z={cfg.satellite.position[2]:g}m"
        )
        return cls(descriptors)

    @staticmethod
    def _create_camera(
        stage,
        path: str,
        position: Vec3,
        optical_axis: Vec3,
        optics: CameraOpticsConfig,
        sensor_id: str,
        calibration_group: str,
        front_separation: float,
        image_rotation_clockwise_deg: float,
    ) -> None:
        camera = UsdGeom.Camera.Define(stage, path)
        camera.CreateProjectionAttr(UsdGeom.Tokens.perspective)
        camera.CreateFocalLengthAttr(optics.focal_length_mm)
        camera.CreateHorizontalApertureAttr(optics.horizontal_aperture_mm)
        camera.CreateVerticalApertureAttr(
            optics.horizontal_aperture_mm * optics.resolution[1] / optics.resolution[0]
        )
        camera.CreateClippingRangeAttr(Gf.Vec2f(*optics.clipping_range))

        xform = UsdGeom.Xformable(camera)
        xform.AddTranslateOp().Set(Gf.Vec3d(*position))
        # USD cameras look along local -Z with local +Y as image up.
        aim = Gf.Rotation(Gf.Vec3d(0.0, 0.0, -1.0), Gf.Vec3d(*optical_axis))
        # Camera roll and image rotation have opposite signs.  Compose in the
        # local frame so the optical axis remains exactly collinear.
        roll = Gf.Rotation(
            Gf.Vec3d(0.0, 0.0, -1.0), -float(image_rotation_clockwise_deg)
        )
        # Gf composes rotations in row-vector order: the local roll must be on
        # the left.  This keeps -Z on optical_axis and maps camera +Y to world
        # +Z for the configured 90-degree correction.
        rotation = roll * aim
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(rotation.GetQuat())

        prim = camera.GetPrim()
        prim.CreateAttribute("simlab:sensorId", Sdf.ValueTypeNames.String).Set(sensor_id)
        prim.CreateAttribute("simlab:sensorRole", Sdf.ValueTypeNames.String).Set(sensor_id)
        prim.CreateAttribute("simlab:calibrationGroup", Sdf.ValueTypeNames.String).Set(
            calibration_group
        )
        prim.CreateAttribute("simlab:opticalAxis", Sdf.ValueTypeNames.Double3).Set(
            Gf.Vec3d(*optical_axis)
        )
        prim.CreateAttribute("simlab:resolutionWidth", Sdf.ValueTypeNames.Int).Set(
            optics.resolution[0]
        )
        prim.CreateAttribute("simlab:resolutionHeight", Sdf.ValueTypeNames.Int).Set(
            optics.resolution[1]
        )
        prim.CreateAttribute("simlab:frontSeparationM", Sdf.ValueTypeNames.Double).Set(
            front_separation
        )
        prim.CreateAttribute("simlab:imageRotationClockwiseDeg", Sdf.ValueTypeNames.Double).Set(
            image_rotation_clockwise_deg
        )
