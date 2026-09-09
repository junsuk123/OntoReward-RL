"""Stage scaffolding: physics world, environment scenery, lighting."""

from __future__ import annotations

from typing import Sequence

import omni.usd
from isaacsim.core.api import World
from pxr import Gf, Sdf, UsdGeom, UsdLux

from simlab.config.schema import AppConfig, WorldConfig
from simlab.scenarios.environments import environment
from simlab.scenarios.occlusion import Box
from simlab.sim.assets import asset_root
from simlab.utils.logging import get_logger

log = get_logger("world")

DOME_LIGHT_PATH = "/World/DomeLight"


def _cube(stage, path: str, box: Box) -> None:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(2.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*box.color)])
    transform = UsdGeom.Xformable(cube)
    transform.AddTranslateOp().Set(Gf.Vec3d(*box.center))
    transform.AddScaleOp().Set(Gf.Vec3f(*(value * 0.5 for value in box.size)))


def _build_scenery(stage, root_path: str, boxes: Sequence[Box]) -> None:
    """Create one cube per box. The list is shared with the occlusion truth."""
    UsdGeom.Xform.Define(stage, root_path)
    for box in boxes:
        _cube(stage, f"{root_path}/{box.name}", box)
    occluders = sum(1 for box in boxes if box.kind == "occluder")
    buildings = sum(1 for box in boxes if box.kind == "building")
    log(
        f"environment scenery: {len(boxes)} prims "
        f"({buildings} building parts, {occluders} sight-line occluders)"
    )


def build_world(
    world_cfg: WorldConfig, app_cfg: AppConfig, boxes: Sequence[Box] = ()
) -> World:
    """Create the physics world and the scenery of the configured environment."""
    world = World(
        stage_units_in_meters=world_cfg.stage_units_in_meters,
        physics_dt=app_cfg.physics_dt,
        rendering_dt=app_cfg.rendering_dt,
    )
    if world_cfg.ground_plane:
        world.scene.add_default_ground_plane()

    stage = omni.usd.get_context().get_stage()
    if boxes:
        _build_scenery(stage, world_cfg.environment_prim_path, boxes)
    if world_cfg.environment_usd:
        environment_url = world_cfg.environment_usd
        if environment_url.startswith("/"):
            environment_url = asset_root() + environment_url
        external = stage.DefinePrim(f"{world_cfg.environment_prim_path}/ExternalUsd", "Xform")
        external.GetReferences().AddReference(environment_url)
        log(f"environment: {world_cfg.environment_usd}")

    preset = environment(world_cfg.environment)
    intensity = world_cfg.dome_light_intensity * preset.light_scale
    dome = UsdLux.DomeLight.Define(stage, Sdf.Path(DOME_LIGHT_PATH))
    dome.CreateIntensityAttr(intensity)
    dome.CreateColorAttr(Gf.Vec3f(*preset.light_color))

    log(
        f"world ready (environment={preset.key}: {preset.description}; "
        f"ground={world_cfg.ground_plane}, dome={intensity:g}, dt={app_cfg.physics_dt:.5f}s)"
    )
    return world


def wait_for_stage(app) -> None:
    """Block until every referenced asset has finished loading."""
    from isaacsim.core.utils.stage import is_stage_loading

    log("loading stage...")
    while is_stage_loading():
        app.update()
