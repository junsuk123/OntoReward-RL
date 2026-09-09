"""omni.anim.people characters: settings, spawning, rig binding, pose readback."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import carb
import omni.kit.app
import omni.kit.commands
import omni.usd
from isaacsim.core.utils import prims
from omni.anim.people.settings import PeopleSettings
from pxr import Gf, Sdf, Usd

from simlab.config.schema import PeopleConfig, PersonConfig
from simlab.sim.assets import BIPED_SETUP, asset_root, character_usd
from simlab.utils.logging import get_logger

log = get_logger("people")

XFORM_OP_ORDER_KEY = "/persistent/app/primCreation/DefaultXformOpType"
BEHAVIOR_SCRIPT_RELPATH = "/omni/anim/people/scripts/character_behavior.py"


# --------------------------------------------------------------------- setup
def configure_settings(cfg: PeopleConfig, command_file: Path) -> None:
    """Point omni.anim.people at our character root and command file."""
    settings = carb.settings.get_settings()
    settings.set(PeopleSettings.CHARACTER_PRIM_PATH, cfg.root_prim)
    settings.set(PeopleSettings.COMMAND_FILE_PATH, str(command_file))
    settings.set(PeopleSettings.NUMBER_OF_LOOP, cfg.loop)
    settings.set(PeopleSettings.NAVMESH_ENABLED, cfg.navmesh)
    settings.set(PeopleSettings.DYNAMIC_AVOIDANCE_ENABLED, cfg.dynamic_avoidance)


def write_command_file(path: Path, names: Sequence[str], agents: Sequence[PersonConfig]) -> None:
    """Emit the omni.anim.people command script that drives the walking loop."""
    lines = ["# <character> GoTo <x> <y> <z> <yaw_deg|_>   |   <character> Idle <seconds>"]
    for name, agent in zip(names, agents):
        for x, y, z in agent.waypoints:
            lines.append(f"{name} GoTo {x} {y} {z} _")
            if agent.idle_s > 0:
                lines.append(f"{name} Idle {agent.idle_s:g}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"wrote command file: {path}")


def _first_prim_of_type(root: Usd.Prim, type_name: str) -> Optional[Usd.Prim]:
    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() == type_name:
            return prim
    return None


def _behavior_script_path() -> str:
    extension_path = (
        omni.kit.app.get_app()
        .get_extension_manager()
        .get_extension_path_by_module("omni.anim.people")
    )
    return extension_path + BEHAVIOR_SCRIPT_RELPATH


# --------------------------------------------------------------------- crowd
class Crowd:
    """The set of animated characters in the stage."""

    def __init__(
        self,
        cfg: PeopleConfig,
        names: Sequence[str],
        character_prims: Sequence[Usd.Prim],
        biped_prim: Usd.Prim,
    ) -> None:
        self.cfg = cfg
        self.names = list(names)
        self.prims = list(character_prims)
        self.biped_prim = biped_prim
        self.skelroot_paths: List[Sdf.Path] = []
        self._handles: List[Any] = []

    def __len__(self) -> int:
        return len(self.prims)

    # -- construction ------------------------------------------------------
    @classmethod
    def spawn(cls, cfg: PeopleConfig, command_file: Path) -> "Crowd":
        agents = cfg.active_agents()
        configure_settings(cfg, command_file)
        names = [f"Person_{i + 1:02d}" for i in range(len(agents))]
        write_command_file(command_file, names, agents)

        biped_prim = cls._spawn_biped_setup(cfg.root_prim)
        character_prims = [
            cls._spawn_character(cfg.root_prim, name, agent)
            for name, agent in zip(names, agents)
        ]
        log(f"spawned {len(character_prims)} characters under {cfg.root_prim}")
        return cls(cfg, names, character_prims, biped_prim)

    @staticmethod
    def _spawn_biped_setup(root_prim: str) -> Usd.Prim:
        """The shared skeleton/anim-graph rig every character binds to. Hidden."""
        prims.create_prim(root_prim, "Xform")
        biped = prims.create_prim(
            f"{root_prim}/Biped_Setup", "Xform", usd_path=asset_root() + BIPED_SETUP
        )
        biped.GetAttribute("visibility").Set("invisible")
        return biped

    @staticmethod
    def _spawn_character(root_prim: str, name: str, agent: PersonConfig) -> Usd.Prim:
        # omni.anim.people steers characters through xformOp:orient, so the prim
        # has to be created with that op order rather than the default.
        settings = carb.settings.get_settings()
        previous = settings.get(XFORM_OP_ORDER_KEY)
        settings.set(XFORM_OP_ORDER_KEY, "Scale, Orient, Translate")
        try:
            prim = prims.create_prim(
                f"{root_prim}/{name}",
                "Xform",
                usd_path=asset_root() + character_usd(agent.asset),
            )
            prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*agent.start))
            quat = Gf.Rotation(Gf.Vec3d(0, 0, 1), float(agent.yaw_deg)).GetQuat()
            orient = prim.GetAttribute("xformOp:orient")
            orient.Set(Gf.Quatf(quat) if isinstance(orient.Get(), Gf.Quatf) else quat)
        finally:
            settings.set(XFORM_OP_ORDER_KEY, previous)
        return prim

    def bind(self) -> None:
        """Attach the animation graph and behavior script to each SkelRoot.

        Run this only after the stage has finished loading -- the SkelRoot prims
        live inside the referenced character USDs.
        """
        anim_graph = _first_prim_of_type(self.biped_prim, "AnimationGraph")
        if anim_graph is None:
            raise RuntimeError(f"no AnimationGraph inside {self.biped_prim.GetPrimPath()}")

        skelroots = [_first_prim_of_type(prim, "SkelRoot") for prim in self.prims]
        missing = [p.GetName() for p, s in zip(self.prims, skelroots) if s is None]
        if missing:
            raise RuntimeError(f"no SkelRoot found for character(s): {missing}")

        paths = [Sdf.Path(skelroot.GetPrimPath()) for skelroot in skelroots]
        omni.kit.commands.execute("RemoveAnimationGraphAPICommand", paths=paths)
        omni.kit.commands.execute(
            "ApplyAnimationGraphAPICommand",
            paths=paths,
            animation_graph_path=Sdf.Path(anim_graph.GetPrimPath()),
        )

        script = _behavior_script_path()
        omni.kit.commands.execute("RemoveScriptingAPICommand", paths=paths)
        omni.kit.commands.execute("ApplyScriptingAPICommand", paths=paths)
        for skelroot in skelroots:
            skelroot.GetAttribute("omni:scripting:scripts").Set([Sdf.AssetPath(script)])

        self.skelroot_paths = paths
        self._handles = [None] * len(paths)
        log(f"bound {len(paths)} characters to {anim_graph.GetPrimPath()}")

    # -- runtime -----------------------------------------------------------
    def positions(self) -> Tuple[Tuple[float, float], ...]:
        """World-frame (x, y) per character.

        Characters are moved by the animation-graph runtime, not by USD
        xformOps, so reading the prim transform would always return the spawn
        pose. Handles are cached because they only exist once the timeline is
        playing.
        """
        import omni.anim.graph.core as anim_graph

        result: List[Tuple[float, float]] = []
        for index, path in enumerate(self.skelroot_paths):
            handle = self._handles[index]
            if handle is None:
                handle = anim_graph.get_character(str(path))
                self._handles[index] = handle
            if handle is None:
                translation = omni.usd.get_world_transform_matrix(
                    self.prims[index]
                ).ExtractTranslation()
                result.append((float(translation[0]), float(translation[1])))
                continue
            position = carb.Float3(0, 0, 0)
            rotation = carb.Float4(0, 0, 0, 0)
            handle.get_world_transform(position, rotation)
            result.append((float(position[0]), float(position[1])))
        return tuple(result)
