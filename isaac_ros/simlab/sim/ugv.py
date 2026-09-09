"""Differential-drive UGV: spawn, wheel resolution, command application."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np
from isaacsim.core.api import World
import omni.usd
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils import prims
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.wheeled_robots.controllers.differential_controller import (
    DifferentialController,
)
from pxr import Usd, UsdPhysics

from simlab.algorithms.base import DriveCommand
from simlab.config.schema import UGVConfig
from simlab.sim.assets import RobotSpec, asset_root, robot_spec
from simlab.utils.logging import get_logger

log = get_logger("ugv")


def _yaw_from_quat(quat: Sequence[float]) -> float:
    """Z rotation from a scalar-first (w, x, y, z) quaternion."""
    w, x, y, z = (float(v) for v in quat)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class UGV:
    """Wraps an articulated wheeled base behind a velocity-command interface."""

    def __init__(self, robot: Robot, spec: RobotSpec, prim_path: str) -> None:
        self.robot = robot
        self.spec = spec
        self.prim_path = prim_path
        self._wheel_indices: List[int] = []
        self._controller: DifferentialController | None = None
        self._articulation = None
        self._articulation_root_path: str | None = None
        self._chassis_prim_path: str | None = None

    # -- construction ------------------------------------------------------
    @classmethod
    def spawn(cls, world: World, cfg: UGVConfig) -> "UGV":
        """Reference the robot USD into the stage and register it with the world."""
        spec = robot_spec(cfg.model)
        position = cfg.spawn if cfg.spawn is not None else (0.0, 0.0, spec.spawn_height)

        parent = cfg.prim_path.rsplit("/", 1)[0]
        if parent and parent != "/World":
            prims.create_prim(parent, "Xform")
        add_reference_to_stage(usd_path=asset_root() + spec.usd, prim_path=cfg.prim_path)

        robot = world.scene.add(
            Robot(prim_path=cfg.prim_path, name=cfg.name, position=np.array(position))
        )
        log(f"spawned {cfg.model} at {tuple(round(v, 3) for v in position)} -> {cfg.prim_path}")
        return cls(robot, spec, cfg.prim_path)

    def initialize(self) -> None:
        """Resolve wheel DOFs and build the drive controller.

        Must run *after* ``world.reset()`` -- DOF names only exist once the
        articulation is initialized.
        """
        self._wheel_indices = self._resolve_wheel_indices()
        self._scan_physics_prims()
        self._controller = DifferentialController(
            name=f"{self.robot.name}_diff",
            wheel_radius=self.spec.wheel_radius,
            wheel_base=self.spec.wheel_base,
        )
        self._articulation = self.robot.get_articulation_controller()

    def _resolve_wheel_indices(self) -> List[int]:
        """Map (left, right) drive wheels to DOF indices, by name then by pattern."""
        dof_names = list(self.robot.dof_names)
        log(f"dof names: {dof_names}")
        left_hint, right_hint = self.spec.wheel_hints
        try:
            return [dof_names.index(left_hint), dof_names.index(right_hint)]
        except ValueError:
            pass

        def match(side: str) -> List[int]:
            return [
                i
                for i, name in enumerate(dof_names)
                if "wheel" in name.lower() and side in name.lower()
            ]

        left, right = match("left"), match("right")
        if not left or not right:
            raise RuntimeError(
                f"could not identify drive wheels for {self.robot.name} among {dof_names}"
            )
        log(f"hints {self.spec.wheel_hints} not found; using "
            f"{dof_names[left[0]]}/{dof_names[right[0]]}")
        return [left[0], right[0]]

    # -- physics prims -----------------------------------------------------
    def _scan_physics_prims(self) -> None:
        """Find the articulation root and the chassis rigid body under the robot.

        OmniGraph's odometry node needs an actual rigid body: pointing it at the
        wrapper Xform fails with "not a valid rigid body or articulation root".
        Nova Carter puts the articulation root on ``chassis_link`` (itself a
        rigid body); Jetbot puts it on the root Xform, so the chassis has to be
        picked up as the first rigid body instead.
        """
        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath(self.prim_path)
        articulation_roots, rigid_bodies = [], []
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                articulation_roots.append(prim)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_bodies.append(prim)

        self._articulation_root_path = (
            str(articulation_roots[0].GetPath()) if articulation_roots else self.prim_path
        )
        chassis = next(
            (p for p in articulation_roots if p.HasAPI(UsdPhysics.RigidBodyAPI)), None
        ) or (rigid_bodies[0] if rigid_bodies else None)
        if chassis is None:
            raise RuntimeError(f"no rigid body found under {self.prim_path}")
        self._chassis_prim_path = str(chassis.GetPath())
        log(f"articulation={self._articulation_root_path} chassis={self._chassis_prim_path}")

    @property
    def articulation_root_path(self) -> str:
        if self._articulation_root_path is None:
            raise RuntimeError("UGV.initialize() must be called first")
        return self._articulation_root_path

    @property
    def chassis_prim_path(self) -> str:
        if self._chassis_prim_path is None:
            raise RuntimeError("UGV.initialize() must be called first")
        return self._chassis_prim_path

    @property
    def wheel_joint_names(self) -> List[str]:
        """The (left, right) drive joint names, resolved by :meth:`initialize`."""
        if not self._wheel_indices:
            raise RuntimeError("UGV.initialize() must be called first")
        names = list(self.robot.dof_names)
        return [names[i] for i in self._wheel_indices]

    # -- runtime -----------------------------------------------------------
    def apply(self, command: DriveCommand) -> None:
        """Convert a body-frame velocity into wheel velocity targets."""
        if self._controller is None or self._articulation is None:
            raise RuntimeError("UGV.initialize() must be called after world.reset()")

        action = self._controller.forward([command.linear, command.angular])
        # NaN leaves a joint untouched, so casters and swings keep free-spinning.
        velocities = np.full(self.robot.num_dof, np.nan)
        velocities[self._wheel_indices[0]] = action.joint_velocities[0]
        velocities[self._wheel_indices[1]] = action.joint_velocities[1]
        action.joint_velocities = velocities
        action.joint_positions = None
        action.joint_efforts = None
        self._articulation.apply_action(action)

    def pose(self) -> Tuple[float, float, float]:
        """World-frame (x, y, yaw[rad])."""
        position, orientation = self.robot.get_world_pose()
        return float(position[0]), float(position[1]), _yaw_from_quat(orientation)
