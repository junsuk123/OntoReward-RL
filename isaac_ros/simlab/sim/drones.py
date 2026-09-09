"""Official Isaac Sim aerial robot assets driven by the swarm controller."""

from __future__ import annotations

import math
from typing import List

import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom

from simlab.algorithms.swarm import DroneState, SwarmController
from simlab.config.schema import DronesConfig, ScenariosConfig, TelemetryConfig
from simlab.sim.assets import asset_root, drone_asset_url, drone_spec
from simlab.utils.logging import get_logger

log = get_logger("drones")


class DroneSwarm:
    """Owns referenced robot prims and advances the configured kinematic aircraft."""

    def __init__(
        self,
        cfg: DronesConfig,
        telemetry: TelemetryConfig,
        scenarios: ScenariosConfig | None = None,
    ) -> None:
        self.cfg = cfg
        self.controller = SwarmController(cfg, scenarios)
        self.telemetry_interval = telemetry.report_every_s
        self._next_report = 0.0
        self._translate_ops: List = []
        self._rotate_ops: List = []
        self._asset_prim_paths: List[str] = []

    def __len__(self) -> int:
        return len(self.controller.states)

    @classmethod
    def spawn(
        cls,
        cfg: DronesConfig,
        telemetry: TelemetryConfig,
        scenarios: ScenariosConfig | None = None,
    ) -> "DroneSwarm":
        swarm = cls(cfg, telemetry, scenarios)
        stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage, cfg.root_prim)

        root_url = asset_root()

        for state in swarm.controller.states:
            path = f"{cfg.root_prim}/{state.name}"
            root = UsdGeom.Xform.Define(stage, path)
            root_xform = UsdGeom.Xformable(root)
            translate_op = root_xform.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(*state.position))
            swarm._translate_ops.append(translate_op)
            swarm._rotate_ops.append(root_xform.AddRotateXYZOp())
            team = cfg.friendly if state.team == "friendly" else cfg.enemy
            spec = drone_spec(state.model)
            visual = UsdGeom.Xform.Define(stage, f"{path}/Visual")
            UsdGeom.Xformable(visual).AddScaleOp().Set(
                Gf.Vec3f(team.asset_scale, team.asset_scale, team.asset_scale)
            )
            asset_path = f"{path}/Visual/Asset"
            asset_prim = UsdGeom.Xform.Define(stage, asset_path).GetPrim()
            asset_url = drone_asset_url(spec, root_url)
            asset_prim.GetReferences().AddReference(asset_url)
            swarm._asset_prim_paths.append(asset_path)
            swarm._add_identity_metadata(
                root.GetPrim(), state, spec.manufacturer, spec.platform, asset_url, team.color
            )

        log(
            f"spawned {cfg.friendly.count} friendly {cfg.friendly.model} + "
            f"{cfg.enemy.count} enemy {cfg.enemy.model} quadrotors; "
            f"trajectory_seed={swarm.controller.seed}"
        )
        return swarm

    @staticmethod
    def _add_identity_metadata(
        prim, state, manufacturer: str, platform: str, asset_url: str, team_color
    ) -> None:
        """Attach stable labels used by segmentation, tracking, and an ontology."""
        fields = {
            "simlab:entityType": "quadrotor",
            "simlab:trackId": state.name,
            "simlab:team": state.team,
            "simlab:model": state.model,
            "simlab:manufacturer": manufacturer,
            "simlab:platform": platform,
            "simlab:assetUri": asset_url,
            "simlab:ontologyClassUri": "urn:simlab:ontology#Quadrotor",
            "simlab:formation": state.plan.formation if state.plan else "",
        }
        for name, value in fields.items():
            prim.CreateAttribute(name, Sdf.ValueTypeNames.String).Set(value)
        prim.CreateAttribute("simlab:teamColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*team_color)
        )
        numeric_id = int(state.name.rsplit("_", 1)[-1])
        if state.team == "enemy":
            numeric_id += 1000
        prim.CreateAttribute("simlab:instanceId", Sdf.ValueTypeNames.Int).Set(numeric_id)
        try:
            from isaacsim.core.utils.semantics import add_labels

            add_labels(prim, labels=["drone"], instance_name="class")
            add_labels(prim, labels=[state.team], instance_name="team")
            add_labels(prim, labels=[state.model], instance_name="model")
            add_labels(prim, labels=[state.name], instance_name="track_id")
        except Exception as exc:
            log(f"semantic labels unavailable for {state.name}: {exc}")

    def finalize_assets(self) -> None:
        """Disable referenced rigid-body dynamics; the swarm supplies flight poses.

        The official USD meshes and articulations remain intact for perception,
        while gravity cannot fight the scripted takeoff/shuttle controller.
        """
        stage = omni.usd.get_context().get_stage()
        disabled = 0
        for path in self._asset_prim_paths:
            root = stage.GetPrimAtPath(path)
            for prim in iter(Usd.PrimRange(root)):
                for attribute_name in (
                    "physics:rigidBodyEnabled",
                    "physics:collisionEnabled",
                    "physics:jointEnabled",
                    "physxArticulation:articulationEnabled",
                ):
                    attribute = prim.GetAttribute(attribute_name)
                    if attribute:
                        attribute.Set(False)
                        disabled += 1
        log(f"loaded official aerial assets; disabled {disabled} physics attributes for kinematic flight")

    def update(self, t: float, dt: float) -> None:
        states = self.controller.step(t, dt)
        for index, state in enumerate(states):
            self._translate_ops[index].Set(Gf.Vec3d(*state.position))
            speed = math.hypot(state.velocity[0], state.velocity[1])
            pitch = max(-10.0, min(10.0, -state.velocity[2] * 3.0))
            roll = max(-12.0, min(12.0, speed * 2.5))
            self._rotate_ops[index].Set(
                Gf.Vec3f(pitch, roll, math.degrees(state.yaw))
            )
        self._report(t, states)

    def _report(self, t: float, states: List[DroneState]) -> None:
        if self.telemetry_interval <= 0 or t < self._next_report:
            return
        self._next_report = (int(t / self.telemetry_interval) + 1) * self.telemetry_interval
        nearest = math.inf
        for index, state in enumerate(states):
            for other in states[index + 1 :]:
                distance = math.dist(state.position, other.position)
                nearest = min(nearest, distance)
        ally = states[0].position if states else (0.0, 0.0, 0.0)
        phase = "takeoff" if t < self.cfg.takeoff_duration_s else "crossing shuttle"
        log(
            f"t={t:5.1f}s phase={phase} drones={len(states)} "
            f"nearest={nearest:.2f}m Ally_01=({ally[0]:.1f},{ally[1]:.1f},{ally[2]:.1f})"
        )
