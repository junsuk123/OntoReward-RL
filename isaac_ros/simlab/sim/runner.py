"""Wires config + assets + algorithm into a running simulation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import omni.timeline

from simlab.algorithms import DriveController, Observation, build_controller
from simlab.config.schema import SceneConfig
from simlab.scenarios.occlusion import lane_occlusion_windows
from simlab.scenarios.scene import camera_positions, scene_boxes
from simlab.scenarios.timeline import ScenarioTimeline
from simlab.sim.drones import DroneSwarm
from simlab.sim.cameras import CameraRig
from simlab.sim.telemetry import PoseReporter
from simlab.sim.world import build_world, wait_for_stage
from simlab.utils.logging import get_logger
from simlab.utils.paths import resolve_path

log = get_logger("scene")

if TYPE_CHECKING:
    from simlab.sim.people import Crowd
    from simlab.sim.ros2 import Ros2Bridge
    from simlab.sim.ugv import UGV
    from simlab.sim.swarm_ros2 import SwarmRos2Bridge


class SimulationRunner:
    """Owns the scene lifecycle: build -> reset -> step loop -> shutdown."""

    def __init__(self, app: Any, cfg: SceneConfig) -> None:
        self.app = app
        self.cfg = cfg
        self.world = None
        self.ugv: UGV | None = None
        self.crowd: Crowd | None = None
        self.controller: DriveController | None = None
        self.reporter: PoseReporter | None = None
        self.bridge: Ros2Bridge | None = None
        self.swarm: DroneSwarm | None = None
        self.camera_rig: CameraRig | None = None
        self.swarm_bridge: SwarmRos2Bridge | None = None
        self.timeline: ScenarioTimeline | None = None
        self.boxes = scene_boxes(cfg)
        #: Wall-clock seconds each camera spent dark, for the episode summary.
        self.blackout_seconds: Dict[str, float] = {}
        #: False when the ROS 2 graph owns the wheels instead of a local controller.
        self.local_control = True

    @property
    def episode_duration_s(self) -> float:
        """An active scenario sets the episode length unless the CLI overrode it."""
        if self.cfg.app.duration_s > 0:
            return self.cfg.app.duration_s
        if self.cfg.scenarios.enabled:
            return self.cfg.scenarios.episode_duration_s
        return 0.0

    @property
    def episode_dir(self) -> Path:
        scenarios = self.cfg.scenarios
        episode_id = self.timeline.episode_id if self.timeline else scenarios.episode_id
        return resolve_path(scenarios.run_dir) / (episode_id or "unnamed")

    # -- setup -------------------------------------------------------------
    def setup(self) -> None:
        cfg = self.cfg
        if cfg.scenarios.enabled:
            self.timeline = ScenarioTimeline.from_scene(cfg)
            log(
                f"episode {self.timeline.episode_id}: scenario={self.timeline.scenario} "
                f"environment={self.timeline.environment} seed={self.timeline.seed} "
                f"duration={self.episode_duration_s:g}s gaps={len(self.timeline.gaps)}"
            )
        self.world = build_world(cfg.world, cfg.app, self.boxes)
        if cfg.drones.enabled:
            self.swarm = DroneSwarm.spawn(cfg.drones, cfg.telemetry, cfg.scenarios)
            if cfg.cameras.enabled:
                self.camera_rig = CameraRig.spawn(cfg.cameras)
            wait_for_stage(self.app)
            self.swarm.finalize_assets()
            self.world.reset()
            if cfg.ros2.enabled:
                from simlab.sim.swarm_ros2 import SwarmRos2Bridge

                self.swarm_bridge = SwarmRos2Bridge(cfg.ros2)
                self.swarm_bridge.build(self.swarm.controller.states, self.camera_rig)
            try:
                from isaacsim.core.utils.viewports import set_camera_view

                set_camera_view(
                    eye=[12.0, 12.0, 10.0],
                    target=[0.0, 0.0, 2.5],
                    camera_prim_path="/OmniverseKit_Persp",
                )
            except Exception as exc:
                log(f"camera view unchanged: {exc}")
            omni.timeline.get_timeline_interface().play()
            return

        from simlab.sim.people import Crowd
        from simlab.sim.ros2 import Ros2Bridge
        from simlab.sim.ugv import UGV

        self.ugv = UGV.spawn(self.world, cfg.ugv)

        command_file: Path = resolve_path(cfg.people.command_file)
        self.crowd = Crowd.spawn(cfg.people, command_file)

        wait_for_stage(self.app)
        self.crowd.bind()
        for _ in range(5):
            self.app.update()

        self.world.reset()
        self.ugv.initialize()

        if cfg.ros2.enabled:
            self.bridge = Ros2Bridge(cfg.ros2, len(self.crowd))
            self.bridge.build(
                robot_prim_path=self.ugv.prim_path,
                articulation_prim_path=self.ugv.articulation_root_path,
                chassis_prim_path=self.ugv.chassis_prim_path,
                wheel_joint_names=self.ugv.wheel_joint_names,
                spec=self.ugv.spec,
                app_cfg=cfg.app,
            )

        self.local_control = not (cfg.ros2.enabled and cfg.ros2.drive_from_cmd_vel)
        if self.local_control:
            self.controller = build_controller(
                cfg.ugv.controller.name, cfg.ugv.controller.params
            )
            self.controller.reset()
        self.reporter = PoseReporter(cfg.telemetry, self.crowd.names)

        # omni.anim.people behavior scripts only tick while the timeline plays.
        omni.timeline.get_timeline_interface().play()

    # -- run ---------------------------------------------------------------
    def observe(self, t: float) -> Observation:
        x, y, yaw = self.ugv.pose()
        return Observation(
            t=t, robot_xy=(x, y), robot_yaw=yaw, people_xy=self.crowd.positions()
        )

    def gate_sensors(self, t: float, step: int, dt: float) -> None:
        """Silence any camera the scenario says is dark at this instant."""
        if self.timeline is None or self.swarm_bridge is None:
            return
        observing = {}
        for sensor_id in ("front_near", "front_far", "satellite_nadir"):
            visible = self.timeline.observing(sensor_id, t, step)
            observing[sensor_id] = visible
            if not visible:
                self.blackout_seconds[sensor_id] = self.blackout_seconds.get(sensor_id, 0.0) + dt
        self.swarm_bridge.apply_sensor_gates(observing)

    def run(self) -> int:
        """Step until the duration elapses or the app closes. Returns steps taken."""
        cfg = self.cfg
        dt = cfg.app.physics_dt
        duration = self.episode_duration_s
        max_steps = round(duration / dt) if duration > 0 else -1
        limit = "Ctrl+C to stop." if max_steps < 0 else f"{duration:g}s."
        if self.swarm is not None:
            scenario = self.timeline.scenario if self.timeline else "crossing shuttle"
            log(
                f"running: {cfg.drones.friendly.count} allies + "
                f"{cfg.drones.enemy.count} enemies; scenario={scenario}. {limit}"
            )
            if self.timeline is not None and self.timeline.gaps and self.swarm_bridge is None:
                log("warning: ros2 bridge is off, so scheduled camera blackouts cannot be applied")
            elif self.timeline is not None and self.timeline.gaps and self.swarm_bridge is not None:
                if not self.swarm_bridge.camera_gating_available:
                    log("warning: this Kit build exposes no camera enable input; blackouts are logged only")
        else:
            driver = self.controller.name if self.controller else f"ros2:{cfg.ros2.cmd_vel_topic}"
            log(f"running: 1x {cfg.ugv.model} [{driver}] + {len(self.crowd)} people. {limit}")

        step = 0
        try:
            while self.app.is_running() and (max_steps < 0 or step < max_steps):
                if self.swarm is not None:
                    t = step * dt
                    self.swarm.update(t, dt)
                    if self.swarm_bridge is not None:
                        self.swarm_bridge.update(self.swarm.controller.states)
                    self.gate_sensors(t, step, dt)
                    self.world.step(render=True)
                    step += 1
                    continue
                obs = self.observe(step * dt)
                self.reporter.maybe_report(obs)
                if self.bridge is not None:
                    self.bridge.update(obs.people_xy)
                if self.controller is not None:
                    self.ugv.apply(self.controller.step(obs))
                self.world.step(render=True)
                step += 1
        except KeyboardInterrupt:
            log("interrupted")
        self.write_episode_summary(step, step * dt)
        log(f"done after {step} steps")
        return step

    # -- episode record ----------------------------------------------------
    def write_episode_summary(self, steps: int, elapsed_s: float) -> None:
        """Record what this episode actually flew, next to its collected frames.

        The occlusion windows are computed from the same geometry the collector
        labels against, so a structural-occlusion episode that failed to hide
        anyone is visible in the summary instead of silently producing a dataset
        with no occlusions in it.
        """
        if self.timeline is None or self.swarm is None:
            return
        cameras = camera_positions(self.cfg)
        lanes = []
        for state in self.swarm.controller.states:
            plan = state.plan
            windows = {}
            for sensor_id in ("front_near", "front_far"):
                found = lane_occlusion_windows(
                    cameras[sensor_id], self.boxes, plan.lane_x, state.cruise_altitude
                )
                windows[sensor_id] = [
                    {"blocker": name, "y_start": round(start, 2), "y_end": round(end, 2)}
                    for name, start, end in found
                    # Only report shadows the aircraft actually flies through.
                    if end >= -plan.half_length and start <= plan.half_length
                ]
            lanes.append({"drone_id": state.name, **plan.as_dict(), "occlusion_windows": windows})

        summary = {
            **self.timeline.describe(),
            "steps": steps,
            "elapsed_s": round(elapsed_s, 3),
            "minimum_separation_m": round(self.swarm.controller.minimum_separation, 3),
            "safety_radius_m": self.cfg.drones.safety_radius,
            "trajectory_seed": self.swarm.controller.seed,
            "blackout_seconds": {k: round(v, 3) for k, v in self.blackout_seconds.items()},
            "camera_gating": bool(
                self.swarm_bridge is not None and self.swarm_bridge.camera_gating_available
            ),
            "lanes": lanes,
        }
        try:
            directory = self.episode_dir
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "episode_summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            log(f"episode summary -> {directory / 'episode_summary.json'}")
        except OSError as exc:
            log(f"episode summary not written: {exc}")
