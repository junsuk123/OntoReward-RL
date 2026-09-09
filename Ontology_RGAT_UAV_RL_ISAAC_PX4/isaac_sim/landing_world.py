#!/usr/bin/env python3
"""Isaac Sim 5.1/Pegasus 5.1 landing world with PX4 Simulator MAVLink.

Launch this file with Pegasus' ``isaac_run`` helper, not system Python.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


ARGS = parse_args()
CONFIG_PATH = Path(ARGS.config).expanduser().resolve()
WORKSPACE = CONFIG_PATH.parent.parent
with CONFIG_PATH.open("r", encoding="utf-8") as stream:
    CONFIG = yaml.safe_load(stream)

# SimulationApp must be constructed before importing Omniverse/Pegasus modules.
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": ARGS.headless})

import carb
import omni.timeline
from isaacsim.core.utils.extensions import enable_extension

# Isaac Sim ships a Python-3.11-compatible rclpy inside the ROS 2 bridge.
# The extension must be enabled and allowed one update before ROS imports.
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

import rclpy
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from std_msgs.msg import Float32, String
from scipy.spatial.transform import Rotation
from isaacsim.core.api import World

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.backends.px4_mavlink_backend import (
    PX4MavlinkBackend,
    PX4MavlinkBackendConfig,
)
from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend
from pegasus.simulator.logic.dynamics import LinearDrag
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from isaacsim.core.api.materials import OmniPBR
from isaacsim.sensors.camera import Camera
from pxr import Gf, Sdf, UsdGeom, UsdShade

sys.path.insert(0, str(Path(__file__).resolve().parent))
from marker_vision import (
    R_BODY_FROM_OPTICAL,
    MarkerBoard,
    MarkerPoseEstimator,
    generate_marker_png,
    intrinsics_from_fov,
    texture_side_ratio,
)


IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


def _quat_wxyz_to_matrix(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _matrix_to_quat_wxyz(m):
    quat = Rotation.from_matrix(m).as_quat()      # scipy returns x, y, z, w
    return np.array([quat[3], quat[0], quat[1], quat[2]])


class LandingPadMarkers:
    """The ArUco tags painted on the pad, as textured quads in the stage."""

    def __init__(self, config: dict, workspace: Path):
        self.board = MarkerBoard.from_config(config["board"])
        self.dictionary = str(config["dictionary"])
        self.texture_dir = workspace / "assets" / "markers"
        self.ratio = texture_side_ratio(self.dictionary)

    def spawn(self, world) -> None:
        stage = world.stage
        UsdGeom.Xform.Define(stage, "/World/landing_pad")
        for marker in self.board.markers.values():
            texture = generate_marker_png(
                self.texture_dir / f"{self.dictionary}_{marker.marker_id}.png",
                self.dictionary, marker.marker_id)
            material = OmniPBR(
                prim_path=f"/World/landing_pad/material_{marker.marker_id}",
                name=f"landing_marker_{marker.marker_id}",
                texture_path=str(texture),
                texture_scale=np.array([1.0, 1.0]),
                texture_translate=np.array([0.0, 0.0]))
            # OmniPBR turns on world-space UV projection in its constructor,
            # which ignores the quad's own UVs and crops the marker's black
            # border and quiet zone away. Without both, it is not a tag.
            material.set_project_uvw(False)
            path = f"/World/landing_pad/marker_{marker.marker_id}"
            self._quad(stage, path, marker.center_xy_m, marker.side_m * self.ratio)
            UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).Bind(
                UsdShade.Material(stage.GetPrimAtPath(material.prim_path)))

    @staticmethod
    def _quad(stage, path: str, center_xy, side: float):
        """A flat quad whose UVs map the padded texture exactly once.

        A cube would be less code, but its faces do not carry [0,1] UVs, so the
        white quiet zone gets cropped away and the tag stops being detectable.
        Laying out the corners explicitly also pins the marker's orientation:
        the texture's top row must face pad north, which is what the pose
        estimator's object points assume.
        """
        mesh = UsdGeom.Mesh.Define(stage, path)
        half = side / 2.0
        corners = [(-half, -half), (half, -half), (half, half), (-half, half)]
        mesh.CreatePointsAttr([Gf.Vec3f(x, y, 0.0) for x, y in corners])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateNormalsAttr([Gf.Vec3f(0.0, 0.0, 1.0)] * 4)
        mesh.CreateExtentAttr([Gf.Vec3f(-half, -half, 0.0), Gf.Vec3f(half, half, 0.0)])
        # st (0,0) is the texture's bottom-left, so pad south-west; pad north
        # therefore lands on the texture's top row, i.e. the marker's own "up".
        primvars = UsdGeom.PrimvarsAPI(mesh)
        st = primvars.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                    UsdGeom.Tokens.varying)
        st.Set([Gf.Vec2f(0.0, 0.0), Gf.Vec2f(1.0, 0.0),
                Gf.Vec2f(1.0, 1.0), Gf.Vec2f(0.0, 1.0)])
        # Lifted a hair so the pad does not z-fight with the ground.
        UsdGeom.XformCommonAPI(mesh).SetTranslate(
            Gf.Vec3d(float(center_xy[0]), float(center_xy[1]), 0.002))
        return mesh


class DownwardCamera:
    """A downward camera on the vehicle plus the pad-relative pose it yields."""

    def __init__(self, config: dict, board: MarkerBoard, dictionary: str):
        camera_cfg = config["camera"]
        self.width, self.height = (int(v) for v in camera_cfg["resolution"])
        self.fov_deg = float(camera_cfg["horizontal_fov_deg"])
        self.rate_hz = float(camera_cfg["rate_hz"])
        self.mount = np.array([float(v) for v in camera_cfg["mount_translation_flu_m"]])
        self.clipping = tuple(float(v) for v in camera_cfg.get("clipping_range_m", (0.02, 60.0)))
        self.camera_matrix = intrinsics_from_fov(self.width, self.height, self.fov_deg)
        self.estimator = MarkerPoseEstimator(
            board, self.camera_matrix, self.mount, dictionary=dictionary,
            quality_reprojection_px=float(config.get("quality_reprojection_px", 3.0)),
            quality_full_scale_px=float(config.get("quality_full_scale_px", 120.0)))
        self.camera = None
        # Set ONTOLOGY_RGAT_VISION_DEBUG_DIR to dump annotated frames; the only
        # way to tell "no image" from "no marker in it" is to look at one.
        self.debug_dir = os.environ.get("ONTOLOGY_RGAT_VISION_DEBUG_DIR", "")
        self.debug_every = int(os.environ.get("ONTOLOGY_RGAT_VISION_DEBUG_EVERY", "30"))
        self.frames = 0

    def attach(self, vehicle_prim_path: str) -> None:
        self.camera = Camera(
            prim_path=vehicle_prim_path + "/body/landing_camera",
            resolution=(self.width, self.height),
            frequency=self.rate_hz)
        self.camera.set_local_pose(translation=self.mount,
                                   orientation=IDENTITY_QUAT, camera_axes="ros")

    def aim_at_nadir(self, vehicle) -> None:
        """Point the optical axis straight down, whatever axis convention applies.

        Isaac's camera_axes conventions are not worth guessing at: an identity
        orientation here comes out looking sideways. So measure what identity
        actually produces, correct by exactly that, and then measure the mount
        again so the pose estimator solves in the frame the camera really has.
        """
        self.camera.set_local_pose(translation=self.mount,
                                   orientation=IDENTITY_QUAT, camera_axes="ros")
        identity_mount, _ = self._measure_mount(vehicle)
        # Isaac composes the requested rotation with a fixed convention, so the
        # rotation that lands on the frame we want is the residual.
        correction = R_BODY_FROM_OPTICAL @ identity_mount.T
        self.camera.set_local_pose(
            translation=self.mount,
            orientation=_matrix_to_quat_wxyz(correction),
            camera_axes="ros")

        body_from_optical, mount = self._measure_mount(vehicle)
        view_in_body = body_from_optical[:, 2]
        if view_in_body[2] > -0.99:
            carb.log_error(
                f"Landing camera is not looking down: optical +Z is {view_in_body} "
                "in body axes. Marker detection will be unusable.")
        self.estimator.body_from_optical = body_from_optical
        self.estimator.mount_translation_body = mount
        print(f"[landing-camera] optical axes in body:\n{body_from_optical}\n"
              f"[landing-camera] mount in body: {mount}", file=sys.stderr, flush=True)

    def _measure_mount(self, vehicle):
        """Camera pose relative to the vehicle body, read back from the stage.

        The convention has to be named on the way out as well as on the way in:
        get_world_pose defaults to Isaac's "world" axes, so reading it back
        after setting "ros" axes compares two different frames and quietly
        reports a camera that is not the one being rendered.
        """
        camera_position, camera_quat = self.camera.get_world_pose(camera_axes="ros")
        r_world_from_optical = _quat_wxyz_to_matrix(np.asarray(camera_quat, dtype=float))
        state = vehicle.state
        r_world_from_body = Rotation.from_quat(state.attitude).as_matrix()
        body_from_optical = r_world_from_body.T @ r_world_from_optical
        mount = r_world_from_body.T @ (np.asarray(camera_position, dtype=float)
                                       - np.asarray(state.position, dtype=float))
        return body_from_optical, mount

    def start(self) -> None:
        self.camera.initialize()
        self.camera.set_clipping_range(*self.clipping)
        # Isaac takes a focal length and an aperture, not a field of view.
        # Derive the aperture from whatever focal length the camera already
        # has: both are in the same units, so their ratio cannot be thrown off
        # by the stage's unit scale the way an absolute focal length can.
        focal = float(self.camera.get_focal_length())
        aperture = 2.0 * focal * math.tan(math.radians(self.fov_deg) / 2.0)
        self.camera.set_horizontal_aperture(aperture)
        self.camera.set_vertical_aperture(aperture * self.height / self.width)
        # Then solve PnP against the renderer's own intrinsics rather than a
        # formula, so a lens setting that did not take cannot silently bias
        # every pose the policy flies on.
        matrix = self.camera.get_intrinsics_matrix()
        if matrix is not None and np.isfinite(matrix).all():
            self.camera_matrix = np.asarray(matrix, dtype=float)
            self.estimator.camera_matrix = self.camera_matrix
        fov = 2.0 * math.degrees(math.atan(
            self.width / (2.0 * float(self.camera_matrix[0, 0]))))
        print(f"[landing-camera] {self.width}x{self.height} fov={fov:.1f}deg "
              f"(asked {self.fov_deg:.1f}) fx={self.camera_matrix[0, 0]:.1f} "
              f"focal={self.camera.get_focal_length():.4f} "
              f"aperture={self.camera.get_horizontal_aperture():.4f} "
              f"world_pose={self.camera.get_world_pose()}", file=sys.stderr, flush=True)

    def observe(self):
        frame = self.camera.get_rgba()
        if frame is None or frame.size == 0:
            if self.frames == 0:
                carb.log_warn("Landing camera produced no frame; is rendering enabled?")
            self.frames += 1
            return None
        image = frame[:, :, :3]
        observation = self.estimator.detect(image)
        self.frames += 1
        if self.debug_dir and self.frames % max(1, self.debug_every) == 0:
            self._dump(image, observation)
        return observation

    def _dump(self, image, observation) -> None:
        import cv2
        path = Path(self.debug_dir)
        path.mkdir(parents=True, exist_ok=True)
        canvas = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2BGR)
        label = ("ids=%s q=%.2f" % (observation.marker_ids, observation.quality)
                 if observation.detected else "no detection")
        cv2.putText(canvas, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 2)
        cv2.imwrite(str(path / f"frame_{self.frames:06d}.png"), canvas)


class WindField:
    def __init__(self, config: dict):
        self.config = config
        self.rng = np.random.default_rng(int(config.get("seed", 49)))
        self.phases = self.rng.uniform(0.0, 2.0 * np.pi, size=(3, 6))
        self.freq = np.linspace(0.23, 1.91, 6)
        self.episode_t0 = 0.0
        self.scale = 1.0

    def reset(self, seed: int, sim_time: float, scale: float = 1.0) -> None:
        self.rng = np.random.default_rng(int(seed) + int(self.config.get("seed", 49)))
        self.phases = self.rng.uniform(0.0, 2.0 * np.pi, size=(3, 6))
        self.episode_t0 = sim_time
        self.scale = float(scale)

    def sample(self, sim_time: float) -> np.ndarray:
        if not self.config.get("enabled", True):
            return np.zeros(3)
        t = sim_time - self.episode_t0
        wind = np.asarray(self.config["mean_enu_m_s"], dtype=float).copy()
        amp = float(self.config.get("turbulence_m_s", 0.0)) / math.sqrt(3.0)
        for axis in range(3):
            wind[axis] += amp * float(np.mean(np.sin(self.freq * t + self.phases[axis])))
        for gust in self.config.get("gusts", []):
            tau = (t - float(gust["t0_s"])) / max(float(gust["sigma_s"]), 1e-3)
            wind += np.asarray(gust["vector_enu_m_s"], dtype=float) * math.exp(-0.5 * tau * tau)
        return self.scale * wind

    def force(self, sim_time: float, vehicle_velocity_enu: np.ndarray,
              attitude_enu_flu_xyzw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        wind = self.sample(sim_time)
        rotation = Rotation.from_quat(attitude_enu_flu_xyzw)
        relative_enu = wind - np.asarray(vehicle_velocity_enu, dtype=float)
        relative_body = rotation.inv().apply(relative_enu)
        rho = float(self.config["air_density_kg_m3"])
        area = np.asarray(self.config["drag_area_m2"], dtype=float)
        cd = np.asarray(self.config["drag_coefficient"], dtype=float)
        force_body = 0.5 * rho * area * cd * relative_body * np.abs(relative_body)
        force_enu = rotation.apply(force_body)
        return wind, force_enu, force_body


class LandingWorld:
    def __init__(self):
        isaac_cfg = CONFIG["isaac"]
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg.set_world_settings(
            physics_dt=float(isaac_cfg["physics_dt"]),
            rendering_dt=float(isaac_cfg["rendering_dt"]),
        )
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        environment = isaac_cfg["environment"]
        if environment not in SIMULATION_ENVIRONMENTS:
            raise KeyError(f"unknown Pegasus environment: {environment}")
        self.pg.load_environment(SIMULATION_ENVIRONMENTS[environment])

        px4_dir = Path(isaac_cfg["px4_dir"])
        if not px4_dir.is_absolute():
            px4_dir = (WORKSPACE / px4_dir).resolve()
        mavlink_cfg = PX4MavlinkBackendConfig({
            "vehicle_id": int(isaac_cfg["vehicle_id"]),
            "connection_type": isaac_cfg["mavlink_connection_type"],
            "connection_ip": isaac_cfg["mavlink_connection_ip"],
            "connection_baseport": int(isaac_cfg["mavlink_connection_baseport"]),
            "enable_lockstep": bool(isaac_cfg["lockstep"]),
            "update_rate": 1.0 / float(isaac_cfg["physics_dt"]),
            "px4_autolaunch": bool(isaac_cfg["px4_autolaunch"]),
            "px4_dir": str(px4_dir),
            "px4_vehicle_model": isaac_cfg["px4_vehicle_model"],
        })
        self.px4_backend = PX4MavlinkBackend(mavlink_cfg)
        self.ros_backend = ROS2Backend(
            vehicle_id=int(isaac_cfg["vehicle_id"]),
            config={
                "namespace": isaac_cfg["namespace"],
                "pub_graphical_sensors": False,
                "pub_sensors": False,
                "pub_state": True,
                "pub_tf": True,
                "sub_control": False,
            },
        )
        vehicle_cfg = MultirotorConfig()
        # Replace Pegasus' still-air linear drag with the wind-relative model below.
        vehicle_cfg.drag = LinearDrag([0.0, 0.0, 0.0])
        vehicle_cfg.backends = [self.px4_backend, self.ros_backend]
        spawn = [float(x) for x in isaac_cfg["spawn_position_enu_m"]]
        self.vehicle = Multirotor(
            "/World/quadrotor",
            ROBOTS[isaac_cfg["robot_asset"]],
            int(isaac_cfg["vehicle_id"]),
            spawn,
            [0.0, 0.0, 0.0, 1.0],
            config=vehicle_cfg,
        )

        vision_cfg = CONFIG["vision"]
        self.vision_enabled = str(vision_cfg.get("mode", "pose_proxy")) == "aruco"
        self.pad = None
        self.camera = None
        if self.vision_enabled:
            self.pad = LandingPadMarkers(vision_cfg, WORKSPACE)
            self.pad.spawn(self.world)
            self.camera = DownwardCamera(
                vision_cfg, self.pad.board, self.pad.dictionary)
            self.camera.attach(self.vehicle.prim_path)

        ns = f"/{isaac_cfg['namespace']}{int(isaac_cfg['vehicle_id'])}"
        node = self.ros_backend.node
        self.wind_pub = node.create_publisher(Vector3Stamped, ns + "/environment/wind", 10)
        self.force_pub = node.create_publisher(Vector3Stamped, ns + "/environment/aero_force", 10)
        self.marker_pub = node.create_publisher(Float32, ns + "/perception/marker_quality", 10)
        self.pad_pose_pub = node.create_publisher(PoseStamped, ns + "/perception/uav_pose_in_pad", 10)
        self.reset_ack_pub = node.create_publisher(String, "/landing_sim/reset_ack", 10)
        node.create_subscription(String, "/landing_sim/reset", self._on_reset_request, 10)

        self.wind = WindField(CONFIG["wind"])
        self.pending_reset: dict | None = None
        self.last_wind = np.zeros(3)
        self.last_force = np.zeros(3)
        self.world.add_physics_callback("/landing_wind", self._apply_wind)
        self.world.reset()
        if self.camera is not None:
            self.camera.start()
            self.camera.aim_at_nadir(self.vehicle)
        self.wind.reset(int(CONFIG["wind"].get("seed", 49)), self.world.current_time)
        self.stop_sim = False

    def _on_reset_request(self, msg: String) -> None:
        try:
            req = json.loads(msg.data)
            if int(req.get("v", -1)) != int(CONFIG["system"]["protocol_version"]):
                raise ValueError("protocol version mismatch")
            scale = float(req.get("wind_scale", 1.0))
            if not math.isfinite(scale) or not 0.0 <= scale <= 4.0:
                raise ValueError("wind_scale outside [0,4]")
            self.pending_reset = {"seq": int(req["seq"]), "seed": int(req.get("seed", 0)),
                                  "wind_scale": scale}
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            carb.log_warn(f"Ignored malformed reset request: {exc}")

    def _perform_reset(self) -> None:
        """Reseed the episode and report the pose the vehicle must fly to.

        Teleporting a running vehicle is not an option here: Pegasus'
        ``PX4MavlinkBackend.reset`` is a documented no-op, so PX4's EKF keeps
        integrating through the jump and diverges for seconds afterwards. The
        controller is contractually fed PX4 estimator data, so a diverged
        estimator makes the episode meaningless. The entry pose is therefore
        drawn here -- keeping the seeded initial-condition distribution of the
        original in-process simulator -- and flown by PX4 itself.
        """
        req, self.pending_reset = self.pending_reset, None
        rng = np.random.default_rng(req["seed"])
        position = np.array([1.2 * rng.normal(), 1.2 * rng.normal(), 3.4 + 1.4 * rng.random()])
        rpy_deg = np.array([4.0 * rng.normal(), 4.0 * rng.normal(), 12.0 * rng.normal()])
        recovered = self._recover_if_tipped_over()
        for backend in (self.px4_backend, self.ros_backend):
            backend.reset()
        self.wind.reset(req["seed"], self.world.current_time, req["wind_scale"])
        ack = String()
        ack.data = json.dumps({"v": int(CONFIG["system"]["protocol_version"]),
                               "seq": req["seq"], "seed": req["seed"],
                               "wind_scale": req["wind_scale"],
                               "entry_position_enu_m": position.tolist(),
                               "entry_rpy_deg": rpy_deg.tolist(),
                               "entry_yaw_enu_rad": math.radians(float(rpy_deg[2])),
                               "recovered_from_tipover": bool(recovered)})
        self.reset_ack_pub.publish(ack)
        carb.log_info(f"Landing episode reset: seq={req['seq']} seed={req['seed']} "
                      f"entry={position.tolist()} recovered={recovered}")

    def _recover_if_tipped_over(self) -> bool:
        """Re-place the vehicle on the pad only when it cannot take off again.

        An upright vehicle is left exactly where the previous episode ended so
        that PX4's estimator is never stepped; a vehicle lying on its side can
        never fly the entry pose, so there the estimator transient is the lesser
        evil and the pad is restored.
        """
        roll, pitch, _ = Rotation.from_quat(self.vehicle.state.attitude).as_euler("XYZ")
        if math.hypot(roll, pitch) <= math.radians(float(CONFIG["landing"]["crash_tilt_deg"])):
            return False
        spawn = np.array([float(x) for x in CONFIG["isaac"]["spawn_position_enu_m"]])
        self.vehicle.set_world_pose(position=spawn, orientation=np.array([1.0, 0.0, 0.0, 0.0]))
        self.vehicle.set_linear_velocity(np.zeros(3))
        self.vehicle.set_angular_velocity(np.zeros(3))
        carb.log_warn("Vehicle had tipped over; restored to the pad before the next episode.")
        return True

    def _apply_wind(self, dt: float) -> None:
        state = self.vehicle.state
        wind, force_enu, force_body = self.wind.force(
            self.world.current_time, state.linear_velocity, state.attitude
        )
        self.vehicle.apply_force(force_body.tolist(), body_part="/body")
        self.last_wind, self.last_force = wind, force_enu

    def _publish_environment(self) -> None:
        stamp = self.ros_backend.node.get_clock().now().to_msg()
        for publisher, vector in ((self.wind_pub, self.last_wind), (self.force_pub, self.last_force)):
            msg = Vector3Stamped()
            msg.header.stamp = stamp
            msg.header.frame_id = "map"
            msg.vector.x, msg.vector.y, msg.vector.z = (float(x) for x in vector)
            publisher.publish(msg)
        if self.vision_enabled:
            self._publish_marker_detection(stamp)
        else:
            self._publish_marker_proxy()

    def _publish_marker_proxy(self) -> None:
        """Analytic stand-in used when the camera is switched off."""
        pos = self.vehicle.state.position
        rotation = Rotation.from_quat(self.vehicle.state.attitude)
        roll, pitch, _ = rotation.as_euler("XYZ")
        vision = CONFIG["vision"]
        quality = math.exp(-((max(pos[2], 0.0) / float(vision["max_range_m"])) ** 2))
        quality *= math.exp(-((math.hypot(roll, pitch) / math.radians(float(vision["tilt_scale_deg"]))) ** 2))
        quality *= math.exp(-((np.linalg.norm(pos[:2]) / float(vision["xy_scale_m"])) ** 2))
        marker = Float32()
        marker.data = float(np.clip(quality, 0.0, 1.0))
        self.marker_pub.publish(marker)

    def _publish_marker_detection(self, stamp) -> None:
        """Run the pad detector on the current frame and publish what it saw.

        Quality is the detector's own confidence, so the ontology consumes real
        perception health rather than a function of ground truth. A miss
        publishes zero and no pose, which is what makes the gateway fall back
        to the PX4 estimate.
        """
        observation = self.camera.observe()
        marker = Float32()
        if observation is None or not observation.detected:
            marker.data = 0.0
            self.marker_pub.publish(marker)
            return
        marker.data = float(observation.quality)
        self.marker_pub.publish(marker)

        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = "landing_pad"
        pose.pose.position.x = float(observation.position_pad_enu[0])
        pose.pose.position.y = float(observation.position_pad_enu[1])
        pose.pose.position.z = float(observation.position_pad_enu[2])
        w, x, y, z = (float(v) for v in observation.quaternion_pad_flu_wxyz)
        pose.pose.orientation.w = w
        pose.pose.orientation.x = x
        pose.pose.orientation.y = y
        pose.pose.orientation.z = z
        self.pad_pose_pub.publish(pose)

    def run(self):
        self.timeline.play()
        # rendering_dt is a multiple of physics_dt, so one frame covers several
        # physics steps. Drawing on every step would render at the physics rate
        # instead, and because PX4 runs in lockstep that slows the flight stack
        # itself, not just the picture.
        render_divider = max(1, round(float(CONFIG["isaac"]["rendering_dt"]) /
                                      float(CONFIG["isaac"]["physics_dt"])))
        step = 0
        while simulation_app.is_running() and not self.stop_sim:
            if self.pending_reset is not None:
                self._perform_reset()
            step += 1
            frame_boundary = step % render_divider == 0
            # The pad camera only produces an image on a rendered frame, so
            # vision costs rendering even in a headless run.
            render = frame_boundary and (self.vision_enabled or not ARGS.headless)
            self.world.step(render=render)
            if frame_boundary:
                self._publish_environment()
        self.timeline.stop()
        simulation_app.close()


def main():
    app = LandingWorld()
    try:
        app.run()
    except KeyboardInterrupt:
        app.stop_sim = True


if __name__ == "__main__":
    main()
