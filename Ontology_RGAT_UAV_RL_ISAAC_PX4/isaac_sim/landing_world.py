#!/usr/bin/env python3
"""Isaac Sim 5.1/Pegasus 5.1 landing world with PX4 Simulator MAVLink.

Launch this file with Pegasus' ``isaac_run`` helper, not system Python.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
import sys
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--parallel-pairs", type=int, default=1)
    return parser.parse_args()


ARGS = parse_args()
if ARGS.parallel_pairs < 1 or ARGS.parallel_pairs > 3:
    raise SystemExit("--parallel-pairs must be between 1 and 3")
CONFIG_PATH = Path(ARGS.config).expanduser().resolve()
WORKSPACE = CONFIG_PATH.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(WORKSPACE / "python"))
from config_loader import load_config
from ontology_rgat.initialization import (camera_centered_hover_offset,
                                          constrain_camera_visible_entry,
                                          curriculum_camera_entry,
                                          yaw_aligned_hover_offset)
from ontology_rgat.benchmarks.randomization import (
    px4_gain_parameters, sample_domain_randomization)

CONFIG = load_config(CONFIG_PATH)
from sensor_profiles import isaac_runtime_profile

RUNTIME = isaac_runtime_profile(CONFIG, ARGS.headless)

# SimulationApp must be constructed before importing Omniverse/Pegasus modules.
from isaacsim import SimulationApp

app_config = {"headless": ARGS.headless}
if not ARGS.headless:
    resolution = RUNTIME.viewport_resolution
    if ARGS.parallel_pairs > 1:
        resolution = tuple(int(value) for value in (
            ((CONFIG.get("parallel") or {}).get("operator_view") or {}).get(
                "viewport_resolution", resolution)))
    app_config.update({"width": resolution[0], "height": resolution[1]})
simulation_app = SimulationApp(app_config)

import carb
import omni.timeline
from isaacsim.core.utils.extensions import enable_extension

# Isaac Sim ships a Python-3.11-compatible rclpy inside the ROS 2 bridge.
# The extension must be enabled and allowed one update before ROS imports.
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

import rclpy
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, String
from scipy.spatial.transform import Rotation
from isaacsim.core.api import World
from isaacsim.core.utils.viewports import set_camera_view

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.backends.px4_mavlink_backend import (
    PX4MavlinkBackend,
    PX4MavlinkBackendConfig,
)
from pegasus.simulator.logic.backends.tools.px4_launch_tool import PX4LaunchTool
from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend
from pegasus.simulator.logic.dynamics import LinearDrag
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from isaacsim.core.api.materials import OmniPBR, PhysicsMaterial
from isaacsim.sensors.physics import ContactSensor
from isaacsim.sensors.camera import Camera
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from marker_vision import (
    nadir_footprint_m,
    R_BODY_FROM_OPTICAL,
    MarkerBoard,
    MarkerPoseEstimator,
    generate_marker_png,
    intrinsics_from_fov,
    texture_side_ratio,
)
from pad_motion import (BENCHMARK_SCENARIOS, PadMotionConfig, PadTrajectory,
                        lorry_parts, ugv_parts)
from urban_scene import UrbanConfig, UrbanLayout, UrbanScene
from gnss import GnssConfig, UrbanGnss
from px4_gnss import UrbanGnssSensor
from px4_sitl_parameters import configured_px4_parameters, px4_rc_script
from live_overlay import LiveOverlay
from metasejong_scene import MetaSejongConfig, MetaSejongScene
from view_geometry import paired_view_pose, street_offset_enu
from wind_sensor import WindSensor
from sensor_profiles import validate_camera_profile, vn100_pegasus_config
from vn100_imu import Vn100Imu


IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


class ParameterizedPX4LaunchTool(PX4LaunchTool):
    """Pegasus launcher using a temporary rcS wrapper owned by this run."""

    def __init__(self, px4_dir, vehicle_id, px4_model, parameters):
        super().__init__(px4_dir, vehicle_id, px4_model)
        wrapper = Path(self.root_fs.name) / "ontology_rgat_rcS"
        wrapper.write_text(px4_rc_script(self.rc_script, parameters), encoding="utf-8")
        self.rc_script = str(wrapper)


class ParameterizedPX4MavlinkBackend(PX4MavlinkBackend):
    """PX4 backend that applies configured parameters before flight starts."""

    def __init__(self, config, parameters):
        super().__init__(config)
        self._startup_parameters = parameters

    def start(self):
        if self._is_running:
            return
        if self._connection is None:
            self.re_initialize_interface()
        self._is_running = True
        if self.px4_autolaunch and self.px4_tool is None:
            carb.log_info("Attempting to launch configured PX4 in background process")
            self.px4_tool = ParameterizedPX4LaunchTool(
                self.px4_dir, self._vehicle_id, self.px4_vehicle_model,
                self._startup_parameters)
            self.px4_tool.launch_px4()

    def set_runtime_parameters(self, values: dict[str, float]) -> None:
        """Send episode gain randomization over the live HIL MAVLink link."""
        if self._connection is None:
            raise RuntimeError("PX4 MAVLink is unavailable for gain randomization")
        target_system = int(getattr(self._connection, "target_system", 0) or 1)
        target_component = int(getattr(self._connection, "target_component", 0) or 1)
        from pymavlink import mavutil
        for name, value in values.items():
            self._connection.mav.param_set_send(
                target_system, target_component, name.encode("ascii"),
                float(value), mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
        carb.log_info("[domain-randomization] PX4 gains sent: " + ", ".join(
            f"{name}={value:.4g}" for name, value in values.items()))


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
    """The ArUco tags painted on the pad, as textured quads in the stage.

    The quads are children of the deck prim, so they ride the rover without any
    per-frame bookkeeping here: moving the parent moves the pad.
    """

    def __init__(self, config: dict, workspace: Path, parent_path: str = "/World/landing_pad"):
        self.board = MarkerBoard.from_config(config["board"])
        self.dictionary = str(config["dictionary"])
        self.texture_dir = workspace / "assets" / "markers"
        self.ratio = texture_side_ratio(self.dictionary)
        self.parent_path = parent_path

    def spawn(self, world) -> None:
        stage = world.stage
        UsdGeom.Xform.Define(stage, self.parent_path)
        for marker in self.board.markers.values():
            texture = generate_marker_png(
                self.texture_dir / f"{self.dictionary}_{marker.marker_id}.png",
                self.dictionary, marker.marker_id)
            material = OmniPBR(
                prim_path=f"{self.parent_path}/material_{marker.marker_id}",
                name=f"landing_marker_{marker.marker_id}",
                texture_path=str(texture),
                texture_scale=np.array([1.0, 1.0]),
                texture_translate=np.array([0.0, 0.0]))
            # OmniPBR turns on world-space UV projection in its constructor,
            # which ignores the quad's own UVs and crops the marker's black
            # border and quiet zone away. Without both, it is not a tag.
            material.set_project_uvw(False)
            path = f"{self.parent_path}/marker_{marker.marker_id}"
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

    def __init__(self, config: dict, board: MarkerBoard, dictionary: str,
                 runtime_rate_hz: int | None = None):
        camera_cfg = config["camera"]
        validate_camera_profile(camera_cfg)
        self.width, self.height = (int(v) for v in camera_cfg["resolution"])
        self.fov_deg = float(camera_cfg["horizontal_fov_deg"])
        self.nominal_rate_hz = float(camera_cfg["rate_hz"])
        self.rate_hz = int(runtime_rate_hz or round(self.nominal_rate_hz))
        self.pitch_down_deg = float(camera_cfg.get("pitch_down_deg", 90.0))
        self.mount = np.array([float(v) for v in camera_cfg["mount_translation_flu_m"]])
        self.clipping = tuple(float(v) for v in camera_cfg.get("clipping_range_m", (0.02, 60.0)))
        self.camera_matrix = intrinsics_from_fov(self.width, self.height, self.fov_deg)
        self.estimator = MarkerPoseEstimator(
            board, self.camera_matrix, self.mount, dictionary=dictionary,
            quality_reprojection_px=float(config.get("quality_reprojection_px", 3.0)),
            quality_full_scale_px=float(config.get("quality_full_scale_px", 120.0)))
        self.camera = None
        self.raw_gray = None
        # Set ONTOLOGY_RGAT_VISION_DEBUG_DIR to dump annotated frames; the only
        # way to tell "no image" from "no marker in it" is to look at one.
        self.debug_dir = os.environ.get("ONTOLOGY_RGAT_VISION_DEBUG_DIR", "")
        self.debug_every = int(os.environ.get("ONTOLOGY_RGAT_VISION_DEBUG_EVERY", "30"))
        self.frames = 0
        self.annotated_rgb = None
        self.domain_visual = None
        carb.log_info(
            f"[landing-camera] {camera_cfg.get('model', 'camera')}: "
            f"{self.width}x{self.height}@{self.rate_hz:g} Hz runtime "
            f"({self.nominal_rate_hz:g} Hz device), "
            f"HFOV={self.fov_deg:g} deg, pitch-down={self.pitch_down_deg:g} deg"
        )

    def attach(self, vehicle_prim_path: str) -> None:
        self.camera = Camera(
            prim_path=vehicle_prim_path + "/body/landing_camera",
            resolution=(self.width, self.height),
            frequency=self.rate_hz)
        self.camera.set_local_pose(translation=self.mount,
                                   orientation=IDENTITY_QUAT, camera_axes="ros")

    def aim_at_nadir(self, vehicle) -> None:
        """Apply the configured down-pitch, despite Isaac's axis convention.

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
        pitch_from_nadir = math.radians(self.pitch_down_deg - 90.0)
        desired = Rotation.from_euler("y", pitch_from_nadir).as_matrix() @ R_BODY_FROM_OPTICAL
        correction = desired @ identity_mount.T
        self.camera.set_local_pose(
            translation=self.mount,
            orientation=_matrix_to_quat_wxyz(correction),
            camera_axes="ros")

        body_from_optical, mount = self._measure_mount(vehicle)
        view_in_body = body_from_optical[:, 2]
        expected_view = desired[:, 2]
        if float(np.dot(view_in_body, expected_view)) < 0.99:
            carb.log_error(
                f"Landing camera optical +Z is {view_in_body}, expected "
                f"{expected_view} in body axes. Marker detection will be unusable.")
        self.estimator.body_from_optical = body_from_optical
        self.estimator.mount_translation_body = mount
        carb.log_info(f"[landing-camera] optical axes in body:\n{body_from_optical}\n"
                      f"[landing-camera] mount in body: {mount}")

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
        carb.log_info(
            f"[landing-camera] {self.width}x{self.height} fov={fov:.1f}deg "
            f"(asked {self.fov_deg:.1f}) fx={self.camera_matrix[0, 0]:.1f} "
            f"focal={self.camera.get_focal_length():.4f} "
            f"aperture={self.camera.get_horizontal_aperture():.4f} "
            f"world_pose={self.camera.get_world_pose()}")

    def observe(self):
        frame = self.camera.get_rgba()
        if frame is None or frame.size == 0:
            if self.frames == 0:
                carb.log_warn("Landing camera produced no frame; is rendering enabled?")
            self.annotated_rgb = None
            self.frames += 1
            return None
        image = frame[:, :, :3]
        if self.domain_visual is not None:
            image = self._randomize_visual(image)
        # Actor topic: unannotated grayscale pixels only.  No pose, marker
        # solve or simulator overlay is rendered into this image.
        self.raw_gray = np.ascontiguousarray(
            np.clip(np.mean(image.astype(np.float32), axis=2), 0, 255).astype(np.uint8))
        observation, self.annotated_rgb = self.estimator.detect_annotated(image)
        self.frames += 1
        if self.debug_dir and self.frames % max(1, self.debug_every) == 0:
            self._dump(self.annotated_rgb)
        return observation

    def configure_domain_randomization(self, sample) -> None:
        """Apply Table-II appearance ranges to the actual actor camera stream."""
        self.domain_visual = None if sample is None else sample.to_dict()

    def _randomize_visual(self, image: np.ndarray) -> np.ndarray:
        values = self.domain_visual
        rgb = np.asarray(image[:, :, :3], dtype=np.float32)
        if float(np.nanmax(rgb)) <= 1.0:
            rgb *= 255.0
        yy, xx = np.mgrid[-1.0:1.0:complex(0, self.height),
                          -1.0:1.0:complex(0, self.width)]
        texture_id = int(values["ground_texture_id"])
        texture_scale = float(values["ground_texture_scale"])
        phase = texture_id * 0.61803398875
        frequency = (1.5 + texture_id % 7) / max(texture_scale, 1e-3)
        texture = 0.90 + 0.10 * np.sin(
            frequency * (xx + 0.73 * yy) + phase)
        angle = math.radians(float(values["light_direction_deg"]))
        directional = np.clip(
            1.0 + 0.10 * (math.cos(angle) * xx + math.sin(angle) * yy),
            0.75, 1.25)
        transform = (float(values["brightness"])
                     * np.asarray(values["rgb_scale"], dtype=np.float32))
        rgb *= transform[None, None, :]
        rgb *= (texture * directional)[:, :, None]
        return np.ascontiguousarray(np.clip(rgb, 0.0, 255.0).astype(np.uint8))

    def _dump(self, annotated_rgb) -> None:
        import cv2
        path = Path(self.debug_dir)
        path.mkdir(parents=True, exist_ok=True)
        canvas = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path / f"frame_{self.frames:06d}.png"), canvas)


# Only the airborne-start hold uses this: it cancels the vehicle's weight so a
# disarmed multirotor can wait in the air. Isaac owns the physics everywhere
# else, including contact with the deck.
GRAVITY_M_S2 = 9.81


class LandingDeck:
    """The ground vehicle the landing pad is bolted to.

    The deck is a kinematic rigid body: PhysX derives its velocity from the
    poses written each physics step, so a vehicle resting on it is carried along
    and friction behaves. A plain static collider that is teleported would let
    the drone slide as if the deck were standing still, and no collider at all
    would let it fall through to the ground plane between episodes.
    """

    PRIM = "/World/landing_rover"
    BODY = "/World/landing_rover/deck"

    def __init__(self, config: dict, *, pair_index: int = 0,
                 prim_path: str | None = None, world_offset=(0.0, 0.0, 0.0),
                 vehicle_prim: str = "/World/quadrotor",
                 route_phase_fraction: float = 0.0):
        self.pair_index = int(pair_index)
        self.PRIM = prim_path or type(self).PRIM
        self.BODY = self.PRIM + "/deck"
        self.vehicle_prim = str(vehicle_prim)
        self.world_offset = np.asarray(world_offset, dtype=float)
        if self.world_offset.shape != (3,) or not np.isfinite(self.world_offset).all():
            raise ValueError("parallel pair world offset must be three finite values")
        self.cfg = PadMotionConfig.from_mapping(config)
        self.route_phase_fraction = float(route_phase_fraction)
        self.trajectory = PadTrajectory(
            self.cfg, initial_route_fraction=self.route_phase_fraction)
        # Asking the trajectory where it starts, rather than assuming the
        # configured origin *is* the start. For the road profile they are not
        # the same point: the route is centred on the city block, so its origin
        # is the middle of the block -- inside a building. Anything placed
        # there, including the vehicle that spawns on this deck, is placed
        # inside the masonry.
        position, self.velocity = self.trajectory.initial_pose()
        self.position = np.asarray(position, dtype=float) + self.world_offset
        self.yaw = 0.0
        self.yaw_rate = 0.0
        self.physics_ok = False
        self._xform = None
        material = (config.get("pad") or {}).get("contact_material", {}) or {}
        self.static_friction = float(material.get("static_friction", 2.0))
        self.dynamic_friction = float(material.get("dynamic_friction", 1.6))
        self.restitution = float(material.get("restitution", 0.0))
        self.contact_min_force_n = float(material.get("contact_min_force_n", 0.25))
        self.contact_sensor = None
        self._contact_warning = False
        # The lorry waits at the kerb until the drone is off its roof. PX4 is
        # booting and levelling its estimator on whatever this deck does, and a
        # vehicle that is carried off at cruise speed and yawed round a corner
        # while it does that fails preflight ("horizontal velocity unstable",
        # "velocity estimate error") and then trips the in-flight mag check
        # ("Compass needs calibration - Land now!") once it is finally armed --
        # which hands the vehicle to a failsafe, out of offboard, in mid-climb.
        # Landing on a moving deck is the experiment; taking off from one is
        # not, so the lap starts when the drone is clear.
        self.held = True

    @property
    def surface_z(self) -> float:
        """World height of the marker plane the drone has to land on."""
        return float(self.position[2])

    def spawn(self, world) -> None:
        stage = world.stage
        root = UsdGeom.Xform.Define(stage, self.PRIM)
        self._xform = UsdGeom.XformCommonAPI(root)
        length, width = self.cfg.deck_size_m
        thickness = 0.08
        body = UsdGeom.Cube.Define(stage, self.BODY)
        body.CreateSizeAttr(1.0)
        # A dark, matte platform makes each marker's white quiet zone readable
        # in both the Isaac viewport and the downward grayscale camera.
        body.CreateDisplayColorAttr([Gf.Vec3f(0.10, 0.18, 0.24)])
        # A unit cube scaled to the deck, so the collider is an exact box and
        # the marker plane sits at the parent's origin (pad-frame z = 0).
        UsdGeom.XformCommonAPI(body).SetScale(
            Gf.Vec3f(float(length), float(width), float(thickness)))
        UsdGeom.XformCommonAPI(body).SetTranslate(Gf.Vec3d(0.0, 0.0, -thickness / 2.0))
        try:
            rigid = UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(self.PRIM))
            rigid.CreateKinematicEnabledAttr(True)
            body_prim = stage.GetPrimAtPath(self.BODY)
            UsdPhysics.CollisionAPI.Apply(body_prim)
            mass = UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(self.PRIM))
            mass.CreateMassAttr(float(self.cfg.vehicle_mass_kg))
            self._apply_grip_material(body_prim)
            # ContactSensor otherwise adds this API lazily and asks for a
            # stop/play cycle. The world has not started yet, so author it now
            # and the very first episode receives contact events.
            # ContactSensor walks up to the rigid-body owner (the lorry root),
            # while its immediate parent remains the roof collision shape.
            report = PhysxSchema.PhysxContactReportAPI.Apply(
                stage.GetPrimAtPath(self.PRIM))
            report.CreateThresholdAttr().Set(self.contact_min_force_n)
            self.contact_sensor = ContactSensor(
                prim_path=self.BODY + "/pad_contact_sensor",
                name=f"landing_pad_contact_{self.pair_index}",
                dt=float(CONFIG["isaac"]["physics_dt"]),
                min_threshold=self.contact_min_force_n,
                max_threshold=1.0e6,
                radius=-1.0,
            )
            self.contact_sensor.add_raw_contact_data_to_frame()
            self.physics_ok = True
            self._build_carrier(stage)
        except Exception as exc:                            # noqa: BLE001
            # Worth continuing without: the episode ends on pad-relative
            # altitude, not on contact. Worth shouting about: nothing will hold
            # the disarmed vehicle up between episodes.
            carb.log_error(
                f"could not make the landing deck a kinematic collider ({exc}); "
                "the vehicle will fall through it")
        self.apply_pose()

    def _apply_grip_material(self, body_prim) -> None:
        """Give the painted roof rubber-like grip and no contact bounce."""
        material = PhysicsMaterial(
            prim_path=f"/World/PhysicsMaterials/LandingDeckGrip_{self.pair_index}",
            name=f"landing_deck_grip_{self.pair_index}",
            static_friction=self.static_friction,
            dynamic_friction=self.dynamic_friction,
            restitution=self.restitution,
        )
        # The maximum combine mode makes the roof coefficient win over the
        # vehicle asset's default landing-gear material instead of averaging
        # the grip back down. Restitution uses the minimum to suppress bounce.
        physx_material = PhysxSchema.PhysxMaterialAPI.Apply(material.prim)
        physx_material.CreateFrictionCombineModeAttr().Set("max")
        physx_material.CreateRestitutionCombineModeAttr().Set("min")
        binding = UsdShade.MaterialBindingAPI.Apply(body_prim)
        binding.Bind(material.material,
                     bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                     materialPurpose="physics")

    def vehicle_contact(self, vehicle_position) -> tuple[bool, float]:
        """Return physical roof contact and normal-force magnitude.

        The sensor is attached to the roof collider. The local footprint gate
        prevents a collision against a lorry side from being called a landing.
        """
        if self.contact_sensor is None:
            return False, 0.0
        try:
            frame = self.contact_sensor.get_current_frame() or {}
            in_contact = bool(frame.get("in_contact", False))
            force = max(0.0, float(frame.get("force", 0.0)))
            contacts = frame.get("contacts") or []
            if contacts:
                in_contact = in_contact and any(
                    self.vehicle_prim in str(contact.get("body0", ""))
                    or self.vehicle_prim in str(contact.get("body1", ""))
                    for contact in contacts)
        except Exception as exc:                         # pragma: no cover - Isaac runtime
            if not self._contact_warning:
                carb.log_error(f"landing-pad contact sensor unavailable: {exc}")
                self._contact_warning = True
            return False, 0.0

        local = self.deck_local_from_world(vehicle_position)
        half_length = 0.5 * float(self.cfg.deck_size_m[0])
        half_width = 0.5 * float(self.cfg.deck_size_m[1])
        over_roof = (abs(float(local[0])) <= half_length
                     and abs(float(local[1])) <= half_width
                     and -0.25 <= float(local[2]) <= 0.60)
        return bool(in_contact and over_roof), force

    def _build_carrier(self, stage) -> None:
        """Draw the configured carrier underneath the landing surface.

        Until now the deck was a bare slab hanging three metres over the road
        with nothing beneath it: the pad moved like a lorry, occluded like a
        lorry and was called one everywhere in the code, but the viewport showed
        a floating plank.

        The shapes live next to their motion in ``pad_motion`` so they can be
        checked without a simulator; this only turns the selected one into
        USD. Everything hangs off the same kinematic body as the deck, so it
        drives, turns and stops with it for free.
        """
        official_visual = self._add_official_ugv_visual(stage)
        parts = (ugv_parts(self.cfg.deck_size_m, self.cfg.deck_height_m,
                           self.cfg.vehicle_dimensions_m)
                 if self.cfg.carrier == "ugv"
                 else lorry_parts(self.cfg.deck_size_m, self.cfg.deck_height_m))
        for part in parts:
            # The imported AgileX mesh supplies the visible body and wheels.
            # Retain only its audited simple body collider underneath it.
            if official_visual and not part.collider:
                continue
            path = f"{self.PRIM}/{part.name}"
            if part.kind == "wheel":
                prim = UsdGeom.Cylinder.Define(stage, path)
                prim.CreateRadiusAttr(0.5 * part.size[0])
                prim.CreateHeightAttr(part.size[1])
                prim.CreateAxisAttr("Y")
            else:
                prim = UsdGeom.Cube.Define(stage, path)
                prim.CreateSizeAttr(1.0)
                UsdGeom.XformCommonAPI(prim).SetScale(
                    Gf.Vec3f(*(float(v) for v in part.size)))
            UsdGeom.XformCommonAPI(prim).SetTranslate(
                Gf.Vec3d(*(float(v) for v in part.centre)))
            prim.CreateDisplayColorAttr([Gf.Vec3f(*part.colour)])
            if part.collider:
                UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(path))
            if official_visual:
                UsdGeom.Imageable(prim).MakeInvisible()

    def _add_official_ugv_visual(self, stage) -> bool:
        """Reference the public AgileX Ranger Mini V3 model when available."""
        if self.cfg.carrier != "ugv" or not self.cfg.vehicle_visual_usd:
            return False
        asset = Path(self.cfg.vehicle_visual_usd)
        if not asset.is_absolute():
            asset = WORKSPACE / asset
        if not asset.is_file():
            carb.log_warn(
                f"RANGER MINI 3.0 USD not found at {asset}; using audited fallback geometry. "
                "Run scripts/import_ranger_mini_v3.sh to generate it.")
            return False

        path = f"{self.PRIM}/ranger_mini_v3_visual"
        root = UsdGeom.Xform.Define(stage, path)
        root.GetPrim().GetReferences().AddReference(str(asset.resolve()))
        offset = self.cfg.vehicle_visual_origin_from_road_m
        UsdGeom.XformCommonAPI(root).SetTranslate(Gf.Vec3d(
            float(offset[0]), float(offset[1]),
            -float(self.cfg.deck_height_m) + float(offset[2])))

        # The pad trajectory owns motion. Remove the imported articulation and
        # collision APIs in the stronger session layer so there is only one
        # kinematic rigid body, while retaining all visual meshes/transforms.
        for prim in Usd.PrimRange(root.GetPrim()):
            if prim.IsA(UsdPhysics.Joint):
                prim.SetActive(False)
                continue
            for api in (UsdPhysics.RigidBodyAPI, UsdPhysics.CollisionAPI,
                        UsdPhysics.MassAPI, UsdPhysics.ArticulationRootAPI):
                if prim.HasAPI(api):
                    prim.RemoveAPI(api)
            for api in (PhysxSchema.PhysxRigidBodyAPI,
                        PhysxSchema.PhysxArticulationAPI):
                if prim.HasAPI(api):
                    prim.RemoveAPI(api)
        carb.log_info(
            f"[landing-pad] loaded official {self.cfg.vehicle_model} visual: {asset}")
        return True

    def reset(self, seed: int, sim_time: float, speed_scale: float = 1.0,
              scenario: str = "training_random_walk") -> dict:
        info = self.trajectory.reset(seed, sim_time, speed_scale, scenario)
        position, self.velocity = self.trajectory.pose(sim_time)
        self.position = np.asarray(position, dtype=float) + self.world_offset
        self.yaw = float(info["yaw_rad"])
        self.yaw_rate = 0.0
        if self.held:
            # The draw still happens -- the seed has to pick the same cruise
            # speed whether or not the lorry is standing -- but a parked deck
            # reports the twist it actually has. The policy feeds this forward,
            # so handing it the speed the lorry has not started driving at yet
            # would make it lead a target that is not moving.
            self.velocity = np.zeros(3)
            info["velocity_enu_m_s"] = self.velocity.tolist()
        self.apply_pose()
        # ``PadTrajectory`` does not know the optional world translation used
        # by an orchestration layout.  The reset acknowledgement must describe
        # the pose that was actually authored into USD, not its unshifted
        # trajectory coordinate.
        info["position_enu_m"] = self.position.tolist()
        info["surface_z_m"] = self.surface_z
        info["held"] = bool(self.held)
        return info

    def advance(self, sim_time: float, dt: float) -> None:
        if self.held:
            # Pin the lap's own clock to the simulation clock so no distance,
            # no lane wander and no traffic dip accrue while the lorry stands.
            # Everything the trajectory reports is a function of sim_time - t0,
            # so this freezes the pose exactly rather than approximately, and
            # the lap resumes from where it stands the moment it is released.
            self.trajectory.t0 = float(sim_time)
            position, _ = self.trajectory.pose(sim_time)
            self.position = np.asarray(position, dtype=float) + self.world_offset
            self.velocity = np.zeros(3)
            self.yaw_rate = 0.0
            self.apply_pose()
            return
        position, self.velocity = self.trajectory.pose(sim_time)
        self.position = np.asarray(position, dtype=float) + self.world_offset
        self.yaw, self.yaw_rate = self.trajectory.step_heading(self.velocity, dt)
        self.apply_pose()

    def release(self, sim_time: float) -> None:
        """Pull away from the kerb, once, at the first episode reset.

        One-way on purpose. The hold exists only so that PX4 boots and levels
        its estimator on a stationary deck; from the first episode onward the
        lorry drives continuously, which is what ``pad.route_start`` and the
        pad-relative entry pose are built around.

        The pull-away is smooth because a kinematic deck whose speed steps in
        one physics tick shears whatever is resting on it -- and what is resting
        on it here is the vehicle that is about to be armed. It is expressed as
        a full stop in the traffic profile centred on this instant, so the lorry
        accelerates away from the light it was waiting at using the same closed
        form the rest of the lap uses. The seed still draws the cruise speed and
        every light after this one; only the phase is re-anchored, and only
        once.
        """
        if not self.held:
            return
        self.held = False
        self.trajectory.pull_away(sim_time)

    def park(self) -> None:
        """Wait at the kerb again, so the next entry climb is over a still deck."""
        self.held = True
        self.velocity = np.zeros(3)
        self.yaw_rate = 0.0

    def apply_pose(self) -> None:
        if self._xform is None:
            return
        self._xform.SetTranslate(Gf.Vec3d(*(float(v) for v in self.position)))
        self._xform.SetRotate(Gf.Vec3f(0.0, 0.0, float(math.degrees(self.yaw))))

    def pad_from_world(self, point) -> np.ndarray:
        """World ENU point expressed in the pad frame."""
        return np.asarray(point, dtype=float) - self.position

    def deck_local_from_world(self, point) -> np.ndarray:
        """World point in the yawing lorry's footprint coordinates."""
        delta = self.pad_from_world(point)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([c * delta[0] + s * delta[1],
                         -s * delta[0] + c * delta[1], delta[2]])

    def world_from_pad(self, offset) -> np.ndarray:
        return self.position + np.asarray(offset, dtype=float)

    def odometry(self, stamp, position_error=None, velocity_error=None,
                 sigma_xy_m: float = 0.0) -> Odometry:
        """The deck's own broadcast.

        A cooperative vehicle sends where it *believes* it is, so the offsets
        are the errors its own receiver is making; passing none of them gives
        the ground truth, which is published on a separate topic and is for
        evaluation only. The reported accuracy goes into the pose covariance,
        because that is the field a consumer is entitled to read.
        """
        position = np.asarray(self.position, dtype=float)
        velocity = np.asarray(self.velocity, dtype=float)
        if position_error is not None:
            position = position + np.asarray(position_error, dtype=float)
        if velocity_error is not None:
            velocity = velocity + np.asarray(velocity_error, dtype=float)
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = "map"
        msg.child_frame_id = f"landing_pad_{self.pair_index}"
        msg.pose.pose.position.x = float(position[0])
        msg.pose.pose.position.y = float(position[1])
        msg.pose.pose.position.z = float(position[2])
        half = 0.5 * self.yaw
        msg.pose.pose.orientation.w = float(math.cos(half))
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = float(math.sin(half))
        msg.twist.twist.linear.x = float(velocity[0])
        msg.twist.twist.linear.y = float(velocity[1])
        msg.twist.twist.linear.z = float(velocity[2])
        msg.twist.twist.angular.z = float(self.yaw_rate)
        variance = float(sigma_xy_m) ** 2
        msg.pose.covariance[0] = variance
        msg.pose.covariance[7] = variance
        msg.pose.covariance[14] = variance
        return msg


class WindField:
    def __init__(self, config: dict):
        self.config = config
        self.rng = np.random.default_rng(int(config.get("seed", 49)))
        self.phases = self.rng.uniform(0.0, 2.0 * np.pi, size=(3, 6))
        self.freq = np.linspace(0.23, 1.91, 6)
        self.episode_t0 = 0.0
        self.scale = 1.0
        self.street: np.ndarray | None = None
        self.openness = 0.0

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
        wind = self._channel(wind)
        amp = float(self.config.get("turbulence_m_s", 0.0)) / math.sqrt(3.0)
        for axis in range(3):
            wind[axis] += amp * float(np.mean(np.sin(self.freq * t + self.phases[axis])))
        for gust in self.config.get("gusts", []):
            tau = (t - float(gust["t0_s"])) / max(float(gust["sigma_s"]), 1e-3)
            wind += np.asarray(gust["vector_enu_m_s"], dtype=float) * math.exp(-0.5 * tau * tau)
        return self.scale * wind

    def set_street_axis(self, heading_rad: float, openness: float = 0.0) -> None:
        """Which way the canyon runs here, and how much of it is still canyon.

        OPENNESS is the local sky view: at an intersection the facades stop
        channeling and the gradient wind comes back, so the drone gets a
        crosswind exactly where the GNSS also recovers.
        """
        self.street = np.array([math.cos(heading_rad), math.sin(heading_rad), 0.0])
        self.openness = float(np.clip(openness, 0.0, 1.0))

    def _channel(self, wind: np.ndarray) -> np.ndarray:
        """Split the gradient wind along and across the street.

        Facades turn a wind that crosses the street into one that runs along
        it: the along-street component is accelerated and the cross-street one
        is largely blocked. Only the mean flow is channeled -- the turbulence
        is what is left after the facades have finished with it.
        """
        canyon = self.config.get("canyon") or {}
        if not canyon.get("enabled", False) or self.street is None:
            return wind
        along_gain = float(canyon.get("along_gain", 1.0))
        cross_gain = float(canyon.get("cross_gain", 1.0))
        if canyon.get("open_sky_blend", True):
            along_gain += (1.0 - along_gain) * self.openness
            cross_gain += (1.0 - cross_gain) * self.openness
        along = float(np.dot(wind, self.street)) * self.street
        cross = wind - along
        return along_gain * along + cross_gain * cross

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


class ViewportFollower:
    """Keep the Isaac GUI camera on the UAV, and out of the buildings.

    A fixed world-frame offset is what an open field allows. A nineteen-metre
    street does not: an offset long enough to frame a six-metre lorry reaches
    past the facade, and a camera inside a facade renders a black screen with
    nothing on it but the unlit debug overlay -- which looks exactly like a
    broken simulator and is not one.

    So the offset is expressed **along the street** by default: back down the
    road behind the lorry and up, which is the natural view of a road chase and
    is the one direction a canyon leaves open. The result is then pulled toward
    the vehicle until it is clear of every building, so no configuration and no
    corner can bury it.
    """

    def __init__(self, config: dict):
        view = config.get("viewport_follow", {}) or {}
        self.enabled = bool(view.get("enabled", True)) and not ARGS.headless
        self.frame = str(view.get("frame", "street")).lower()
        if self.frame not in ("street", "world"):
            raise ValueError("isaac.viewport_follow.frame must be 'street' or 'world'")
        # street: [along the road (negative is behind), across, up].
        # world:  plain ENU, the open-field behaviour.
        self.offset = np.asarray(
            view.get("offset_m", view.get("offset_enu_m", [-14.0, 0.0, 7.0])), dtype=float)
        self.look_at_offset = np.asarray(view.get("look_at_offset_enu_m", [0.0, 0.0, 0.0]), dtype=float)
        self.focus = str(view.get("focus", "uav")).lower()
        if self.focus not in ("uav", "pair", "group"):
            raise ValueError(
                "isaac.viewport_follow.focus must be 'uav', 'pair' or 'group'")
        self.pair_span_m = float(view.get("pair_span_m", 7.0))
        if self.pair_span_m <= 0.0:
            raise ValueError("isaac.viewport_follow.pair_span_m must be positive")
        self.smoothing = float(np.clip(view.get("smoothing", 0.18), 0.01, 1.0))
        self.eye: np.ndarray | None = None
        self._warned = False

    def update(self, vehicle_position: np.ndarray, street_heading_rad: float = 0.0,
               layout=None, deck_position: np.ndarray | None = None) -> None:
        if not self.enabled:
            return
        position = np.asarray(vehicle_position, dtype=float).reshape(3)
        offset = self._offset_enu(street_heading_rad)
        if self.focus == "pair" and deck_position is not None:
            desired_eye, target, _ = paired_view_pose(
                position, deck_position, offset, self.pair_span_m)
        else:
            target = position
            desired_eye = target + offset
        target = target + self.look_at_offset
        if self.eye is None:
            self.eye = desired_eye
        else:
            self.eye += self.smoothing * (desired_eye - self.eye)
        # The offset can still point into a corner building on the outside of a
        # turn, so the layout gets the last word on where the camera may sit.
        eye = (self.eye if layout is None
               else layout.clear_of_buildings(self.eye, target))
        try:
            set_camera_view(eye=eye.tolist(), target=target.tolist())
        except Exception as exc:  # pragma: no cover - Isaac GUI/runtime dependent
            if not self._warned:
                print(f"Isaac viewport follow disabled: {exc}", file=sys.stderr, flush=True)
                self._warned = True
            self.enabled = False

    def _offset_enu(self, street_heading_rad: float) -> np.ndarray:
        """The configured offset in world ENU."""
        if self.frame == "world":
            return self.offset
        return street_offset_enu(self.offset, street_heading_rad)

    def update_group(self, vehicle_positions, deck_positions, headings,
                     layout=None) -> None:
        """Frame every live pair while retaining the street-relative camera."""
        if not self.enabled:
            return
        vehicles = np.asarray(vehicle_positions, dtype=float).reshape(-1, 3)
        decks = np.asarray(deck_positions, dtype=float).reshape(-1, 3)
        if not len(vehicles) or len(vehicles) != len(decks):
            return
        angles = np.asarray(headings, dtype=float).reshape(-1)
        heading = (math.atan2(float(np.mean(np.sin(angles))),
                              float(np.mean(np.cos(angles))))
                   if angles.size else 0.0)
        vehicle_center = np.mean(vehicles, axis=0)
        deck_center = np.mean(decks, axis=0)
        target = 0.5 * (vehicle_center + deck_center)
        all_positions = np.vstack((vehicles, decks))
        fleet_span = max(
            self.pair_span_m,
            2.0 * float(np.max(np.linalg.norm(
                all_positions[:, :2] - target[None, :2], axis=1))))
        offset = self._offset_enu(heading).copy()
        horizontal = max(float(np.linalg.norm(offset[:2])), 1.0)
        scale = max(1.0, 1.20 * fleet_span / horizontal)
        offset[:2] *= scale
        offset[2] = max(float(offset[2]), 0.55 * fleet_span)
        desired_eye, target, _ = paired_view_pose(
            vehicle_center, deck_center, offset, fleet_span)
        target = target + self.look_at_offset
        if self.eye is None:
            self.eye = desired_eye
        else:
            self.eye += self.smoothing * (desired_eye - self.eye)
        eye = (self.eye if layout is None
               else layout.clear_of_buildings(self.eye, target))
        try:
            set_camera_view(eye=eye.tolist(), target=target.tolist())
        except Exception as exc:  # pragma: no cover - Isaac GUI/runtime dependent
            if not self._warned:
                print(f"Isaac group viewport disabled: {exc}", file=sys.stderr, flush=True)
                self._warned = True
            self.enabled = False


class LandingWorld:
    """One independently controlled UAV/UGV pair in a possibly shared stage."""

    def __init__(self, *, pair_index: int = 0, pair_count: int = 1,
                 shared=None, defer_world_reset: bool = False):
        isaac_cfg = CONFIG["isaac"]
        self.pair_index = int(pair_index)
        self.pair_count = int(pair_count)
        self.parallel = self.pair_count > 1
        parallel_cfg = CONFIG.get("parallel") or {}
        configured_offsets = parallel_cfg.get("pair_offsets_enu_m") or (
            (0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (-4.0, 0.0, 0.0))
        if len(configured_offsets) < self.pair_count:
            raise ValueError("parallel.pair_offsets_enu_m has fewer entries than pairs")
        self.world_offset = np.asarray(
            configured_offsets[self.pair_index], dtype=float)
        configured_phases = parallel_cfg.get("route_phase_fractions") or (
            0.0, 0.08, 0.16)
        if len(configured_phases) < self.pair_count:
            raise ValueError("parallel.route_phase_fractions has fewer entries than pairs")
        self.route_phase_fraction = float(configured_phases[self.pair_index])
        if (not math.isfinite(self.route_phase_fraction)
                or not 0.0 <= self.route_phase_fraction < 1.0):
            raise ValueError("parallel route phases must be finite in [0,1)")
        self.topic_root = (f"/landing_pair_{self.pair_index}"
                           if self.parallel else "")
        self.pair_root = (f"/World/landing_pairs/pair_{self.pair_index}"
                          if self.parallel else "/World")
        self.runtime = RUNTIME
        self.startup_render_released = False
        if shared is None:
            self.timeline = omni.timeline.get_timeline_interface()
            self.pg = PegasusInterface()
            self.pg.set_world_settings(
                physics_dt=float(isaac_cfg["physics_dt"]),
                rendering_dt=self.runtime.rendering_dt,
            )
            self.pg._world = World(**self.pg._world_settings)
            self.world = self.pg.world
            environment = isaac_cfg["environment"]
            if environment not in SIMULATION_ENVIRONMENTS:
                raise KeyError(f"unknown Pegasus environment: {environment}")
            self.pg.load_environment(SIMULATION_ENVIRONMENTS[environment])

            self.metasejong = MetaSejongConfig.from_mapping(CONFIG, WORKSPACE)
            MetaSejongScene(self.metasejong).spawn(self.world)
            self.urban = UrbanLayout(UrbanConfig.from_mapping(CONFIG))
            if self.urban.cfg.latitude_deg or self.urban.cfg.longitude_deg:
                self.pg.set_global_coordinates(
                    latitude=self.urban.cfg.latitude_deg,
                    longitude=self.urban.cfg.longitude_deg,
                    altitude=self.urban.cfg.altitude_m)
                where = (f"{self.urban.cfg.latitude_deg:.5f}, "
                         f"{self.urban.cfg.longitude_deg:.5f}")
                if self.urban.extract_name:
                    carb.log_warn(f"City: {self.urban.extract_name} at {where} "
                                  f"({len(self.urban.buildings)} real footprints, "
                                  f"{self.urban.attribution})")
                else:
                    carb.log_warn(f"World origin set to {where}.")
            UrbanScene(self.urban).spawn(self.world)
        else:
            self.timeline = shared.timeline
            self.pg = shared.pg
            self.world = shared.world
            self.metasejong = shared.metasejong
            self.urban = shared.urban
        gnss_cfg = GnssConfig.from_mapping(CONFIG)
        self.gnss = UrbanGnss(gnss_cfg, self.urban if self.urban.cfg.enabled else None)
        self.gnss_enabled = bool(gnss_cfg.enabled)
        self.gnss_injected_into_px4 = bool(
            self.gnss_enabled and gnss_cfg.inject_into_px4)
        self.gnss_time = 0.0

        px4_dir = Path(isaac_cfg["px4_dir"])
        if not px4_dir.is_absolute():
            px4_dir = (WORKSPACE / px4_dir).resolve()
        vehicle_id = int(isaac_cfg["vehicle_id"]) + self.pair_index
        mavlink_cfg = PX4MavlinkBackendConfig({
            "vehicle_id": vehicle_id,
            "connection_type": isaac_cfg["mavlink_connection_type"],
            "connection_ip": isaac_cfg["mavlink_connection_ip"],
            "connection_baseport": int(isaac_cfg["mavlink_connection_baseport"]),
            "enable_lockstep": bool(isaac_cfg["lockstep"]),
            "update_rate": 1.0 / float(isaac_cfg["physics_dt"]),
            "px4_autolaunch": bool(isaac_cfg["px4_autolaunch"]),
            "px4_dir": str(px4_dir),
            "px4_vehicle_model": isaac_cfg["px4_vehicle_model"],
        })
        px4_cfg = CONFIG.get("px4") or {}
        self.px4_sitl_parameters = configured_px4_parameters(
            px4_cfg.get("sitl_parameters"))
        self.px4_backend = ParameterizedPX4MavlinkBackend(
            mavlink_cfg, self.px4_sitl_parameters)
        self.ros_backend = ROS2Backend(
            vehicle_id=vehicle_id,
            config={
                "namespace": isaac_cfg["namespace"],
                "pub_graphical_sensors": False,
                "pub_sensors": False,
                "pub_state": True,
                "pub_tf": True,
                "sub_control": False,
            },
        )
        # The pad rides on this, so it has to exist before the vehicle is
        # placed: the vehicle starts parked on the deck, not on the ground.
        vehicle_prim = (f"{self.pair_root}/quadrotor_{self.pair_index}"
                        if self.parallel else "/World/quadrotor")
        deck_prim = (f"{self.pair_root}/landing_rover"
                     if self.parallel else LandingDeck.PRIM)
        self.deck = LandingDeck(
            CONFIG, pair_index=self.pair_index, prim_path=deck_prim,
            world_offset=self.world_offset, vehicle_prim=vehicle_prim,
            route_phase_fraction=self.route_phase_fraction)
        self.deck.spawn(self.world)

        vehicle_cfg = MultirotorConfig()
        imu_cfg = vn100_pegasus_config(CONFIG, float(isaac_cfg["physics_dt"]))
        vehicle_cfg.sensors[1] = Vn100Imu(imu_cfg)
        carb.log_info(
            "[imu] VectorNav VN-100: "
            f"hardware max={(CONFIG.get('imu') or {}).get('hardware_output_rate_hz', 800):g} Hz, "
            f"Isaac injection={imu_cfg['update_rate']:g} Hz"
        )
        if self.gnss_injected_into_px4:
            # Replace Pegasus' generic clean GPS with the receiver that sees the
            # same buildings as the rendered camera. The IMU above carries the
            # VN-100 noise/range profile; PX4 EKF2 performs the actual
            # covariance-weighted fusion and inertial dead reckoning.
            self.urban_gps_sensor = UrbanGnssSensor(
                self.gnss, self.deck, gnss_cfg)
            vehicle_cfg.sensors[-1] = self.urban_gps_sensor
        else:
            self.urban_gps_sensor = None
        # Replace Pegasus' still-air linear drag with the wind-relative model below.
        vehicle_cfg.drag = LinearDrag([0.0, 0.0, 0.0])
        vehicle_cfg.backends = [self.px4_backend, self.ros_backend]
        # Where the vehicle waits before the autopilot has it. A hovering start
        # skips the takeoff entirely: the episode is a landing, so climbing off
        # the roof first only costs wall-clock and gives PX4 a chance to fail
        # before the part being measured begins.
        self.start_airborne = bool(isaac_cfg.get("start_airborne", False))
        self.hover_start_pad_m = np.array(
            [float(v) for v in isaac_cfg.get("hover_start_offset_pad_m", (0.0, 0.0, 4.5))],
            dtype=float)
        if bool(isaac_cfg.get("center_hover_on_landing_camera", False)):
            camera_cfg = CONFIG["vision"]["camera"]
            self.hover_start_pad_m = camera_centered_hover_offset(
                self.hover_start_pad_m[2],
                float(camera_cfg.get("pitch_down_deg", 60.0)),
                camera_cfg.get("mount_translation_flu_m", (0.0, 0.0, -0.16)))
        self.hover_hold_release_s = float(
            isaac_cfg.get("airborne_hold_release_s", 2.0))
        if not math.isfinite(self.hover_hold_release_s) or self.hover_hold_release_s < 0.0:
            raise ValueError("isaac.airborne_hold_release_s must be finite and non-negative")
        entry_altitudes = ((CONFIG.get("benchmark") or {}).get(
            "initial_conditions") or {}).get("relative_altitude_m", (2.0, 8.0))
        # Setup-only catch floor.  The ordinary flight remains entirely under
        # PX4, but a rare OFFBOARD/actuator handover race used to let the
        # airborne start fall through its 1.6 m entry target and strike the
        # deck before a measured episode existed.  Start assisting below a
        # conservative margin under the lowest legal entry altitude.
        lowest_entry_m = float(min(entry_altitudes))
        self.preentry_recovery_floor_m = max(0.60, lowest_entry_m - 0.30)
        self.preentry_recovery_target_m = max(
            self.preentry_recovery_floor_m + 0.20, lowest_entry_m)
        self.deck_clearance_pad_m = np.array(
            [float(v) for v in isaac_cfg["spawn_position_enu_m"]], dtype=float)
        # Only the airborne pre-arm hold prices weight with this; it is not the
        # vehicle's inertia and nothing else reads it. An error here is a
        # steady-state offset the PD closes, not a wrong flight model.
        self.hover_hold_mass_kg = float(
            (CONFIG.get("battery") or {}).get("vehicle_mass_kg", 1.5))
        self.px4_hover_thrust = float(
            (CONFIG.get("px4") or {}).get("hover_thrust", 0.58))
        self.px4_normalized_thrust = 0.0
        clearance = (self.hover_start_pad_m if self.start_airborne
                     else self.deck_clearance_pad_m)
        spawn = self.deck.world_from_pad(clearance).tolist()
        self.vehicle = Multirotor(
            vehicle_prim,
            ROBOTS[isaac_cfg["robot_asset"]],
            vehicle_id,
            spawn,
            [0.0, 0.0, 0.0, 1.0],
            config=vehicle_cfg,
        )

        vision_cfg = CONFIG["vision"]
        self.vision_enabled = str(vision_cfg.get("mode", "pose_proxy")) == "aruco"
        self.pad = None
        self.camera = None
        if self.vision_enabled:
            # Parented to the deck: the tags move with the rover for free.
            marker_cfg = vision_cfg
            if self.parallel:
                marker_cfg = deepcopy(vision_cfg)
                marker_cfg["dictionary"] = str((CONFIG.get("parallel") or {}).get(
                    "marker_dictionary", "DICT_5X5_250"))
                marker_stride = int((CONFIG.get("parallel") or {}).get(
                    "marker_id_stride", 60))
                for marker in marker_cfg["board"]:
                    marker["id"] = int(marker["id"]) + marker_stride * self.pair_index
            self.pad = LandingPadMarkers(marker_cfg, WORKSPACE, self.deck.PRIM)
            self.pad.spawn(self.world)
            self.camera = DownwardCamera(
                marker_cfg, self.pad.board, self.pad.dictionary,
                runtime_rate_hz=self.runtime.camera_rate_hz)
            self.camera.attach(self.vehicle.prim_path)

        battery_cfg = CONFIG.get("battery", {}) or {}
        self.battery_enabled = bool(battery_cfg.get("enabled", True))
        self.battery_hover_range = tuple(
            float(v) for v in battery_cfg.get("episode_hover_seconds_range", (6.0, 45.0)))

        ns = (f"{self.topic_root}/uav" if self.parallel else
              f"/{isaac_cfg['namespace']}{vehicle_id}")
        landing_topic = lambda legacy, relative: (
            f"{self.topic_root}/{relative}" if self.parallel else legacy)
        node = self.ros_backend.node
        # Physics consumes /environment/wind truth.  The learner consumes only
        # the separately modelled UAV anemometer measurement on /sensors/wind.
        self.wind_pub = node.create_publisher(Vector3Stamped, ns + "/sensors/wind", 10)
        self.wind_truth_pub = node.create_publisher(
            Vector3Stamped, ns + "/environment/wind", 10)
        self.force_pub = node.create_publisher(Vector3Stamped, ns + "/environment/aero_force", 10)
        self.marker_pub = node.create_publisher(Float32, ns + "/perception/marker_quality", 10)
        self.pad_pose_pub = node.create_publisher(PoseStamped, ns + "/perception/uav_pose_in_pad", 10)
        self.pad_contact_pub = node.create_publisher(Bool, ns + "/perception/pad_contact", 1)
        self.pad_contact_force_pub = node.create_publisher(
            Float32, ns + "/perception/pad_contact_force", 1)
        # Depth one prevents a slow/hidden RViz window from queuing full-size
        # frames. The frame itself carries all recognition diagnostics, so no
        # custom visualization message or cv_bridge dependency is needed.
        self.marker_image_pub = node.create_publisher(
            Image, ns + "/perception/landing_camera/annotated", 1)
        self.actor_image_pub = node.create_publisher(
            Image, ns + "/perception/landing_camera/image_raw", 1)
        self.reset_ack_pub = node.create_publisher(
            String, landing_topic("/landing_sim/reset_ack", "sim/reset_ack"), 10)
        # The deck broadcasts its own state, the way a cooperative ground
        # vehicle would. The drone's own estimate of the pad still comes from
        # its camera; this is what lets the gateway fall back to the PX4
        # estimate when the tags are out of frame.
        # What the lorry broadcasts: its own receiver's answer, canyon errors
        # and all. This is the only deck pose any consumer on the drone side is
        # allowed to read.
        self.deck_pub = node.create_publisher(
            Odometry, landing_topic("/landing_pad/state/odom", "pad/state/odom"), 10)
        # The simulator's truth, for scoring the episode afterwards. Nothing on
        # the control path subscribes to it; see docs/ARCHITECTURE.md, "GNSS".
        self.deck_truth_pub = node.create_publisher(
            Odometry, landing_topic(
                "/landing_pad/state/odom_truth", "pad/state/odom_truth"), 10)
        self.uav_truth_pub = node.create_publisher(
            Odometry, landing_topic(
                "/landing_uav0/state/odom_truth", "uav/state/odom_truth"), 10)
        self.gnss_pub = node.create_publisher(
            String, landing_topic("/landing_uav0/gnss/status", "uav/gnss/status"), 10)
        node.create_subscription(
            String, landing_topic("/landing_sim/reset", "sim/reset"),
            self._on_reset_request, 10)
        node.create_subscription(String, landing_topic(
                                 "/landing_sim/flight_state", "sim/flight_state"),
                                 self._on_flight_state, 10)
        # Held aloft until the autopilot is armed and flying it. Nothing else
        # can hold a disarmed multirotor in the air, and dropping it for the
        # second PX4 spends arming is the takeoff this start exists to avoid.
        self.autopilot_flying = False
        self.hover_hold_release_started: float | None = None
        # Wind belongs to the measured landing episode. Applying it while PX4
        # spends tens of seconds booting on a 1.5 m deck can push the unpowered
        # vehicle off the roof before the estimator has even fixed its origin.
        # Keep the field and sensor alive, but apply aerodynamic force only
        # after the first policy action marks the true episode handover.
        self.policy_handover = False
        self.domain_randomization = None
        self.domain_px4_gains = {}
        self.domain_initial_perturbation_pending = False
        self.last_pad_contact = False

        # A live 3D view of the episode inside the simulator window: the two
        # trails, the vector still to be closed and the success tolerance. Off
        # in a headless run, where nothing would read it.
        self.overlay = (shared.overlay if shared is not None else LiveOverlay(
            node, enabled=not ARGS.headless,
            telemetry_topics=(
                [f"/landing_rl/pair_{index}/telemetry"
                 for index in range(self.pair_count)] if self.parallel else None),
            success_radius_m=float(CONFIG["landing"]["success_xy_m"])))
        viewport_cfg = CONFIG["isaac"]
        if self.parallel and self.pair_index == 0:
            viewport_cfg = deepcopy(viewport_cfg)
            operator_view = dict(parallel_cfg.get("operator_view") or {})
            operator_view.pop("viewport_resolution", None)
            viewport_cfg.setdefault("viewport_follow", {}).update(operator_view)
        elif self.pair_index != 0:
            viewport_cfg = deepcopy(viewport_cfg)
            viewport_cfg.setdefault("viewport_follow", {})["enabled"] = False
        self.viewport_follower = ViewportFollower(viewport_cfg)

        self.wind = WindField(CONFIG["wind"])
        self.wind_sensor = WindSensor(CONFIG["wind"].get("sensor", {}))
        self.pending_reset: dict | None = None
        self.last_wind = np.zeros(3)
        self.last_wind_measurement = np.zeros(3)
        self.last_force = np.zeros(3)
        self.world.add_physics_callback(
            f"/landing_wind_{self.pair_index}", self._apply_wind)
        # Stepped with physics, not with rendering: PhysX derives the kinematic
        # deck's velocity from consecutive poses, so a pose written once per
        # rendered frame would give it a stale, chunky velocity.
        self.world.add_physics_callback(
            f"/landing_deck_{self.pair_index}", self._advance_deck)
        self.world.reset()
        if self.camera is not None:
            self.camera.start()
            self.camera.aim_at_nadir(self.vehicle)
        self.wind.reset(int(CONFIG["wind"].get("seed", 49)), self.world.current_time)
        self.wind_sensor.reset(int(CONFIG["wind"].get("seed", 49)), self.world.current_time)
        pad_cfg = CONFIG.get("pad") or {}
        preview_motion = bool(pad_cfg.get("preview_motion", False))
        preview_scale = float(pad_cfg.get("preview_speed_scale", 1.0))
        if not math.isfinite(preview_scale) or preview_scale < 0.0:
            raise ValueError("pad.preview_speed_scale must be finite and non-negative")
        self.deck.reset(
            int(CONFIG["wind"].get("seed", 49)), self.world.current_time,
            preview_scale if preview_motion else 1.0)
        # A manually started simulator has no learner to send a policy
        # handover, which previously made the UGV look broken because it stayed
        # parked forever.  Preview motion applies only to this initial idle
        # scene.  Every episode reset below still parks the deck and releases it
        # only at policy handover, preserving the safe entry-climb contract.
        if preview_motion:
            self.deck.release(self.world.current_time)
        self.gnss.reset(int((CONFIG.get("gnss") or {}).get("seed", 17)))
        self.gnss_time = float(self.world.current_time)
        self.stop_sim = False

    def _on_reset_request(self, msg: String) -> None:
        try:
            req = json.loads(msg.data)
            if int(req.get("v", -1)) != int(CONFIG["system"]["protocol_version"]):
                raise ValueError("protocol version mismatch")
            scale = float(req.get("wind_scale", 1.0))
            if not math.isfinite(scale) or not 0.0 <= scale <= 4.0:
                raise ValueError("wind_scale outside [0,4]")
            pad_scale = float(req.get("pad_scale", 1.0))
            if not math.isfinite(pad_scale) or not 0.0 <= pad_scale <= 4.0:
                raise ValueError("pad_scale outside [0,4]")
            initial_condition_scale = float(req.get(
                "initial_condition_scale", min(pad_scale, 1.0)))
            if (not math.isfinite(initial_condition_scale)
                    or not 0.0 <= initial_condition_scale <= 1.0):
                raise ValueError("initial_condition_scale outside [0,1]")
            # Scales the error mechanisms, not the buildings: 0.0 is the
            # open-sky control condition with the same city still standing.
            gnss_scale = float(req.get("gnss_scale", 1.0))
            if not math.isfinite(gnss_scale) or not 0.0 <= gnss_scale <= 4.0:
                raise ValueError("gnss_scale outside [0,4]")
            scenario = str(req.get("scenario", "training_random_walk"))
            if scenario not in BENCHMARK_SCENARIOS:
                raise ValueError(f"unknown benchmark scenario {scenario!r}")
            self.pending_reset = {"seq": int(req["seq"]), "seed": int(req.get("seed", 0)),
                                  "wind_scale": scale, "pad_scale": pad_scale,
                                  "initial_condition_scale": initial_condition_scale,
                                  "gnss_scale": gnss_scale,
                                  "scenario": scenario}
            self.startup_render_released = True
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
        self.policy_handover = False
        rng = np.random.default_rng(req["seed"])
        benchmark = CONFIG.get("benchmark") or {}
        domain_cfg = CONFIG.get("domain_randomization") or {}
        self.domain_randomization = (
            sample_domain_randomization(req["seed"])
            if bool(domain_cfg.get("enabled", False)) else None)
        self.domain_initial_perturbation_pending = self.domain_randomization is not None
        self.domain_px4_gains = {}
        if self.domain_randomization is not None:
            self.domain_px4_gains = px4_gain_parameters(
                self.domain_randomization,
                dict(domain_cfg.get("px4_nominal_gains") or {}))
            self.px4_backend.set_runtime_parameters(self.domain_px4_gains)
            if self.camera is not None:
                self.camera.configure_domain_randomization(
                    self.domain_randomization)
        elif self.camera is not None:
            self.camera.configure_domain_randomization(None)
        initial = benchmark.get("initial_conditions") or {}
        if str(benchmark.get("profile", "")).lower() == "shin2026":
            # Draw the Table-I box first. During training its curriculum starts
            # at the supported airborne hover and blends toward that draw;
            # every level is then conditioned on the paper's initial-FOV rule.
            x_range = initial.get("relative_lateral_x_m", (-3.0, 3.0))
            y_range = initial.get("relative_lateral_y_m", (-3.0, 3.0))
            z_range = initial.get("relative_altitude_m", (2.0, 8.0))
            yaw_range = initial.get("platform_yaw_misalignment_deg", (-60.0, 60.0))
            offset = np.array([rng.uniform(*x_range), rng.uniform(*y_range),
                               rng.uniform(*z_range)])
            rpy_deg = np.array([0.0, 0.0, rng.uniform(*yaw_range)])
            offset, rpy_deg[2] = curriculum_camera_entry(
                offset, rpy_deg[2],
                req.get("initial_condition_scale", min(
                    req.get("pad_scale", 1.0), 1.0)),
                minimum_scale=float(benchmark.get(
                    "initial_condition_curriculum_min_scale", 0.0)),
                # The gateway's pad-relative state is translated ENU, not the
                # yawing deck's local coordinates. Rotate the forward-camera
                # hover with the UGV heading or it is centred only at yaw=0.
                hover_offset_pad_m=yaw_aligned_hover_offset(
                    self.hover_start_pad_m, self.deck.yaw),
            )
            camera_cfg = (CONFIG.get("vision") or {}).get("camera", {}) or {}
            offset = constrain_camera_visible_entry(
                offset, self.deck.yaw + math.radians(float(rpy_deg[2])),
                image_size=camera_cfg.get("resolution", (512, 320)),
                horizontal_fov_deg=float(camera_cfg.get(
                    "horizontal_fov_deg", 90.0)),
                pitch_down_deg=float(camera_cfg.get("pitch_down_deg", 60.0)),
                mount_translation_flu_m=camera_cfg.get(
                    "mount_translation_flu_m", (0.0, 0.0, -0.16)),
                footprint_fraction=float(camera_cfg.get(
                    "entry_visible_footprint_fraction", 0.65)),
                target_radius_m=0.5 * math.hypot(*self.deck.cfg.deck_size_m),
            )
        else:
            # Retained urban distribution, expressed as an offset from the
            # moving deck rather than an absolute world point.
            reference_hover_m = 4.5
            entry_x = 1.8 * rng.normal()
            entry_y = 1.4 * rng.normal()
            entry_height = (4.0 + 1.8 * rng.random()) * max(
                float(self.hover_start_pad_m[2]), 0.5) / reference_hover_m
            offset = np.array([entry_x, entry_y, entry_height])
            offset = self._entry_within_camera(offset)
            rpy_deg = np.array([4.0 * rng.normal(), 4.0 * rng.normal(),
                                12.0 * rng.normal()])
        # A Gaussian tail can put the entry point inside a facade -- about one
        # seed in three thousand on the outer lane -- and PX4 flies into it,
        # never reaches the pose and times out the reset. Pulling the offset
        # back toward the deck keeps the drawn direction and the seeded
        # distribution everywhere it was already legal.
        deck_world = self.deck.world_from_pad(np.zeros(3))
        offset = self.urban.clear_of_buildings(
            deck_world + offset, deck_world) - deck_world
        # Drawn from the same generator as the entry pose so the whole initial
        # condition -- geometry, wind, deck motion and energy -- is one seed.
        hover_seconds = float(rng.uniform(*self.battery_hover_range))
        reseated = self._seat_on_deck()
        self.deck.park()
        # The lorry does NOT pull away here. It waits until handover, so PX4
        # flies the entry climb over a deck that is standing still and only the
        # landing -- the part being measured -- has to track a moving one. See
        # _on_flight_state.
        for backend in (self.px4_backend, self.ros_backend):
            backend.reset()
        self.wind.reset(req["seed"], self.world.current_time, req["wind_scale"])
        self.wind_sensor.reset(req["seed"], self.world.current_time)
        deck = self.deck.reset(req["seed"], self.world.current_time,
                               req.get("pad_scale", 1.0),
                               req.get("scenario", "training_random_walk"))
        entry_yaw_enu = (self.deck.yaw + math.radians(float(rpy_deg[2]))
                         if str(benchmark.get("profile", "")).lower() == "shin2026"
                         else math.radians(float(rpy_deg[2])))
        self.gnss.reset(req["seed"], req.get("gnss_scale", 1.0))
        self.gnss_time = float(self.world.current_time)
        self._update_environment_sensors(0.0)
        self._publish_deck()
        ack = String()
        ack.data = json.dumps({"v": int(CONFIG["system"]["protocol_version"]),
                               "seq": req["seq"], "seed": req["seed"],
                               "wind_scale": req["wind_scale"],
                               "pad_scale": req.get("pad_scale", 1.0),
                               "initial_condition_scale": req.get(
                                   "initial_condition_scale", 1.0),
                               "gnss_scale": req.get("gnss_scale", 1.0),
                               "gnss": {
                                   "enabled": self.gnss_enabled,
                                   "satellites": int(self.gnss.constellation.size),
                                   "uav": self.gnss.uav.last.to_dict(),
                                   "deck": self.gnss.deck.last.to_dict()},
                               "entry_offset_pad_m": offset.tolist(),
                               "entry_position_enu_m":
                                   self.deck.world_from_pad(offset).tolist(),
                               "entry_rpy_deg": rpy_deg.tolist(),
                               "entry_yaw_enu_rad": entry_yaw_enu,
                               "battery_hover_seconds":
                                   (hover_seconds if self.battery_enabled else None),
                               "domain_randomization": (
                                   None if self.domain_randomization is None else {
                                       **self.domain_randomization.to_dict(),
                                       "applied_px4_gains": self.domain_px4_gains,
                                       "visual_application":
                                           "live camera photometric/procedural texture",
                                   }),
                               "pad": deck,
                               "reseated_on_deck": bool(reseated)})
        self.reset_ack_pub.publish(ack)
        self.overlay.reset(self.pair_index if self.parallel else None)
        carb.log_info(f"Landing episode reset: seq={req['seq']} seed={req['seed']} "
                      f"entry_offset={offset.tolist()} pad={deck['mode']}@"
                      f"{deck['speed_m_s']:.2f} m/s battery={hover_seconds:.1f} s "
                      f"gnss={self.gnss.uav.last.satellites_tracked} sats "
                      f"({self.gnss.uav.last.satellites_nlos} NLOS, "
                      f"q={self.gnss.uav.last.quality:.2f}) reseated={reseated}")

    def _entry_within_camera(self, offset: np.ndarray) -> np.ndarray:
        """Pull the entry point in until the pad is inside the camera frame.

        Every episode is supposed to begin with the pad already in view, so the
        policy starts from a marker fix rather than spending its first seconds
        hunting for the deck on GNSS alone. The drawn offset does not respect
        that on its own: at the entry altitude the camera's short axis reaches
        about three quarters of its long one, and the Gaussian tails put the
        deck outside both.

        The direction that was drawn is kept and only the reach is shortened --
        the same treatment ``clear_of_buildings`` gives a point inside a facade.
        The budget is taken against the *short* axis, because the vehicle's yaw
        at handover is not controlled and either image axis could be the one
        pointing at the deck, and it leaves room for the entry tilt and for the
        vehicle sitting a little off the point it was aimed at.
        """
        vision = CONFIG.get("vision") or {}
        camera = vision.get("camera") or {}
        width, height = (int(v) for v in camera.get("resolution", (800, 600)))
        fov = float(camera.get("horizontal_fov_deg", 90.0))
        altitude = float(offset[2])
        _, half_short = nadir_footprint_m(width, height, fov, altitude)
        # What the vehicle may be off by and still see the deck: the entry tilt
        # swings the footprint, and it only has to hold the entry pose to
        # within the handover tolerance.
        tilt_m = altitude * math.tan(math.radians(float(
            vision.get("entry_tilt_allowance_deg", 8.0))))
        slack = float(vision.get("entry_fov_margin_m", 1.2))
        allowed = max(0.35 * half_short, half_short - tilt_m - slack)
        reach = float(math.hypot(offset[0], offset[1]))
        if reach > allowed > 0.0:
            offset = offset.copy()
            offset[:2] *= allowed / reach
        return offset

    def _seat_on_deck(self) -> bool:
        """Put a grounded vehicle back on the lorry's roof before the next episode.

        An *airborne* vehicle is left exactly where the previous episode ended,
        so that PX4's estimator is never stepped mid-flight -- that is the rule
        the whole reset design is built on and it does not change.

        In a deck-start run a grounded vehicle is re-seated. It has usually
        slid: the deck is redrawn with a new cruise speed at every reset, and a
        kinematic body whose speed steps in one physics tick shears whatever is
        resting on it. Airborne-start runs return above and never write a pose;
        their physical hover support and PX4 entry flight own all repositioning.
        """
        # Airborne-start runs never teleport the PX4 estimator. After a
        # touchdown the same physical support used at process startup lifts
        # the disarmed vehicle smoothly toward its hover; after a timeout PX4
        # remains airborne and flies the next entry pose itself.
        if self.start_airborne:
            return False
        state = self.vehicle.state
        roll, pitch, _ = Rotation.from_quat(state.attitude).as_euler("XYZ")
        tipped = math.hypot(roll, pitch) > math.radians(
            float(CONFIG["landing"]["crash_tilt_deg"]))
        # Anything within a metre of the roof is on the roof or on the road
        # beside it; either way it is not flying and the next episode starts
        # from the deck.
        grounded = float(state.position[2]) <= self.deck.surface_z + 1.0
        if not (tipped or grounded):
            return False
        spawn = self.deck.world_from_pad(self.deck_clearance_pad_m)
        moved = float(np.linalg.norm(np.asarray(state.position, dtype=float) - spawn))
        # Vehicle inherits Isaac Robot.set_world_pose, whose direct API is
        # scalar-first [w,x,y,z] even though Pegasus vehicle state is xyzw.
        self.vehicle.set_world_pose(
            position=spawn, orientation=np.array([1.0, 0.0, 0.0, 0.0]))
        self.vehicle.set_linear_velocity(
            np.zeros(3))
        self.vehicle.set_angular_velocity(np.zeros(3))
        if tipped:
            carb.log_warn(f"Vehicle had tipped over; re-placed on the deck ({moved:.2f} m).")
        elif moved > 0.5:
            carb.log_info(f"Vehicle re-placed on the deck ({moved:.2f} m).")
        return True

    def _on_flight_state(self, msg: String) -> None:
        """Latch when the autopilot takes the vehicle over."""
        try:
            state = json.loads(msg.data)
            if int(state.get("v", -1)) != int(CONFIG["system"]["protocol_version"]):
                raise ValueError("protocol version mismatch")
            flying = bool(state["armed"])
            handover = bool(state.get("handover", False))
            controlled = bool(state.get("controlled", flying))
            px4_thrust = float(state.get("px4_thrust", 0.0))
            if not math.isfinite(px4_thrust):
                raise ValueError("non-finite PX4 thrust")
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            carb.log_warn(f"Ignored malformed flight state: {exc}")
            return
        previous_control = self.autopilot_flying
        previous_handover = self.policy_handover
        self.autopilot_flying = bool(flying and controlled)
        self.px4_normalized_thrust = float(np.clip(px4_thrust, 0.0, 1.0))
        if self.autopilot_flying and not previous_control:
            self.hover_hold_release_started = float(self.world.current_time)
        elif not self.autopilot_flying:
            self.hover_hold_release_started = None
        self.policy_handover = bool(flying and handover)
        if (self.policy_handover and not previous_handover
                and self.domain_initial_perturbation_pending
                and self.domain_randomization is not None):
            state_now = self.vehicle.state
            rotation = Rotation.from_quat(state_now.attitude)
            velocity = (np.asarray(state_now.linear_velocity, dtype=float)
                        + rotation.apply(
                            self.domain_randomization.initial_velocity_m_s))
            angular_rate = (np.asarray(state_now.angular_velocity, dtype=float)
                            + self.domain_randomization.initial_angular_rate_rad_s)
            self.vehicle.set_linear_velocity(velocity)
            self.vehicle.set_angular_velocity(angular_rate)
            self.domain_initial_perturbation_pending = False
            carb.log_info(
                "[domain-randomization] applied policy-handover velocity/rate perturbation")
        # The gateway starts only after PX4 reports Ready for takeoff.  Its
        # first valid state therefore releases the cheap startup render loop
        # even for viewers such as run_metasejong_demo that do not reset first.
        self.startup_render_released = True
        if handover and self.deck.held:
            carb.log_warn("Policy has the vehicle; the lorry pulls away.")
            self.deck.release(self.world.current_time)

    def _hold_prearm_start(self, dt: float) -> None:
        """Keep a disarmed airborne start aloft until PX4 takes over.

        A deck-start vehicle must keep its weight on the roof so its configured
        friction can hold it. Cancelling gravity there removes the normal force
        and lets even a small disturbance slide it away. Pre-handover wind is
        gated separately in :meth:`_apply_wind`, so deck starts need no
        artificial force at all.

        An airborne start does need a real force: a multirotor with no
        autopilot has nothing holding it up while PX4 aligns and arms.

        It is held by force rather than by writing the pose, because the IMU is
        differentiated from successive velocities: pinning the transform every
        tick makes the accelerometer read a vehicle in free fall while GPS reads
        one standing still, and PX4 refuses to arm on the contradiction
        ("Preflight Fail: velocity estimate error"). Cancelling weight and
        closing a PD onto the hover point instead means the vehicle really is
        hovering, so every sensor agrees and the estimator aligns on a genuine
        hover -- which is the state the episode is supposed to start from.

        It is held *with* the deck, so the pad-relative pose the episode starts
        from is the one the reset drew.
        """
        if not self.start_airborne:
            return
        gain = 1.0
        emergency_target = None
        if self.autopilot_flying:
            elapsed = (float(self.world.current_time)
                       - float(self.hover_hold_release_started
                               if self.hover_hold_release_started is not None
                               else self.world.current_time))
            released = (
                self.hover_hold_release_s <= 0.0
                or elapsed >= self.hover_hold_release_s)
            state = self.vehicle.state
            pad_position = self.deck.pad_from_world(state.position)
            entry_too_low = bool(
                not self.policy_handover
                and float(pad_position[2]) < self.preentry_recovery_floor_m)
            if released and not entry_too_low:
                return
            scheduled_gain = (0.0 if released else
                              max(0.0, 1.0 - elapsed / self.hover_hold_release_s))
            # Do not add a full gravity-cancelling force on top of PX4's own
            # hover thrust. For a commanded descent PX4 initially asks for
            # almost zero thrust, so the scheduled taper lets it descend gently
            # until the controller catches the velocity. For a climb, thrust
            # rises immediately and the assist gets out of the way.
            thrust_gain = max(
                0.0,
                1.0 - self.px4_normalized_thrust / max(self.px4_hover_thrust, 1e-6))
            gain = min(scheduled_gain, thrust_gain)
            if entry_too_low:
                # The rescue acts only before policy handover and only below
                # the legal entry envelope. Hold current XY and recover Z so it
                # cannot steer the sampled initial condition or any RL action.
                emergency_target = np.asarray(state.position, dtype=float).copy()
                emergency_target[2] += (
                    self.preentry_recovery_target_m - float(pad_position[2]))
                penetration = (self.preentry_recovery_floor_m
                               - float(pad_position[2]))
                gain = max(gain, float(np.clip(
                    0.35 + penetration / 0.30, 0.35, 1.0)))
        state = self.vehicle.state
        position = np.asarray(state.position, dtype=float)
        velocity = np.asarray(state.linear_velocity, dtype=float)
        target = (emergency_target if emergency_target is not None
                  else self.deck.world_from_pad(self.hover_start_pad_m))
        target_velocity = np.asarray(self.deck.velocity, dtype=float)
        # Weight, plus a PD that is stiff enough to hold station against the
        # wind this environment blows and soft enough not to ring at 250 Hz.
        accel = (np.array([0.0, 0.0, GRAVITY_M_S2])
                 + 4.0 * (target - position)
                 + 3.5 * (target_velocity - velocity))
        force_world = gain * self.hover_hold_mass_kg * accel
        force_body = Rotation.from_quat(state.attitude).inv().apply(force_world)
        self.vehicle.apply_force(force_body.tolist(), body_part="/body")
        # And hold it level. A disarmed multirotor has no rotors to stabilise
        # attitude, so over the twenty-odd seconds PX4 spends aligning its
        # estimator it tips under any residual torque and trips the attitude
        # check before it will arm. Pinning the body rate is enough: the vehicle
        # spawns level and, with no rate, it stays there.
        self.vehicle.set_angular_velocity(np.zeros(3))

    def _advance_deck(self, dt: float) -> None:
        self.deck.advance(self.world.current_time, dt)
        # After the deck, so the vehicle is held against the pose the deck has
        # this tick rather than the one it had last tick.
        self._hold_prearm_start(dt)
        self._report_deck_carry()

    def _report_deck_carry(self) -> None:
        """Say out loud when the deck drives out from under a parked vehicle.

        The whole moving-deck design rests on the roof carrying whatever is
        resting on it, so that a disarmed vehicle rides the lorry and the
        pad-relative entry pose stays a few metres away. When it does not, every
        reset fails identically -- PX4 flies at an entry point that is receding
        at cruise speed -- and nothing upstream says why.
        """
        now = float(self.world.current_time)
        if now - getattr(self, "_carry_report_t", -1e9) < 2.0:
            return
        self._carry_report_t = now
        if self.start_airborne:
            return                          # it never rides the roof by design
        position = np.asarray(self.vehicle.state.position, dtype=float)
        pad = self.deck.pad_from_world(position)
        if pad[2] > 1.0:                    # airborne: nothing to be carried by
            return
        half_length = 0.5 * float(self.deck.cfg.deck_size_m[0])
        if float(np.hypot(pad[0], pad[1])) <= half_length:
            return
        carb.log_warn(
            f"vehicle is {np.hypot(pad[0], pad[1]):.1f} m from the deck centre "
            f"(roof half-length {half_length:.1f} m) at pad z={pad[2]:.2f} m: it is "
            f"not being carried, so the entry pose recedes at the lorry's speed "
            f"({np.linalg.norm(self.deck.velocity[:2]):.1f} m/s)")

    def _apply_wind(self, dt: float) -> None:
        state = self.vehicle.state
        wind, force_enu, force_body = self.wind.force(
            self.world.current_time, state.linear_velocity, state.attitude
        )
        if self.policy_handover:
            domain_force_body = (np.zeros(3) if self.domain_randomization is None
                                 else self.domain_randomization.external_force_n)
            domain_torque_body = (np.zeros(3) if self.domain_randomization is None
                                  else self.domain_randomization.external_torque_nm)
            total_force_body = force_body + domain_force_body
            self.vehicle.apply_force(total_force_body.tolist(), body_part="/body")
            if np.any(domain_torque_body):
                self.vehicle.apply_torque(
                    domain_torque_body.tolist(), body_part="/body")
            self.last_force = (force_enu
                               + Rotation.from_quat(state.attitude).apply(
                                   domain_force_body))
        else:
            # Wind speed remains observable during reset, but this is the
            # physical force actually applied to the vehicle.
            self.last_force = np.zeros(3)
        self.last_wind = wind

    def _update_environment_sensors(self, dt: float) -> None:
        """Both receivers' fixes and the local wind geometry, once per frame.

        Not at the physics rate: a fix is an observation, and re-solving it two
        hundred and fifty times a second would only add cost and a whiter noise
        than a real receiver's. The wind is updated on the same tick because it
        is the same geometry -- the facades that hide the satellites are the
        ones channeling the flow, and both relax at the intersections.
        """
        # When HIL injection is disabled this preserves the legacy telemetry-
        # only model.  With injection enabled UrbanGnssSensor updates both
        # receivers at the configured GNSS rate; updating again here would make
        # the pseudorange process run twice as fast as its timestamps.
        if self.gnss_enabled and not self.gnss_injected_into_px4:
            self.gnss.update(self.vehicle.state.position, self.deck.position, dt)
        if self.wind.config.get("canyon", {}).get("enabled", False):
            self.wind.set_street_axis(self.deck.yaw,
                                      self.urban.openness(self.deck.position))

    def _publish_deck(self) -> None:
        stamp = self.ros_backend.node.get_clock().now().to_msg()
        fix = self.gnss.deck.last
        # The cooperative vehicle knows its own velocity from wheel odometry;
        # GNSS supplies its global position and covariance. Injecting a fraction
        # of pseudorange position bias into speed made relative DR drift by
        # metres even though Doppler/wheel speed is the well-observed channel.
        self.deck_pub.publish(self.deck.odometry(
            stamp, fix.error_enu_m, None, fix.sigma_xy_m))
        self.deck_truth_pub.publish(self.deck.odometry(stamp))
        state = self.vehicle.state
        truth = Odometry()
        truth.header.stamp = stamp
        truth.header.frame_id = "map"
        truth.child_frame_id = f"landing_uav{self.pair_index}/base_link"
        truth.pose.pose.position.x = float(state.position[0])
        truth.pose.pose.position.y = float(state.position[1])
        truth.pose.pose.position.z = float(state.position[2])
        qx, qy, qz, qw = state.attitude
        truth.pose.pose.orientation.w = float(qw)
        truth.pose.pose.orientation.x = float(qx)
        truth.pose.pose.orientation.y = float(qy)
        truth.pose.pose.orientation.z = float(qz)
        truth.twist.twist.linear.x = float(state.linear_velocity[0])
        truth.twist.twist.linear.y = float(state.linear_velocity[1])
        truth.twist.twist.linear.z = float(state.linear_velocity[2])
        self.uav_truth_pub.publish(truth)

    def _publish_gnss(self) -> None:
        """The drone's own receiver, as a receiver would report it.

        The truth subobject remains simulator-only diagnostics.  When
        ``injected_into_px4`` is true the gateway must not apply that error a
        second time: it has already crossed HIL_GPS and been fused by EKF2.
        """
        if not self.gnss_enabled:
            return
        msg = String()
        msg.data = json.dumps({"v": int(CONFIG["system"]["protocol_version"]),
                               "injected_into_px4": self.gnss_injected_into_px4,
                               "hil_gps_mode": (self.urban_gps_sensor.mode
                                                if self.urban_gps_sensor is not None
                                                else "legacy"),
                               "uav": self.gnss.uav.last.to_dict(),
                               "deck": self.gnss.deck.last.to_dict()})
        self.gnss_pub.publish(msg)

    def _publish_environment(self) -> None:
        now = float(self.world.current_time)
        self._update_environment_sensors(max(now - self.gnss_time, 0.0))
        self.gnss_time = now
        self._publish_gnss()
        self._publish_deck()
        stamp = self.ros_backend.node.get_clock().now().to_msg()
        self.last_wind_measurement = self.wind_sensor.measure(self.last_wind, now)
        for publisher, vector in (
                (self.wind_pub, self.last_wind_measurement),
                (self.wind_truth_pub, self.last_wind),
                (self.force_pub, self.last_force)):
            msg = Vector3Stamped()
            msg.header.stamp = stamp
            msg.header.frame_id = "map"
            msg.vector.x, msg.vector.y, msg.vector.z = (float(x) for x in vector)
            publisher.publish(msg)
        self._publish_pad_contact()
        if self.vision_enabled:
            self._publish_marker_detection(stamp)
        else:
            self._publish_marker_proxy()

    def _publish_pad_contact(self) -> None:
        contact, force = self.deck.vehicle_contact(self.vehicle.state.position)
        contact_msg = Bool()
        contact_msg.data = bool(contact)
        self.pad_contact_pub.publish(contact_msg)
        force_msg = Float32()
        force_msg.data = float(force if contact else 0.0)
        self.pad_contact_force_pub.publish(force_msg)
        if contact != self.last_pad_contact:
            state = "ON" if contact else "OFF"
            carb.log_info(f"Landing pad contact {state}: force={force:.2f} N")
            self.last_pad_contact = contact

    def _publish_marker_proxy(self) -> None:
        """Analytic stand-in used when the camera is switched off.

        Measured against the deck, not the world origin, so switching the
        camera off does not silently change what "over the pad" means.
        """
        rel = self.deck.pad_from_world(self.vehicle.state.position)
        rotation = Rotation.from_quat(self.vehicle.state.attitude)
        roll, pitch, _ = rotation.as_euler("XYZ")
        vision = CONFIG["vision"]
        quality = math.exp(-((max(rel[2], 0.0) / float(vision["max_range_m"])) ** 2))
        quality *= math.exp(-((math.hypot(roll, pitch) / math.radians(float(vision["tilt_scale_deg"]))) ** 2))
        quality *= math.exp(-((np.linalg.norm(rel[:2]) / float(vision["xy_scale_m"])) ** 2))
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
        self._publish_actor_camera(stamp)
        self._publish_annotated_camera(stamp)
        marker = Float32()
        if observation is None or not observation.detected:
            marker.data = 0.0
            self.marker_pub.publish(marker)
            return
        marker.data = float(observation.quality)
        self.marker_pub.publish(marker)

        # The solve comes back in the marker board's own frame, which yaws with
        # the deck. Publish it in pad ENU instead -- ENU axes translated to the
        # deck origin, deliberately not rotated with it -- so the axes stay
        # gravity-aligned, the gateway's PX4 fallback (a plain subtraction of
        # the deck pose) means the same thing, and no Coriolis term appears in
        # the relative velocity. The deck heading used for the rotation is the
        # one the rover broadcasts; a drone could equally recover it from its
        # own yaw and the yaw this solve already measures.
        deck_yaw = Rotation.from_euler("z", self.deck.yaw)
        position = deck_yaw.apply(np.asarray(observation.position_pad_enu, dtype=float))
        q = np.asarray(observation.quaternion_pad_flu_wxyz, dtype=float)
        attitude = deck_yaw * Rotation.from_quat([q[1], q[2], q[3], q[0]])
        qx, qy, qz, qw = attitude.as_quat()

        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = f"landing_pad_{self.pair_index}"
        pose.pose.position.x = float(position[0])
        pose.pose.position.y = float(position[1])
        pose.pose.position.z = float(position[2])
        pose.pose.orientation.w = float(qw)
        pose.pose.orientation.x = float(qx)
        pose.pose.orientation.y = float(qy)
        pose.pose.orientation.z = float(qz)
        self.pad_pose_pub.publish(pose)

    def _publish_annotated_camera(self, stamp) -> None:
        """Publish the most recent operator view as a standard ROS ``rgb8`` image."""
        image = self.camera.annotated_rgb
        if image is None or image.size == 0:
            return
        image = np.ascontiguousarray(image, dtype=np.uint8)
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = f"landing_camera_{self.pair_index}_optical"
        msg.height = int(image.shape[0])
        msg.width = int(image.shape[1])
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = int(msg.width * 3)
        msg.data = image.tobytes()
        self.marker_image_pub.publish(msg)

    def _publish_actor_camera(self, stamp) -> None:
        """Publish the unmodified actor frame as ROS ``mono8``."""
        image = self.camera.raw_gray
        if image is None or image.size == 0:
            return
        image = np.ascontiguousarray(image, dtype=np.uint8)
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = f"landing_camera_{self.pair_index}_optical"
        msg.height = int(image.shape[0])
        msg.width = int(image.shape[1])
        msg.encoding = "mono8"
        msg.is_bigendian = 0
        msg.step = int(msg.width)
        msg.data = image.tobytes()
        self.actor_image_pub.publish(msg)

    def run(self, pairs=None):
        pairs = list(pairs or (self,))
        self.timeline.play()
        # Accumulate physical time instead of rounding to an integer divider.
        # ZED 2i's 60 Hz period is 4.1667 of the retained 250 Hz physics steps,
        # so a divider would silently run it at either 50 or 62.5 Hz. The
        # accumulator alternates four/five-step intervals for exactly 60 Hz on
        # average without changing the flight dynamics clock.
        physics_dt = float(CONFIG["isaac"]["physics_dt"])
        rendering_dt = self.runtime.rendering_dt
        render_elapsed = 0.0
        previous_rendering_dt = None
        try:
            while (simulation_app.is_running()
                   and not any(pair.stop_sim for pair in pairs)):
                for pair in pairs:
                    if pair.pending_reset is not None:
                        pair._perform_reset()
                startup_rendering = (
                    not all(pair.startup_render_released for pair in pairs)
                    and float(self.world.current_time) < self.runtime.startup_max_sim_s
                    and self.runtime.startup_rendering_dt > rendering_dt)
                active_rendering_dt = (self.runtime.startup_rendering_dt
                                       if startup_rendering else rendering_dt)
                if active_rendering_dt != previous_rendering_dt:
                    render_elapsed = 0.0
                    previous_rendering_dt = active_rendering_dt
                    mode = "PX4 startup" if startup_rendering else "run-time"
                    carb.log_warn(
                        f"Isaac {mode} render rate: {1.0 / active_rendering_dt:.1f} Hz "
                        f"(physics {1.0 / physics_dt:.0f} Hz)")
                render_elapsed += physics_dt
                frame_boundary = render_elapsed + 1e-12 >= active_rendering_dt
                if frame_boundary:
                    render_elapsed -= active_rendering_dt
                # The pad camera only produces an image on a rendered frame, so
                # vision costs rendering even in a headless run.
                render = frame_boundary and (
                    any(pair.vision_enabled for pair in pairs) or not ARGS.headless)
                self.world.step(render=render)
                if frame_boundary:
                    for pair in pairs:
                        pair._publish_environment()
                    if render:
                        primary = pairs[0]
                        vehicles = [pair.vehicle.state.position for pair in pairs]
                        decks = [pair.deck.world_from_pad(np.zeros(3))
                                 for pair in pairs]
                        primary.overlay.update_many(tuple(zip(vehicles, decks)))
                        if len(pairs) > 1:
                            primary.viewport_follower.update_group(
                                vehicles, decks,
                                [pair.deck.yaw for pair in pairs], primary.urban)
                        else:
                            primary.viewport_follower.update(
                                vehicles[0], primary.deck.yaw, primary.urban,
                                deck_position=decks[0])
        except Exception as exc:
            # Isaac's ROS bridge invalidates its context as soon as the process
            # receives the stack's shutdown signal. A publisher can race that
            # teardown by one frame; this is a clean exit, not a simulator crash.
            if "context is invalid" not in str(exc):
                raise
        finally:
            self.timeline.stop()
            simulation_app.close()


def main():
    app = LandingWorld(pair_index=0, pair_count=ARGS.parallel_pairs)
    pairs = [app]
    for index in range(1, ARGS.parallel_pairs):
        pairs.append(LandingWorld(
            pair_index=index, pair_count=ARGS.parallel_pairs, shared=app))
    if len(pairs) > 1:
        offsets = ", ".join(
            f"pair {pair.pair_index}=route {pair.route_phase_fraction:.0%}"
            f" + {pair.world_offset.tolist()} m"
            for pair in pairs)
        carb.log_warn(
            f"Parallel landing world ready with {len(pairs)} UAV/UGV pairs; {offsets}")
    try:
        app.run(pairs)
    except KeyboardInterrupt:
        for pair in pairs:
            pair.stop_sim = True


if __name__ == "__main__":
    main()
