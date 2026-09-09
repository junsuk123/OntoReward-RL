"""ROS 2 bridge: an OmniGraph that exchanges the scene state over DDS.

Published
    /clock                  rosgraph_msgs/Clock       simulation time
    /odom                   nav_msgs/Odometry         UGV odometry
    /tf                     tf2_msgs/TFMessage        odom -> base_link,
                                                      odom -> person_NN,
                                                      base_link -> sensor frames
    /scan                   sensor_msgs/LaserScan     front 2D lidar (Nav2 input)
    /front/rgb              sensor_msgs/Image         front camera
    /front/camera_info      sensor_msgs/CameraInfo
Subscribed
    /cmd_vel                geometry_msgs/Twist       drives the UGV wheels

The people transforms are pushed from Python each step: characters are moved by
the animation-graph runtime, so their USD transforms never change and a prim
based TF publisher would emit their spawn pose forever.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import omni.graph.core as og
from isaacsim.core.nodes.scripts.utils import set_target_prims

from simlab.config.schema import AppConfig, Ros2Config, SensorConfig
from simlab.sim.assets import RobotSpec
from simlab.utils.logging import get_logger

log = get_logger("ros2")

GRAPH_PATH = "/ActionGraph"

#: Extensions required on top of the scene's own set.
ROS2_EXTENSIONS: Sequence[str] = (
    "isaacsim.ros2.bridge",
    "isaacsim.robot.wheeled_robots",
    "omni.graph.action",
    "omni.graph.nodes",
)


def _yaw_to_quat(yaw: float) -> Tuple[float, float, float, float]:
    """Z-axis rotation as an (x, y, z, w) quaternion -- the layout OmniGraph wants."""
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))



def _relative_transform(parent_path: str, child_path: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    """Child pose expressed in the parent's frame, as (xyz, quaternion xyzw)."""
    import omni.usd
    from pxr import Gf, Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    parent_world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(parent_path))
    child_world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(child_path))
    relative = child_world * parent_world.GetInverse()

    translation = relative.ExtractTranslation()
    quat = relative.ExtractRotationQuat()
    imaginary = quat.GetImaginary()
    return (
        (translation[0], translation[1], translation[2]),
        (imaginary[0], imaginary[1], imaginary[2], quat.GetReal()),
    )


def _render_product_path(prim_path: str, resolution: Tuple[int, int]) -> str:
    """Create a render product for a camera or RTX lidar prim."""
    import omni.replicator.core as rep

    product = rep.create.render_product(prim_path, resolution, name="simlab")
    return product.path


class Ros2Bridge:
    """Owns the action graph and the per-step attribute writes it needs."""

    def __init__(self, cfg: Ros2Config, people_count: int) -> None:
        self.cfg = cfg
        self.people_count = people_count if cfg.publish_people_tf else 0
        self._person_translation: List[og.Attribute] = []
        self._person_rotation: List[og.Attribute] = []

    # -- construction ------------------------------------------------------
    def build(
        self,
        robot_prim_path: str,
        articulation_prim_path: str,
        chassis_prim_path: str,
        wheel_joint_names: Sequence[str],
        spec: RobotSpec,
        app_cfg: AppConfig,
    ) -> None:
        """Create the graph. Call after the UGV articulation is initialized."""
        cfg = self.cfg
        keys = og.Controller.Keys

        nodes = [
            ("Tick", "omni.graph.action.OnPlaybackTick"),
            ("SimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
            ("Context", "isaacsim.ros2.bridge.ROS2Context"),
            ("PubClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
            ("SubTwist", "isaacsim.ros2.bridge.ROS2SubscribeTwist"),
            ("BreakLinear", "omni.graph.nodes.BreakVector3"),
            ("BreakAngular", "omni.graph.nodes.BreakVector3"),
            ("DiffDrive", "isaacsim.robot.wheeled_robots.DifferentialController"),
            ("ArtController", "isaacsim.core.nodes.IsaacArticulationController"),
            ("Odometry", "isaacsim.core.nodes.IsaacComputeOdometry"),
            ("PubOdom", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
            ("PubBaseTf", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
        ]
        connect = [
            # clock
            ("Tick.outputs:tick", "PubClock.inputs:execIn"),
            ("Context.outputs:context", "PubClock.inputs:context"),
            ("SimTime.outputs:simulationTime", "PubClock.inputs:timeStamp"),
            # cmd_vel -> wheels
            ("Tick.outputs:tick", "SubTwist.inputs:execIn"),
            ("Context.outputs:context", "SubTwist.inputs:context"),
            ("SubTwist.outputs:linearVelocity", "BreakLinear.inputs:tuple"),
            ("SubTwist.outputs:angularVelocity", "BreakAngular.inputs:tuple"),
            ("BreakLinear.outputs:x", "DiffDrive.inputs:linearVelocity"),
            ("BreakAngular.outputs:z", "DiffDrive.inputs:angularVelocity"),
            # Tick-driven rather than message-driven, so the last command is
            # held instead of the wheels dropping to zero between messages.
            ("Tick.outputs:tick", "DiffDrive.inputs:execIn"),
            ("Tick.outputs:tick", "ArtController.inputs:execIn"),
            ("DiffDrive.outputs:velocityCommand", "ArtController.inputs:velocityCommand"),
            # odometry
            ("Tick.outputs:tick", "Odometry.inputs:execIn"),
            ("Odometry.outputs:execOut", "PubOdom.inputs:execIn"),
            ("Context.outputs:context", "PubOdom.inputs:context"),
            ("SimTime.outputs:simulationTime", "PubOdom.inputs:timeStamp"),
            ("Odometry.outputs:position", "PubOdom.inputs:position"),
            ("Odometry.outputs:orientation", "PubOdom.inputs:orientation"),
            ("Odometry.outputs:linearVelocity", "PubOdom.inputs:linearVelocity"),
            ("Odometry.outputs:angularVelocity", "PubOdom.inputs:angularVelocity"),
            # odom -> base_link
            ("Odometry.outputs:execOut", "PubBaseTf.inputs:execIn"),
            ("Context.outputs:context", "PubBaseTf.inputs:context"),
            ("SimTime.outputs:simulationTime", "PubBaseTf.inputs:timeStamp"),
            ("Odometry.outputs:position", "PubBaseTf.inputs:translation"),
            ("Odometry.outputs:orientation", "PubBaseTf.inputs:rotation"),
        ]
        values = [
            ("Context.inputs:useDomainIDEnvVar", cfg.domain_id is None),
            ("PubClock.inputs:topicName", cfg.clock_topic),
            ("PubClock.inputs:nodeNamespace", cfg.node_namespace),
            ("SubTwist.inputs:topicName", cfg.cmd_vel_topic),
            ("SubTwist.inputs:nodeNamespace", cfg.node_namespace),
            ("DiffDrive.inputs:wheelRadius", spec.wheel_radius),
            ("DiffDrive.inputs:wheelDistance", spec.wheel_base),
            ("DiffDrive.inputs:maxLinearSpeed", cfg.max_linear_speed),
            ("DiffDrive.inputs:maxAngularSpeed", cfg.max_angular_speed),
            ("DiffDrive.inputs:dt", app_cfg.physics_dt),
            ("ArtController.inputs:jointNames", list(wheel_joint_names)),
            ("PubOdom.inputs:topicName", cfg.odom_topic),
            ("PubOdom.inputs:nodeNamespace", cfg.node_namespace),
            ("PubOdom.inputs:odomFrameId", cfg.odom_frame),
            ("PubOdom.inputs:chassisFrameId", cfg.base_frame),
            ("PubBaseTf.inputs:topicName", cfg.tf_topic),
            ("PubBaseTf.inputs:nodeNamespace", cfg.node_namespace),
            ("PubBaseTf.inputs:parentFrameId", cfg.odom_frame),
            ("PubBaseTf.inputs:childFrameId", cfg.base_frame),
        ]
        if cfg.domain_id is not None:
            values.append(("Context.inputs:domain_id", cfg.domain_id))

        # One raw-TF publisher per person; values are pushed in update().
        for index in range(self.people_count):
            node = f"PubPersonTf{index:02d}"
            frame = cfg.person_frame(index)
            nodes.append((node, "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"))
            connect += [
                ("Tick.outputs:tick", f"{node}.inputs:execIn"),
                ("Context.outputs:context", f"{node}.inputs:context"),
                ("SimTime.outputs:simulationTime", f"{node}.inputs:timeStamp"),
            ]
            values += [
                (f"{node}.inputs:topicName", cfg.tf_topic),
                (f"{node}.inputs:nodeNamespace", cfg.node_namespace),
                (f"{node}.inputs:parentFrameId", cfg.odom_frame),
                (f"{node}.inputs:childFrameId", frame),
            ]

        sensors = cfg.sensors

        # Sensor frames: publish the fixed base_link -> sensor offsets read
        # straight out of the USD, so the frame names stay ours rather than
        # inheriting prim names.
        for label, enabled, rel_prim, frame in (
            ("Lidar", sensors.lidar_enabled, sensors.lidar_prim, sensors.lidar_frame),
            ("Camera", sensors.camera_enabled, sensors.camera_prim, sensors.camera_frame),
        ):
            if not enabled:
                continue
            node = f"Pub{label}Tf"
            nodes.append((node, "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"))
            connect += [
                ("Tick.outputs:tick", f"{node}.inputs:execIn"),
                ("Context.outputs:context", f"{node}.inputs:context"),
                ("SimTime.outputs:simulationTime", f"{node}.inputs:timeStamp"),
            ]
            values += [
                (f"{node}.inputs:topicName", cfg.tf_topic),
                (f"{node}.inputs:nodeNamespace", cfg.node_namespace),
                (f"{node}.inputs:parentFrameId", cfg.base_frame),
                (f"{node}.inputs:childFrameId", frame),
            ]

        if sensors.lidar_enabled:
            lidar_prim = f"{robot_prim_path}/{sensors.lidar_prim}"
            nodes.append(("PubScan", "isaacsim.ros2.bridge.ROS2RtxLidarHelper"))
            connect += [
                ("Tick.outputs:tick", "PubScan.inputs:execIn"),
                ("Context.outputs:context", "PubScan.inputs:context"),
            ]
            values += [
                # An RTX lidar renders into a 1x1 product; the scan comes from
                # the sensor pattern, not the texture size.
                ("PubScan.inputs:renderProductPath", _render_product_path(lidar_prim, (1, 1))),
                ("PubScan.inputs:topicName", sensors.lidar_topic),
                ("PubScan.inputs:nodeNamespace", cfg.node_namespace),
                ("PubScan.inputs:frameId", sensors.lidar_frame),
                ("PubScan.inputs:type", "laser_scan"),
            ]

        if sensors.camera_enabled:
            camera_prim = f"{robot_prim_path}/{sensors.camera_prim}"
            camera_product = _render_product_path(
                camera_prim, (sensors.camera_width, sensors.camera_height)
            )
            nodes += [
                ("PubImage", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("PubCameraInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
            ]
            connect += [
                ("Tick.outputs:tick", "PubImage.inputs:execIn"),
                ("Context.outputs:context", "PubImage.inputs:context"),
                ("Tick.outputs:tick", "PubCameraInfo.inputs:execIn"),
                ("Context.outputs:context", "PubCameraInfo.inputs:context"),
            ]
            values += [
                ("PubImage.inputs:renderProductPath", camera_product),
                ("PubImage.inputs:topicName", sensors.camera_topic),
                ("PubImage.inputs:nodeNamespace", cfg.node_namespace),
                ("PubImage.inputs:frameId", sensors.camera_frame),
                ("PubImage.inputs:type", "rgb"),
                ("PubCameraInfo.inputs:renderProductPath", camera_product),
                ("PubCameraInfo.inputs:topicName", sensors.camera_info_topic),
                ("PubCameraInfo.inputs:nodeNamespace", cfg.node_namespace),
                ("PubCameraInfo.inputs:frameId", sensors.camera_frame),
            ]

        og.Controller.edit(
            {"graph_path": GRAPH_PATH, "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: nodes,
                keys.CONNECT: connect,
                keys.SET_VALUES: values,
            },
        )

        # Sensor offsets are static, so they are written once, after the graph
        # exists but before it first ticks.
        for label, enabled, rel_prim in (
            ("Lidar", sensors.lidar_enabled, sensors.lidar_prim),
            ("Camera", sensors.camera_enabled, sensors.camera_prim),
        ):
            if not enabled:
                continue
            translation, rotation = _relative_transform(
                chassis_prim_path, f"{robot_prim_path}/{rel_prim}"
            )
            og.Controller.set(
                og.Controller.attribute(f"{GRAPH_PATH}/Pub{label}Tf.inputs:translation"),
                list(translation),
            )
            og.Controller.set(
                og.Controller.attribute(f"{GRAPH_PATH}/Pub{label}Tf.inputs:rotation"),
                list(rotation),
            )

        # Relationship inputs cannot go through SET_VALUES.
        set_target_prims(
            primPath=f"{GRAPH_PATH}/ArtController",
            targetPrimPaths=[articulation_prim_path],
            inputName="inputs:targetPrim",
        )
        # Must be a rigid body, not the wrapper Xform.
        set_target_prims(
            primPath=f"{GRAPH_PATH}/Odometry",
            targetPrimPaths=[chassis_prim_path],
            inputName="inputs:chassisPrim",
        )

        self._person_translation = [
            og.Controller.attribute(f"{GRAPH_PATH}/PubPersonTf{i:02d}.inputs:translation")
            for i in range(self.people_count)
        ]
        self._person_rotation = [
            og.Controller.attribute(f"{GRAPH_PATH}/PubPersonTf{i:02d}.inputs:rotation")
            for i in range(self.people_count)
        ]

        published = [cfg.clock_topic, cfg.odom_topic, cfg.tf_topic]
        if sensors.lidar_enabled:
            published.append(sensors.lidar_topic)
        if sensors.camera_enabled:
            published += [sensors.camera_topic, sensors.camera_info_topic]
        log(
            f"graph ready: pub {', '.join(published)} "
            f"({self.people_count} people frames) | sub {cfg.cmd_vel_topic}"
        )

    # -- runtime -----------------------------------------------------------
    def update(self, people_xy: Sequence[Tuple[float, float]], yaws: Sequence[float] | None = None) -> None:
        """Push the current people poses into their TF publisher nodes."""
        for index, (x, y) in enumerate(people_xy[: self.people_count]):
            yaw = yaws[index] if yaws is not None and index < len(yaws) else 0.0
            og.Controller.set(self._person_translation[index], [float(x), float(y), 0.0])
            og.Controller.set(self._person_rotation[index], list(_yaw_to_quat(yaw)))
