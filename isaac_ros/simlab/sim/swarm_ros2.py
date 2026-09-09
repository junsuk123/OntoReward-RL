"""ROS 2 export for the drone swarm, camera rig, and RViz pipeline."""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence

import omni.graph.core as og
import omni.usd

from simlab.algorithms.swarm import DroneState
from simlab.config.schema import Ros2Config
from simlab.sim.cameras import CameraRig
from simlab.utils.logging import get_logger

log = get_logger("ros2")
GRAPH_PATH = "/SwarmRosActionGraph"


def _render_product(camera_path: str, resolution, name: str) -> str:
    import omni.replicator.core as rep

    return rep.create.render_product(camera_path, resolution, name=name).path


def _world_pose(path: str):
    matrix = omni.usd.get_world_transform_matrix(
        omni.usd.get_context().get_stage().GetPrimAtPath(path)
    )
    translation = matrix.ExtractTranslation()
    quaternion = matrix.ExtractRotationQuat()
    imaginary = quaternion.GetImaginary()
    return (
        [float(translation[0]), float(translation[1]), float(translation[2])],
        [float(imaginary[0]), float(imaginary[1]), float(imaginary[2]), float(quaternion.GetReal())],
    )


class SwarmRos2Bridge:
    """Publish camera images, camera info, clock, and map-frame drone TFs.

    Also owns the camera gates the sensor-dropout scenario uses: a blackout is
    the image publisher going quiet, not a flag on an image that still arrives,
    so every consumer downstream loses the observation exactly as it would if
    the link had dropped.
    """

    def __init__(self, cfg: Ros2Config) -> None:
        self.cfg = cfg
        self._translations: List[og.Attribute] = []
        self._rotations: List[og.Attribute] = []
        self._camera_gates: Dict[str, og.Attribute] = {}
        self._gate_state: Dict[str, bool] = {}

    def build(self, states: Sequence[DroneState], camera_rig: CameraRig | None) -> None:
        cfg = self.cfg
        keys = og.Controller.Keys
        nodes = [
            ("Tick", "omni.graph.action.OnPlaybackTick"),
            ("SimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
            ("Context", "isaacsim.ros2.bridge.ROS2Context"),
            ("PubClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
        ]
        connect = [
            ("Tick.outputs:tick", "PubClock.inputs:execIn"),
            ("Context.outputs:context", "PubClock.inputs:context"),
            ("SimTime.outputs:simulationTime", "PubClock.inputs:timeStamp"),
        ]
        values = [
            ("Context.inputs:useDomainIDEnvVar", cfg.domain_id is None),
            ("PubClock.inputs:topicName", cfg.clock_topic),
            ("PubClock.inputs:nodeNamespace", cfg.node_namespace),
        ]
        if cfg.domain_id is not None:
            values.append(("Context.inputs:domain_id", cfg.domain_id))

        if cfg.publish_swarm_tf:
            for index, state in enumerate(states):
                node = f"PubDroneTf{index:02d}"
                nodes.append((node, "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"))
                connect += [
                    ("Tick.outputs:tick", f"{node}.inputs:execIn"),
                    ("Context.outputs:context", f"{node}.inputs:context"),
                    ("SimTime.outputs:simulationTime", f"{node}.inputs:timeStamp"),
                ]
                values += [
                    (f"{node}.inputs:topicName", cfg.tf_topic),
                    (f"{node}.inputs:nodeNamespace", cfg.node_namespace),
                    (f"{node}.inputs:parentFrameId", cfg.map_frame),
                    (f"{node}.inputs:childFrameId", state.name),
                ]

        camera_topics = []
        if camera_rig is not None:
            for index, descriptor in enumerate(camera_rig.descriptors.values()):
                tf_node = f"PubCameraTf{index:02d}"
                nodes.append((tf_node, "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"))
                connect += [
                    ("Tick.outputs:tick", f"{tf_node}.inputs:execIn"),
                    ("Context.outputs:context", f"{tf_node}.inputs:context"),
                    ("SimTime.outputs:simulationTime", f"{tf_node}.inputs:timeStamp"),
                ]
                values += [
                    (f"{tf_node}.inputs:topicName", cfg.tf_topic),
                    (f"{tf_node}.inputs:nodeNamespace", cfg.node_namespace),
                    (f"{tf_node}.inputs:parentFrameId", cfg.map_frame),
                    (f"{tf_node}.inputs:childFrameId", descriptor.sensor_id),
                ]
                if not cfg.publish_swarm_cameras:
                    continue
                width = max(1, round(descriptor.resolution[0] * cfg.swarm_image_scale))
                height = max(1, round(descriptor.resolution[1] * cfg.swarm_image_scale))
                product = _render_product(
                    descriptor.prim_path, (width, height), f"simlab_{descriptor.sensor_id}"
                )
                image_node = f"PubImage{index:02d}"
                info_node = f"PubCameraInfo{index:02d}"
                nodes += [
                    (image_node, "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    (info_node, "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
                ]
                for node in (image_node, info_node):
                    connect += [
                        ("Tick.outputs:tick", f"{node}.inputs:execIn"),
                        ("Context.outputs:context", f"{node}.inputs:context"),
                    ]
                topic_root = f"simlab/{descriptor.sensor_id}"
                values += [
                    (f"{image_node}.inputs:renderProductPath", product),
                    (f"{image_node}.inputs:topicName", f"{topic_root}/image"),
                    (f"{image_node}.inputs:nodeNamespace", cfg.node_namespace),
                    (f"{image_node}.inputs:frameId", descriptor.sensor_id),
                    (f"{image_node}.inputs:type", "rgb"),
                    (f"{info_node}.inputs:renderProductPath", product),
                    (f"{info_node}.inputs:topicName", f"{topic_root}/camera_info"),
                    (f"{info_node}.inputs:nodeNamespace", cfg.node_namespace),
                    (f"{info_node}.inputs:frameId", descriptor.sensor_id),
                ]
                camera_topics.append(f"/{topic_root}/image")

        og.Controller.edit(
            {"graph_path": GRAPH_PATH, "evaluator_name": "execution"},
            {keys.CREATE_NODES: nodes, keys.CONNECT: connect, keys.SET_VALUES: values},
        )

        self._translations = []
        self._rotations = []
        if cfg.publish_swarm_tf:
            for index in range(len(states)):
                self._translations.append(
                    og.Controller.attribute(f"{GRAPH_PATH}/PubDroneTf{index:02d}.inputs:translation")
                )
                self._rotations.append(
                    og.Controller.attribute(f"{GRAPH_PATH}/PubDroneTf{index:02d}.inputs:rotation")
                )

        if camera_rig is not None and cfg.publish_swarm_cameras:
            self._bind_camera_gates(camera_rig)

        if camera_rig is not None:
            for index, descriptor in enumerate(camera_rig.descriptors.values()):
                translation, rotation = _world_pose(descriptor.prim_path)
                og.Controller.set(
                    og.Controller.attribute(f"{GRAPH_PATH}/PubCameraTf{index:02d}.inputs:translation"),
                    translation,
                )
                og.Controller.set(
                    og.Controller.attribute(f"{GRAPH_PATH}/PubCameraTf{index:02d}.inputs:rotation"),
                    rotation,
                )
        self.update(states)
        log(f"swarm graph ready: {len(states)} TF frames; camera topics={camera_topics}")

    def _bind_camera_gates(self, camera_rig: CameraRig) -> None:
        """Look up each image publisher's enable input, if the node exposes one."""
        for index, descriptor in enumerate(camera_rig.descriptors.values()):
            try:
                self._camera_gates[descriptor.sensor_id] = og.Controller.attribute(
                    f"{GRAPH_PATH}/PubImage{index:02d}.inputs:enabled"
                )
                self._gate_state[descriptor.sensor_id] = True
            except Exception as exc:  # attribute name differs across Kit versions
                log(f"camera gate unavailable for {descriptor.sensor_id}: {exc}")

    @property
    def camera_gating_available(self) -> bool:
        return bool(self._camera_gates)

    def apply_sensor_gates(self, observing: Mapping[str, bool]) -> None:
        """Enable or silence each image publisher. Only writes on a change."""
        for sensor_id, gate in self._camera_gates.items():
            wanted = bool(observing.get(sensor_id, True))
            if self._gate_state.get(sensor_id) == wanted:
                continue
            og.Controller.set(gate, wanted)
            self._gate_state[sensor_id] = wanted

    def update(self, states: Sequence[DroneState]) -> None:
        for index, state in enumerate(states[: len(self._translations)]):
            og.Controller.set(self._translations[index], [float(v) for v in state.position])
            half = 0.5 * state.yaw
            og.Controller.set(
                self._rotations[index], [0.0, 0.0, math.sin(half), math.cos(half)]
            )

