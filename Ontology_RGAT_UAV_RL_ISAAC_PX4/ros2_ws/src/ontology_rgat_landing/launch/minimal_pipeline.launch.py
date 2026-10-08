"""Minimal-observation Ontology -> R-GAT pipeline.

    ros2 launch ontology_rgat_landing minimal_pipeline.launch.py \
        use_sim_time:=true checkpoint:=/path/policy.pt

The supervised setpoint is published on <ns>/safety/acceleration_setpoint and
is NOT sent to PX4 by this launch file.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

ARGS = {
    "namespace": "/landing_uav0",
    "px4_namespace": "/fmu",
    "use_sim_time": "true",
    "checkpoint": "",
    "profile": "",
    "with_detector": "true",
    "with_px4_bridge": "true",
    # Isaac publishes sim time only as <ns>/simulation/clock (JSON); bridge it
    # to /clock. Leave false when something else already provides /clock.
    "sim_clock_from_isaac": "true",
}


def generate_launch_description():
    from launch.conditions import IfCondition
    cfg = {name: LaunchConfiguration(name) for name in ARGS}
    sim = ParameterValue(cfg["use_sim_time"], value_type=bool)
    common = {"namespace": cfg["namespace"], "use_sim_time": sim}
    detector_params = dict(common)
    detector_params["use_camera_info"] = False
    detector_params["profile"] = cfg["profile"]

    def node(executable, params, condition=None):
        return Node(package="ontology_rgat_landing", executable=executable,
                    name=executable, output="screen", parameters=[params],
                    condition=condition)

    nodes = [
        node("sim_clock_bridge", {"namespace": cfg["namespace"]},
             IfCondition(cfg["sim_clock_from_isaac"])),
        node("aruco_pad_detector", detector_params, IfCondition(cfg["with_detector"])),
        node("px4_localization_bridge", {**common, "px4_namespace": cfg["px4_namespace"]},
             IfCondition(cfg["with_px4_bridge"])),
        node("observation_assembler", common),
        node("ontology_node", common),
        node("rgat_policy_node", {**common, "checkpoint": cfg["checkpoint"]}),
        node("safety_supervisor_node", common),
    ]
    return LaunchDescription(
        [DeclareLaunchArgument(name, default_value=value) for name, value in ARGS.items()]
        + nodes)
