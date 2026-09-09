"""Bring up Isaac Sim + swarm perception prototype + RViz2.

    ros2 launch launch/simlab.launch.py                       # swarm + tracker + RViz2
    ros2 launch launch/simlab.launch.py pipeline:=false       # simulation/RViz only
    ros2 launch launch/simlab.launch.py headless:=true seconds:=60
    ros2 launch launch/simlab.launch.py rviz:=false

Exactly one thing should publish /cmd_vel, so turning `nav2` on turns the
simlab controller node off unless `controller` is set explicitly.

Each process is started through its script in ``scripts/`` because they need
different environments: Isaac Sim runs on Python 3.11 with ROS 2's Python paths
stripped, while the controller, Nav2 and RViz run on the system Python 3.10.
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"


def generate_launch_description() -> LaunchDescription:
    headless = LaunchConfiguration("headless")
    seconds = LaunchConfiguration("seconds")
    config = LaunchConfiguration("config")
    rviz = LaunchConfiguration("rviz")
    nav2 = LaunchConfiguration("nav2")
    controller = LaunchConfiguration("controller")
    pipeline = LaunchConfiguration("pipeline")
    yolo = LaunchConfiguration("yolo")

    args = [
        DeclareLaunchArgument("headless", default_value="false",
                              description="run Isaac Sim without its GUI"),
        DeclareLaunchArgument("seconds", default_value="0",
                              description="simulation duration; 0 = until closed"),
        DeclareLaunchArgument("config", default_value="configs/default.yaml",
                              description="scene YAML, shared by the sim and the controller"),
        DeclareLaunchArgument("rviz", default_value="true",
                              description="start RViz2 with swarm tracking and camera views"),
        DeclareLaunchArgument("pipeline", default_value="true",
                              description="run satellite-label/front-association prototype"),
        DeclareLaunchArgument("yolo", default_value="true",
                              description="run YOLO RGB inference and publish annotated images"),
        DeclareLaunchArgument("nav2", default_value="false",
                              description="drive with Nav2 instead of the simlab controller"),
        DeclareLaunchArgument(
            "controller", default_value="false",
            description="run the legacy UGV ROS 2 controller node"),
    ]

    sim = ExecuteProcess(
        name="isaac_sim",
        cmd=[
            str(SCRIPTS / "run_sim.sh"),
            "--config", config,
            "--seconds", seconds,
            "--set", "ros2.enabled=true",
            # --set takes a YAML value, so the launch argument passes straight through
            # instead of needing a present/absent --headless flag.
            "--set", ["app.headless=", headless],
        ],
        output="screen",
        shell=False,
    )

    algorithm = ExecuteProcess(
        name="simlab_controller",
        cmd=[str(SCRIPTS / "run_controller.sh")],
        additional_env={"SIMLAB_CONFIG": config},
        output="screen",
        shell=False,
        condition=IfCondition(controller),
    )

    swarm_pipeline = ExecuteProcess(
        name="swarm_pipeline",
        cmd=[str(SCRIPTS / "run_swarm_pipeline.sh")],
        additional_env={"SIMLAB_CONFIG": config},
        output="screen",
        shell=False,
        condition=IfCondition(pipeline),
    )

    yolo_detector = ExecuteProcess(
        name="yolo_detector",
        cmd=[str(SCRIPTS / "run_yolo_detector.sh")],
        additional_env={"SIMLAB_CONFIG": config},
        output="screen",
        shell=False,
        condition=IfCondition(yolo),
    )

    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(Path(__file__).parent / "nav2.launch.py")),
        condition=IfCondition(nav2),
    )

    rviz2 = ExecuteProcess(
        name="rviz2",
        cmd=[str(SCRIPTS / "run_rviz.sh")],
        output="log",
        shell=False,
        condition=IfCondition(rviz),
    )

    stop_all_when_sim_closes = RegisterEventHandler(
        OnProcessExit(
            target_action=sim,
            on_exit=[EmitEvent(event=Shutdown(reason="Isaac Sim closed"))],
        )
    )

    return LaunchDescription(
        args + [sim, swarm_pipeline, yolo_detector, algorithm, navigation, rviz2, stop_all_when_sim_closes]
    )
