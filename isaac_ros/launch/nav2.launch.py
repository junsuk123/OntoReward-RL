"""Nav2 against the Isaac Sim scene, without a static map.

`map` is pinned to `odom` by a static transform because the ground plane has no
geometry for AMCL or a SLAM scan matcher to localize against. Both costmaps roll
with the robot and are fed by /scan, so the walking people appear as moving
obstacles. Replace the static publisher with amcl + map_server (or slam_toolbox)
once the scene has real structure.

    ros2 launch launch/nav2.launch.py

Requires the Nav2 servers:

    sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup
"""

import ctypes.util
import sys
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARAMS = str(ROOT / "configs" / "nav2_params.yaml")

#: Brought up together by the lifecycle manager, in this order.
LIFECYCLE_NODES = [
    "controller_server",
    "planner_server",
    "behavior_server",
    "bt_navigator",
    "velocity_smoother",
]


def check_dependencies() -> None:
    """Fail loudly on the one missing library Nav2 does not report clearly.

    nav2_lifecycle_manager links libdiagnostic_updater.so, but older
    ros-humble-diagnostic-updater builds (<= 4.0.6) shipped only the Python
    package. apt is happy -- the dependency is unversioned -- and the manager
    then dies with a bare `exit code 127`, leaving every server stuck in
    `unconfigured` with no hint as to why.
    """
    if ctypes.util.find_library("diagnostic_updater"):
        return
    print(
        "\n[nav2.launch.py] libdiagnostic_updater.so not found.\n"
        "  nav2_lifecycle_manager needs it and will exit 127 without it,\n"
        "  leaving the Nav2 servers unconfigured.\n"
        "  Fix:  sudo apt install --only-upgrade ros-humble-diagnostic-updater\n",
        file=sys.stderr,
    )


def generate_launch_description() -> LaunchDescription:
    check_dependencies()

    params = LaunchConfiguration("params_file")
    use_sim_time = LaunchConfiguration("use_sim_time")
    common = [{"use_sim_time": use_sim_time}, params]

    args = [
        DeclareLaunchArgument("params_file", default_value=DEFAULT_PARAMS,
                              description="Nav2 parameter YAML"),
        DeclareLaunchArgument("use_sim_time", default_value="true",
                              description="follow /clock from Isaac Sim"),
    ]

    # Stand-in for localization: map and odom coincide.
    map_to_odom = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="map_to_odom",
        arguments=["0", "0", "0", "0", "0", "0", "map", "odom"],
        parameters=[{"use_sim_time": use_sim_time}],
        output="log",
    )

    servers = [
        Node(package="nav2_controller", executable="controller_server",
             name="controller_server", parameters=common, output="screen",
             remappings=[("cmd_vel", "cmd_vel_nav")]),
        Node(package="nav2_planner", executable="planner_server",
             name="planner_server", parameters=common, output="screen"),
        Node(package="nav2_behaviors", executable="behavior_server",
             name="behavior_server", parameters=common, output="screen"),
        Node(package="nav2_bt_navigator", executable="bt_navigator",
             name="bt_navigator", parameters=common, output="screen"),
        # Smooths controller output and is the only publisher on /cmd_vel.
        Node(package="nav2_velocity_smoother", executable="velocity_smoother",
             name="velocity_smoother", parameters=common, output="screen",
             remappings=[("cmd_vel", "cmd_vel_nav"), ("cmd_vel_smoothed", "cmd_vel")]),
        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager",
             name="lifecycle_manager_navigation", output="screen",
             parameters=[{
                 "use_sim_time": use_sim_time,
                 "autostart": True,
                 "node_names": LIFECYCLE_NODES,
             }]),
    ]

    return LaunchDescription(args + [map_to_odom] + servers)
