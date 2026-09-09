#!/usr/bin/env bash
# The live landing view. Start this beside the learner; the learner publishes
# /landing_rl/* and the simulator publishes the deck pose and the marker solve.
set -euo pipefail
workspace_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ascii_ws=${ASCII_ROS2_WS:-$HOME/.local/share/ontology_rgat_uav_rl/ros2_ws}

# Everything that touches /fmu/* must speak Fast DDS, because that is what the
# XRCE-DDS agent speaks. A shell with a different default sees the topics but
# reads nothing from them.
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
unset CYCLONEDDS_URI

# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
if [ -f "$ascii_ws/install/setup.bash" ]; then
  # shellcheck disable=SC1091
  source "$ascii_ws/install/setup.bash"
fi

exec rviz2 -d "$workspace_root/rviz/ontology_rgat.rviz" "$@"
