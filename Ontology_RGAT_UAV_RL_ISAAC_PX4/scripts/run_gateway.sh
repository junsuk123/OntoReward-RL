#!/usr/bin/env bash
set -euo pipefail
workspace_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ros2_runtime=${ASCII_ROS2_WS:-/home/${USER}/.local/share/ontology_rgat_uav_rl/ros2_ws}
unset CYCLONEDDS_URI
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
set +u
source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
set -u
if [[ -f "$ros2_runtime/install/local_setup.bash" ]]; then
  set +u
  source "$ros2_runtime/install/local_setup.bash"
  set -u
else
  printf 'ROS workspace is not built; run scripts/bootstrap_px4_ros2.sh.\n' >&2
  exit 1
fi
exec ros2 run ontology_rgat_px4 ros2_gateway \
  --config "$workspace_root/config/system.yaml" "$@"
