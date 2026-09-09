#!/usr/bin/env bash
set -euo pipefail

workspace_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ros2_runtime=${ASCII_ROS2_WS:-/home/${USER}/.local/share/ontology_rgat_uav_rl/ros2_ws}

printf '%s\n' '=== Processes ==='
ps -eo pid,etime,state,%cpu,%mem,cmd | \
  grep -E '[l]anding_world.py|[M]icroXRCEAgent|px4_sitl_default/bin/[p]x4|[r]os2_gateway' || true

printf '%s\n' '=== Ports ==='
ss -lntup | grep -E ':4560|:8888|:14650|:14651' || true

if [[ ! -f "$ros2_runtime/install/local_setup.bash" ]]; then
  printf 'ROS 2 runtime not built: %s\n' "$ros2_runtime" >&2
  exit 1
fi

unset CYCLONEDDS_URI
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}
set +u
source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
source "$ros2_runtime/install/local_setup.bash"
set -u

printf '%s\n' '=== ROS 2 data-rate check (6 seconds) ==='
timeout 6 ros2 topic hz /fmu/out/vehicle_odometry || true

printf '%s\n' '=== MATLAB gateway state ==='
python3 "$workspace_root/tools/protocol_probe.py" state
