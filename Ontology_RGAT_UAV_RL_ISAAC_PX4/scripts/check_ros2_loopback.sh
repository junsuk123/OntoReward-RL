#!/usr/bin/env bash
set -euo pipefail
workspace_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ros2_runtime=${ASCII_ROS2_WS:-/home/${USER}/.local/share/ontology_rgat_uav_rl/ros2_ws}
unset CYCLONEDDS_URI
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
set +u
source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
set -u
if [[ ! -f "$ros2_runtime/install/local_setup.bash" ]]; then
  printf 'Build ros2_ws first with scripts/bootstrap_px4_ros2.sh.\n' >&2
  exit 1
fi
set +u
source "$ros2_runtime/install/local_setup.bash"
set -u
# The probe publishes mock PX4 telemetry. A live PX4 publishes the same topics
# and wins, which shows up as a baffling assertion failure rather than a clear
# conflict, so refuse to run alongside the real stack.
if pgrep -f 'px4_sitl_default/bin/px4' >/dev/null 2>&1; then
  printf 'A live PX4 SITL is running; stop Isaac before the loopback check.\n' >&2
  exit 1
fi
log=/tmp/ontology_rgat_ros_gateway.log
# setsid so the trap can take down ros2's gateway child too; killing only the
# wrapper leaves the real gateway holding the UDP port for the next check.
setsid ros2 run ontology_rgat_px4 ros2_gateway \
  --config "$workspace_root/config/system.yaml" --target sitl >"$log" 2>&1 &
gateway_pid=$!
trap 'kill -TERM -"$gateway_pid" 2>/dev/null || true' EXIT
sleep 2
if ! kill -0 "$gateway_pid" 2>/dev/null; then
  printf 'Gateway failed to start; stop any other gateway on the UDP port.\n' >&2
  cat "$log" >&2
  exit 1
fi
python3 "$workspace_root/tools/ros2_loopback_probe.py"
