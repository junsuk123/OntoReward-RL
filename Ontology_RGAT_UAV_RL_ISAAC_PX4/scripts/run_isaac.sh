#!/usr/bin/env bash
set -euo pipefail
workspace_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
config_path=${1:-$workspace_root/config/system.yaml}
if [[ $# -gt 0 ]]; then shift; fi
if [[ $config_path != /* ]]; then config_path="$workspace_root/$config_path"; fi
isaacsim_path=${ISAACSIM_PATH:-}
isaac_python=${ISAACSIM_PYTHON:-${isaacsim_path:+$isaacsim_path/python.sh}}
if [[ -z "$isaac_python" || ! -x "$isaac_python" ]]; then
  printf 'Isaac Python not found. Set ISAACSIM_PATH or ISAACSIM_PYTHON.\n' >&2
  exit 1
fi
headless_arg=()
if [[ ${HEADLESS:-0} == 1 ]]; then headless_arg=(--headless); fi
unset ROS_VERSION ROS_PYTHON_VERSION AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH PYTHONPATH CYCLONEDDS_URI
export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
if [[ -n "$isaacsim_path" ]]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$isaacsim_path/exts/isaacsim.ros2.bridge/humble/lib"
fi
exec "$isaac_python" "$workspace_root/isaac_sim/landing_world.py" \
  --config "$config_path" "${headless_arg[@]}" "$@"
