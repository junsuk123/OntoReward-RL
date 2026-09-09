#!/usr/bin/env bash
# Run a learner entry point with ROS 2 on the path, so the RViz 2 live view is
# available. Without this the learner still runs -- it prints why the view is
# off and carries on -- but a training run you wanted to watch is not the place
# to discover that rclpy was not importable.
#
#   ./scripts/run_learner.sh run_pipeline.py --mode full
#   ./scripts/run_learner.sh run_episode.py --policy proposed
set -euo pipefail
workspace_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ascii_ws=${ASCII_ROS2_WS:-$HOME/.local/share/ontology_rgat_uav_rl/ros2_ws}

if [ "$#" -lt 1 ]; then
  printf 'usage: %s <run_pipeline.py|run_episode.py|...> [args]\n' "$0" >&2
  exit 2
fi

# Everything that touches /fmu/* must speak Fast DDS, because that is what the
# XRCE-DDS agent speaks.
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
unset CYCLONEDDS_URI

# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
if [ -f "$ascii_ws/install/setup.bash" ]; then
  # shellcheck disable=SC1091
  source "$ascii_ws/install/setup.bash"
fi

entry=$1
shift
exec python3 "$workspace_root/python/$entry" "$@"
