#!/usr/bin/env bash
# The algorithm, as a ROS 2 node. Runs on the system Python 3.10 (rclpy), not
# on Isaac Sim's 3.11 -- simlab.algorithms is plain Python so both can import it.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
source "$HERE/ros2_env.sh"
cd "$ROOT"
exec env PYTHONUNBUFFERED=1 PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -m simlab.ros.controller_node --ros-args \
    -p use_sim_time:=true \
    -p config:="${SIMLAB_CONFIG:-configs/default.yaml}" \
    "$@"
