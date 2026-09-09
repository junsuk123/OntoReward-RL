#!/usr/bin/env bash
# Isaac Sim scene runner. The drone swarm does not require the ROS 2 bridge.
#   scripts/run_sim.sh                    GUI
#   scripts/run_sim.sh --headless --seconds 60
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/ros2_env.sh"
PYTHON="${SIMLAB_PYTHON:-/home/j/env_isaaclab/bin/python}"
cd "$HERE/.."
exec env PYTHONUNBUFFERED=1 PYTHONPATH="$(strip_ros_python_paths)" \
    "$PYTHON" -m simlab "$@"
