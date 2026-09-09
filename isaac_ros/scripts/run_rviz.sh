#!/usr/bin/env bash
# RViz2 with the simlab layout, on simulation time from /clock.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
source "$HERE/ros2_env.sh"
strip_snap_gtk_path
exec rviz2 -d "$ROOT/rviz/simlab.rviz" --ros-args -p use_sim_time:=true "$@"
