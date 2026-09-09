#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
source "$HERE/ros2_env.sh"
PYTHON_BIN="$("$HERE/setup_perception.sh" | tail -n 1)"
cd "$ROOT"
exec env PYTHONUNBUFFERED=1 PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" -m simlab.ros.yolo_detector_node --ros-args \
    -p use_sim_time:=true -p perception_config:=configs/perception.yaml \
    -p scene_config:="${SIMLAB_CONFIG:-configs/default.yaml}" "$@"
