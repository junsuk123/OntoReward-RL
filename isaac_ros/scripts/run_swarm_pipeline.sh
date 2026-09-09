#!/usr/bin/env bash
# Prototype ontology-gated multi-camera tracking and RViz marker publisher.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
source "$HERE/ros2_env.sh"
cd "$ROOT"
exec env PYTHONUNBUFFERED=1 PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -m simlab.ros.swarm_pipeline_node --ros-args \
    -p use_sim_time:=true \
    -p config:="${SIMLAB_CONFIG:-configs/default.yaml}" \
    "$@"
