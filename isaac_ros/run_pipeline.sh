#!/usr/bin/env bash
# Default simulation entry point: collect, continually train, gate, then deploy.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
if [[ "${SIMLAB_INFERENCE_ONLY:-0}" != "1" ]]; then
  exec "$HERE/run_auto_pipeline.sh" "$@"
fi
source "$HERE/scripts/ros2_env.sh"
cd "$HERE"
exec ros2 launch launch/simlab.launch.py "$@"
