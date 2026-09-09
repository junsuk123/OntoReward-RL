#!/usr/bin/env bash
# One command: persistent collection, cumulative SAM labels, continual YOLO training, deployment.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/scripts/ros2_env.sh"
PYTHON_BIN="$("$HERE/scripts/setup_perception.sh" | tail -n 1)"
cd "$HERE"
export SIMLAB_PERCEPTION_PYTHON="$PYTHON_BIN"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" -m simlab.ml.auto_pipeline "$@"
