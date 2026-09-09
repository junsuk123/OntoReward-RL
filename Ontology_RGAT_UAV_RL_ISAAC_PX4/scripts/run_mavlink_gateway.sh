#!/usr/bin/env bash
set -euo pipefail
workspace_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$workspace_root/ros2_ws/src/ontology_rgat_px4${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m ontology_rgat_px4.mavlink_gateway \
  --config "$workspace_root/config/system.yaml" "$@"

