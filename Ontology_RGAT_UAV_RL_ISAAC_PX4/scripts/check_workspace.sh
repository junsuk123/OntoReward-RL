#!/usr/bin/env bash
set -euo pipefail
workspace_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python3 -m compileall -q \
  "$workspace_root/ros2_ws/src/ontology_rgat_px4/ontology_rgat_px4" \
  "$workspace_root/tools"
bash -n "$workspace_root"/scripts/*.sh
PYTHONPATH="$workspace_root/ros2_ws/src/ontology_rgat_px4" \
  python3 -m pytest -q "$workspace_root/tests"
# grep, not rg: a missing ripgrep used to make this guard pass silently.
if grep -rEn 'dynamics\.rk4Step|aero\.distributedAero|sensor\.observe|control\.actionToRotorCmd|viz\.plotSurfaceLoads' \
  "$workspace_root/matlab/src"; then
  printf 'External sim adapter still calls the in-process simulator.\n' >&2
  exit 1
fi
printf 'Workspace checks passed.\n'
