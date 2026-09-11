#!/usr/bin/env bash
# One-command live controlled benchmark: stack -> train -> paired eval -> report.
set -euo pipefail
workspace_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -z ${ISAACSIM_PATH:-} && -x /home/j/isaacsim/_build/linux-x86_64/release/python.sh ]]; then
  export ISAACSIM_PATH=/home/j/isaacsim/_build/linux-x86_64/release
fi
# The runtime ROS workspace is an ASCII-path mirror. Keep the executable
# gateway synchronized so the new velocity/scenario protocol is live.
if ! "$workspace_root/scripts/sync_gateway.sh" --check >/dev/null 2>&1; then
  "$workspace_root/scripts/sync_gateway.sh"
fi
exec "$workspace_root/scripts/run_learner.sh" run_shin2026_pipeline.py "$@"
