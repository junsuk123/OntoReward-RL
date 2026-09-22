#!/usr/bin/env bash
set -Eeuo pipefail

workspace_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# The runtime ROS workspace is an ASCII-path mirror of ros2_ws/src, and the
# gateway the stack launches is the mirror's copy, not this repository's. A
# stale mirror starts cleanly and then refuses the protocol check, which costs
# an Isaac boot and two relaunches before it says so -- exactly what happened
# on 2026-09-22 after a new deck scenario was added to protocol.py.
#
# Both legacy launchers have done this since they were written. This one --
# the launcher a bare ./run.sh uses, and therefore the one that matters most --
# did not, so the repository's own entry point was the only way to start a
# gateway that does not match the repository.
if ! "$workspace_root/scripts/sync_gateway.sh" --check >/dev/null 2>&1; then
  "$workspace_root/scripts/sync_gateway.sh"
fi

export PYTHONPATH="$workspace_root/python:${PYTHONPATH:-}"
exec python3 "$workspace_root/python/run_two_pipeline.py" "$@"

