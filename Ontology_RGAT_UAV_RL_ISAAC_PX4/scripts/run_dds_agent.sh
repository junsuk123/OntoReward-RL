#!/usr/bin/env bash
set -euo pipefail
workspace_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
agent="$workspace_root/external/install/bin/MicroXRCEAgent"
if [[ -d "$workspace_root/external/install/lib" ]]; then
  export LD_LIBRARY_PATH="$workspace_root/external/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [[ ! -x "$agent" ]]; then
  agent=$(command -v MicroXRCEAgent || true)
fi
if [[ -z "$agent" || ! -x "$agent" ]]; then
  printf 'MicroXRCEAgent not found; run scripts/bootstrap_px4_ros2.sh first.\n' >&2
  exit 1
fi
if (( $# > 0 )); then
  exec "$agent" "$@"
fi
exec "$agent" udp4 -p "${XRCE_UDP_PORT:-8888}"
