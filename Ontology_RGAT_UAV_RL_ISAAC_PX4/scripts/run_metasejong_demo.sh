#!/usr/bin/env bash
# One graphical command: DDS + Meta-Sejong Isaac/PX4 + gateway + RViz + takeoff.
set -euo pipefail
workspace_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# This is the verified installation on the project workstation. An explicit
# environment value still wins, so the script remains portable.
if [[ -z ${ISAACSIM_PATH:-} && -x /home/j/isaacsim/_build/linux-x86_64/release/python.sh ]]; then
  export ISAACSIM_PATH=/home/j/isaacsim/_build/linux-x86_64/release
fi

exec python3 "$workspace_root/python/run_metasejong_demo.py" "$@"
