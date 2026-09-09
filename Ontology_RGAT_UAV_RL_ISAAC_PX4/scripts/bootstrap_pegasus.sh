#!/usr/bin/env bash
set -euo pipefail
workspace_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
isaacsim_path=${ISAACSIM_PATH:-}
isaac_python=${ISAACSIM_PYTHON:-${isaacsim_path:+$isaacsim_path/python.sh}}
pegasus_version=${PEGASUS_VERSION:-v5.1.0}

if [[ ${1:-} == "--help" ]]; then
  printf '%s\n' \
    "Usage: ISAACSIM_PATH=/path/to/isaacsim $0" \
    "Overrides: ISAACSIM_PYTHON=/path/to/python.sh PEGASUS_VERSION=$pegasus_version"
  exit 0
fi
if [[ -z "$isaac_python" || ! -x "$isaac_python" ]]; then
  printf 'Isaac Python not found. Set ISAACSIM_PATH or ISAACSIM_PYTHON.\n' >&2
  exit 1
fi
mkdir -p "$workspace_root/external"
if [[ ! -d "$workspace_root/external/PegasusSimulator/.git" ]]; then
  git clone --branch "$pegasus_version" --depth 1 \
    https://github.com/PegasusSimulator/PegasusSimulator.git \
    "$workspace_root/external/PegasusSimulator"
fi
"$isaac_python" -m pip install --editable \
  "$workspace_root/external/PegasusSimulator/extensions/pegasus.simulator"
printf 'Pegasus %s installed into Isaac Python.\n' "$pegasus_version"
