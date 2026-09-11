#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd -- "$script_dir/.." && pwd)
isaac_root=${ISAACSIM_ROOT:-/home/j/isaacsim}
isaac_python="$isaac_root/python.sh"
if [[ ! -x "$isaac_python" && -x "$isaac_root/_build/linux-x86_64/release/python.sh" ]]; then
  isaac_python="$isaac_root/_build/linux-x86_64/release/python.sh"
fi

if [[ ! -x "$isaac_python" ]]; then
  echo "Isaac Sim python.sh not found under: $isaac_root" >&2
  echo "Set ISAACSIM_ROOT to the Isaac Sim installation directory." >&2
  exit 1
fi

exec "$isaac_python" "$project_dir/tools/import_ranger_mini_v3.py" \
  --source "$project_dir/assets/ranger_mini_v3/source" \
  --output "$project_dir/assets/ranger_mini_v3/ranger_mini_v3.usd"
