#!/usr/bin/env bash
# Launch the simlab scene with the project's Isaac Sim interpreter.
#   ./run.sh                       GUI, runs until closed
#   ./run.sh --headless --seconds 20
#   ./run.sh --allies 4 --enemies 7
#   ./run.sh --set drones.avoidance_radius=2.5
set -euo pipefail
PYTHON=${SIMLAB_PYTHON:-/home/j/env_isaaclab/bin/python}
cd "$(dirname "$0")"
exec env PYTHONUNBUFFERED=1 "$PYTHON" -m simlab "$@"
