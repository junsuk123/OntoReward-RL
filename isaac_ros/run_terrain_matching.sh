#!/usr/bin/env bash
# Run the GNSS-denied ontology-guided terrain matching experiment.
#
#   ./run_terrain_matching.sh
#   ./run_terrain_matching.sh --config config/my_experiment.yaml
#   TERRAIN_MATCHING_PYTHON=/path/to/python ./run_terrain_matching.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${TERRAIN_MATCHING_PYTHON:-python3}"
DEFAULT_CONFIG="${TERRAIN_MATCHING_CONFIG:-config/experiment.yaml}"

cd "$HERE"

# Let an explicit command-line --config override the environment/default path.
for argument in "$@"; do
  if [[ "$argument" == "--config" || "$argument" == --config=* ]]; then
    exec env PYTHONUNBUFFERED=1 "$PYTHON" -m simlab.terrain_matching "$@"
  fi
done

exec env PYTHONUNBUFFERED=1 "$PYTHON" -m simlab.terrain_matching \
  --config "$DEFAULT_CONFIG" "$@"
