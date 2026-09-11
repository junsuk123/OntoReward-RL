#!/usr/bin/env bash
# Repository entry point for the complete Shin/OntoReward flight benchmark.
set -Eeuo pipefail

repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root="$repository_root/Ontology_RGAT_UAV_RL_ISAAC_PX4"
launcher="$project_root/scripts/run_shin2026_benchmark.sh"

if [[ ! -x "$launcher" ]]; then
  echo "ERROR: benchmark launcher is missing or not executable: $launcher" >&2
  exit 1
fi

# A bare ./run.sh means the publication-scale pipeline.  An explicitly supplied
# mode always wins, and every other option is passed through unchanged.
mode_supplied=false
for argument in "$@"; do
  case "$argument" in
    --mode|--mode=*) mode_supplied=true ;;
  esac
done

arguments=("$@")
if [[ "$mode_supplied" == false ]]; then
  arguments=(--mode full "${arguments[@]}")
fi

exec "$launcher" "${arguments[@]}"
