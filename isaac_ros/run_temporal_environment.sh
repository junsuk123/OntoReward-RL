#!/usr/bin/env bash
# Accelerated Year0-to-Year10 environment experiment.
#
#   ./run_temporal_environment.sh
#   ./run_temporal_environment.sh --with-isaac
#   ./run_temporal_environment.sh --gui
#   ./run_temporal_environment.sh --config config/my_temporal.yaml --with-isaac
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${TEMPORAL_PYTHON:-python3}"
ISAAC_PYTHON="${TEMPORAL_ISAAC_PYTHON:-/home/j/env_isaaclab/bin/python}"
CONFIG="${TEMPORAL_CONFIG:-config/temporal_environment.yaml}"
WITH_ISAAC=0
ISAAC_GUI=0
ARGS=()
while (($#)); do
  case "$1" in
    --with-isaac) WITH_ISAAC=1; shift ;;
    --gui) WITH_ISAAC=1; ISAAC_GUI=1; shift ;;
    --config) CONFIG="$2"; shift 2 ;;
    --config=*) CONFIG="${1#--config=}"; shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

cd "$HERE"
OUTPUT="$(env PYTHONUNBUFFERED=1 "$PYTHON" -m simlab.temporal --config "$CONFIG" "${ARGS[@]}")"
echo "$OUTPUT"
if [[ "$WITH_ISAAC" == "1" ]]; then
  if [[ ! -x "$ISAAC_PYTHON" ]]; then
    echo "Isaac Sim interpreter not found: $ISAAC_PYTHON" >&2
    exit 2
  fi
  ISAAC_LOG="$OUTPUT/isaac_sim.log"
  ISAAC_ARGS=()
  if [[ "$ISAAC_GUI" == "1" ]]; then ISAAC_ARGS+=(--gui --hold); fi
  env PYTHONUNBUFFERED=1 "$ISAAC_PYTHON" -m simlab.temporal.isaac_capture \
    --config "$CONFIG" --output "$OUTPUT" "${ISAAC_ARGS[@]}" >"$ISAAC_LOG" 2>&1 || true
  if [[ ! -f "$OUTPUT/isaac_validation.json" ]]; then
    echo "Isaac validation did not complete. Last log lines:" >&2
    tail -80 "$ISAAC_LOG" >&2
    exit 1
  fi
  grep '\[temporal-isaac\]' "$ISAAC_LOG" || true
  echo "Isaac log: $ISAAC_LOG"
fi
