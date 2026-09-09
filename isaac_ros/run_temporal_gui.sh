#!/usr/bin/env bash
# One-command workflow: batch generation -> Isaac Sim -> validation GUI.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
exec "$HERE/run_temporal_environment.sh" --gui "$@"
