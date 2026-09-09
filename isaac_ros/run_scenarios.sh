#!/usr/bin/env bash
# The three tracking-failure scenarios, each flown as its own episode, followed
# by incremental training on whatever has not been learned yet.
#
#   ./run_scenarios.sh                       full plan, then deploy
#   ./run_scenarios.sh --dry-run             write the episode configs only
#   ./run_scenarios.sh --scenario sensor_dropout --no-deploy
#   ./run_scenarios.sh --seed 7              reproduce a previous run exactly
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/scripts/ros2_env.sh"
PYTHON_BIN="$("$HERE/scripts/setup_perception.sh" | tail -n 1)"
cd "$HERE"
export SIMLAB_PERCEPTION_PYTHON="$PYTHON_BIN"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" -m simlab.scenarios.orchestrator "$@"
