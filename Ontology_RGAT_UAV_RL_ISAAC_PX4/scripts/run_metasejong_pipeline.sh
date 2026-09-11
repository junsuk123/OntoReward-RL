#!/usr/bin/env bash
# One command for DDS + Meta-Sejong Isaac/PX4 + gateway + RViz/dashboard and
# the ontology -> R-GAT -> PPO -> evaluation experiment.
set -euo pipefail
workspace_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if [[ -z ${ISAACSIM_PATH:-} && -x /home/j/isaacsim/_build/linux-x86_64/release/python.sh ]]; then
  export ISAACSIM_PATH=/home/j/isaacsim/_build/linux-x86_64/release
fi

exec "$workspace_root/scripts/run_learner.sh" run_pipeline.py \
  --system-config "$workspace_root/config/metasejong-pipeline.yaml" "$@"
