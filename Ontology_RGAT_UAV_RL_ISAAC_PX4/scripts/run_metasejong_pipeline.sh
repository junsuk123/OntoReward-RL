#!/usr/bin/env bash
# One command for DDS + Meta-Sejong Isaac/PX4 + gateway + RViz/dashboard and
# the ontology -> R-GAT -> PPO -> evaluation experiment.
set -euo pipefail
workspace_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# The controlled Shin benchmark has its own runner and system profile.  Strip
# only the selector; every remaining option is forwarded verbatim.
benchmark=false
forward=()
while (($#)); do
  if [[ $1 == --experiment && ${2:-} == shin2026 ]]; then
    benchmark=true
    shift 2
  else
    forward+=("$1")
    shift
  fi
done

if [[ -z ${ISAACSIM_PATH:-} && -x /home/j/isaacsim/_build/linux-x86_64/release/python.sh ]]; then
  export ISAACSIM_PATH=/home/j/isaacsim/_build/linux-x86_64/release
fi

if $benchmark; then
  exec "$workspace_root/scripts/run_learner.sh" run_shin2026_benchmark.py \
    --config "$workspace_root/config/experiments/shin2026_ablation.yaml" "${forward[@]}"
fi

exec "$workspace_root/scripts/run_learner.sh" run_pipeline.py \
  --system-config "$workspace_root/config/metasejong-pipeline.yaml" "${forward[@]}"
