#!/usr/bin/env bash
# Explicit experiment routing. Help/status never acquire or stop a flight stack.
set -Eeuo pipefail
repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root="$repository_root/Ontology_RGAT_UAV_RL_ISAAC_PX4"
export PYTHONUNBUFFERED=1
case "${1:---help}" in
  --help|-h)
    echo 'Usage:'
    echo '  ./run.sh reference-smoke [--steps 32 --ppo-minibatch]'
    echo '  ./run.sh reference --stage train|evaluate|aggregate|tune|all [options]'
    echo '  ./run.sh status'
    echo '  ./run.sh spatial --stage train|evaluate|isaac|all --output PATH [options]'
    echo '  ./run.sh isaac-legacy [existing flight options]'
    echo 'Reference = two_axis_reference_v28_active.yaml, local dynamics, no Isaac/PX4.'
    echo 'Isaac legacy = planar_three_arm_comparison.yaml; NOT reference-equivalent.'
    echo 'No automatic training or process takeover. Use --takeover explicitly on isaac-legacy.'
    ;;
  reference-smoke)
    shift
    exec python3 "$project_root/python/run_two_axis_experiment.py" \
      --config "$project_root/config/experiments/two_axis_reference_v28_active.yaml" --smoke "$@"
    ;;
  reference)
    shift
    exec python3 "$project_root/python/run_two_axis_pipeline.py" "$@"
    ;;
  status)
    shift
    exec python3 "$project_root/tools/runtime_audit.py" "$@"
    ;;
  spatial)
    shift
    exec python3 "$project_root/python/run_spatial_pipeline.py" "$@"
    ;;
  isaac-legacy)
    shift
    exec bash "$project_root/scripts/run_isaac_legacy_entry.sh" "$@"
    ;;
  *)
    echo 'Select reference, reference-smoke, spatial, status, or isaac-legacy. See ./run.sh --help.' >&2
    exit 2
    ;;
esac
