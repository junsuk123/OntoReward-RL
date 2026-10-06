#!/usr/bin/env bash
# Experiment routing. Bare invocation runs the whole LOCAL pipeline; status,
# help and smoke never acquire or stop a flight stack, and nothing here reaches
# Isaac/PX4 without the explicit `spatial --stage isaac` route.
set -Eeuo pipefail
repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root="$repository_root/Ontology_RGAT_UAV_RL_ISAAC_PX4"
export PYTHONUNBUFFERED=1
case "${1:-all}" in
  --help|-h)
    echo 'Usage:'
    echo '  ./run.sh                     run the whole local pipeline (see all)'
    echo '  ./run.sh all [options]       clone -> train every cell -> evaluate -> report'
    echo '  ./run.sh reference-smoke [--steps 32 --ppo-minibatch]'
    echo '  ./run.sh reference --stage train|evaluate|aggregate|tune|all [options]'
    echo '  ./run.sh status'
    echo '  ./run.sh spatial --stage train|evaluate|isaac|all --output PATH [options]'
    echo '  ./run.sh isaac-legacy [existing flight options]'
    echo 'Reference = two_axis_reference_v28_active.yaml, local dynamics, no Isaac/PX4.'
    echo 'Isaac legacy = planar_three_arm_comparison.yaml; NOT reference-equivalent.'
    echo 'Bare ./run.sh TRAINS. It is long-running and writes under results/.'
    echo 'It never starts Isaac/PX4; use spatial --stage isaac or isaac-legacy for that,'
    echo 'and --takeover stays explicit on isaac-legacy.'
    echo 'Preview the plan without running it: ./run.sh all --dry-run'
    ;;
  all)
    shift || true
    exec python3 "$project_root/tools/run_full_pipeline.py" "$@"
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
