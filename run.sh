#!/usr/bin/env bash
# Experiment routing. Bare invocation runs the WHOLE minimal-observation system
# (user decision 2026-10-08). The MATLAB-port migration is an explicit route;
# it must not replace the repository's primary execution contract.
set -Eeuo pipefail
repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root="$repository_root/Ontology_RGAT_UAV_RL_ISAAC_PX4"
export PYTHONUNBUFFERED=1
first=${1:-}
if [ -z "$first" ] || { [ "${first#-}" != "$first" ] && [ "$first" != "--help" ] && [ "$first" != "-h" ]; }; then
  # Bare, or options only (./run.sh --no-isaac, ./run.sh --dry-run): the system.
  exec bash "$project_root/scripts/run_minimal_system.sh" "$@"
fi
case "$1" in
  --help|-h)
    echo 'Usage:'
    echo '  ./run.sh [--no-isaac] [--dry-run] [--nominal-training] [--run-root PATH]'
    echo '                               the WHOLE minimal-observation system: build ROS'
    echo '                               nodes, teacher -> BC -> PPO -> held-out -> stress'
    echo '                               -> Isaac/PX4 (when installed and not running)'
    echo '  ./run.sh system [options]    the same, by name'
    echo '  ./run.sh all [options]       spatial-reference pipeline: clone -> train every'
    echo '                               cell -> evaluate -> report (local unless --isaac)'
    echo '  ./run.sh reference-smoke [--steps 32 --ppo-minibatch]'
    echo '  ./run.sh reference --stage train|evaluate|aggregate|tune|all [options]'
    echo '  ./run.sh status'
    echo '  ./run.sh spatial --stage train|evaluate|isaac|all --output PATH [options]'
    echo '  ./run.sh matlab-port --stage audit|parity|smoke|train|evaluate|report [options]'
    echo '  ./run.sh matlab-port-all [options]'
    echo '                               audit -> parity -> smoke -> train -> local'
    echo '                               deterministic/sampled -> owned Isaac/PX4 -> report'
    echo '  ./run.sh matlab-port-final [options]'
    echo '                               lean final 3D route: train -> owned Isaac/PX4 only'
    echo '  ./run.sh isaac-legacy [existing flight options]'
    echo 'Reference = two_axis_reference_v28_active.yaml, local dynamics, no Isaac/PX4.'
    echo 'Isaac legacy = planar_three_arm_comparison.yaml; NOT reference-equivalent.'
    echo 'Bare ./run.sh TRAINS and FLIES: hours, writes under results/, and starts an'
    echo 'owned Isaac/PX4 stack at the end when one is installed; it never takes over a'
    echo 'running Isaac. --no-isaac keeps it local. --takeover stays explicit on isaac-legacy.'
    echo 'Preview without running anything: ./run.sh --dry-run'
    echo 'MATLAB-port preview: ./run.sh matlab-port-all --dry-run --no-isaac'
    ;;
  system)
    shift
    exec bash "$project_root/scripts/run_minimal_system.sh" "$@"
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
  matlab-port)
    shift
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
    exec python3 "$project_root/python/run_matlab_port.py" "$@"
    ;;
  matlab-port-all)
    shift
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
    exec python3 "$project_root/python/run_matlab_port_pipeline.py" "$@"
    ;;
  matlab-port-final)
    shift
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
    exec python3 "$project_root/python/run_matlab_port_pipeline.py" \
      --final-isaac-only "$@"
    ;;
  isaac-legacy)
    shift
    exec bash "$project_root/scripts/run_isaac_legacy_entry.sh" "$@"
    ;;
  *)
    echo 'Select system, all, reference, reference-smoke, spatial, matlab-port, matlab-port-all, matlab-port-final, status, or isaac-legacy. See ./run.sh --help.' >&2
    exit 2
    ;;
esac
