#!/usr/bin/env bash
# Experiment routing.  Since 2026-11-20 the repository holds one route: the current-observation
# 3D study of the ORL landing policy and its comparison arms (current_semantics/).  Every other
# route (minimal-observation ROS system, spatial-reference ladder, two-axis reference, MATLAB
# port, Isaac legacy, navigation-degradation demo) was archived in
# archive/non_orl_20261120.tar.zst.  A bare ./run.sh only prints this usage; it starts nothing.
set -Eeuo pipefail
repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root="$repository_root/Ontology_RGAT_UAV_RL_ISAAC_PX4"
export PYTHONUNBUFFERED=1

usage() {
  echo 'Usage:'
  echo '  ./run.sh current-semantics-3d --stage STAGE [options]'
  echo '      audit | fixtures | smoke | coupling | learnability | report     contract and gates'
  echo '      orl-learnability | register-orl-development | orl-develop       ORL protocols'
  echo '      register-speed | speed | speed-report                           learning-speed/1'
  echo '      register-development | develop                                  development stage'
  echo '      isaac-eval | isaac-trace | register-isaac-development | isaac-development'
  echo '                                  owned Isaac Sim + PX4 flights, only with --allow-isaac'
  echo '      monitor | dashboard                                             read-only pages'
  echo '  ./run.sh current-semantics-3d --help      every stage and option'
  echo
  echo 'A bare ./run.sh starts nothing.  Other routes were archived on 2026-11-20'
  echo '(archive/non_orl_20261120.tar.zst).'
}

case "${1:-}" in
  ""|--help|-h)
    usage
    ;;
  current-semantics-3d)
    shift
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
    exec python3 "$project_root/python/run_current_semantics_3d.py" "$@"
    ;;
  *)
    echo "unknown route '$1'; only current-semantics-3d remains (other routes are archived)" >&2
    usage >&2
    exit 2
    ;;
esac
