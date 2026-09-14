#!/usr/bin/env bash
# Repository entry point for the strict two-pipeline flight benchmark.
set -Eeuo pipefail

repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root="$repository_root/Ontology_RGAT_UAV_RL_ISAAC_PX4"
launcher="$project_root/scripts/run_two_pipeline.sh"

# Isaac/PX4 and the learner gateway are one physical control resource. Two
# learners cannot safely adopt them at once: UDP replies can go to the wrong
# client and one learner's reset/disarm interrupts the other. Keep the lock FD
# open across both launcher execs so a second run fails before touching results,
# RViz, the dashboard, or the flight stack. A crashed process releases flock
# automatically.
flight_lock=/tmp/ontology_rgat_flight_pipeline.lock
exec 9>>"$flight_lock"
if ! flock -n 9; then
  active_run=$(tr '\n' ' ' <"$flight_lock" 2>/dev/null || true)
  echo "ERROR: another Ontology-RGAT flight pipeline is already active${active_run:+ ($active_run)}." >&2
  echo "Wait for it to finish or stop that process before starting another run.sh." >&2
  exit 73
fi
: >"$flight_lock"
printf 'pid=%s command=%q\n' "$$" "$0 $*" >&9

# Preserve old commands only through explicitly legacy interfaces. The normal
# path cannot select estimator-free, adaptive-weight, or PBRS pipelines.
legacy_invocation=false
legacy_multi=false
for argument in "$@"; do
  case "$argument" in
    --methods|--methods=*|--reward|--reward=*) legacy_invocation=true ;;
    --legacy-multi-pipeline) legacy_multi=true ;;
  esac
done
if [[ "$legacy_invocation" == true ]]; then
  launcher="$project_root/scripts/run_shin2026_benchmark.sh"
elif [[ "$legacy_multi" == true ]]; then
  launcher="$project_root/scripts/run_three_pipeline.sh"
fi

if [[ ! -x "$launcher" ]]; then
  echo "ERROR: benchmark launcher is missing or not executable: $launcher" >&2
  exit 1
fi

# A bare ./run.sh means the deadline/seminar full pipeline. Explicit overrides
# always win, and every other option is passed through unchanged.
mode_supplied=false
selected_mode=full
training_budget_supplied=false
evaluation_budget_supplied=false
rgat_budget_supplied=false
expect_mode_value=false
expect_config_value=false
seminar_fast=false
# The repository's zero-argument contract is the currently supported seminar
# experiment: two isolated UAV/UGV pairs in one Isaac stage, one method
# per pair, with both operator views kept alive after the saved result.  Any
# explicit command line keeps the former opt-in behaviour and can override the
# profile below.
if [[ $# -eq 0 ]]; then
  seminar_fast=true
fi
for argument in "$@"; do
  if [[ "$expect_config_value" == true ]]; then
    expect_config_value=false
    continue
  fi
  if [[ "$expect_mode_value" == true ]]; then
    selected_mode="$argument"
    expect_mode_value=false
    continue
  fi
  case "$argument" in
    --seminar-fast) seminar_fast=true ;;
    --mode)
      mode_supplied=true
      expect_mode_value=true
      ;;
    --mode=*)
      mode_supplied=true
      selected_mode="${argument#--mode=}"
      ;;
    --config) expect_config_value=true ;;
    --train-episodes|--train-episodes=*|--total-train-episodes|--total-train-episodes=*)
      training_budget_supplied=true
      ;;
    --eval-episodes|--eval-episodes=*) evaluation_budget_supplied=true ;;
    --rgat-data-episodes|--rgat-data-episodes=*) rgat_budget_supplied=true ;;
  esac
done

arguments=()
for argument in "$@"; do
  [[ "$argument" == "--seminar-fast" || "$argument" == "--legacy-multi-pipeline" ]] \
    || arguments+=("$argument")
done
if [[ "$seminar_fast" == true ]]; then
  # The baseline and proposed agents receive the same flight profile, budgets,
  # seeds and PPO configuration on two isolated vehicle pairs.
  profile_arguments=(
    --mode full
    --config "$project_root/config/experiments/seminar_10h_two_pipeline.yaml"
    --system-config "$project_root/config/seminar-fast-system.yaml"
    --experiment two_pipeline_fov_risk
    --results-dir "$project_root/results/seminar_10h/two_pipe_parallel_144"
    --pipelines shin_se_fixed shin_se_onto_rgat_fov
    --parallel-pairs 2
    --stay-open
    --rgat-max-data-episodes 120
    --rgat-epochs 80
  )
  if [[ "$training_budget_supplied" == false ]]; then
    profile_arguments+=(--train-episodes 144)
    training_budget_supplied=true
  fi
  if [[ "$evaluation_budget_supplied" == false ]]; then
    profile_arguments+=(--eval-episodes 5)
    evaluation_budget_supplied=true
  fi
  if [[ "$rgat_budget_supplied" == false ]]; then
    profile_arguments+=(--rgat-data-episodes 40)
    rgat_budget_supplied=true
  fi
  arguments=("${profile_arguments[@]}" "${arguments[@]}")
  mode_supplied=true
  selected_mode=full
fi
if [[ "$mode_supplied" == false ]]; then
  arguments=(--mode full "${arguments[@]}")
fi
if [[ "$selected_mode" == full ]]; then
  # Equal PPO budgets and equal estimator warm-up apply to both SE-enabled arms.
  if [[ "$training_budget_supplied" == false ]]; then
    arguments+=(--total-train-episodes 800)
  fi
  # The original 32,000-flight evaluation and 400-flight reward-design pass are
  # also incompatible with a two-day seminar deadline. Keep every scenario but
  # use a clearly labelled preview sample size; CLI overrides recover any larger
  # budget without editing this launcher.
  if [[ "$evaluation_budget_supplied" == false ]]; then
    arguments+=(--eval-episodes 5)
  fi
  if [[ "$rgat_budget_supplied" == false ]]; then
    arguments+=(--rgat-data-episodes 40)
  fi
fi

exec "$launcher" "${arguments[@]}"
