#!/usr/bin/env bash
# Repository entry point for the controlled three-pipeline flight benchmark.
set -Eeuo pipefail

repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root="$repository_root/Ontology_RGAT_UAV_RL_ISAAC_PX4"
launcher="$project_root/scripts/run_three_pipeline.sh"

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

# Preserve the former reward-mode CLI explicitly. Legacy invocations continue
# to use the old runner; a bare run and the new --pipelines interface use the
# scientifically separated three-pipeline runner.
legacy_invocation=false
for argument in "$@"; do
  case "$argument" in
    --methods|--methods=*|--reward|--reward=*) legacy_invocation=true ;;
  esac
done
if [[ "$legacy_invocation" == true ]]; then
  launcher="$project_root/scripts/run_shin2026_benchmark.sh"
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
for argument in "$@"; do
  if [[ "$expect_mode_value" == true ]]; then
    selected_mode="$argument"
    expect_mode_value=false
    continue
  fi
  case "$argument" in
    --mode)
      mode_supplied=true
      expect_mode_value=true
      ;;
    --mode=*)
      mode_supplied=true
      selected_mode="${argument#--mode=}"
      ;;
    --train-episodes|--train-episodes=*|--total-train-episodes|--total-train-episodes=*)
      training_budget_supplied=true
      ;;
    --eval-episodes|--eval-episodes=*) evaluation_budget_supplied=true ;;
    --rgat-data-episodes|--rgat-data-episodes=*) rgat_budget_supplied=true ;;
  esac
done

arguments=("$@")
if [[ "$mode_supplied" == false ]]; then
  arguments=(--mode full "${arguments[@]}")
fi
if [[ "$selected_mode" == full ]]; then
  # Equal PPO budgets: 264 x 3, plus the Shin-only 8-episode estimator warm-up,
  # gives exactly 800 live training episodes in the default seminar run.
  # The runner subtracts that explicit overhead before dividing the remaining
  # PPO budget equally, including when --pipelines selects a subset.
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
