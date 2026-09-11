#!/usr/bin/env bash
# Repository entry point for the complete Shin/OntoReward flight benchmark.
set -Eeuo pipefail

repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root="$repository_root/Ontology_RGAT_UAV_RL_ISAAC_PX4"
launcher="$project_root/scripts/run_shin2026_benchmark.sh"

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
  # Default two-method run: 400 Shin + 400 OntoReward episodes. Five explicitly
  # selected ablation methods receive 160 each; a single --reward receives 800.
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
