#!/usr/bin/env bash
# Repository entry point for the strict two-pipeline flight benchmark.
set -Eeuo pipefail

repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root="$repository_root/Ontology_RGAT_UAV_RL_ISAAC_PX4"
launcher="$project_root/scripts/run_two_pipeline.sh"

# Isaac/PX4 and the learner gateway are one physical control resource. Two
# learners cannot safely adopt them at once: UDP replies can go to the wrong
# client and one learner's reset/disarm interrupts the other. Starting a new
# run therefore means the previous one ends: this launcher stops the active
# pipeline and every flight process left behind by an earlier crash, so the
# new run always owns a fresh stack instead of adopting a degraded one.
# ``--no-takeover`` restores the former behaviour of refusing to start.
flight_lock=/tmp/ontology_rgat_flight_pipeline.lock
exec 9>>"$flight_lock"

takeover=true
for argument in "$@"; do
  [[ "$argument" == "--no-takeover" ]] && takeover=false
done

# The same processes ``scripts/stack_status.sh`` reports: the learner itself,
# the simulator, PX4 SITL, the DDS agent and the velocity gateways. Each is
# started in its own session, so the process group takes the whole subtree
# (Pegasus' PX4 child included) down with its parent.
flight_process_pattern='python/run_two_pipeline\.py|python/run_three_pipeline\.py|run_shin2026_benchmark|isaac_sim/landing_world\.py|MicroXRCEAgent|px4_sitl_default/bin/px4|ros2_gateway|rviz2'
own_group=$(ps -o pgid= -p "$$" | tr -d ' ')

# A shell or a grep whose command line merely mentions those names is not a
# flight process, and killing an operator's terminal would be far worse than
# adopting a stale simulator. Accept a match only when it is the program being
# run: never an inline ``-c`` script, never a search command.
is_flight_process() {
  local pid=$1 first second token
  local -a tokens
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  mapfile -d '' -t tokens < "/proc/$pid/cmdline" 2>/dev/null || return 1
  (( ${#tokens[@]} )) || return 1
  first=${tokens[0]}
  second=${tokens[1]:-}
  [[ "$second" == "-c" ]] && return 1
  for token in "${tokens[@]}"; do
    case "${token##*/}" in
      grep|pgrep|pkill|egrep|ps|tail|sed|awk|kill) return 1 ;;
    esac
  done
  case "${first##*/}" in
    python|python3|python3.*|bash|sh|MicroXRCEAgent|px4|ros2|python.sh|rviz2) ;;
    *) return 1 ;;
  esac
  return 0
}

flight_process_pids() {
  local pid
  {
    # A lock holder that has not exec'd the learner yet is only named here.
    sed -n 's/^pid=\([0-9]\{1,\}\).*/\1/p' "$flight_lock" 2>/dev/null || true
    pgrep -f "$flight_process_pattern" 2>/dev/null || true
  } | sort -u | while read -r pid; do
    [[ -n "$pid" && "$pid" != "$$" ]] || continue
    kill -0 "$pid" 2>/dev/null || continue
    is_flight_process "$pid" || continue
    printf '%s\n' "$pid"
  done
}

# Signal targets: a whole process group where the previous run has its own
# (the normal case, since every stack process starts a new session), and bare
# pids for anything that shares this launcher's group, which must be signalled
# individually so the new run does not kill itself.
flight_targets() {
  local pid group
  for pid in $(flight_process_pids); do
    group=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    if [[ -n "$group" && "$group" != "$own_group" ]]; then
      printf -- '-%s\n' "$group"
    else
      printf -- '%s\n' "$pid"
    fi
  done | sort -u
}

stop_previous_flight_pipeline() {
  local targets signal target waited pid
  targets=$(flight_targets)
  [[ -n "$targets" ]] || return 0
  echo "Stopping the flight pipeline left running by an earlier command:"
  for pid in $(flight_process_pids); do
    echo "  $(ps -o pid=,etime=,cmd= -p "$pid" 2>/dev/null | cut -c1-140)"
  done
  # SIGINT first so Isaac and rclpy close their contexts cleanly, exactly as
  # ExternalStack.stop() does; escalate only if the tree ignores it.
  for signal in INT TERM KILL; do
    for target in $targets; do
      kill -"$signal" -- "$target" 2>/dev/null || true
    done
    waited=0
    while (( waited < 40 )); do
      [[ -n "$(flight_process_pids)" ]] || break
      sleep 0.5
      waited=$((waited + 1))
    done
    targets=$(flight_targets)
    [[ -n "$targets" ]] || break
  done
  if [[ -n "$(flight_process_pids)" ]]; then
    echo "ERROR: could not stop the previous flight pipeline; it is still running." >&2
    flight_process_pids | sed 's/^/  pid /' >&2
    exit 73
  fi
  echo "Previous flight pipeline stopped."
}

if ! flock -n 9; then
  if [[ "$takeover" == false ]]; then
    active_run=$(tr '\n' ' ' <"$flight_lock" 2>/dev/null || true)
    echo "ERROR: another Ontology-RGAT flight pipeline is already active${active_run:+ ($active_run)}." >&2
    echo "Wait for it to finish or stop that process before starting another run.sh." >&2
    exit 73
  fi
  stop_previous_flight_pipeline
  # The dead holder releases flock, but only once the kernel reaps it.
  if ! flock -w 30 9; then
    echo "ERROR: the flight lock is still held after stopping the previous run." >&2
    exit 73
  fi
fi
# The lock is ours from here; the holder line must not name a dead run any more.
: >"$flight_lock"
# An orphaned simulator survives its learner and would otherwise be adopted in
# a degraded state, which is what the entry-hover and arming failures look like.
if [[ "$takeover" == true ]]; then
  stop_previous_flight_pipeline
fi
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

# A bare ./run.sh means the full reference experiment. Explicit overrides
# always win, and every other option is passed through unchanged.
mode_supplied=false
selected_mode=full
training_budget_supplied=false
evaluation_budget_supplied=false
rgat_budget_supplied=false
expect_mode_value=false
expect_config_value=false
seminar_fast=false
# The repository's zero-argument contract is the full reference experiment:
# ./run.sh alone runs config/experiments/three_arm_burst_comparison.yaml at the
# budgets that file declares, on as many isolated UAV/UGV pairs as the machine
# measures room for.
#
# That is the three-arm burst comparison (2026-09-22): one deck -- the pad
# cruises straight, the vehicle settles into following it, and then it doubles
# speed and leaves the camera frame -- flown by a non-learned visual servo
# control condition and the two learned arms. The six-deck two-arm design it
# replaced is still declared and still runnable:
#   ./run.sh --config config/experiments/two_pipeline_comparison.yaml
#
# It used to select the seminar preview here instead -- the profile marked
# publication_claim_allowed: false. So the command that reads as "run the
# experiment" quietly ran the one whose results may not be published, and an
# edit to the experiment config landed on a file that path never opens. The
# preview is still one flag away: ./run.sh --seminar-fast.
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
  [[ "$argument" == "--seminar-fast" || "$argument" == "--legacy-multi-pipeline" \
     || "$argument" == "--no-takeover" ]] || arguments+=("$argument")
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
    --pipelines shin_se_fixed shin_se_onto_rgat_recovery
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
# --mode full runs the budget the experiment config declares, which is the
# reference budget: 1,000 PPO episodes per arm, a 400-episode floor on the
# FOV-risk design set, and 200 paired evaluation flights per scenario per arm.
# Those are sized against this machine's measured ~20 episodes per hour per
# pair -- about five days end to end -- rather than the round 40,000 that
# preceded them, which was months of flying and could not have completed.
# Equal PPO budgets and equal estimator warm-up still apply to both
# SE-enabled arms, because --total-train-episodes is not imposed here and each
# arm takes training.episodes_full.
#
# The seminar preview budget still exists as its own profile:
# ./run.sh --seminar-fast, which is the one marked
# publication_claim_allowed: false. A shorter ad-hoc run needs no edit here
# either -- --train-episodes, --total-train-episodes, --eval-episodes and
# --rgat-data-episodes all still override the config.

exec "$launcher" "${arguments[@]}"
