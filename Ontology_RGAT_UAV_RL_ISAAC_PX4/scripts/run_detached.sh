#!/usr/bin/env bash
# Start a run that outlives the terminal, the SSH session and the editor.
#
# 2026-09-25 a 22-hour run reached episode 630/1008 and then stopped mid-line:
# no traceback, no teardown, the runner's log simply ending and Isaac and the
# gateways dying a minute later as orphans. Nothing had failed. The run had
# been started as a background job of an editor session, and when that session
# ended it took the whole process group with it.
#
# ``setsid`` puts the run in its own session and process group, so it has no
# controlling terminal to be hung up on and no parent whose exit can reap it.
# Checkpoints already make a run resumable; this makes it survivable.
#
#   scripts/run_detached.sh                    start, print the log path and PID
#   scripts/run_detached.sh --status           is one running, and how far along
#   scripts/run_detached.sh --stop             interrupt it the way Ctrl-C would
#   scripts/run_detached.sh -- --mode quick    anything after -- goes to run.sh
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project=$(cd "$here/.." && pwd)
launcher="$(cd "$project/.." && pwd)/run.sh"
state_dir="$project/logs"
pid_file="$state_dir/run_detached.pid"

running_pid() {
  [[ -f $pid_file ]] || return 1
  local pid
  pid=$(cat "$pid_file" 2>/dev/null) || return 1
  [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null && printf '%s' "$pid"
}

case "${1:-}" in
  --status)
    if pid=$(running_pid); then
      printf 'running: pid %s, up %s s\n' "$pid" "$(ps -o etimes= -p "$pid" | tr -d ' ')"
      log=$(cat "$state_dir/run_detached.log.path" 2>/dev/null || true)
      [[ -n $log && -f $log ]] &&
        grep -oE '(shin_se_fixed|shin_se_onto_rgat_state) episode [0-9]+/[0-9]+' "$log" | tail -2
    else
      printf 'no detached run is active\n'
    fi
    exit 0 ;;
  --stop)
    if pid=$(running_pid); then
      printf 'interrupting pid %s; the stack teardown takes a couple of minutes\n' "$pid"
      kill -INT "$pid"
    else
      printf 'no detached run is active\n'
    fi
    exit 0 ;;
  --) shift ;;
  "") ;;
  *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
esac

if pid=$(running_pid); then
  printf 'a detached run is already active (pid %s); stop it first\n' "$pid" >&2
  exit 1
fi

mkdir -p "$state_dir"
log="$state_dir/run-$(date +%Y%m%d-%H%M%S).log"
printf '%s' "$log" > "$state_dir/run_detached.log.path"

# PYTHONUNBUFFERED because the run's own log otherwise stays empty for minutes
# and the first sign of trouble arrives late.
PYTHONUNBUFFERED=1 setsid nohup "$launcher" "$@" > "$log" 2>&1 < /dev/null &
pid=$!
printf '%s' "$pid" > "$pid_file"
sleep 1
printf 'started pid %s in session %s\n' "$pid" "$(ps -o sid= -p "$pid" | tr -d ' ')"
printf 'log: %s\n' "$log"
printf 'it survives this terminal; scripts/run_detached.sh --status to check on it\n'
