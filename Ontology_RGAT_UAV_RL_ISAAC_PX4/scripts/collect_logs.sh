#!/usr/bin/env bash
# Gather every log the stack produces into the project, and therefore into
# Synology Drive.
#
# Four writers put their logs in four places outside the repository, and three
# of them are destroyed on the next run:
#
#   Isaac/gateway/agent   /tmp/ontology_rgat_stack/<name>.log, kept for exactly
#                         two generations -- every stack rebuild rotates
#                         <name>.log to <name>.previous.log and drops the one
#                         before it. Rebuilds ran every ~20 minutes on
#                         2026-09-24, so the log that explained a failure was
#                         routinely gone before anyone read it.
#   PX4 flight logs       Pegasus starts each PX4 in its own
#                         tempfile.TemporaryDirectory and PX4 writes
#                         ./log/<date>/<time>.ulg there. Nothing reads them and
#                         nothing removes them: 69.9 GB had accumulated by
#                         2026-09-24.
#   ROS 2 node logs       ~/.ros/log
#   Isaac Sim (Kit)       ~/.local/share/ov/data/Kit
#
#   scripts/collect_logs.sh                 one pass, logs since the last pass
#   scripts/collect_logs.sh --all           include the historical backlog
#   scripts/collect_logs.sh --watch [secs]  keep collecting while a run flies
#
# Volume, because it decides whether this is a good idea: PX4 alone writes
# about 15 GB a day, and everything here lands in a synced folder. The ulogs
# are compressed on arrival, which gets roughly 4x on this data. logs/ is in
# .gitignore -- none of it belongs in the repository.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project=$(cd "$here/.." && pwd)
logs="$project/logs"

mode=once
interval=300
all=0
while [ $# -gt 0 ]; do
  case "$1" in
    --all) all=1 ;;
    --watch) mode=watch; [ "${2:-}" ] && [[ "${2}" =~ ^[0-9]+$ ]] && { interval=$2; shift; } ;;
    -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
  shift
done

# One directory per run, named for the runner that is flying now. Falling back
# to a timestamp keeps a manual collection from landing in another run's tree.
run_started=$(ps -o lstart= -C python3 2>/dev/null | head -1 || true)
runner_pid=$(pgrep -f 'python/run_(two|three)_pipeline\.py' | head -1 || true)
if [ -n "$runner_pid" ]; then
  stamp=$(date -d "@$(($(date +%s) - $(ps -o etimes= -p "$runner_pid" | tr -d ' ')))" +%Y%m%d-%H%M)
  run="run-$stamp"
else
  run=$(ls -1 "$logs/runs" 2>/dev/null | sort | tail -1)
  [ -n "$run" ] || run="run-$(date +%Y%m%d-%H%M)"
fi
destination="$logs/runs/$run"
mkdir -p "$destination/stack" "$destination/px4" "$logs/ros2" "$logs/isaac_kit"

newer_than() {
  # Only what this run produced, unless --all was asked for.
  if [ "$all" = 1 ]; then printf '%s' "1970-01-01"; else printf '%s' "$(date +%Y-%m-%d)"; fi
}

collect_once() {
  local since; since=$(newer_than)

  # PX4 flight logs. Copied and compressed, never moved: PX4 may still have
  # the file open, and a half-written ulog is worth more than no ulog.
  local copied=0
  while IFS= read -r ulg; do
    [ -n "$ulg" ] || continue
    local owner target
    owner=$(printf '%s' "$ulg" | sed -E 's#^/tmp/([^/]+)/log/.*#\1#')
    target="$destination/px4/$owner/$(basename "$(dirname "$ulg")")"
    mkdir -p "$target"
    if [ ! -f "$target/$(basename "$ulg").gz" ]; then
      gzip -c "$ulg" > "$target/$(basename "$ulg").gz.part" 2>/dev/null &&
        mv "$target/$(basename "$ulg").gz.part" "$target/$(basename "$ulg").gz" &&
        copied=$((copied + 1))
    fi
  done < <(find /tmp -path '*/log/*' -name '*.ulg' -newermt "$since" -type f 2>/dev/null)

  # Stack logs. The live tail collectors own <name>.cumulative.log; this picks
  # up the rotated generations they cannot see.
  for source in /tmp/ontology_rgat_stack/*.log; do
    [ -f "$source" ] || continue
    cp -u "$source" "$destination/stack/$(basename "$source")" 2>/dev/null || true
  done

  # ROS 2 and Isaac Sim.
  if [ -d "$HOME/.ros/log" ]; then
    find "$HOME/.ros/log" -newermt "$since" -type f -print0 2>/dev/null |
      rsync -a --files-from=- --from0 / "$logs/ros2/" 2>/dev/null || true
  fi
  if [ -d "$HOME/.local/share/ov/data/Kit" ]; then
    find "$HOME/.local/share/ov/data/Kit" -newermt "$since" -type f -print0 2>/dev/null |
      rsync -a --files-from=- --from0 / "$logs/isaac_kit/" 2>/dev/null || true
  fi

  printf '%s  %s: +%d ulog(s), %s total, %s free on /\n' \
    "$(date +%H:%M:%S)" "$run" "$copied" \
    "$(du -sh "$logs" 2>/dev/null | cut -f1)" \
    "$(df -h / | awk 'NR==2 {print $4}')"
}

if [ "$mode" = watch ]; then
  printf 'collecting into %s every %s s; Ctrl-C to stop\n' "$destination" "$interval"
  while true; do collect_once; sleep "$interval"; done
else
  collect_once
fi
