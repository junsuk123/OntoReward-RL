#!/usr/bin/env bash
# Kill every simlab / Nav2 / RViz process this project starts.
#
# Leftovers are easy to create and hard to spot: two generations of Nav2 servers
# register the same node names, so the lifecycle manager talks to whichever
# answers first and bringup fails in confusing ways. Match on `comm` (the
# executable name, which ps truncates to 15 chars) rather than the full command
# line -- a `pgrep -f simlab` pattern also matches the shell running it.
set -uo pipefail

PATTERNS='^(controller_serv|planner_server|behavior_serve|bt_navigator|velocity_smoot|lifecycle_mana|static_transfo|rviz2)'

pids=$(ps -eo pid,comm --no-headers | awk -v pat="$PATTERNS" '$2 ~ pat {print $1}')
pids="$pids $(ps -eo pid,comm,args --no-headers |
    awk '($2=="python"||$2=="python3") && /simlab/ {print $1}')"

found=0
for pid in $pids; do
    [[ -z "$pid" ]] && continue
    kill "$pid" 2>/dev/null && { echo "stopped $pid ($(cat /proc/$pid/comm 2>/dev/null))"; found=1; }
done
[[ $found -eq 0 ]] && echo "nothing running"

sleep 3
remaining=$(ps -eo pid,comm --no-headers | awk -v pat="$PATTERNS" '$2 ~ pat {print $1}')
for pid in $remaining; do
    kill -9 "$pid" 2>/dev/null && echo "force-killed $pid"
done
exit 0
