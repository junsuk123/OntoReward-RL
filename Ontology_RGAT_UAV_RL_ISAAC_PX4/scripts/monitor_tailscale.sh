#!/bin/bash
# Copyright (c) 2026 Jun Suk Kim (김준석)
# Maintainer: Jun Suk Kim (김준석) <junsuk55944587@gmail.com>
# SPDX-License-Identifier: Apache-2.0

# Read-only study monitor (current_semantics/monitor.py) on this machine's Tailscale address,
# so it is reachable from devices on the same tailnet only (not localhost-only, not public).
#   scripts/monitor_tailscale.sh start|stop|status   [port, default 8780]
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
port="${2:-8780}"
log="$here/results/current_semantics_3d/monitor_tailscale.log"
find_pid() { pgrep -f "run_current_semantics_3d[.]py --stage monitor --host" || true; }
case "${1:-status}" in
  start)
    if [ -n "$(find_pid)" ]; then echo "already running: $(find_pid)"; exit 0; fi
    ip="$(tailscale ip -4 | head -1)"
    [ -n "$ip" ] || { echo "tailscale has no IPv4 address (is tailscaled up?)" >&2; exit 1; }
    CS3D_DASHBOARD_NO_BROWSER=1 setsid nohup nice -n 10 \
      bash "$here/../run.sh" current-semantics-3d --stage monitor --host "$ip" --port "$port" \
      > "$log" 2>&1 < /dev/null &
    for _ in $(seq 1 40); do grep -q "^monitor:" "$log" 2>/dev/null && break; sleep 0.5; done
    head -1 "$log"
    dns="$(tailscale status --json | python3 -c 'import sys,json;print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))')"
    echo "also: http://$dns:$port/"
    ;;
  stop)
    pid="$(find_pid)"; [ -n "$pid" ] && kill $pid && echo "stopped $pid" || echo "not running"
    ;;
  status)
    pid="$(find_pid)"; [ -n "$pid" ] && echo "running: $pid" && head -1 "$log" || echo "not running"
    ;;
  *) echo "usage: $0 start|stop|status [port]" >&2; exit 2 ;;
esac
