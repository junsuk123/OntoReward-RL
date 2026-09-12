#!/usr/bin/env bash
set -Eeuo pipefail

workspace_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if ! "$workspace_root/scripts/sync_gateway.sh" --check >/dev/null 2>&1; then
  "$workspace_root/scripts/sync_gateway.sh"
fi

exec "$workspace_root/scripts/run_learner.sh" run_three_pipeline.py "$@"
