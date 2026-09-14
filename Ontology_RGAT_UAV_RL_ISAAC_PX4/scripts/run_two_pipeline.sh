#!/usr/bin/env bash
set -Eeuo pipefail

workspace_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$workspace_root/python:${PYTHONPATH:-}"
exec python3 "$workspace_root/python/run_two_pipeline.py" "$@"

