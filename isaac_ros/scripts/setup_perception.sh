#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
VENV="$ROOT/.venv-perception"
if [[ ! -x "$VENV/bin/python" ]]; then
    python3 -m venv --system-site-packages "$VENV"
fi
if ! "$VENV/bin/python" - <<'PY' 2>/dev/null
import ultralytics, polars
parts = tuple(int(x) for x in ultralytics.__version__.split('.')[:3])
healthy = polars.DataFrame({"x": [1]}).height == 1
raise SystemExit(0 if parts >= (8, 3, 237) and healthy else 1)
PY
then
    "$VENV/bin/python" -m pip install -r "$ROOT/requirements-perception.txt"
fi
printf '%s\n' "$VENV/bin/python"
