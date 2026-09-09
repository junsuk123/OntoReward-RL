"""Filesystem locations used by the package."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_ROOT.parent


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    """Resolve ``path``; relative paths are taken against ``base`` (default: project root)."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return ((base or PROJECT_ROOT) / candidate).resolve()
