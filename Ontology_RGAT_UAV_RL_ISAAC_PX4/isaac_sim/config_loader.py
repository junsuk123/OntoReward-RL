"""Load simulator YAML files with an optional local base configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings; lists and scalar values replace the base."""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    """Read PATH and merge its ``extends`` file, resolved beside PATH."""
    config_path = Path(path).expanduser().resolve()
    seen = set() if _seen is None else set(_seen)
    if config_path in seen:
        chain = " -> ".join(str(item) for item in (*seen, config_path))
        raise ValueError(f"cyclic config extends chain: {chain}")
    seen.add(config_path)

    with config_path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    if not isinstance(document, dict):
        raise ValueError(f"configuration must be a mapping: {config_path}")

    parent = document.pop("extends", None)
    if parent is None:
        return document
    if not isinstance(parent, str) or not parent.strip():
        raise ValueError(f"extends must be a non-empty path: {config_path}")
    parent_path = Path(parent).expanduser()
    if not parent_path.is_absolute():
        parent_path = config_path.parent / parent_path
    return _merge(load_config(parent_path, seen), document)
