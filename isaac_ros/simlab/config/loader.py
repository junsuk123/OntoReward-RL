"""Load a YAML scene config and apply overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from simlab.config.schema import SceneConfig
from simlab.utils.paths import PROJECT_ROOT, resolve_path

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base``. Lists replace, they don't append."""
    merged: Dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def read_yaml(path: str | Path) -> Dict[str, Any]:
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"config file not found: {resolved}")
    with resolved.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{resolved}: top level must be a mapping, got {type(data).__name__}")
    return data


def load_config(
    path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> SceneConfig:
    """Read ``path`` (default: ``configs/default.yaml``) and merge ``overrides`` on top."""
    data = read_yaml(path or DEFAULT_CONFIG)
    if overrides:
        data = deep_merge(data, overrides)
    return SceneConfig.from_dict(data)
