"""Experiment configuration, hashes and paired-seed plans."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


METHODS = ("shin2026", "sparse", "manual_no_active", "ontoreward",
           "ontoreward_plus_active")


def _merge(base: dict, overlay: dict) -> dict:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_experiment(path: str | Path, seen=None) -> dict:
    path = Path(path).resolve()
    visited = set() if seen is None else set(seen)
    if path in visited:
        raise ValueError(f"cyclic experiment configuration at {path}")
    visited.add(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = data.pop("extends", None)
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        data = _merge(load_experiment(parent_path, visited), data)
    return data


def configuration_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def paired_seed_plan(methods, scenarios: dict[str, int], seed0=5000):
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f"unknown benchmark methods: {sorted(unknown)}")
    plan = []
    offset = 0
    for scenario, episodes in scenarios.items():
        seeds = list(range(int(seed0) + offset, int(seed0) + offset + int(episodes)))
        offset += int(episodes)
        for method in methods:
            plan.extend({"method": method, "scenario": scenario, "seed": seed}
                        for seed in seeds)
    return plan
