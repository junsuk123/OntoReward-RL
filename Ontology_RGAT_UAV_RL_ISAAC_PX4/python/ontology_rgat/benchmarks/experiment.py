"""Experiment configuration, hashes and paired-seed plans."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


PRIMARY_PIPELINES = ("shin_se_fixed", "shin_se_onto_rgat_fov")
ADAPTIVE_PIPELINES = (
    "shin_se_rgat_weight", "no_se_fixed",
    "onto_rgat_adaptive_weight_no_se", "onto_rgat_potential_pbrs_no_se",
    "mlp_adaptive_weight_no_se", "gat_adaptive_weight_no_se",
    "rgat_adaptive_weight_no_se",
)
LEGACY_METHODS = ("shin2026", "sparse", "manual_no_active", "ontoreward",
                  "ontoreward_plus_active")
METHODS = PRIMARY_PIPELINES + ADAPTIVE_PIPELINES + LEGACY_METHODS


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


def episodes_per_method(total_episodes: int, method_count: int,
                        overhead_episodes: int = 0) -> int:
    """Divide an exact run-wide budget after explicit non-PPO overhead.

    ``overhead_episodes`` is used by the three-pipeline experiment for the
    Shin-only estimator warm-up.  Legacy callers omit it and retain the old
    exact-division behavior.
    """
    total = int(total_episodes)
    count = int(method_count)
    overhead = int(overhead_episodes)
    if total < 1 or count < 1 or overhead < 0:
        raise ValueError("episode budget and method count must be positive")
    ppo_total = total - overhead
    if ppo_total < count:
        raise ValueError(
            f"total training budget {total} leaves fewer than one PPO episode "
            f"per method after {overhead} overhead episodes")
    if ppo_total % count:
        raise ValueError(
            f"PPO budget {ppo_total} (total {total} minus overhead {overhead}) "
            f"is not divisible by {count} methods")
    return ppo_total // count


def controlled_training_seeds(ppo_seed0: int, ppo_episodes: int, *,
                              warmup_episodes: int = 0,
                              warmup_seed0: int | None = None) -> list[int]:
    """Put disjoint estimator warm-up seeds before a common PPO seed range."""
    count = int(ppo_episodes)
    warmup = int(warmup_episodes)
    if count < 1 or warmup < 0:
        raise ValueError("PPO episodes must be positive and warm-up non-negative")
    start = int(ppo_seed0)
    warm_start = start - warmup if warmup_seed0 is None else int(warmup_seed0)
    warm_seeds = list(range(warm_start, warm_start + warmup))
    ppo_seeds = list(range(start, start + count))
    if set(warm_seeds) & set(ppo_seeds):
        raise ValueError("estimator warm-up and PPO seed ranges must be disjoint")
    return warm_seeds + ppo_seeds


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
