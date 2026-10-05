#!/usr/bin/env python3
"""Byte-level behavioural snapshot of the 2D and 3D task contracts.

Refactoring toward one axis-generic contract must not silently move the ported
2D reference or the 3D task. This records what a refactor could change --
registry/graph hashes, every packet value, every graph cell, every reward term
and every terminal -- over fixed seeds under a fixed open-loop action script.
No policy, no training, no simulator truth leaves the evaluator.

    python tools/contract_snapshot.py --out before.json
    ... refactor ...
    python tools/contract_snapshot.py --out after.json
    python tools/contract_snapshot.py --compare before.json after.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

SEEDS = (3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008)
STEPS = 40


def scripted_action(step: int, width: int) -> np.ndarray:
    """Deterministic open-loop probe: no policy, no state feedback, no truth.

    Sweeps each axis independently so a channel that silently moves between
    axes changes the digest instead of cancelling out.
    """
    phase = np.arange(width) * 1.7 + 0.3
    return np.clip(0.6 * np.sin(0.21 * step + phase), -1.0, 1.0)


def digest(array) -> str:
    data = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    return hashlib.sha256(data.tobytes()).hexdigest()[:16]


def snapshot_two_axis(config_path: Path) -> dict:
    from ontology_rgat.two_axis.config import load_config
    from ontology_rgat.two_axis.environment import TwoAxisLandingEnv

    cfg = load_config(config_path)
    env = TwoAxisLandingEnv(cfg)
    episodes = []
    for seed in SEEDS:
        obs, _ = env.reset(seed=seed)
        packets, graphs, rewards, terms = [], [], [], []
        status = "RUNNING"
        for step in range(STEPS):
            packets.append(np.asarray(obs.packet.values, dtype=float))
            graphs.append(np.asarray(obs.graph.X, dtype=float).ravel())
            obs, reward, done, _, info = env.step(scripted_action(step, 2))
            rewards.append(float(reward))
            terms.append([float(v) for v in sorted(
                (info.get("reward_terms") or {}).items())] if isinstance(
                    info.get("reward_terms"), dict) else [])
            if done:
                status = str(info.get("terminal_reason") or info.get("status") or "DONE")
                break
        episodes.append({
            "seed": seed, "steps": len(rewards), "status": status,
            "packet_digest": digest(packets), "graph_digest": digest(graphs),
            "reward_digest": digest(rewards),
            "reward_sum": round(float(np.sum(rewards)), 9),
        })
    return {
        "route": "two_axis",
        "config": str(config_path.relative_to(ROOT)),
        "registry_sha256": env.registry.sha256,
        "terminal_table": {k: float(v) for k, v in sorted(
            dict(cfg.reward.terminal_bonus).items())},
        "discount_time_constant_s": float(cfg.timing.discount_time_constant_s),
        "curriculum_start_unsafe": float(cfg.curriculum.start_unsafe_contact_penalty),
        "episodes": episodes,
    }


def snapshot_spatial(schema: str) -> dict:
    from dataclasses import replace
    from ontology_rgat.spatial.core import SpatialConfig
    from ontology_rgat.spatial.environment import SpatialLandingEnv, Evaluator

    cfg = SpatialConfig() if schema is None else replace(SpatialConfig(), schema=schema)
    env = SpatialLandingEnv(cfg)
    episodes = []
    for seed in SEEDS:
        obs, _ = env.reset(seed=seed)
        packets, graphs, rewards = [], [], []
        status = "RUNNING"
        for step in range(STEPS):
            packets.append(np.asarray(obs.packet.values, dtype=float))
            graphs.append(np.asarray(obs.graph.X, dtype=float).ravel())
            obs, reward, done, _, info = env.step(scripted_action(step, 3))
            rewards.append(float(reward))
            if done:
                status = str(info["status"])
                break
        episodes.append({
            "seed": seed, "steps": len(rewards), "status": status,
            "packet_digest": digest(packets), "graph_digest": digest(graphs),
            "reward_digest": digest(rewards),
            "reward_sum": round(float(np.sum(rewards)), 9),
        })
    env.close()
    return {
        "route": "spatial",
        "schema": cfg.schema,
        "registry_sha256": cfg.registry_hash,
        "packet_fields": list(cfg.packet_fields),
        "terminal_table": {k: float(v) for k, v in sorted(Evaluator.BONUSES.items())},
        "discount_time_constant_s": float(cfg.discount_tau),
        "curriculum_start_unsafe": float(
            cfg.curriculum.start_unsafe_contact_penalty),
        "episodes": episodes,
    }


def _identity(contract: dict) -> str:
    return f"{contract['route']}:{contract.get('schema') or contract.get('config')}"


def compare(before: Path, after: Path) -> int:
    """Diff two snapshots, matching contracts by identity rather than position.

    Adding a contract must not make every other one look changed; by index it
    did, and the result was an unreadable dump of both files.
    """
    a, b = json.loads(before.read_text()), json.loads(after.read_text())
    differences = []

    def walk(path, x, y):
        if isinstance(x, dict) and isinstance(y, dict):
            for key in sorted(set(x) | set(y)):
                walk(f"{path}.{key}", x.get(key), y.get(key))
        elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
            for i, (u, v) in enumerate(zip(x, y)):
                walk(f"{path}[{i}]", u, v)
        elif x != y:
            differences.append((path, x, y))

    left = {_identity(c): c for c in a.pop("contracts", [])}
    right = {_identity(c): c for c in b.pop("contracts", [])}
    walk("", a, b)
    for name in sorted(set(left) & set(right)):
        walk(f"[{name}]", left[name], right[name])
    added = sorted(set(right) - set(left))
    removed = sorted(set(left) - set(right))
    shared = sorted(set(left) & set(right))
    if not differences and not removed:
        print(f"identical on {len(shared)} shared contract(s): "
              + ", ".join(shared))
        if added:
            print("  new (not in the baseline): " + ", ".join(added))
        return 0
    print(f"{len(differences)} difference(s)"
          + (f", {len(removed)} contract(s) removed" if removed else "") + ":")
    for name in removed:
        print(f"  MISSING contract {name}")
    for path, x, y in differences:
        print(f"  {path}\n    before: {x}\n    after:  {y}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BEFORE", "AFTER"))
    parser.add_argument("--reference-config", type=Path,
                        default=ROOT / "config/experiments/two_axis_reference_v28_active.yaml")
    parser.add_argument("--spatial-schemas", nargs="*", default=["spatial-causal-rgat/5"])
    args = parser.parse_args()
    if args.compare:
        return compare(*args.compare)
    if not args.out:
        parser.error("--out or --compare is required")
    payload = {
        "steps_per_episode": STEPS,
        "seeds": list(SEEDS),
        "action_script": "0.6*sin(0.21*step + 1.7*axis + 0.3), clipped",
        "contracts": [snapshot_two_axis(args.reference_config)]
        + [snapshot_spatial(schema) for schema in args.spatial_schemas],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {args.out}")
    for contract in payload["contracts"]:
        name = contract.get("schema", contract.get("config"))
        statuses = {}
        for episode in contract["episodes"]:
            statuses[episode["status"]] = statuses.get(episode["status"], 0) + 1
        print(f"  {contract['route']:9s} {name}  registry={contract['registry_sha256'][:12]}"
              f"  tau={contract['discount_time_constant_s']}  {statuses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
