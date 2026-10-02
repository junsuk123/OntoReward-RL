#!/usr/bin/env python3
"""Seed planar learners from their best weights without losing progress.

The current/latest artifacts are copied into a timestamped recovery directory.
Each latest checkpoint keeps its completed episode, curriculum and history but
receives the corresponding best checkpoint's model weights. Reward traces are
append-only observations, so they remain in place and continue at the next
episode. Both arms receive a one-shot optimizer reset at the configured initial
learning rate. This makes recovery fair and avoids replaying completed flights.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

import torch


CONTRACT_ID = "planar_pair_provenance_kl_recovery_v2"
BASELINE = "shin_se_fixed"
PROPOSED = "shin_se_onto_rgat_state"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_torch_save(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".recovery-tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _running_pid(project: Path) -> int | None:
    pid_path = project / "logs/run_detached.pid"
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        return None


def _archive_copy(path: Path, archive: Path, records: list[dict]) -> None:
    if not path.is_file():
        return
    destination = archive / path.name
    shutil.copy2(path, destination)
    records.append({
        "source": str(path.resolve()),
        "archive": str(destination.resolve()),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(destination),
    })


def _archive_trace(model_dir: Path, method: str, archive: Path,
                   records: list[dict]) -> None:
    for suffix in ("_reward_steps.jsonl", "_reward_steps.manifest.json"):
        path = model_dir / f"{method}{suffix}"
        if not path.is_file():
            continue
        destination = archive / path.name
        shutil.copy2(path, destination)
        records.append({
            "source": str(path.resolve()),
            "archive": str(destination.resolve()),
            "bytes": int(path.stat().st_size),
            "sha256": _sha256(destination),
            "copied": True,
        })


def _recover_checkpoint(latest: dict, best: dict, *, method: str) -> dict:
    payload = dict(latest)
    payload["model"] = best["model"]
    payload["reset_optimizer_on_resume"] = True
    payload["recovery"] = {
        "method": method,
        "source": "best checkpoint model weights",
        "source_best_episode": int(best["episode"]),
        "preserved_progress_episode": int(latest["episode"]),
        "source_selection_score": best.get("selection_score"),
        "optimizer": "fresh Adam at configured initial learning rate",
        "contract_id": latest.get("training_contract_id"),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path("results/three_pipeline/full"))
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    running = _running_pid(project)
    if running is not None:
        raise SystemExit(
            f"refusing to rewrite live checkpoints while pipeline pid {running} is active")

    results = (project / args.results_dir).resolve()
    model_root = results / "models"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    archive = results / "recovery" / f"paired-best-weights-{timestamp}"
    archive.mkdir(parents=True, exist_ok=False)
    records: list[dict] = []

    baseline_dir = model_root / BASELINE
    proposed_dir = model_root / PROPOSED
    baseline_latest = baseline_dir / f"{BASELINE}.pt"
    baseline_best = baseline_dir / f"{BASELINE}.best.pt"
    proposed_latest = proposed_dir / f"{PROPOSED}.pt"
    proposed_best = proposed_dir / f"{PROPOSED}.best.pt"
    baseline_history = baseline_dir / f"{BASELINE}_training.csv"
    proposed_history = proposed_dir / f"{PROPOSED}_training.csv"

    required = (baseline_latest, baseline_best, proposed_latest,
                proposed_best, baseline_history, proposed_history)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing recovery input(s): " + ", ".join(missing))

    baseline = torch.load(baseline_latest, map_location="cpu", weights_only=False)
    baseline_best_payload = torch.load(
        baseline_best, map_location="cpu", weights_only=False)
    proposed = torch.load(proposed_latest, map_location="cpu", weights_only=False)
    proposed_best_payload = torch.load(
        proposed_best, map_location="cpu", weights_only=False)

    for method, latest, best in (
            (BASELINE, baseline, baseline_best_payload),
            (PROPOSED, proposed, proposed_best_payload)):
        if latest.get("training_contract_id") != CONTRACT_ID:
            raise SystemExit(
                f"{method} latest checkpoint has incompatible training contract")
        if best.get("training_contract_id") != CONTRACT_ID:
            raise SystemExit(
                f"{method} best checkpoint has incompatible training contract")
        if latest.get("config_hash") != best.get("config_hash"):
            raise SystemExit(f"{method} best/latest config hashes differ")
        if int(best.get("episode", -1)) > int(latest.get("episode", -1)):
            raise SystemExit(f"{method} best checkpoint is ahead of latest")

    for path in required:
        _archive_copy(path, archive, records)
    _archive_trace(baseline_dir, BASELINE, archive, records)
    _archive_trace(proposed_dir, PROPOSED, archive, records)

    _atomic_torch_save(
        baseline_latest,
        _recover_checkpoint(
            baseline, baseline_best_payload, method=BASELINE))
    _atomic_torch_save(
        proposed_latest,
        _recover_checkpoint(
            proposed, proposed_best_payload, method=PROPOSED))

    manifest = {
        "format": "ontology-rgat-training-recovery-v1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "training_contract_id": CONTRACT_ID,
        "proposed": {
            "method": PROPOSED,
            "source_best_episode": int(proposed_best_payload["episode"]),
            "preserved_progress_episode": int(proposed["episode"]),
            "optimizer_reset_on_resume": True,
        },
        "baseline": {
            "method": BASELINE,
            "source_best_episode": int(baseline_best_payload["episode"]),
            "preserved_progress_episode": int(baseline["episode"]),
            "optimizer_reset_on_resume": True,
        },
        "reward_trace_policy": (
            "archived copy plus in-place append; episode progress preserved"),
        "archives": records,
    }
    manifest_path = archive / "recovery_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"Recovery archive: {archive}")
    print(
        f"Proposed: best weights episode {int(proposed_best_payload['episode'])}; "
        f"progress episode {int(proposed['episode'])} preserved")
    print(
        f"Baseline: best weights episode {int(baseline_best_payload['episode'])}; "
        f"progress episode {int(baseline['episode'])} preserved")
    print(f"Training contract: {CONTRACT_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
