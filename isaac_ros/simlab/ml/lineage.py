"""Model-lineage helpers for safe continual training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from simlab.config.schema import SceneConfig


def drone_model_signature(scene: SceneConfig) -> str:
    """Identify the visual aircraft configuration, independent of fleet count."""
    payload = {
        "friendly": {
            "model": scene.drones.friendly.model,
            "asset_scale": scene.drones.friendly.asset_scale,
        },
        "enemy": {
            "model": scene.drones.enemy.model,
            "asset_scale": scene.drones.enemy.asset_scale,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def lineage_dir(artifacts: Path, signature: str) -> Path:
    return artifacts / "model_lineages" / signature


def read_model_pointer(path: Path) -> str | None:
    if not path.exists():
        return None
    model = path.read_text(encoding="utf-8").strip()
    return model if model and Path(model).is_file() else None


def write_model_pointer(path: Path, model: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(model.resolve()) + "\n", encoding="utf-8")


def active_model_for_signature(artifacts: Path, signature: str) -> str | None:
    signature_file = artifacts / "active_signature.txt"
    if not signature_file.exists() or signature_file.read_text(encoding="utf-8").strip() != signature:
        return None
    return read_model_pointer(artifacts / "active_model.txt")


def latest_model_for_signature(artifacts: Path, signature: str) -> str | None:
    return read_model_pointer(lineage_dir(artifacts, signature) / "latest_model.txt")


# -- session ledgers --------------------------------------------------------
# Two questions have to be answered separately on every start-up: which
# collected sessions are already in the training set, and which of those the
# deployed weights have actually seen. A session can be in the dataset but not
# in the model -- that is exactly the "not yet learned" state the orchestrator
# looks for.

def dataset_dir(artifacts: Path, signature: str) -> Path:
    """The cumulative dataset for one airframe configuration."""
    return artifacts / "datasets" / f"cumulative_{signature}"


def ingested_sessions(dataset: Path) -> set[str]:
    """Collection sessions already folded into the dataset."""
    directory = dataset / "sessions"
    if not directory.is_dir():
        return set()
    return {path.stem for path in directory.glob("*.json")}


def pending_raw_sessions(raw_root: Path, dataset: Path) -> list[Path]:
    """Collected sessions on disk that the dataset has not absorbed yet."""
    if not raw_root.is_dir():
        return []
    known = ingested_sessions(dataset)
    return sorted(
        path
        for path in raw_root.iterdir()
        if path.is_dir() and path.name not in known and (path / "prompts.jsonl").is_file()
    )


def trained_sessions(lineage: Path) -> set[str]:
    path = lineage / "trained_sessions.json"
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return set(payload.get("sessions", []))


def record_trained_sessions(lineage: Path, sessions: Iterable[str], model: Path) -> None:
    """Mark sessions as learned. Only called once a training run has finished."""
    lineage.mkdir(parents=True, exist_ok=True)
    path = lineage / "trained_sessions.json"
    known = trained_sessions(lineage) | set(sessions)
    path.write_text(
        json.dumps({"model": str(model.resolve()), "sessions": sorted(known)}, indent=2),
        encoding="utf-8",
    )


def untrained_sessions(dataset: Path, lineage: Path) -> list[str]:
    """Ingested sessions no training run has covered yet."""
    return sorted(ingested_sessions(dataset) - trained_sessions(lineage))
