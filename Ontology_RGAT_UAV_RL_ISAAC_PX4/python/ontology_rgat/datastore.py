"""Accumulating SQLite store for the flight data offline training consumes.

Collecting a landing episode costs real simulator time, and until now every
run threw its collection away: the FOV-risk rollouts and the keypoint
calibration frames live in one run's results directory and are reused only
when that exact directory comes back. This keeps them in one database so a
later run can train on everything collected so far.

*What may be reused, and what may not, is the whole design of this module.*

On-policy trajectories may never be reused. PPO's update is only valid for
data drawn from the policy being updated; replaying a stored rollout into it
is wrong however the samples are weighted. The learner's own resume path
already continues from a checkpoint with *fresh* seeds rather than
re-consuming anything, and :meth:`CollectedDataStore.store_episode` refuses
the on-policy kinds outright rather than leaving that to a caller's
discipline.

What is safe is the offline *supervised* collection -- the future-FOV-risk
graphs and the geometry-labelled keypoint frames. Every model trained from
them is built from scratch each time (a fresh ``FOVRiskModel``; a keypoint
encoder restarted from the synthetic weights), so a growing store gives those
trainings more data rather than more epochs over the same data.

Two traps are closed explicitly here, because both would bias a run in exactly
the way repeating data on a resumed checkpoint would:

* **Double weighting.** Re-collecting an episode the store already holds must
  not add a second copy of it. A row's identity is a digest of what the
  episode *is* -- its kind, its data fingerprint, its seed and the policy that
  flew it -- so a re-run of an interrupted collection is idempotent.
* **A split that moves.** If the train/validation split were redrawn whenever
  the store grew, an episode held out in one run would train in the next, and
  the validation loss that selects the frozen artifact would be measured on
  data its own lineage had already fitted. Each row carries a split hash
  derived from its *seed*, so the side an episode falls on is a pure function
  of the episode and the requested fraction -- never of what else happens to
  be in the store, and never of collection order. Two flights of one initial
  condition therefore always land on the same side.

Reuse is gated on a ``fingerprint``: a digest of everything that decides what
the data *means* -- the graph and feature definitions, the camera and pad
geometry, the prediction horizon, the trajectory distribution. It is
deliberately not the experiment's whole configuration hash, which also covers
learning rates and episode budgets that change nothing about a collected
frame. Rows whose fingerprint no longer matches stay in the database for audit
and are simply invisible to the new architecture.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping, Sequence
import uuid

import numpy as np


SCHEMA_VERSION = 1

# Kinds whose samples are only valid for the policy that produced them. They
# are named so that an attempt to persist them fails loudly instead of
# quietly creating a replay buffer PPO must not have.
ON_POLICY_KINDS = frozenset({
    "ppo_rollout", "policy_transition", "on_policy_trajectory",
})

KIND_FOV_RISK = "fov_risk_episode"
KIND_KEYPOINT_CALIBRATION = "keypoint_calibration_viewpoint"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> str:
    """A stable JSON rendering, so a fingerprint does not depend on key order."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      default=_canonical_default, allow_nan=False)


def _canonical_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value)!r} is not part of a stable fingerprint")


def data_fingerprint(parts: Mapping[str, Any]) -> str:
    """Digest the properties that decide what a collected sample means.

    Callers pass the geometry, the graph definition and the trajectory
    distribution -- never learning rates or episode budgets. Two runs that
    agree on this may share data; two that do not must not.
    """
    if not parts:
        raise ValueError("a data fingerprint needs at least one property")
    return hashlib.sha256(_canonical(dict(parts)).encode("utf-8")).hexdigest()


def _episode_uid(kind: str, fingerprint: str, identity: Mapping[str, Any]) -> str:
    if not identity:
        raise ValueError("a stored episode needs an identity to deduplicate on")
    payload = _canonical({"kind": str(kind), "fingerprint": str(fingerprint),
                          "identity": dict(identity)})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _split_hash(kind: str, fingerprint: str, split_key: str) -> float:
    """A stable [0,1) coordinate for one initial condition.

    Keyed on the *seed group* rather than the episode, so re-flying a seed
    under a new policy cannot put the same initial condition on both sides of
    the split.
    """
    digest = hashlib.sha256(
        f"{kind}\x00{fingerprint}\x00{split_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _encode_payload(payload: Mapping[str, np.ndarray]) -> bytes:
    if not payload:
        raise ValueError("a stored episode needs at least one array")
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{
        str(name): np.asarray(value) for name, value in payload.items()})
    return buffer.getvalue()


def _decode_payload(blob: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


@dataclass(frozen=True)
class StoredEpisode:
    """One reusable unit of collection, with its provenance."""

    uid: str
    kind: str
    fingerprint: str
    seed: int
    split_key: str
    split_hash: float
    samples: int
    environment_steps: int
    payload_sha256: str
    provenance: dict[str, Any]
    run_id: str
    created_at: str
    _payload: bytes | None

    def payload(self) -> dict[str, np.ndarray]:
        if self._payload is None:
            raise ValueError(
                f"{self.uid[:12]} was read without its payload; query the "
                "store again with payloads=True to use its arrays")
        return _decode_payload(self._payload)

    def split(self, validation_fraction: float) -> str:
        return ("validation" if self.split_hash < float(validation_fraction)
                else "train")


class CollectedDataStore:
    """SQLite-backed accumulation of reusable, offline-supervised collection."""

    def __init__(self, path: str | Path, *, run_id: str | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = str(run_id or uuid.uuid4())
        self._connection = sqlite3.connect(str(self.path), timeout=30.0)
        self._connection.row_factory = sqlite3.Row
        # Several pair workers write from one process, and a run can be
        # interrupted at any point; WAL keeps a reader from blocking them and
        # keeps a half-written transaction from corrupting the accumulation.
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    # ---------------------------------------------------------------- schema
    def _create_schema(self) -> None:
        with self._connection as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS run (
                    run_id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    experiment TEXT,
                    results_dir TEXT,
                    config_hash TEXT,
                    notes TEXT);
                CREATE TABLE IF NOT EXISTS episode (
                    uid TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    split_key TEXT NOT NULL,
                    split_hash REAL NOT NULL,
                    samples INTEGER NOT NULL,
                    environment_steps INTEGER NOT NULL,
                    payload BLOB NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    provenance TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    created_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS episode_by_architecture
                    ON episode (kind, fingerprint);
                CREATE TABLE IF NOT EXISTS consumption (
                    artifact_sha256 TEXT NOT NULL,
                    uid TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    trained_from_scratch INTEGER NOT NULL,
                    validation_fraction REAL NOT NULL,
                    split TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    PRIMARY KEY (artifact_sha256, uid));
                """)
            version = connection.execute(
                "SELECT version FROM schema_version").fetchone()
            if version is None:
                connection.execute(
                    "INSERT INTO schema_version (version) VALUES (?)",
                    (SCHEMA_VERSION,))
            elif int(version["version"]) != SCHEMA_VERSION:
                raise ValueError(
                    f"{self.path} holds datastore schema v{version['version']}, "
                    f"not v{SCHEMA_VERSION}; point --datastore at a new file "
                    "rather than mixing incompatible accumulations.")

    # ------------------------------------------------------------------ runs
    def record_run(self, *, experiment: str | None = None,
                   results_dir: str | Path | None = None,
                   config_hash: str | None = None,
                   notes: str | None = None) -> str:
        with self._connection as connection:
            connection.execute(
                "INSERT OR REPLACE INTO run "
                "(run_id, started_at, experiment, results_dir, config_hash, notes) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (self.run_id, _utc_now(), experiment,
                 None if results_dir is None else str(results_dir),
                 config_hash, notes))
        return self.run_id

    # -------------------------------------------------------------- episodes
    def store_episode(self, kind: str, fingerprint: str, *,
                      seed: int,
                      payload: Mapping[str, np.ndarray],
                      identity: Mapping[str, Any],
                      provenance: Mapping[str, Any] | None = None,
                      samples: int | None = None,
                      environment_steps: int = 0,
                      split_key: str | None = None) -> tuple[str, bool]:
        """Add one collected episode, ignoring a re-collection of the same one.

        Returns ``(uid, inserted)``. ``inserted`` is ``False`` when the store
        already held this identity, which is what makes resuming an
        interrupted collection idempotent instead of double-weighting it.
        """
        kind = str(kind)
        if kind in ON_POLICY_KINDS:
            raise ValueError(
                f"{kind!r} is on-policy data. Replaying it into PPO is invalid "
                "for the policy being updated, so it is not stored for reuse; "
                "the learner resumes from its checkpoint with fresh seeds.")
        blob = _encode_payload(payload)
        uid = _episode_uid(kind, fingerprint, identity)
        key = str(seed) if split_key is None else str(split_key)
        record = (uid, kind, str(fingerprint), int(seed), key,
                  _split_hash(kind, str(fingerprint), key),
                  int(samples if samples is not None
                      else len(next(iter(payload.values())))),
                  int(environment_steps), blob,
                  hashlib.sha256(blob).hexdigest(),
                  _canonical(dict(provenance or {})), self.run_id, _utc_now())
        with self._connection as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO episode (uid, kind, fingerprint, seed, "
                "split_key, split_hash, samples, environment_steps, payload, "
                "payload_sha256, provenance, run_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", record)
            return uid, bool(cursor.rowcount)

    def episodes(self, kind: str, fingerprint: str, *,
                 limit: int | None = None,
                 payloads: bool = True) -> list[StoredEpisode]:
        """Every stored episode this architecture may reuse, oldest first.

        ``payloads=False`` leaves the arrays in the database. A store that has
        accumulated calibration frames over many runs holds megabytes of
        images, and counting them or reading their split does not need them.
        """
        columns = "*" if payloads else (
            "uid, kind, fingerprint, seed, split_key, split_hash, samples, "
            "environment_steps, payload_sha256, provenance, run_id, created_at")
        query = (f"SELECT {columns} FROM episode WHERE kind = ? AND "
                 "fingerprint = ? ORDER BY created_at, uid")
        parameters: tuple[Any, ...] = (str(kind), str(fingerprint))
        if limit is not None:
            query += " LIMIT ?"
            parameters += (int(limit),)
        return [StoredEpisode(
            uid=row["uid"], kind=row["kind"], fingerprint=row["fingerprint"],
            seed=int(row["seed"]), split_key=row["split_key"],
            split_hash=float(row["split_hash"]), samples=int(row["samples"]),
            environment_steps=int(row["environment_steps"]),
            payload_sha256=row["payload_sha256"],
            provenance=json.loads(row["provenance"]), run_id=row["run_id"],
            created_at=row["created_at"],
            _payload=row["payload"] if payloads else None)
            for row in self._connection.execute(query, parameters)]

    def stored_identities(self, kind: str, fingerprint: str) -> set[str]:
        return {row["uid"] for row in self._connection.execute(
            "SELECT uid FROM episode WHERE kind = ? AND fingerprint = ?",
            (str(kind), str(fingerprint)))}

    def holds(self, kind: str, fingerprint: str,
              identity: Mapping[str, Any]) -> bool:
        return bool(self._connection.execute(
            "SELECT 1 FROM episode WHERE uid = ?",
            (_episode_uid(kind, fingerprint, identity),)).fetchone())

    def summary(self, kind: str, fingerprint: str, *,
                validation_fraction: float = 0.2) -> dict[str, Any]:
        rows = self.episodes(kind, fingerprint, payloads=False)
        sides = [row.split(validation_fraction) for row in rows]
        return {
            "kind": str(kind), "fingerprint": str(fingerprint),
            "episodes": len(rows),
            "training_episodes": sides.count("train"),
            "validation_episodes": sides.count("validation"),
            "samples": int(sum(row.samples for row in rows)),
            "environment_steps": int(sum(row.environment_steps for row in rows)),
            "seeds": sorted({row.seed for row in rows}),
            "runs": sorted({row.run_id for row in rows}),
        }

    # ----------------------------------------------------------- consumption
    def record_consumption(self, artifact_sha256: str,
                           episodes: Sequence[StoredEpisode], *,
                           validation_fraction: float,
                           trained_from_scratch: bool) -> int:
        """Record which episodes went into one frozen artifact, and how.

        ``trained_from_scratch`` is written down rather than assumed: an
        artifact that was fine-tuned on an accumulation has seen its older
        episodes more often than its newer ones, and that is exactly the bias
        this ledger exists to make visible after the fact.
        """
        stamp = _utc_now()
        rows = [(str(artifact_sha256), episode.uid, episode.kind,
                 episode.fingerprint, int(bool(trained_from_scratch)),
                 float(validation_fraction),
                 episode.split(validation_fraction), self.run_id, stamp)
                for episode in episodes]
        with self._connection as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO consumption (artifact_sha256, uid, "
                "kind, fingerprint, trained_from_scratch, validation_fraction, "
                "split, run_id, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows)
        return len(rows)

    def consumed_by(self, artifact_sha256: str) -> set[str]:
        return {row["uid"] for row in self._connection.execute(
            "SELECT uid FROM consumption WHERE artifact_sha256 = ?",
            (str(artifact_sha256),))}

    # ------------------------------------------------------------- lifecycle
    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "CollectedDataStore":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def open_datastore(path: str | Path | None, **kwargs) -> CollectedDataStore | None:
    """Open the store, or return ``None`` when accumulation is switched off."""
    if path is None:
        return None
    return CollectedDataStore(path, **kwargs)


def split_episodes(episodes: Iterable[StoredEpisode], *,
                   validation_fraction: float) -> tuple[list, list]:
    """Partition stored episodes on their frozen split coordinate."""
    training, validation = [], []
    for episode in episodes:
        (validation if episode.split(validation_fraction) == "validation"
         else training).append(episode)
    return training, validation
