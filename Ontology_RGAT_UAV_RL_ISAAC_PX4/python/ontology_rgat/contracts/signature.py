"""Strict signature for model and trace reuse after scientific refactors."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping


@dataclass(frozen=True)
class CheckpointSignature:
    algorithm_version: str
    environment_config_hash: str
    observation_registry_hash: str
    reward_config_hash: str
    action_contract_hash: str
    graph_schema_hash: str
    pretrained_artifact_hash: str
    relation_partition_hash: str
    normalization_hash: str
    seed_contract_hash: str
    training_budget_hash: str

    @property
    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True,
                             separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def assert_compatible(self, stored: Mapping[str, str]) -> None:
        expected = asdict(self)
        mismatches = {key: (stored.get(key), value)
                      for key, value in expected.items()
                      if stored.get(key) != value}
        if mismatches:
            raise ValueError(f"checkpoint signature mismatch: {mismatches}")


def checkpoint_signature(**values: str) -> CheckpointSignature:
    return CheckpointSignature(**values)
