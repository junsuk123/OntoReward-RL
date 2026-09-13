"""상태 적응형 보상 가중치용 실제 rollout 데이터셋 계약."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..reward_modes.adaptive_weight import RewardComponentNormalizer
from .adaptive_model import (
    ADAPTIVE_GRAPH_INPUT_DIM, ADAPTIVE_GRAPH_VERSION, ADAPTIVE_NODE_NAMES,
    ADAPTIVE_RELATION_NAMES, empty_adaptive_reward_graph)


ADAPTIVE_DATASET_FORMAT = "ontology_rgat.adaptive_reward_rollouts/2-stratified-episode"
TRANSITION_FIELDS = frozenset({
    "graph_X", "rho_raw", "episode_id", "time_index", "success",
    "failure_type", "touchdown_error", "touchdown_vertical_speed",
    "touchdown_roll", "touchdown_pitch", "duration", "terminal_reason",
    "scenario", "seed", "phase", "disturbance_level",
})
OUTCOME_ONLY_FIELDS = frozenset({
    "success", "failure_type", "touchdown_error", "touchdown_vertical_speed",
    "touchdown_roll", "touchdown_pitch", "duration", "terminal_reason",
})


def adaptive_episode_records(rows: Sequence[Mapping[str, Any]], metric: Mapping[str, Any],
                             *, episode_id: int, seed: int, scenario: str,
                             phase: str = "offline_design") -> list[dict[str, Any]]:
    """``collect_episode`` 결과를 명시적 transition/episode schema로 좁힌다."""
    if not rows:
        raise ValueError("adaptive reward episode cannot be empty")
    status = str(metric.get("status", "failure"))
    success = bool(float(metric.get("paper_success", 0.0)))
    disturbance = float(metric.get(
        "domain_external_force_n", metric.get("disturbance_level", 0.0)))
    common = {
        "episode_id": int(episode_id), "success": success,
        "failure_type": "none" if success else status,
        "touchdown_error": float(metric["touchdown_lateral_error"]),
        "touchdown_vertical_speed": float(metric["touchdown_vertical_velocity"]),
        "touchdown_roll": float(metric.get("touchdown_roll", 0.0)),
        "touchdown_pitch": float(metric.get("touchdown_pitch", 0.0)),
        "duration": float(metric.get("touchdown_time_s", len(rows))),
        "terminal_reason": status, "scenario": str(scenario), "seed": int(seed),
        "phase": str(phase), "disturbance_level": disturbance,
    }
    records = []
    for index, row in enumerate(rows):
        graph = np.asarray(row.get("adaptive_graph_X"), dtype=np.float32)
        rho = np.asarray(row.get("rho_raw"), dtype=np.float32)
        if graph.shape != (ADAPTIVE_GRAPH_INPUT_DIM, len(ADAPTIVE_NODE_NAMES)):
            raise ValueError("episode row lacks the adaptive semantic graph")
        if rho.shape != (5,) or not np.isfinite(rho).all():
            raise ValueError("episode row lacks five finite raw reward components")
        records.append({**common, "graph_X": graph, "rho_raw": rho,
                        "time_index": int(row.get("time_index", index))})
    return records


def _array(dataset, name, dtype=None):
    return np.asarray(dataset[name], dtype=dtype)


def validate_adaptive_dataset(dataset: Mapping[str, Any], *,
                              require_both_classes=False):
    required = {
        "X", "rho_raw", "rho_normalized", "episode_id", "time_index",
        "success", "failure_type", "touchdown_error",
        "touchdown_vertical_speed", "touchdown_roll", "touchdown_pitch",
        "duration", "terminal_reason", "scenario", "seed", "phase",
        "disturbance_level", "split", "normalization",
        "edge_src", "edge_dst", "edge_relation",
    }
    missing = required - set(dataset)
    if missing:
        raise ValueError(f"adaptive dataset missing fields: {sorted(missing)}")
    X = _array(dataset, "X", np.float32)
    raw = _array(dataset, "rho_raw", np.float32)
    normalized = _array(dataset, "rho_normalized", np.float32)
    n = X.shape[0] if X.ndim else 0
    if X.shape != (n, len(ADAPTIVE_NODE_NAMES), ADAPTIVE_GRAPH_INPUT_DIM):
        raise ValueError("adaptive graph data has an invalid shape")
    if raw.shape != (n, 5) or normalized.shape != (n, 5):
        raise ValueError("adaptive reward component data must be [transition,5]")
    if n == 0 or not (np.isfinite(X).all() and np.isfinite(raw).all()
                      and np.isfinite(normalized).all()):
        raise ValueError("adaptive dataset must contain finite transitions")
    template = empty_adaptive_reward_graph()
    for name, expected in (("edge_src", template.src), ("edge_dst", template.dst),
                           ("edge_relation", template.rel)):
        value = _array(dataset, name, np.int64)
        if value.shape != expected.shape or not np.array_equal(value, expected):
            raise ValueError(f"adaptive dataset {name} topology mismatch")
    vector_fields = required - {
        "X", "rho_raw", "rho_normalized", "normalization",
        "edge_src", "edge_dst", "edge_relation"}
    for name in vector_fields:
        if _array(dataset, name).shape != (n,):
            raise ValueError(f"adaptive dataset field {name} must have one value per transition")
    episode = _array(dataset, "episode_id", np.int64)
    time_index = _array(dataset, "time_index", np.int64)
    seed = _array(dataset, "seed", np.int64)
    success = _array(dataset, "success", np.int64)
    if not set(np.unique(success)).issubset({0, 1}):
        raise ValueError("adaptive dataset outcomes must be binary")
    if require_both_classes and set(np.unique(success)) != {0, 1}:
        raise ValueError("adaptive dataset requires success and failure episodes")
    # Episode metadata must be constant and time indices unique/ordered.
    scenario = _array(dataset, "scenario").astype(str)
    split = _array(dataset, "split").astype(str)
    if not set(np.unique(split)).issubset({"train", "validation", "test"}):
        raise ValueError("adaptive dataset split must be train/validation/test")
    group_split = {}
    for ep in np.unique(episode):
        idx = np.flatnonzero(episode == ep)
        for name in ("success", "failure_type", "touchdown_error",
                     "touchdown_vertical_speed", "touchdown_roll",
                     "touchdown_pitch", "duration", "terminal_reason",
                     "scenario", "seed", "phase", "disturbance_level", "split"):
            if np.unique(_array(dataset, name)[idx]).size != 1:
                raise ValueError(f"episode-level field {name} changes within episode {ep}")
        if np.unique(time_index[idx]).size != idx.size:
            raise ValueError(f"duplicate time index in episode {ep}")
        key = (int(ep), str(scenario[idx[0]]))
        group_split[key] = str(split[idx[0]])
    # This explicit group map is what prevents transition-level leakage.
    if len(group_split) != len(np.unique(episode)):
        raise ValueError("episode identifiers must not span scenarios")
    if not np.any(split == "train"):
        raise ValueError("adaptive dataset has no training episodes")
    return X, raw, normalized


def _episode_split(records: Sequence[Mapping[str, Any]], *, seed: int,
                   validation_fraction: float, test_fraction: float):
    groups = sorted({(int(row["episode_id"]), str(row["scenario"])) for row in records})
    if len(groups) < 2:
        raise ValueError("episode/scenario split requires at least two episodes")
    if not 0.0 <= validation_fraction < 1.0 or not 0.0 <= test_fraction < 1.0:
        raise ValueError("split fractions must be in [0,1)")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation and test fractions leave no training data")
    # Split complete episodes and stratify by terminal class.  The old random
    # split could put the sole validation episode in one class (the six-flight
    # seminar run did exactly that), making 0/1 validation accuracy meaningless.
    # We keep at least one item of every sufficiently represented class in both
    # training and validation.
    outcome = {}
    for group in groups:
        values = {bool(row["success"]) for row in records
                  if (int(row["episode_id"]), str(row["scenario"])) == group}
        if len(values) != 1:
            raise ValueError(f"terminal class changes within episode group {group}")
        outcome[group] = int(values.pop())
    rng = np.random.default_rng(int(seed))
    strata = {}
    for group in groups:
        strata.setdefault(outcome[group], []).append(group)
    for label, values in strata.items():
        strata[label] = [values[index] for index in rng.permutation(len(values))]

    n_test = int(round(test_fraction * len(groups)))
    test = set()
    if n_test:
        candidates = [group for label in sorted(strata) for group in strata[label]
                      if len(strata[label]) >= 3]
        for group in candidates[:n_test]:
            test.add(group)

    available = [group for group in groups if group not in test]
    n_val = int(round(validation_fraction * len(groups)))
    if validation_fraction > 0.0:
        n_val = max(1, n_val)
    represented = [label for label, values in strata.items()
                   if sum(group not in test for group in values) >= 2]
    if validation_fraction > 0.0 and len(represented) > 1:
        n_val = max(n_val, len(represented))
    n_val = min(n_val, max(0, len(available) - 1))
    validation = set()
    for label in sorted(represented):
        candidate = next((group for group in strata[label]
                          if group not in test), None)
        if candidate is not None and len(validation) < n_val:
            validation.add(candidate)
    remainder = [group for label in sorted(strata) for group in strata[label]
                 if group not in test and group not in validation
                 and sum(item not in test and item not in validation
                         for item in strata[label]) > 1]
    validation.update(remainder[:max(0, n_val - len(validation))])
    return {group: ("test" if group in test else
                    "validation" if group in validation else "train")
            for group in groups}


def build_adaptive_dataset(records: Sequence[Mapping[str, Any]], *, seed=42,
                           validation_fraction=0.2, test_fraction=0.0,
                           physical_scales=None, normalization_quantile=0.99,
                           exact_paper_raw=False) -> dict[str, Any]:
    """실제 episode transition record를 검증하고 dataset으로 변환한다."""
    rows = list(records)
    if not rows:
        raise ValueError("cannot build adaptive dataset from no transitions")
    for index, row in enumerate(rows):
        if set(row) != TRANSITION_FIELDS:
            missing = TRANSITION_FIELDS - set(row)
            extra = set(row) - TRANSITION_FIELDS
            raise ValueError(
                f"transition {index} schema mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    split_map = _episode_split(
        rows, seed=int(seed), validation_fraction=float(validation_fraction),
        test_fraction=float(test_fraction))
    X = np.stack([np.asarray(row["graph_X"], dtype=np.float32).T for row in rows])
    raw = np.stack([np.asarray(row["rho_raw"], dtype=np.float32) for row in rows])
    split = np.asarray([split_map[(int(row["episode_id"]), str(row["scenario"]))]
                        for row in rows])
    if exact_paper_raw:
        normalizer = RewardComponentNormalizer(exact_paper_raw=True,
                                               source="exact_paper_raw")
    else:
        normalizer = RewardComponentNormalizer.fit_training_split(
            raw[split == "train"], physical_scales=physical_scales,
            quantile=float(normalization_quantile))
    dataset = {"X": X, "rho_raw": raw,
               "rho_normalized": normalizer.transform(raw), "split": split,
               "normalization": normalizer.to_manifest()}
    template = empty_adaptive_reward_graph()
    dataset.update({"edge_src": template.src.copy(),
                    "edge_dst": template.dst.copy(),
                    "edge_relation": template.rel.copy()})
    for name in TRANSITION_FIELDS - {"graph_X", "rho_raw"}:
        dataset[name] = np.asarray([row[name] for row in rows])
    validate_adaptive_dataset(dataset)
    return dataset


def adaptive_dataset_digest(dataset: Mapping[str, Any]) -> str:
    validate_adaptive_dataset(dataset)
    digest = hashlib.sha256()
    for name in sorted(set(dataset) - {"normalization"}):
        value = np.ascontiguousarray(np.asarray(dataset[name]))
        digest.update(name.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.dtype.str.encode())
        digest.update(value.tobytes())
    digest.update(json.dumps(dataset["normalization"], sort_keys=True).encode())
    return digest.hexdigest()


def save_adaptive_dataset(dataset: Mapping[str, Any], path: str | Path, *,
                          config_hash: str, source_behavior_policy: Mapping[str, Any]):
    validate_adaptive_dataset(dataset)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.asarray(value) for name, value in dataset.items()
              if name != "normalization"}
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)
    episode_ids = np.unique(np.asarray(dataset["episode_id"], dtype=np.int64))
    success = np.asarray(dataset["success"], dtype=np.int64)
    representative = [int(np.flatnonzero(dataset["episode_id"] == ep)[0])
                      for ep in episode_ids]
    failure_types = [str(dataset["failure_type"][index])
                     for index in representative if not int(success[index])]
    near_miss = sum(
        not int(success[index])
        and float(dataset["touchdown_error"][index]) <= 0.70
        and abs(float(dataset["touchdown_vertical_speed"][index])) <= 1.0
        for index in representative)
    manifest = {
        "format": ADAPTIVE_DATASET_FORMAT,
        "graph_schema_version": ADAPTIVE_GRAPH_VERSION,
        "dataset_config_hash": str(config_hash),
        "dataset_sha256": adaptive_dataset_digest(dataset),
        "node_names": list(ADAPTIVE_NODE_NAMES),
        "relation_names": list(ADAPTIVE_RELATION_NAMES),
        "graph_input_source": (
            "keypoint heatmaps, UAV proprioception and onboard battery only"),
        "label_source": "Isaac/PX4 physical transition and terminal outcome",
        "outcome_fields_are_graph_inputs": False,
        "outcome_only_fields": sorted(OUTCOME_ONLY_FIELDS),
        "split_unit": "(episode_id, scenario)",
        "normalization": dict(dataset["normalization"]),
        "source_behavior_policy": dict(source_behavior_policy),
        "transitions": int(len(success)),
        "episodes": int(len(episode_ids)),
        "successful_episodes": int(sum(
            int(success[np.flatnonzero(dataset["episode_id"] == ep)[0]])
            for ep in episode_ids)),
        "outcome_strata": {
            "success": int(sum(int(success[index]) for index in representative)),
            "failure": int(sum(not int(success[index]) for index in representative)),
            "collision": int(sum(value == "collision" for value in failure_types)),
            "excessive_drift": int(sum(value == "excessive_drift"
                                        for value in failure_types)),
            "near_miss": int(near_miss),
        },
    }
    manifest_path = path.with_suffix(".manifest.json")
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, allow_nan=False),
                                  encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    return manifest


def load_adaptive_dataset(path: str | Path, *, config_hash: str | None = None):
    path = Path(path)
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != ADAPTIVE_DATASET_FORMAT:
        raise ValueError("adaptive rollout dataset format mismatch")
    if manifest.get("graph_schema_version") != ADAPTIVE_GRAPH_VERSION:
        raise ValueError("adaptive rollout graph schema mismatch")
    if config_hash is not None and manifest.get("dataset_config_hash") != str(config_hash):
        raise ValueError("adaptive rollout configuration mismatch")
    with np.load(path, allow_pickle=False) as blob:
        dataset = {name: blob[name] for name in blob.files}
    dataset["normalization"] = dict(manifest["normalization"])
    validate_adaptive_dataset(dataset)
    if adaptive_dataset_digest(dataset) != manifest.get("dataset_sha256"):
        raise ValueError("adaptive rollout dataset digest mismatch")
    return dataset, manifest
