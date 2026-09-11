"""Build the controlled-profile frozen potential through an R-GAT stage."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from ..config import default_config
from ..rgat.train import train_potential
from ..semantic import OntologyGraph
from .controlled_potential import FEATURES


NODE_NAMES = tuple(FEATURES) + ("SafeLanding",)
RELATION_NAMES = ("contributes", "self")


def controlled_graph(values=None):
    values = np.zeros(5) if values is None else np.asarray(values, dtype=float)
    if values.shape != (5,):
        raise ValueError("controlled ontology needs five node values")
    src = np.asarray([0, 1, 2, 3, 0, 1, 2, 3, 4], dtype=np.int64)
    dst = np.asarray([4, 4, 4, 4, 0, 1, 2, 3, 4], dtype=np.int64)
    rel = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64)
    features = np.zeros((9, 5), dtype=float)
    features[0] = values
    features[1] = 1.0 - values
    features[2, :4] = 1.0
    features[3] = 1.0
    features[4:] = np.eye(5)
    return OntologyGraph(
        X=features, src=src, dst=dst, rel=rel, goal_node=4,
        node_names=NODE_NAMES, relation_names=RELATION_NAMES)


def _bootstrap_dataset(seed: int, samples: int):
    """Table-I-bounded semantic pretraining set, not simulator outcomes."""
    rng = np.random.default_rng(int(seed))
    costs = rng.uniform(0.0, 1.0, size=(int(samples), 4)).astype(np.float32)
    # Smooth safety proxy used only to initialize reward design. Actual policy
    # evaluation always uses physical touchdown outcomes.
    target = (1.0 - 2.0 * np.max(costs, axis=1)).astype(np.float32)
    template = controlled_graph()
    features = np.stack([
        controlled_graph(np.r_[value, 0.0]).X.T for value in costs
    ]).astype(np.float32)
    return {"X": features, "y": target, "graph": template}, costs


def prepare_controlled_rgat_artifact(path: str | Path, *, mode="quick",
                                     seed=42, samples=None, epochs=None):
    """Train, distill and atomically freeze a controlled benchmark artifact.

    Since the paper does not specify an ontology or R-GAT training dataset,
    this stage is an explicitly reported OntoReward design choice. It never
    uses the legacy UGV/GNSS/wind channels.
    """
    cfg = default_config(mode, "sitl")
    cfg.seed = int(seed)
    cfg.ontology.node_names = list(NODE_NAMES)
    cfg.ontology.relation_names = list(RELATION_NAMES)
    cfg.ontology.n_nodes = 5
    cfg.ontology.n_relations = 2
    cfg.ontology.in_dim = 9
    cfg.rgat.epochs = int(epochs if epochs is not None else (10 if mode == "quick" else 80))
    cfg.rgat.batch_size = 64
    count = int(samples if samples is not None else (4096 if mode == "quick" else 32768))
    dataset, _ = _bootstrap_dataset(seed, count)
    model, history = train_potential(dataset, cfg, verbose=True)

    X = dataset["X"]
    baseline = model.predict_batch(X)
    importance = []
    signed = []
    for node in range(4):
        counterfactual = X.copy()
        counterfactual[:, node, 0] = 0.0
        counterfactual[:, node, 1] = 1.0
        delta = model.predict_batch(counterfactual) - baseline
        importance.append(float(np.mean(np.abs(delta))))
        signed.append(float(np.mean(delta)))
    raw = np.asarray(importance) + 1e-6
    weights = raw / raw.sum()
    payload = {
        "format": "ontology_rgat.controlled_reward/1",
        "profile": "controlled_landing", "frozen": True,
        "provenance": "rgat_distillation",
        "dataset_provenance": "synthetic_table_i_semantic_bootstrap",
        "dataset_note": "configurable OntoReward design choice; not specified by Shin et al.",
        "features": list(FEATURES),
        "weights": {name: float(value) for name, value in zip(FEATURES, weights)},
        "importance": {name: value for name, value in zip(FEATURES, importance)},
        "signed_effect": {name: value for name, value in zip(FEATURES, signed)},
        "dataset_samples": count, "rgat_epochs": cfg.rgat.epochs,
        "rgat_final_validation_mse": float(history["val_loss"][-1]),
        "seed": int(seed),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["design_id"] = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return path, payload
