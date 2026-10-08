"""Evaluation-only consistency, authority and MATLAB-export helpers."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def policy_consistency(reference_action, perturbed_actions, action_limit, *, epsilon=1e-9):
    reference = np.asarray(reference_action, dtype=float)
    perturbed = np.asarray(perturbed_actions, dtype=float)
    limit = np.asarray(action_limit, dtype=float)
    if perturbed.ndim != 2 or perturbed.shape[1:] != reference.shape:
        raise ValueError("perturbed actions must be [probe,action_axis]")
    delta = (perturbed-reference)/np.maximum(np.abs(limit), epsilon)
    return {"normalized_l2": np.linalg.norm(delta, axis=1),
            "normalized_axis": delta}


def authority_metrics(records):
    """Summarise policy request separately from limit/gateway/measurement."""
    if not records:
        return {"steps": 0, "intervention_rate": 0.0}
    requested = np.asarray([r["requested"] for r in records], dtype=float)
    limited = np.asarray([r["limited"] for r in records], dtype=float)
    measured = np.asarray([r.get("measured", [np.nan]*requested.shape[1])
                           for r in records], dtype=float)
    stamp = np.asarray([r["command_stamp_s"] for r in records], dtype=float)
    decision = np.asarray([r["decision_stamp_s"] for r in records], dtype=float)
    dt = np.diff(stamp)
    jerk = np.diff(limited, axis=0)/np.maximum(dt[:, None], 1e-9)
    latency = stamp-decision
    intervention = np.linalg.norm(limited-requested, axis=1) > 1e-12
    return {"steps": len(records), "intervention_rate": float(intervention.mean()),
            "mean_intervention_l2": float(np.linalg.norm(limited-requested, axis=1).mean()),
            "requested_jerk_l2_mean": float(np.linalg.norm(
                np.diff(requested, axis=0)/np.maximum(dt[:, None], 1e-9), axis=1).mean())
                if len(records) > 1 else 0.0,
            "limited_jerk_l2_mean": float(np.linalg.norm(jerk, axis=1).mean())
                if len(records) > 1 else 0.0,
            "measurement_rmse": float(np.sqrt(np.nanmean((measured-limited)**2))),
            "latency_p50_s": float(np.percentile(latency, 50)),
            "latency_p95_s": float(np.percentile(latency, 95)),
            "latency_p99_s": float(np.percentile(latency, 99))}


def export_matlab(records, summary, output_prefix, metadata):
    """Write one source table to CSV/MAT plus its exact metadata JSON."""
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    rows = list(records)
    if rows:
        fields = list(rows[0])
        with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
    else:
        prefix.with_suffix(".csv").write_text("", encoding="utf-8")
    payload = {"records": rows, "summary": summary, "metadata": metadata}
    prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2, default=float),
                                           encoding="utf-8")
    try:
        from scipy.io import savemat
        arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]} if rows else {}
        arrays["summary_json"] = json.dumps(summary, default=float)
        arrays["metadata_json"] = json.dumps(metadata, default=float)
        savemat(prefix.with_suffix(".mat"), arrays)
        return {"csv": str(prefix.with_suffix('.csv')),
                "mat": str(prefix.with_suffix('.mat')),
                "metadata": str(prefix.with_suffix('.json'))}
    except ImportError:
        return {"csv": str(prefix.with_suffix('.csv')), "mat": "NOT_RUN: scipy unavailable",
                "metadata": str(prefix.with_suffix('.json'))}

