"""Episode recording and the altitude-plateau detector.

The recurring failure this module exists to detect is not a policy that stops
commanding descent. It is a safety intervention triggered upstream:

    horizontal tracking error grows
    -> FOV margin shrinks
    -> detection confidence falls
    -> landing_inhibited becomes true
    -> the supervisor overrides the commanded descent
    -> applied az differs from requested az
    -> altitude plateaus until SAFE_ABORT or TASK_TIMEOUT

The detector reports each link separately, so a plateau is attributed to the
link that actually failed rather than to the supervisor that reacted to it.
Removing the supervisor or lowering the confidence threshold would hide the
symptom; the upstream link is the one to fix.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .contracts import packet_fields
from .environment import TwoAxisLandingEnv


@dataclass
class EpisodeRecord:
    """Per-decision traces plus the episode outcome."""

    seed: int
    difficulty: float
    terminal_reason: str = "RUNNING"
    time_s: list[float] = field(default_factory=list)
    height_m: list[float] = field(default_factory=list)
    horizontal_error_m: list[float] = field(default_factory=list)
    relative_speed_m_s: list[float] = field(default_factory=list)
    requested_ax: list[float] = field(default_factory=list)
    requested_az: list[float] = field(default_factory=list)
    applied_ax: list[float] = field(default_factory=list)
    applied_az: list[float] = field(default_factory=list)
    bearing: list[float] = field(default_factory=list)
    fov_margin: list[float] = field(default_factory=list)
    confidence: list[float] = field(default_factory=list)
    time_since_detection_s: list[float] = field(default_factory=list)
    landing_inhibited: list[bool] = field(default_factory=list)
    abort_requested: list[bool] = field(default_factory=list)
    terminal_descent: list[bool] = field(default_factory=list)
    intervened: list[bool] = field(default_factory=list)
    readiness: list[float] = field(default_factory=list)
    pad_x_estimate_m: list[float] = field(default_factory=list)
    pad_vx_estimate_m_s: list[float] = field(default_factory=list)
    pad_ax_estimate_m_s2: list[float] = field(default_factory=list)
    reward: list[float] = field(default_factory=list)

    @property
    def landed(self) -> bool:
        return self.terminal_reason == "SUCCESS"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def record_episode(env: TwoAxisLandingEnv,
                   policy: Callable[[TwoAxisLandingEnv], np.ndarray], *,
                   seed: int, difficulty: float = 1.0,
                   max_decisions: int = 10_000) -> EpisodeRecord:
    """Run one episode, capturing every quantity the plateau chain needs."""
    observation, info = env.reset(seed=seed, difficulty=difficulty)
    record = EpisodeRecord(seed=int(seed),
                           difficulty=float(info["curriculum_difficulty"]))
    for _ in range(int(max_decisions)):
        action = np.asarray(policy(env), dtype=float).reshape(2)
        observation, reward, terminated, _truncated, info = env.step(action)
        assert env.state is not None and env.scenario is not None
        assert env.track is not None
        pad_x, pad_v, _ = env.scenario.state_at(env.state.time_s)
        values = packet_fields(observation.packet, env.registry)
        record.time_s.append(env.state.time_s)
        record.height_m.append(env.state.z_m)
        record.horizontal_error_m.append(pad_x - env.state.x_m)
        record.relative_speed_m_s.append(pad_v - env.state.vx_m_s)
        record.requested_ax.append(float(info["requested_acceleration_m_s2"][0]))
        record.requested_az.append(float(info["requested_acceleration_m_s2"][1]))
        record.applied_ax.append(float(info["applied_acceleration_m_s2"][0]))
        record.applied_az.append(float(info["applied_acceleration_m_s2"][1]))
        record.bearing.append(values["measuredBearing"])
        record.fov_margin.append(values["predictedFovMargin"])
        record.confidence.append(values["detectionConfidence"])
        record.time_since_detection_s.append(env.track.time_since_detection_s)
        record.landing_inhibited.append(bool(info["landing_inhibited"]))
        record.abort_requested.append(bool(info["abort_requested"]))
        record.terminal_descent.append(bool(info["terminal_descent"]))
        record.intervened.append(bool(info["safety_intervened"]))
        record.readiness.append(float(info["landing_readiness"]))
        record.pad_x_estimate_m.append(env.track.pad_x_m)
        record.pad_vx_estimate_m_s.append(env.track.pad_vx_m_s)
        record.pad_ax_estimate_m_s2.append(env.track.pad_ax_m_s2)
        record.reward.append(float(reward))
        if terminated:
            record.terminal_reason = str(info["status"])
            break
    return record


def detect_altitude_plateau(record: EpisodeRecord, *,
                            tail_fraction: float = 0.4,
                            height_range_m: float = 0.5,
                            minimum_height_m: float = 0.3,
                            minimum_decisions: int = 20) -> dict[str, Any]:
    """Classify a near-constant-altitude tail and attribute it upstream."""
    count = len(record.height_m)
    if count < minimum_decisions:
        return {"plateau": False, "reason": "episode too short to classify"}
    start = int(count * (1.0 - float(tail_fraction)))
    height = np.asarray(record.height_m[start:], dtype=float)
    requested = np.asarray(record.requested_az[start:], dtype=float)
    applied = np.asarray(record.applied_az[start:], dtype=float)
    # A descent was asked for and the supervisor did not deliver it.
    overridden = np.logical_and(requested < -1e-9, applied > requested + 1e-9)
    plateau = bool(height.ptp() < float(height_range_m)
                   and height.mean() > float(minimum_height_m)
                   and overridden.any())
    error = np.abs(np.asarray(record.horizontal_error_m[start:], dtype=float))
    margin = np.asarray(record.fov_margin[start:], dtype=float)
    confidence = np.asarray(record.confidence[start:], dtype=float)
    inhibited = np.asarray(record.landing_inhibited[start:], dtype=bool)
    half = max(1, len(error) // 2)
    chain = {
        "horizontal_error_growing":
            bool(error[-half:].mean() > error[:half].mean() + 1e-6),
        "fov_margin_shrinking":
            bool(margin[-half:].mean() < margin[:half].mean() - 1e-6),
        "confidence_falling":
            bool(confidence[-half:].mean() < confidence[:half].mean() - 1e-6),
        "landing_inhibited_active": bool(inhibited.any()),
        "descent_overridden": bool(overridden.any()),
    }
    if not plateau:
        upstream = None
    elif chain["horizontal_error_growing"]:
        upstream = "horizontal_tracking"
    elif chain["fov_margin_shrinking"] or chain["confidence_falling"]:
        upstream = "visibility_geometry"
    elif chain["landing_inhibited_active"]:
        upstream = "track_trust_or_stopping_margin"
    else:
        upstream = "unattributed"
    return {
        "plateau": plateau,
        "terminal_reason": record.terminal_reason,
        "tail_height_range_m": float(height.ptp()),
        "tail_mean_height_m": float(height.mean()),
        "override_fraction": float(overridden.mean()),
        "chain": chain,
        # The link to repair. Suppressing the supervisor or the confidence
        # threshold would remove the symptom and leave this cause in place.
        "upstream_cause": upstream,
    }


def summarize(records: Sequence[EpisodeRecord]) -> dict[str, Any]:
    """Aggregate outcome rates and plateau attribution across episodes."""
    if not records:
        return {"episodes": 0}
    unsafe = {"UNSAFE_CONTACT", "UNAUTHORIZED_CONTACT", "MISSED_PAD_CONTACT",
              "SAFETY_ENVELOPE_VIOLATION"}
    reports = [detect_altitude_plateau(item) for item in records]
    landed = [item for item in records if item.landed]
    causes: dict[str, int] = {}
    for report in reports:
        if report.get("plateau"):
            key = str(report["upstream_cause"])
            causes[key] = causes.get(key, 0) + 1

    def rate(predicate: Callable[[EpisodeRecord], bool]) -> float:
        return float(np.mean([predicate(item) for item in records]))

    return {
        "episodes": len(records),
        "landing_rate": rate(lambda r: r.landed),
        "safe_abort_rate": rate(lambda r: r.terminal_reason == "SAFE_ABORT"),
        "task_timeout_rate": rate(lambda r: r.terminal_reason == "TASK_TIMEOUT"),
        "unsafe_rate": rate(lambda r: r.terminal_reason in unsafe),
        "fov_capture_rate": float(np.mean(
            [np.mean(np.asarray(r.time_since_detection_s) <= 1e-9)
             for r in records])),
        "supervisor_intervention_rate": float(np.mean(
            [np.mean(r.intervened) for r in records])),
        "plateau_rate": float(np.mean([bool(r["plateau"]) for r in reports])),
        "plateau_upstream_causes": causes,
        "mean_landing_time_s": (float(np.mean([r.time_s[-1] for r in landed]))
                                if landed else None),
        "mean_terminal_horizontal_error_m": (
            float(np.mean([abs(r.horizontal_error_m[-1]) for r in landed]))
            if landed else None),
    }


def graph_utilisation(records_X: Sequence[np.ndarray]) -> dict[str, Any]:
    """Which graph nodes and channels actually carry information.

    Answers the structural half of the ontology audit: dead nodes, permanently
    zero channels, constant channels and duplicate node rows. The other half --
    whether a relation's attention collapses to zero -- needs a trained
    encoder and is not inferable from the features alone.
    """
    from .ontology import (DECLARED_EDGES, FEATURE_CHANNELS, GRAPH_EDGES,
                           NODE_NAMES, QUERY_NODES)

    if not records_X:
        return {"samples": 0}
    stacked = np.stack([np.asarray(item, dtype=float) for item in records_X])
    if stacked.shape[1:] == (9, 12):
        from .ontology_v28 import FEATURE_CHANNELS, GRAPH_EDGES, NODE_NAMES
        QUERY_NODES = ()
    width = len(FEATURE_CHANNELS)
    base = stacked[:, :, :width]
    nodes: dict[str, Any] = {}
    zero_pairs: list[str] = []
    constant_pairs: dict[str, float] = {}
    for i, name in enumerate(NODE_NAMES):
        block = base[:, i, :]
        active = int((np.abs(block).max(axis=0) > 1e-9).sum())
        nodes[name] = {
            "active_channels": active,
            "max_abs": float(np.abs(block).max()),
            "std": float(block.std()),
            "dead": bool(active == 0),
            "query_node": name in QUERY_NODES,
        }
        for j, channel in enumerate(FEATURE_CHANNELS):
            column = block[:, j]
            if np.abs(column).max() <= 1e-9:
                zero_pairs.append(f"{name}.{channel}")
            elif column.std() <= 1e-12:
                constant_pairs[f"{name}.{channel}"] = float(column[0])
    duplicates = [f"{NODE_NAMES[i]}=={NODE_NAMES[j]}"
                  for i in range(len(NODE_NAMES))
                  for j in range(i + 1, len(NODE_NAMES))
                  if np.allclose(base[:, i, :], base[:, j, :])]
    return {
        "samples": int(stacked.shape[0]),
        "nodes": nodes,
        "dead_nodes": [name for name, row in nodes.items() if row["dead"]],
        "permanently_zero_channels": zero_pairs,
        "constant_channels": constant_pairs,
        "duplicate_node_rows": duplicates,
        "declared_edges": len(DECLARED_EDGES),
        "total_edges": len(GRAPH_EDGES),
        "attention_sparsity": (
            "not inferable from features; requires a trained encoder"),
    }
