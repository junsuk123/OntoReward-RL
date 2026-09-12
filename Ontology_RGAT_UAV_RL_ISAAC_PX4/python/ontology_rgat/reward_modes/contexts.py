"""Narrow reward-side contracts for the controlled pipelines."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..semantic import OntologyGraph
from ..perception.semantic_observation import (
    SEMANTIC_GRAPH_INPUT_DIM, SEMANTIC_NODE_NAMES, SEMANTIC_RELATION_NAMES)


def _vector(value, size: int, name: str) -> np.ndarray:
    output = np.asarray(value, dtype=np.float64).reshape(-1)
    if output.shape != (size,) or not np.isfinite(output).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return output


@dataclass(frozen=True)
class TerminalFlags:
    physical_contact: bool = False
    crash: bool = False
    excessive_drift: bool = False
    battery_depleted: bool = False
    terminal: bool = False

    def as_kwargs(self) -> dict:
        return {
            "physical_contact": bool(self.physical_contact),
            "crash": bool(self.crash),
            "excessive_drift": bool(self.excessive_drift),
            "battery_depleted": bool(self.battery_depleted),
            "terminal": bool(self.terminal),
        }


@dataclass(frozen=True)
class ShinSERewardContext:
    current_training_relative_state: np.ndarray
    next_training_relative_state: np.ndarray
    next_estimation_loss: float
    action: np.ndarray
    uav_vertical_velocity: float
    terminal: TerminalFlags

    def __post_init__(self) -> None:
        object.__setattr__(self, "current_training_relative_state", _vector(
            self.current_training_relative_state, 6, "current training relative state"))
        object.__setattr__(self, "next_training_relative_state", _vector(
            self.next_training_relative_state, 6, "next training relative state"))
        object.__setattr__(self, "action", _vector(self.action, 4, "action"))
        if not np.isfinite(float(self.next_estimation_loss)):
            raise ValueError("next estimation loss must be finite")
        if not np.isfinite(float(self.uav_vertical_velocity)):
            raise ValueError("UAV vertical velocity must be finite")
        if not isinstance(self.terminal, TerminalFlags):
            raise TypeError("terminal must be TerminalFlags")


@dataclass(frozen=True)
class NoSERewardContext:
    current_training_relative_state: np.ndarray
    next_training_relative_state: np.ndarray
    action: np.ndarray
    uav_vertical_velocity: float
    terminal: TerminalFlags

    def __post_init__(self) -> None:
        object.__setattr__(self, "current_training_relative_state", _vector(
            self.current_training_relative_state, 6, "current training relative state"))
        object.__setattr__(self, "next_training_relative_state", _vector(
            self.next_training_relative_state, 6, "next training relative state"))
        object.__setattr__(self, "action", _vector(self.action, 4, "action"))
        if not np.isfinite(float(self.uav_vertical_velocity)):
            raise ValueError("UAV vertical velocity must be finite")
        if not isinstance(self.terminal, TerminalFlags):
            raise TypeError("terminal must be TerminalFlags")


@dataclass(frozen=True)
class OntologyRewardContext:
    """PBRS context deliberately incapable of carrying metric relative state."""

    graph: OntologyGraph
    next_graph: OntologyGraph
    terminal: TerminalFlags

    def __post_init__(self) -> None:
        for name, graph in (("graph", self.graph), ("next_graph", self.next_graph)):
            if not isinstance(graph, OntologyGraph):
                raise TypeError(f"{name} must be an estimator-free ontology graph")
            if not np.isfinite(graph.X).all():
                raise ValueError(f"{name} contains non-finite semantic features")
            if tuple(graph.node_names) != SEMANTIC_NODE_NAMES:
                raise ValueError(f"{name} is not the estimator-free semantic node schema")
            if tuple(graph.relation_names) != SEMANTIC_RELATION_NAMES:
                raise ValueError(f"{name} is not the semantic relation schema")
            if graph.X.shape != (SEMANTIC_GRAPH_INPUT_DIM, len(SEMANTIC_NODE_NAMES)):
                raise ValueError(f"{name} has an invalid semantic feature shape")
        if not isinstance(self.terminal, TerminalFlags):
            raise TypeError("terminal must be TerminalFlags")
