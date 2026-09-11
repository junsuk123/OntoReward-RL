"""Ontology/R-GAT potential-based shaping for the controlled benchmark."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .sparse import sparse_terminal_reward


@dataclass(frozen=True)
class OntoRewardPBRS:
    potential: Callable[[object], float]
    gamma: float
    ppo_gamma: float
    shaping_lambda: float = 1.0
    design_id: str = "unversioned"
    frozen: bool = True

    def __post_init__(self) -> None:
        if abs(float(self.gamma) - float(self.ppo_gamma)) > 1e-12:
            raise ValueError("PBRS gamma must equal PPO gamma")
        if not self.frozen:
            raise ValueError("R-GAT reward design must be frozen before PPO training")

    def __call__(self, state, next_state, *, physical_contact=False, crash=False,
                 excessive_drift=False, terminal=False):
        task = sparse_terminal_reward(
            physical_contact=physical_contact, crash=crash,
            excessive_drift=excessive_drift, terminal=terminal)
        phi = float(self.potential(state))
        absorbing = bool(terminal or physical_contact or crash or excessive_drift)
        phi_next = 0.0 if absorbing else float(self.potential(next_state))
        shaping = self.shaping_lambda * (self.gamma * phi_next - phi)
        return float(task + shaping), {
            "task": task, "shape": shaping, "phi": phi, "phi_next": phi_next,
            "design_id": self.design_id,
        }
