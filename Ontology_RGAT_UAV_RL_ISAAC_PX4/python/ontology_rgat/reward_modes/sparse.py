"""Common terminal task reward."""
from __future__ import annotations


def sparse_terminal_reward(*, physical_contact: bool, crash: bool,
                           excessive_drift: bool, terminal: bool,
                           success_value: float = 10.0,
                           failure_value: float = -10.0) -> float:
    if physical_contact:
        return float(success_value)
    if terminal and (crash or excessive_drift):
        return float(failure_value)
    return 0.0
