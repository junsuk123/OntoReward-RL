"""Common terminal task reward."""
from __future__ import annotations


def sparse_terminal_reward(*, physical_contact: bool, crash: bool,
                           excessive_drift: bool, terminal: bool,
                           battery_depleted: bool = False,
                           success_value: float = 10.0,
                           failure_value: float = -10.0,
                           timeout_value: float = 0.0) -> float:
    """Score the terminal outcome.

    ``physical_contact`` means a *valid landing*, not raw deck contact: the
    caller resolves an off-centre or tilted strike into ``crash``.

    An episode that simply runs out of horizon scores ``timeout_value``. It
    used to score zero, which made it the cheapest terminal available and let a
    policy that never approached the deck outscore one that tried and
    occasionally struck it badly.
    """
    if physical_contact:
        return float(success_value)
    if terminal and (crash or excessive_drift or battery_depleted):
        return float(failure_value)
    if terminal:
        return float(timeout_value)
    return 0.0
