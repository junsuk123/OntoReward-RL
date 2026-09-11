"""The three reward paths the experiment compares.

``manual``    hand-designed dense baseline, with explicit arbitrary weights.
``sparse``    binary task reward; the PBRS base that shaping must not alter.
``proposed``  the sparse base plus potential-based shaping with coefficients
              distilled from R-GAT and frozen before PPO starts.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .config import Config

__all__ = ["manual_dense", "sparse_task", "proposed_pbrs", "reward_for"]

TOUCHDOWN_FAILURES = ("unsafe_touchdown", "flight_failure")


def manual_dense(cur, nxt, action, status: str, viol: float, cfg: Config) -> float:
    """Hand-designed baseline reward.

    The moving deck and the energy budget get hand-tuned weights here, which is
    the point of the baseline: the proposed arm has to derive the same
    trade-offs from the ontology instead of being told them.
    """
    w = cfg.reward.manual
    a = np.asarray(action, dtype=float)
    r = -cfg.sim.dt * (
        w.w_pos * min(cur.sem.position_error, 3.0)
        + w.w_vel * float(np.linalg.norm(cur.meas["vel"]))
        + w.w_tilt * cur.sem.tilt
        + w.w_rate * cur.sem.angular_rate
        + w.w_wind * cur.sem.wind_risk
        + w.w_pad_track * min(cur.sem.closing_speed, 3.0)
        + w.w_energy * (1.0 - cur.sem.battery_reserve)
        # Descending on a pose nothing can vouch for. The policy can act on
        # this: keeping the markers in frame is what raises nav_confidence when
        # the fix is bad, and holding altitude is what buys the time to do it.
        + w.w_nav * (1.0 - cur.sem.nav_confidence)
        + w.w_act * float(np.sum(a ** 2))
        + w.time)
    # Reward progress, so the policy is not paid only in static penalties.
    r += 0.8 * (cur.sem.position_error - nxt.sem.position_error)
    if status == "success":
        r += w.success
    elif status in TOUCHDOWN_FAILURES:
        # Graded rather than a cliff: a near miss costs a fraction of a hard
        # crash, so the policy gets a gradient toward safe touchdown instead of
        # an information-free binary penalty.
        grade = min(1.0, max(0.0, (viol - 1.0) / w.viol_span))
        r += w.failure * (w.failure_floor + (1.0 - w.failure_floor) * grade)
    elif status == "battery_depleted":
        r += w.battery_depleted
    elif status == "timeout":
        # Never touching down is a mission failure. Without this, hovering out
        # the clock strictly dominates attempting a landing.
        r += w.timeout
    return float(r)


def sparse_task(status: str, cfg: Config) -> float:
    """Binary task reward.

    Running out of energy in the air is a distinct failure and gets its own
    terminal value rather than being scored as a timeout.
    """
    s = cfg.reward.sparse
    r = s.time * cfg.sim.dt
    if status == "success":
        r += s.success
    elif status in TOUCHDOWN_FAILURES:
        r += s.failure
    elif status == "battery_depleted":
        r += s.battery_depleted
    elif status == "timeout":
        r += s.timeout
    return float(r)


def proposed_pbrs(cur, nxt, status: str, potential, cfg: Config) -> tuple[float, dict[str, Any]]:
    """Fixed R-GAT-distilled potential-based reward shaping.

    ``F(s, s') = lambda * (gamma * Phi(s') - Phi(s))`` with ``Phi = 0`` in the
    absorbing terminal state, which is what keeps the optimal policy the one
    the sparse task reward defines. ``cfg.reward.pbrs.gamma`` must equal
    ``cfg.ppo.gamma`` for that invariance to hold.
    """
    if potential is None:
        raise ValueError("the 'proposed' reward needs a frozen R-GAT reward design")
    base = sparse_task(status, cfg)
    phi0 = potential.predict(cur.graph)
    phi1 = potential.predict(nxt.graph) if status == "running" else 0.0
    shape = cfg.reward.pbrs["lambda"] * (cfg.reward.pbrs.gamma * phi1 - phi0)
    parts: dict[str, Any] = {"base": base, "shape": float(shape),
                             "phi0": float(phi0), "phi1": float(phi1)}
    if hasattr(potential, "design_id"):
        parts["design_id"] = str(potential.design_id)
    if hasattr(potential, "explain_terms"):
        parts["weighted_terms"] = potential.explain_terms(cur.graph)
    return float(base + shape), parts


def reward_for(mode: str, cur, nxt, action, status: str, viol: float,
               potential, cfg: Config) -> tuple[float, dict[str, Any]]:
    """Dispatch on the reward mode, returning the value and its parts."""
    mode = mode.lower()
    if mode == "manual":
        return manual_dense(cur, nxt, action, status, viol, cfg), {}
    if mode == "sparse":
        return sparse_task(status, cfg), {}
    if mode == "proposed":
        return proposed_pbrs(cur, nxt, status, potential, cfg)
    raise ValueError(f"Unknown reward mode: {mode}")
