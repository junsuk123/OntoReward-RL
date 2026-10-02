"""Deterministic reward-hacking fixtures; these are not learned-policy results."""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from .config import ExperimentConfig
from .reward import compute_reward, discount_for_dt
from .safety import TerminalReason


FIXTURES = {
    "safe efficient landing": (8.0, TerminalReason.SUCCESS, "approach"),
    "safe delayed landing": (20.0, TerminalReason.SUCCESS, "approach"),
    "stable near-pad hovering until deadline": (70.0, TerminalReason.TASK_TIMEOUT, "hover"),
    "unnecessary induced loss followed by abort": (8.0, TerminalReason.SAFE_ABORT, "blind"),
    "appropriate abort under unrecoverable loss": (4.0, TerminalReason.SAFE_ABORT, "blind"),
    "fast impact": (2.0, TerminalReason.UNSAFE_CONTACT, "impact"),
    "late impact": (50.0, TerminalReason.UNSAFE_CONTACT, "impact"),
    "prohibited blind contact": (10.0, TerminalReason.UNAUTHORIZED_CONTACT, "blind"),
    "oscillatory approach": (40.0, TerminalReason.TASK_TIMEOUT, "oscillatory"),
    "recoverable short loss followed by landing": (12.0, TerminalReason.SUCCESS, "short_loss"),
}


def _state(kind: str, fraction: float, elapsed_s: float
           ) -> tuple[float, float, bool, float, np.ndarray]:
    if kind == "approach":
        return 2.5 * (1 - fraction), 5.0 * (1 - fraction), True, 0.0, np.array([.2, -.2])
    if kind == "hover":
        return .2, .5, True, 0.0, np.zeros(2)
    if kind == "blind":
        return 1.5, 3.0, False, 0.0, np.array([.3, .2])
    if kind == "impact":
        return 1.0 * (1 - fraction), 4.0 * (1 - fraction), True, 0.0, np.array([.8, -1.0])
    if kind == "oscillatory":
        return 1.5 * math.sin(8 * math.pi * fraction), 2.0, True, .2, np.array([
            math.sin(12 * math.pi * fraction), 0.2])
    if kind == "short_loss":
        visible = not (5.0 <= elapsed_s < 5.3)
        return 2.0 * (1 - fraction), 5.0 * (1 - fraction), visible, 0.0, np.array([.25, -.2])
    raise ValueError(kind)


def run_reward_audit(config: ExperimentConfig) -> dict:
    records = {}
    dt_nominal = config.timing.policy_dt_s
    for name, (duration, terminal, kind) in FIXTURES.items():
        elapsed = 0.0
        discount = 1.0
        total = 0.0
        terminal_count = 0
        running_total = 0.0
        while elapsed < duration - 1e-12:
            dt = min(dt_nominal, duration - elapsed)
            elapsed += dt
            fraction = elapsed / duration
            ex, h, visible, bearing, action = _state(kind, fraction, elapsed)
            reason = terminal if elapsed >= duration - 1e-12 else None
            reward = compute_reward(
                ex_true_m=ex, h_true_m=h, measured_bearing_rad=bearing,
                bearing_valid=visible, normalized_policy_action=action,
                fov_rad=config.camera.fov_rad, dt_s=dt,
                terminal_reason=reason, config=config.reward)
            total += discount * reward.total
            running_total += discount * reward.running
            terminal_count += int(reward.terminal != 0.0)
            discount *= discount_for_dt(
                dt, config.timing.discount_time_constant_s)
        records[name] = {
            "duration_s": duration, "terminal_reason": terminal.value,
            "discounted_return": total,
            "discounted_running_component": running_total,
            "terminal_payment_count": terminal_count,
            "fixture_kind": kind,
        }
    payload = {
        "schema": "two_axis_reward_audit/1",
        "warning": "deterministic design fixtures, not learned-policy performance",
        "trajectories": records,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload
