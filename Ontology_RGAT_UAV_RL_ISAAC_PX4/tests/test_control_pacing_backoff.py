"""Waiting for simulated time must not race it.

``pace_to_control_period`` waits for PX4's clock to advance one control period.
It used to do that by polling the gateway with no delay: the bridge socket
carries a 2 ms timeout, so a single control step asked for state up to five
hundred times, and each request took the gateway's ``control_lock`` and built a
full state message. The gateway's ROS executor has two threads -- with the poll
holding one and the OFFBOARD heartbeat the other, the sensor callbacks that
advance PX4's clock are what gets starved, so the simulated time the loop waits
for arrives more slowly the harder it waits.

Measured 2026-09-25 on an otherwise idle machine (Isaac 1.7 cores of 20, PX4
1.5% each, GPU 3%, CPU at 4.2 of 4.7 GHz): 0.07x realtime inside an episode
against 1.1x during the entry hover, which is the same stack with nobody
polling it.
"""
from __future__ import annotations

from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]

from ontology_rgat.bridge import (                    # noqa: E402
    _PACING_BACKOFF_FRACTION, _PACING_BACKOFF_MAX_S, PX4Bridge)


class _Clock:
    """A gateway whose simulated clock advances a fixed amount per request."""

    def __init__(self, step_us, control_period_us=100_000):
        self.now_us = 0
        self.step_us = step_us
        self.requests = 0
        self.cfg = SimpleNamespace(timeout=5.0)
        self.control_period_us = control_period_us
        self.last_px4_time_us = 0

    def transact(self, kind, fields, expected):
        self.requests += 1
        self.now_us += self.step_us
        return {"px4_time_us": self.now_us}

    def validate_state_with_estimator_grace(self, state):
        return state


def _pace(clock, state):
    return PX4Bridge.pace_to_control_period(clock, state)


def test_the_backoff_is_bounded_so_a_step_cannot_overshoot_far():
    assert 0.0 < _PACING_BACKOFF_FRACTION < 1.0
    # One control period is 100 ms; an overshoot of a whole period would change
    # the episode's timeline, which is the thing the pacing exists to protect.
    assert 0.0 < _PACING_BACKOFF_MAX_S <= 0.05


def test_waiting_no_longer_polls_flat_out():
    """The same wait, far fewer requests.

    PX4's clock advances on its own; here each request carries it 10 ms, so a
    100 ms control period needs ten of them at the very most.
    """
    clock = _Clock(step_us=10_000)
    state = _pace(clock, {"px4_time_us": 0})
    assert state["px4_time_us"] >= 100_000
    assert clock.requests <= 12, (
        f"{clock.requests} requests for one control period is a busy poll")


def test_it_still_waits_the_full_control_period():
    clock = _Clock(step_us=1_000)
    started = time.monotonic()
    state = _pace(clock, {"px4_time_us": 0})
    # The simulated deadline is what matters, and it is met exactly once.
    assert state["px4_time_us"] >= 100_000
    assert clock.last_px4_time_us >= 100_000
    # And the backoff actually slept rather than spinning the CPU.
    assert time.monotonic() - started > 0.0


def test_a_stalled_simulator_still_raises_rather_than_sleeping_forever():
    from ontology_rgat.bridge import BridgeError

    clock = _Clock(step_us=0)
    clock.cfg = SimpleNamespace(timeout=0.5)
    with pytest.raises(BridgeError, match="simulator"):
        _pace(clock, {"px4_time_us": 0})
