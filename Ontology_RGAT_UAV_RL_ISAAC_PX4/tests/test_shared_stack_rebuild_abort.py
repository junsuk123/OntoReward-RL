"""A sibling's rebuild must not cost every other pair worker a full timeout.

Four pair workers share one Isaac/PX4 stack. When one of them fails for real it
cycles that stack, and the others are mid-exchange with gateways that are being
killed. On 2026-09-24 they each waited out ``benchmark.gateway_timeout_s`` and
then reported an infrastructure failure of their own -- the run log shows
"Stopping gateway_0" followed immediately by two sibling timeouts, three times
in one hour.

Instrumenting the gateway is what ruled everything else out: no wedge (its
timers kept running and its lock was never held), no starved input, and no
dropped datagram on any socket. The process was simply gone.
"""
from __future__ import annotations

from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]

from ontology_rgat.bridge import _STACK_CHECK_PERIOD_S     # noqa: E402
from ontology_rgat.stack import ExternalStack              # noqa: E402


def _stack_with_restart_lock() -> ExternalStack:
    stack = ExternalStack.__new__(ExternalStack)
    stack._restart_lock = threading.RLock()
    stack.generation = 0
    return stack


def test_a_worker_can_see_a_rebuild_without_blocking_on_it():
    stack = _stack_with_restart_lock()
    assert not stack.restart_in_progress()

    holding = threading.Event()
    release = threading.Event()

    def rebuild():
        with stack._restart_lock:
            holding.set()
            release.wait(timeout=5.0)

    worker = threading.Thread(target=rebuild, daemon=True)
    worker.start()
    assert holding.wait(timeout=5.0)
    try:
        # The probe must answer immediately; a worker asking this is already
        # blocked on something else and cannot afford to wait here too.
        started = time.monotonic()
        assert stack.restart_in_progress()
        assert time.monotonic() - started < 0.5
    finally:
        release.set()
        worker.join(timeout=5.0)
    assert not stack.restart_in_progress()


def test_the_probe_is_cheap_enough_to_run_inside_the_wait_loop():
    """The exchange polls a 2 ms socket, so the probe cannot be per-iteration."""
    assert 0.02 <= _STACK_CHECK_PERIOD_S <= 0.5
    stack = _stack_with_restart_lock()
    started = time.monotonic()
    for _ in range(10000):
        stack.restart_in_progress()
    assert time.monotonic() - started < 1.0


def test_the_timeout_budget_is_not_the_lever_for_a_torn_down_gateway():
    """20 s, and the reason it is not 45.

    A gateway that has been torn down never answers, so a longer budget buys
    nothing and costs every sibling the difference on every rebuild.
    """
    import yaml

    system = yaml.safe_load(
        (ROOT / "config/shin2026-system.yaml").read_text(encoding="utf-8"))
    assert float(system["benchmark"]["gateway_timeout_s"]) == pytest.approx(20.0)
