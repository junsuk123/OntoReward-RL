"""Bounded relaunch after a crashed simulator startup.

Isaac aborts intermittently during boot on this hardware (a carb tasking
mutex assertion). No episode, reward or label exists at that point, so a
relaunch costs one boot and loses nothing -- whereas failing out discards a
queued multi-hour run. It must stay *bounded*: a simulator that genuinely
cannot start still has to fail the run.
"""
from __future__ import annotations

import pytest

from ontology_rgat.stack import ExternalStack, StackError


def _stack(monkeypatch, failures: int, attempts: int = 3):
    stack = ExternalStack.__new__(ExternalStack)
    stack.log_dir = "/tmp/ontology_rgat_stack"
    stack.startup_attempts = attempts
    stack._shutdown_requested = False
    calls = {"agent": 0, "isaac": 0, "gateway": 0, "stop": 0}

    def start_agent():
        calls["agent"] += 1

    def start_isaac():
        calls["isaac"] += 1
        if calls["isaac"] <= failures:
            raise StackError("Isaac Sim + PX4 SITL exited during startup. Log:\nabort")

    def start_gateway():
        calls["gateway"] += 1

    def stop():
        calls["stop"] += 1

    monkeypatch.setattr(stack, "start_agent", start_agent, raising=False)
    monkeypatch.setattr(stack, "start_isaac", start_isaac, raising=False)
    monkeypatch.setattr(stack, "start_gateway", start_gateway, raising=False)
    monkeypatch.setattr(stack, "stop", stop, raising=False)
    monkeypatch.setattr("ontology_rgat.stack.time.sleep", lambda _s: None)
    return stack, calls


def test_a_crashed_boot_is_relaunched_and_the_run_continues(monkeypatch, capsys):
    stack, calls = _stack(monkeypatch, failures=1)

    stack.start()

    assert calls["isaac"] == 2
    assert calls["gateway"] == 1
    # The dead processes are cleaned up before the relaunch.
    assert calls["stop"] == 1
    output = capsys.readouterr().out
    assert "Relaunching (1 of 2)" in output
    assert "External stack ready" in output


def test_a_simulator_that_never_boots_still_fails_the_run(monkeypatch):
    stack, calls = _stack(monkeypatch, failures=99, attempts=3)

    with pytest.raises(StackError, match="exited during startup"):
        stack.start()

    assert calls["isaac"] == 3
    assert calls["gateway"] == 0


def test_shutdown_during_startup_is_not_fought_with_a_relaunch(monkeypatch):
    stack, calls = _stack(monkeypatch, failures=99)
    stack._shutdown_requested = True

    with pytest.raises(StackError):
        stack.start()

    assert calls["isaac"] == 1


def test_retries_can_be_disabled_for_a_single_call(monkeypatch):
    stack, calls = _stack(monkeypatch, failures=1)

    with pytest.raises(StackError):
        stack.start(attempts=1)

    assert calls["isaac"] == 1
