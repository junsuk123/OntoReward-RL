"""Owned group shutdown must reap leaders without abandoning living children."""
import io
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from unittest.mock import Mock

import pytest

from ontology_rgat.stack import ExternalStack


def test_reap_real_exited_owned_child_without_escalation_wait():
    child = subprocess.Popen([sys.executable, '-c', 'pass'], start_new_session=True)
    try:
        deadline = time.monotonic()+5.
        while time.monotonic() < deadline:
            if Path(f'/proc/{child.pid}/stat').read_text().split()[2] == 'Z':
                break
            time.sleep(.01)
        else:
            pytest.fail('owned fixture child did not exit')
        assert child.returncode is None  # not reaped by the test
        started = time.monotonic()
        ExternalStack._kill_group(child.pid, process=child)
        assert time.monotonic()-started < 3.
        assert child.returncode == 0
        assert not Path(f'/proc/{child.pid}').exists()
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5.)


def test_real_owned_child_gets_graceful_interrupt_and_is_reaped():
    code = ('import signal,sys,time; '
            'signal.signal(signal.SIGINT,lambda *_:sys.exit(0)); '
            'print("ready",flush=True); time.sleep(60)')
    child = subprocess.Popen([sys.executable, '-c', code], start_new_session=True,
                             stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'ready'
        started = time.monotonic()
        ExternalStack._kill_group(child.pid, process=child)
        assert time.monotonic()-started < 3.
        assert child.returncode == 0
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5.)
        child.stdout.close()


def test_reaped_leader_does_not_abandon_remaining_group(monkeypatch):
    child = Mock(pid=987654)
    child.poll.return_value = 0
    calls, probes = [], 0
    def killpg(pid, sig):
        nonlocal probes
        assert pid == child.pid
        calls.append(sig)
        if sig == 0:
            probes += 1
            if probes >= 4:
                raise ProcessLookupError
    monkeypatch.setattr(os, 'killpg', killpg)
    monkeypatch.setattr(time, 'sleep', lambda _: None)
    ExternalStack._kill_group(child.pid, process=child)
    assert signal.SIGINT in calls
    assert probes == 4
    assert child.poll.call_count == 4


def test_stubborn_owned_group_retains_bounded_escalation(monkeypatch):
    child = Mock(pid=987654)
    child.poll.return_value = None
    calls, sleeps = [], []
    monkeypatch.setattr(os, 'killpg', lambda pid, sig: calls.append((pid, sig)))
    monkeypatch.setattr(time, 'sleep', sleeps.append)
    ExternalStack._kill_group(child.pid, process=child)
    assert [sig for _, sig in calls if sig] == [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]
    assert len(sleeps) == 80
    assert sum(sleeps) == 20.
    child.wait.assert_called_once_with(timeout=1.)


def test_stop_passes_ownership_and_closes_handle():
    child = Mock(pid=987654)
    handle = io.BytesIO()
    stack = object.__new__(ExternalStack)
    stack.managed = [dict(name='owned', pid=child.pid, process=child, handle=handle)]
    stack._kill_group = Mock()
    stack.stop()
    stack._kill_group.assert_called_once_with(child.pid, process=child)
    assert not stack.managed and handle.closed


@pytest.mark.parametrize('pid', [0, 1, os.getpgrp()])
def test_broad_or_current_group_is_never_signalled(pid, monkeypatch):
    send = Mock()
    monkeypatch.setattr(os, 'killpg', send)
    with pytest.raises(ValueError, match='broad or current'):
        ExternalStack._kill_group(pid)
    send.assert_not_called()


def test_mismatched_owner_is_never_signalled(monkeypatch):
    send = Mock()
    monkeypatch.setattr(os, 'killpg', send)
    with pytest.raises(ValueError, match='owned process'):
        ExternalStack._kill_group(987654, process=Mock(pid=987653))
    send.assert_not_called()
