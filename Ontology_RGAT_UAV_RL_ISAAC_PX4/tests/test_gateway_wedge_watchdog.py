"""The gateway has to say something when it stops answering.

2026-09-24 it stopped serving UDP roughly four times an hour, for longer than
the 45 s the learner waits, and the only trace was silence: no exception, no
last words beyond whatever routine line preceded the gap. Three explanations
fit that evidence equally well -- blocked on the control lock, blocked inside a
DDS publish, starved of CPU -- and nothing in the log chose between them.

Two timers share ``control_lock`` and the executor has two threads, so one
callback blocking while it holds the lock stops the UDP replies and the
OFFBOARD heartbeat together. That is the pair of symptoms that was actually
observed, which is why the watchdog reports the lock holder by name.
"""
from __future__ import annotations

from pathlib import Path
import sys
import threading
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2_ws/src/ontology_rgat_px4"))

from ontology_rgat_px4.ros2_gateway import (        # noqa: E402
    WEDGE_REPORT_COOLDOWN_S, WEDGE_THRESHOLD_S, _all_thread_stacks)


def test_the_threshold_sits_between_normal_and_the_learners_patience():
    """Wide enough not to cry wolf, tight enough to catch what killed runs.

    The learner's control period is about 0.1 s and PX4 calls an OFFBOARD
    signal lost after 0.5 s, so anything past a second is already abnormal; the
    learner abandons the episode at ``benchmark.gateway_timeout_s``, and a
    watchdog that only fired after that would report the wedge no sooner than
    the failure it is meant to explain.
    """
    assert 1.0 <= WEDGE_THRESHOLD_S <= 5.0
    assert WEDGE_THRESHOLD_S < 20.0, "must fire before the learner gives up"
    assert WEDGE_REPORT_COOLDOWN_S >= WEDGE_THRESHOLD_S, (
        "one report per wedge, not one per poll of it")


def test_the_dump_names_the_thread_and_where_it_is_blocked():
    held = threading.Lock()
    held.acquire()
    started = threading.Event()

    def blocked_callback():
        started.set()
        held.acquire()          # stands in for a callback that never returns

    worker = threading.Thread(target=blocked_callback, daemon=True,
                              name="pretend-control-tick")
    worker.start()
    assert started.wait(timeout=2.0)
    time.sleep(0.1)
    try:
        dump = _all_thread_stacks()
    finally:
        held.release()

    assert "pretend-control-tick" in dump, "the blocked thread must be named"
    assert "blocked_callback" in dump, (
        "and the frame it is stuck in, or the dump explains nothing")
    assert dump.count("thread ") >= 2, "every thread, not just the culprit"


def test_the_two_timers_that_share_the_lock_are_both_instrumented():
    """A wedge report is only useful if it can name either holder.

    The UDP poll and the OFFBOARD heartbeat are the two callbacks that take
    ``control_lock``; the watchdog distinguishes them by the site name each
    passes to ``_holding``.
    """
    source = (ROOT / "ros2_ws/src/ontology_rgat_px4/ontology_rgat_px4"
              / "ros2_gateway.py").read_text(encoding="utf-8")
    assert 'self._holding("udp")' in source
    assert 'self._holding("control_tick")' in source
    # The watchdog must not be a ROS timer: the thing it measures is the
    # executor failing to run timers.
    assert 'threading.Thread(target=self._watch_for_wedges' in source
    assert "daemon=True" in source
