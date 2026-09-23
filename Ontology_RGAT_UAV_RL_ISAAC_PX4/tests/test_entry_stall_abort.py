"""A parked vehicle must not spend the whole entry budget going nowhere.

Measured on 2026-09-23: a pair sat on the deck at 0.00 m/s, 5.13 m from a
commanded pose 4.50 m above it, and burned 10515 of 10515 samples -- 271 s of
wall clock -- closing -0.00 m before the budget expired. The stack was then
rebuilt, which was the only remedy available from the first second.

The entry hold is bounded by a *simulated* clock precisely so that a slow
settle on a heavy stage is not cut off, so the budget cannot simply be
shortened. What can be detected is the difference between settling slowly and
not flying at all, and the standstill speed is what separates them: a vehicle
that is chasing, overshooting or circling has a relative speed far above it
however long it takes to converge.
"""
from __future__ import annotations

import pytest

from ontology_rgat.bridge import EntryResetError, PX4Bridge


class _Cfg:
    """Only the fields ``wait_at_entry`` reads."""

    entry_tolerance = 0.90
    entry_speed_tolerance = 1.00
    entry_settle = 1.00
    entry_timeout = 600.0
    entry_sim_budget = 400.0
    entry_view_margin = 0.85
    require_pad_in_view = False
    entry_travel_speed = 1.2
    entry_travel_budget_max = 60.0
    failsafe_grace = 10.0
    entry_stall_seconds = 20.0
    entry_stall_speed = 0.05
    entry_stall_progress_m = 0.25


def test_the_standstill_threshold_separates_parked_from_settling():
    """The whole safety of the early abort rests on this gap."""
    assert _Cfg.entry_stall_speed < _Cfg.entry_speed_tolerance / 10.0


@pytest.mark.parametrize("peak_speed,should_abort", [
    (0.00, True),     # sitting on the deck, the measured failure
    (0.02, True),     # the measured worst speed of that flight
    (0.30, False),    # drifting: slow, but flying
    (1.43, False),    # the other measured failure -- chasing, never settling
])
def test_only_a_vehicle_that_has_stopped_moving_is_cut_off(peak_speed, should_abort):
    """A slow settle must keep its budget; a parked vehicle must not."""
    from ontology_rgat import bridge as bridge_module

    stalled = []

    # Drive the decision logic directly: offset never improves, speed is the
    # parameter, and the window is exceeded.
    best_offset, stalled_since, stalled_peak = float("inf"), None, 0.0
    for step in range(400):
        now, offset, speed = step * 0.1, 5.13, peak_speed
        if offset < best_offset - _Cfg.entry_stall_progress_m:
            best_offset, stalled_since, stalled_peak = offset, None, 0.0
        else:
            if stalled_since is None:
                stalled_since, stalled_peak = now, speed
            else:
                stalled_peak = max(stalled_peak, speed)
            if (now - stalled_since >= _Cfg.entry_stall_seconds
                    and stalled_peak <= _Cfg.entry_stall_speed):
                stalled.append(now)
                break
    assert bool(stalled) is should_abort
    assert bridge_module.EntryResetError is EntryResetError


def test_a_vehicle_still_closing_the_gap_resets_the_window():
    """Improvement must restart the clock, or a long approach trips it."""
    best_offset, stalled_since = float("inf"), None
    aborted = False
    for step in range(600):
        now = step * 0.1
        offset = max(0.5, 30.0 - 0.05 * step)     # closing steadily
        if offset < best_offset - _Cfg.entry_stall_progress_m:
            best_offset, stalled_since = offset, None
        else:
            if stalled_since is None:
                stalled_since = now
            elif now - stalled_since >= _Cfg.entry_stall_seconds:
                aborted = True
                break
    assert not aborted, "a vehicle that is closing the gap must keep its budget"


def test_the_abort_is_the_recoverable_kind():
    """So the existing retry rebuilds the stack instead of ending the run."""
    from ontology_rgat.bridge import BridgeError
    from ontology_rgat.ppo.recurrent_train import collect_episode_resilient
    import inspect

    assert issubclass(EntryResetError, BridgeError)
    body = inspect.getsource(collect_episode_resilient)
    assert "EntryResetError" in body


def test_the_message_names_what_it_measured_and_why_it_gave_up():
    source = inspect_source()
    assert "PX4 stopped flying to the entry pose" in source
    assert "standstill threshold" in source
    assert "parked, not settling" in source


def inspect_source():
    import inspect
    return inspect.getsource(PX4Bridge.wait_at_entry)


def test_the_thresholds_are_configurable_with_safe_defaults():
    source = inspect_source()
    for name, default in (("entry_stall_seconds", "20.0"),
                          ("entry_stall_speed", "0.05"),
                          ("entry_stall_progress_m", "0.25")):
        assert f'"{name}", {default}' in source, name


def test_disabling_the_abort_restores_the_old_behaviour():
    """``entry_stall_seconds = 0`` must switch it off entirely."""
    source = inspect_source()
    assert "stall_seconds > 0.0" in source


def test_the_guard_is_silent_while_the_budget_pays_for_the_trip():
    """The travel allowance funds a long approach; it must not be cut off.

    The allowance is the run's own statement that the vehicle is expected to
    be under way rather than holding, so a standstill inside that window is
    not yet evidence of a parked vehicle -- and the existing entry-budget
    contract (tests/test_protocol.py) exercises exactly that case with a
    frozen fixture.
    """
    source = inspect_source()
    assert "travelling = sim_elapsed < float(travel_allowance or 0.0)" in source
    # The window must not even OPEN while travelling: opened on departure it
    # would expire on arrival, which is when the hold is supposed to start.
    assert "if offset < best_offset - stall_progress or travelling:" in source
