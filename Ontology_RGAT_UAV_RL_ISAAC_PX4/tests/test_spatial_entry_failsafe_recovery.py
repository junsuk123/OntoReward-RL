"""A recoverable link failsafe during the spatial entry hover is an entry failure."""
from __future__ import annotations

import pytest

from ontology_rgat.bridge import EntryResetError, PX4Failsafe
from ontology_rgat.spatial import environment as env_module
from ontology_rgat.spatial import lifecycle


def _backend(monkeypatch, failures):
    backend = object.__new__(env_module.IsaacBackend)
    backend.cfg, backend.pair, backend.scenario = object(), 0, "s"
    backend.bridge = type("B", (), {"close": lambda self: None})()
    calls = {"reset": 0, "reinit": 0, "recovered": []}

    def reset_once(seed):
        calls["reset"] += 1
        if failures:
            raise failures.pop(0)
        return ("measurement", seed)

    backend._reset_once = reset_once
    monkeypatch.setattr(env_module.IsaacBackend, "__init__",
                        lambda self, *a, **k: calls.__setitem__("reinit", calls["reinit"] + 1))
    monkeypatch.setattr(lifecycle, "prepare_isolated_episode", lambda **_: False)

    def recover(error, *, seed, release):
        calls["recovered"].append((type(error).__name__, seed))
        return True

    monkeypatch.setattr(lifecycle, "recover_refused_reset", recover)
    return backend, calls


def test_recoverable_entry_failsafe_restarts_and_retries_the_same_seed(monkeypatch):
    backend, calls = _backend(monkeypatch, [PX4Failsafe(["gcs_connection_lost"], recoverable=True)])
    assert backend.reset(12000) == ("measurement", 12000)
    assert calls["recovered"] == [("EntryResetError", 12000)]
    assert calls["reset"] == 2 and calls["reinit"] == 1


def test_hard_entry_failsafe_still_propagates(monkeypatch):
    backend, calls = _backend(monkeypatch, [PX4Failsafe(["battery_unhealthy"], recoverable=False)])
    with pytest.raises(PX4Failsafe):
        backend.reset(12000)
    assert calls["recovered"] == []
