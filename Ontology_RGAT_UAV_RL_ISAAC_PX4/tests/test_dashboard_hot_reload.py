"""The read-only UI mirror reloads HTML without touching the live run."""
from __future__ import annotations

import os
import json

import run_dashboard_mirror as mirror
from run_dashboard_mirror import LivePage, StateRelay


def _dashboard_source(page: str) -> str:
    return f'PAGE = {page!r}\n'


def test_page_source_changes_are_visible_on_the_next_refresh(tmp_path):
    source = tmp_path / "dashboard.py"
    source.write_text(_dashboard_source("first UI"), encoding="utf-8")
    live = LivePage(source)
    assert live.page() == "first UI"

    source.write_text(_dashboard_source("updated R-GAT UI"), encoding="utf-8")
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    assert live.page() == "updated R-GAT UI"


def test_a_broken_ui_edit_keeps_the_last_good_page(tmp_path):
    source = tmp_path / "dashboard.py"
    source.write_text(_dashboard_source("known good"), encoding="utf-8")
    live = LivePage(source)
    assert live.page() == "known good"

    source.write_text("PAGE = (\n", encoding="utf-8")
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    assert live.page() == "known good"


class _Enricher:
    def enrich(self, state):
        return {**state, "enriched": True}


class _Response:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def read(self):
        return self.body


def test_relay_reuses_one_encoded_snapshot_for_concurrent_pollers(monkeypatch):
    calls = []

    def fetch(_request, timeout):
        calls.append(timeout)
        return _Response(json.dumps({
            "revision": 7, "graphs": {"pair_0": {"nodes": []}},
        }).encode())

    monkeypatch.setattr(mirror, "urlopen", fetch)
    relay = StateRelay("http://source/api/state", _Enricher(),
                       refresh_seconds=60)
    first = relay.get()
    second = relay.get()
    assert first.status == second.status == 200
    assert first.body is second.body
    assert len(calls) == 1
    assert json.loads(first.body)["revision"] == 7


def test_relay_serves_last_good_snapshot_when_upstream_drops(monkeypatch):
    responses = [
        _Response(json.dumps({"revision": 3, "graphs": {}}).encode()),
        OSError("connection refused"),
    ]

    def fetch(_request, timeout):
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(mirror, "urlopen", fetch)
    relay = StateRelay("http://source/api/state", _Enricher(),
                       refresh_seconds=0, retry_seconds=0)
    good = relay.get()
    stale = relay.get()
    assert good.status == stale.status == 200
    assert stale.stale is True
    assert stale.body is good.body
    assert "connection refused" in stale.error


def test_relay_restores_a_persisted_snapshot_before_upstream_returns(
        monkeypatch, tmp_path):
    cached = json.dumps({"revision": 11, "stage": {"name": "training"}}).encode()
    cache_path = tmp_path / "last-state.json"
    cache_path.write_bytes(cached)
    monkeypatch.setattr(
        mirror, "urlopen", lambda *_args, **_kwargs:
        (_ for _ in ()).throw(OSError("offline")))
    relay = StateRelay("http://source/api/state", _Enricher(),
                       refresh_seconds=0, retry_seconds=0,
                       cache_path=cache_path)
    response = relay.get()
    assert response.status == 200 and response.stale is True
    assert response.body == cached
