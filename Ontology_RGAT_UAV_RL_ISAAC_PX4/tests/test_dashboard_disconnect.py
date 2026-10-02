"""Expected browser disconnects do not flood the experiment log."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler

import pytest

from ontology_rgat.viz.dashboard import _Handler


@pytest.mark.parametrize("disconnect", [BrokenPipeError, ConnectionResetError])
def test_handler_swallows_expected_client_disconnects(monkeypatch, disconnect):
    def disconnected(_self):
        raise disconnect("dashboard browser closed")

    monkeypatch.setattr(BaseHTTPRequestHandler, "handle", disconnected)
    _Handler.handle(object.__new__(_Handler))


def test_handler_does_not_hide_unexpected_server_errors(monkeypatch):
    def failed(_self):
        raise RuntimeError("dashboard bug")

    monkeypatch.setattr(BaseHTTPRequestHandler, "handle", failed)
    with pytest.raises(RuntimeError, match="dashboard bug"):
        _Handler.handle(object.__new__(_Handler))
