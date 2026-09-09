from __future__ import annotations

import socket
from collections.abc import Callable
from typing import Any

from .protocol import MAX_DATAGRAM, ProtocolError, decode, encode, now_ns


class DatagramServer:
    """Small non-blocking, versioned command endpoint owned by the ROS timer."""

    def __init__(self, host: str, port: int, version: int, handler: Callable[[dict[str, Any]], None]):
        self.version = version
        self.handler = handler
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # No SO_REUSEADDR: on Linux it lets a second gateway bind this unicast
        # port, and the control link then silently goes to whichever bound
        # last. Failing to start is the useful outcome.
        self.socket.bind((host, port))
        self.socket.setblocking(False)
        self.peer: tuple[str, int] | None = None

    def poll(self, limit: int = 20) -> None:
        for _ in range(limit):
            try:
                raw, peer = self.socket.recvfrom(MAX_DATAGRAM + 1)
            except BlockingIOError:
                return
            self.peer = peer
            try:
                self.handler(decode(raw, self.version))
            except (ProtocolError, PermissionError, ValueError) as exc:
                self.send({"v": self.version, "type": "error", "seq": 0,
                           "time_ns": now_ns(), "error": str(exc)})

    def send(self, message: dict[str, Any], peer: tuple[str, int] | None = None) -> None:
        destination = peer or self.peer
        if destination is not None:
            self.socket.sendto(encode(message), destination)

    def close(self) -> None:
        self.socket.close()

