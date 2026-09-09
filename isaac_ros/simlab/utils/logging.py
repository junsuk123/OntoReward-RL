"""Line logging that survives Kit's hard shutdown.

Kit can terminate the process without flushing Python's buffers, so every line
is written with ``flush=True`` rather than going through ``logging``.
"""

from __future__ import annotations

import sys
from typing import Callable

Logger = Callable[[str], None]


def get_logger(tag: str, stream=None) -> Logger:
    """Return ``log(message)`` that prints ``[tag] message`` and flushes."""
    target = stream if stream is not None else sys.stdout

    def log(message: str) -> None:
        print(f"[{tag}] {message}", file=target, flush=True)

    return log
