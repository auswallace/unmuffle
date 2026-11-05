"""Progress messages. Prints by default; a listener (the web page) can also receive them."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable

_listeners: list[Callable[[str], None]] = []


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    for fn in list(_listeners):
        with contextlib.suppress(Exception):  # a broken listener must never stop a run
            fn(msg)


def add_listener(fn: Callable[[str], None]) -> None:
    _listeners.append(fn)


def remove_listener(fn: Callable[[str], None]) -> None:
    if fn in _listeners:
        _listeners.remove(fn)


def stamp(t: float) -> str:
    """Seconds as m:ss."""
    return f"{int(t // 60)}:{int(t % 60):02d}"
