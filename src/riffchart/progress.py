"""How far through the current stage the charting is (0..1), for the app's progress bar. The product code sets a
hook per stage (product/core.py); the slow loops (Demucs separation, the chart model's windows) report into it.
Per thread, so a charting job never reports into another one."""
from __future__ import annotations

import threading
from typing import Callable

_local = threading.local()


def set_hook(fn: Callable[[float], None] | None) -> None:
    _local.fn = fn


def report(frac: float) -> None:
    fn = getattr(_local, "fn", None)
    if fn is not None:
        try:
            fn(min(1.0, max(0.0, float(frac))))
        except Exception:                                # a progress display must never stop the charting
            pass
