"""One way to rewrite a song's notes.chart: under the song's lock, old version into history, atomic replace.

The editor's Save, Bring up to date (bass / keys / lyrics), Find lyrics and Remove part all read the chart,
change one thing and write it back. They run on different threads (the web server, the job queue), so each
read-modify-write holds lock(folder); a long step (charting a part) runs outside the lock and re-reads the
chart inside it, so an editor save made meanwhile is never overwritten.
"""
from __future__ import annotations

import os
import shutil
import threading
import time
from pathlib import Path

import numpy as np

_locks: dict[str, threading.RLock] = {}
_guard = threading.Lock()


def lock(folder: str | Path) -> threading.RLock:
    key = os.path.normcase(str(Path(folder).resolve()))
    with _guard:
        return _locks.setdefault(key, threading.RLock())


def same_timing(a, b) -> bool:
    """Same resolution, tempo map and offset: ticks mean the same times in both songs."""
    ta, tb = a.tempo, b.tempo
    return (a.resolution == b.resolution and abs(a.chart_offset_s - b.chart_offset_s) < 1e-6
            and np.array_equal(ta.ticks, tb.ticks) and np.allclose(ta.bpm, tb.bpm)
            and np.array_equal(ta.ts_ticks, tb.ts_ticks) and np.array_equal(ta.ts_num, tb.ts_num)
            and np.array_equal(ta.ts_den, tb.ts_den))


def history_copy(folder: Path, chart_path: Path, reason: str) -> Path:
    """Copy the current chart into .riffchart/history as <stamp>-<reason>-notes.<ext>."""
    hist = Path(folder) / ".riffchart" / "history"
    hist.mkdir(parents=True, exist_ok=True)
    dest = hist / time.strftime(f"%Y%m%d-%H%M%S-{reason}-notes{chart_path.suffix}")
    shutil.copy2(chart_path, dest)
    return dest


def write_atomic(path: Path, text: str) -> None:
    """Write next to it, then replace: a reader never sees half a chart, a full disk never leaves one."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def rewrite(folder: Path, chart_path: Path, song, ini: dict, reason: str) -> None:
    """Old chart into history, then the song written as notes.chart (atomically). Hold lock(folder)."""
    from ..formats.chart_writer import write_chart_text
    history_copy(folder, chart_path, reason)
    write_atomic(Path(folder) / "notes.chart", write_chart_text(
        song, name=ini.get("name", ""), artist=ini.get("artist", ""),
        charter=ini.get("charter", "RiffChart"), ini=ini))
