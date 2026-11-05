"""Audio -> Clone Hero tempo map, with the Phase 3 defaults (see NOTES.md):
fine-tuned Beat This (fp32) on the full mix -> smoothed tempo map (20 ms tolerance) with
time signatures at every downbeat-spacing change, resolution 192, notes on a 48/beat grid.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..formats.song import TempoMap
from .tempo_map import beats_to_tempo_map
from .trackers import get_tracker

DEFAULTS = dict(tracker="finetuned", mode="smooth", tol_s=0.020, ts_mode="raw", resolution=192)


def track(y: np.ndarray, sr: int, tracker: str = DEFAULTS["tracker"], cache: Path | None = None):
    if cache is not None and cache.exists():
        z = np.load(cache)
        return z["beats"], z["downbeats"]
    b, d = get_tracker(tracker)(y, sr)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, beats=b, downbeats=d)
    return b, d


def tempo_map_from_audio(y: np.ndarray, sr: int, cache: Path | None = None, **kw) -> TempoMap:
    o = {**DEFAULTS, **kw}
    beats, downbeats = track(y, sr, o["tracker"], cache)
    return beats_to_tempo_map(beats, downbeats, o["resolution"], mode=o["mode"], tol_s=o["tol_s"], ts_mode=o["ts_mode"])
