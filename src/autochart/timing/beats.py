"""Beat This! beat/downbeat tracking with an on-disk cache (data/cache/beats/<song_id>.npz)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

_MODEL = None


def _model(device: str = "cuda"):
    global _MODEL
    if _MODEL is None:
        from beat_this.inference import Audio2Beats

        _MODEL = Audio2Beats(checkpoint_path="final0", device=device, dbn=False)
    return _MODEL


def track_beats(y: np.ndarray, sr: int, cache: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    if cache is not None and cache.exists():
        z = np.load(cache)
        return z["beats"], z["downbeats"]
    beats, downbeats = _model()(y, sr)
    beats, downbeats = np.asarray(beats, float), np.asarray(downbeats, float)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, beats=beats, downbeats=downbeats)
    return beats, downbeats
