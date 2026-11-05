"""Audio decoding (libsndfile via soundfile: ogg/opus/mp3/wav/flac) and resampling (soxr)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import soxr


def load_audio(path: str | Path, sr: int | None = None, mono: bool = True) -> tuple[np.ndarray, int]:
    """Returns float32 audio shaped (n,) if mono else (n, ch), and its sample rate."""
    y, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    if mono:
        y = y.mean(axis=1)
    if sr is not None and sr != file_sr:
        y = soxr.resample(y, file_sr, sr, quality="HQ").astype(np.float32)
        file_sr = sr
    return y, file_sr


def load_mix(paths: list[Path], sr: int, mono: bool = True) -> np.ndarray:
    """Sum several stems (e.g. song + guitar + rhythm) into one mix, padding to the longest."""
    """Non-mono output is always stereo (mono files are duplicated, >2 channels keep the first two)."""
    parts = [load_audio(p, sr, mono)[0] for p in paths]
    n = max(len(p) for p in parts)
    out = np.zeros((n,) if mono else (n, 2), np.float32)
    for p in parts:
        if not mono:
            p = np.repeat(p[:, :1], 2, axis=1) if p.shape[1] == 1 else p[:, :2]
        out[: len(p)] += p
    return out
