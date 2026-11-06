"""Audio decoding and resampling (soxr).

libsndfile (via soundfile) reads ogg/opus/mp3/wav/flac. Everything else that people get from video sites
and music stores (m4a/aac, webm, mp4, mkv, wma) goes through ffmpeg, via PyAV's bundled build."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

SNDFILE_EXT = {".mp3", ".ogg", ".opus", ".flac", ".wav"}
FFMPEG_EXT = {".m4a", ".aac", ".webm", ".mp4", ".mkv", ".mka", ".wma"}
AUDIO_EXT = SNDFILE_EXT | FFMPEG_EXT


def _decode_ffmpeg(path: Path) -> tuple[np.ndarray, int]:
    """First audio stream as float32 (n, ch). Video streams are ignored."""
    import av
    with av.open(str(path)) as c:
        if not c.streams.audio:
            raise ValueError(f"{path.name}: no audio stream")
        s = c.streams.audio[0]
        rs = av.AudioResampler(format="fltp")         # planar float, source layout and rate
        parts = []
        for frame in c.decode(s):
            for f in rs.resample(frame):
                parts.append(f.to_ndarray())
        for f in rs.resample(None):
            parts.append(f.to_ndarray())
        if not parts:
            raise ValueError(f"{path.name}: no audio decoded")
        return np.ascontiguousarray(np.concatenate(parts, axis=1).T, dtype=np.float32), s.rate


def load_audio(path: str | Path, sr: int | None = None, mono: bool = True) -> tuple[np.ndarray, int]:
    """Returns float32 audio shaped (n,) if mono else (n, ch), and its sample rate."""
    path = Path(path)
    if path.suffix.lower() in FFMPEG_EXT:
        y, file_sr = _decode_ffmpeg(path)
    else:
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
