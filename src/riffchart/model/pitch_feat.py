"""Pitch input for the chart model: a semitone-resolution constant-Q spectrum of the guitar track.

The mel input (128 bands, 30 Hz-11 kHz) is coarse at guitar fundamentals (one band spans several
semitones below ~500 Hz), so the model can hear *when* notes start but not reliably *which way* the
melody moves; fret choice is its weakest skill (Expert exact ~0.2, contour ~0.67). A hand-written
pitch -> fret rule did not help (NOTES, "pitch-following frets step 1"); this input lets the model learn
the mapping from the charts instead.

    64 bins, 12 per octave, G1 (49 Hz) .. B6 (1976 Hz): the guitar's fundamental range incl. drop tunings
    same frames as the mel input (sr 22050, hop 256, centred), so both go through window_features alike
    stored as uint8 log-magnitude, LO..HI (-12..1.5 natural log), ~0.05 per step: 64 B per frame

    python -m riffchart.model.pitch_feat [--workers 8]     # data/features/model/pitch/<id>.npy, resumable
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..paths import PROJECT

SR, HOP = 22050, 256
BINS, BPO, FMIN_MIDI = 64, 12, 31
LO, HI = -12.0, 1.5
ROOT = PROJECT / "data/features/model"
PITCH_DIR = ROOT / "pitch"
STATS = ROOT / "pitch_stats.npz"


def cqt_uint8(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Mono audio at 22.05 kHz -> (T, 64) uint8, T = 1 + len // 256 (matches the mel frames)."""
    import librosa
    C = np.abs(librosa.cqt(y.astype(np.float32), sr=sr, hop_length=HOP, fmin=librosa.midi_to_hz(FMIN_MIDI),
                           n_bins=BINS, bins_per_octave=BPO))
    L = np.log(C + 1e-6).T
    return np.clip(np.round((L - LO) / (HI - LO) * 255), 0, 255).astype(np.uint8)


def dequant(q: np.ndarray) -> np.ndarray:
    return np.asarray(q, np.float32) * ((HI - LO) / 255.0) + LO


def to_bands(x: np.ndarray) -> np.ndarray:
    """(..., 64) -> (..., 128): each semitone bin twice, so the pitch channel has the mel channels' shape."""
    return np.repeat(x, 128 // BINS, axis=-1)


def _one(sid: str) -> tuple[str, float]:
    from ..audio.io import load_audio
    out = PITCH_DIR / f"{sid}.npy"
    if out.exists():
        return sid, 0.0
    t = time.perf_counter()
    y, _ = load_audio(PROJECT / "data/stems" / sid / "guitar.ogg", SR)
    y = y.mean(1) if y.ndim == 2 else y
    q = cqt_uint8(y)
    mel_T = np.load(ROOT / "mel" / f"{sid}.npy", mmap_mode="r").shape[0]
    if len(q) < mel_T:                                   # the mel was cut to min(guitar, mix): match it
        q = np.concatenate([q, np.zeros((mel_T - len(q), BINS), np.uint8)])
    tmp = out.with_suffix(".tmp.npy")
    np.save(tmp, q[:mel_T])
    tmp.replace(out)
    return sid, time.perf_counter() - t


def compute_stats(n_songs: int = 200) -> None:
    """Per-band mean / std of the (dequantised, 128-band) pitch input, like mel_stats.npz."""
    ids = sorted(p.stem for p in PITCH_DIR.glob("*.npy"))
    rng = np.random.default_rng(0)
    chunks = []
    for sid in rng.choice(ids, min(n_songs, len(ids)), replace=False):
        q = np.load(PITCH_DIR / f"{sid}.npy", mmap_mode="r")
        chunks.append(dequant(q[:: 8]))
    a = to_bands(np.concatenate(chunks))
    np.savez(STATS, mean=a.mean(0)[None], std=a.std(0)[None] + 1e-3)     # (1, 128) like one mel channel


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    PITCH_DIR.mkdir(parents=True, exist_ok=True)
    ids = sorted(p.stem for p in (ROOT / "mel").glob("*.npy"))
    todo = [s for s in ids if not (PITCH_DIR / f"{s}.npy").exists()]
    print(f"{len(ids)} songs, {len(todo)} to do", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        for k, (sid, sec) in enumerate(ex.map(_one, todo, chunksize=4), 1):
            if k % 25 == 0 or k == len(todo):
                el = time.time() - t0
                print(f"[{k}/{len(todo)}] {sid} {sec:.1f}s  ({el / 60:.1f} min, about {el / k * (len(todo) - k) / 60:.0f} min left)",
                      flush=True)
    compute_stats()
    print(f"done: {len(ids)} songs, stats -> {STATS}", flush=True)


if __name__ == "__main__":
    main()
