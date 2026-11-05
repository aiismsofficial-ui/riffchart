"""Make fast single-note runs follow the guitar's pitch.

The chart model hears the guitar as a mel spectrogram with no explicit pitch, so its timing is good
(Expert F1@50 ~0.85) but its fret choice is weak (exact ~0.2, contour ~0.67): in fast runs and solos it
tends to repeat a fret where the guitar walks up or down (user report, 2026-11-04). This pass:

  1. tracks the pitch of the guitar (play-along) track with CREPE (torchcrepe, GPU, 10 ms frames) and
     takes each note's pitch as the median over its first ~100 ms where CREPE is confident
     (periodicity >= `voiced`). A constant-Q peak picker was tried first and failed: its pitch direction
     agreed with human fret direction only 58% of the time (CREPE: 75%, 89% when confident);
  2. finds runs of single-fret notes (>= `min_run` notes, gaps <= `run_gap_s`) whose pitch is clear
     (median periodicity >= `min_conf`);
  3. where the model's frets disagree with the pitch movement (agreement < `agree_thr`), re-assigns that
     run's frets: local pitch rank (unique semitones within +-`window` notes; <= 5 of them map
     one-to-one, more are spread over the frets), placed near the model's hand position, then adjacent
     notes are nudged so a pitch change is never the same fret and a repeated pitch is.

Timing, chords, sustains and note types are untouched; the playability fixer runs afterwards (HOPOs that
land on a repeated fret become strums there).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..formats.song import OPEN_BIT, GuitarTrack, Song, popcount
from .playability import rebuild

CREPE_SR, CREPE_HOP = 16000, 160          # 10 ms frames
MAX_FRET = {"expert": 4, "hard": 4, "medium": 3, "easy": 3}
DEFAULTS = {"min_run": 4, "run_gap_s": 0.30, "agree_thr": 0.7, "min_conf": 0.6, "voiced": 0.3, "window": 6}


@dataclass
class PitchTrack:
    midi: np.ndarray          # per frame
    periodicity: np.ndarray   # 0..1, CREPE's voicing confidence
    frame_s: float = CREPE_HOP / CREPE_SR


def track_pitch(y: np.ndarray, sr: int, device: str = "cuda", model: str = "full") -> PitchTrack:
    """Mono audio -> CREPE pitch track (guitar range 60-1400 Hz)."""
    import soxr
    import torch
    import torchcrepe
    x = soxr.resample(y.astype(np.float32), sr, CREPE_SR) if sr != CREPE_SR else y.astype(np.float32)
    dev = device if torch.cuda.is_available() or device == "cpu" else "cpu"
    f0, per = torchcrepe.predict(torch.from_numpy(x)[None], CREPE_SR, CREPE_HOP, 60, 1400, model,
                                 batch_size=2048, device=dev, return_periodicity=True)
    f0, per = f0[0].cpu().numpy(), per[0].cpu().numpy()
    return PitchTrack(69 + 12 * np.log2(np.maximum(f0, 1.0) / 440.0), per)


def onset_pitches(pt: PitchTrack, on: np.ndarray, end: np.ndarray, voiced: float = 0.3):
    """Pitch (MIDI, float; nan if unclear) and confidence (mean periodicity) at each note start."""
    n = len(pt.midi)
    pitch, conf = np.full(len(on), np.nan), np.zeros(len(on))
    for i, (a, b) in enumerate(zip(on, end)):
        f0 = int((a + 0.02) / pt.frame_s)
        f1 = min(int(min(b, a + 0.12) / pt.frame_s), n)
        if f1 <= f0:
            continue
        w = pt.periodicity[f0:f1]
        if w.max() < voiced:
            continue
        pitch[i] = float(np.median(pt.midi[f0:f1][w >= voiced]))
        conf[i] = float(w.mean())
    return pitch, conf


def _agreement(frets: np.ndarray, q: np.ndarray) -> float:
    dq, df = np.diff(q), np.diff(frets)
    ok = np.where(dq == 0, df == 0, np.sign(df) == np.sign(dq))
    return float(ok.mean()) if len(ok) else 1.0


def assign_run(frets: np.ndarray, q: np.ndarray, max_fret: int, window: int) -> np.ndarray:
    """New frets for one run: local pitch rank near the model's hand position."""
    n, width = len(q), max_fret + 1
    out = np.zeros(n, int)
    for i in range(n):
        lo, hi = max(0, i - window), min(n, i + window + 1)
        u = np.unique(q[lo:hi])
        r = int(np.searchsorted(u, q[i]))
        if len(u) <= width:
            off = int(np.clip(round(frets[lo:hi].mean() - (len(u) - 1) / 2), 0, width - len(u)))
            out[i] = r + off
        else:
            out[i] = int(round(r / (len(u) - 1) * max_fret))
    for i in range(1, n):                                    # never contradict the pitch movement
        d = q[i] - q[i - 1]
        if d == 0:
            out[i] = out[i - 1]
        elif (d > 0) != (out[i] > out[i - 1]) or out[i] == out[i - 1]:
            step = 1 if d > 0 else -1
            for cand in (out[i - 1] + step, out[i - 1] - step):
                if 0 <= cand <= max_fret and cand != out[i - 1]:
                    out[i] = cand
                    break
    return np.clip(out, 0, max_fret)


def follow_pitch(song: Song, tr: GuitarTrack, pt: PitchTrack, offset_s: float = 0.0,
                 cfg: dict | None = None) -> tuple[GuitarTrack, int]:
    """Returns the track and the number of notes whose fret changed."""
    c = {**DEFAULTS, **(cfg or {})}
    n = len(tr)
    if n < c["min_run"]:
        return tr, 0
    on = song.tempo.tick_to_sec(tr.tick) + offset_s
    nxt = np.concatenate([on[1:], [on[-1] + 0.5]])
    pitch, conf = onset_pitches(pt, on, nxt, c["voiced"])
    single = (popcount(tr.mask) == 1) & ((tr.mask & OPEN_BIT) == 0)
    fret = np.array([int(m).bit_length() - 1 if m else 0 for m in tr.mask])
    mask = tr.mask.copy()
    changed = np.zeros(n, bool)
    max_fret = MAX_FRET.get(tr.difficulty, 4)
    i = 0
    while i < n:
        if not single[i] or np.isnan(pitch[i]):
            i += 1
            continue
        j = i + 1
        while j < n and single[j] and not np.isnan(pitch[j]) and on[j] - on[j - 1] <= c["run_gap_s"]:
            j += 1
        if j - i >= c["min_run"] and np.median(conf[i:j]) >= c["min_conf"]:
            q = np.round(pitch[i:j]).astype(int)
            if _agreement(fret[i:j], q) < c["agree_thr"]:
                new = assign_run(np.minimum(fret[i:j], max_fret), q, max_fret, c["window"])
                for k, f in enumerate(new, start=i):
                    if f != fret[k]:
                        mask[k] = 1 << f
                        changed[k] = True
        i = j
    if not changed.any():
        return tr, 0
    out = rebuild(tr.difficulty, tr.tick, mask, tr.sustain, tr.ntype.copy(), song.resolution,
                  tr.star_power, tr.solos)
    return out, int(changed.sum())
