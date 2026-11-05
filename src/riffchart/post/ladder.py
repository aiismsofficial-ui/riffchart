"""Difficulty ladder: each easier difficulty stays below the one above it.

Hard / Medium / Easy are generated directly by the model, one difficulty at a time, with nothing tying
them to Expert. Only ~25% of training songs have genuine reductions (the rest copy Expert), so on some
songs the lower difficulties come out as dense as Expert, and locally they can be busier than Expert
(Creep, 2026-11-04: Hard kept charting through a quiet-guitar stretch at twice Expert's rate). This pass,
run after decoding (and the density control), enforces:

  song-wide  chords <= cap x Expert chords (human median ratios 0.78 / 0.56 / 0.43; caps sit above them)
  local      in every window of `window_beats` beats: no more chords, and no more multi-fret chords,
             than the next harder difficulty in the same window

Excess notes go weakest-first by the learned reducer's keep probability (post/reduce.py); excess chords
become single notes (the lowest fret of the chord, which keeps the melodic contour). HOPOs are then
re-derived as in the reducer; the playability fixer runs afterwards.
"""
from __future__ import annotations

import numpy as np

from ..formats.song import OPEN_BIT, GuitarTrack, Song, popcount
from .reduce import _finish, features

ORDER = ("expert", "hard", "medium", "easy")
DEFAULT_CAPS = {"hard": 0.90, "medium": 0.75, "easy": 0.60}


def _scores(song: Song, tr: GuitarTrack, models: dict | None) -> np.ndarray:
    if models is None or not len(tr):                 # no reducer: prefer on-beat chords, then earlier
        res = song.resolution
        return 1.0 - (tr.tick % res > 0) * 0.5 - (tr.tick % (res // 2) > 0) * 0.25
    return models["keep"].predict_proba(features(song, tr, tr.difficulty))[:, 1]


def _lowest_fret(m: int) -> int:
    if m & OPEN_BIT:
        return OPEN_BIT
    return m & -m


def ladder(song: Song, tracks: dict[str, GuitarTrack], models: dict | None = None,
           caps: dict[str, float] | None = None, window_beats: int = 16) -> dict[str, GuitarTrack]:
    caps = {**DEFAULT_CAPS, **(caps or {})}
    out = dict(tracks)
    ex = out.get("expert")
    if ex is None or not len(ex):
        return out
    win = song.resolution * window_beats
    for k in range(1, len(ORDER)):
        d = ORDER[k]
        tr = out.get(d)
        if tr is None or not len(tr):
            continue
        upper = next((out[u] for u in reversed(ORDER[:k]) if u in out and len(out[u])), ex)
        score = _scores(song, tr, models)
        keep = np.ones(len(tr), bool)
        # local: per window, at most as many chords as the harder difficulty
        wi = tr.tick // win
        up_w = upper.tick // win
        up_count = np.bincount(up_w, minlength=int(max(wi.max(), up_w.max())) + 1)
        for w in np.unique(wi):
            idx = np.flatnonzero(wi == w)
            allow = int(up_count[w]) if w < len(up_count) else 0
            if len(idx) > allow:
                keep[idx[np.argsort(score[idx], kind="stable")[: len(idx) - allow]]] = False
        # song-wide cap relative to Expert
        n_max = int(round(caps[d] * len(ex)))
        if keep.sum() > n_max:
            alive = np.flatnonzero(keep)
            keep[alive[np.argsort(score[alive], kind="stable")[: len(alive) - n_max]]] = False
        masks = tr.mask.astype(int).copy()
        # local: no more multi-fret chords than the harder difficulty in the same window
        up_chord = popcount(upper.mask) > 1
        up_chords = np.bincount(up_w[up_chord], minlength=len(up_count)) if up_chord.any() else np.zeros(len(up_count), int)
        is_chord = (popcount(tr.mask) > 1) & keep
        for w in np.unique(wi[is_chord]):
            idx = np.flatnonzero((wi == w) & is_chord)
            allow = int(up_chords[w]) if w < len(up_chords) else 0
            if len(idx) > allow:
                for i in idx[np.argsort(score[idx], kind="stable")[: len(idx) - allow]]:
                    masks[i] = _lowest_fret(int(masks[i]))
        if keep.all() and (masks == tr.mask).all():
            continue
        out[d] = _finish(song, tr, d, keep, masks)
    return out
