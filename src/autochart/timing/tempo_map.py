"""Beat/downbeat times -> Clone Hero tempo map.

Tick 0 is audio time 0. The lead-in before the first beat is one tempo segment covering a whole
number of beats, so every detected beat lands exactly on a beat tick.

Tempo modes
  per_beat  one BPM event per beat (follows every wobble of the tracker)
  smooth    beats are grouped into steady regions; each region gets one BPM fitted by least squares
            with its start anchored at the previous region's end (so the map stays continuous). A
            region grows while every beat stays within `tol_s` of the fitted grid; drifting
            passages therefore fall back to (near) per-beat events. This mirrors how charters map:
            constant tempo where the band is steady, beat-by-beat where it isn't.

Time signatures come from downbeats: bar length = beats between consecutive downbeats, cleaned with
a run-length filter (a new bar length must hold for `min_ts_bars` bars) so single tracker mistakes
don't create one-bar TS flips. A pickup before the first downbeat gets its own short bar.
"""
from __future__ import annotations

import numpy as np

from ..formats.song import TempoMap


def _clean_beats(beats: np.ndarray) -> np.ndarray:
    b = np.asarray(beats, float)
    return b[np.concatenate([[True], np.diff(b) > 1e-3])] if len(b) else b


def _smooth_regions(b: np.ndarray, tol_s: float, min_beats: int) -> list[tuple[int, int, float]]:
    """Greedy anchored LS fit. Returns regions (i, j, period) covering beats i..j (j exclusive end idx)."""
    regions = []
    i, start = 0, b[0]
    n = len(b)
    while i < n - 1:
        best_j, best_p = i + 1, b[i + 1] - start
        j = i + 2
        while j < n:
            k = np.arange(1, j - i + 1)
            y = b[i + 1 : j + 1] - start
            p = float((k * y).sum() / (k * k).sum())
            if np.abs(y - k * p).max() > tol_s:
                break
            best_j, best_p = j, p
            j += 1
        if best_j - i < min_beats:  # too short to be "steady": take one beat exactly
            best_j, best_p = i + 1, b[i + 1] - start
        regions.append((i, best_j, best_p))
        start = start + (best_j - i) * best_p
        i = best_j
    return regions


def _ts_events_raw(beats: np.ndarray, downbeats: np.ndarray) -> list[tuple[int, int]]:
    """A TS event at every downbeat where the bar length changes (bar lines always re-anchor)."""
    idx = np.unique(np.clip(np.searchsorted(beats, np.asarray(downbeats) - 0.035), 0, len(beats) - 1))
    out = []
    for k, ln in enumerate(np.diff(idx)):
        if 1 <= ln <= 16 and (not out or out[-1][1] != ln):
            out.append((int(idx[k]), int(ln)))
    return out


def _ts_events(beats: np.ndarray, downbeats: np.ndarray, min_ts_bars: int) -> list[tuple[int, int]]:
    """Downbeats -> [(beat_index, bar_length)] time-signature events on a bar lattice.

    The current bar length is kept while detected downbeats fall on its lattice. A single downbeat
    off the lattice whose successor is back on it is treated as a tracker error and ignored. When
    downbeats leave the lattice for good, the map re-anchors at the new downbeat: an odd bar fills
    the gap and the new bar length is the most common length of the next `min_ts_bars` bars.
    """
    idx = np.unique(np.clip(np.searchsorted(beats, np.asarray(downbeats) - 0.035), 0, len(beats) - 1))
    if len(idx) < 2:
        return []
    lens = np.diff(idx)

    def mode(k):
        w = lens[k : k + max(min_ts_bars, 1)]
        w = w[(w >= 1) & (w <= 16)]
        return int(np.bincount(w).argmax()) if len(w) else 4

    cur = mode(0)
    anchor = int(idx[0])
    events = [(anchor, cur)]
    k = 1
    while k < len(idx):
        d = int(idx[k])
        if (d - anchor) % cur == 0:
            k += 1
            continue
        nxt = int(idx[k + 1]) if k + 1 < len(idx) else None
        if nxt is not None and (nxt - anchor) % cur == 0:   # one stray downbeat
            k += 1
            continue
        last = anchor + ((d - anchor) // cur) * cur          # last lattice bar line before d
        if last == events[-1][0]:
            events[-1] = (last, d - last)                    # the odd bar replaces that event
        else:
            events.append((last, d - last))                  # odd bar to re-sync
        cur = mode(k)
        anchor = d
        events.append((d, cur))
        k += 1
    # drop no-op repeats
    out = []
    for e in events:
        if not out or out[-1][1] != e[1]:
            out.append(e)
        elif out and out[-1][0] == e[0]:
            out[-1] = e
    return out


def beats_to_tempo_map(beats: np.ndarray, downbeats: np.ndarray | None = None, resolution: int = 480,
                       mode: str = "per_beat", tol_s: float = 0.012, min_beats: int = 4,
                       min_ts_bars: int = 2, ts_mode: str = "raw", bpm_decimals: int = 3) -> TempoMap:
    b = _clean_beats(beats)
    if len(b) < 2:
        return TempoMap(resolution, np.array([0]), np.array([120.0]))
    period = np.diff(b)
    n_lead = max(1, int(round(b[0] / period[0]))) if b[0] > 1e-3 else 0
    ticks, bpm = [], []
    if n_lead:
        ticks.append(0)
        bpm.append(60.0 * n_lead / b[0])
    if mode == "per_beat":
        for i, p in enumerate(period):
            ticks.append((n_lead + i) * resolution)
            bpm.append(60.0 / p)
    elif mode == "smooth":
        for i, j, p in _smooth_regions(b, tol_s, min_beats):
            ticks.append((n_lead + i) * resolution)
            bpm.append(60.0 / p)
    else:
        raise ValueError(mode)
    bpm = np.round(np.array(bpm), bpm_decimals)
    # merge consecutive identical tempos
    keep = np.concatenate([[True], bpm[1:] != bpm[:-1]])
    ticks, bpm = np.array(ticks)[keep], bpm[keep]

    ts_t, ts_n = [0], [4]
    if downbeats is not None and len(downbeats) >= 2:
        ev = _ts_events_raw(b, downbeats) if ts_mode == "raw" else _ts_events(b, downbeats, min_ts_bars)
        if ev:
            ts_t, ts_n = [], []
            first, first_len = ev[0]
            pickup = (n_lead + first) % first_len
            if pickup:
                ts_t.append((n_lead + first - pickup) * resolution)
                ts_n.append(int(pickup))
            for beat_i, ln in ev:
                t = (n_lead + beat_i) * resolution
                if ts_t and ts_t[-1] == t:
                    ts_n[-1] = int(ln)
                else:
                    ts_t.append(t)
                    ts_n.append(int(ln))
            if ts_t[0] != 0:
                ts_t.insert(0, 0)
                ts_n.insert(0, ts_n[0])
    return TempoMap(resolution, ticks, bpm.astype(float), np.array(ts_t), np.array(ts_n), np.full(len(ts_t), 4))
