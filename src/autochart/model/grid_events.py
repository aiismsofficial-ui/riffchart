"""Notes in audio seconds <-> events on a beat grid (beat times in seconds)."""
from __future__ import annotations

import numpy as np

from ..formats.song import OPEN_BIT
from ..timing.grid import GRID, MODEL_POSITIONS, MUSICAL_POSITIONS


def beat_position(t, beats: np.ndarray) -> np.ndarray:
    """Fractional beat index of times t (linear extrapolation with the edge periods)."""
    t = np.asarray(t, float)
    idx = np.arange(len(beats), dtype=float)
    out = np.interp(t, beats, idx)
    p0, p1 = max(beats[1] - beats[0], 1e-3), max(beats[-1] - beats[-2], 1e-3)   # guard duplicated edge beats
    out = np.where(t < beats[0], (t - beats[0]) / p0, out)
    return np.where(t > beats[-1], len(beats) - 1 + (t - beats[-1]) / p1, out)


def beat_time(units, beats: np.ndarray) -> np.ndarray:
    """Inverse of beat_position for grid units (beat * 48 + pos)."""
    x = np.asarray(units, float) / GRID
    idx = np.arange(len(beats), dtype=float)
    out = np.interp(x, idx, beats)
    p0, p1 = beats[1] - beats[0], beats[-1] - beats[-2]
    out = np.where(x < 0, beats[0] + x * p0, out)
    return np.where(x > len(beats) - 1, beats[-1] + (x - len(beats) + 1) * p1, out)


def snap_units(t, beats: np.ndarray, positions: np.ndarray = MUSICAL_POSITIONS) -> np.ndarray:
    bp = beat_position(t, beats)
    base = np.floor(bp)
    frac = (bp - base) * GRID
    cand = np.concatenate([positions, [GRID]])
    k = cand[np.abs(frac[:, None] - cand[None, :]).argmin(1)]
    return (base * GRID + k).astype(np.int64)


def downbeat_flags(beats: np.ndarray, downbeats: np.ndarray) -> np.ndarray:
    flags = np.zeros(len(beats), bool)
    if len(downbeats) and len(beats) > 1:
        i = np.clip(np.searchsorted(beats, downbeats), 1, len(beats) - 1)
        left = np.where(np.abs(downbeats - beats[i - 1]) <= np.abs(beats[i] - downbeats), i - 1, i)
        period = np.median(np.diff(beats))
        ok = np.abs(beats[left] - downbeats) < 0.3 * period
        flags[left[ok]] = True
    return flags


def notes_to_units(on, end, mask, ntype, beats: np.ndarray, positions: np.ndarray = MUSICAL_POSITIONS):
    """Snap chords to the grid. Returns (unit, mask, type, sus_units) with chords landing on the
    same unit merged (mask union, longest sustain)."""
    u = snap_units(on, beats, positions)
    e = np.round(beat_position(end, beats) * GRID).astype(np.int64)
    sus = np.where(np.asarray(end) > np.asarray(on) + 1e-6, np.maximum(e - u, 0), 0)
    sus = np.where(sus < 3, 0, sus)
    order = np.argsort(u, kind="stable")
    u, m, t, s = u[order], np.asarray(mask)[order].astype(int), np.asarray(ntype)[order], sus[order]
    keep = np.concatenate([[True], np.diff(u) > 0])
    gid = np.cumsum(keep) - 1
    mm = np.zeros(keep.sum(), int)
    np.bitwise_or.at(mm, gid, m)
    mm = np.where((mm & 0x1F) > 0, mm & 0x1F, mm)        # open + frets never occurs in human data
    ss = np.zeros(keep.sum(), int)
    np.maximum.at(ss, gid, s)
    return u[keep], mm, t[keep], ss
