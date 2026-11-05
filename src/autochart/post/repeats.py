"""Repeated-section consistency: bars that repeat earlier bars in the audio get the earlier bars' chart.

Bar features: the beat-synchronous mel (guitar stem + mix) pooled to 32 bands x 12 steps per beat,
centred per song and L2-normalised per bar. Bar j repeats bar i (i < j, same number of beats) when
cos(i, j) >= thr and a neighbouring bar pair also matches (cos(i-1, j-1) or cos(i+1, j+1) >= thr), so
an isolated look-alike bar is not enough. Chains resolve to the first occurrence.
The generated chart's tempo map has one tick-beat per audio beat (generate.events_to_song), so copying
bar i to bar j is a tick shift of (j0 - i0) * resolution.
"""
from __future__ import annotations

import numpy as np

from ..formats.song import GuitarTrack, Song
from ..model.dataset import window_features
from ..model.grid_events import downbeat_flags
from .playability import rebuild


def bars_from_grid(beats: np.ndarray, downbeats: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(downbeat_flags(beats, downbeats))
    return [(int(a), int(b)) for a, b in zip(starts[:-1], starts[1:])]


def bar_features(mel: np.ndarray, beats: np.ndarray, bars: list[tuple[int, int]]) -> list[np.ndarray]:
    nb = len(beats) - 1
    x = window_features(mel, beats, 0, nb)                       # (2, 128, nb*48)
    x = x.reshape(2, 32, 4, nb, 12, 4).mean(axis=(2, 5))         # (2, 32 bands, nb beats, 12 steps)
    x = x - x.mean(axis=(2, 3), keepdims=True)
    out = []
    for a, b in bars:
        f = x[:, :, a:b].ravel()
        out.append(f / (np.linalg.norm(f) + 1e-8))
    return out


def find_repeats(feats: list[np.ndarray], bars: list[tuple[int, int]], thr: float) -> dict[int, int]:
    n = len(bars)
    length = np.array([b - a for a, b in bars])
    sim = np.full((n, n), -1.0)
    for L in np.unique(length):
        idx = np.flatnonzero(length == L)
        F = np.stack([feats[i] for i in idx])
        sim[np.ix_(idx, idx)] = F @ F.T
    rep: dict[int, int] = {}
    for j in range(1, n):
        best, bi = thr, -1
        for i in range(j):
            if sim[i, j] < best:
                continue
            nb = max(sim[i - 1, j - 1] if i > 0 else -1, sim[i + 1, j + 1] if j + 1 < n else -1)
            if nb >= thr:
                best, bi = sim[i, j], i
        if bi >= 0:
            rep[j] = rep.get(bi, bi)
    return rep


def apply_repeats(song: Song, tr: GuitarTrack, beats: np.ndarray, bars: list[tuple[int, int]],
                  rep: dict[int, int]) -> GuitarTrack:
    if not rep or len(tr) == 0:
        return tr
    res = song.resolution
    lead = int(round(float(song.tempo.sec_to_tick(beats[0])) / res))
    span = lambda k: ((lead + bars[k][0]) * res, (lead + bars[k][1]) * res)
    keep = np.ones(len(tr), bool)
    add = []
    for j, i in rep.items():
        j0, j1 = span(j)
        i0, i1 = span(i)
        keep &= ~((tr.tick >= j0) & (tr.tick < j1))
        sel = (tr.tick >= i0) & (tr.tick < i1)
        add.append((tr.tick[sel] - i0 + j0, tr.mask[sel], tr.sustain[sel], tr.ntype[sel]))
    parts = [(tr.tick[keep], tr.mask[keep], tr.sustain[keep], tr.ntype[keep])] + add
    tick, mask, sus, ntype = (np.concatenate([p[k] for p in parts]) for k in range(4))
    o = np.argsort(tick, kind="stable")
    tick, mask, sus, ntype = tick[o], mask[o], sus[o], ntype[o]
    nxt = np.concatenate([tick[1:], [np.iinfo(np.int64).max]])
    sus = np.minimum(sus, nxt - tick)                             # copied sustains may now overlap
    return rebuild(tr.difficulty, tick, mask.astype(int), sus, ntype, res, tr.star_power, tr.solos)
