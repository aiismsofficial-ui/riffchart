"""Star power phrase placement.

Human statistics (127 train songs, Expert): 2.3 phrases per minute, about 52 beats apart, a median of
5 beats (p10 4, p90 8.8) and 11 notes (p10 5, p90 18) per phrase. 64-76% of lower difficulties use
exactly the Expert phrases, so the same ranges are copied to every difficulty.

Placement: split the charted span into equal segments, one per phrase. In each segment pick the
best one- or two-bar window starting on a downbeat. A window scores well with about 11 notes,
ending on a sustain (an easy place to whammy), and few chords spread over the window.
"""
from __future__ import annotations

import numpy as np

from ..formats.song import GuitarTrack, Song, popcount

RATE_PER_MIN = 2.3
TARGET_NOTES = 11


def _windows(song: Song, tr: GuitarTrack, end_s: float):
    res = song.resolution
    bars = song.tempo.sec_to_tick(song.tempo.downbeat_times(end_s)).round().astype(np.int64)
    for i, b in enumerate(bars):
        for nb in (1, 2):
            if i + nb >= len(bars):
                continue
            e = int(bars[i + nb])
            if (e - b) / res < 3 or (e - b) / res > 10:
                continue
            sel = np.flatnonzero((tr.tick >= b) & (tr.tick < e))
            if len(sel) < 4:
                continue
            n = len(sel)
            score = -abs(np.log(n / TARGET_NOTES))
            score += 0.6 * (tr.sustain[sel[-1]] > res // 2)
            score -= 0.4 * (popcount(tr.mask[sel]) > 1).mean()
            yield int(tr.tick[sel[0]]), int(tr.tick[sel[-1]]) + 1, score


def place_star_power(song: Song, tr: GuitarTrack, rate_per_min: float = RATE_PER_MIN) -> list[tuple[int, int]]:
    """Star power phrases (tick, length) for one track (normally Expert)."""
    if len(tr) < 20:
        return []
    on = song.tempo.tick_to_sec(tr.tick)
    span_s = on[-1] - on[0]
    n = max(1, int(round(rate_per_min * span_s / 60)))
    cands = list(_windows(song, tr, on[-1] + 1))
    if not cands:
        return []
    starts = song.tempo.tick_to_sec(np.array([c[0] for c in cands]))
    edges = np.linspace(on[0], on[-1] + 1e-3, n + 1)
    out: list[tuple[int, int]] = []
    for k in range(n):
        idx = [j for j in range(len(cands)) if edges[k] <= starts[j] < edges[k + 1]
               and (not out or cands[j][0] >= out[-1][0] + out[-1][1])]
        if idx:
            s, e, _ = cands[max(idx, key=lambda j: cands[j][2])]
            out.append((s, e - s))
    return out


def copy_phrases(phrases: list[tuple[int, int]], tr: GuitarTrack) -> list[tuple[int, int]]:
    """Keep the phrases that still contain a note of `tr` (lower difficulties)."""
    return [(s, n) for s, n in phrases if ((tr.tick >= s) & (tr.tick < s + n)).any()]
