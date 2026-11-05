"""Repair a generated track so it passes the playability checker (eval/playability.py, same thresholds).

Fixes, in order (thresholds per difficulty from configs/eval.yaml `playability`):
  out_of_audio     drop notes before 0 s or after the audio end
  open_chord       open + frets -> frets only (CH has no such chord)
  chord_size/span  drop inner / outermost frets until the chord fits
  stacked/too_fast of two chords closer than min_ioi_s keep the more salient one (beat position, then
                   chord size, then sustain), so off-beat ghost notes go first
  fast_jump        a jump of >= jump_frets faster than jump_ioi_s is pulled toward the previous chord
  sustains         shorter than min_sustain_s -> none; never past the next chord (legato ending allowed)
  bad_hopo         HOPO on the same frets as the previous chord -> strum
Note types are then re-derived under CH .chart rules: natural HOPO threshold res*65/192, the forced
flag set wherever the wanted type differs from the natural one (formats/hopo.py, from YARG.Core).
"""
from __future__ import annotations

import numpy as np

from ..formats.hopo import build_track, chart_hopo_threshold
from ..formats.song import OPEN_BIT, GuitarTrack, NoteType, Song, popcount

# Rhythmic salience of a 48th position: beat > 8th > 16th > triplets > finer
_SAL = np.zeros(48)
for _p in range(48):
    _SAL[_p] = 5 if _p == 0 else 4 if _p % 24 == 0 else 3 if _p % 12 == 0 else 2 if _p % 8 == 0 else \
        1.5 if _p % 4 == 0 else 1 if _p % 6 == 0 else 0


def salience(tick: np.ndarray, mask: np.ndarray, sustain: np.ndarray, res: int) -> np.ndarray:
    pos = (np.asarray(tick) % res) * 48 // res
    return _SAL[pos] + 0.3 * np.minimum(popcount(mask), 3) + 0.5 * (np.asarray(sustain) > 0)


def _fit_chord(m: int, max_chord: int, max_span: int) -> int:
    if m & OPEN_BIT:
        m = m & 0x1F or OPEN_BIT
    if m == OPEN_BIT:
        return m
    frets = [f for f in range(5) if m >> f & 1]
    while len(frets) > max_chord:              # drop inner frets first: keeps the chord's shape
        frets.pop(len(frets) // 2)
    while len(frets) > 1 and frets[-1] - frets[0] > max_span:
        frets.pop()
    return sum(1 << f for f in frets)


def _shift(m: int, k: int) -> int:
    if m & OPEN_BIT or k == 0:
        return m
    frets = [f for f in range(5) if m >> f & 1]
    k = max(-frets[0], min(4 - frets[-1], k))
    return sum(1 << (f + k) for f in frets)


def _pitch(m: int) -> float:
    if m & OPEN_BIT:
        return -1.0
    frets = [f for f in range(5) if m >> f & 1]
    return sum(frets) / len(frets)


def fix_track(song: Song, tr: GuitarTrack, rules: dict, audio_s: float | None = None,
              offset_s: float = 0.0) -> GuitarTrack:
    """Return a new track that satisfies `rules` (one difficulty's playability thresholds)."""
    if len(tr) == 0:
        return tr
    res, tm = song.resolution, song.tempo
    tick = tr.tick.astype(np.int64)
    t = tm.tick_to_sec(tick) + offset_s
    keep = t >= 0
    if audio_s is not None:
        keep &= t <= audio_s
    tick, mask, sus, ntype = tick[keep], tr.mask[keep].astype(int), tr.sustain[keep].astype(np.int64), tr.ntype[keep]
    t = t[keep]
    mask = np.array([_fit_chord(int(m), rules["max_chord"], rules["max_span"]) for m in mask], int)

    # too-close chords: keep the more salient of each pair, scanning left to right
    sal = salience(tick, mask.astype(np.uint8), sus, res)
    kept: list[int] = []
    for i in range(len(tick)):
        while kept and t[i] - t[kept[-1]] < rules["min_ioi_s"]:
            if sal[i] > sal[kept[-1]]:
                kept.pop()
            else:
                break
        if not kept or t[i] - t[kept[-1]] >= rules["min_ioi_s"]:
            kept.append(i)
    idx = np.array(kept, int)
    tick, mask, sus, ntype, t = tick[idx], mask[idx], sus[idx], ntype[idx], t[idx]

    # fast jumps: pull the later chord toward the previous one
    for i in range(1, len(tick)):
        if t[i] - t[i - 1] < rules["jump_ioi_s"] and not (mask[i] | mask[i - 1]) & OPEN_BIT:
            d = _pitch(int(mask[i])) - _pitch(int(mask[i - 1]))
            if abs(d) >= rules["jump_frets"]:
                need = int(np.ceil(abs(d) - (rules["jump_frets"] - 1)))
                mask[i] = _shift(int(mask[i]), -need if d > 0 else need)

    # sustains: drop tiny ones, end no later than the next chord
    nxt = np.concatenate([tick[1:], [np.iinfo(np.int64).max]])
    sus = np.minimum(sus, nxt - tick)
    sus_s = tm.tick_to_sec(tick + sus) - tm.tick_to_sec(tick)
    sus = np.where(sus_s < rules["min_sustain_s"], 0, sus)

    # note types: no HOPO on a repeated chord, then encode under CH rules
    ntype = ntype.copy()
    same = np.concatenate([[False], mask[1:] == mask[:-1]])
    ntype[same & (ntype == NoteType.HOPO)] = NoteType.STRUM
    return rebuild(tr.difficulty, tick, mask, sus, ntype, res, tr.star_power, tr.solos)


def rebuild(difficulty: str, tick, mask, sus, ntype, res: int, star_power=None, solos=None) -> GuitarTrack:
    """Build a .chart track whose final note types equal `ntype` (forced flags where needed)."""
    thr = chart_hopo_threshold(res)
    notes = [(int(tk), f, int(s)) for tk, m, s in zip(tick, mask, sus) for f in range(6) if int(m) >> f & 1]
    taps = {int(tk) for tk, nt in zip(tick, ntype) if nt == NoteType.TAP}
    want = {int(tk): int(nt) for tk, nt in zip(tick, ntype)}
    nat = build_track(difficulty, notes, thr, midi=False, tap_ticks=taps)
    forced = {int(tk) for tk, nt in zip(nat.tick, nat.ntype) if nt != NoteType.TAP and (nt == 1) != (want[int(tk)] == 1)}
    return build_track(difficulty, notes, thr, midi=False, tap_ticks=taps, chart_forced_ticks=forced,
                       star_power=list(star_power or []), solos=list(solos or []))
