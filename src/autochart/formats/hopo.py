"""Chord assembly and Clone Hero / YARG note-type resolution.

Rules mirror YARG.Core (MoonscraperChartParser/Events/MoonNote.cs, IsNaturalHopo/IsHopo,
and MidReader.cs ProcessEventAsGuitarForcedTypePostDelay):

natural HOPO  <=>  single note (not a chord)
              and a previous note exists
              and (prev is single: fret differs from prev
                   prev is chord : .chart -> always; .mid -> note not contained in prev chord)
              and tick - prev_tick <= threshold

.chart: final = TAP if tap flag, else HOPO if (natural XOR forced-flag) else STRUM.
.mid  : TAP marker wins, then force-HOPO marker, then force-strum marker, else natural.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .song import GuitarTrack, NoteType, popcount


def chart_hopo_threshold(resolution: int, ini: dict[str, str] | None = None) -> int:
    """Threshold in ticks for .chart files (YARG ChartReader.cs + SongEntry.IniBase.cs)."""
    t = _ini_threshold(resolution, ini)
    if t is None:
        return resolution * 65 // 192
    # IniBase: value derived from eighthnote_hopo/hopofreq gets +res/192 leniency for .chart;
    # an explicit hopo_frequency is used as-is.
    return t


def midi_hopo_threshold(resolution: int, ini: dict[str, str] | None = None) -> int:
    """Threshold in ticks for .mid files (YARG MidReader.cs: setting + 1, default res/3 + 1)."""
    t = _ini_threshold(resolution, ini, chart=False)
    return (t if t is not None else resolution // 3) + 1


def _ini_threshold(resolution: int, ini: dict[str, str] | None, chart: bool = True) -> int | None:
    if not ini:
        return None
    hf = _int(ini.get("hopo_frequency"))
    if hf is not None and hf > 0:
        return hf
    t = None
    if "eighthnote_hopo" in ini:
        eighth = str(ini["eighthnote_hopo"]).strip().lower() in ("1", "true", "yes")
        t = resolution // (2 if eighth else 3)
    elif _int(ini.get("hopofreq")) is not None:
        denom = {0: 24, 1: 16, 2: 12, 3: 8, 4: 6, 5: 4}.get(_int(ini["hopofreq"]))
        if denom:
            t = 4 * resolution // denom
    if t is not None and chart:
        t += int(resolution / 192)
    return t


def _int(v) -> int | None:
    try:
        return int(float(str(v).strip()))
    except (TypeError, ValueError):
        return None


def assemble_chords(notes: list[tuple[int, int, int]]):
    """notes: (tick, fret 0..5 with 5=open, sustain). Duplicate (tick, fret) keeps the last entry
    like Moonscraper's PushNote. Returns (ticks, masks, sustains, disjoint) sorted by tick."""
    by_tick: dict[int, dict[int, int]] = defaultdict(dict)
    for tick, fret, sus in notes:
        by_tick[tick][fret] = sus
    ticks = np.array(sorted(by_tick), np.int64)
    masks = np.zeros(len(ticks), np.uint8)
    sus = np.zeros(len(ticks), np.int64)
    disjoint = np.zeros(len(ticks), bool)
    for i, t in enumerate(ticks):
        d = by_tick[int(t)]
        m = 0
        for f in d:
            m |= 1 << f
        masks[i] = m
        vals = list(d.values())
        sus[i] = max(vals)
        disjoint[i] = len(set(vals)) > 1
    return ticks, masks, sus, disjoint


def natural_hopo(ticks: np.ndarray, masks: np.ndarray, threshold: int, midi_exception: bool) -> np.ndarray:
    n = len(ticks)
    out = np.zeros(n, bool)
    if n < 2:
        return out
    pc = popcount(masks)
    single = pc == 1
    prev_mask = masks[:-1]
    cur = masks[1:]
    prev_single = single[:-1]
    differs = np.where(prev_single, cur != prev_mask, True)
    if midi_exception:
        differs = np.where(prev_single, differs, (prev_mask & cur) == 0)
    close = (ticks[1:] - ticks[:-1]) <= threshold
    out[1:] = single[1:] & differs & close
    return out


def build_track(
    difficulty: str,
    notes: list[tuple[int, int, int]],
    threshold: int,
    *,
    midi: bool,
    chart_forced_ticks: set[int] | None = None,
    tap_ticks: set[int] | None = None,
    hopo_ranges: list[tuple[int, int]] | None = None,
    strum_ranges: list[tuple[int, int]] | None = None,
    tap_ranges: list[tuple[int, int]] | None = None,
    star_power: list[tuple[int, int]] | None = None,
    solos: list[tuple[int, int]] | None = None,
) -> GuitarTrack:
    if not notes:
        tr = GuitarTrack.empty(difficulty)
        tr.star_power, tr.solos = star_power or [], solos or []
        return tr
    ticks, masks, sus, disjoint = assemble_chords(notes)
    nat = natural_hopo(ticks, masks, threshold, midi_exception=midi)

    if midi:
        tap = _in_ranges(ticks, tap_ranges)
        fh = _in_ranges(ticks, hopo_ranges)
        fs = _in_ranges(ticks, strum_ranges)
        is_hopo = np.where(fh, True, np.where(fs, False, nat))
    else:
        tap = np.isin(ticks, list(tap_ticks or ()))
        fflag = np.isin(ticks, list(chart_forced_ticks or ()))
        is_hopo = nat ^ fflag
    ntype = np.where(tap, NoteType.TAP, np.where(is_hopo, NoteType.HOPO, NoteType.STRUM)).astype(np.uint8)
    forced = (~tap) & (is_hopo != nat)
    return GuitarTrack(difficulty, ticks, masks, sus, ntype, forced, disjoint, star_power or [], solos or [])


def _in_ranges(ticks: np.ndarray, ranges: list[tuple[int, int]] | None) -> np.ndarray:
    """Phrase ranges cover [start, end-1] (YARG excludes the phrase's last tick)."""
    out = np.zeros(len(ticks), bool)
    for s, e in ranges or ():
        last = e - 1 if e > s else e
        out |= (ticks >= s) & (ticks <= last)
    return out
