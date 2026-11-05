""".mid parser for the PART GUITAR track (the .mid chart conventions Clone Hero reads).

References: YARG.Core MoonscraperChartParser/IO/Midi/{MidReader.cs, MidReader.ProcessLists.cs,
MidIOHelper.cs, PhaseShiftSysEx.cs}; TheNathannator/GuitarGame_ChartFormats 5-Fret Guitar.md.
  difficulty base keys: easy 60, medium 72, hard 84, expert 96
  base+0..4 frets, base+5 force HOPO, base+6 force strum, base-1 open (only with ENHANCED_OPENS)
  104 tap (all difficulties), 103 solo, 116 star power (103 if song.ini multiplier_note=103)
  Phase Shift SysEx `50 53 00 00 <diff|FF> <code> <1=start|0=end>`: code 1 open, code 4 tap
Processing order follows YARG: SysEx (opens) first, then forced types; tap beats HOPO beats strum.
Sustains shorter than the cutoff (default res/3) become 0.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import mido
import numpy as np

from .hopo import build_track, midi_hopo_threshold
from .song import DIFFICULTIES, Song, TempoMap

DIFF_BASE = {"easy": 60, "medium": 72, "hard": 84, "expert": 96}
SYSEX_DIFF = {0: "easy", 1: "medium", 2: "hard", 3: "expert"}
GUITAR_TRACK_NAMES = ("PART GUITAR", "T1 GEMS")
TAP_NOTE, SOLO_NOTE, SP_NOTE = 104, 103, 116
TEXT_TYPES = {"text", "lyrics", "marker", "cue_marker", "track_name", "instrument_name"}


def _abs_events(track: mido.MidiTrack):
    t = 0
    for msg in track:
        t += msg.time
        yield t, msg


def _track_name(track: mido.MidiTrack) -> str:
    for msg in track:
        if msg.type == "track_name":
            return msg.name.strip()
    return ""


def parse_midi(path: str | Path, ini: dict[str, str] | None = None) -> Song:
    mid = mido.MidiFile(str(path), clip=True)
    if mid.type == 2:
        raise ValueError("MIDI type 2 not supported")
    res = mid.ticks_per_beat
    if res <= 0:
        raise ValueError("MIDI has no tick resolution (SMPTE timing?)")
    warnings: list[str] = []

    tempo = _parse_tempo(mid.tracks[0], res)

    sections = []
    guitar = None
    for tr in mid.tracks[1:]:
        name = _track_name(tr).upper()
        if name == "EVENTS":
            for t, msg in _abs_events(tr):
                if msg.type in TEXT_TYPES and msg.type != "track_name":
                    txt = msg.text.strip().strip("[]").strip()
                    if txt.startswith("section "):
                        sections.append((t, txt[8:].strip()))
                    elif txt.startswith("prc_"):
                        sections.append((t, txt[4:].strip()))
        elif name in GUITAR_TRACK_NAMES and guitar is None:
            guitar = tr
    if guitar is None:
        raise ValueError("no PART GUITAR track")

    sp_note = SP_NOTE
    if ini and str(ini.get("multiplier_note", "")).strip() == "103":
        sp_note = SOLO_NOTE
    try:
        cutoff = int(float(ini["sustain_cutoff_threshold"])) if ini and "sustain_cutoff_threshold" in ini else res // 3
    except ValueError:
        cutoff = res // 3
    cutoff = max(cutoff, 1)
    threshold = midi_hopo_threshold(res, ini)

    enhanced_opens = False
    open_notes: dict[int, list[tuple[int, int]]] = defaultdict(list)  # key -> (start, end)
    active: dict[tuple[int, int], list[int]] = defaultdict(list)
    spans: dict[int, list[tuple[int, int]]] = defaultdict(list)
    sysex_open: dict[str, list[tuple[int, int]]] = defaultdict(list)
    sysex_tap: dict[str, list[tuple[int, int]]] = defaultdict(list)
    sysex_active: dict[tuple[int, int], int] = {}

    for t, msg in _abs_events(guitar):
        if msg.type == "note_on" and msg.velocity > 0:
            active[(msg.note, msg.channel)].append(t)
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            stack = active.get((msg.note, msg.channel))
            if stack:
                spans[msg.note].append((stack.pop(0), t))
        elif msg.type in TEXT_TYPES:
            if "ENHANCED_OPENS" in getattr(msg, "text", getattr(msg, "name", "")):
                enhanced_opens = True
        elif msg.type == "sysex":
            d = msg.data
            if len(d) >= 7 and tuple(d[:3]) == (0x50, 0x53, 0x00) and d[3] == 0x00:
                diff, code, val = d[4], d[5], d[6]
                if code not in (1, 4):
                    continue
                key = (diff, code)
                if val == 1:
                    sysex_active[key] = t
                elif key in sysex_active:
                    start = sysex_active.pop(key)
                    # 0xFF = all difficulties; mido(clip=True) reads it as 0x7F.
                    diffs = DIFFICULTIES if diff in (0xFF, 0x7F) else (SYSEX_DIFF.get(diff),)
                    for dn in diffs:
                        if dn:
                            (sysex_open if code == 1 else sysex_tap)[dn].append((start, t))
    if any(active.values()):
        warnings.append("unterminated MIDI notes ignored")

    solos = spans.get(SOLO_NOTE, []) if sp_note != SOLO_NOTE else []
    star = [(s, e - s) for s, e in spans.get(sp_note, [])]
    taps = spans.get(TAP_NOTE, [])

    tracks = {}
    for diff, base in DIFF_BASE.items():
        notes: list[tuple[int, int, int]] = []
        for fret in range(5):
            for s, e in spans.get(base + fret, []):
                notes.append((s, fret, (e - s) if (e - s) >= cutoff else 0))
        if enhanced_opens:
            for s, e in spans.get(base - 1, []):
                notes.append((s, 5, (e - s) if (e - s) >= cutoff else 0))
        if sysex_open.get(diff):
            ranges = sysex_open[diff]
            notes = [(s, 5 if _covered(s, ranges) else f, l) for s, f, l in notes]
        if not notes and not star:
            continue
        notes.sort(key=lambda n: n[0])
        tracks[diff] = build_track(
            diff, notes, threshold, midi=True,
            hopo_ranges=spans.get(base + 5, []), strum_ranges=spans.get(base + 6, []),
            tap_ranges=taps + sysex_tap.get(diff, []),
            star_power=sorted(star), solos=sorted(solos),
        )
    # Drop difficulty entries that ended up with no notes (only phrases).
    tracks = {k: v for k, v in tracks.items() if len(v)}
    return Song(res, tempo, tracks, sorted(sections), {}, 0.0, "mid", warnings)


def _covered(t: int, ranges: list[tuple[int, int]]) -> bool:
    # Open-note SysEx phrases exclude their last tick (YARG MidReader.cs).
    return any(s <= t <= (e - 1 if e > s else e) for s, e in ranges)


def _parse_tempo(track: mido.MidiTrack, res: int) -> TempoMap:
    tk, bpm, ts_t, ts_n, ts_d = [], [], [], [], []
    for t, msg in _abs_events(track):
        if msg.type == "set_tempo" and msg.tempo > 0:
            if tk and tk[-1] == t:
                bpm[-1] = 60_000_000 / msg.tempo
            else:
                tk.append(t)
                bpm.append(60_000_000 / msg.tempo)
        elif msg.type == "time_signature":
            if ts_t and ts_t[-1] == t:
                ts_n[-1], ts_d[-1] = msg.numerator, msg.denominator
            else:
                ts_t.append(t)
                ts_n.append(msg.numerator)
                ts_d.append(msg.denominator)
    if not ts_t or ts_t[0] != 0:
        ts_t, ts_n, ts_d = [0] + ts_t, [4] + ts_n, [4] + ts_d
    return TempoMap(res, np.array(tk, np.int64), np.array(bpm), np.array(ts_t, np.int64),
                    np.array(ts_n, np.int64), np.array(ts_d, np.int64))
