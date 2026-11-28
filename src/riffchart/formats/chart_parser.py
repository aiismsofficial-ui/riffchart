""".chart parser (Moonscraper / Clone Hero text format).

Format references: YARG.Core MoonscraperChartParser/IO/Chart/ChartReader*.cs and
TheNathannator/GuitarGame_ChartFormats (.chart docs).
  [Song]       Resolution (default 192), Offset (seconds)
  [SyncTrack]  `t = B bpm*1000`, `t = TS num [log2(den)]`, `t = A microseconds` (anchor: the marker
  is locked at that time; kept for editors)
  [Events]     `t = E "section name"` (lyrics and phrases kept in Song.lyrics)
  [<Diff>Single]  `t = N fret len` (0-4 frets, 5 forced, 6 tap, 7 open), `t = S 2 len` star power,
                  `t = E solo` / `t = E soloend`
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .hopo import build_track, chart_hopo_threshold
from .song import DIFFICULTIES, Song, TempoMap
from .textio import decode_text

_SECTION = re.compile(r"^\s*\[(.+?)\]\s*$")
_LINE = re.compile(r"^\s*(\d+)\s*=\s*(\S+)\s*(.*?)\s*$")
_KV = re.compile(r'^\s*(\w+)\s*=\s*"?(.*?)"?\s*$')

TRACK_NAMES = {d: f"{d.capitalize()}Single" for d in DIFFICULTIES}
# .chart section suffix per instrument (Clone Hero / Moonscraper names)
INSTRUMENT_SECTIONS = {"guitar": "Single", "bass": "DoubleBass", "rhythm": "DoubleRhythm", "keys": "Keyboard"}


def track_names(instrument: str = "guitar") -> dict[str, str]:
    return {d: f"{d.capitalize()}{INSTRUMENT_SECTIONS[instrument]}" for d in DIFFICULTIES}


def split_sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    name = None
    body: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _SECTION.match(line)
        if m:
            if name is not None:
                sections.setdefault(name, body)
            name, body = m.group(1), []
            continue
        if line in ("{", "}"):
            continue
        if name is not None:
            body.append(line)
    if name is not None:
        sections.setdefault(name, body)
    return sections


def parse_chart(path: str | Path, ini: dict[str, str] | None = None, instrument: str = "guitar") -> Song:
    text = decode_text(Path(path).read_bytes())
    return parse_chart_text(text, ini, instrument)


def parse_chart_text(text: str, ini: dict[str, str] | None = None, instrument: str = "guitar") -> Song:
    secs = split_sections(text)
    warnings: list[str] = []
    meta: dict[str, str] = {}
    for line in secs.get("Song", []):
        m = _KV.match(line)
        if m:
            meta[m.group(1).lower()] = m.group(2)
    try:
        resolution = int(float(meta.get("resolution", "192")))
    except ValueError:
        resolution = 192
        warnings.append("bad Resolution, assuming 192")
    if resolution <= 0:
        raise ValueError(f"invalid resolution {resolution}")
    try:
        offset = float(meta.get("offset", "0") or 0)
    except ValueError:
        offset = 0.0

    tempo = _parse_sync(secs.get("SyncTrack", []), resolution, warnings)

    sections, lyrics = [], []
    for line in secs.get("Events", []):
        m = _LINE.match(line)
        if m and m.group(2) == "E":
            ev = m.group(3).strip().strip('"')
            if ev.startswith("section "):
                sections.append((int(m.group(1)), ev[8:].strip()))
            elif ev in ("phrase_start", "phrase_end") or ev.startswith("lyric "):
                lyrics.append((int(m.group(1)), ev))
    sections.sort()

    threshold = chart_hopo_threshold(resolution, ini)
    cutoff = _sustain_cutoff(ini)
    parts: dict[str, dict] = {}
    for inst in INSTRUMENT_SECTIONS:
        got = {diff: _parse_track(diff, secs[sec_name], threshold, cutoff, warnings if inst == instrument else [])
               for diff, sec_name in track_names(inst).items() if sec_name in secs}
        if got or inst == instrument:
            parts[inst] = got
    tracks = parts.pop(instrument)
    known = {"Song", "SyncTrack", "Events"} | {n for i in INSTRUMENT_SECTIONS for n in track_names(i).values()}
    # a part's section with no notes (only star power or solo marks) is kept as it was, not dropped
    empty = {sec for i, got in parts.items() for d, sec in track_names(i).items() if d in got and not len(got[d])}
    raw = {k: v for k, v in secs.items() if k not in known or k in empty}
    return Song(resolution, tempo, tracks, sections, meta, offset, "chart", warnings, lyrics, instrument,
                {k: v for k, v in parts.items() if any(len(t) for t in v.values())}, raw)


def _sustain_cutoff(ini: dict[str, str] | None) -> int:
    # .chart defaults to no sustain cutoff unless song.ini sets one (YARG SongEntry.IniBase.cs).
    try:
        return int(float(ini["sustain_cutoff_threshold"])) if ini and "sustain_cutoff_threshold" in ini else 0
    except ValueError:
        return 0


def _parse_sync(lines: list[str], resolution: int, warnings: list[str]) -> TempoMap:
    bpm_ev: dict[int, float] = {}
    ts_ev: dict[int, tuple[int, int]] = {}
    anchors: dict[int, float] = {}
    for line in lines:
        m = _LINE.match(line)
        if not m:
            continue
        tick, kind, rest = int(m.group(1)), m.group(2), m.group(3).split()
        try:
            if kind == "B":
                bpm = int(rest[0]) / 1000.0
                if bpm <= 0:
                    warnings.append(f"non-positive BPM at {tick}")
                    continue
                bpm_ev[tick] = bpm
            elif kind == "A":
                anchors[tick] = int(rest[0]) / 1e6
            elif kind == "TS":
                num = int(rest[0])
                den = 2 ** int(rest[1]) if len(rest) > 1 else 4
                ts_ev[tick] = (num, den)
        except (ValueError, IndexError):
            warnings.append(f"bad sync line: {line}")
    if not bpm_ev:
        warnings.append("no tempo events, assuming 120 BPM")
    for t in sorted(anchors):                       # an anchor without a tempo marker gets one (as Moonscraper does)
        if t not in bpm_ev and bpm_ev:
            before = [x for x in bpm_ev if x < t]
            bpm_ev[t] = bpm_ev[max(before)] if before else 120.0
    tk = sorted(bpm_ev)
    ts = sorted(ts_ev) or [0]
    if not ts_ev:
        ts_ev[0] = (4, 4)
    return TempoMap(
        resolution,
        np.array(tk, np.int64),
        np.array([bpm_ev[t] for t in tk], np.float64),
        np.array(ts, np.int64),
        np.array([ts_ev[t][0] for t in ts], np.int64),
        np.array([ts_ev[t][1] for t in ts], np.int64),
        anchors={t: v for t, v in anchors.items() if t in bpm_ev and t > 0},
    )


def _parse_track(diff: str, lines: list[str], threshold: int, cutoff: int, warnings: list[str]):
    notes: list[tuple[int, int, int]] = []
    forced: set[int] = set()
    taps: set[int] = set()
    sp: list[tuple[int, int]] = []
    solos: list[tuple[int, int]] = []
    solo_start = None
    for line in lines:
        m = _LINE.match(line)
        if not m:
            continue
        tick, kind, rest = int(m.group(1)), m.group(2), m.group(3)
        if kind == "N":
            parts = rest.split()
            try:
                fret, length = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
            except ValueError:
                warnings.append(f"bad note line: {line}")
                continue
            if 0 <= fret <= 4:
                notes.append((tick, fret, length if length >= cutoff else 0))
            elif fret == 7:
                notes.append((tick, 5, length if length >= cutoff else 0))
            elif fret == 5:
                forced.add(tick)
            elif fret == 6:
                taps.add(tick)
        elif kind == "S":
            parts = rest.split()
            if len(parts) >= 2 and parts[0] == "2":
                sp.append((tick, int(parts[1])))
        elif kind == "E":
            ev = rest.strip().strip('"')
            if ev == "solo":
                solo_start = tick
            elif ev == "soloend" and solo_start is not None:
                solos.append((solo_start, tick))
                solo_start = None
    notes.sort(key=lambda n: n[0])
    return build_track(diff, notes, threshold, midi=False, chart_forced_ticks=forced, tap_ticks=taps,
                       star_power=sorted(sp), solos=solos)
