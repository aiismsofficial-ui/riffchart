""".chart parser (Moonscraper / Clone Hero text format).

Format references: YARG.Core MoonscraperChartParser/IO/Chart/ChartReader*.cs and
TheNathannator/GuitarGame_ChartFormats (.chart docs).
  [Song]       Resolution (default 192), Offset (seconds)
  [SyncTrack]  `t = B bpm*1000`, `t = TS num [log2(den)]`, `t = A ...` (anchor, ignored)
  [Events]     `t = E "section name"` (also lyrics/phrases, ignored)
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


def parse_chart(path: str | Path, ini: dict[str, str] | None = None) -> Song:
    text = decode_text(Path(path).read_bytes())
    return parse_chart_text(text, ini)


def parse_chart_text(text: str, ini: dict[str, str] | None = None) -> Song:
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

    sections = []
    for line in secs.get("Events", []):
        m = _LINE.match(line)
        if m and m.group(2) == "E":
            ev = m.group(3).strip().strip('"')
            if ev.startswith("section "):
                sections.append((int(m.group(1)), ev[8:].strip()))
    sections.sort()

    threshold = chart_hopo_threshold(resolution, ini)
    cutoff = _sustain_cutoff(ini)
    tracks = {}
    for diff, sec_name in TRACK_NAMES.items():
        if sec_name in secs:
            tracks[diff] = _parse_track(diff, secs[sec_name], threshold, cutoff, warnings)
    return Song(resolution, tempo, tracks, sections, meta, offset, "chart", warnings)


def _sustain_cutoff(ini: dict[str, str] | None) -> int:
    # .chart defaults to no sustain cutoff unless song.ini sets one (YARG SongEntry.IniBase.cs).
    try:
        return int(float(ini["sustain_cutoff_threshold"])) if ini and "sustain_cutoff_threshold" in ini else 0
    except ValueError:
        return 0


def _parse_sync(lines: list[str], resolution: int, warnings: list[str]) -> TempoMap:
    bpm_ev: dict[int, float] = {}
    ts_ev: dict[int, tuple[int, int]] = {}
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
            elif kind == "TS":
                num = int(rest[0])
                den = 2 ** int(rest[1]) if len(rest) > 1 else 4
                ts_ev[tick] = (num, den)
        except (ValueError, IndexError):
            warnings.append(f"bad sync line: {line}")
    if not bpm_ev:
        warnings.append("no tempo events, assuming 120 BPM")
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
