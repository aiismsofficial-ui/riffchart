"""Write a Song to Clone Hero .chart text.

Forced (N 5) flags are recomputed against .chart natural-HOPO rules at the output resolution, so a
track's resolved note types survive the round trip even when the source was a .mid (whose HOPO
threshold and chord-subset rule differ).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .chart_parser import TRACK_NAMES
from .hopo import chart_hopo_threshold, natural_hopo
from .song import GuitarTrack, NoteType, Song


def _q(s: str) -> str:
    return '"' + str(s).replace('"', "'") + '"'


def write_chart_text(song: Song, *, name: str = "", artist: str = "", charter: str = "riffchart",
                     music_stream: str = "song.ogg", ini: dict[str, str] | None = None) -> str:
    res = song.resolution
    out = ["[Song]", "{",
           f"  Name = {_q(name)}", f"  Artist = {_q(artist)}", f"  Charter = {_q(charter)}",
           f"  Offset = {song.chart_offset_s:g}", f"  Resolution = {res}",
           '  Player2 = bass', '  Difficulty = 0', '  PreviewStart = 0', '  PreviewEnd = 0',
           f"  MusicStream = {_q(music_stream)}", "}", "[SyncTrack]", "{"]
    tm = song.tempo
    sync = []
    for t, n, d in zip(tm.ts_ticks, tm.ts_num, tm.ts_den):
        exp = int(round(np.log2(d)))
        sync.append((int(t), 0, f"TS {int(n)}" + (f" {exp}" if exp != 2 else "")))
    for t, b in zip(tm.ticks, tm.bpm):
        sync.append((int(t), 1, f"B {int(round(b * 1000))}"))
    for t, _, txt in sorted(sync):
        out.append(f"  {t} = {txt}")
    out += ["}", "[Events]", "{"]
    for t, s in song.sections:
        out.append(f'  {t} = E "section {s}"')
    out.append("}")

    threshold = chart_hopo_threshold(res, ini)
    for diff, sec in TRACK_NAMES.items():
        tr = song.tracks.get(diff)
        if tr is None or len(tr) == 0:
            continue
        out += [f"[{sec}]", "{"]
        out += _track_lines(tr, threshold)
        out.append("}")
    return "\n".join(out) + "\n"


def _track_lines(tr: GuitarTrack, threshold: int) -> list[str]:
    nat = natural_hopo(tr.tick, tr.mask, threshold, midi_exception=False)
    events: list[tuple[int, int, str]] = []
    for i in range(len(tr)):
        t, m, sus, nt = int(tr.tick[i]), int(tr.mask[i]), int(tr.sustain[i]), int(tr.ntype[i])
        for f in range(6):
            if m >> f & 1:
                events.append((t, 1, f"N {7 if f == 5 else f} {sus}"))
        if nt == NoteType.TAP:
            events.append((t, 2, "N 6 0"))
        elif (nt == NoteType.HOPO) != bool(nat[i]):
            events.append((t, 2, "N 5 0"))
    for t, length in tr.star_power:
        events.append((int(t), 3, f"S 2 {int(length)}"))
    for s, e in tr.solos:
        events.append((int(s), 0, "E solo"))
        events.append((int(e), 4, "E soloend"))
    events.sort(key=lambda x: (x[0], x[1], x[2]))
    return [f"  {t} = {txt}" for t, _, txt in events]


def write_chart(song: Song, path: str | Path, **kw) -> None:
    Path(path).write_text(write_chart_text(song, **kw), encoding="utf-8")
