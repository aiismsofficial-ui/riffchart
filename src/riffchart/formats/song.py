"""Internal chart representation shared by the .chart and .mid parsers.

Notes are stored per chord event (one row per tick) as numpy arrays:
    tick     int64   position in source ticks
    mask     uint8   bit0..4 = G R Y B O, bit5 = open
    sustain  int64   sustain length in ticks (max over the chord's notes)
    ntype    uint8   NoteType (strum / hopo / tap), resolved with CH rules
    forced   bool    True when the final type differs from the natural type
                     (i.e. a .chart writer must emit N 5 for this chord)
    disjoint bool    chord notes had different sustain lengths in the source

Seconds are derived from the TempoMap on demand (`times()`), so tick data stays
authoritative and lossless.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

DIFFICULTIES = ("expert", "hard", "medium", "easy")
OPEN_BIT = 1 << 5
FRET_NAMES = ("G", "R", "Y", "B", "O", "open")


class NoteType(IntEnum):
    STRUM = 0
    HOPO = 1
    TAP = 2


@dataclass
class TempoMap:
    """Piecewise-constant tempo. `ticks`/`bpm` are parallel arrays sorted by tick, ticks[0] == 0."""

    resolution: int
    ticks: np.ndarray  # int64
    bpm: np.ndarray  # float64
    ts_ticks: np.ndarray = field(default_factory=lambda: np.zeros(1, np.int64))
    ts_num: np.ndarray = field(default_factory=lambda: np.full(1, 4, np.int64))
    ts_den: np.ndarray = field(default_factory=lambda: np.full(1, 4, np.int64))
    # locked tempo markers (Moonscraper's anchors): tick -> the time in seconds the marker is held at. Only
    # editors use them (they keep the marker there when an earlier tempo changes); playback ignores them.
    anchors: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.ticks = np.asarray(self.ticks, np.int64)
        self.bpm = np.asarray(self.bpm, np.float64)
        if len(self.ticks) == 0 or self.ticks[0] != 0:
            # CH assumes 120 BPM until the first tempo event.
            self.ticks = np.concatenate([[0], self.ticks])
            self.bpm = np.concatenate([[120.0], self.bpm if len(self.bpm) else []])
        self.ts_ticks = np.asarray(self.ts_ticks, np.int64)
        self.ts_num = np.asarray(self.ts_num, np.int64)
        self.ts_den = np.asarray(self.ts_den, np.int64)
        if len(self.ts_ticks) == 0 or self.ts_ticks[0] != 0:
            self.ts_ticks = np.concatenate([[0], self.ts_ticks]).astype(np.int64)
            self.ts_num = np.concatenate([[4], self.ts_num]).astype(np.int64)
            self.ts_den = np.concatenate([[4], self.ts_den]).astype(np.int64)
        sec_per_tick = 60.0 / (self.bpm * self.resolution)
        seg = np.diff(self.ticks) * sec_per_tick[:-1]
        self._sec = np.concatenate([[0.0], np.cumsum(seg)])
        self._spt = sec_per_tick

    def tick_to_sec(self, ticks) -> np.ndarray:
        t = np.asarray(ticks, np.float64)
        i = np.clip(np.searchsorted(self.ticks, t, side="right") - 1, 0, len(self.ticks) - 1)
        return self._sec[i] + (t - self.ticks[i]) * self._spt[i]

    def sec_to_tick(self, secs) -> np.ndarray:
        s = np.asarray(secs, np.float64)
        i = np.clip(np.searchsorted(self._sec, s, side="right") - 1, 0, len(self.ticks) - 1)
        return self.ticks[i] + (s - self._sec[i]) / self._spt[i]

    def beat_times(self, end_sec: float) -> np.ndarray:
        """Seconds of every beat (quarter-note grid line) up to end_sec."""
        end_tick = float(self.sec_to_tick(end_sec))
        return self.tick_to_sec(np.arange(0, end_tick + 1, self.resolution))

    def downbeat_times(self, end_sec: float) -> np.ndarray:
        """Seconds of each bar start, honouring time-signature changes (TS den: beat = 4/den quarters)."""
        end_tick = float(self.sec_to_tick(end_sec))
        out = []
        for k, start in enumerate(self.ts_ticks):
            stop = self.ts_ticks[k + 1] if k + 1 < len(self.ts_ticks) else end_tick + 1
            bar = self.resolution * self.ts_num[k] * 4 / self.ts_den[k]
            if bar <= 0:
                continue
            out.append(np.arange(start, stop, bar))
        return self.tick_to_sec(np.concatenate(out)) if out else np.zeros(0)


@dataclass
class GuitarTrack:
    difficulty: str
    tick: np.ndarray
    mask: np.ndarray
    sustain: np.ndarray
    ntype: np.ndarray
    forced: np.ndarray
    disjoint: np.ndarray
    star_power: list[tuple[int, int]] = field(default_factory=list)  # (tick, length)
    solos: list[tuple[int, int]] = field(default_factory=list)  # (start_tick, end_tick)

    def __len__(self) -> int:
        return len(self.tick)

    @staticmethod
    def empty(difficulty: str) -> "GuitarTrack":
        z = np.zeros(0, np.int64)
        return GuitarTrack(difficulty, z, z.astype(np.uint8), z, z.astype(np.uint8), z.astype(bool), z.astype(bool))


@dataclass
class Song:
    resolution: int
    tempo: TempoMap
    tracks: dict[str, GuitarTrack]
    sections: list[tuple[int, str]] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    chart_offset_s: float = 0.0  # [Song] Offset from .chart (seconds), 0 for .mid
    source_format: str = "chart"
    warnings: list[str] = field(default_factory=list)
    lyrics: list[tuple[int, str]] = field(default_factory=list)   # .chart events: phrase_start / lyric x / phrase_end
    instrument: str = "guitar"            # what `tracks` is (guitar, bass, rhythm, keys)
    # the chart's other 5-fret parts {instrument: {difficulty: track}} and, from a .chart, the sections nothing
    # here reads (drums, 6-fret...) as their lines: kept so that writing the chart again never drops them
    parts: dict[str, dict[str, "GuitarTrack"]] = field(default_factory=dict)
    raw_sections: dict[str, list[str]] = field(default_factory=dict)

    def times(self, track: GuitarTrack) -> tuple[np.ndarray, np.ndarray]:
        """(onset seconds, sustain seconds) in chart time (no audio offset applied)."""
        on = self.tempo.tick_to_sec(track.tick)
        off = self.tempo.tick_to_sec(track.tick + track.sustain)
        return on, off - on

    def last_tick(self) -> int:
        ends = [int((t.tick + t.sustain).max()) for t in self.tracks.values() if len(t)]
        return max(ends) if ends else 0


def popcount(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, np.uint8)
    return np.unpackbits(m[:, None], axis=1).sum(1) if len(m) else np.zeros(0, np.int64)


def rescale_parts(song: "Song", resolution: int) -> tuple[dict, dict]:
    """song.parts and song.raw_sections with every tick moved to `resolution` (the same beats)."""
    import copy
    import re
    if resolution == song.resolution:
        return song.parts, song.raw_sections
    f = resolution / song.resolution
    r = lambda x: np.round(np.asarray(x) * f).astype(np.int64)
    parts = copy.deepcopy(song.parts)
    for tracks in parts.values():
        for tr in tracks.values():
            tr.tick, tr.sustain = r(tr.tick), r(tr.sustain)
            tr.star_power = [(int(round(a * f)), int(round(b * f))) for a, b in tr.star_power]
            tr.solos = [(int(round(a * f)), int(round(b * f))) for a, b in tr.solos]
    line = re.compile(r"^(\d+)(\s*=\s*\w+\s+)(\d+)(\s+)(\d+)$")      # "t = N fret len" / "t = S 2 len"
    raw = {}
    for name, lines in song.raw_sections.items():
        out = []
        for ln in lines:
            m = line.match(ln)
            if m:
                out.append(f"{round(int(m.group(1)) * f)}{m.group(2)}{m.group(3)}{m.group(4)}{round(int(m.group(5)) * f)}")
            else:
                head, _, rest = ln.partition("=")
                out.append(f"{round(int(head) * f)} ={rest}" if head.strip().isdigit() else ln)
        raw[name] = out
    return parts, raw
