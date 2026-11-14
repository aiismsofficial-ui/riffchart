"""notes.mid writer: the same song as notes.chart, plus PART VOCALS (pitched vocals, which only .mid can carry;
YARG sings them, Clone Hero shows their lyrics). The inverse of midi_parser.py, using the same conventions:

  tempo track   set_tempo / time_signature
  EVENTS        [section <name>]
  PART GUITAR   difficulty base keys (easy 60, medium 72, hard 84, expert 96) + fret 0..4, base-1 open (with
                [ENHANCED_OPENS]), base+5 / base+6 forced HOPO / strum where the note's type differs from what
                the game works out by itself, tap as Phase Shift SysEx per difficulty (note 104 would tap every
                difficulty at that tick), 116 star power, 103 solo
  PART VOCALS   notes 36-84 (one per word, the word as a lyric event at its tick; "#" = no clear pitch),
                105 = phrase

Round trip: parse_midi(write_midi(song)) gives back the same notes, types, star power and solos (tests).
"""
from __future__ import annotations

from pathlib import Path

import mido
import numpy as np

from .hopo import midi_hopo_threshold, natural_hopo
from .midi_parser import DIFF_BASE, SOLO_NOTE, SP_NOTE
from .song import Song

SYSEX_DIFF = {"easy": 0, "medium": 1, "hard": 2, "expert": 3}
PHRASE_NOTE = 105
VOCAL_LO, VOCAL_HI = 36, 84


def _t(s: str) -> str:
    """MIDI text is bytes (mido writes latin-1): store UTF-8, which YARG and Clone Hero read."""
    return str(s).encode("utf-8").decode("latin-1")


def _track(name: str, events: list[tuple[int, int, mido.Message]]) -> mido.MidiTrack:
    """events: (tick, order, message) -> a track with delta times. At one tick: note offs (order 0) before
    text / sysex (1) before note ons (2), so back-to-back notes on one key never overlap."""
    tr = mido.MidiTrack()
    tr.append(mido.MetaMessage("track_name", name=_t(name), time=0))
    last = 0
    for t, _, msg in sorted(events, key=lambda e: (e[0], e[1])):
        tr.append(msg.copy(time=int(t - last)))
        last = t
    tr.append(mido.MetaMessage("end_of_track", time=0))
    return tr


def _note(events, key: int, start: int, end: int, vel: int = 100):
    events.append((start, 2, mido.Message("note_on", note=key, velocity=vel)))
    events.append((max(end, start + 1), 0, mido.Message("note_off", note=key, velocity=0)))


def _guitar(song: Song, ini: dict | None, tracks: dict | None = None) -> list:
    """One 5-fret part (song.tracks, or `tracks`: another part of the song such as the bass)."""
    ev: list = []
    res = song.resolution
    thr = midi_hopo_threshold(res, ini)
    tracks = song.tracks if tracks is None else tracks
    opens = any(len(tr) and (np.asarray(tr.mask) & 0x20).any() for tr in tracks.values())
    if opens:
        ev.append((0, 1, mido.MetaMessage("text", text="[ENHANCED_OPENS]")))
    first = None
    for diff, tr in tracks.items():
        if diff not in DIFF_BASE or not len(tr):
            continue
        first = first or tr
        base = DIFF_BASE[diff]
        tick = np.asarray(tr.tick, np.int64)
        mask = np.asarray(tr.mask, np.int64)
        sus = np.asarray(tr.sustain, np.int64)
        ntype = np.asarray(tr.ntype, np.int64)
        nat = natural_hopo(tick, mask.astype(np.uint8), thr, True)
        next_on: dict[int, int] = {}                        # key -> next start, to keep notes from overlapping
        for i in range(len(tick) - 1, -1, -1):
            t, m, s = int(tick[i]), int(mask[i]), int(sus[i])
            keys = [base - 1] if m & 0x20 else [base + f for f in range(5) if m >> f & 1]
            for k in keys:
                end = t + (s if s > 0 else 1)
                if k in next_on:
                    end = min(end, next_on[k])
                _note(ev, k, t, end)
                next_on[k] = t
            if ntype[i] == 2:                               # tap: SysEx for this difficulty only
                d = SYSEX_DIFF[diff]
                ev.append((t, 1, mido.Message("sysex", data=(0x50, 0x53, 0, 0, d, 4, 1))))
                ev.append((t + 1, 1, mido.Message("sysex", data=(0x50, 0x53, 0, 0, d, 4, 0))))
            elif ntype[i] == 1 and not nat[i]:
                _note(ev, base + 5, t, t + 1)
            elif ntype[i] == 0 and nat[i]:
                _note(ev, base + 6, t, t + 1)
    src = tracks.get("expert") or first
    if src is not None:
        for t, ln in src.star_power:
            _note(ev, SP_NOTE, int(t), int(t) + max(int(ln), 1))
        for a, b in src.solos:
            _note(ev, SOLO_NOTE, int(a), int(b))
    return ev


def _vocals(notes: list[tuple[int, int, str, int | None]], phrases: list[tuple[int, int]]) -> list:
    """notes: (start, end, word, midi pitch or None); phrases: (start, end) ticks."""
    ev: list = []
    for a, b in phrases:
        _note(ev, PHRASE_NOTE, a, b)
    for a, b, w, p in notes:
        if p is None:
            text, key = w + "#", 60                         # unpitched: the game shows the word, no pitch
        else:
            text, key = w, int(min(max(round(p), VOCAL_LO), VOCAL_HI))
        ev.append((a, 1, mido.MetaMessage("lyrics", text=_t(text))))
        _note(ev, key, a, b)
    return ev


def write_midi(song: Song, path: str | Path, ini: dict | None = None, name: str = "",
               vocals: tuple[list, list] | None = None) -> Path:
    """Write `song` (and vocals = (notes, phrases) from product/vocals.py) as a type-1 notes.mid."""
    if abs(song.chart_offset_s) > 1e-6:
        raise ValueError("this chart starts with an audio offset, which notes.mid can't carry")
    slow = [float(b) for b in song.tempo.bpm if 60_000_000 / float(b) > 0xFFFFFF]
    if slow:                                 # a set_tempo holds at most 16777215 us per beat (3.58 BPM)
        raise ValueError(f"this chart has a {min(slow):g} BPM tempo marker; notes.mid can't go below 3.6 BPM")
    mid = mido.MidiFile(type=1, ticks_per_beat=int(song.resolution))
    tm = song.tempo
    tempo_ev = []
    for t, bpm in zip(tm.ticks, tm.bpm):
        tempo_ev.append((int(t), 1, mido.MetaMessage("set_tempo", tempo=int(round(60_000_000 / float(bpm))))))
    for t, n, d in zip(tm.ts_ticks, tm.ts_num, tm.ts_den):
        tempo_ev.append((int(t), 1, mido.MetaMessage("time_signature", numerator=int(n), denominator=int(d))))
    mid.tracks.append(_track(name or "notes", tempo_ev))
    events = [(int(t), 1, mido.MetaMessage("text", text=_t(f"[section {n.replace('[', '').replace(']', '')}]")))
              for t, n in song.sections]
    mid.tracks.append(_track("EVENTS", events))
    parts = {**song.parts, song.instrument: song.tracks}
    for inst, name in (("guitar", "PART GUITAR"), ("bass", "PART BASS"), ("rhythm", "PART RHYTHM"),
                       ("keys", "PART KEYS")):
        if any(len(t) for t in parts.get(inst, {}).values()) or inst == "guitar":
            mid.tracks.append(_track(name, _guitar(song, ini, parts.get(inst, {}))))
    if vocals and vocals[0]:
        mid.tracks.append(_track("PART VOCALS", _vocals(*vocals)))
    path = Path(path)
    tmp = path.with_suffix(".mid.tmp")
    mid.save(str(tmp))
    tmp.replace(path)
    return path
