"""Beat-grid chart tokenizer (Phase 4).

Window sequence:
    BOS DIFF DENS CHORD  <context beats>  SEP  <target beats>  EOS
Every beat starts with BAR (downbeat) or BEAT, then one group per chord, ordered by position:
    POS_p  [HOPO|TAP]  NOTE_m  [SUS_NEXT | SUS_k]
  POS_p     p in MUSICAL_POSITIONS (24 positions of the 48 grid). Which positions targets are
            snapped to (and decoding may emit) is a config choice: `data.positions` musical24 | model16
  NOTE_m    31 fret combinations (bitmask 1..31, G=bit0) + OPEN
  HOPO/TAP  final note type (strum is implicit); the chart writer derives forced flags
  SUS_NEXT  sustain held until the next chord (42% of human sustains end exactly there)
  SUS_k     sustain length in 48ths from SUS_LENGTHS (nearest in log space)
DENS / CHORD are style controls (notes per beat, chord share) with an AUTO bin used for
conditioning dropout and as the inference default. Models trained with `model_source` add a SRC token
after CHORD (main library / extra library / AUTO); SRC ids sit at the end of the vocabulary
so checkpoints trained without them keep their token ids.

Besides token ids the encoder returns, per token, the beat index within the window and the
48th-position, which the model uses to align decoder steps with the audio.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..formats.song import OPEN_BIT, DIFFICULTIES, NoteType
from ..timing.grid import GRID, MODEL_POSITIONS, MUSICAL_POSITIONS

SUS_LENGTHS = np.array([6, 8, 12, 16, 18, 24, 28, 30, 32, 36, 40, 42, 48, 56, 60, 64, 72, 84, 88, 96, 120, 144,
                        168, 192, 240, 288, 384, 480, 576, 768, 960, 1152])
DENS_EDGES = np.array([0.75, 1.25, 1.75, 2.5, 3.5])        # chords per beat -> 6 bins
CHORD_EDGES = np.array([0.1, 0.25, 0.4, 0.55])            # chord share -> 5 bins


class V:
    PAD, BOS, EOS, BEAT, BAR, SEP = 0, 1, 2, 3, 4, 5
    DIFF = 6                                  # 4
    DENS = DIFF + 4                           # 6 bins + AUTO
    N_DENS = len(DENS_EDGES) + 2
    CHORD = DENS + N_DENS                     # 5 bins + AUTO
    N_CHORD = len(CHORD_EDGES) + 2
    POS = CHORD + N_CHORD                     # 24
    HOPO = POS + len(MUSICAL_POSITIONS)
    TAP = HOPO + 1
    NOTE = TAP + 1                            # 31 combos + open
    OPEN = NOTE + 31
    SUS_NEXT = OPEN + 1
    SUS = SUS_NEXT + 1
    SRC = SUS + len(SUS_LENGTHS)              # main, extra, AUTO
    SIZE = SRC + 3


SOURCES = ("main", "extra", "auto")       # which training library a chart came from


def source_token(source: str) -> int:
    return V.SRC + SOURCES.index(source)


POS_INDEX = {int(p): i for i, p in enumerate(MUSICAL_POSITIONS)}


@dataclass
class Event:
    """One chord on the grid. sus = 0 none, -1 until next chord, >0 length in 48ths."""
    unit: int          # beat * 48 + position (window-relative)
    mask: int
    ntype: int
    sus: int = 0


def style_tokens(n_events: int, n_beats: int, chord_share: float, drop: bool = False) -> list[int]:
    if drop or n_beats <= 0:
        return [V.DENS + V.N_DENS - 1, V.CHORD + V.N_CHORD - 1]
    d = int(np.searchsorted(DENS_EDGES, n_events / n_beats, side="right"))
    c = int(np.searchsorted(CHORD_EDGES, chord_share, side="right"))
    return [V.DENS + d, V.CHORD + c]


def sus_token(ev: Event, next_unit: int | None) -> int | None:
    if ev.sus == 0:
        return None
    if ev.sus == -1 or (next_unit is not None and abs(ev.unit + ev.sus - next_unit) <= 1):
        return V.SUS_NEXT
    i = int(np.argmin(np.abs(np.log(SUS_LENGTHS) - np.log(max(ev.sus, 1)))))
    return V.SUS + i


def note_token(mask: int) -> int:
    if mask & OPEN_BIT:
        return V.OPEN
    return V.NOTE + (mask & 0x1F) - 1


def encode_beats(events: list[Event], n_beats: int, downbeat: np.ndarray, beat0: int = 0):
    """Events (units relative to window start) for beats [beat0, beat0 + n_beats) -> (ids, beat_idx, pos)."""
    ids, bidx, pos = [], [], []
    ev = sorted((e for e in events if beat0 * GRID <= e.unit < (beat0 + n_beats) * GRID), key=lambda e: e.unit)
    units = [e.unit for e in events]
    later = sorted(units)
    k = 0
    for b in range(beat0, beat0 + n_beats):
        ids.append(V.BAR if downbeat[b] else V.BEAT); bidx.append(b); pos.append(0)
        while k < len(ev) and ev[k].unit < (b + 1) * GRID:
            e = ev[k]
            p = e.unit - b * GRID
            ids.append(V.POS + POS_INDEX[p]); bidx.append(b); pos.append(p)
            if e.ntype == NoteType.HOPO:
                ids.append(V.HOPO); bidx.append(b); pos.append(p)
            elif e.ntype == NoteType.TAP:
                ids.append(V.TAP); bidx.append(b); pos.append(p)
            ids.append(note_token(e.mask)); bidx.append(b); pos.append(p)
            j = np.searchsorted(later, e.unit, side="right")
            st = sus_token(e, later[j] if j < len(later) else None)
            if st is not None:
                ids.append(st); bidx.append(b); pos.append(p)
            k += 1
    return ids, bidx, pos


def encode_window(events: list[Event], downbeat: np.ndarray, n_ctx: int, n_tgt: int, difficulty: str,
                  style: list[int]):
    """events: window-relative units, beats 0..n_ctx+n_tgt-1. Returns ids, beat_idx, pos, loss_mask."""
    head = [V.BOS, V.DIFF + DIFFICULTIES.index(difficulty)] + style
    c_ids, c_b, c_p = encode_beats(events, n_ctx, downbeat, 0)
    t_ids, t_b, t_p = encode_beats(events, n_tgt, downbeat, n_ctx)
    ids = head + c_ids + [V.SEP] + t_ids + [V.EOS]
    bidx = [0] * len(head) + c_b + [n_ctx] + t_b + [n_ctx + n_tgt - 1]
    pos = [0] * len(head) + c_p + [0] + t_p + [0]
    loss = [0] * (len(head) + len(c_ids) + 1) + [1] * (len(t_ids) + 1)
    return np.array(ids), np.array(bidx), np.array(pos), np.array(loss, bool)


def decode(ids, beat0: int = 0) -> list[Event]:
    """Token ids (any slice that starts at a beat marker, prefix tokens ignored) -> events.
    Beat counting starts at beat0 - 1 so the first BEAT/BAR opens beat beat0."""
    out: list[Event] = []
    b, p, t = beat0 - 1, 0, NoteType.STRUM
    for tok in ids:
        tok = int(tok)
        if tok in (V.BEAT, V.BAR):
            b += 1; p = 0; t = NoteType.STRUM
        elif tok == V.EOS:
            break
        elif V.POS <= tok < V.HOPO:
            p = int(MUSICAL_POSITIONS[tok - V.POS]); t = NoteType.STRUM
        elif tok == V.HOPO:
            t = NoteType.HOPO
        elif tok == V.TAP:
            t = NoteType.TAP
        elif V.NOTE <= tok <= V.OPEN:
            mask = OPEN_BIT if tok == V.OPEN else tok - V.NOTE + 1
            out.append(Event(max(b, 0) * GRID + p, mask, t, 0)); t = NoteType.STRUM
        elif tok == V.SUS_NEXT and out:
            out[-1].sus = -1
        elif V.SUS <= tok < V.SRC and out:
            out[-1].sus = int(SUS_LENGTHS[tok - V.SUS])
    return out


def resolve_sustains(events: list[Event], gap_units: int = 0) -> list[Event]:
    """Replace SUS_NEXT (-1) by the distance to the next chord minus gap_units."""
    events = sorted(events, key=lambda e: e.unit)
    for i, e in enumerate(events):
        if e.sus == -1:
            nxt = events[i + 1].unit if i + 1 < len(events) else e.unit + GRID
            e.sus = max(nxt - e.unit - gap_units, 0)
    return events
