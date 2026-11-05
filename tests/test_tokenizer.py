import numpy as np

from riffchart.formats.song import OPEN_BIT, NoteType
from riffchart.model.grid_events import beat_position, beat_time, downbeat_flags, notes_to_units
from riffchart.model.tokenizer import (SUS_LENGTHS, Event, V, decode, encode_beats, encode_window,
                                       resolve_sustains, style_tokens)
from riffchart.timing.grid import GRID

S, H, T = NoteType.STRUM, NoteType.HOPO, NoteType.TAP


def test_vocab_layout_is_contiguous():
    assert V.OPEN - V.NOTE == 31 and V.SRC == V.SUS + len(SUS_LENGTHS) and V.SIZE == V.SRC + 3
    assert V.HOPO - V.POS == 24


def test_round_trip_events_tokens():
    evs = [Event(0, 1, S, 0), Event(12, 3, H, 24), Event(48 + 16, OPEN_BIT, S, 0),
           Event(48 + 32, 16, T, -1), Event(3 * 48, 31, S, 96), Event(3 * 48 + 42, 2, H, 0)]
    down = np.array([True, False, False, False, True, False])
    ids, bidx, pos = encode_beats(evs, 6, down)
    assert ids[0] == V.BAR and ids.count(V.BEAT) + ids.count(V.BAR) == 6
    back = decode(ids)
    assert [(e.unit, e.mask, e.ntype) for e in back] == [(e.unit, e.mask, e.ntype) for e in evs]
    assert [e.sus for e in back] == [0, 24, 0, -1, 96, 0]
    # beat index / position side channels
    i = ids.index(V.TAP)
    assert bidx[i] == 1 and pos[i] == 32


def test_sus_next_detection_and_resolution():
    evs = [Event(0, 1, S, 24), Event(24, 2, S, 30), Event(60, 4, S, 0)]
    ids, _, _ = encode_beats(evs, 2, np.zeros(2, bool))
    assert ids.count(V.SUS_NEXT) == 1                       # first sustain ends on the next onset
    back = resolve_sustains(decode(ids))
    assert back[0].sus == 24 and back[1].sus == 30


def test_window_context_target_split():
    evs = [Event(b * GRID, 1 << (b % 5), S, 0) for b in range(8)]
    down = np.array([b % 4 == 0 for b in range(8)])
    ids, bidx, pos, loss = encode_window(evs, down, 3, 5, "hard", style_tokens(8, 8, 0.0))
    sep = list(ids).index(V.SEP)
    assert not loss[: sep + 1].any() and loss[sep + 1 :].all()
    assert ids[1] == V.DIFF + 1 and ids[-1] == V.EOS
    tgt = decode(ids[sep + 1 :], beat0=3)
    assert [e.unit for e in tgt] == [b * GRID for b in range(3, 8)]


def test_grid_snap_and_merge():
    beats = 1.0 + 0.5 * np.arange(20)
    assert np.allclose(beat_time(beat_position([1.1, 3.3], beats) * GRID, beats), [1.1, 3.3])
    on = np.array([1.0, 1.125, 1.126, 1.5 + 0.5 / 3])       # beat 0, 16th (x2 -> merged), 8th-triplet
    end = np.array([1.0, 1.25, 1.125, 2.5])
    u, m, t, s = notes_to_units(on, end, [1, 2, 4, 8], [S, H, S, S], beats)
    assert list(u) == [0, 12, 48 + 16] and list(m) == [1, 6, 8]
    assert s[1] == 12 and s[2] == 96 - 16
    flags = downbeat_flags(beats, beats[::4] + 0.01)
    assert flags[::4].all() and flags.sum() == 5
