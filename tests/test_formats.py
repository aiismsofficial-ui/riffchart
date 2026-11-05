import mido
import numpy as np
import pytest

from riffchart.formats.chart_parser import parse_chart_text
from riffchart.formats.chart_writer import write_chart_text
from riffchart.formats.hopo import chart_hopo_threshold, midi_hopo_threshold
from riffchart.formats.midi_parser import parse_midi
from riffchart.formats.song import OPEN_BIT, NoteType, TempoMap

S, H, T = NoteType.STRUM, NoteType.HOPO, NoteType.TAP


def chart(track_lines: str, sync: str = "0 = TS 4\n0 = B 120000", res: int = 192, offset: float = 0) -> str:
    return (f"[Song]\n{{\n  Resolution = {res}\n  Offset = {offset}\n}}\n"
            f"[SyncTrack]\n{{\n{sync}\n}}\n[Events]\n{{\n  0 = E \"section Intro\"\n}}\n"
            f"[ExpertSingle]\n{{\n{track_lines}\n}}\n")


# ---------------------------------------------------------------- tempo map
def test_tempo_map_conversion_and_inverse():
    tm = TempoMap(192, np.array([0, 768]), np.array([120.0, 60.0]))
    assert tm.tick_to_sec(768) == pytest.approx(2.0)          # 4 beats @120
    assert tm.tick_to_sec(768 + 192) == pytest.approx(3.0)    # +1 beat @60
    ticks = np.array([0, 100, 768, 1000, 5000])
    assert np.allclose(tm.sec_to_tick(tm.tick_to_sec(ticks)), ticks)


def test_tempo_defaults_to_120_before_first_event():
    tm = TempoMap(480, np.array([960]), np.array([60.0]))
    assert tm.tick_to_sec(960) == pytest.approx(1.0)


def test_downbeats_follow_time_signature_changes():
    tm = TempoMap(192, np.array([0]), np.array([60.0]), np.array([0, 768]), np.array([4, 3]), np.array([4, 4]))
    db = tm.downbeat_times(10.0)
    assert np.allclose(db[:4], [0, 4, 7, 10])


# ---------------------------------------------------------------- .chart parsing
def test_chart_basic_fields_and_sections():
    song = parse_chart_text(chart("  0 = N 0 0\n  192 = N 1 96", sync="0 = TS 4\n0 = B 120000\n384 = B 60000", offset=0.25))
    assert song.resolution == 192 and song.chart_offset_s == 0.25
    assert song.sections == [(0, "Intro")]
    tr = song.tracks["expert"]
    on, sus = song.times(tr)
    assert np.allclose(on, [0.0, 0.5]) and np.allclose(sus, [0.0, 0.25])


def test_chart_natural_hopo_threshold_inclusive_65():
    assert chart_hopo_threshold(192) == 65 and chart_hopo_threshold(480) == 162
    tr = parse_chart_text(chart("  0 = N 0 0\n  65 = N 1 0\n  131 = N 2 0")).tracks["expert"]
    assert list(tr.ntype) == [S, H, S]  # 65 ticks -> HOPO, 66 ticks -> strum


def test_chart_same_fret_and_chords_are_strums():
    lines = "  0 = N 0 0\n  48 = N 0 0\n  96 = N 0 0\n  96 = N 1 0\n  144 = N 2 0\n  192 = N 0 0"
    tr = parse_chart_text(chart(lines)).tracks["expert"]
    # repeat -> strum; chord -> strum; single after chord -> HOPO; G after Y -> HOPO
    assert list(tr.mask) == [1, 1, 3, 4, 1]
    assert list(tr.ntype) == [S, S, S, H, H]


def test_chart_single_after_chord_containing_it_is_hopo():
    # .chart has no "subset" exception (only .mid does).
    tr = parse_chart_text(chart("  0 = N 0 0\n  0 = N 1 0\n  48 = N 0 0")).tracks["expert"]
    assert list(tr.ntype) == [S, H]


def test_chart_forced_flag_flips_and_tap_wins():
    lines = ("  0 = N 0 0\n  48 = N 1 0\n  48 = N 5 0\n"      # natural HOPO forced -> strum
             "  480 = N 2 0\n  480 = N 5 0\n"                 # natural strum forced -> HOPO
             "  528 = N 0 0\n  528 = N 2 0\n  528 = N 5 0\n"  # chord forced -> HOPO chord
             "  576 = N 3 0\n  576 = N 6 0\n  576 = N 5 0")   # tap beats forced
    tr = parse_chart_text(chart(lines)).tracks["expert"]
    assert list(tr.ntype) == [S, S, H, H, T]
    assert list(tr.forced) == [False, True, True, True, False]


def test_chart_open_star_power_solo_disjoint():
    lines = ("  0 = N 7 0\n  0 = S 2 768\n  0 = E solo\n  192 = N 0 100\n  192 = N 1 50\n"
             "  384 = N 7 0\n  768 = E soloend")
    tr = parse_chart_text(chart(lines)).tracks["expert"]
    assert tr.mask[0] == OPEN_BIT and tr.mask[2] == OPEN_BIT
    assert tr.star_power == [(0, 768)] and tr.solos == [(0, 768)]
    assert tr.sustain[1] == 100 and tr.disjoint[1]


def test_chart_ini_overrides_hopo_threshold():
    lines = "  0 = N 0 0\n  96 = N 1 0"
    assert parse_chart_text(chart(lines)).tracks["expert"].ntype[1] == S
    tr = parse_chart_text(chart(lines), ini={"eighthnote_hopo": "1"}).tracks["expert"]
    assert chart_hopo_threshold(192, {"eighthnote_hopo": "1"}) == 97 and tr.ntype[1] == H
    assert chart_hopo_threshold(192, {"hopo_frequency": "100"}) == 100


def test_chart_tolerates_unsorted_and_junk():
    tr = parse_chart_text(chart("  96 = N 1 0\n  0 = N 0 0\n  garbage line\n  48 = Q 1 2")).tracks["expert"]
    assert list(tr.tick) == [0, 96]


# ---------------------------------------------------------------- .mid parsing
def _midi(tmp_path, events, res=480, track_events_extra=(), name="PART GUITAR"):
    mf = mido.MidiFile(ticks_per_beat=res)
    t0 = mido.MidiTrack([mido.MetaMessage("set_tempo", tempo=500000, time=0),
                         mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0)])
    mf.tracks.append(t0)
    abs_ev = []
    for start, note, length in events:
        abs_ev.append((start, 1, mido.Message("note_on", note=note, velocity=100)))
        abs_ev.append((start + length, 0, mido.Message("note_off", note=note, velocity=0)))
    abs_ev += list(track_events_extra)
    abs_ev.sort(key=lambda x: (x[0], x[1]))
    tr = mido.MidiTrack([mido.MetaMessage("track_name", name=name, time=0)])
    last = 0
    for t, _, msg in abs_ev:
        tr.append(msg.copy(time=t - last))
        last = t
    mf.tracks.append(tr)
    p = tmp_path / "notes.mid"
    mf.save(p)
    return p


def test_midi_notes_hopo_and_sustain_cutoff(tmp_path):
    assert midi_hopo_threshold(480) == 161
    ev = [(0, 96, 120), (160, 97, 480), (480, 98, 60), (960, 98, 10)]
    song = parse_midi(_midi(tmp_path, ev))
    tr = song.tracks["expert"]
    assert list(tr.ntype) == [S, H, S, S]
    assert list(tr.sustain) == [0, 480, 0, 0]  # < res/3 = 160 cut to 0
    assert song.tempo.tick_to_sec(480) == pytest.approx(0.5)


def test_midi_chord_subset_exception(tmp_path):
    ev = [(0, 96, 1), (0, 97, 1), (100, 96, 1), (200, 98, 1)]
    tr = parse_midi(_midi(tmp_path, ev)).tracks["expert"]
    assert list(tr.ntype) == [S, S, H]  # G after G+R chord -> strum in .mid


def test_midi_forcing_markers_and_tap(tmp_path):
    ev = [(0, 96, 1), (100, 97, 1), (100, 102, 1),   # natural HOPO forced strum
          (960, 98, 1), (960, 101, 1),               # natural strum forced HOPO
          (1920, 96, 1), (1920, 101, 1), (1920, 102, 1),  # both markers -> HOPO wins
          (2880, 99, 1), (2880, 104, 1)]             # tap
    tr = parse_midi(_midi(tmp_path, ev)).tracks["expert"]
    assert list(tr.ntype) == [S, S, H, H, T]


def test_midi_marker_excludes_last_tick(tmp_path):
    ev = [(0, 96, 1), (960, 97, 1), (960, 101, 960), (1920, 98, 1)]
    tr = parse_midi(_midi(tmp_path, ev)).tracks["expert"]
    assert list(tr.ntype) == [S, H, S]


def test_midi_enhanced_opens_and_sysex_opens(tmp_path):
    enh = [(0, 2, mido.MetaMessage("text", text="[ENHANCED_OPENS]"))]
    tr = parse_midi(_midi(tmp_path, [(0, 95, 1), (480, 96, 1)], track_events_extra=enh)).tracks["expert"]
    assert list(tr.mask) == [OPEN_BIT, 1]

    sx = [(0, 2, mido.Message("sysex", data=[0x50, 0x53, 0, 0, 3, 1, 1])),
          (10, 2, mido.Message("sysex", data=[0x50, 0x53, 0, 0, 3, 1, 0])),
          (480, 2, mido.Message("sysex", data=[0x50, 0x53, 0, 0, 0x7F, 4, 1])),
          (500, 2, mido.Message("sysex", data=[0x50, 0x53, 0, 0, 0x7F, 4, 0]))]
    tr = parse_midi(_midi(tmp_path, [(0, 96, 1), (480, 97, 1), (960, 98, 1)], track_events_extra=sx)).tracks["expert"]
    assert list(tr.mask) == [OPEN_BIT, 2, 4]
    assert list(tr.ntype) == [S, T, S]


def _vlq(n):
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.insert(0, (n & 0x7F) | 0x80)
        n >>= 7
    return bytes(out)


def test_midi_raw_sysex_with_ff_difficulty(tmp_path):
    # Real charts put 0xFF in the SysEx difficulty byte, which mido cannot write; build raw bytes.
    def trk(evs):
        body = b"".join(_vlq(dt) + e for dt, e in evs) + bytes([0, 0xFF, 0x2F, 0])
        return b"MTrk" + len(body).to_bytes(4, "big") + body

    t0 = trk([(0, bytes([0xFF, 0x51, 3, 0x07, 0xA1, 0x20]))])
    name = b"PART GUITAR"

    def sx(v):
        return bytes([0xF0]) + _vlq(8) + bytes([0x50, 0x53, 0, 0, 0xFF, 1, v, 0xF7])

    t1 = trk([(0, bytes([0xFF, 0x03]) + _vlq(len(name)) + name), (0, sx(1)), (0, bytes([0x90, 97, 100])),
              (10, sx(0)), (0, bytes([0x80, 97, 0])), (470, bytes([0x90, 98, 100])), (10, bytes([0x80, 98, 0]))])
    p = tmp_path / "raw.mid"
    p.write_bytes(b"MThd" + (6).to_bytes(4, "big") + (1).to_bytes(2, "big") + (2).to_bytes(2, "big")
                  + (480).to_bytes(2, "big") + t0 + t1)
    tr = parse_midi(p).tracks["expert"]
    assert list(tr.mask) == [OPEN_BIT, 4]


def test_midi_star_power_solo_and_difficulties(tmp_path):
    ev = [(0, 96, 1), (0, 84, 1), (0, 72, 1), (0, 60, 1), (0, 116, 1920), (0, 103, 960)]
    song = parse_midi(_midi(tmp_path, ev))
    assert set(song.tracks) == {"expert", "hard", "medium", "easy"}
    assert song.tracks["expert"].star_power == [(0, 1920)] and song.tracks["easy"].solos == [(0, 960)]


def test_midi_without_guitar_track_raises(tmp_path):
    with pytest.raises(ValueError):
        parse_midi(_midi(tmp_path, [(0, 96, 1)], name="PART DRUMS"))


# ---------------------------------------------------------------- writer round trip
def _same(a, b):
    for d in a.tracks:
        x, y = a.tracks[d], b.tracks[d]
        for f in ("tick", "mask", "sustain", "ntype"):
            assert np.array_equal(getattr(x, f), getattr(y, f)), (d, f)
        assert x.star_power == y.star_power and x.solos == y.solos


def test_chart_write_parse_round_trip():
    lines = ("  0 = N 0 0\n  0 = S 2 400\n  48 = N 1 0\n  48 = N 5 0\n  96 = N 7 200\n  300 = N 2 0\n"
             "  300 = N 3 0\n  300 = N 5 0\n  400 = N 4 0\n  400 = N 6 0\n  500 = E solo\n  600 = N 0 0\n  700 = E soloend")
    a = parse_chart_text(chart(lines, sync="0 = TS 4\n0 = B 120000\n384 = TS 7 3\n384 = B 133337", offset=-0.1))
    b = parse_chart_text(write_chart_text(a))
    _same(a, b)
    assert np.array_equal(a.tempo.bpm, b.tempo.bpm) and np.array_equal(a.tempo.ts_den, b.tempo.ts_den)
    assert b.chart_offset_s == pytest.approx(-0.1) and b.sections == a.sections


def test_midi_to_chart_preserves_note_types(tmp_path):
    # G+R chord then G at 100 ticks: strum in .mid (subset rule) but natural HOPO in .chart,
    # so the writer must emit N 5 to keep it a strum.
    ev = [(0, 96, 1), (0, 97, 1), (100, 96, 1), (200, 98, 1), (1000, 99, 1), (1000, 101, 1)]
    a = parse_midi(_midi(tmp_path, ev))
    b = parse_chart_text(write_chart_text(a))
    _same(a, b)


# ---------------------------------------------------------------- beats -> tempo map
def test_beats_to_tempo_map_hits_every_beat_and_bar():
    from riffchart.timing.tempo_map import beats_to_tempo_map

    rng = np.random.default_rng(0)
    beats = 0.73 + np.cumsum(0.5 + rng.normal(0, 0.01, 40))
    downbeats = beats[1::4]  # pickup of one beat, 4/4 after
    tm = beats_to_tempo_map(beats, downbeats, 480)
    assert np.allclose(tm.tick_to_sec(tm.sec_to_tick(beats).round()), beats, atol=1e-3)
    t = tm.sec_to_tick(beats)
    assert np.abs(t - 480 * np.round(t / 480)).max() < 0.5  # every beat on a beat tick
    db = tm.downbeat_times(beats[-1])
    assert np.allclose(db[np.searchsorted(db, downbeats[0] - 1e-3):][:5], downbeats[:5], atol=1e-3)


def test_smooth_tempo_map_steady_and_ts_lattice():
    from riffchart.timing.tempo_map import beats_to_tempo_map

    beats = 1.0 + 0.5 * np.arange(64)                 # steady 120 BPM
    downbeats = beats[::4].copy()
    downbeats = np.delete(downbeats, 5)               # missed downbeat -> must not shift bars
    downbeats[3] = beats[13]                          # one stray downbeat (next is back on lattice)
    tm = beats_to_tempo_map(beats, downbeats, 192, mode="smooth", ts_mode="lattice")
    assert len(tm.bpm) <= 2 and abs(tm.bpm[-1] - 120.0) < 1e-6
    db = tm.downbeat_times(beats[-1])
    assert np.allclose(db[db >= 1.0 - 1e-6][:16], beats[::4][:16], atol=1e-3)
    # a genuine switch to 3/4 at beat 32 re-anchors the lattice
    d2 = np.concatenate([beats[0:32:4], beats[32::3]])
    tm = beats_to_tempo_map(beats, d2, 192, mode="smooth", ts_mode="lattice")
    db = tm.downbeat_times(beats[-1])
    assert np.allclose(db[db >= 1.0 - 1e-6][:14], d2[:14], atol=1e-3)
    assert 3 in list(tm.ts_num)
