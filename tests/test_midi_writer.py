"""notes.mid writer (round trip through the YARG-rules parser) and pitched vocals (product/vocals.py)."""
import json

import mido
import numpy as np
import pytest

from riffchart.formats.chart_parser import parse_chart_text
from riffchart.formats.loader import load_song
from riffchart.formats.midi_parser import parse_midi
from riffchart.formats.midi_writer import write_midi
from riffchart.product import vocals

CHART = """[Song]
{
  Resolution = 192
  Offset = 0
}
[SyncTrack]
{
  0 = TS 4
  0 = B 120000
  768 = B 150000
  1536 = TS 7 3
}
[Events]
{
  0 = E "section Intro"
  768 = E "section Verse ü"
  768 = E "phrase_start"
  768 = E "lyric Hello"
  960 = E "lyric world"
  1152 = E "phrase_end"
  1344 = E "phrase_start"
  1344 = E "lyric again"
  1500 = E "phrase_end"
}
[ExpertSingle]
{
  0 = N 0 0
  48 = N 1 0
  96 = N 1 0
  96 = N 5 0
  192 = N 2 200
  480 = N 0 0
  480 = N 1 0
  540 = N 7 0
  600 = N 3 0
  600 = N 6 0
  700 = N 4 0
  760 = N 0 0
  760 = N 5 0
  0 = S 2 400
  1000 = N 2 0
}
[HardSingle]
{
  0 = N 0 0
  600 = N 3 0
  700 = N 7 0
}
"""


def _song():
    return parse_chart_text(CHART, {})


def test_round_trip_keeps_notes_types_and_phrases(tmp_path):
    a = _song()
    write_midi(a, tmp_path / "notes.mid")
    b = parse_midi(tmp_path / "notes.mid")
    assert set(b.tracks) == set(a.tracks)
    for d, x in a.tracks.items():
        y = b.tracks[d]
        assert list(x.tick) == list(y.tick) and list(x.mask) == list(y.mask), d
        assert list(x.ntype) == list(y.ntype), d                      # forced HOPO / strum / tap / open
        assert list(np.where(x.sustain >= 64, x.sustain, 0)) == list(y.sustain)
    assert b.tracks["expert"].star_power == a.tracks["expert"].star_power
    assert b.sections == a.sections                                   # UTF-8 names survive
    assert list(b.tempo.ts_num) == [4, 7] and list(b.tempo.ts_den) == [4, 8]
    assert np.allclose(b.tempo.bpm, a.tempo.bpm, atol=1e-3)


def test_vocal_notes_from_lyrics():
    s = _song()
    notes, phrases = vocals.vocal_notes(s, None)
    assert phrases == [(768, 1152), (1344, 1500)]
    assert [(a, b, w) for a, b, w, _ in notes] == [(768, 960, "Hello"), (960, 1152, "world"), (1344, 1500, "again")]
    sec = s.tempo.tick_to_sec(np.array([768]))[0]
    fs = 0.01
    n = 500
    pitch = {"midi": np.full(n, 64.0, np.float32), "periodicity": np.ones(n, np.float32), "frame_s": np.float32(fs)}
    pitch["periodicity"][int((sec + 0.6) / fs):] = 0.0                # only the first word is sung clearly
    notes, _ = vocals.vocal_notes(s, pitch)
    assert notes[0][3] == pytest.approx(64.0) and notes[2][3] is None


def test_sync_mid_follows_chart_and_loader_reads_the_chart(tmp_path):
    f = tmp_path / "song"
    (f / ".riffchart").mkdir(parents=True)
    (f / "notes.chart").write_text(CHART, encoding="utf-8")
    (f / "song.ini").write_text("[song]\nname = Tést\n", encoding="utf-8")
    assert vocals.sync_mid(f) == "" and not (f / "notes.mid").exists()  # vocals off: nothing written
    np.savez_compressed(f / vocals.PITCH, midi=np.full(800, 60.0, np.float32),
                        periodicity=np.ones(800, np.float32), frame_s=np.float32(0.01))
    assert vocals.sync_mid(f) == "written"
    m = mido.MidiFile(str(f / "notes.mid"))
    names = [next(x.name for x in t if x.type == "track_name") for t in m.tracks]
    assert names[1:] == ["EVENTS", "PART GUITAR", "PART VOCALS"]
    voc = m.tracks[3]
    assert [x.text for x in voc if x.type == "lyrics"] == ["Hello", "world", "again"]
    assert sorted({x.note for x in voc if x.type == "note_on"}) == [60, 105]
    assert vocals.summary(f) == {"notes": 3, "pitched": 3}
    song, _, path = load_song(f)
    assert path.name == "notes.chart" and song.lyrics                 # the chart stays the source
    vocals.disable(f)
    assert not (f / "notes.mid").exists() and not (f / vocals.MARK).exists()
    assert load_song(f)[2].name == "notes.chart"


def test_offset_charts_are_refused(tmp_path):
    s = _song()
    s.chart_offset_s = 0.5
    with pytest.raises(ValueError):
        write_midi(s, tmp_path / "x.mid")
