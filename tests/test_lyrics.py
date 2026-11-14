"""Lyrics: parsing LRCLIB records, fitting them to a vocal track, and .chart lyric events (no network)."""
import numpy as np

from riffchart.formats.chart_parser import parse_chart_text
from riffchart.formats.chart_writer import write_chart_text
from riffchart.formats.song import GuitarTrack, Song, TempoMap
from riffchart.product import lyrics as L

WORD_FILE = """version: '1.0'
lines:
- text: Hello there world
  start_ms: 2000
  end_ms: 3500
  words:
  - text: 'Hello '
    start_ms: 2000
  - text: 'there '
    start_ms: 2500
  - text: world
    start_ms: 3000
"""


def _tempo(bpm=120.0, res=192):
    return TempoMap(res, np.array([0]), np.array([bpm]), np.array([0]), np.array([4]), np.array([4]))


def test_parse_word_timed():
    lines = L.parse({"lyricsfile": WORD_FILE})
    assert len(lines) == 1
    assert [w for _, w in lines[0].words] == ["Hello", "there", "world"]
    assert lines[0].words[1][0] == 2.5 and lines[0].end == 3.5


def test_parse_line_timed_spreads_words():
    lines = L.parse({"syncedLyrics": "[00:01.00] one two three\n[00:04.00] four\n[00:08.00] "})
    assert [len(ln.words) for ln in lines] == [3, 1]
    t = [t for t, _ in lines[0].words]
    assert t[0] == 1.0 and t[0] < t[1] < t[2] < lines[0].end <= 4.0


def test_fit_shift_finds_the_offset():
    sr, rng = 8000, np.random.default_rng(0)
    starts = np.sort(rng.uniform(5, 55, 40))
    vocals = np.zeros(sr * 60, np.float32)
    for s in starts + 3.0:                                # the singing is 3 s later than the database says
        i = int(s * sr)
        vocals[i: i + int(0.2 * sr)] = rng.normal(0, 0.5, int(0.2 * sr))
    lines = [L.Line(float(s), float(s) + 0.3, [(float(s), "la")]) for s in starts]
    shift, z = L.fit_shift(lines, vocals, sr)
    assert abs(shift - 3.0) < 0.05 and z > 3


def test_events_round_trip_through_chart():
    tm = _tempo()
    lines = [L.Line(1.0, 2.0, [(1.0, "Don't"), (1.5, 'say "no"#')])]
    ev = L.to_events(lines, tm)
    assert [e for _, e in ev] == ["phrase_start", "lyric Don't", "lyric say no", "phrase_end"]
    assert [t for t, _ in ev] == sorted(t for t, _ in ev)
    tr = GuitarTrack.empty("expert")
    song = Song(192, tm, {"expert": tr}, [(0, "Intro")], lyrics=ev)
    back = parse_chart_text(write_chart_text(song))
    assert back.lyrics == ev and back.sections == [(0, "Intro")]


def test_events_follow_the_audio_offset():
    tm = _tempo()                                         # 120 bpm, 192 ticks a beat: 384 ticks a second
    ev = L.to_events([L.Line(2.0, 2.5, [(2.0, "hey")])], tm, offset_s=0.5)
    assert dict((e, t) for t, e in ev)["lyric hey"] == 576   # audio 2.0 s = chart 1.5 s


def test_attach_reads_saved_lyrics(tmp_path):
    L.save(tmp_path, [L.Line(1.0, 1.5, [(1.0, "yo")])], {"source": "test"})
    song = Song(192, _tempo(), {})
    assert L.attach(song, tmp_path) == 1
    assert [e for _, e in song.lyrics] == ["phrase_start", "lyric yo", "phrase_end"]


def test_transcribe_fallback_when_nothing_online(tmp_path, monkeypatch):
    import numpy as np
    from riffchart.product import lyrics as lyr
    monkeypatch.setattr(lyr, "fetch", lambda *a, **k: None)
    monkeypatch.setattr(lyr, "can_transcribe", lambda: True)
    words = [(1.0 + i * 0.5, f"w{i}") for i in range(30)]
    monkeypatch.setattr(lyr, "transcribe", lambda v, sr, device=None: [lyr.Line(1.0, 16.0, words)])
    loud = np.ones(44100 * 20, np.float32) * 0.1
    rep = lyr.make(tmp_path, "", "", loud, 44100, 20.0)               # no names: straight to transcribing
    assert rep["found"] and rep["source"] == "whisper" and rep["words"] == 30
    assert lyr.summary(tmp_path)["source"] == "whisper"
    lyr.remove(tmp_path)
    quiet = loud * 0.01                                                # an instrumental's leftover vocal track
    assert lyr.make(tmp_path, "", "", quiet, 44100, 20.0, mix=loud)["found"] is False
    monkeypatch.setattr(lyr, "transcribe", lambda v, sr, device=None: [lyr.Line(1.0, 2.0, words[:3])])
    assert lyr.make(tmp_path, "", "", loud, 44100, 20.0)["found"] is False   # a few stray words: no lyrics
    assert lyr.make(tmp_path, "a", "b", loud, 44100, 20.0, local=False)["found"] is False
