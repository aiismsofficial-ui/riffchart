"""Fixes from the pre-1.1 review: chart round trips, lyric marks, name matching, Bring up to date guards."""
from pathlib import Path

import numpy as np
import pytest

from riffchart.formats.chart_parser import parse_chart_text
from riffchart.formats.chart_writer import write_chart_text
from riffchart.product import chartfile, lyrics, update

CHART = """[Song]
{
  Resolution = 192
}
[SyncTrack]
{
  0 = TS 4
  0 = B 120000
}
[Events]
{
  0 = E "phrase_start"
  0 = E "lyric one"
  384 = E "phrase_end"
  384 = E "phrase_start"
  384 = E "lyric two"
  768 = E "phrase_end"
}
[ExpertSingle]
{
  0 = N 0 0
  192 = N 1 0
}
[HardDoubleBass]
{
  0 = S 2 192
}
"""


def test_phrases_meeting_at_one_tick_round_trip():
    song = parse_chart_text(CHART)
    again = parse_chart_text(write_chart_text(song))
    assert again.lyrics == song.lyrics
    lines = lyrics.from_events(again.lyrics, again.tempo)
    assert len(lines) == 2 and lines[0].end == pytest.approx(1.0) and lines[1].words[0][1] == "two"


def test_part_section_without_notes_is_kept_once():
    song = parse_chart_text(CHART)
    text = write_chart_text(song)
    assert text.count("[HardDoubleBass]") == 1 and "0 = S 2 192" in text


def test_lyric_marks_survive_an_edit():
    assert lyrics._word("Hel=", syllables=True) == "Hel="
    assert lyrics._word("lo#", syllables=True) == "lo#"
    assert lyrics._word("Hel-", syllables=True) == "Hel-"
    assert lyrics._word("double-four") == "double=four"
    assert lyrics._word("Hel-") == "Hel" and lyrics._word("-") == ""
    lines = lyrics.from_events([(0, "phrase_start"), (0, "lyric dou=ble"), (10, "lyric Hel="), (20, "phrase_end")],
                               parse_chart_text(CHART).tempo)
    assert [w for _, w in lines[0].words] == ["dou-ble", "Hel="]


def test_name_matching_any_script():
    assert lyrics._same("Måneskin", "Maneskin") and lyrics._same("AC/DC", "ac dc")
    assert lyrics._same("Кино", "кино") and not lyrics._same("Кино", "Ария")
    assert not lyrics._same("!!!", "???")                # nothing left to compare: never a match


def test_same_timing():
    a = parse_chart_text(CHART)
    b = parse_chart_text(CHART.replace("B 120000", "B 140000"))
    assert chartfile.same_timing(a, parse_chart_text(CHART)) and not chartfile.same_timing(a, b)


def test_bring_up_to_date_leaves_existing_lyrics_and_other_charts(tmp_path):
    song = tmp_path / "song"
    song.mkdir()
    (song / "song.ini").write_text("[song]\nname = X\nartist = Y\ncharter = RiffChart\n", encoding="utf-8")
    (song / "notes.chart").write_text(CHART.replace('  0 = E "lyric one"\n', ""), encoding="utf-8")
    assert update.chart_has_lyrics(song)                  # "lyric two" is still there
    assert "lyrics" not in update.needs(song, split=False, lyrics_online=True, webm=False)
    no_lyrics = "\n".join(ln for ln in CHART.splitlines() if "lyric" not in ln and "phrase" not in ln)
    (song / "notes.chart").write_text(no_lyrics, encoding="utf-8")
    assert "lyrics" in update.needs(song, split=False, lyrics_online=True, webm=False)
    (song / "song.ini").write_text("[song]\nname = X\nartist = Y\ncharter = Someone\n", encoding="utf-8")
    assert "lyrics" not in update.needs(song, split=False, lyrics_online=True, webm=False)


def test_write_atomic_leaves_no_temp(tmp_path):
    p = tmp_path / "notes.chart"
    chartfile.write_atomic(p, "x")
    assert p.read_text() == "x" and not list(tmp_path.glob("*.tmp"))


def test_chart_file_never_rebinds_its_stage_timer():
    """chart_file times its stages with `t` (done() reads it): a loop variable named t broke every chart in a
    pre-release build, which the fake-charter tests can't see. Only the timer's own assignments may bind it."""
    import ast
    import inspect
    import textwrap
    from riffchart.product.core import RiffCharter
    tree = ast.parse(textwrap.dedent(inspect.getsource(RiffCharter.chart_file)))
    binds = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == "t" and isinstance(n.ctx, ast.Store)]
    assert len(binds) == 2, [b.lineno for b in binds]          # t = time.perf_counter(); t = now (in done)


def test_lyrics_hear_the_whole_song(tmp_path):
    """Where the guitar is quiet, guitar.ogg borrows the vocals (follow.py): lyrics written down or timed
    from song.ogg / vocals.ogg alone missed whole verses. lyrics_audio separates every track summed."""
    import soundfile as sf
    from riffchart.product.core import OUT_SR, RiffCharter
    n = OUT_SR
    verse = np.zeros((n, 2), np.float32)
    verse[: n // 2] = 0.5                                    # the singing, only in the guitar track
    sf.write(tmp_path / "guitar.ogg", verse, OUT_SR)
    sf.write(tmp_path / "song.ogg", np.zeros((n, 2), np.float32), OUT_SR)
    sf.write(tmp_path / "vocals.ogg", np.zeros((n, 2), np.float32), OUT_SR)
    seen = []

    class Sep:
        sr = OUT_SR

        def __call__(self, mix):
            seen.append(mix)
            return {"vocals": mix}

    ac = RiffCharter.__new__(RiffCharter)
    ac.sep = Sep()
    voc, sr, mix = ac.lyrics_audio(tmp_path)
    assert sr == OUT_SR and float(np.abs(voc[: n // 4]).mean()) > 0.3      # the verse is heard
