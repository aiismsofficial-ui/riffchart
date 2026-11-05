from pathlib import Path

import numpy as np
import pytest

from autochart.model.tokenizer import V
from autochart.product.cli import find_audio, parse_diffs
from autochart.product.core import CHORD_BIN, Style, estimate_diff, preview_start, safe_name, song_ini
from autochart.product.meta import SongMeta, from_filename


def test_style_parse_and_tokens():
    st = Style.parse("density=-0.5, chords=low")
    assert st.density == -0.5 and st.chords == "low"
    assert st.tokens() == [V.DENS + V.N_DENS - 1, V.CHORD + CHORD_BIN["low"]]
    assert Style.parse("").tokens() == [V.DENS + V.N_DENS - 1, V.CHORD + V.N_CHORD - 1]   # AUTO, AUTO
    for bad in ("density=2", "chords=lots", "speed=1"):
        with pytest.raises(ValueError):
            Style.parse(bad)


def test_names_from_files_and_folders():
    assert from_filename(Path("x/Biohazard - Sellout/song.ogg")) == SongMeta("Sellout", "Biohazard")
    assert from_filename(Path("x/03 - Metallica - One.mp3")) == SongMeta("One", "Metallica")
    assert from_filename(Path("x/untitled.wav")) == SongMeta("untitled", "Unknown Artist")
    assert from_filename(Path("x/200 Stabs - Drilling Your Head/song.ogg")) == SongMeta("Drilling Your Head", "200 Stabs")
    assert from_filename(Path("x/07. Muse - Uprising.flac")) == SongMeta("Uprising", "Muse")
    assert safe_name('AC/DC - What?: "Live"') == "ACDC - What Live"


def test_song_ini_and_difficulty():
    cuts = [2.6, 3.5, 4.05, 4.75, 5.8, 7.5]
    assert estimate_diff(300, 0.0, 200.0, cuts) == 0                # 1.5 notes/s
    assert estimate_diff(900, 0.0, 200.0, cuts) == 3                # 4.5 notes/s
    assert estimate_diff(2000, 0.0, 200.0, cuts) == 6
    ini = song_ini(SongMeta("One", "Metallica", year="1988"), 446.2, 120.5, 4)
    for line in ("name = One", "artist = Metallica", "charter = autochart", "song_length = 446200",
                 "preview_start_time = 120500", "diff_guitar = 4", "year = 1988"):
        assert line in ini


def test_preview_picks_loud_section():
    sr = 1000
    y = np.full(200 * sr, 0.01, np.float32)
    y[100 * sr : 130 * sr] = 0.5                                  # loud "chorus" at 100-130 s
    downbeats = np.arange(0, 200, 2.0)
    assert preview_start(y, sr, downbeats, 200.0) == 100.0


def test_cli_inputs(tmp_path):
    (tmp_path / "a").mkdir()
    for n in ("a/x.mp3", "a/y.flac", "a/z.m4a", "a/notes.txt"):
        (tmp_path / n).write_bytes(b"")
    files, unsupported = find_audio([str(tmp_path)])
    assert [f.name for f in files] == ["x.mp3", "y.flac"] and [f.name for f in unsupported] == ["z.m4a"]
    assert parse_diffs("all") == ["expert", "hard", "medium", "easy"]
    assert parse_diffs("Expert, easy") == ["expert", "easy"]
