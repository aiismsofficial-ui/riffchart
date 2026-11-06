from pathlib import Path

import numpy as np
import pytest

from riffchart.model.tokenizer import V
from riffchart.product.cli import find_audio, parse_diffs
from riffchart.product.core import CHORD_BIN, Style, estimate_diff, preview_start, safe_name, song_ini
from riffchart.product.meta import SongMeta, clean, from_filename


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
    for line in ("name = One", "artist = Metallica", "charter = RiffChart", "song_length = 446200",
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
    for n in ("a/x.mp3", "a/y.flac", "a/z.m4a", "a/w.webm", "a/video.mp4", "a/v.aiff", "a/notes.txt"):
        (tmp_path / n).write_bytes(b"")
    files, unsupported = find_audio([str(tmp_path)])
    assert [f.name for f in files] == ["w.webm", "x.mp3", "y.flac", "z.m4a"]   # not the background video
    assert [f.name for f in unsupported] == ["v.aiff"]
    assert parse_diffs("all") == ["expert", "hard", "medium", "easy"]
    assert parse_diffs("Expert, easy") == ["expert", "easy"]


def test_names_from_video_downloads():
    def name(f):
        return clean(from_filename(Path(f)))
    assert name("Pierce The Veil - Emergency Contact (Official Video) [dQw4w9WgXcQ].webm") ==         SongMeta("Emergency Contact", "Pierce The Veil")
    assert name("Muse - Uprising [Official Audio] [a-b_c1234Z9].m4a") == SongMeta("Uprising", "Muse")
    assert name("Sellout [abcdefghijk].m4a") == SongMeta("Sellout", "Unknown Artist")
    assert clean(SongMeta("One", "Metallica - Topic")) == SongMeta("One", "Metallica")
    assert name("Song [not an id].mp3").title == "Song [not an id]"


@pytest.mark.parametrize("ext,codec,video", [(".m4a", "aac", False), (".webm", "libopus", False),
                                             (".mp4", "aac", True)])
def test_decode_video_site_formats(tmp_path, ext, codec, video):
    av = pytest.importorskip("av")
    from riffchart.audio.io import load_audio
    sr, secs = 48000, 2.0
    t = np.arange(int(sr * secs)) / sr
    x = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 660 * t)]).astype(np.float32) * 0.5
    path = tmp_path / f"Artist - Song{ext}"
    with av.open(str(path), "w") as c:
        if video:                                       # a music video: the video stream must be ignored
            v = c.add_stream("mpeg4", rate=10); v.width = v.height = 64; v.pix_fmt = "yuv420p"
        a = c.add_stream(codec, rate=sr, layout="stereo")
        if video:
            for i in range(int(secs * 10)):
                f = av.VideoFrame.from_ndarray(np.full((64, 64, 3), i * 10, np.uint8), format="rgb24")
                for pkt in v.encode(f):
                    c.mux(pkt)
            for pkt in v.encode():
                c.mux(pkt)
        n = 960
        for i in range(0, x.shape[1], n):
            f = av.AudioFrame.from_ndarray(np.ascontiguousarray(x[:, i:i + n]), format="fltp", layout="stereo")
            f.sample_rate = sr; f.pts = i
            for pkt in a.encode(f):
                c.mux(pkt)
        for pkt in a.encode():
            c.mux(pkt)
    y, r = load_audio(path, 44100, mono=False)
    assert r == 44100 and y.shape[1] == 2 and y.dtype == np.float32
    assert abs(len(y) / r - secs) < 0.1
    mid = y[int(0.5 * r):int(1.5 * r)]
    spec = np.abs(np.fft.rfft(mid[:, 0]))
    assert abs(np.argmax(spec) * r / len(mid) - 440) < 5      # left channel kept its tone
    assert 0.3 < np.abs(mid).max() < 0.7


@pytest.mark.parametrize("artist,title,album,want", [
    ("Unknown Artist", "front-porch-step-drown-128-ytshorts.savetube.me", "", ("Unknown Artist", "Front Porch Step Drown", "")),
    ("Unknown Artist", "night-drive-lyric-video-128-ytshorts.savetube.me", "", ("Unknown Artist", "Night Drive", "")),
    ("Måneskin", "TIMEZONE - ManeskinVEVO", "", ("Måneskin", "TIMEZONE", "")),
    ("Bullet For My Valentine", "Your Betrayal - bulletvalentineVEVO", "", ("Bullet For My Valentine", "Your Betrayal", "")),
    ("Some Uploader", "Rammstein - ''Pussy'' - - - (English CC)", "", ("Some Uploader", "Rammstein - Pussy", "")),
    ("Kate Bush", "Running Up That Hill (Rock Cover by Our Last Night) - Our Last Night", "",
     ("Kate Bush", "Running Up That Hill (Rock Cover by Our Last Night)", "")),
    ("Unknown Artist", "Choosin_Texas_-_Ella_Langley_Rock_Cover_by_Our_Last_Night", "",
     ("Our Last Night", "Choosin Texas - Ella Langley (Rock Cover)", "")),
    ("Roadrunner Records", "Devildriver - Clouds Over California", "Devildriver - Clouds Over California [OFFICIAL VIDEO]",
     ("Devildriver", "Clouds Over California", "")),
    ("LØLØ", "whiskey & coke (Official Visualizer) - LØLØ", "", ("LØLØ", "whiskey & coke", "")),
    ("Unknown Artist", "y2mate.com - Muse - Uprising (320 kbps)", "", ("Muse", "Uprising", "")),
    # already fine: left alone
    ("Silverstein", "Bad Habits - UNFD", "", ("Silverstein", "Bad Habits - UNFD", "")),
    ("coolchannel", "Night Drive - The Midnight Hours", "", ("coolchannel", "Night Drive - The Midnight Hours", "")),
    ("DRAGONFORCE", "Through the Fire and Flames", "Inhuman Rampage", ("DRAGONFORCE", "Through the Fire and Flames", "Inhuman Rampage")),
    ("AC/DC", "T.N.T.", "", ("AC/DC", "T.N.T.", "")),
    ("blink-182", "all-the-small-things", "", ("blink-182", "All The Small Things", "")),
    ("Metallica", "One", "", ("Metallica", "One", "")),
])
def test_tidy_download_names(artist, title, album, want):
    m = clean(SongMeta(title, artist, album))
    assert (m.artist, m.title, m.album) == want
    assert clean(m) == m                                     # tidying twice changes nothing


@pytest.mark.parametrize("artist,title,space_ok,want", [
    ("SOiL", "SOiL Halo", True, "Halo"),
    ("SOiL", "SOiL Halo", False, "SOiL Halo"),                  # space only: not when unsure
    ("Muse", "Muse: Uprising", False, "Uprising"),
    ("Muse", "MUSE | Uprising", False, "Uprising"),
    ("Måneskin", "Maneskin – Timezone", False, "Timezone"),
    ("Guns N' Roses", "Guns N Roses Paradise City", True, "Paradise City"),
    ("AC/DC", "ACDC Thunderstruck", True, "Thunderstruck"),
    # the artist's name is part of the title: left alone
    ("Queen", "Queen of the Night", True, "Queen of the Night"),
    ("Kiss", "Kissin' Time", True, "Kissin' Time"),
    ("Bad Company", "Bad Company", True, "Bad Company"),
    ("Motörhead", "Motorhead", True, "Motorhead"),
    ("Death", "Death by Metal", True, "Death by Metal"),
    ("Boston", "Boston's Theme", True, "Boston's Theme"),
    ("Queen", "Queens of the Stone Age", True, "Queens of the Stone Age"),
    ("Iron Maiden", "Iron Maiden (Live)", True, "Iron Maiden (Live)"),
    ("Unknown Artist", "Unknown Artist Song", True, "Unknown Artist Song"),
    ("The Who", "Who Are You", True, "Who Are You"),
])
def test_strip_artist(artist, title, space_ok, want):
    from riffchart.product.meta import strip_artist
    assert strip_artist(title, artist, space_ok) == want


def test_clean_strips_artist_from_titles():
    assert clean(SongMeta("Muse: Uprising", "Muse")).title == "Uprising"
    assert clean(SongMeta("SOiL Halo", "SOiL")).title == "SOiL Halo"            # automatic: not from a space
    assert clean(SongMeta("SOiL Halo", "SOiL"), loose=True).title == "Halo"     # Tidy names suggestion
    assert clean(SongMeta("soil-halo", "SOiL")).title == "Halo"                 # file-name slug
    m = clean(SongMeta("SOiL Halo", "SOiL"), loose=True)
    assert clean(m, loose=True) == m
