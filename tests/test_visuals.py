"""Backgrounds from the cover, music-video backgrounds, online cover search (no network: canned answers)."""
import io
import json

import numpy as np
import pytest
from PIL import Image

from riffchart.product import covers
from riffchart.product.visuals import (background_from_cover, bg_mode, refresh_background, save_video,
                                       set_bg_mode, write_own_background)
from riffchart.ui.library import remove_art, write_art


def _img(color, size=(300, 300), fmt="PNG") -> bytes:
    b = io.BytesIO()
    Image.new("RGB", size, color).save(b, fmt)
    return b.getvalue()


def test_background_from_cover_is_dark_16_9_jpeg():
    data = background_from_cover(_img((250, 60, 40)))
    assert data[:3] == b"\xff\xd8\xff"
    im = Image.open(io.BytesIO(data))
    assert im.size == (1920, 1080)
    px = np.asarray(im, np.float32)
    assert px[540, 960, 0] < 160 and px[540, 960, 0] > px[540, 960, 2]       # darkened, same hue
    assert px[5, 5].mean() < px[540, 960].mean()                               # vignette


def test_background_follows_cover_unless_chosen(tmp_path):
    f = tmp_path
    assert bg_mode(f) == "auto"
    write_art(f, _img((0, 0, 255)))
    assert (f / "background.jpg").exists()
    write_art(f, _img((0, 255, 0), fmt="JPEG"))                                # a new cover: a new background
    assert np.asarray(Image.open(f / "background.jpg"))[540, 960, 1] > 40
    remove_art(f)
    assert not (f / "background.jpg").exists()
    write_own_background(f, _img((9, 9, 9), (1280, 720)))
    assert bg_mode(f) == "own" and (f / "background.png").exists()
    write_art(f, _img((255, 0, 0)))                                            # own background untouched
    assert (f / "background.png").exists() and not (f / "background.jpg").exists()
    set_bg_mode(f, "none")
    refresh_background(f)
    assert not list(f.glob("background.*"))
    with pytest.raises(ValueError):
        write_own_background(f, b"GIF89a")


def test_existing_background_counts_as_own(tmp_path):
    (tmp_path / "background.png").write_bytes(_img((1, 2, 3)))
    assert bg_mode(tmp_path) == "own"
    (tmp_path / "album.png").write_bytes(_img((200, 0, 0)))
    refresh_background(tmp_path)
    assert (tmp_path / "background.png").exists()


def _encode(path, video=True, audio=True, secs=1.0):
    import av
    sr = 48000
    with av.open(str(path), "w") as c:
        if video:
            v = c.add_stream("mpeg4", rate=10)
            v.width = v.height = 64
            v.pix_fmt = "yuv420p"
        a = c.add_stream("aac", rate=sr, layout="stereo") if audio else None
        if video:
            for i in range(int(secs * 10)):
                fr = av.VideoFrame.from_ndarray(np.full((64, 64, 3), i * 20, np.uint8), format="rgb24")
                for pkt in v.encode(fr):
                    c.mux(pkt)
            for pkt in v.encode():
                c.mux(pkt)
        if audio:
            x = np.zeros((2, int(sr * secs)), np.float32)
            for i in range(0, x.shape[1], 960):
                fr = av.AudioFrame.from_ndarray(np.ascontiguousarray(x[:, i:i + 960]), format="fltp", layout="stereo")
                fr.sample_rate = sr
                fr.pts = i
                for pkt in a.encode(fr):
                    c.mux(pkt)
            for pkt in a.encode():
                c.mux(pkt)


def test_music_video_kept_without_sound(tmp_path):
    import av
    src = tmp_path / "Band - Song (Official Video).mp4"
    _encode(src)
    out = tmp_path / "song"
    out.mkdir()
    r = save_video(src, out)
    assert r["file"] == "video.mp4" and r["codec"] == "mpeg4" and abs(r["start_ms"]) < 100
    with av.open(str(out / "video.mp4")) as c:
        assert len(c.streams.video) == 1 and not c.streams.audio
        assert sum(1 for _ in c.decode(video=0)) == 10
    assert not list(out.glob("*.tmp"))


def test_no_video_from_audio_files(tmp_path):
    from mutagen.mp4 import MP4, MP4Cover
    a = tmp_path / "a.m4a"
    _encode(a, video=False)
    out = tmp_path / "o"
    out.mkdir()
    assert save_video(a, out) is None
    m = MP4(a)                                       # an embedded cover is a still "video" stream: not a video
    m["covr"] = [MP4Cover(_img((5, 5, 5), fmt="JPEG"), imageformat=MP4Cover.FORMAT_JPEG)]
    m.save()
    assert save_video(a, out) is None and not list(out.iterdir())


def test_cover_search_parsing(monkeypatch):
    itunes = {"results": [
        {"artworkUrl100": "https://is1-ssl.mzstatic.com/x/100x100bb.jpg", "collectionName": "Album A", "artistName": "Band", "releaseDate": "1999-01-01T00:00:00Z"},
        {"artworkUrl100": "https://is1-ssl.mzstatic.com/y/100x100bb.jpg", "collectionName": "album a", "artistName": "Band"},
        {"collectionName": "No Art"}]}
    mb = {"recordings": [{"artist-credit": [{"name": "Band"}], "releases": [
        {"title": "Album A", "date": "1999", "release-group": {"id": "rg1"}},
        {"title": "Album A (deluxe)", "release-group": {"id": "rg1"}},
        {"title": "Live", "release-group": {"id": "rg2"}}]}]}
    seen = []

    def fake(url):
        seen.append(url)
        return itunes if "itunes.apple.com" in url else mb
    monkeypatch.setattr(covers, "_get_json", fake)
    monkeypatch.setattr(covers, "_last_mb", 0.0)
    r = covers.search("Band", 'Song "Live"', "")
    assert not r["errors"]
    assert [x["source"] for x in r["results"]] == ["Apple Music", "MusicBrainz", "MusicBrainz"]
    assert r["results"][0]["full"].endswith("1200x1200bb.jpg") and r["results"][0]["year"] == "1999"
    assert r["results"][1]["full"] == "https://coverartarchive.org/release-group/rg1/front-1200"
    assert "Song+%5C%22Live%5C%22" in seen[1]                                    # quotes escaped for MusicBrainz


def test_cover_fetch_only_from_cover_hosts():
    for bad in ("http://is1-ssl.mzstatic.com/x.jpg", "https://example.com/x.jpg", "https://mzstatic.com.evil.io/x",
                "file:///C:/Windows/win.ini", "https://127.0.0.1/x"):
        with pytest.raises(ValueError):
            covers.fetch(bad, 1000)
    assert covers._host_ok("https://ia800.us.archive.org/1/items/x.jpg")
    assert covers._host_ok("https://coverartarchive.org/release-group/x/front-250")


def test_highways_tile_and_follow_the_choice(tmp_path):
    from riffchart.product.visuals import (highway_from_cover, hw_mode, refresh_highway, riffchart_highway,
                                           set_hw_mode, write_own_highway)
    rc = np.asarray(Image.open(io.BytesIO(riffchart_highway())).convert("RGB"), np.int16)
    assert rc.shape == (1024, 512, 3)
    assert np.abs(rc[0] - rc[-1]).max() <= 6                            # seamless top to bottom
    assert tuple(rc[500, 3]) == (245, 165, 36)                           # amber rail
    cov = np.asarray(Image.open(io.BytesIO(highway_from_cover(_img((20, 60, 230))))).convert("RGB"), np.int16)
    assert cov[500, 3, 2] > cov[500, 3, 0]                               # rail takes the cover's blue
    assert np.abs(cov[0] - cov[-1]).max() <= 12
    f = tmp_path
    assert hw_mode(f) == "riffchart"                                     # library default
    refresh_highway(f)
    assert (f / "highway.png").exists()
    set_hw_mode(f, "cover")
    refresh_highway(f)                                                   # no cover yet: the RiffChart one
    assert np.asarray(Image.open(f / "highway.png").convert("RGB"))[500, 3].tolist() == [245, 165, 36]
    write_art(f, _img((20, 200, 40)))                                    # a cover: the highway follows it
    px = np.asarray(Image.open(f / "highway.png").convert("RGB"), np.int16)[500, 3]
    assert px[1] > px[0] and px[1] > px[2]
    write_own_highway(f, _img((5, 5, 5), (512, 1024), "JPEG"))
    assert hw_mode(f) == "own" and (f / "highway.jpg").exists() and not (f / "highway.png").exists()
    write_art(f, _img((200, 0, 0)))                                      # own highway untouched
    assert (f / "highway.jpg").exists()
    set_hw_mode(f, "none")
    refresh_highway(f)
    assert not list(f.glob("highway.*"))
    with pytest.raises(ValueError):
        set_hw_mode(f, "rainbow")


def test_background_choice_saved_alongside_highway(tmp_path):
    """The background mode must be remembered even when the state file already holds the highway choice."""
    from riffchart.product.visuals import refresh_highway
    refresh_highway(tmp_path)                                            # state file now exists (highway only)
    write_art(tmp_path, _img((0, 0, 255)))                               # makes background.jpg (auto)
    assert bg_mode(tmp_path) == "auto"
    write_art(tmp_path, _img((0, 255, 0)))
    assert np.asarray(Image.open(tmp_path / "background.jpg"))[540, 960, 1] > 30


def test_search_errors_are_reported(monkeypatch):
    def boom(url):
        raise OSError("offline")
    monkeypatch.setattr(covers, "_get_json", boom)
    monkeypatch.setattr(covers, "_last_mb", 0.0)
    r = covers.search("Band", "Song")
    assert r["results"] == [] and len(r["errors"]) == 2 and "offline" in json.dumps(r)
