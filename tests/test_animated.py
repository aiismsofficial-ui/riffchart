"""Animated highways and backgrounds: every effect renders, loops are seamless, repeats reach the song's length."""
import io

import av
import numpy as np
import pytest

from riffchart.product import background_anim as ba
from riffchart.product import highways
from riffchart.product.highway_render import EFFECTS, Animator, layers


def _frames(data):
    with av.open(io.BytesIO(data)) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]


def test_old_designs_get_defaults():
    d = highways.clean_design({"name": "x", "top": "#000000"})
    assert d["anim"] == "none" and d["rail_style"] == "solid" and d["fret_tint"] == 0
    with pytest.raises(ValueError):
        highways.clean_design({"anim_color": "red"})
    assert highways.clean_design({"anim": "bogus", "anim_speed": 99})["anim"] == "none"


@pytest.mark.parametrize("effect", [e for e in EFFECTS if e != "none"])
def test_highway_effects_loop(effect):
    d = highways.clean_design({"anim": effect, "anim_seconds": 2, "rail_style": "neon", "lane_style": "glow",
                               "fret_tint": 50, "vignette": 30, "center_glow": 30})
    L = layers(d, None, None, (64, 128))
    a = Animator(d, L, fps=8)
    f = [a.frame(i).astype(np.float32) for i in range(a.n)]
    assert f[0].shape == (128, 64, 3)
    jump = np.abs(f[0] - f[-1]).mean()                    # last -> first is an ordinary step, not a jump
    steps = [np.abs(f[i + 1] - f[i]).mean() for i in range(len(f) - 1)]
    assert jump <= max(steps) * 1.5 + 1.0


def test_highway_video_is_vp8():
    data = highways.render_video({"anim": "sparks", "anim_seconds": 2}, preview=True)
    with av.open(io.BytesIO(data)) as c:
        assert c.streams.video[0].codec_context.name == "vp8"
    assert len(_frames(data)) == 48


@pytest.mark.parametrize("preset", list(ba.PRESETS))
def test_background_presets_render(preset):
    data = ba.render(preset, None, size=(64, 36), seconds=1, fps=8, crf=40, bitrate="200k")
    assert len(_frames(data)) == 8


def test_repeat_reaches_song_length(tmp_path):
    loop = ba.render("embers", None, size=(64, 36), seconds=1, fps=ba.FPS, crf=40, bitrate="200k")
    ba.repeat_to(loop, 4.5, tmp_path / "v.webm")
    with av.open(str(tmp_path / "v.webm")) as c:
        n = sum(1 for _ in c.decode(video=0))
    assert n == 5 * ba.FPS                                # whole loops until past 4.5 s


def test_background_matches_screen_shape():
    assert ba.target_size((3440, 1440)) == (2560, 1072)          # ultrawide keeps 21:9
    assert ba.target_size((1920, 1080)) == (1920, 1080)
    assert ba.target_size((2560, 1440)) == (1920, 1080)          # 16:9, capped at 1080 high
    w, h = ba.target_size((1366, 768))
    assert w % 2 == 0 and h % 2 == 0
