"""Per-instrument audio: borrowed melodic parts aren't played twice; near-silent parts get no file."""
import numpy as np

from riffchart.product.follow import Playable, instrument_parts


def test_parts_skip_borrowed_and_silent():
    sr, n, rng = 1000, 4000, np.random.default_rng(1)
    stems = {k: rng.normal(0, 0.3, (n, 2)).astype(np.float32) for k in ("guitar", "bass", "drums", "vocals", "other")}
    stems["piano"] = np.zeros((n, 2), np.float32)               # no piano in this song
    w = np.r_[np.zeros(8), np.ones(8)]                           # second half: guitar borrows the melodic parts
    melodic = stems["vocals"] + stems["other"]
    ws = np.repeat(w, 250)[:, None]
    play = Playable(stems["guitar"] + ws * melodic, stems["guitar"], w, 0.25, 2.0)
    parts = instrument_parts(stems, play, sr)
    assert set(parts) == {"bass", "drums", "vocals"}             # keys: silent, stays in song.ogg
    assert np.allclose(parts["vocals"][:2000], stems["vocals"][:2000])
    assert np.allclose(parts["vocals"][2000:], 0)                # already in guitar.ogg there
    mix = sum(stems.values())
    rest = mix - play.audio - sum(parts.values())
    assert np.allclose(rest[2000:], 0, atol=1e-5)                # song.ogg there: nothing but separation error
    assert np.allclose(rest[:2000], stems["other"][:2000], atol=1e-5)
