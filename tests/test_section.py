"""Re-chart a section (M5): range decoding, splicing into the existing chart, the API (no GPU)."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from riffchart.formats.song import GuitarTrack, Song, TempoMap
from riffchart.model.generate import Charter, events_to_song
from riffchart.model.net import ChartTransformer, ModelConfig
from riffchart.model.tokenizer import Event
from riffchart.product import section as sec
from riffchart.product.core import Style
from riffchart.timing.grid import GRID, MUSICAL_POSITIONS

from test_ui import _charted, env  # noqa: F401  (fixture)


def tiny_charter(n_ctx=4, n_tgt=4):
    torch.manual_seed(0)
    ch = object.__new__(Charter)
    ch.model = ChartTransformer(ModelConfig(d_model=32, n_heads=4, enc_layers=1, dec_layers=1, dropout=0.0,
                                            max_beats=n_ctx + n_tgt)).eval()
    ch.cfg = {"data": {"input": "both"}}
    ch.mean, ch.std = np.zeros((2, 128, 1), np.float32), np.ones((2, 128, 1), np.float32)
    ch.device, ch.use_pitch, ch.n_ctx, ch.n_tgt = "cpu", False, n_ctx, n_tgt
    ch.pos_allowed = np.ones(len(MUSICAL_POSITIONS), bool)
    return ch


def test_chart_range_only_returns_new_beats():
    ch = tiny_charter()
    beats = 0.5 * np.arange(40)
    mel = np.random.default_rng(0).normal(size=(int(20 * 86.13), 2, 128)).astype(np.float16)
    prefix = {"expert": [Event(u, 1, 0, 0) for u in range(0, 30 * GRID, GRID)]}
    out = ch.chart(mel, beats, beats[::4], ["expert"], temperature=1.0, seed=3, audio_s=19.0,
                   start_beat=10, end_beat=17, prefix=prefix)
    units = [e.unit for e in out["expert"]]
    assert units and min(units) >= 10 * GRID and max(units) < 17 * GRID
    assert len(prefix["expert"]) == 30                              # caller's list untouched


def test_events_to_song_with_existing_tempo_keeps_ticks():
    tm = TempoMap(192, np.array([0, 192 * 8]), np.array([120.0, 150.0]))
    beats = tm.beat_times(20)
    song = events_to_song({"expert": [Event(9 * GRID + 12, 2, 0, 24)]}, beats, beats[::4], tempo=tm)
    tr = song.tracks["expert"]
    assert song.tempo is tm and list(tr.tick) == [9 * 192 + 48] and list(tr.sustain) == [96]


class StubCharter:
    """Returns fixed chords on every beat of the range (fret by difficulty), like Charter.chart."""
    use_pitch = False

    def __init__(self):
        self.kw = None

    def chart(self, mel, beats, downs, diffs, temperature, top_p, seed=0, start_beat=0, end_beat=None,
              prefix=None, **kw):
        self.kw = dict(start_beat=start_beat, end_beat=end_beat, prefix=prefix, seed=seed, diffs=diffs)
        return {d: [Event(b * GRID, 1 << i, 0, 4 * GRID if b == end_beat - 1 else 0) for b in range(start_beat, end_beat)]
                for i, d in enumerate(diffs)}


def _song():
    tm = TempoMap(192, np.array([0]), np.array([120.0]))

    def track(d, ticks):
        n = len(ticks)
        return GuitarTrack(d, np.array(ticks, np.int64), np.full(n, 16, np.uint8), np.zeros(n, np.int64),
                           np.zeros(n, np.uint8), np.zeros(n, bool), np.zeros(n, bool))
    every = list(range(0, 192 * 40, 192))
    return Song(192, tm, {"expert": track("expert", every), "hard": track("hard", every[::2])})


def test_chart_section_splices_range_and_clips_last_sustain():
    ac = SimpleNamespace(charter=StubCharter(), cfg={"decoding": {"temperature": 0.7, "top_p": 1.0, "min_gap": "auto"}},
                         min_gap={}, post_cfg={"fix": False, "star_power": False, "lower": "direct"})
    feats = {"audio_s": np.float64(20.0), "auto": np.zeros((1800, 2, 128), np.float16)}
    out = sec.chart_section(ac, feats, _song(), 0.0, start_tick=192 * 8 + 50, end_tick=192 * 12 - 10,
                            difficulties=["expert"], style=Style(follow="auto"), seed=5)
    assert (out["start"], out["end"]) == (192 * 8, 192 * 12)        # snapped out to whole beats
    kw = ac.charter.kw
    assert (kw["start_beat"], kw["end_beat"], kw["seed"], kw["diffs"]) == (8, 12, 5, ["expert"])
    assert len(kw["prefix"]["expert"]) == 40 and kw["prefix"]["expert"][3].unit == 3 * GRID
    rows = out["notes"]["expert"]
    assert [r[0] for r in rows] == [192 * k for k in range(8, 12)] and set(out["notes"]) == {"expert"}
    assert rows[-1][2] == 192                                       # 4-beat sustain cut at the next chord


def test_chart_section_rejects_range_past_the_audio():
    ac = SimpleNamespace(charter=StubCharter(), cfg={"decoding": {"temperature": 0.7, "top_p": 1.0, "min_gap": None}},
                         min_gap={}, post_cfg={"fix": False, "star_power": False})
    feats = {"audio_s": np.float64(5.0), "auto": np.zeros((450, 2, 128), np.float16)}
    with pytest.raises(ValueError):
        sec.chart_section(ac, feats, _song(), 0.0, 192 * 30, 192 * 32, ["expert"], Style(follow="auto"), 0)


def test_feature_cache_round_trip_and_key(tmp_path):
    f = tmp_path / "song.ogg"
    f.write_bytes(b"abc")
    key = sec.audio_key([f])
    feats = {"audio_s": np.float64(3.0), "version": np.int64(sec.FEATS_VERSION), "auto": np.ones((5, 2, 128), np.float16)}
    p = tmp_path / "c" / "section_feats.npz"
    sec.save_features(p, key, feats)
    got = sec.load_features(p, key)
    assert got is not None and got["auto"].shape == (5, 2, 128) and float(got["audio_s"]) == 3.0
    assert sec.load_features(p, key + "x") is None
    p.write_bytes(b"junk")
    assert sec.load_features(p, key) is None


def test_section_endpoint_caches_analysis(env, monkeypatch):
    import soundfile as sf
    folder = _charted(env)
    sf.write(folder / "song.ogg", np.zeros((8000, 2), np.float32), 8000, format="OGG", subtype="VORBIS")
    calls = {"feats": 0, "chart": []}

    def fake_features(ac, mix, sr):
        calls["feats"] += 1
        return {"audio_s": np.float64(1.0), "version": np.int64(sec.FEATS_VERSION), "auto": np.zeros((4, 2, 128), np.float16)}

    def fake_chart(ac, feats, song, offset_s, start, end, diffs, style, seed):
        calls["chart"].append((start, end, diffs, style.density, seed))
        return {"start": start, "end": end, "notes": {d: [[start, 1, 0, 0]] for d in diffs}}
    monkeypatch.setattr(sec, "section_features", fake_features)
    monkeypatch.setattr(sec, "chart_section", fake_chart)
    c, sid = env.client, "Muse - Uprising"
    chart = c.get(f"/api/songs/{sid}/chart").json()
    before = (folder / "notes.chart").read_text(encoding="utf-8")
    assert c.get(f"/api/songs/{sid}/section").json()["analysed"] is False
    body = {"chart": chart, "start": 192, "end": 768, "seed": 7, "options": {"difficulties": ["expert"], "density": 0.3}}
    r1 = c.post(f"/api/songs/{sid}/section", json=body)
    r2 = c.post(f"/api/songs/{sid}/section", json=body)
    assert r1.status_code == 200 and r1.json()["analysed"] and not r2.json()["analysed"]
    assert calls["feats"] == 1 and calls["chart"][0] == (192, 768, ["expert"], 0.3, 7)
    assert c.get(f"/api/songs/{sid}/section").json()["analysed"] is True
    assert c.post(f"/api/songs/{sid}/section", json=dict(body, end=100)).status_code == 400
    assert (folder / "notes.chart").read_text(encoding="utf-8") == before            # nothing written
