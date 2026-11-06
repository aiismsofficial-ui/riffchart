"""Guitar difficulty estimator (product/difficulty.py) and its fitted model (configs/difficulty.yaml)."""
import numpy as np
import yaml

from riffchart.formats.song import GuitarTrack, NoteType, Song, TempoMap
from riffchart.paths import project_path
from riffchart.product.difficulty import FEATURES, chart_features, estimate, load_model, score

RES = 192


def _song(step_ticks, n=400, masks=(1, 2, 4, 8, 16), hopo=False, burst=None):
    """120 BPM (1 beat = 192 ticks = 0.5 s); notes every step_ticks; burst=(start, count, step) inserts a run."""
    ticks = list(range(0, n * step_ticks, step_ticks))
    if burst:
        s0, cnt, st = burst
        ticks = sorted(set(ticks) | set(range(s0, s0 + cnt * st, st)))
    t = np.array(ticks, np.int64)
    k = len(t)
    mask = np.array([masks[i % len(masks)] for i in range(k)], np.uint8)
    ntype = np.full(k, NoteType.HOPO if hopo else NoteType.STRUM, np.uint8)
    tr = GuitarTrack("expert", t, mask, np.zeros(k, np.int64), ntype, np.zeros(k, bool), np.zeros(k, bool))
    return Song(RES, TempoMap(RES, np.array([0]), np.array([120.0])), {"expert": tr})


def test_features():
    f = chart_features(_song(96))                       # 8th notes at 120 BPM = 4 notes/s, single frets
    assert set(f) == set(FEATURES)
    assert abs(f["nps"] - 4.0) < 0.05 and abs(f["peak_nps"] - 4.0) < 0.3
    assert f["chord_frac"] == 0 and f["fast_frac"] == 0 and f["move_ps"] > 0
    assert chart_features(_song(96, n=10)) is None      # too few notes to judge


def test_model_ranks_harder_charts_higher():
    m = load_model(project_path("configs/difficulty.yaml"))
    assert m["features"] == list(FEATURES) and len(m["weights"]) == len(FEATURES)
    assert list(m["cuts"]) == sorted(m["cuts"]) and len(m["cuts"]) == 6
    slow, mid, fast = _song(192), _song(96), _song(32)  # 2, 4 and 12 notes/s
    assert score(chart_features(slow), m) < score(chart_features(mid), m) < score(chart_features(fast), m)
    assert estimate(slow, m) <= 2 and estimate(fast, m) >= 5
    # a fast solo in an otherwise easy song raises the rating (the busiest stretch counts most)
    solo = _song(192, burst=(192 * 100, 120, 24))
    assert estimate(solo, m) > estimate(slow, m)
    assert 0 <= estimate(Song(RES, TempoMap(RES, np.array([0]), np.array([120.0])), {}), m) <= 6


def test_product_config_points_at_model():
    cfg = yaml.safe_load(open(project_path("configs/product.yaml"), encoding="utf-8"))
    assert project_path(cfg["difficulty_model"]).exists()
