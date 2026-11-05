import numpy as np

from autochart.data.audit import _peak, hit_stats, note_train, xcorr_lags
from autochart.data.dedup import _prep, chroma_similarity, norm_text


def test_xcorr_recovers_known_lag():
    rng = np.random.default_rng(0)
    notes = np.sort(rng.uniform(1, 59, 300))
    env = note_train(notes + 0.037, 6000, 100.0, 1.5) + rng.normal(0, 0.02, 6000)
    lags, c = xcorr_lags(env, note_train(notes, 6000, 100.0, 1.5), 50)
    lag, _, _ = _peak(lags, c, -15, 15)
    assert abs(lag * 10 - 37) < 3


def test_hit_stats():
    peaks = np.array([1.0, 2.0, 3.0, 4.0])
    rate, med, _ = hit_stats(np.array([1.01, 2.02, 3.2, 3.99]), peaks)
    assert rate == 0.75 and abs(med - (-10)) < 1e-6


def test_norm_text():
    assert norm_text("The Foo (Remastered) feat. Bar") == "foo"
    assert norm_text("<color=red>Héllo</color> - World!") == "hello world"


def test_chroma_similarity_same_vs_different():
    rng = np.random.default_rng(1)
    a = rng.random((200, 12))
    shifted = np.concatenate([rng.random((5, 12)), a[:-5]]) + rng.normal(0, 0.05, (200, 12))
    other = rng.random((200, 12))
    assert chroma_similarity(_prep(a), _prep(shifted)) > 0.9
    assert chroma_similarity(_prep(a), _prep(other)) < 0.3


def test_alignment_correction_sign():
    """A chart whose notes are 40 ms early must get a +40 ms correction (beyond detector latency)."""
    import pandas as pd

    from autochart.data.split import flag_songs

    base = dict(status="ok", n_expert=100, hit50=0.9, local_ratio=1.0, drift_range_ms=5.0, hit_med_off_ms=0.0,
                chart_end_past_audio_s=-5.0, bpm_min=120.0, bpm_max=120.0, delay_s=0.0)
    rows = [dict(base, lag_ms=25.0) for _ in range(9)] + [dict(base, lag_ms=65.0)]
    cfg = {"audit": {"min_hit50": 0.5, "min_local_ratio": 0.85, "max_drift_ms": 60, "min_bpm": 20, "max_bpm": 1000}}
    df = flag_songs(pd.DataFrame(rows), cfg)
    assert df.align_correction_ms.iloc[-1] == 40.0 and (df.align_correction_ms.iloc[:-1] == 0).all()
