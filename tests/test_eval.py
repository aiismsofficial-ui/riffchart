import numpy as np
import pytest

from riffchart.eval.metrics import Events, evaluate, pitch
from riffchart.eval.playability import check
from riffchart.formats.song import OPEN_BIT, NoteType

S, H, T = NoteType.STRUM, NoteType.HOPO, NoteType.TAP
RULES = {"stack_s": 0.01, "sustain_gap_s": -0.001, "min_sustain_s": 0.05, "jump_frets": 3,
         "max_chord": 2, "max_span": 3, "min_ioi_s": 0.1, "jump_ioi_s": 0.15}


def ev(t, mask, sus=None, ntype=None):
    t = np.asarray(t, float)
    return Events(t, np.asarray(mask, np.uint8), np.zeros(len(t)) if sus is None else np.asarray(sus, float),
                  np.zeros(len(t), np.uint8) if ntype is None else np.asarray(ntype, np.uint8))


def test_identical_charts_score_perfect():
    r = ev([0, 0.5, 1.0, 1.5], [1, 2, 3, 4], [0, 0.3, 0, 0], [S, H, S, T])
    m = evaluate(r, r)
    for k in ("F1@25", "F1@50", "fret_exact", "contour", "sustain_acc", "type_acc", "hopo_f1", "tap_f1"):
        assert m[k] == 1.0, k
    assert m["density_ratio"] == 1.0


def test_onset_tolerances_and_prf():
    r = ev([0, 1, 2, 3], [1, 1, 1, 1])
    e = ev([0.03, 1.0, 2.2, 3.01, 4.0], [1, 1, 1, 1, 1])
    m = evaluate(r, e)
    assert m["R@25"] == 0.5 and m["P@25"] == pytest.approx(2 / 5)   # 1.0, 3.01 within 25 ms
    assert m["R@50"] == 0.75 and m["P@50"] == pytest.approx(3 / 5)  # + 0.03


def test_one_estimate_matches_one_reference_only():
    r = ev([1.0, 1.02], [1, 2])
    e = ev([1.01], [1])
    assert evaluate(r, e)["R@50"] == 0.5


def test_fret_chord_and_contour():
    r = ev([0, 1, 2, 3], [1, 2, 4, 3])      # G R Y GR : up up down
    e = ev([0, 1, 2, 3], [2, 4, 8, 6])      # R Y B RY : up up down, shifted by one fret
    m = evaluate(r, e)
    assert m["fret_exact"] == 0 and m["contour"] == 1.0 and m["chord_agree"] == 1.0
    assert m["chord_exact"] == 0.0


def test_open_is_lowest_pitch():
    assert pitch(np.array([OPEN_BIT, 1, 16, 3]))[0] == -1 and list(pitch(np.array([1, 16, 3]))) == [0, 4, 0.5]


def test_hopo_tap_f1_nan_when_absent_and_empty_estimate():
    r = ev([0, 1], [1, 2])
    m = evaluate(r, r)
    assert np.isnan(m["hopo_f1"]) and np.isnan(m["tap_f1"])
    m = evaluate(r, Events.empty())
    assert m["F1@50"] == 0 and np.isnan(m["fret_exact"])


def test_playability_rules():
    e = ev([0.0, 0.005, 0.5, 0.55, 1.0, 1.5, 2.0, 3.0],
           [1 | OPEN_BIT, 1, 7, 16, 1, 1, 17, 2],
           [0, 0, 0, 0, 0.7, 0.02, 0, 0],
           [S, S, S, S, S, H, S, S])
    c = check(e, RULES, audio_s=2.5)
    per_min = 2.5 / 60
    counts = {k: round(v * per_min) for k, v in c.items()}
    assert counts["pv_open_chord"] == 1
    assert counts["pv_stacked"] == 1           # 0.0 -> 0.005
    assert counts["pv_chord_size"] == 1        # G R Y
    assert counts["pv_chord_span"] == 1        # G+O span 4
    assert counts["pv_too_fast"] == 1          # 0.5 -> 0.55
    assert counts["pv_fast_jump"] == 1         # Y-ish chord -> O in 50 ms
    assert counts["pv_sustain_overlap"] == 1   # 1.0 + 0.7 > 1.5
    assert counts["pv_tiny_sustain"] == 1      # 0.02 s
    assert counts["pv_bad_hopo"] == 1          # G HOPO after G
    assert counts["pv_out_of_audio"] == 1      # 3.0 > 2.5
