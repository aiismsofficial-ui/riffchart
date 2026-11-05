import numpy as np

from riffchart.eval.metrics import Events
from riffchart.eval.playability import check
from riffchart.formats.song import OPEN_BIT, GuitarTrack, NoteType, Song, TempoMap
from riffchart.post.playability import fix_track, rebuild
from riffchart.post.star_power import copy_phrases, place_star_power

RES = 192
RULES = {"stack_s": 0.010, "sustain_gap_s": -0.001, "min_sustain_s": 0.050, "jump_frets": 3,
         "max_chord": 3, "max_span": 4, "min_ioi_s": 0.055, "jump_ioi_s": 0.060}
S, H = NoteType.STRUM, NoteType.HOPO


def _song(tick, mask, sus, ntype, bpm=120.0):
    tm = TempoMap(RES, np.array([0]), np.array([bpm]))
    tr = rebuild("expert", np.array(tick), np.array(mask), np.array(sus), np.array(ntype), RES)
    return Song(RES, tm, {"expert": tr})


def _pv(song, tr):
    return check(Events.from_track(song, tr), RULES, audio_s=60.0)


def test_fix_removes_every_violation():
    # 120 BPM: 1 tick = 1/384 s. 64th at 12 ticks = 31 ms (too fast), open+fret, 4-note chord,
    # sustain overlapping the next note, HOPO on a repeated chord, tiny sustain, fast big jump.
    tick = [0, 12, 192, 384, 576, 600, 768, 960, 980]
    mask = [1, 2, OPEN_BIT | 1, 15, 4, 4, 1, 1, 16]
    sus = [0, 0, 0, 400, 0, 0, 10, 0, 0]
    ntype = [S, S, S, S, S, H, S, S, S]
    song = _song(tick, mask, sus, ntype)
    before = _pv(song, song.tracks["expert"])
    assert before["pv_total"] > 0
    fixed = fix_track(song, song.tracks["expert"], RULES, audio_s=60.0)
    after = _pv(song, fixed)
    assert after["pv_total"] == 0, after
    assert 0 in fixed.tick and 12 not in fixed.tick          # the on-beat note wins over the 64th


def test_fix_keeps_clean_chart_and_types():
    tick = [0, 96, 192, 288, 384]
    mask = [1, 2, 4, 2, 3]
    sus = [0, 0, 90, 0, 0]
    ntype = [S, H, H, S, S]
    song = _song(tick, mask, sus, ntype)
    fixed = fix_track(song, song.tracks["expert"], RULES, audio_s=60.0)
    tr = song.tracks["expert"]
    assert np.array_equal(fixed.tick, tr.tick) and np.array_equal(fixed.mask, tr.mask)
    assert np.array_equal(fixed.ntype, tr.ntype) and np.array_equal(fixed.sustain, tr.sustain)


def test_star_power_rate_and_copy():
    n = 8 * 120                                              # 4 minutes of 8th notes at 120 BPM
    tick = np.arange(n) * 96
    song = _song(tick, (1 << (np.arange(n) % 5)), np.zeros(n, int), np.zeros(n, int))
    sp = place_star_power(song, song.tracks["expert"])
    assert 7 <= len(sp) <= 11
    assert all(b > 0 for _, b in sp) and all(sp[i][0] + sp[i][1] <= sp[i + 1][0] for i in range(len(sp) - 1))
    lower = rebuild("hard", tick[:40], np.ones(40, int), np.zeros(40, int), np.zeros(40, int), RES)
    assert all(s < 40 * 96 for s, _ in copy_phrases(sp, lower))


def test_density_thin_and_thicken_are_predictable():
    from riffchart.post.density import adjust

    class Keep:                                   # stub reducer: prefer chords on the beat
        def predict_proba(self, X):
            p = np.where(np.arange(len(X)) % 2 == 0, 0.9, 0.2)
            return np.stack([1 - p, p], 1)

    models = {"keep": Keep()}
    ex = _song(np.arange(0, 192 * 40, 96), np.tile([1, 2, 4, 8], 20), np.zeros(80), np.zeros(80)).tracks["expert"]
    hard = _song(np.arange(0, 192 * 40, 192), np.tile([1, 2], 20), np.zeros(40), np.zeros(40)).tracks["expert"]
    hard.difficulty = "hard"
    song = Song(RES, TempoMap(RES, np.array([0]), np.array([120.0])), {"expert": ex, "hard": hard})
    out = adjust(song, -1.0, models)
    assert len(out["expert"]) == 40 and len(out["hard"]) == 20                    # -1 keeps half
    assert set(out["expert"].tick) <= set(ex.tick)
    out = adjust(song, 1.0, models)
    assert len(out["expert"]) == 80                                               # Expert can't borrow
    assert len(out["hard"]) == 40 + 24 and set(out["hard"].tick) <= set(ex.tick)  # +60%, from Expert
    assert adjust(song, 0.0, models)["hard"] is hard


def test_ladder_keeps_each_difficulty_below_the_one_above():
    from riffchart.formats.song import popcount
    from riffchart.post.ladder import ladder
    tm = TempoMap(RES, np.array([0]), np.array([120.0]))
    win = RES * 16

    def tr(diff, ticks, masks):
        t = GuitarTrack.empty(diff) if False else rebuild(diff, np.array(ticks), np.array(masks), np.zeros(len(ticks)),
                                                           np.zeros(len(ticks), np.uint8), RES)
        return t
    ex_t = list(range(0, win, RES // 2)) + list(range(win, 2 * win, RES * 2))    # busy window, then sparse
    ex = tr("expert", ex_t, [1] * len(ex_t))
    hard_t = list(range(0, 2 * win, RES // 2))                                    # busy in both windows
    hard = tr("hard", hard_t, [3] * len(hard_t))                                  # all chords
    song = Song(RES, tm, {"expert": ex, "hard": hard})
    out = ladder(song, dict(song.tracks), None, {"hard": 0.85})
    h = out["hard"]
    w0, w1 = (h.tick < win).sum(), (h.tick >= win).sum()
    assert w1 <= (ex.tick >= win).sum()                      # local: never busier than Expert there
    assert len(h) <= round(0.85 * len(ex))                   # song-wide cap
    assert (popcount(h.mask) > 1).sum() == 0                 # Expert has no chords -> neither does Hard
    assert w0 > 0


def test_follow_fills_quiet_guitar_with_other_instruments():
    from riffchart.product.follow import playable_part
    sr, n = 1000, 40_000
    t = np.arange(n) / sr
    guitar = np.where(t < 20, 0.3, 0.003)[:, None] * np.sin(2 * np.pi * 50 * t)[:, None] * np.ones((1, 2))
    piano = 0.2 * np.sin(2 * np.pi * 70 * t)[:, None] * np.ones((1, 2))
    stems = {"guitar": guitar.astype(np.float32), "piano": piano.astype(np.float32),
             "drums": np.zeros((n, 2), np.float32)}
    g = playable_part(stems, sr, "guitar")
    assert g.filled_s == 0 and g.model_in is stems["guitar"]
    a = playable_part(stems, sr, "auto")
    assert 15 <= a.filled_s <= 21                            # the quiet second half
    assert a.weight[:60].max() == 0 and a.weight[-20:].min() == 1
    assert np.allclose(a.audio[:15 * sr], guitar[:15 * sr])  # loud part: guitar only
    assert abs(a.audio[-5 * sr:]).max() > 0.15               # quiet part: piano in the play-along track
    al = playable_part(stems, sr, "all")
    assert al.weight.min() == 1
