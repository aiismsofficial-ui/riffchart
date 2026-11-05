import torch

from autochart.model.net import ChartTransformer, ModelConfig, rope
from autochart.model.tokenizer import V

CFG = ModelConfig(d_model=64, n_heads=4, enc_layers=1, dec_layers=2, dropout=0.0, max_beats=8)


def _inputs(B=2, beats=6, T=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, 2, 128, beats * 48, generator=g)
    ids = torch.randint(0, V.SIZE, (B, T), generator=g)
    bidx = torch.randint(0, beats, (B, T), generator=g)
    pos = torch.randint(0, 48, (B, T), generator=g)
    return x, ids, bidx, pos


def test_forward_shapes():
    m = ChartTransformer(CFG).eval()
    x, ids, b, p = _inputs()
    assert m(x, ids, b, p).shape == (2, 20, V.SIZE)
    mem, bm = m.encode(x)
    assert mem.shape == (2, 6 * 16, 64) and bm.shape == (2, 6, 64)


def test_rope_is_relative():
    q = torch.randn(1, 1, 5, 8)
    k = torch.randn(1, 1, 5, 8)
    pos = torch.arange(5)
    a = rope(q, pos) @ rope(k, pos).transpose(-1, -2)
    b = rope(q, pos + 7) @ rope(k, pos + 7).transpose(-1, -2)
    assert torch.allclose(a, b, atol=1e-5)


def test_incremental_decoding_matches_full_forward_with_left_padding():
    torch.manual_seed(0)
    m = ChartTransformer(CFG).eval()
    x, ids, b, p = _inputs(B=1, T=12)
    full = m(x, ids, b, p)                                  # (1, 12, V)
    mem, bm = m.encode(x)
    # row 0 = the sequence, row 1 = same sequence left-padded by 3 tokens of junk
    P = 4
    pad = 3
    ids2 = torch.cat([torch.full((1, pad), 7), ids[:, :P]], 1)
    b2 = torch.cat([torch.zeros(1, pad, dtype=torch.long), b[:, :P]], 1)
    p2 = torch.cat([torch.zeros(1, pad, dtype=torch.long), p[:, :P]], 1)
    ids1 = torch.cat([torch.zeros(1, pad, dtype=torch.long), ids[:, :P]], 1)
    batch_ids = torch.cat([ids1, ids2]); batch_b = torch.cat([b2, b2]); batch_p = torch.cat([p2, p2])
    batch_ids[0, :pad] = 0
    valid = torch.ones(2, pad + 12, dtype=torch.bool)
    valid[:, :pad] = False
    cache = m.new_cache()
    out = m.step(batch_ids, batch_b, batch_p, 0, mem.expand(2, -1, -1), bm.expand(2, -1, -1), cache, valid[:, : pad + P])
    assert torch.allclose(out[0], full[0, P - 1], atol=1e-4) and torch.allclose(out[1], full[0, P - 1], atol=1e-4)
    t = pad + P
    for j in range(P, 12):
        out = m.step(ids[:, j : j + 1].expand(2, -1), b[:, j : j + 1].expand(2, -1), p[:, j : j + 1].expand(2, -1), t,
                     mem.expand(2, -1, -1), bm.expand(2, -1, -1), cache, valid[:, : t + 1])
        assert torch.allclose(out[0], full[0, j], atol=1e-4), j
        assert torch.allclose(out[1], full[0, j], atol=1e-4), j
        t += 1


def test_events_to_song_clips_sustain_at_next_chord():
    import numpy as np
    from autochart.model.generate import events_to_song
    from autochart.model.tokenizer import Event
    beats = 1.0 + 0.5 * np.arange(16)
    evs = [Event(0, 1, 0, 96), Event(24, 2, 0, 0), Event(48, 4, 0, 30)]   # first sustain overshoots by 72
    song = events_to_song({"expert": evs}, beats, beats[::4])
    tr = song.tracks["expert"]
    k = song.resolution // 48
    first = tr.sustain[tr.tick == tr.tick.min()][0]
    assert first == 24 * k and tr.sustain.max() == 30 * k


def test_pitch_channel_starts_identical_to_the_old_model(tmp_path):
    import torch
    from autochart.model.net import ChartTransformer, ModelConfig
    from autochart.model.train import init_from
    kw = dict(d_model=64, n_heads=4, enc_layers=1, dec_layers=1, dropout=0.0)
    torch.manual_seed(0)
    old = ChartTransformer(ModelConfig(in_channels=2, **kw)).eval()
    torch.save({"model": old.state_dict()}, tmp_path / "old.pt")
    new = ChartTransformer(ModelConfig(in_channels=3, **kw)).eval()
    init_from(new, str(tmp_path / "old.pt"), "cpu")
    x = torch.randn(1, 2, 128, 48 * 4)
    xp = torch.cat([x, torch.randn(1, 1, 128, 48 * 4)], 1)              # any pitch input
    with torch.no_grad():
        a, b = old.encode(x), new.encode(xp)
    assert torch.allclose(a[0], b[0], atol=1e-5) and torch.allclose(a[1], b[1], atol=1e-5)


def test_pitch_features_shape_and_window():
    import numpy as np
    from autochart.model.dataset import pitch_window
    from autochart.model.pitch_feat import BINS, LO, cqt_uint8, dequant
    sr = 22050
    t = np.arange(sr * 2) / sr
    y = 0.05 * np.sin(2 * np.pi * 110.0 * t).astype(np.float32)         # A2 = MIDI 45, stem-like level
    q = cqt_uint8(y)
    assert q.shape == (1 + len(y) // 256, BINS) and q.dtype == np.uint8
    assert int(np.argmax(dequant(q[40:120]).mean(0))) == 45 - 31       # bin of A2 (G1 = bin 0)
    beats = np.arange(0, 3, 0.5)
    w = pitch_window(q, beats, 0, 4)
    assert w.shape == (1, 128, 4 * 48) and w.min() >= LO
    assert pitch_window(None, beats, 0, 4).max() == LO                    # no pitch -> silence
