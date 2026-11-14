"""Automatic sections on a made-up song: intro, verse / chorus twice (sung), solo, chorus, outro."""
import numpy as np

from riffchart.formats.song import Song, TempoMap
from riffchart.product import structure as st

SR = 22050
BAR = 2.0                      # 120 BPM, 4/4


def _tone(freqs, n, rng, noise=0.02):
    t = np.arange(n) / SR
    return sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs) * 0.3 + noise * rng.standard_normal(n)


def _song():
    rng = np.random.default_rng(0)
    plan = [("intro", 4), ("verse", 8), ("chorus", 8), ("verse", 8), ("chorus", 8), ("solo", 8), ("chorus", 8),
            ("outro", 4)]
    chords = {"intro": [110, 165], "verse": [147, 220, 294], "chorus": [196, 247, 392, 494], "solo": [131, 196],
              "outro": [110, 165]}
    mix, voc, gtr = [], [], []
    starts, bar = [], 0
    for kind, bars in plan:
        n = int(bars * BAR * SR)
        starts.append(bar)
        bar += bars
        mix.append(_tone(chords[kind], n, rng, 0.05 if kind == "chorus" else 0.02) * (1.6 if kind == "chorus" else 1))
        sung = kind in ("verse", "chorus")
        voc.append(_tone([440 if kind == "chorus" else 330], n, rng) if sung else 0.0005 * rng.standard_normal(n))
        gtr.append(_tone([660, 880], n, rng) * (3.0 if kind == "solo" else 0.5))
    mix, voc, gtr = (np.concatenate(x).astype(np.float32) for x in (mix, voc, gtr))
    song = Song(192, TempoMap(192, np.array([0]), np.array([120.0])), {})
    return song, mix + voc, voc, gtr, starts


def test_auto_sections_finds_and_names_the_parts():
    song, mix, voc, gtr, starts = _song()
    secs = st.auto_sections(song, mix, SR, vocals=voc, guitar=gtr)
    got = [t // (192 * 4) for t, _ in secs]
    names = [n for _, n in secs]
    hits = sum(any(abs(g - s) <= 1 for g in got) for s in starts[1:])
    assert hits >= len(starts[1:]) - 1, (got, starts)
    kinds = [st.category(n) for n in names]
    assert kinds[0] == "intro" and kinds[-1] == "outro"
    assert "chorus" in kinds and "verse" in kinds and "solo" in kinds, names
    assert len(set(names)) == len(names)                          # numbered: Verse 1, Verse 2, ...


def test_category_of_human_names():
    assert [st.category(n) for n in ("Verse 1A", "chorus_2", "Pre-Chorus", "Gtr Solo B", "Big Rock Ending",
                                     "Intro Riff", "Bridge", "Main Riff")] == \
        ["verse", "chorus", "prechorus", "solo", "outro", "intro", "bridge", "instrumental"]
