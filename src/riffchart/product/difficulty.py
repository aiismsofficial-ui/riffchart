"""Guitar difficulty (song.ini diff_guitar, 0-6) estimated from the Expert chart.

More than raw speed: busy-section density, fast runs, chords, hand movement, HOPOs / taps and sustains,
combined linearly (weights fitted on human-rated charts by scripts/fit_difficulty.py, stored in
configs/difficulty.yaml) and cut into the 0-6 scale."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from ..formats.song import OPEN_BIT, NoteType, Song

FEATURES = ("nps", "peak_nps", "p90_nps", "fast_frac", "very_fast_frac", "chord_frac", "chord3_frac",
            "move_ps", "hopo_frac", "tap_frac", "sustain_frac", "open_frac")
WINDOW_S = 5.0


def chart_features(song: Song, difficulty: str = "expert") -> dict[str, float] | None:
    """Difficulty features of one difficulty of a chart, or None if it has fewer than 20 notes."""
    t = song.tracks.get(difficulty)
    if t is None or len(t) < 20:
        return None
    order = np.argsort(t.tick, kind="stable")
    on, sus = song.times(t)
    on, sus = on[order], sus[order]
    mask = np.asarray(t.mask)[order].astype(np.int64)
    ntype = np.asarray(t.ntype)[order]
    span = max(float(on[-1] - on[0]), 1.0)
    n = len(on)
    # density in 5 s windows starting at every note: the busiest stretches decide how hard a song feels
    ends = np.searchsorted(on, on + WINDOW_S, side="left")
    win = (ends - np.arange(n)) / WINDOW_S
    ioi = np.diff(on)
    frets = mask & 0x1F
    nfret = np.array([bin(int(m)).count("1") for m in frets])
    lanes = np.array([np.mean([i for i in range(5) if m >> i & 1]) if m else 2.0 for m in frets])
    move = np.abs(np.diff(lanes)).sum() / span
    return {
        "nps": n / span,
        "peak_nps": float(win.max()),
        "p90_nps": float(np.percentile(win, 90)),
        "fast_frac": float((ioi < 0.15).mean()),
        "very_fast_frac": float((ioi < 0.10).mean()),
        "chord_frac": float((nfret >= 2).mean()),
        "chord3_frac": float((nfret >= 3).mean()),
        "move_ps": float(move),
        "hopo_frac": float((ntype == NoteType.HOPO).mean()),
        "tap_frac": float((ntype == NoteType.TAP).mean()),
        "sustain_frac": float((sus > 0.05).mean()),
        "open_frac": float(((mask & OPEN_BIT) > 0).mean()),
    }


def load_model(path: str | Path) -> dict:
    return yaml.safe_load(open(path, encoding="utf-8"))


def score(feats: dict[str, float], model: dict) -> float:
    """Continuous difficulty (about 0-6) from the fitted linear model."""
    x = np.array([feats[k] for k in model["features"]], np.float64)
    z = (x - np.array(model["mean"])) / np.array(model["std"])
    return float(model["intercept"] + z @ np.array(model["weights"]))


def estimate(song: Song, model: dict) -> int:
    """diff_guitar 0-6 for a chart (0 when the Expert chart is nearly empty)."""
    f = chart_features(song)
    if f is None:
        return 0
    return int(np.searchsorted(np.array(model["cuts"]), score(f, model)))
