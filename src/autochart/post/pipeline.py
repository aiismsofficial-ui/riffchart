"""Post-processing pipeline applied to a generated Song (all difficulties).

Order: repeated-section consistency on Expert -> playability fixes on Expert -> lower difficulties
(kept as generated, or reduced from the fixed Expert) -> playability fixes on them -> star power
(placed on Expert, copied to the lower difficulties).
"""
from __future__ import annotations

import pickle

import numpy as np
import yaml

from ..formats.song import DIFFICULTIES, Song
from ..paths import project_path
from .density import adjust as adjust_density
from .ladder import ladder
from .playability import fix_track
from .reduce import LOWER, MODEL_PATH, reduce_learned
from .repeats import apply_repeats, bar_features, bars_from_grid, find_repeats
from .star_power import copy_phrases, place_star_power

_REDUCER = None


def _reducer():
    global _REDUCER
    if _REDUCER is None:
        _REDUCER = pickle.load(open(MODEL_PATH, "rb"))
    return _REDUCER


def postprocess(song: Song, cfg: dict, audio_s: float | None = None, mel: np.ndarray | None = None,
                beats: np.ndarray | None = None, downbeats: np.ndarray | None = None,
                lower: list[str] | None = None, density: float = 0.0) -> Song:
    """lower: difficulties to produce besides Expert when cfg['lower'] == 'reduce'
    (default: the ones present in `song`). density: the user's sparser (-1) / denser (+1) control
    (post/density.py), applied before the playability fixes."""
    rules_file = project_path(cfg.get("eval_config", "configs/eval.yaml"))
    rules = yaml.safe_load(open(rules_file, encoding="utf-8"))["playability"]
    tracks = dict(song.tracks)
    ex = tracks.get("expert")
    if ex is not None and cfg.get("repeats", {}).get("enabled") and mel is not None and beats is not None:
        bars = bars_from_grid(beats, downbeats)
        if len(bars) >= 8:
            rep = find_repeats(bar_features(mel, beats, bars), bars, cfg["repeats"]["thr"])
            ex = apply_repeats(song, ex, beats, bars, rep)
    if ex is not None and cfg.get("fix", True):
        ex = fix_track(song, ex, rules["expert"], audio_s=audio_s)
    if ex is not None:
        tracks["expert"] = ex
    if cfg.get("lower", "direct") == "reduce" and ex is not None:
        want = lower if lower is not None else [d for d in LOWER if d in song.tracks]
        thr = cfg.get("reduce_thr", {})
        for d in LOWER:
            tracks.pop(d, None)
        for d in want:
            tracks[d] = reduce_learned(song, ex, d, _reducer(), thr.get(d, 0.5))
    if density:
        tracks = adjust_density(Song(song.resolution, song.tempo, tracks, song.sections), density, _reducer())
    lad = cfg.get("ladder", {})
    if lad.get("enabled") and "expert" in tracks:
        caps = {d: c + lad.get("denser_cap_step", 0.12) * max(density, 0.0) for d, c in lad.get("caps", {}).items()}
        tracks = ladder(Song(song.resolution, song.tempo, tracks, song.sections), tracks, _reducer(), caps,
                        int(lad.get("window_beats", 16)))
    if cfg.get("fix", True):
        for d in (DIFFICULTIES if density else LOWER):
            if d in tracks:
                tracks[d] = fix_track(song, tracks[d], rules[d], audio_s=audio_s)
    if cfg.get("star_power", True) and "expert" in tracks:
        phrases = place_star_power(song, tracks["expert"], cfg.get("sp_rate_per_min", 2.3))
        for d in DIFFICULTIES:
            if d in tracks:
                tracks[d].star_power = copy_phrases(phrases, tracks[d])
    return Song(song.resolution, song.tempo, tracks, song.sections, song.metadata, song.chart_offset_s,
                song.source_format, song.warnings)
