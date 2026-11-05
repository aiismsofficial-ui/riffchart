"""Re-chart one section of an existing chart (the editor's "re-chart section", M5).

The song's tempo map is kept exactly (beat k = tick k * resolution), the model continues from the
chords already in the chart before the section (its left context), and only beats [start, end) are
generated again. The usual post-processing (density, ladder, playability fixes) then runs on the section
alone, with the current notes of any difficulty that is not being re-charted, and only the notes inside
the section are returned (the editor splices them in as a take).

The audio analysis (stem separation -> model inputs for every play-along mode) is the slow part, so it
is computed once per song and cached; after that a take is a few seconds.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import soxr

from ..formats.song import DIFFICULTIES, GuitarTrack, Song
from ..model.data_prep import SR as MEL_SR
from ..model.generate import events_to_song
from ..model.tokenizer import Event
from ..post.pipeline import postprocess
from ..timing.grid import GRID
from .follow import MODES as FOLLOW_MODES, playable_part

FEATS_VERSION = 1


def section_features(ac, mix: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    """Mix (n, 2) at `sr` -> {"audio_s": ..., <follow mode>: model input features (frames, 2, 128) fp16}."""
    from .core import OUT_SR
    if sr != OUT_SR:
        mix = soxr.resample(mix, sr, OUT_SR)
    mix = np.repeat(mix[:, :1], 2, 1) if mix.shape[1] == 1 else mix[:, :2]
    stems = ac.sep(soxr.resample(mix, OUT_SR, ac.sep.sr) if ac.sep.sr != OUT_SR else mix)
    m22 = soxr.resample(mix.mean(1), OUT_SR, MEL_SR).astype(np.float32)
    out = {"audio_s": np.float64(len(mix) / OUT_SR), "version": np.int64(FEATS_VERSION)}
    for mode in FOLLOW_MODES:                      # same inputs AutoCharter.chart_analysis builds
        play = playable_part(stems, ac.sep.sr, mode)
        g22 = soxr.resample(play.model_in.mean(1), ac.sep.sr, MEL_SR).astype(np.float32)
        n = min(len(m22), len(g22))
        out[mode] = np.stack([ac.mel(g22[:n]), ac.mel(m22[:n])], 1).astype(np.float16)
    return out


def audio_key(files: list[Path]) -> str:
    return "|".join(f"{f.name}:{f.stat().st_size}:{int(f.stat().st_mtime)}" for f in files)


def load_features(path: Path, key: str) -> dict | None:
    if not path.exists():
        return None
    try:
        z = np.load(path, allow_pickle=False)
        if str(z["key"]) != key or int(z["version"]) != FEATS_VERSION:
            return None
        return {k: z[k] for k in z.files}
    except Exception:                               # damaged cache: compute again
        return None


def save_features(path: Path, key: str, feats: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, key=np.array(key), **feats)
    tmp.replace(path)


def track_events(tr: GuitarTrack, res: int) -> list[Event]:
    k = GRID / res
    return [Event(int(round(t * k)), int(m), int(n), int(round(s * k)))
            for t, m, s, n in zip(tr.tick, tr.mask, tr.sustain, tr.ntype)]


def snap_range(song: Song, start_tick: int, end_tick: int) -> tuple[int, int]:
    """Section bounds -> whole beats [b0, b1) (the model charts beat by beat)."""
    res = song.resolution
    b0 = max(0, int(start_tick) // res)
    b1 = max(b0 + 1, -(-int(end_tick) // res))
    return b0, b1


def chart_section(ac, feats: dict, song: Song, offset_s: float, start_tick: int, end_tick: int,
                  difficulties, style, seed: int) -> dict:
    """-> {"start": tick, "end": tick, "notes": {difficulty: [[tick, mask, sustain, ntype], ...]}}.
    `song` is the chart as it is now in the editor (edits included); offset_s: audio time = chart time + offset_s."""
    if getattr(ac.charter, "use_pitch", False):
        raise RuntimeError("section re-chart does not support pitch-input models")
    mode = style.follow or ac.cfg.get("follow", "guitar")
    mel = feats[mode]
    audio_s = float(feats["audio_s"])
    res, tm = song.resolution, song.tempo
    end_chart = audio_s - offset_s
    beats = tm.beat_times(end_chart + 2) + offset_s
    downs = tm.downbeat_times(end_chart + 2) + offset_s
    b0, b1 = snap_range(song, start_tick, end_tick)
    b1 = min(b1, int(np.searchsorted(beats, audio_s - 0.05)))
    if b1 <= b0:
        raise ValueError("the section is outside the song's audio")
    diffs = [d for d in DIFFICULTIES if d in difficulties]
    if not diffs:
        raise ValueError("pick at least one difficulty")
    existing = {d: track_events(song.tracks[d], res) for d in DIFFICULTIES if d in song.tracks}
    from .core import DENSER_BIAS
    dc = ac.cfg["decoding"]
    new = ac.charter.chart(mel, beats, downs, diffs, dc["temperature"], dc["top_p"], seed=int(seed),
                           style=style.tokens(), audio_s=audio_s,
                           pos_bias={"expert": DENSER_BIAS * max(style.density, 0.0)},
                           min_gap_s=ac.min_gap if dc["min_gap"] == "auto" else None,
                           start_beat=b0, end_beat=b1, prefix=existing)
    # Post-process the section on its own: new chords for the re-charted difficulties, the current ones for
    # the others (the ladder compares against them). With the rest of the song included, density / ladder
    # thinning could land outside the section and be thrown away, leaving the section unbalanced.
    lo, hi = b0 * GRID, b1 * GRID
    merged = {}
    for d in DIFFICULTIES:
        if d in diffs:
            merged[d] = new[d]
        elif d in existing:
            merged[d] = [e for e in existing[d] if lo <= e.unit < hi]
    whole = events_to_song(merged, beats, downs, tempo=tm)
    whole.sections = list(song.sections)
    post_cfg = copy.deepcopy(ac.post_cfg)
    post_cfg.setdefault("repeats", {})["enabled"] = False     # would copy other bars over the new take
    post_cfg["star_power"] = False                             # the editor keeps its own phrases
    whole = postprocess(whole, post_cfg, audio_s=audio_s, mel=mel, beats=beats, downbeats=downs,
                        density=style.density)
    t0, t1 = b0 * res, b1 * res
    out = {}
    for d in diffs:
        tr = whole.tracks.get(d)
        after = song.tracks[d].tick[song.tracks[d].tick >= t1] if d in song.tracks else []
        nxt = int(after[0]) if len(after) else None          # first chord after the section
        rows = []
        if tr is not None:
            for t, m, s, n in zip(tr.tick, tr.mask, tr.sustain, tr.ntype):
                if t0 <= t < t1:
                    s = min(int(s), nxt - int(t)) if nxt is not None and s else int(s)
                    rows.append([int(t), int(m), s, int(n)])
        out[d] = rows
    return {"start": int(t0), "end": int(t1), "notes": out}
