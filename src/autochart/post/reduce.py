"""Expert -> Hard / Medium / Easy reduction.

Human reductions are subsets of Expert: 96-97% of lower-difficulty chords sit on an Expert chord tick
(127 train songs). So the main decision is, per Expert chord, whether it survives at this difficulty
(gradient-boosted classifier). It is trained only on songs with a genuine human reduction (Phase 1
`real_*`): 92 / 77 / 61 train songs for H / M / E. Many charts in the library are Expert-only or copy
Expert into every difficulty.

Frets are kept as in Expert. On val, Expert frets match the human reduced chord 65 / 51 / 43% of the
time (H / M / E). A classifier over absolute target masks scored 57 / 48 / 42% and one over relative
transformations (shift down, reduce to a single fret, ...) 61 / 46 / 46%: fret mappings are
charter-specific and don't transfer across the artist-disjoint split (Phase 5 notes).

Note types after reduction: a HOPO stays a HOPO only when the chord before it also survived (the
hammer-on still starts from the same note); taps stay taps. Sustains are kept.

    python -m autochart.post.reduce train          # features + keep model, ~1 min
    python -m autochart.post.reduce apply <system> <new_system> [--split val] [--rules]
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from tqdm import tqdm

from ..baselines.common import split_songs, write_song_folder
from ..formats.loader import load_song
from ..formats.textio import song_delay_s
from ..formats.song import OPEN_BIT, GuitarTrack, NoteType, Song, popcount
from ..paths import PROJECT
from .playability import _SAL, rebuild

LOWER = ("hard", "medium", "easy")
MODEL_PATH = PROJECT / "data/checkpoints/reduce/reduce.pkl"


def features(song: Song, tr: GuitarTrack, diff: str) -> np.ndarray:
    res, tm = song.resolution, song.tempo
    tick = tr.tick.astype(np.int64)
    n = len(tick)
    t = tm.tick_to_sec(tick)
    spb = tm.tick_to_sec(tick + res) - t
    beat = tick / res
    pos = (tick % res) * 48 // res
    end_s = float(t[-1]) + 1.0 if n else 1.0
    bars = np.round(tm.sec_to_tick(tm.downbeat_times(end_s))).astype(np.int64)
    bi = np.clip(np.searchsorted(bars, tick, side="right") - 1, 0, max(len(bars) - 1, 0))
    in_bar = (tick - bars[bi]) / res if len(bars) else beat % 4
    big = 8.0
    ioi_p = np.concatenate([[big], np.diff(beat)]).clip(0, big)
    ioi_n = np.concatenate([np.diff(beat), [big]]).clip(0, big)
    ioi_ps = np.concatenate([[big], np.diff(t)]).clip(0, big)
    ioi_ns = np.concatenate([np.diff(t), [big]]).clip(0, big)
    dens_b = np.searchsorted(beat, beat + 1, "right") - np.searchsorted(beat, beat - 1, "left")
    dens_s = np.searchsorted(t, t + 2, "right") - np.searchsorted(t, t - 2, "left")
    m = tr.mask.astype(int)
    bits = (m[:, None] >> np.arange(6)) & 1
    pc = popcount(tr.mask)
    pitch = np.where(m & OPEN_BIT, -1.0, (bits[:, :5] * np.arange(5)).sum(1) / np.maximum(bits[:, :5].sum(1), 1))
    dp_p = np.concatenate([[0], np.diff(pitch)])
    dp_n = np.concatenate([np.diff(pitch), [0]])
    same_p = np.concatenate([[0], (m[1:] == m[:-1]).astype(int)])
    sus_b = tr.sustain / res
    first_in_beat = np.concatenate([[1], (np.floor(beat[1:]) != np.floor(beat[:-1])).astype(int)])
    nps = n / max(t[-1] - t[0], 1.0) if n > 1 else 0.0
    song_f = np.array([nps, bits[:, 4].mean(), bits[:, 5].mean(), (pc > 1).mean(), np.median(spb)])
    d = np.full(n, LOWER.index(diff))
    cols = [d, pos, _SAL[pos], in_bar, ioi_p, ioi_n, ioi_ps, ioi_ns, spb, dens_b, dens_s, pc, pitch, dp_p, dp_n,
            same_p, sus_b, tr.ntype, first_in_beat, m]
    X = np.column_stack(cols + [bits] + [np.tile(song_f, (n, 1))]).astype(np.float32)
    return X


CAT_COLS = [0, 19]  # categorical: difficulty and expert mask


def song_examples(song: Song, diff: str):
    ex, tr = song.tracks.get("expert"), song.tracks.get(diff)
    if ex is None or tr is None or len(ex) < 20 or len(tr) < 5:
        return None
    X = features(song, ex, diff)
    keep = np.isin(ex.tick, tr.tick)
    tmask = dict(zip(tr.tick.tolist(), tr.mask.tolist()))
    y_mask = np.array([tmask.get(int(t), 0) for t in ex.tick])
    return X, keep, y_mask


def build(split: str):
    Xs, ks, ms, gs = [], [], [], []
    rows = split_songs("data", split)
    for r in tqdm(list(rows.itertuples()), desc=f"reduce features {split}", mininterval=10):
        diffs = [d for d in LOWER if getattr(r, f"real_{d}")]
        if not diffs:
            continue
        song, _, _ = load_song(r.path)
        for d in diffs:
            out = song_examples(song, d)
            if out is None:
                continue
            Xs.append(out[0]); ks.append(out[1]); ms.append(out[2]); gs.append(np.full(len(out[1]), hash(r.song_id) % 10**6))
    return np.concatenate(Xs), np.concatenate(ks), np.concatenate(ms), np.concatenate(gs)


def train(seed: int = 0) -> dict:
    X, keep, y_mask, _ = build("train")
    print(f"{len(X)} expert chords, keep rate {keep.mean():.3f}", flush=True)
    cat = np.zeros(X.shape[1], bool); cat[CAT_COLS] = True
    km = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
                                        categorical_features=cat, random_state=seed, early_stopping=True)
    km.fit(X, keep)
    models = {"keep": km}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(models, open(MODEL_PATH, "wb"))
    print(f"saved {MODEL_PATH}", flush=True)
    return models


def _finish(song: Song, ex: GuitarTrack, diff: str, keep: np.ndarray, masks: np.ndarray) -> GuitarTrack:
    idx = np.flatnonzero(keep)
    ntype = ex.ntype[idx].copy()
    prev_kept = np.concatenate([[False], np.diff(idx) == 1])        # the chord right before survived
    ntype[(ntype == NoteType.HOPO) & ~prev_kept] = NoteType.STRUM
    m = masks[idx]
    ntype[np.concatenate([[False], m[1:] == m[:-1]]) & (ntype == NoteType.HOPO)] = NoteType.STRUM
    return rebuild(diff, ex.tick[idx], m, ex.sustain[idx], ntype, song.resolution, ex.star_power, ex.solos)


def reduce_learned(song: Song, ex: GuitarTrack, diff: str, models: dict, thr: float = 0.5) -> GuitarTrack:
    if len(ex) == 0:
        return GuitarTrack.empty(diff)
    X = features(song, ex, diff)
    keep = models["keep"].predict_proba(X)[:, 1] >= thr
    return _finish(song, ex, diff, keep, ex.mask.astype(int))


# Rule-based reference: keep by rhythmic salience under a per-difficulty minimum gap (the human 5th
# percentile onset gap: 0.155 / 0.201 / 0.270 s), squash 4+-note chords, Easy drops orange and shifts down.
RULE_GAP_S = {"hard": 0.155, "medium": 0.20, "easy": 0.27}


def reduce_rules(song: Song, ex: GuitarTrack, diff: str) -> GuitarTrack:
    if len(ex) == 0:
        return GuitarTrack.empty(diff)
    from .playability import _fit_chord, salience
    t = song.tempo.tick_to_sec(ex.tick)
    sal = salience(ex.tick, ex.mask, ex.sustain, song.resolution)
    kept: list[int] = []
    for i in range(len(t)):
        while kept and t[i] - t[kept[-1]] < RULE_GAP_S[diff] and sal[i] > sal[kept[-1]]:
            kept.pop()
        if not kept or t[i] - t[kept[-1]] >= RULE_GAP_S[diff]:
            kept.append(i)
    keep = np.zeros(len(t), bool); keep[kept] = True
    masks = np.array([_fit_chord(int(m), 3 if diff == "hard" else 2, 4) for m in ex.mask])
    if diff == "easy":
        masks = np.array([m if m & OPEN_BIT else (m >> 1 if m & 16 else m) or 1 for m in masks])
    return _finish(song, ex, diff, keep, masks)


def apply(system: str, new_system: str, split: str, rules: bool, thr: dict[str, float] | None = None,
          only_real: bool = True) -> None:
    models = None if rules else pickle.load(open(MODEL_PATH, "rb"))
    src, dst = Path("data/outputs") / system, Path("data/outputs") / new_system
    for r in tqdm(list(split_songs("data", split).itertuples()), desc=new_system, mininterval=10):
        f = Path(r.path) if system == "human" else src / r.song_id      # "human": oracle Expert input
        if system != "human" and not (f / "notes.chart").exists():
            continue
        song, ini, _ = load_song(f)
        ex = song.tracks["expert"]
        tracks = {"expert": ex}
        for d in LOWER:
            if only_real and not getattr(r, f"real_{d}"):
                continue
            tracks[d] = reduce_rules(song, ex, d) if rules else \
                reduce_learned(song, ex, d, models, (thr or {}).get(d, 0.5))
        out = Song(song.resolution, song.tempo, tracks, song.sections, song.metadata)
        delay = r.delay_s + r.align_correction_ms / 1000.0 if system == "human" else song_delay_s(ini)
        write_song_folder(dst / r.song_id, out, str(r.name), str(r.artist), "autochart", float(r.audio_s), delay)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train")
    a = sub.add_parser("apply")
    a.add_argument("system"); a.add_argument("new_system")
    a.add_argument("--split", default="val")
    a.add_argument("--rules", action="store_true")
    a.add_argument("--thr", type=float, nargs=3, default=None, metavar=("HARD", "MEDIUM", "EASY"))
    args = ap.parse_args(argv)
    if args.cmd == "train":
        train()
    else:
        thr = dict(zip(LOWER, args.thr)) if args.thr else None
        apply(args.system, args.new_system, args.split, args.rules, thr)


if __name__ == "__main__":
    main()
