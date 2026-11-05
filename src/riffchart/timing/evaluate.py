"""Tempo-map quality vs the human charts' SyncTracks.

For each song: tracker beats/downbeats (cached per tracker) -> tempo map (mode) -> metrics within
[first human note - 1 s, last human note + 0.5 s]:
  beat_f1, beat_f1_metrical (half/double allowed), downbeat_f1     mir_eval, 70 ms
  n_tempo_events / human n_tempo_events                            map complexity
  grid fit: human Expert onsets snapped to the predicted grid's musical positions
    snap_med_ms / snap_p90_ms   how far notes have to move
    snap_f1_25                  F1@25 ms of snapped vs original onsets
    phase_match                 snapped 48th-position == the human chart's own 48th-position
                                (the rhythm can be written the same way on the predicted grid)

Trackers
  beatthis       Beat This! final0 on the full mix (MIT)
  beatthis_ens   mean of final0/1/2 framewise logits
  fretformer_ref Fretformer 2's fine-tuned Beat This + smoother (CC BY-NC-SA, reference only:
                 cannot ship in our product) read from the Phase 2 feature cache (test only)
  finetuned      our Beat This fine-tuned on the train split's tempo maps (Phase 3)

    python -m riffchart.timing.evaluate --tracker beatthis --mode smooth --split val
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mir_eval
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..baselines.common import load_full_mix, split_songs
from ..eval.metrics import Events, match
from ..formats.loader import load_song
from .grid import GRID, snap
from .tempo_map import beats_to_tempo_map

SR = 22050


def tracker_beats(tracker: str, row, root: Path, ckpt: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    if tracker == "fretformer_ref":
        z = np.load(root / "cache" / "fretformer" / f"{row.song_id}.npz")
        return z["beats"], z["downbeats"]
    cache = root / "cache" / f"beats_{tracker}" / f"{row.song_id}.npz"
    if tracker == "beatthis":  # Phase 2 cache location
        cache = root / "cache" / "beats" / f"{row.song_id}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["beats"], z["downbeats"]
    from .trackers import get_tracker

    y = load_full_mix(row, SR)
    b, d = get_tracker(tracker, ckpt)(y, SR)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, beats=b, downbeats=d)
    return b, d


def song_metrics(row, beats, downbeats, mode: str, **kw) -> dict:
    ref, _, _ = load_song(row.path)
    off = row.delay_s + row.align_correction_ms / 1000.0
    tm = beats_to_tempo_map(beats, downbeats, 192, mode=mode, **kw)
    ex = ref.tracks["expert"]
    on = ref.tempo.tick_to_sec(ex.tick) + off
    lo, hi = on.min() - 1.0, on.max() + 0.5
    rb = ref.tempo.beat_times(hi - off + 2) + off
    rd = ref.tempo.downbeat_times(hi - off + 2) + off
    eb = tm.beat_times(hi + 2)
    ed = tm.downbeat_times(hi + 2)
    w = lambda x: x[(x >= lo) & (x <= hi)]  # noqa: E731
    rb, rd, eb, ed = w(rb), w(rd), w(eb), w(ed)
    out = {"song_id": row.song_id}
    out["beat_f1"] = mir_eval.beat.f_measure(rb, eb)
    half, dbl = eb[::2], np.sort(np.concatenate([eb, (eb[:-1] + eb[1:]) / 2]))
    out["beat_f1_metrical"] = max(out["beat_f1"], mir_eval.beat.f_measure(rb, half), mir_eval.beat.f_measure(rb, dbl))
    out["downbeat_f1"] = mir_eval.beat.f_measure(rd, ed) if len(rd) > 1 and len(ed) > 1 else np.nan
    in_win = lambda t: ((t >= lo) & (t <= hi)).sum()  # noqa: E731
    out["tempo_events"] = int(in_win(tm.tick_to_sec(tm.ticks)))
    out["tempo_events_human"] = int(in_win(ref.tempo.tick_to_sec(ref.tempo.ticks) + off))
    # grid fit
    units = snap(tm, on)
    t_snap = tm.tick_to_sec(units * tm.resolution / GRID)
    shift = np.abs(t_snap - on) * 1000
    out["snap_med_ms"] = float(np.median(shift))
    out["snap_p90_ms"] = float(np.percentile(shift, 90))
    z = np.zeros(len(on))
    k = len(match(Events(on, ex.mask, z, ex.ntype), Events(t_snap, ex.mask, z, ex.ntype), 0.025))
    out["snap_f1_25"] = k / len(on)
    human_pos = np.round((ex.tick % ref.resolution) / ref.resolution * GRID).astype(int) % GRID
    dpos = np.abs(((units % GRID) - human_pos + GRID // 2) % GRID - GRID // 2)
    out["phase_match"] = float((dpos == 0).mean())
    out["phase_match_tol1"] = float((dpos <= 1).mean())   # within 1/48 beat
    return out


def run(tracker: str, mode: str, split: str, root: Path, ckpt: str | None = None, limit: int | None = None, **kw) -> pd.DataFrame:
    rows = []
    for r in tqdm(list(split_songs(root, split).itertuples())[:limit], desc=f"{tracker}/{mode}", mininterval=10):
        b, d = tracker_beats(tracker, r, root, ckpt)
        rows.append(song_metrics(r, b, d, mode, **kw))
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", default="beatthis")
    ap.add_argument("--mode", default="smooth")
    ap.add_argument("--split", default="val")
    ap.add_argument("--ckpt")
    ap.add_argument("--tol-ms", type=float, default=12.0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ts-mode", default="raw")
    args = ap.parse_args(argv)
    root = Path("data")
    df = run(args.tracker, args.mode, args.split, root, args.ckpt, args.limit, tol_s=args.tol_ms / 1000, ts_mode=args.ts_mode)
    name = f"{args.tracker}_{args.mode}" + (f"_tol{int(args.tol_ms)}" if args.mode == "smooth" else "") + f"_ts{args.ts_mode}"
    out = root / "eval" / "tempo" / args.split
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}.csv", index=False)
    s = df.drop(columns="song_id").mean().to_dict()
    s["tempo_events_ratio_median"] = float((df.tempo_events / df.tempo_events_human.clip(lower=1)).median())
    (out / f"{name}.json").write_text(json.dumps(s, indent=2))
    print(name, json.dumps({k: round(v, 4) for k, v in s.items()}))


if __name__ == "__main__":
    main()
